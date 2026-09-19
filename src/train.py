# src/train.py
"""Pipeline complet du PoC GCO-DQN.

Sources de données :
- json  : data/sample_org.json (généré si absent)
- neo4j : graphe chargé depuis Neo4j (chargé depuis la génération si la base est vide)
- auto  : neo4j si disponible, sinon json

Exemples :
    python -m src.train --source json --episodes 50
    python -m src.train --source neo4j --generate-only   # peuple Neo4j puis quitte
    python -m src.train --source auto
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Set

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (Docker)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import yaml

from src.environment.team_formation_env import TeamFormationEnv
from src.graph.metrics import skill_coverage
from src.graph.org_graph import centrality_team, evaluate_team, graph_summary, greedy_team, random_team
from src.models.dqn_agent import GCODQNAgent


# --------------------------------------------------------------------------- #
# Chargement des données
# --------------------------------------------------------------------------- #

def graph_from_payload(data: dict) -> tuple[nx.Graph, List[str], Set[str]]:
    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(
            node["id"],
            name=node.get("name", f"Employee_{node['id']}"),
            department=node["department"],
            skills=set(node["skills"]),
            level=node["level"],
        )
    G.add_edges_from(data["edges"])
    return G, list(data["all_skills"]), set(data["project_skills"])


def load_from_json(cfg: dict) -> tuple[nx.Graph, List[str], Set[str]]:
    path = cfg["data"]["json_path"]
    if not os.path.exists(path):
        print(f"[data] {path} absent → génération synthétique...")
        from src.data.generate_synthetic import main as generate
        data = generate(json_path=path, seed=cfg.get("seed", 42))
    else:
        with open(path, "r") as f:
            data = json.load(f)
    return graph_from_payload(data)


def load_from_neo4j(cfg: dict, force_reload: bool = False) -> tuple[nx.Graph, List[str], Set[str]]:
    from src.db.neo4j_store import open_store
    from src.data.generate_synthetic import generate_organization, generate_project_requirements, organization_to_dict

    with open_store() as store:
        stats = store.stats()
        if stats["nodes"] == 0 or force_reload:
            print("[neo4j] base vide → génération + chargement...")
            seed = cfg.get("seed", 42)
            G, employees, all_skills = generate_organization(
                n_employees=cfg["data"]["n_employees"],
                n_departments=cfg["data"]["n_departments"],
                n_skills=cfg["data"]["n_skills"],
                avg_skills_per_emp=cfg["data"]["avg_skills_per_emp"],
                communication_prob=cfg["data"]["communication_prob"],
                seed=seed,
            )
            project_skills = generate_project_requirements(
                all_skills, cfg["data"]["n_required_skills"], seed=seed
            )
            payload = organization_to_dict(G, employees, all_skills, project_skills)
            store.clear()
            store.load_organization(payload["nodes"], payload["edges"])
            store.store_project(payload["all_skills"], payload["project_skills"])

        G = store.export_graph()
        all_skills, project_skills = store.get_project()
        stats = store.stats()
        print(f"[neo4j] {stats['nodes']} employés, {stats['edges']} communications")
    return G, all_skills, project_skills


def load_data(cfg: dict, source: str) -> tuple[nx.Graph, List[str], Set[str]]:
    if source == "json":
        return load_from_json(cfg)
    if source == "neo4j":
        return load_from_neo4j(cfg)
    # auto
    from src.db.neo4j_store import Neo4jStore
    if Neo4jStore.is_available():
        print("[data] Neo4j détecté → source neo4j")
        return load_from_neo4j(cfg)
    print("[data] Neo4j indisponible → source json")
    return load_from_json(cfg)


# --------------------------------------------------------------------------- #
# État pour l'agent (masques compacts ; le graphe vit dans l'agent)
# --------------------------------------------------------------------------- #

def build_static_arrays(G: nx.Graph, all_skills: List[str]):
    n = G.number_of_nodes()
    s = len(all_skills)
    skill_index = {sk: i for i, sk in enumerate(all_skills)}

    node_skill_matrix = np.zeros((n, s), dtype=np.float32)
    for nid, d in G.nodes(data=True):
        for sk in d.get("skills", set()):
            j = skill_index.get(sk)
            if j is not None:
                node_skill_matrix[nid, j] = 1.0
    return node_skill_matrix


def state_masks(env: TeamFormationEnv) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(env.n_nodes, dtype=np.float32)
    for i in env.selected:
        mask[i] = 1.0
    cov = np.zeros(env.n_skills, dtype=np.float32)
    for sk in env.covered_skills:
        idx = env.skill_index.get(sk)
        if idx is not None:
            cov[idx] = 1.0
    return mask, cov


def run_episode(env, agent, training: bool, grad_steps: int = 1) -> Dict:
    env.reset()
    mask, cov = state_masks(env)
    total_reward, done, losses = 0.0, False, []
    info = {}

    while not done:
        action = agent.select_action(mask, cov, env.valid_actions(), training=training)
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask, next_cov = state_masks(env)
        # Masque des actions valides de l'état suivant → indispensable au
        # bootstrap TD masqué (sinon l'argmax porte sur des actions invalides).
        next_valid = env.valid_action_mask()

        if training:
            agent.store_transition(mask, cov, action, reward, done,
                                   next_mask, next_cov, next_valid)
            for _ in range(grad_steps):
                loss = agent.train_step()
                if loss > 0:
                    losses.append(loss)

        mask, cov = next_mask, next_cov
        total_reward += reward

    return {
        "total_reward": total_reward,
        "team_size": info.get("team_size", len(env.selected)),
        "skill_coverage": info.get("skill_coverage", 0.0),
        "disruption": info.get("disruption", 0.0),
        "selected": list(env.selected),
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
    }


def evaluate(env, agent, n_episodes: int = 10) -> Dict:
    results = [run_episode(env, agent, training=False) for _ in range(n_episodes)]
    return {
        "avg_reward": float(np.mean([r["total_reward"] for r in results])),
        "avg_team_size": float(np.mean([r["team_size"] for r in results])),
        "avg_skill_coverage": float(np.mean([r["skill_coverage"] for r in results])),
        "avg_disruption": float(np.mean([r["disruption"] for r in results])),
        "success_rate": float(np.mean([r["skill_coverage"] >= 1.0 for r in results])),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="PoC GCO-DQN")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", choices=["json", "neo4j", "auto"], default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override du nombre d'épisodes")
    parser.add_argument("--generate-only", action="store_true", help="Génère/charge les données puis quitte")
    parser.add_argument("--eval-only", action="store_true", help="Évalue un modèle sauvegardé (outputs/model.pt)")
    parser.add_argument("--allow-removal", dest="allow_removal", action="store_true",
                        default=None, help="Active l'action 'remove' (cœur de GCO-DQN)")
    parser.add_argument("--no-removal", dest="allow_removal", action="store_false",
                        help="Désactive l'action 'remove'")
    parser.add_argument("--tag", default=None,
                        help="Suffixe des fichiers de sortie (ex: 'fix' → outputs/model_fix.pt)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    source = args.source or cfg["data"].get("source", "auto")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    # Données ---------------------------------------------------------------
    t0 = time.time()
    G, all_skills, project_skills = load_data(cfg, source)
    print(f"[data] source={source} | {G.number_of_nodes()} nœuds, "
          f"{G.number_of_edges()} arêtes | chargé en {time.time() - t0:.2f}s")
    summary = graph_summary(G)
    print(f"[data] départements: {summary['n_departments']} | densité: {summary['density']:.4f}")
    print(f"[data] compétences projet: {sorted(project_skills)}")

    if args.generate_only:
        print("[done] données générées/chargées, arrêt (--generate-only).")
        return

    # Environnement + agent --------------------------------------------------
    # allow_removal : CLI prioritaire, sinon config, sinon False
    allow_removal = args.allow_removal
    if allow_removal is None:
        allow_removal = bool(cfg["env"].get("allow_removal", False))

    env = TeamFormationEnv(
        graph=G,
        project_skills=project_skills,
        all_skills=all_skills,
        max_team_size=cfg["env"]["max_team_size"],
        skill_weight=cfg["env"]["skill_weight"],
        disruption_weight=cfg["env"]["disruption_weight"],
        step_penalty=cfg["env"]["step_penalty"],
        completion_bonus=cfg["env"]["completion_bonus"],
        size_penalty=cfg["env"].get("size_penalty", 0.1),
        delay_penalty=cfg["env"].get("delay_penalty", 1.0),
        allow_removal=allow_removal,
        remove_penalty=cfg["env"].get("remove_penalty", 0.2),
    )
    print(f"[env] allow_removal={allow_removal} | n_actions={env.n_actions} "
          f"({G.number_of_nodes()} add + "
          f"{G.number_of_nodes() if allow_removal else 0} remove + 1 terminate)")

    agent = GCODQNAgent(
        n_skills=len(all_skills),
        n_nodes=G.number_of_nodes(),
        gnn_hidden_dim=cfg["agent"]["gnn_hidden_dim"],
        q_hidden_dim=cfg["agent"]["q_hidden_dim"],
        n_gnn_layers=cfg["agent"]["n_gnn_layers"],
        dropout=cfg["agent"]["dropout"],
        lr=cfg["agent"]["lr"],
        gamma=cfg["agent"]["gamma"],
        epsilon_start=cfg["agent"]["epsilon_start"],
        epsilon_end=cfg["agent"]["epsilon_end"],
        epsilon_decay=cfg["agent"]["epsilon_decay"],
        buffer_capacity=cfg["agent"]["buffer_capacity"],
        batch_size=cfg["agent"]["batch_size"],
        target_update_freq=cfg["agent"]["target_update_freq"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_actions=env.n_actions,
    )

    node_skill_matrix = build_static_arrays(G, all_skills)
    agent.set_graph(env.static_node_features, env.edge_index(), node_skill_matrix,
                    project_mask=env.project_skill_mask())

    # Suffixe optionnel pour ne pas écraser les runs précédents (A/B test)
    suffix = f"_{args.tag}" if args.tag else ""
    model_path = os.path.join(cfg["output"]["dir"], f"model{suffix}.pt")

    if args.eval_only:
        agent.load(model_path)
        final = evaluate(env, agent, n_episodes=20)
        print(json.dumps(final, indent=2))
        return

    # Entraînement -----------------------------------------------------------
    n_episodes = args.episodes or cfg["train"]["n_episodes"]
    grad_steps = cfg["agent"].get("grad_steps_per_env_step", 1)
    print(f"\n[train] {n_episodes} épisodes sur {agent.device}...")
    rewards, coverages, disruptions = [], [], []

    for episode in range(n_episodes):
        result = run_episode(env, agent, training=True, grad_steps=grad_steps)
        agent.decay_epsilon()  # fix : déclin par épisode
        rewards.append(result["total_reward"])
        coverages.append(result["skill_coverage"])
        disruptions.append(result["disruption"])

        if (episode + 1) % cfg["train"]["eval_every"] == 0:
            ev = evaluate(env, agent, n_episodes=cfg["train"]["eval_episodes"])
            print(f"  Épisode {episode + 1:>4}/{n_episodes} | "
                  f"eps={agent.epsilon:.3f} | "
                  f"reward(train)={np.mean(rewards[-50:]):7.2f} | "
                  f"loss={result['avg_loss']:.4f} | "
                  f"couverture(eval)={ev['avg_skill_coverage']:.0%} | "
                  f"équipe={ev['avg_team_size']:.1f} | "
                  f"perturbation={ev['avg_disruption']:.3f} | "
                  f"succès={ev['success_rate']:.0%}")

    agent.save(model_path)

    # Évaluation finale -------------------------------------------------------
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE (DQN, 20 épisodes)")
    print("=" * 60)
    final = evaluate(env, agent, n_episodes=20)
    for k, v in final.items():
        print(f"  {k:>20}: {v:.3f}" if isinstance(v, float) else f"  {k:>20}: {v}")

    # Baselines (section 5.2 du PoC) ------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARAISON AVEC LES BASELINES")
    print("=" * 60)
    rows = {}
    # Équipe DQN (épisode déterministe) scorée avec la MÊME formule que les
    # baselines (evaluate_team) pour une comparaison équitable.
    dqn_team = run_episode(env, agent, training=False)["selected"]
    rows["GCO-DQN"] = evaluate_team(
        G, dqn_team, project_skills,
        disruption_weight=cfg["env"]["disruption_weight"],
        skill_weight=cfg["env"]["skill_weight"],
    )
    for name, fn in [("Greedy", greedy_team), ("Random", random_team), ("Centralité", centrality_team)]:
        team = fn(G, project_skills, max_team_size=cfg["env"]["max_team_size"])
        rows[name] = evaluate_team(
            G, team, project_skills,
            disruption_weight=cfg["env"]["disruption_weight"],
            skill_weight=cfg["env"]["skill_weight"],
        )

    def metric(m, *keys):
        for k in keys:
            if k in m:
                return m[k]
        return 0.0

    header = f"{'Méthode':<12} {'Taille':>7} {'Couverture':>11} {'Perturb.':>9} {'Score':>8} {'Complet':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        size = metric(m, "team_size", "avg_team_size")
        cov = metric(m, "skill_coverage", "avg_skill_coverage")
        disr = metric(m, "disruption", "avg_disruption")
        score = metric(m, "score", "avg_reward")
        complete = bool(m.get("complete", metric(m, "success_rate") >= 1.0))
        print(f"{name:<12} {size:>7.1f} {cov:>11.1%} {disr:>9.3f} {score:>8.2f} {str(complete):>8}")

    # Plots -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title("Récompense par épisode")
    axes[0, 1].plot(coverages)
    axes[0, 1].set_title("Couverture des compétences")
    axes[1, 0].plot(disruptions)
    axes[1, 0].set_title("Perturbation organisationnelle")
    window = min(20, max(1, n_episodes // 10))
    if len(rewards) >= window:
        axes[1, 1].plot(np.convolve(rewards, np.ones(window) / window, mode="valid"))
        axes[1, 1].set_title(f"Récompense (moyenne mobile {window})")
    for ax in axes.flat:
        ax.set_xlabel("Épisode")
    plt.tight_layout()
    plot_path = os.path.join(cfg["output"]["dir"], f"training_results{suffix}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[done] courbes → {plot_path} | modèle → {model_path}")

    # Métriques brutes ---------------------------------------------------------
    metrics_path = os.path.join(cfg["output"]["dir"], f"final_metrics{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump({"dqn": final, "baselines": rows, "graph": summary}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
