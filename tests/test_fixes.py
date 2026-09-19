# tests/test_fixes.py
"""Non-régression des correctifs « Phase 0 » (GCO-DQN).

Couvre :
1. Le bootstrap TD est **masqué** (pas d'argmax sur une action invalide).
2. L'espace d'action est unifié à ``2N+1`` (add / remove / terminate).
3. Le réseau Q produit bien ``2N+1`` valeurs (tête de retrait incluse).
4. ``valid_action_mask()`` est cohérent avec ``valid_actions()``.
5. ``DisruptionTracker.reset_state()`` restaure l'état.

Exécution : ``python tests/test_fixes.py`` ou ``pytest tests/test_fixes.py``
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.generate_synthetic import (
    generate_organization,
    generate_project_requirements,
)
from src.environment.team_formation_env import TeamFormationEnv
from src.graph.metrics import DisruptionTracker
from src.models.dqn_agent import GCODQNAgent, ReplayBuffer
from src.models.gnn_encoder import TeamFormationQNetwork


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _tiny_env(allow_removal: bool) -> TeamFormationEnv:
    G, _, all_skills = generate_organization(
        n_employees=30, n_departments=3, n_skills=8, seed=0
    )
    project = generate_project_requirements(all_skills, 3, seed=0)
    return TeamFormationEnv(
        graph=G,
        project_skills=project,
        all_skills=all_skills,
        max_team_size=6,
        allow_removal=allow_removal,
    )


def _skill_matrix(env: TeamFormationEnv) -> np.ndarray:
    """Les n_skills premières colonnes des features statiques = one-hot skills."""
    return env.static_node_features[:, : env.n_skills].copy()


# --------------------------------------------------------------------------- #
# 1. Le cœur du fix : bootstrap masqué
# --------------------------------------------------------------------------- #

def test_masked_max_next_q_ignores_invalid_action():
    """Une action invalide à Q très haute ne doit PAS être retenue."""
    online = torch.tensor([[10.0, -1.0, 0.0]])   # Q action 0 = 10 (invalide)
    target = torch.tensor([[10.0, 5.0, 7.0]])
    valid = torch.tensor([[0.0, 1.0, 1.0]])      # action 0 masquée

    got = GCODQNAgent.masked_max_next_q(online, target, valid)

    # Avec masque : argmax → action 2 → target[2] = 7
    assert abs(float(got[0]) - 7.0) < 1e-6, f"attendu 7.0, obtenu {got}"

    # Sans masque (ancien comportement buggé) : argmax → action 0 → target[0] = 10
    unmasked = GCODQNAgent.masked_max_next_q(
        online, target, torch.ones_like(valid)
    )
    assert abs(float(unmasked[0]) - 10.0) < 1e-6


def test_masked_max_next_q_no_nan_when_all_masked():
    """Ligne entièrement masquée → 0.0, jamais NaN."""
    online = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 2.0]])
    valid = torch.tensor([[0.0, 0.0]])

    got = GCODQNAgent.masked_max_next_q(online, target, valid)
    assert torch.isfinite(got).all(), got
    assert float(got[0]) == 0.0


# --------------------------------------------------------------------------- #
# 2. Espace d'action unifié 2N+1
# --------------------------------------------------------------------------- #

def test_action_space_is_2n_plus_1():
    for allow in (False, True):
        env = _tiny_env(allow)
        n = env.n_nodes
        assert env.n_actions == 2 * n + 1, f"n_actions={env.n_actions} (n={n})"
        assert env.terminate_action == 2 * n
        assert env.terminate_action in env.valid_actions()


def test_tiny_env_accepts_all_action_types():
    """Chaque type d'action (add / remove / terminate) doit fonctionner."""
    env = _tiny_env(allow_removal=True)
    n = env.n_nodes

    env.reset()
    _, _, term, trunc, info = env.step(0)              # add_0
    assert not term and not trunc and info["op"] == "add"

    _, _, term, trunc, info = env.step(n + 0)          # remove_0
    assert not term and not trunc and info["op"] == "remove"

    _, _, term, trunc, info = env.step(env.terminate_action)
    assert term and info["op"] == "terminate"


# --------------------------------------------------------------------------- #
# 3. Masque d'actions valides
# --------------------------------------------------------------------------- #

def test_valid_action_mask_shape_and_content():
    env = _tiny_env(allow_removal=True)
    n = env.n_nodes

    m = env.valid_action_mask()
    assert m.shape == (2 * n + 1,)
    assert m[:n].sum() == n              # tous les ajouts possibles
    assert m[n:2 * n].sum() == 0.0       # aucun retrait au départ
    assert m[env.terminate_action] == 1.0


def test_valid_action_mask_after_selection():
    env = _tiny_env(allow_removal=True)
    n = env.n_nodes

    env.reset()
    env.step(0)
    env.step(1)
    m = env.valid_action_mask()

    assert m[:n].sum() == n - 2, "les employés déjà sélectionnés ne sont pas ajoutables"
    assert m[n:2 * n].sum() == 2.0, "les employés sélectionnés sont retirables"
    assert m[0] == 0.0 and m[n] == 1.0
    assert m[env.terminate_action] == 1.0


def test_no_removal_masks_all_removes():
    env = _tiny_env(allow_removal=False)
    n = env.n_nodes

    env.reset()
    env.step(0)
    m = env.valid_action_mask()
    assert m[n:2 * n].sum() == 0.0, "aucun retrait ne doit être valide"
    assert m[env.terminate_action] == 1.0


# --------------------------------------------------------------------------- #
# 4. Réseau Q : 2N+1 sorties (tête de retrait présente)
# --------------------------------------------------------------------------- #

def test_network_outputs_2n_plus_1():
    env = _tiny_env(allow_removal=True)
    n, s = env.n_nodes, env.n_skills

    net = TeamFormationQNetwork(
        n_skills=s, gnn_hidden_dim=8, q_hidden_dim=8, n_gnn_layers=1
    )
    net.eval()

    x = torch.as_tensor(env.static_node_features, dtype=torch.float32)
    ei = torch.as_tensor(env.edge_index(), dtype=torch.long)
    mtx = torch.as_tensor(_skill_matrix(env), dtype=torch.float32)

    # Cas mono-échantillon : tenseurs 1-D (comme agent.q_values)
    q = net(x, ei, mtx, torch.zeros(n), torch.zeros(s))
    assert q.shape == (2 * n + 1,), f"shape={tuple(q.shape)} attendu {(2 * n + 1,)}"

    # Cas batché : tenseurs 2-D [B, ...]
    qb = net(x, ei, mtx, torch.zeros(4, n), torch.zeros(4, s))
    assert qb.shape == (4, 2 * n + 1), qb.shape

    assert any("remove_head" in name for name, _ in net.named_parameters())


# --------------------------------------------------------------------------- #
# 5. Buffer + agent + tracker
# --------------------------------------------------------------------------- #

def test_buffer_stores_valid_mask():
    buf = ReplayBuffer(capacity=4, n_nodes=3, n_skills=2, n_actions=7)
    buf.push(
        np.zeros(3, np.float32), np.zeros(2, np.float32), 1, 0.5, False,
        np.zeros(3, np.float32), np.zeros(2, np.float32), np.ones(7, np.float32),
    )
    *_, next_valids = buf.sample(1)
    assert next_valids.shape == (1, 7)
    assert next_valids[0, 0] == 1.0


def test_train_step_runs_end_to_end():
    """Un train_step complet ne doit pas planter (formes 2N+1 cohérentes)."""
    env = _tiny_env(allow_removal=True)
    agent = GCODQNAgent(
        n_skills=env.n_skills,
        n_nodes=env.n_nodes,
        gnn_hidden_dim=8,
        q_hidden_dim=8,
        n_gnn_layers=1,
        batch_size=4,
        device="cpu",
        n_actions=env.n_actions,
    )
    agent.set_graph(env.static_node_features, env.edge_index(), _skill_matrix(env))

    env.reset()
    mask = np.zeros(env.n_nodes, np.float32)
    cov = np.zeros(env.n_skills, np.float32)
    valid = env.valid_action_mask()

    for _ in range(8):
        agent.store_transition(mask, cov, 0, 1.0, False, mask, cov, valid)

    loss = agent.train_step()
    assert np.isfinite(loss) and loss >= 0.0, loss


def test_tracker_reset_state_matches_fresh():
    """reset_state() doit donner le même état qu'un tracker neuf."""
    G, _, _ = generate_organization(
        n_employees=20, n_departments=2, n_skills=5, seed=1
    )

    fresh = DisruptionTracker(G)
    reused = DisruptionTracker(G)

    for node in (0, 1, 2):
        reused.add(node)
    reused.reset_state()

    assert reused.debug_state() == fresh.debug_state()


# --------------------------------------------------------------------------- #
# 6. Sanity check du reward shaping (la leçon de la Phase 0)
# --------------------------------------------------------------------------- #

def test_config_reward_is_not_self_defeating():
    """Le reward de config.yaml ne doit pas rendre « ne rien faire » optimal.

    Piège découvert : si ``size_penalty`` dépasse le gain marginal d'un employé
    utile (``skill_weight * 2 / n_required_skills``), alors AUCUN ajout n'est
    rentable → l'agent apprend à terminer immédiatement (équipe de taille 0).
    """
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    n_req = cfg["data"]["n_required_skills"]

    # Gain d'un employé apportant 2 compétences projet neuves
    marginal = env_cfg["skill_weight"] * 2 / n_req - env_cfg["size_penalty"]

    assert marginal > 0, (
        f"gain marginal d'un employé à 2 skills neuves = {marginal:.4f} ≤ 0 → "
        f"l'agent n'a aucune incitation à construire une équipe "
        f"(size_penalty={env_cfg['size_penalty']} trop élevé)"
    )


def test_complete_team_beats_immediate_terminate():
    """Construire une équipe couvrante doit rapporter plus que terminer tout de suite."""
    env = _tiny_env(allow_removal=False)
    project = env.project_skills

    # (a) Terminer immédiatement
    env.reset()
    _, reward_quit, term, _, _ = env.step(env.terminate_action)
    assert term

    # (b) Ajouter les employés utiles jusqu'à couvrir le projet, puis terminer
    env.reset()
    total, covered = 0.0, set()
    for node in sorted(env.graph.nodes()):
        if covered >= project:
            break
        skills = env.graph.nodes[node].get("skills", set())
        if skills - covered:
            _, r, term, trunc, _ = env.step(node)
            total += r
            covered |= skills
            if term or trunc:
                break

    assert env.covered_skills >= project, "le graphe de test doit pouvoir couvrir le projet"
    _, r_term, term, _, _ = env.step(env.terminate_action)
    total += r_term

    assert total > reward_quit, (
        f"construire l'équipe ({total:.2f}) doit battre terminer tout de suite "
        f"({reward_quit:.2f})"
    )


# --------------------------------------------------------------------------- #
# 7. Masque projet (l'observation doit encoder les compétences REQUISES)
# --------------------------------------------------------------------------- #

def test_project_skill_mask_marks_required_skills():
    env = _tiny_env(allow_removal=False)
    m = env.project_skill_mask()
    assert m.shape == (env.n_skills,)
    assert m.sum() == len(env.project_skills)
    marked = {i for i, v in enumerate(m) if v == 1.0}
    assert marked == {env.skill_index[s] for s in env.project_skills}


def test_terminate_q_depends_on_project_mask():
    """À couverture globale IDENTIQUE, changer le masque projet doit changer
    Q(terminer) — sinon le réseau ne peut pas savoir si le projet est couvert,
    et n'apprend donc jamais quand s'arrêter."""
    env = _tiny_env(allow_removal=False)
    n, s = env.n_nodes, env.n_skills

    net = TeamFormationQNetwork(
        n_skills=s, gnn_hidden_dim=8, q_hidden_dim=8, n_gnn_layers=1
    )
    net.eval()

    x = torch.as_tensor(env.static_node_features, dtype=torch.float32)
    ei = torch.as_tensor(env.edge_index(), dtype=torch.long)
    mtx = torch.as_tensor(_skill_matrix(env), dtype=torch.float32)
    sel = torch.zeros(n)

    cov = torch.zeros(s)
    cov[0] = 1.0
    cov[1] = 1.0

    pm_a = torch.zeros(s); pm_a[0] = 1.0     # la compétence 0 est requise
    pm_b = torch.zeros(s); pm_b[1] = 1.0     # la compétence 1 est requise

    with torch.no_grad():
        qa = net(x, ei, mtx, sel, cov, project_mask=pm_a)
        qb = net(x, ei, mtx, sel, cov, project_mask=pm_b)

    assert abs(float(qa[-1]) - float(qb[-1])) > 1e-6, (
        "Q(terminer) doit dépendre du masque projet"
    )


def test_terminate_head_consumes_project_coverage():
    """L'architecture doit bien recevoir 2*n_skills + hidden en entrée."""
    env = _tiny_env(allow_removal=False)
    net = TeamFormationQNetwork(
        n_skills=env.n_skills, gnn_hidden_dim=8, q_hidden_dim=8, n_gnn_layers=1
    )
    first_linear = net.terminate_head[0]
    assert first_linear.in_features == 2 * env.n_skills + 8, first_linear.in_features


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  ❌ {name} → {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {name} → {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passent.")
