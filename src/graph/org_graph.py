# src/graph/org_graph.py
"""Helpers sur le graphe organisationnel : centralité, synthèse, baselines.

Les baselines (greedy / random / centralité) servent de point de comparaison
au DQN — elles répondent à la section 5.2 du PoC.
"""
from __future__ import annotations

import random
from typing import Dict, List, Set

import networkx as nx


def graph_summary(G: nx.Graph) -> Dict:
    depts: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        depts[d.get("department", "unknown")] = depts.get(d.get("department", "unknown"), 0) + 1
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_departments": len(depts),
        "departments": depts,
        "density": nx.density(G),
    }


def greedy_team(
    G: nx.Graph,
    required: Set[str],
    max_team_size: int = 20,
) -> List[int]:
    """Baseline greedy : ajoute itérativement l'employé couvrant le plus de
    compétences requises manquantes (ties brisés par degré croissant pour
    limiter la perturbation)."""
    covered: Set[str] = set()
    team: List[int] = []
    candidates = set(G.nodes())

    while covered < required and len(team) < max_team_size and candidates:
        best, best_gain, best_deg = None, 0, -1
        for n in candidates:
            gain = len(G.nodes[n].get("skills", set()) & (required - covered))
            deg = G.degree(n)
            if gain > best_gain or (gain == best_gain and deg < best_deg):
                best, best_gain, best_deg = n, gain, deg
        if best is None or best_gain == 0:
            break
        team.append(best)
        candidates.discard(best)
        covered |= G.nodes[best].get("skills", set()) & required

    return team


def random_team(
    G: nx.Graph,
    required: Set[str],
    max_team_size: int = 20,
    seed: int | None = None,
) -> List[int]:
    """Baseline aléatoire : pioche des employés jusqu'à couvrir les compétences
    (ou atteindre la taille max)."""
    rng = random.Random(seed)
    covered: Set[str] = set()
    team: List[int] = []
    pool = list(G.nodes())
    rng.shuffle(pool)

    for n in pool:
        if covered >= required or len(team) >= max_team_size:
            break
        team.append(n)
        covered |= G.nodes[n].get("skills", set()) & required

    return team


def centrality_team(
    G: nx.Graph,
    required: Set[str],
    max_team_size: int = 20,
) -> List[int]:
    """Baseline centralité : sélectionne les employés les plus centraux
    (degré) tant que les compétences ne sont pas couvertes."""
    covered: Set[str] = set()
    team: List[int] = []
    ranked = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)

    for n in ranked:
        if covered >= required or len(team) >= max_team_size:
            break
        team.append(n)
        covered |= G.nodes[n].get("skills", set()) & required

    return team


def evaluate_team(
    G: nx.Graph,
    team: List[int],
    required: Set[str],
    disruption_weight: float = 0.5,
    skill_weight: float = 2.0,
) -> Dict:
    """Score d'une équipe avec les mêmes métriques que l'env RL (comparabilité)."""
    from src.graph.metrics import DisruptionTracker, skill_coverage

    tracker = DisruptionTracker(G)
    covered: Set[str] = set()
    for n in team:
        tracker.add(n)
        covered |= G.nodes[n].get("skills", set())

    cov = skill_coverage(covered, required)
    disr = tracker.disruption()
    return {
        "team_size": len(team),
        "skill_coverage": cov,
        "disruption": disr,
        "score": skill_weight * cov - disruption_weight * disr,
        "complete": covered >= required,
    }
