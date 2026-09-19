# src/graph/metrics.py
"""Métriques de perturbation organisationnelle et de couverture de compétences.

Fix majeur vs PoC initial : la perturbation est maintenue de façon
*incrémentale* (O(deg) par ajout) au lieu d'un recalcul global O(V+E) à chaque
step.

Nouveauté : méthode `remove()` symétrique de `add()` pour supporter l'action
"remove" de GCO-DQN (l'agent peut retirer un employé déjà sélectionné).
"""
from __future__ import annotations

from typing import Dict, Set

import networkx as nx


class DisruptionTracker:
    """Suivi incrémental de la perturbation organisationnelle.

    Définition (identique au PoC initial, mais calcul incrémental) :
    perturbation = moyenne, sur les départements affectés, de la fraction
    d'arêtes internes "coupées" (une extrémité sélectionnée, l'autre non).

    Invariant : `add()` puis `remove()` sur le même nœud ramène l'état
    exactement à ce qu'il était avant.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph

        # Par département : nœuds, arêtes internes
        self._dept_nodes: Dict[str, set] = {}
        self._dept_edges: Dict[str, set] = {}
        for u, v in graph.edges():
            du = graph.nodes[u].get("department", "unknown")
            dv = graph.nodes[v].get("department", "unknown")
            if du == dv:
                self._dept_edges.setdefault(du, set()).add((u, v))

        for n, data in graph.nodes(data=True):
            self._dept_nodes.setdefault(data.get("department", "unknown"), set()).add(n)

        # État courant
        self.selected: Set[int] = set()
        self._cut_edges_by_dept: Dict[str, int] = {}

    def reset_state(self):
        """Remet l'état à zéro SANS reconstruire l'index (O(1) au lieu de O(E)).

        L'index des départements/arêtes internes est statique : il est calculé
        une seule fois dans __init__ et réutilisé à chaque épisode.
        """
        self.selected.clear()
        self._cut_edges_by_dept.clear()

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #

    def add(self, node: int):
        """Ajoute un employé sélectionné et met à jour la perturbation en O(deg).

        Pour chaque voisin `nb` non sélectionné du même département, l'arête
        interne (node, nb) devient coupée → +1 sur le compteur du département.
        """
        if node in self.selected:
            return
        self.selected.add(node)

        dept = self.graph.nodes[node].get("department", "unknown")
        cut = self._cut_edges_by_dept.get(dept, 0)

        for nb in self.graph.neighbors(node):
            if nb in self.selected:
                continue
            dnb = self.graph.nodes[nb].get("department", "unknown")
            if dnb == dept:
                cut += 1

        self._cut_edges_by_dept[dept] = cut

    def remove(self, node: int):
        """Retire un employé sélectionné et met à jour la perturbation en O(deg).

        Symétrique exact de `add()`. Si `add()` a été appelé avant `remove()`
        sur le même nœud, l'état est restauré à l'identique.
        """
        if node not in self.selected:
            return
        self.selected.discard(node)

        dept = self.graph.nodes[node].get("department", "unknown")
        cut = self._cut_edges_by_dept.get(dept, 0)

        for nb in self.graph.neighbors(node):
            if nb in self.selected:
                continue
            dnb = self.graph.nodes[nb].get("department", "unknown")
            if dnb == dept:
                # Cette arête était coupée (par node) ; elle ne l'est plus.
                cut -= 1

        # Nettoyage : évite les clés à 0 qui polluent affected_departments.
        if cut <= 0:
            self._cut_edges_by_dept.pop(dept, None)
        else:
            self._cut_edges_by_dept[dept] = cut

    # ------------------------------------------------------------------ #
    # Lectures
    # ------------------------------------------------------------------ #

    @property
    def affected_departments(self) -> Set[str]:
        """Départements ayant au moins un employé sélectionné."""
        return {
            self.graph.nodes[n].get("department", "unknown") for n in self.selected
        }

    def disruption(self) -> float:
        """Moyenne, sur les départements affectés, de la fraction d'arêtes internes coupées."""
        affected = self.affected_departments
        if not affected:
            return 0.0
        total = 0.0
        n_affected_with_edges = 0
        for dept in affected:
            edges = self._dept_edges.get(dept)
            if not edges:
                continue
            total += self._cut_edges_by_dept.get(dept, 0) / len(edges)
            n_affected_with_edges += 1
        if n_affected_with_edges == 0:
            return 0.0
        return total / n_affected_with_edges

    def per_department(self) -> Dict[str, float]:
        """Détail par département affecté (pour l'explicabilité)."""
        out = {}
        for dept in self.affected_departments:
            edges = self._dept_edges.get(dept)
            if edges:
                out[dept] = self._cut_edges_by_dept.get(dept, 0) / len(edges)
        return out

    # ------------------------------------------------------------------ #
    # Debug / tests
    # ------------------------------------------------------------------ #

    def debug_state(self) -> Dict:
        """Retourne l'état interne complet (pour tests de non-régression)."""
        return {
            "selected": sorted(self.selected),
            "cut_edges_by_dept": dict(self._cut_edges_by_dept),
            "disruption": self.disruption(),
        }


def skill_coverage(covered: Set[str], required: Set[str]) -> float:
    """Fraction des compétences projet couvertes."""
    if not required:
        return 1.0
    return len(covered & required) / len(required)