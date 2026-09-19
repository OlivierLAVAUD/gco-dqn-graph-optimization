# src/data/generate_synthetic.py
"""Génération d'une organisation synthétique (employés, skills, communications).

Version corrigée du PoC :
- seed reproductible (numpy Generator, plus de np.random.seed global)
- génération d'arêtes optimisée (groupée par département, évite le double scan)
- biais de compétences dispersé : chaque département a 8 skills "favorites"
  choisies aléatoirement (évite les clusters contigus qui rendent le problème
  trivial : un employé pouvait couvrir 4/5 skills projet à lui seul)
- export JSON ; le push Neo4j est géré par src/db/neo4j_store.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Set

import networkx as nx
import numpy as np


@dataclass
class Employee:
    id: int
    name: str
    department: str
    skills: Set[str] = field(default_factory=set)
    level: int = 3              # 1-5
    communication_degree: int = 0


def generate_organization(
    n_employees: int = 200,
    n_departments: int = 8,
    n_skills: int = 15,
    avg_skills_per_emp: int = 4,
    communication_prob: float = 0.02,
    seed: int | None = 42,
) -> tuple[nx.Graph, List[Employee], List[str]]:
    """Génère une organisation synthétique sous forme de graphe.

    Returns:
        G: networkx.Graph avec attributs sur les nœuds
        employees: liste des employés
        all_skills: liste ordonnée de toutes les compétences
    """
    rng = np.random.default_rng(seed)

    all_skills = [f"skill_{i}" for i in range(n_skills)]
    departments = [f"dept_{i}" for i in range(n_departments)]

    # Pré-calcul : chaque département a un ensemble de "skills favorites"
    # choisi aléatoirement dans l'espace des skills (pas contigu).
    # → réaliste : un département Data Science aime Python, ML, SQL, etc.
    #   mais ces skills ne sont pas contigus dans l'espace des compétences.
    n_favorites = min(8, n_skills)
    dept_favorites: List[np.ndarray] = [
        rng.choice(n_skills, size=n_favorites, replace=False)
        for _ in range(n_departments)
    ]

    G = nx.Graph()
    employees: List[Employee] = []

    for i in range(n_employees):
        # Assigner un département et des compétences (avec biais par département)
        dept_idx = int(rng.integers(0, n_departments))
        dept = departments[dept_idx]

        # Biais doux : les skills favorites du département ont un poids ×2
        # (au lieu de ×3 sur 3 skills contiguës → clusters artificiels)
        skill_weights = np.ones(n_skills)
        skill_weights[dept_favorites[dept_idx]] *= 2.0
        skill_weights /= skill_weights.sum()

        n_skills_emp = int(rng.poisson(avg_skills_per_emp))
        n_skills_emp = max(1, min(n_skills_emp, n_skills))
        skills = set(rng.choice(
            all_skills, size=n_skills_emp, replace=False, p=skill_weights
        ).tolist())

        level = int(rng.integers(1, 6))
        emp = Employee(
            id=i,
            name=f"Employee_{i}",
            department=dept,
            skills=skills,
            level=level,
        )
        employees.append(emp)

        G.add_node(
            i,
            department=dept,
            skills=skills,
            level=level,
            name=emp.name,
            skill_vector=np.array([1 if s in skills else 0 for s in all_skills]),
        )

    # Créer les arêtes (communications)
    # Biais : plus de communications intra-département, moins inter-dept,
    # atténuées par l'écart de niveau hiérarchique.
    for d_idx, dept in enumerate(departments):
        members = [e.id for e in employees if e.department == dept]
        # Intra-département
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                p = communication_prob * 8
                p *= 1.0 / (1.0 + abs(employees[i].level - employees[j].level) * 0.3)
                if rng.random() < p:
                    G.add_edge(i, j, weight=1.0)

    # Inter-départements : échantillonnage creux (évite le scan O(n²) complet)
    n_inter = int(n_employees * (n_employees - 1) / 2 * communication_prob * 0.5 * 0.2)
    for _ in range(max(1, n_inter)):
        i, j = rng.integers(0, n_employees, size=2)
        if i == j:
            continue
        i, j = int(i), int(j)
        p = communication_prob * 0.5
        p *= 1.0 / (1.0 + abs(employees[i].level - employees[j].level) * 0.3)
        if rng.random() < p:
            G.add_edge(i, j, weight=1.0)

    # Mettre à jour les degrés
    for emp in employees:
        emp.communication_degree = G.degree(emp.id)

    return G, employees, all_skills


def generate_project_requirements(
    all_skills: List[str],
    n_required_skills: int = 5,
    seed: int | None = None,
) -> Set[str]:
    """Génère les compétences requises pour un projet."""
    rng = np.random.default_rng(seed)
    return set(rng.choice(all_skills, size=n_required_skills, replace=False).tolist())


def organization_to_dict(
    G: nx.Graph,
    employees: List[Employee],
    all_skills: List[str],
    project_skills: Set[str],
) -> dict:
    """Sérialise l'organisation pour export JSON / push Neo4j."""
    return {
        "n_nodes": G.number_of_nodes(),
        "edges": [[int(u), int(v)] for u, v in G.edges()],
        "nodes": [
            {
                "id": e.id,
                "name": e.name,
                "department": e.department,
                "skills": sorted(e.skills),
                "level": e.level,
            }
            for e in employees
        ],
        "all_skills": all_skills,
        "project_skills": sorted(project_skills),
    }


def main(json_path: str = "data/sample_org.json", seed: int = 42) -> dict:
    G, employees, all_skills = generate_organization(seed=seed)
    project_skills = generate_project_requirements(all_skills, seed=seed)

    print(f"Organisation: {G.number_of_nodes()} employés, {G.number_of_edges()} connexions")
    print(f"Compétences projet requises: {sorted(project_skills)}")

    data = organization_to_dict(G, employees, all_skills, project_skills)

    import os
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Sauvegardé dans {json_path}")
    return data


if __name__ == "__main__":
    main()