# src/db/neo4j_store.py
"""Couche de persistance Neo4j pour le graphe organisationnel.

Rôle dans l'architecture :
- Charger le graphe (nœuds employés + arêtes de communication) dans Neo4j
- L'exporter vers NetworkX pour l'entraînement (interface identique au JSON)
- Exposer des requêtes utiles : sous-graphe par département, centralité, voisins

Le driver est thread-safe et conçu pour être réutilisé ; on le ferme via un
context manager ou en fin de programme.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Optional

import networkx as nx

from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "gcodqn-poc")


class Neo4jStore:
    """Wrapper minimal autour du driver Neo4j pour le graphe organisationnel."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #

    def close(self):
        self.driver.close()

    def __enter__(self):
        self.driver.verify_connectivity()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @staticmethod
    def is_available(uri: str = NEO4J_URI, user: str = NEO4J_USER,
                     password: str = NEO4J_PASSWORD, timeout: float = 3.0) -> bool:
        """Vérifie rapidement qu'un serveur Neo4j est joignable."""
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password),
                                          connection_timeout=timeout)
            driver.verify_connectivity()
            driver.close()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Écriture : schéma + chargement
    # ------------------------------------------------------------------ #

    def clear(self):
        """Vide la base (PoC uniquement : DELETE tout le graphe)."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def create_schema(self):
        """Contraintes et index pour la recherche rapide."""
        statements = [
            "CREATE CONSTRAINT employee_id IF NOT EXISTS "
            "FOR (e:Employee) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX employee_dept IF NOT EXISTS FOR (e:Employee) ON (e.department)",
            "CREATE INDEX communication_weight IF NOT EXISTS FOR ()-[c:COMMUNICATES_WITH]-() ON (c.weight)",
        ]
        with self.driver.session() as s:
            for stmt in statements:
                s.run(stmt)

    def load_organization(
        self,
        nodes: Iterable[dict],
        edges: Iterable[tuple[int, int]],
    ):
        """Charge employés + communications depuis un dict JSON-like.

        nodes : [{"id": int, "name": str, "department": str, "skills": [..], "level": int}, ...]
        edges : [(u, v), ...]
        """
        self.create_schema()

        def _tx(tx, nodes_batch, edges_batch):
            tx.run(
                """
                UNWIND $nodes AS n
                MERGE (e:Employee {id: n.id})
                SET e.name = n.name,
                    e.department = n.department,
                    e.skills = n.skills,
                    e.level = n.level
                """,
                nodes=nodes_batch,
            )
            tx.run(
                """
                UNWIND $edges AS e
                MATCH (a:Employee {id: e[0]}), (b:Employee {id: e[1]})
                MERGE (a)-[c:COMMUNICATES_WITH]->(b)
                SET c.weight = 1.0
                """,
                edges=edges_batch,
            )

        nodes_list = list(nodes)
        edges_list = [(int(u), int(v)) for u, v in edges]

        with self.driver.session() as s:
            # Batchs de 500 pour éviter de saturer la mémoire de la transaction
            for i in range(0, len(nodes_list), 500):
                s.execute_write(_tx, nodes_list[i:i + 500], [])
            for i in range(0, len(edges_list), 500):
                s.execute_write(_tx, [], edges_list[i:i + 500])

    # ------------------------------------------------------------------ #
    # Persistance du projet (compétences requises)
    # ------------------------------------------------------------------ #

    def store_project(self, all_skills: list[str], project_skills: list[str]):
        """Stocke les compétences (globales + requises par le projet)."""
        with self.driver.session() as s:
            s.run(
                "MERGE (p:Project {name: 'current'}) "
                "SET p.all_skills = $all, p.required_skills = $req",
                all=list(all_skills), req=list(project_skills),
            )

    def get_project(self) -> tuple[list[str], set[str]]:
        """Récupère (all_skills, project_skills)."""
        with self.driver.session() as s:
            rec = s.run(
                "MATCH (p:Project {name: 'current'}) "
                "RETURN p.all_skills AS all_skills, p.required_skills AS req"
            ).single()
        if rec is None:
            raise RuntimeError("Aucun projet stocké dans Neo4j (charger l'organisation d'abord)")
        return list(rec["all_skills"] or []), set(rec["req"] or [])

    # ------------------------------------------------------------------ #
    # Lecture : exports NetworkX
    # ------------------------------------------------------------------ #

    def export_graph(self, department: Optional[str] = None) -> nx.Graph:
        """Exporte le graphe (ou un sous-graphe par département) vers NetworkX.

        Non-dirigé : les communications sont symétriques pour le calcul de
        perturbation et le GNN.
        """
        G = nx.Graph()

        node_filter = "WHERE e.department = $dept" if department else ""
        edge_filter = "WHERE a.department = $dept AND b.department = $dept" if department else ""
        params = {"dept": department} if department else {}

        with self.driver.session() as s:
            nodes = s.run(
                f"MATCH (e:Employee) {node_filter} "
                "RETURN e.id AS id, e.name AS name, e.department AS department, "
                "e.skills AS skills, e.level AS level",
                **params,
            ).data()
            edges = s.run(
                f"MATCH (a:Employee)-[c:COMMUNICATES_WITH]-(b:Employee) {edge_filter} "
                "RETURN a.id AS u, b.id AS v, c.weight AS weight",
                **params,
            ).data()

        for n in nodes:
            G.add_node(
                n["id"],
                name=n["name"],
                department=n["department"],
                skills=set(n["skills"] or []),
                level=n["level"],
            )
        for e in edges:
            if e["u"] in G and e["v"] in G:
                G.add_edge(e["u"], e["v"], weight=e["weight"] or 1.0)

        return G

    # ------------------------------------------------------------------ #
    # Requêtes analytiques utiles (baselines / inspection)
    # ------------------------------------------------------------------ #

    def top_degree_employees(self, k: int = 10) -> list[dict]:
        """Employés les plus connectés (baseline 'centralité' côté DB)."""
        with self.driver.session() as s:
            return s.run(
                "MATCH (e:Employee)-[c:COMMUNICATES_WITH]-() "
                "RETURN e.id AS id, e.name AS name, e.department AS department, "
                "count(c) AS degree ORDER BY degree DESC LIMIT $k",
                k=k,
            ).data()

    def neighbors(self, employee_id: int, k_hops: int = 1) -> list[dict]:
        """Voisins à k sauts d'un employé (utile pour inspecter l'équipe choisie)."""
        with self.driver.session() as s:
            return s.run(
                "MATCH (e:Employee {id: $id})-[c:COMMUNICATES_WITH*1..%d]-(n:Employee) "
                "WHERE n.id <> $id "
                "RETURN DISTINCT n.id AS id, n.name AS name, n.department AS department" % k_hops,
                id=employee_id,
            ).data()

    def stats(self) -> dict:
        """Compteurs rapides pour vérifier le chargement."""
        with self.driver.session() as s:
            s.run("MATCH (p:Project) RETURN count(p) AS c").single()
            n_nodes = s.run("MATCH (e:Employee) RETURN count(e) AS c").single()["c"]
            n_edges = s.run(
                "MATCH ()-[c:COMMUNICATES_WITH]->() RETURN count(c) AS c"
            ).single()["c"]
        return {"nodes": n_nodes, "edges": n_edges}


@contextmanager
def open_store(uri: str = NEO4J_URI, user: str = NEO4J_USER,
               password: str = NEO4J_PASSWORD):
    """Context manager pratique : `with open_store() as store: ...`"""
    store = Neo4jStore(uri, user, password)
    try:
        yield store
    finally:
        store.close()
