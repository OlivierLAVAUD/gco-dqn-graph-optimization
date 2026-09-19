# src/environment/team_formation_env.py
"""Environnement Gym de formation d'équipe.

Fixes vs PoC initial :
- Perturbation calculée de façon incrémentale (DisruptionTracker) au lieu
  d'un recalcul O(V+E) à chaque step.
- Reward shaping par *delta* de couverture : l'agent est récompensé pour les
  nouvelles compétences apportées, ce qui évite le biais "grosses équipes".
- Récompense de couverture retirée à l'action "terminer" si incomplet
  (sinon terminer immédiatement donne presque le même reward que continuer).

Nouveauté (optionnelle) :
- Action "remove" (allow_removal=True) : l'agent peut retirer un employé
  déjà sélectionné. C'est le cœur de GCO-DQN (Lv et al., 2024) : l'agent
  construit incrémentalement la solution ET peut revenir sur ses décisions.

Espace d'action (toujours 2N+1, indépendant de allow_removal) :
    i < N        : ajouter l'employé i
    N <= i < 2N  : retirer l'employé (i - N)   [masqué si allow_removal=False]
    i = 2N       : terminer
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from src.graph.metrics import DisruptionTracker, skill_coverage


class TeamFormationEnv(gym.Env):
    """L'agent ajoute (et optionnellement retire) des employés, puis termine.

    Espace d'action unifié (toujours 2N+1, indépendant de allow_removal) :
        i < N        : ajouter l'employé i
        N <= i < 2N  : retirer l'employé (i - N)   [masqué si allow_removal=False]
        i = 2N       : terminer

    Quand allow_removal=False, les actions "remove" sont décodées mais
    rejetées par valid_actions() → l'agent ne peut jamais les sélectionner.
    """

    TERMINATE = None  # sentinel

    def __init__(
        self,
        graph: nx.Graph,
        project_skills: Set[str],
        all_skills: List[str],
        max_team_size: int = 20,
        skill_weight: float = 1.0,
        disruption_weight: float = 0.5,
        step_penalty: float = 0.01,
        completion_bonus: float = 10.0,
        size_penalty: float = 0.1,
        delay_penalty: float = 1.0,
        # --- Nouveaux paramètres (action "remove") ---
        allow_removal: bool = False,
        remove_penalty: float = 2.0,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.graph = graph
        self.project_skills = set(project_skills)
        self.all_skills = list(all_skills)
        self.skill_index = {s: i for i, s in enumerate(self.all_skills)}
        self.max_team_size = max_team_size
        self.skill_weight = skill_weight
        self.disruption_weight = disruption_weight
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus
        self.size_penalty = size_penalty
        self.delay_penalty = delay_penalty

        # Nouveaux params
        self.allow_removal = allow_removal
        self.remove_penalty = remove_penalty
        # Garde-fou anti-oscillation : borne dure sur le nombre d'actions
        self.max_steps = max_steps if max_steps is not None else 3 * max_team_size

        self.n_nodes = graph.number_of_nodes()
        self.n_skills = len(self.all_skills)

        # Espace d'action unifié (taille constante 2N+1) :
        #   [add_0..add_{N-1}, remove_0..remove_{N-1}, terminate]
        # Les "remove" sont simplement masqués par valid_actions() quand
        # allow_removal=False → le layout reste aligné sur la tête Q du réseau.
        self.n_actions = 2 * self.n_nodes + 1
        self.terminate_action = 2 * self.n_nodes

        # Pré-calcul des features statiques des nœuds
        self._static_node_features = self._build_node_features()

        self.action_space = spaces.Discrete(self.n_actions)
        obs_dim = self.n_nodes + self.n_skills + self.n_nodes
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------ #
    # Décodage d'action
    # ------------------------------------------------------------------ #

    def _decode_action(self, action: int) -> Tuple[str, int | None]:
        """Retourne ('add'|'remove'|'terminate', idx).

        Le décodage est indépendant de allow_removal : une action "remove" est
        décodée puis rejetée par valid_actions() si le retrait est désactivé.
        """
        if action < self.n_nodes:
            return ("add", action)
        if action < 2 * self.n_nodes:
            return ("remove", action - self.n_nodes)
        return ("terminate", None)

    # ------------------------------------------------------------------ #
    # Features statiques
    # ------------------------------------------------------------------ #

    def _build_node_features(self) -> np.ndarray:
        """[n_nodes, n_skills + 2] : skills one-hot + level normalisé + degré normalisé."""
        feats = np.zeros((self.n_nodes, self.n_skills + 2), dtype=np.float32)
        for n, data in self.graph.nodes(data=True):
            for s in data.get("skills", set()):
                idx = self.skill_index.get(s)
                if idx is not None:
                    feats[n, idx] = 1.0
            feats[n, self.n_skills] = data.get("level", 3) / 5.0
            feats[n, self.n_skills + 1] = self.graph.degree(n) / max(1, self.n_nodes - 1)
        return feats

    @property
    def static_node_features(self) -> np.ndarray:
        return self._static_node_features

    def edge_index(self) -> np.ndarray:
        """[2, 2*n_edges] : arêtes symétrisées pour PyG (cache après 1er appel)."""
        if not hasattr(self, "_edge_index_cache"):
            edges = list(self.graph.edges())
            if not edges:
                self._edge_index_cache = np.zeros((2, 0), dtype=np.int64)
            else:
                src = [u for u, v in edges] + [v for u, v in edges]
                dst = [v for u, v in edges] + [u for u, v in edges]
                self._edge_index_cache = np.array([src, dst], dtype=np.int64)
        return self._edge_index_cache

    def project_skill_mask(self) -> np.ndarray:
        """[n_skills] masque binaire des compétences requises par le projet.

        Indispensable au réseau : sans lui, il ne peut pas distinguer « projet
        couvert » de « compétences au hasard couvertes » → il n'apprend jamais
        à terminer.
        """
        m = np.zeros(self.n_skills, dtype=np.float32)
        for sk in self.project_skills:
            idx = self.skill_index.get(sk)
            if idx is not None:
                m[idx] = 1.0
        return m

    # ------------------------------------------------------------------ #
    # Cycle de vie Gym
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.selected: Set[int] = set()
        self.covered_skills: Set[str] = set()
        # Compteur de skills : permet le retrait incrémental sans recalcul
        self._skill_count: Dict[str, int] = {}
        # Le tracker (indexation départements/arêtes en O(E)) est construit une
        # seule fois, puis seulement remis à zéro en O(1) → plus de O(E)/épisode.
        if getattr(self, "tracker", None) is None:
            self.tracker = DisruptionTracker(self.graph)
        else:
            self.tracker.reset_state()
        self.current_step = 0
        return self._get_observation(), self._info()

    def _get_observation(self) -> np.ndarray:
        selection = np.zeros(self.n_nodes, dtype=np.float32)
        for i in self.selected:
            selection[i] = 1.0

        coverage = np.zeros(self.n_skills, dtype=np.float32)
        for s in self.covered_skills:
            idx = self.skill_index.get(s)
            if idx is not None:
                coverage[idx] = 1.0

        degrees = self._static_node_features[:, self.n_skills + 1]
        return np.concatenate([selection, coverage, degrees]).astype(np.float32)

    def _info(self) -> Dict:
        return {
            "selected": list(self.selected),
            "covered_skills": set(self.covered_skills),
            "skill_coverage": skill_coverage(self.covered_skills, self.project_skills),
            "disruption": self.tracker.disruption(),
            "team_size": len(self.selected),
        }

    # ------------------------------------------------------------------ #
    # Mise à jour incrémentale de la couverture
    # ------------------------------------------------------------------ #

    def _add_skills(self, idx: int) -> Set[str]:
        """Ajoute les skills du nœud idx. Retourne les nouvelles skills projet couvertes."""
        emp_skills = self.graph.nodes[idx].get("skills", set())
        new_project_skills = (emp_skills & self.project_skills) - self.covered_skills

        for sk in emp_skills:
            self._skill_count[sk] = self._skill_count.get(sk, 0) + 1
            if self._skill_count[sk] == 1:
                self.covered_skills.add(sk)
        return new_project_skills

    def _remove_skills(self, idx: int) -> Set[str]:
        """Retire les skills du nœud idx. Retourne les skills projet perdues."""
        emp_skills = self.graph.nodes[idx].get("skills", set())
        lost_project_skills: Set[str] = set()

        for sk in emp_skills:
            self._skill_count[sk] = self._skill_count.get(sk, 0) - 1
            if self._skill_count[sk] <= 0:
                self._skill_count.pop(sk, None)
                self.covered_skills.discard(sk)
                if sk in self.project_skills:
                    lost_project_skills.add(sk)
        return lost_project_skills

    # ------------------------------------------------------------------ #
    # Récompense
    # ------------------------------------------------------------------ #

    def _reward(
        self,
        op: str,
        new_skills: Set[str],
        lost_skills: Set[str],
        terminated: bool,
    ) -> float:
        """Reward = gain de couverture - pénalité perturbation - coût step (+ bonus)."""
        r = -self.step_penalty

        # Coût spécifique pour l'action "remove" (décourage le yo-yo)
        if op == "remove":
            r -= self.remove_penalty

        if terminated:
            # Termine : bonus si complet, forte pénalité sinon.
            if self.covered_skills >= self.project_skills:
                r += self.completion_bonus
            else:
                r -= self.skill_weight * skill_coverage(
                    self.project_skills - self.covered_skills,
                    self.project_skills,
                )
        else:
            # Delta de couverture
            if new_skills:
                r += self.skill_weight * len(new_skills) / len(self.project_skills)
            # Pénalité si on perd une compétence projet en retirant
            if lost_skills:
                r -= self.skill_weight * len(lost_skills) / len(self.project_skills)

            # Pénalité de perturbation marginale
            r -= self.disruption_weight * max(
                0.0, self.tracker.disruption()
            ) / max(1, len(self.tracker.affected_departments))

            # Coût de recrutement (pression vers équipes minimales)
            r -= self.size_penalty

            # Recruter/retirer alors que la couverture est complète retarde le
            # bonus : pénalité forte pour accélérer le choix "terminer".
            if self.covered_skills >= self.project_skills:
                r -= self.delay_penalty

        return float(r)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        terminated = False
        truncated = False
        new_skills: Set[str] = set()
        lost_skills: Set[str] = set()

        op, idx = self._decode_action(action)

        if op == "terminate":
            terminated = True

        elif op == "add":
            if idx in self.selected:
                # Action invalide : pénalité légère, l'épisode continue
                obs, info = self._get_observation(), self._info()
                return obs, -1.0, False, False, {**info, "error": "already_selected"}
            self.selected.add(idx)
            self.tracker.add(idx)
            new_skills = self._add_skills(idx)
            if len(self.selected) >= self.max_team_size:
                terminated = True

        elif op == "remove":
            if idx not in self.selected:
                # Action invalide : pénalité légère, l'épisode continue
                obs, info = self._get_observation(), self._info()
                return obs, -1.0, False, False, {**info, "error": "not_selected"}
            self.selected.discard(idx)
            self.tracker.remove(idx)
            lost_skills = self._remove_skills(idx)

        # Garde-fou anti-oscillation
        if self.current_step >= self.max_steps:
            truncated = True

        reward = self._reward(op, new_skills, lost_skills, terminated)
        info = self._info()
        info["new_skills"] = sorted(new_skills)
        info["lost_skills"] = sorted(lost_skills)
        info["op"] = op
        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers pour l'agent
    # ------------------------------------------------------------------ #

    def valid_actions(self) -> List[int]:
        """Actions valides selon l'état courant."""
        valid: List[int] = [i for i in range(self.n_nodes) if i not in self.selected]
        if self.allow_removal:
            valid += [self.n_nodes + i for i in self.selected]
        valid.append(self.terminate_action)
        return valid

    def valid_action_mask(self) -> np.ndarray:
        """Masque binaire de taille 2N+1 des actions valides.

        Utilisé pour le bootstrap TD (argmax masqué) — sans ce masque, la cible
        peut être maximisée sur une action invalide (employé déjà sélectionné).
        """
        m = np.zeros(self.n_actions, dtype=np.float32)
        m[self.valid_actions()] = 1.0
        return m

    @property
    def n_actions_total(self) -> int:
        """Nombre total d'actions (utile pour l'agent)."""
        return self.n_actions