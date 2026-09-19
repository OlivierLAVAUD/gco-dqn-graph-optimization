# src/models/dqn_agent.py
"""Agent DQN pour la formation d'équipe.

Fixes vs PoC initial :
- Replay buffer **compact** : on ne stocke que selected_mask + skill_coverage
  (le graphe statique vit dans l'agent, une seule fois) — ~600 Mo économisés.
- train_step vectorisé : un seul forward online + un seul forward target pour
  tout le batch (au lieu de 2 × batch_size forwards GNN).
- Epsilon décroît par **épisode** (decay_epsilon), plus par step d'entraînement.
- Double DQN léger : la cible utilise l'action argmax du réseau online, évaluée
  par le réseau target (réduit la sur-estimation des Q-values).
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.gnn_encoder import TeamFormationQNetwork


class ReplayBuffer:
    """Buffer compact : (selected_mask, skill_coverage, action, reward, done,
    next_selected_mask, next_skill_coverage, next_valid_mask).

    Seuls les masques d'état sont stockés (quelques Ko par transition) ; le
    graphe (node_features, edge_index, node_skill_matrix) est statique et vit
    une seule fois dans l'agent — vs ~600 Mo dans le PoC initial.

    ``next_valid_mask`` est indispensable : sans lui, l'argmax du bootstrap TD
    porte sur des actions invalides (employé déjà sélectionné) dont la Q-value
    est hors distribution → cible surestimée → l'agent n'apprend jamais à
    terminer.
    """

    def __init__(self, capacity: int, n_nodes: int, n_skills: int, n_actions: int):
        self.capacity = capacity
        self.n_nodes = n_nodes
        self.n_skills = n_skills
        self.n_actions = n_actions
        self.masks = np.zeros((capacity, n_nodes), dtype=np.float32)
        self.covs = np.zeros((capacity, n_skills), dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_nodes), dtype=np.float32)
        self.next_covs = np.zeros((capacity, n_skills), dtype=np.float32)
        self.next_valids = np.zeros((capacity, n_actions), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, mask, cov, action, reward, done, next_mask, next_cov, next_valid):
        p = self.pos
        self.masks[p] = mask
        self.covs[p] = cov
        self.next_masks[p] = next_mask
        self.next_covs[p] = next_cov
        self.next_valids[p] = next_valid
        self.actions[p] = action
        self.rewards[p] = reward
        self.dones[p] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.masks[idx], self.covs[idx], self.actions[idx],
            self.rewards[idx], self.dones[idx],
            self.next_masks[idx], self.next_covs[idx], self.next_valids[idx],
        )

    def __len__(self):
        return self.size


class GCODQNAgent:
    """DQN avec réseau Q par nœud et buffer compact."""

    def __init__(
        self,
        n_skills: int,
        n_nodes: int,
        gnn_hidden_dim: int = 64,
        q_hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        device: str = "cpu",
        n_actions: Optional[int] = None,
    ):
        self.n_skills = n_skills
        self.n_nodes = n_nodes
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Graphe statique : transféré une seule fois sur le device
        self.node_features: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.node_skill_matrix: Optional[torch.Tensor] = None
        self.project_mask: Optional[torch.Tensor] = None

        self.q_network = TeamFormationQNetwork(
            n_skills=n_skills,
            gnn_hidden_dim=gnn_hidden_dim,
            q_hidden_dim=q_hidden_dim,
            n_gnn_layers=n_gnn_layers,
            dropout=dropout,
        ).to(self.device)

        self.target_network = TeamFormationQNetwork(
            n_skills=n_skills,
            gnn_hidden_dim=gnn_hidden_dim,
            q_hidden_dim=q_hidden_dim,
            n_gnn_layers=n_gnn_layers,
            dropout=dropout,
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # Layout d'action unifié 2N+1 (add / remove / terminate), même quand
        # allow_removal=False : les "remove" sont alors simplement masqués.
        self.n_actions = n_actions if n_actions is not None else 2 * n_nodes + 1
        self.replay_buffer = ReplayBuffer(
            buffer_capacity, n_nodes, n_skills, self.n_actions
        )
        self._train_steps = 0

    # ------------------------------------------------------------------ #
    # État
    # ------------------------------------------------------------------ #

    def set_graph(
        self,
        node_features: np.ndarray,      # [N, S+2]
        edge_index: np.ndarray,         # [2, E]
        node_skill_matrix: np.ndarray,  # [N, S]
        project_mask: Optional[np.ndarray] = None,  # [S] compétences requises
    ):
        """Enregistre le graphe statique (une seule fois, sur le device)."""
        self.node_features = torch.as_tensor(node_features, dtype=torch.float32,
                                             device=self.device)
        self.edge_index = torch.as_tensor(edge_index, dtype=torch.long,
                                          device=self.device)
        self.node_skill_matrix = torch.as_tensor(node_skill_matrix, dtype=torch.float32,
                                                 device=self.device)
        self.project_mask = (
            torch.as_tensor(project_mask, dtype=torch.float32, device=self.device)
            if project_mask is not None else None
        )

    # ------------------------------------------------------------------ #
    # Q-values et sélection d'action
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def q_values(self, selected_mask: np.ndarray, skill_coverage: np.ndarray) -> torch.Tensor:
        return self.q_network(
            self.node_features, self.edge_index, self.node_skill_matrix,
            torch.as_tensor(selected_mask, dtype=torch.float32, device=self.device),
            torch.as_tensor(skill_coverage, dtype=torch.float32, device=self.device),
            project_mask=self.project_mask,
        )

    def select_action(
        self,
        selected_mask: np.ndarray,
        skill_coverage: np.ndarray,
        valid_actions: Sequence[int],
        training: bool = True,
    ) -> int:
        """Epsilon-greedy avec masquage des actions invalides."""
        if training and random.random() < self.epsilon:
            return random.choice(list(valid_actions))

        q = self.q_values(selected_mask, skill_coverage)
        q_masked = torch.full_like(q, float("-inf"))
        q_masked[list(valid_actions)] = q[list(valid_actions)]
        return int(q_masked.argmax().item())

    # ------------------------------------------------------------------ #
    # Buffer + entraînement
    # ------------------------------------------------------------------ #

    def store_transition(self, mask, cov, action, reward, done, next_mask, next_cov,
                         next_valid):
        self.replay_buffer.push(mask, cov, action, reward, done, next_mask, next_cov,
                                next_valid)

    @staticmethod
    def masked_max_next_q(
        next_q_online: torch.Tensor,
        next_q_target: torch.Tensor,
        next_valids: torch.Tensor,
    ) -> torch.Tensor:
        """Bootstrap Double-DQN **masqué**.

        L'action est choisie par le réseau online (argmax) puis évaluée par le
        réseau target. Le masquage AVANT l'argmax est essentiel : sans lui,
        l'argmax peut désigner un employé déjà sélectionné (Q-value hors
        distribution) et surestimer la cible → l'agent n'apprend jamais à
        terminer (équipes toujours à max_team_size).
        """
        masked = next_q_online.masked_fill(next_valids < 0.5, float("-inf"))
        max_q = next_q_target.gather(
            1, masked.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        # Garde-fou : si une ligne est entièrement masquée, argmax() renvoie
        # l'index 0 et gather une valeur arbitraire → on force un bootstrap nul.
        # (En pratique impossible : "terminer" est toujours valide.)
        all_masked = next_valids.sum(dim=1) < 0.5
        return torch.where(all_masked, torch.zeros_like(max_q), max_q)

    def train_step(self) -> float:
        """Une étape d'optimisation vectorisée. Retourne la loss (0 si buffer trop petit)."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        (masks, covs, actions, rewards, dones,
         next_masks, next_covs, next_valids) = self.replay_buffer.sample(self.batch_size)

        masks_t = torch.as_tensor(masks, device=self.device)
        covs_t = torch.as_tensor(covs, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, device=self.device)
        next_covs_t = torch.as_tensor(next_covs, device=self.device)
        next_valids_t = torch.as_tensor(next_valids, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # Q(s, a) pour les actions réellement prises — un seul forward
        q_all = self.q_network(
            self.node_features, self.edge_index, self.node_skill_matrix,
            masks_t, covs_t, project_mask=self.project_mask,
        )
        q_taken = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Cible Double-DQN : argmax par online (état suivant), évaluation par target
        with torch.no_grad():
            next_q_online = self.q_network(
                self.node_features, self.edge_index, self.node_skill_matrix,
                next_masks_t, next_covs_t, project_mask=self.project_mask,
            )
            next_q_target = self.target_network(
                self.node_features, self.edge_index, self.node_skill_matrix,
                next_masks_t, next_covs_t, project_mask=self.project_mask,
            )
            # FIX : bootstrap masqué — l'argmax ne peut plus désigner une
            # action invalide (voir masked_max_next_q pour le détail).
            max_next_q = self.masked_max_next_q(
                next_q_online, next_q_target, next_valids_t
            )
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        """À appeler une fois par ÉPISODE (fix vs PoC initial)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistance
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["q_network"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
