# src/models/gnn_encoder.py
"""Encodeur GNN + réseau Q pour le DQN.

Fixes vs PoC initial :
- Tête de Q-values **par nœud** (au lieu d'un MLP sur le vecteur aplati de
  13k dimensions) : scalable et respecte la structure du graphe.
- Les embeddings GNN ne dépendent que du graphe statique → un seul forward
  GNN par batch d'entraînement (au lieu de 2 × batch_size).
- Entrée dynamique compacte par nœud : [embedding, selected, couvre-skill-manquant].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class EmployeeGNNEncoder(nn.Module):
    """Encode les employés (nœuds) à partir de skills, niveau, degré + structure."""

    def __init__(
        self,
        n_skills: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_gat: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_skills + 2, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            if use_gat:
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout
                ))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [n_nodes, n_skills+2], edge_index: [2, n_edges] → [n_nodes, hidden]."""
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new  # résiduel
        return self.output_proj(h)


class TeamFormationQNetwork(nn.Module):
    """Q-network par nœud (GCO-DQN).

    Layout de sortie : ``[add_0..add_{N-1}, remove_0..remove_{N-1}, terminate]``
    soit ``2N+1`` actions (index ``2N`` = terminer).

    Q(ajouter i)  = f([emb_i, selected_i, apporte_une_skill_manquante_i])
    Q(retirer i)  = h([emb_i, selected_i, apporte_une_skill_manquante_i])
    Q(terminer)   = g([couverture_skills, moyenne des embeddings sélectionnés])

    Le graphe (features + arêtes) est passé une fois par forward ; seul l'état
    de sélection/couverture varie — d'où un seul passage GNN même en batch.
    """

    def __init__(
        self,
        n_skills: int,
        gnn_hidden_dim: int = 64,
        q_hidden_dim: int = 128,
        n_gnn_layers: int = 3,
        dropout: float = 0.1,
        use_gat: bool = True,
    ):
        super().__init__()
        self.gnn = EmployeeGNNEncoder(
            n_skills=n_skills,
            hidden_dim=gnn_hidden_dim,
            n_layers=n_gnn_layers,
            dropout=dropout,
            use_gat=use_gat,
        )

        # Tête par nœud (ajout) : embedding + selected (1) + skill_match (1)
        self.node_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim + 2, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, 1),
        )

        # Tête de retrait (cœur de GCO-DQN) : mêmes features, poids distincts.
        # Sans elle, l'action "remove" n'a pas de Q-value → impossible d'apprendre
        # à revenir sur une décision.
        self.remove_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim + 2, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, 1),
        )

        # Tête de terminaison : couverture globale + couverture PROJET + embedding
        # moyen → 2*n_skills + gnn_hidden_dim
        self.terminate_head = nn.Sequential(
            nn.Linear(2 * n_skills + gnn_hidden_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,      # [N, n_skills+2] (statique)
        edge_index: torch.Tensor,         # [2, E] (statique)
        node_skill_matrix: torch.Tensor,  # [N, n_skills] one-hot skills (statique)
        selected_mask: torch.Tensor,      # [B, N]
        skill_coverage: torch.Tensor,     # [B, n_skills]
        batch_mask: Optional[torch.Tensor] = None,  # None si B=1
        project_mask: Optional[torch.Tensor] = None,  # [n_skills] skills requises
    ) -> torch.Tensor:
        """Retourne les Q-values [B, 2N+1] (add.., remove.., terminer)."""
        single = selected_mask.dim() == 1
        if single:  # uniformiser en batch
            selected_mask = selected_mask.unsqueeze(0)
            skill_coverage = skill_coverage.unsqueeze(0)
        B, N = selected_mask.shape

        # Un seul forward GNN (graphe statique, partagé par tout le batch)
        emb = self.gnn(node_features, edge_index)              # [N, H]

        # Sans le masque projet, le réseau ne peut PAS savoir si le projet est
        # couvert : une couverture globale sur n_skills compétences ne distingue
        # pas "12 compétences requises présentes" de "12 compétences au hasard".
        if project_mask is not None:
            pm = project_mask.unsqueeze(0)                     # [1, S]
            uncovered = (1.0 - skill_coverage) * pm            # [B, S]
            proj_cov = skill_coverage * pm                     # [B, S]
        else:
            uncovered = 1.0 - skill_coverage
            proj_cov = skill_coverage

        # skill_match : le nœud couvre-t-il une compétence REQUISE non couverte ?
        match = torch.clamp(uncovered @ node_skill_matrix.T, max=1.0)  # [B, N]

        # Têtes par nœud, vectorisées sur B et N
        sel = selected_mask.unsqueeze(-1)                      # [B, N, 1]
        mt = match.unsqueeze(-1)                               # [B, N, 1]
        e = emb.unsqueeze(0).expand(B, N, emb.size(-1))        # [B, N, H]
        node_feats = torch.cat([e, sel, mt], dim=-1)           # [B, N, H+2]
        add_q = self.node_head(node_feats).squeeze(-1)         # [B, N]  ajouter i
        rem_q = self.remove_head(node_feats).squeeze(-1)       # [B, N]  retirer i

        # Terminaison : couverture globale + couverture PROJET + embedding moyen
        sel_sum = selected_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_emb = (selected_mask @ emb) / sel_sum             # [B, H]
        term_q = self.terminate_head(
            torch.cat([skill_coverage, proj_cov, mean_emb], dim=-1)
        )                                                      # [B, 1]

        # Layout : [add_0..add_{N-1}, remove_0..remove_{N-1}, terminate] → [B, 2N+1]
        q = torch.cat([add_q, rem_q, term_q], dim=-1)
        return q.squeeze(0) if single else q
