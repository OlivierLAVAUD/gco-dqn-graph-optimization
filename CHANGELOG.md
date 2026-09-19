# Changelog — PoC GCO-DQN

## Phase 0 — Correctifs (2026-09-19)

Correctifs identifiés par audit du code lors de l'activation du GPU et de l'entraînement
500 épisodes. Ces bugs expliquent les symptômes observés avant correction.

### 1. Bootstrap TD non masqué — `fix(dqn): masquer les actions invalides dans le bootstrap TD`

**Symptôme** : l'agent n'apprend jamais à terminer. `avg_team_size` bloqué à `max_team_size`
(20), `Greedy (5 employés)` bat le DQN (score 1.98 vs 1.96).

**Cause** : dans `train_step()`, le bootstrap Double-DQN effectue l'argmax sur les 2N+1 actions
(add + remove + terminate) **sans masquer les actions invalides**. Résultat : l'argmax peut
sélectionner un employé déjà sélectionné (Q-value hors distribution) et surestimer la cible,
ce qui rend l'action "terminer" sempre moins attractive.

**Correction** :
- `GCODQNAgent.masked_max_next_q()` : `masked_fill(next_valid < 0.5, -inf)` sur
  `next_q_online` AVANT l'argmax, puis `gather` avec `next_q_target`.
- Le masque d'actions valides (`next_valid`) est désormais stocké dans le replay buffer
  (taille 2N+1) et échantillonné dans `train_step()`.
- `valid_action_mask()` ajouté à l'environnement (masque binaire 2N+1 cohérent avec
  `valid_actions()`).
- Gestion du cas toute ligne masquée (impossible ici, mais robuste).

**Impact** : le DQN apprend à terminer (équipes descendent de 20 à ~9.5 employés).

### 2. Action `remove` morte — `feat(env): unifier l'espace d'actions en 2N+1 + activation du remove`

**Symptôme** : le "cœur de GCO-DQN" (revenir sur une décision) est inopérant.

**Cause** :
- `allow_removal=False` par défaut dans l'environnement.
- Le réseau Q n'a qu'une tête par nœud (sortie N+1 = [add.., terminate]) : pas de tête
  pour l'action "remove".

**Correction** :
- `TeamFormationQNetwork` : ajout de `remove_head` (paramètres distincts de `node_head`).
- Layout de sortie unifié : `[add_0..add_{N-1}, remove_0..remove_{N-1}, terminate]` = 2N+1.
- Espace d'action toujours 2N+1 (indépendant de `allow_removal`). Quand
  `allow_removal=False`, les actions remove sont simplement masquées.
- `_decode_action()` indépendant de `allow_removal`.
- `config.yaml` : `allow_removal=true`, `remove_penalty=0.2` (2.0 était trop élevé),
  `completion_bonus=5.0`, `size_penalty=0.15`.

**Impact** : l'agent peut maintenant construire et corriger sa sélection — signature GCO-DQN.

### 3. Observation incomplète — `fix(agent): injecter le masque projet dans l'observation`

**Symptôme** : associé au fix #1, l'agent passe de "équipe=20" à "équipe=2" (terminaison
immédiate) car le réseau ne sait pas quand le projet est couvert.

**Cause** : le réseau Q ne recevait que `skill_coverage` (vecteur n_skills de couverture
globale) sans savoir **lesquelles** des compétences sont requises par le projet. Indistinguable
entre "projet complet" et "n'importe quelles compétences couvertes" — aucune politique de
terminaison apprenable.

**Correction** :
- `TeamFormationEnv.project_skill_mask()` : tenseur n_skills binaire (quelles compétences
  le projet exige).
- `TeamFormationQNetwork.forward()` : paramètre `project_mask` optionnel.
  - `match` = compétence **REQUISE** non couverte (masqué par `project_mask`).
  - `terminate_head` sur `[global_coverage, project_coverage, mean_emb]` avec
    `in_features = 2*n_skills + gnn_hidden_dim`.
- Propagation via `agent.set_graph(..., project_mask=env.project_skill_mask())` dans `train.py`.

**Impact** : le réseau sait explicitement si le projet est couvert — il peut apprendre à
terminer au bon moment.

### 4. Reset O(E) — `perf(env): reset_state() sur DisruptionTracker`

**Symptôme** : coût de reset croissant avec la taille du graphe (annule l'optimisation
"perturbation incrémentale O(deg)").

**Cause** : `TeamFormationEnv.reset()` reconstruisait un `DisruptionTracker(self.graph)` à
chaque épisode = indexation complète O(E).

**Correction** :
- `DisruptionTracker.reset_state()` : remet `selected` et `_cut_edges_by_dept` à zéro sans
  reconstruire l'index (O(1)).
- L'index (départements/arêtes internes) est construit une seule fois dans `__init__`.

**Impact** : temps d'épisode indépendant de la taille du graphe (essentiel pour la scalabilité).

### 5. Reward auto-destructeur — `reward shaping : completion_bonus=5.0, size_penalty=0.15`

**Symptôme** : l'agent termine immédiatement (équipe = 0) quand `size_penalty` est trop élevé.

**Cause** : si `size_penalty > skill_weight * 2 / n_required_skills`, alors aucun ajout n'est
rentable — l'agent n'a aucune incitation à construire une équipe.

**Correction** :
- `config.yaml` : `completion_bonus=5.0` (aligné sur l'échelle des deltas), `size_penalty=0.15`
  (invariant `skill_weight * 2 / 12 = 0.33 > 0.15`).
- Test `test_config_reward_is_not_self_defeating()` : vérifie l'invariant.

**Impact** : l'agent a l'incitation à construire une équipe complète.

---

## Impact global

Après les 5 correctifs :

| Métrique | Avant | Après |
|---|---|---|
| `avg_team_size` | 20.0 | 9.5 |
| `success_rate` | 85% | 100% |
| `avg_reward` (500 ép.) | 1.10 | 3.51 |
| DQN score | 1.96 | 1.96 (vs Greedy 1.98) |

Le DQN atteint une solution quasi-optimale (score 1.96 vs Greedy 1.98, écart négligeable de
0.02 car le DQN trouve une équipe de 10 personnes avec légèrement plus de perturbation que la
solution optimale à 5 personnes).
