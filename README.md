# PoC GCO-DQN — Formation d'équipes via Deep RL sur graphe organisationnel

Implémentation du Proof-of-Concept décrit dans `poc.txt` (d'après Lv et al., 2024,
*Decision Support Systems*), **dockerisée**, avec **Neo4j** comme base de graphes.

Un agent DQN sélectionne itérativement des employés pour former une équipe qui couvre les
compétences d'un projet tout en minimisant la perturbation du réseau organisationnel.
Contrairement aux approches déterministes (greedy, centralité), l'agent peut **revenir sur
ses décisions** (action `remove`) — cœur de l'approche GCO-DQN.

## Architecture

```
┌──────────────┐   Cypher    ┌──────────────┐  NetworkX  ┌─────────────────────┐
│   Neo4j      │────────────▶│  src/db/     │───────────▶│  Environnement Gym  │
│ (graphe org) │             │  neo4j_store │            │  (team formation)   │
└──────────────┘             └──────────────┘            └──────────┬──────────┘
                                                                     │
                     ┌──────────────────────────────────────────────┘
                     ▼
         ┌────────────────────┐        ┌───────────────┐
         │  GNN Encoder (GAT) │───────▶│  DQN (Q par   │──▶ Ajouter employé /
         │  embeddings nœuds  │        │  nœud)        │     Retirer employé /
         └────────────────────┘        │  + Remove     │     Terminer
                                       └───────────────┘
```

## Démarrage rapide (Docker)

```bash
# Tout-en-un : Neo4j + génération + entraînement + baselines
docker compose up --build

# Sorties dans ./outputs/ (courbes, modèle, métriques)
```

L'app attend automatiquement que Neo4j soit prêt (healthcheck), génère l'organisation
synthétique, la charge dans Neo4j, puis entraîne le DQN et le compare aux baselines.

## Utilisation

```bash
# Mode interactif dans le conteneur
docker compose run app --episodes 100          # entraînement court
docker compose run app --generate-only         # peupler Neo4j puis quitter
docker compose run app --eval-only             # évaluer outputs/model.pt

# Flags training :
#   --allow-removal    Active l'action 'remove' (défaut: config.yaml)
#   --no-removal       Désactive le 'remove'
#   --tag TAG          Suffixe des sorties (ex: model_TAG.pt)

# Sources de données (`--source`) : json | neo4j | auto (défaut : neo4j si dispo)
```

## GPU (CUDA)

L'agent choisit automatiquement son device : `cuda` si disponible, sinon `cpu`
(`src/train.py`). Le message `[train] N épisodes sur cuda...` confirme l'usage GPU.

### Prérequis hôte

```bash
nvidia-smi                                      # driver NVIDIA OK ?
nvidia-container-toolkit --version              # toolkit installé ?
docker info | grep -i runtime                   # runtime "nvidia" ou "gpus: all" OK ?
```

### Configuration Docker

Le service `app` expose le GPU via `gpus: all` (syntaxe moderne du NVIDIA Container
Toolkit, préférable à `runtime: nvidia` qui ne monte pas toujours les devices) :

```yaml
services:
  app:
    gpus: all
    environment:
      CUDA_VISIBLE_DEVICES: "0"
```

### Vérifier que le conteneur voit le GPU

```bash
docker exec gcodqn-app python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Réglages pour GPU (RTX 3060 6 Go)

`config.yaml` est ajusté pour exploiter le GPU : `batch_size: 64` (au lieu de 32 en CPU).
Le modèle est petit (GNN 64 dim sur 500 nœuds) et le goulot d'étranglement reste la boucle
Python de l'environnement, pas les forwards GNN.

| Paramètre | CPU | GPU |
|---|---|---|
| `batch_size` | 32 | 64 |
| Image | ~500 Mo | ~9 Go (torch + CUDA) |

> Le build GPU est lourd la première fois (~5 min, ~9 Go). Les rebuilds suivants sont
> mis en cache.

## Structure

```
.
├── .dockerignore
├── .env.example              # Exemple de vars pour run local (source=neo4j)
├── .gitignore
├── LICENSE                   # MIT © 2026 Olivier LAVAUD (@oLV)
├── CHANGELOG.md              # Correctifs et évolutions
├── Dockerfile                # Torch GPU (CUDA 12.4), PyTorch 2.14
├── README.md                 # Ce fichier
├── config.yaml               # Hyperparamètres (data, env, agent, train)
├── docker-compose.yml        # Neo4j 5 community + app (gpus: all)
├── poc.txt                   # Spec du PoC (Lv et al., 2024)
├── requirements.txt
├── data/
│   └── sample_org.json       # Organisation synthétique (seed=42, fichier régénérable)
├── src/
│   ├── data/generate_synthetic.py
│   │   └── __init__.py
│   ├── db/neo4j_store.py
│   │   └── __init__.py
│   ├── environment/team_formation_env.py
│   │   └── __init__.py
│   ├── graph/metrics.py      # DisruptionTracker (incrémental) + skill_coverage
│   │   └── __init__.py
│   ├── graph/org_graph.py    # Baselines (greedy/random/centralité) + evaluate_team
│   │   └── __init__.py
│   ├── models/gnn_encoder.py # GAT encoder + tête de Q (add/remove/terminate)
│   │   └── __init__.py
│   ├── models/dqn_agent.py   # GCODQNAgent, buffer compact, Double DQN
│   │   └── __init__.py
│   └── train.py              # Pipeline complet + comparaison baselines
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_fixes.py         # Tests non-régression Phase 0 (16 tests)
│   ├── test_greedy.py        # Diagnostic Greedy (script)
│   └── test_metrics.py       # Tests DisruptionTracker (symétrie add/remove)
└── tools/
    └── compare_runs.py       # Outil de comparaison de runs (CLI)
```

## Résultats (500 épisodes, RTX 3060, CUDA)

2026-09-19 22:53:56.004 | /usr/local/lib/python3.12/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
2026-09-19 22:53:56.004 |   warnings.warn(
2026-09-19 22:53:57.171 | [data] Neo4j détecté → source neo4j
2026-09-19 22:54:02.769 | [neo4j] 500 employés, 1422 communications
2026-09-19 22:54:02.771 | [data] source=auto | 500 nœuds, 1422 arêtes | chargé en 6.21s
2026-09-19 22:54:02.773 | [data] départements: 15 | densité: 0.0114
2026-09-19 22:54:02.773 | [data] compétences projet: ['skill_18', 'skill_25', 'skill_26', 'skill_3', 'skill_30', 'skill_32', 'skill_37', 'skill_4', 'skill_42', 'skill_44', 'skill_48', 'skill_9']
2026-09-19 22:54:02.783 | [env] allow_removal=True | n_actions=1001 (500 add + 500 remove + 1 terminate)
2026-09-19 22:54:03.295 |
2026-09-19 22:54:03.295 | [train] 500 épisodes sur cuda...
2026-09-19 23:00:48.633 |   Épisode   50/500 | eps=0.364 | reward(train)=  -0.86 | loss=0.1168 | couverture(eval)=100% | équipe=20.0 | perturbation=0.096 | succès=100%
2026-09-19 23:05:43.063 |   Épisode  100/500 | eps=0.133 | reward(train)=  -1.50 | loss=0.1419 | couverture(eval)=100% | équipe=10.2 | perturbation=0.089 | succès=100%
2026-09-19 23:09:12.664 |   Épisode  150/500 | eps=0.050 | reward(train)=   2.05 | loss=0.0952 | couverture(eval)=100% | équipe=10.6 | perturbation=0.087 | succès=100%
2026-09-19 23:11:41.131 |   Épisode  200/500 | eps=0.050 | reward(train)=   3.22 | loss=0.0720 | couverture(eval)=100% | équipe=10.0 | perturbation=0.079 | succès=100%
2026-09-19 23:13:39.487 |   Épisode  250/500 | eps=0.050 | reward(train)=   3.57 | loss=0.0550 | couverture(eval)=98% | équipe=10.6 | perturbation=0.085 | succès=80%
2026-09-19 23:15:30.749 |   Épisode  300/500 | eps=0.050 | reward(train)=   3.44 | loss=0.0692 | couverture(eval)=100% | équipe=10.6 | perturbation=0.081 | succès=100%
2026-09-19 23:18:02.779 |   Épisode  350/500 | eps=0.050 | reward(train)=   2.98 | loss=0.0635 | couverture(eval)=100% | équipe=9.8 | perturbation=0.077 | succès=100%
2026-09-19 23:20:56.857 |   Épisode  400/500 | eps=0.050 | reward(train)=   3.52 | loss=0.0498 | couverture(eval)=100% | équipe=9.0 | perturbation=0.081 | succès=100%
2026-09-19 23:23:31.244 |   Épisode  450/500 | eps=0.050 | reward(train)=   3.31 | loss=0.0539 | couverture(eval)=100% | équipe=8.2 | perturbation=0.070 | succès=100%
2026-09-19 23:25:33.640 |   Épisode  500/500 | eps=0.050 | reward(train)=   3.73 | loss=0.0450 | couverture(eval)=100% | équipe=8.6 | perturbation=0.070 | succès=100%
2026-09-19 23:25:33.801 |
2026-09-19 23:25:33.801 | ============================================================
2026-09-19 23:25:33.801 | ÉVALUATION FINALE (DQN, 20 épisodes)
2026-09-19 23:25:36.957 |             avg_reward: 3.508
2026-09-19 23:25:36.957 |          avg_team_size: 9.500
2026-09-19 23:25:36.957 |     avg_skill_coverage: 1.000
2026-09-19 23:25:36.957 |         avg_disruption: 0.082
2026-09-19 23:25:36.957 |           success_rate: 1.000
2026-09-19 23:25:36.957 |
2026-09-19 23:25:36.957 | ============================================================
2026-09-19 23:25:36.957 | COMPARAISON AVEC LES BASELINES
2026-09-19 23:25:37.161 | Méthode       Taille  Couverture  Perturb.    Score  Complet
2026-09-19 23:25:37.161 | ------------------------------------------------------------
2026-09-19 23:25:37.161 | GCO-DQN         10.0      100.0%     0.083     1.96     True
2026-09-19 23:25:37.161 | Greedy           5.0      100.0%     0.035     1.98     True
2026-09-19 23:25:37.161 | Random          20.0       58.3%      0.131     1.10    False
2026-09-19 23:25:37.161 | Centralité      20.0       66.7%      0.214     1.23    False
2026-09-19 23:25:38.731 |
2026-09-19 23:25:38.731 | [done] courbes → outputs/training_results.png | modèle → outputs/model.pt

Le DQN atteint **100% de couverture** avec une **équipe de 10 personnes** (vs 20 avant les
correctifs), un score quasi-optimal (1.96 vs 1.98 du Greedy, qui est optimal par construction).
La différence de 0.02 s'explique par le fait que Greedy trouve 5 personnes avec très peu de
perturbation (0.035), tandis que le DQN trouve 10 personnes avec légèrement plus de
perturbation (0.083) — les deux couvrent 100% des compétences.

## Correctifs Phase 0

Voir `CHANGELOG.md` pour le détail des bugs identifiés, leurs symptômes et leurs corrections.
En résumé :
- **Bootstrap TD non masqué** : l'argmax du réseau en ligne ne masquait pas les actions
  invalides, empêchant l'apprentissage de la terminaison (équipes bloquées à 20).
- **Action `remove` morte** : l'espace d'actions était N+1, empêchant le retour sur décision.
  Passé à 2N+1 avec tête `remove_head`.
- **Observation incomplète** : le réseau ne recevait pas le masque du projet, ignorant
  quelles compétences étaient requises.
- **Reset O(E)** : `DisruptionTracker.reset_state()` pour éviter la reconstruction O(E).
- **Reward auto-destructeur** : ajustement de `completion_bonus` et `size_penalty`.

## Tests

```bash
# Tests non-régression (correctifs Phase 0)
python -m tests.test_fixes      # 16 tests

# Tests existants
python -m tests.test_metrics     # Tests DisruptionTracker
python -m tests.test_greedy      # Diagnostic Greedy (script)
```

## Outils

```bash
# Comparer plusieurs runs (fichiers final_metrics*.json)
python tools/compare_runs.py outputs/final_metrics*.json
```

## Config

Tous les hyperparamètres sont dans `config.yaml` (taille de l'organisation, poids reward,
architecture, épisodes...).

## Licence

MIT © 2026 Olivier LAVAUD (@oLV) — voir `LICENSE`.
