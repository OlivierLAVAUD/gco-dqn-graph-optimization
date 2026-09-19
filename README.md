# PoC GCO-DQN — Formation d'équipes via Deep RL sur graphe organisationnel

Implémentation du PoC décrit dans `poc.txt` (d'après Lv et al., 2024, *Decision Support Systems*), **dockerisée**, avec **Neo4j** comme base de graphes.

Un agent DQN sélectionne itérativement des employés pour former une équipe qui couvre les compétences d'un projet tout en minimisant la perturbation du réseau organisationnel.

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
        │  GNN Encoder (GAT) │───────▶│  DQN (Q par   │──▶ Ajouter employé / Terminer
        │  embeddings nœuds  │        │  nœud)        │
        └────────────────────┘        └───────────────┘
```

## Démarrage rapide (Docker)

```bash
# Tout-en-un : Neo4j + génération + entraînement + baselines
docker compose up --build

# Sorties dans ./outputs/ (courbes, modèle, métriques)
```

L'app attend automatiquement que Neo4j soit prêt (healthcheck), génère l'organisation synthétique, la charge dans Neo4j, puis entraîne le DQN et le compare aux baselines.

## GPU (CUDA)

L'agent choisit automatiquement son device : `cuda` si disponible, sinon `cpu`
(`src/train.py`). Le message `[train] N épisodes sur cuda...` confirme l'usage GPU.

### Prérequis hôte

```bash
nvidia-smi                                    # driver NVIDIA OK ?
nvidia-container-toolkit --version            # toolkit installé ?
docker info | grep -i runtime                 # runtime "nvidia" présent ?
```

### Configuration Docker

Le service `app` demande le GPU via `runtime: nvidia` (et `CUDA_VISIBLE_DEVICES=0`) :

```yaml
services:
  app:
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: "0"
```

Alternative Docker moderne (si `runtime:` est refusé par votre version) :

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Vérifier que le conteneur voit le GPU

```bash
docker run --rm --runtime=nvidia --gpus all poc-gco-dqn-app:latest \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Réglages pour GPU (RTX 3060 6 Go)

`config.yaml` est déjà ajusté pour exploiter le GPU : `batch_size: 64`
(au lieu de 32 en CPU). Le modèle est petit (GNN 64 dim sur 500 nœuds) et le
goulot d'étranglement reste la boucle Python de l'environnement, pas les forwards.

| Paramètre | CPU | GPU |
|---|---|---|
| `batch_size` | 32 | 64 |
| Épisode | ~1.5 s | ~3.8 s (500 épisodes ≈ 30 min) |

> Le build GPU est lourd (~3 Go d'image, ~5 min la première fois) car le wheel
> torch embarque CUDA + cuDNN. Les builds suivants sont mis en cache.


## Utilisation

```bash
# Mode interactif dans le conteneur
docker compose run app --episodes 100          # entraînement court
docker compose run app --generate-only         # peupler Neo4j puis quitter
docker compose run app --eval-only             # évaluer outputs/model.pt

# Hors Docker (Neo4j local requis pour --source neo4j)
pip install -r requirements.txt
python -m src.train --source json              # fallback sans base de graphes
python -m src.train --source neo4j             # via bolt://localhost:7687
```

Sources de données (`--source`) : `json` | `neo4j` | `auto` (défaut : neo4j si dispo).

## Explorer Neo4j

Browser web sur http://localhost:7474 (login `neo4j` / `gcodqn-poc`) :

```cypher
MATCH (e:Employee)-[c:COMMUNICATES_WITH]-(f:Employee) WHERE e.department='dept_3' RETURN e, c, f LIMIT 100
MATCH (e:Employee) RETURN e.department AS dept, count(*) AS effectif ORDER BY effectif DESC
MATCH (e:Employee)-[c:COMMUNICATES_WITH]-() RETURN e.name, count(c) AS degre ORDER BY degre DESC LIMIT 10
```

## Structure

```
├── Dockerfile / docker-compose.yml   # app + Neo4j 5 community
├── config.yaml                       # hyperparamètres (data, env, agent, train)
├── src/
│   ├── data/generate_synthetic.py    # organisation synthétique (seed reproductible)
│   ├── db/neo4j_store.py             # persistance Cypher, export NetworkX
│   ├── graph/metrics.py              # perturbation incrémentale (fix O(V+E))
│   ├── graph/org_graph.py            # baselines : greedy / random / centralité
│   ├── environment/team_formation_env.py
│   ├── models/gnn_encoder.py         # GAT encoder + tête de Q par nœud
│   ├── models/dqn_agent.py           # Double DQN, buffer compact, eps par épisode
│   └── train.py                      # pipeline complet + comparaison baselines
└── outputs/                          # model.pt, training_results.png, métriques
```

## Correctifs vs PoC initial

| Problème du PoC | Fix |
|---|---|
| Graphe statique stocké dans chaque transition (~600 Mo) | Graphe une fois dans l'agent ; buffer ne stocke que les masques |
| MLP sur vecteur aplati de 13k dim | Tête de Q-values **par nœud** sur embeddings GNN |
| 2 × batch_size forwards GNN par step | 1 forward online + 1 target vectorisés |
| Recalcul perturbation O(V+E) par step | `DisruptionTracker` incrémental O(deg) |
| Epsilon décroît par step | Décroissance par **épisode** (`decay_epsilon`) |
| Reward = couverture absolue (biais grosses équipes) | Reward shaping par **delta** de couverture |
| Baselines absentes (section 5.2) | Greedy / Random / Centralité implémentées |
| État env divergent du modèle | Masques compacts partagés agent/env |

## Correctifs « Phase 0 » (bugs trouvés par audit du code)

| Bug | Symptôme | Correctif |
|---|---|---|
| **Bootstrap TD non masqué** : `next_q_online.argmax()` portait sur les `2N+1` actions, y compris les employés déjà sélectionnés (Q hors distribution) | L'agent n'apprenait jamais à terminer → `avg_team_size` bloqué à `max_team_size` (20), **Greedy (5 employés) battait le DQN** | `GCODQNAgent.masked_max_next_q()` : `masked_fill(next_valid < 0.5, -inf)` **avant** l'argmax. Le masque d'actions valides est stocké dans le replay buffer |
| **Action `remove` morte** : `allow_removal=False` par défaut, jamais activé ; le réseau n'avait pas de tête de retrait (sortie `N+1`) | Le « cœur de GCO-DQN » (revenir sur une décision) était inopérant | Tête `remove_head` ajoutée → sortie `2N+1` `[add.., remove.., terminate]`, layout unifié dans l'env |
| **`DisruptionTracker` recréé à chaque `reset()`** → O(E)/épisode, annulant l'optimisation « incrémentale O(deg) » | Coût croissant avec la taille du graphe | `DisruptionTracker.reset_state()` : index statique réutilisé (reset O(1)) |
| **Reward auto-destructeur** : `size_penalty` > gain marginal d'un employé utile → aucun ajout rentable | L'agent termine immédiatement (équipe = 0) | `size_penalty` maintenu à `0.15` (invariant `skill_weight × 2/n_required > size_penalty`), vérifié par test |
| **Observation incomplète** : le réseau ne recevait QUE la couverture globale sur `n_skills` compétences — **il ne savait pas quelles compétences le projet exige** | Impossible de distinguer « projet couvert » de « compétences au hasard couvertes » → aucune politique de terminaison apprenable (équipes à 20 sans le fix, à 2 après le fix du bootstrap) | `project_skill_mask` exposé par l'env et injecté dans le réseau : `match` = compétence **requise** non couverte, et tête de terminaison sur `[couverture globale, couverture PROJET, mean_emb]` |

Tests de non-régression : `python -m tests.test_fixes` (16 tests).

## Config

Tous les hyperparamètres sont dans `config.yaml` (taille de l'organisation, poids reward, architecture, épisodes...).
