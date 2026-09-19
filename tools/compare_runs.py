#!/usr/bin/env python3
"""Compare plusieurs fichiers de métriques GCO-DQN côte à côte.

Usage :
    python tools/compare_runs.py outputs/final_metrics*.json

Affiche un tableau `méthode × run` pour taille / couverture / perturbation /
score, plus la ligne DQN d'évaluation finale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main(paths: list[str]) -> int:
    if not paths:
        paths = sorted(str(p) for p in Path("outputs").glob("final_metrics*.json"))
    if not paths:
        print("Aucun fichier de métriques trouvé.")
        return 1

    runs = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[skip] {p} introuvable")
            continue
        runs[path.name] = _load(path)

    if not runs:
        return 1

    # --- Évaluation finale du DQN ----------------------------------------- #
    print("\n=== ÉVALUATION FINALE (DQN) ===")
    cols = list(runs)
    header = f"{'run':<34} {'taille':>8} {'couv.':>7} {'perturb.':>9} {'succès':>7} {'reward':>8}"
    print(header)
    print("-" * len(header))
    for name in cols:
        d = runs[name].get("dqn", {})
        print(f"{name:<34} "
              f"{d.get('avg_team_size', 0):>8.2f} "
              f"{d.get('avg_skill_coverage', 0):>7.1%} "
              f"{d.get('avg_disruption', 0):>9.3f} "
              f"{d.get('success_rate', 0):>7.1%} "
              f"{d.get('avg_reward', 0):>8.2f}")

    # --- Baselines --------------------------------------------------------- #
    methods: list[str] = []
    for name in cols:
        for m in runs[name].get("baselines", {}):
            if m not in methods:
                methods.append(m)

    print("\n=== BASELINES / SCORE (skill_weight*couv - disruption_weight*disrupt) ===")
    header = f"{'méthode':<14}" + "".join(f"{n[:22]:>24}" for n in cols)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<14}"
        for name in cols:
            b = runs[name].get("baselines", {}).get(m)
            if b is None:
                row += f"{'-':>24}"
            else:
                row += (f"{b.get('team_size', 0):>5.0f} emp "
                        f"{b.get('skill_coverage', 0):>5.0%} "
                        f"score {b.get('score', 0):>5.2f}").rjust(24)
        print(row)

    # --- Graphe ----------------------------------------------------------- #
    print("\n=== GRAPHE ===")
    for name in cols:
        g = runs[name].get("graph", {})
        print(f"{name:<34} {g.get('n_nodes', '?')} nœuds, "
              f"{g.get('n_edges', '?')} arêtes, {g.get('n_departments', '?')} dépts")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
