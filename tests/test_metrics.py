# tests/test_metrics.py
import networkx as nx
from src.graph.metrics import DisruptionTracker

def test_add_remove_symmetry():
    """add() puis remove() doit ramener l'état initial."""
    G = nx.Graph()
    # 2 départements, 4 employés
    for i in range(4):
        G.add_node(i, department=f"dept_{i // 2}")
    G.add_edges_from([(0, 1), (2, 3), (0, 2), (1, 3)])

    tracker = DisruptionTracker(G)
    state_initial = tracker.debug_state()

    # Séquence : add, add, remove, add, remove, remove
    seq = [0, 2, 0, 1, 1, 2]
    for node in seq:
        if node in tracker.selected:
            tracker.remove(node)
        else:
            tracker.add(node)

    state_final = tracker.debug_state()

    assert state_final["selected"] == state_initial["selected"], \
        f"Sélection non restaurée : {state_final['selected']}"
    assert state_final["cut_edges_by_dept"] == state_initial["cut_edges_by_dept"], \
        f"Cut edges non restaurés : {state_final} vs {state_initial}"
    assert abs(state_final["disruption"] - state_initial["disruption"]) < 1e-9
    print("✅ test_add_remove_symmetry passed")


def test_remove_nonexistent():
    """remove() sur un nœud non sélectionné ne fait rien."""
    G = nx.Graph()
    G.add_node(0, department="d1")
    G.add_node(1, department="d1")
    G.add_edge(0, 1)

    tracker = DisruptionTracker(G)
    tracker.remove(0)  # ne doit pas planter
    assert tracker.debug_state()["selected"] == []
    print("✅ test_remove_nonexistent passed")


if __name__ == "__main__":
    test_add_remove_symmetry()
    test_remove_nonexistent()
    print("\n🎉 Tous les tests passent.")