"""The incremental coordination graph must equal a from-scratch rebuild."""
import numpy as np

from base.amorphous_structure import AmorphousStruc_factory


def _edge_set(g):
    return set(map(frozenset, g.edges))


def test_incremental_commit_matches_full_rebuild():
    rng = np.random.default_rng(0)
    cell = (12.0, 12.0, 12.0)
    elements = ["Si", "O", "Al", "H"]

    # Start from one atom and an up-to-date graph, then commit many atoms one by one
    # (the incremental path). Keep querying so the graph stays non-stale.
    s = AmorphousStruc_factory(symbols=["Si"], positions=[[6.0, 6.0, 6.0]],
                               cell=list(cell), pbc=True, seed=0)
    s.get_graph()  # build initial graph

    for _ in range(40):
        sym = elements[int(rng.integers(len(elements)))]
        pos = rng.uniform(0.0, 12.0, size=3)
        s.commit_atom(sym, pos)
        s.get_graph()  # incremental update stays active (not stale)

    inc = s.get_graph()
    inc_nodes = set(inc.nodes)
    inc_edges = _edge_set(inc)
    inc_symbols = {i: inc.nodes[i]["symbol"] for i in inc.nodes}

    full = s.get_graph(force_rebuild=True)

    assert set(full.nodes) == inc_nodes
    assert _edge_set(full) == inc_edges
    assert {i: full.nodes[i]["symbol"] for i in full.nodes} == inc_symbols


def test_incremental_then_remove_falls_back_to_full_rebuild(make_struct):
    # remove_atom reindexes, so it must mark the graph stale (full rebuild), and the
    # result must still be correct.
    s = make_struct(["Si", "O", "O"], [[6, 6, 6], [7.6, 6, 6], [4.4, 6, 6]], cell=(12.0, 12.0, 12.0))
    s.get_graph()
    s.commit_atom("O", [6.0, 7.6, 6.0])     # incremental
    s.remove_atom(0)                          # marks stale
    after = s.get_graph()
    full = s.get_graph(force_rebuild=True)
    assert _edge_set(after) == _edge_set(full)
