"""The sequencer: exactness, and the move set that asymmetry demands."""

import itertools
import math
import random

import pytest

from automix.model.cost import Weights, target_energy
from automix.model.sequencer import _greedy, _held_karp, _improve, _total


def random_problem(n, seed, asymmetric=True):
    rng = random.Random(seed)
    pair = [[math.inf if i == j else rng.uniform(0.1, 9.0) for j in range(n)] for i in range(n)]
    if not asymmetric:
        for i in range(n):
            for j in range(i + 1, n):
                pair[j][i] = pair[i][j]
    arc = [[rng.uniform(0.0, 2.0) for _ in range(n)] for _ in range(n)]
    return pair, arc


def brute_force(n, pair, arc):
    best, best_order = math.inf, None
    for order in itertools.permutations(range(n)):
        cost = _total(list(order), pair, arc)
        if cost < best:
            best, best_order = cost, list(order)
    return best_order, best


@pytest.mark.parametrize("seed", range(6))
def test_held_karp_matches_brute_force(seed):
    """The exact solver has to actually be exact."""
    n = 7
    pair, arc = random_problem(n, seed)
    _, want = brute_force(n, pair, arc)
    order, got = _held_karp(n, pair, arc, start=None)
    assert got == pytest.approx(want, rel=1e-9)
    assert sorted(order) == list(range(n))
    assert _total(order, pair, arc) == pytest.approx(want, rel=1e-9)


def test_held_karp_respects_a_forced_start():
    n = 7
    pair, arc = random_problem(n, 42)
    order, cost = _held_karp(n, pair, arc, start=3)
    assert order[0] == 3
    best = min(_total([3] + list(rest), pair, arc)
               for rest in itertools.permutations([i for i in range(n) if i != 3]))
    assert cost == pytest.approx(best, rel=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_local_search_never_makes_things_worse(seed):
    n = 9
    pair, arc = random_problem(n, seed)
    start = _greedy(n, pair, arc, 0)
    improved = _improve(start, pair, arc)
    assert sorted(improved) == list(range(n))
    assert _total(improved, pair, arc) <= _total(start, pair, arc) + 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_local_search_gets_close_to_optimal(seed):
    n = 8
    pair, arc = random_problem(n, seed)
    _, optimal = brute_force(n, pair, arc)
    best = min(_total(_improve(_greedy(n, pair, arc, s), pair, arc), pair, arc)
               for s in range(n))
    assert best <= optimal * 1.25 + 1e-9


def test_local_search_is_correct_under_asymmetric_costs():
    """The reason 2-opt is not used here, made concrete.

    A segment reversal assumes an edge costs the same in both directions. This
    builds a problem where it emphatically does not - the forward chain is cheap
    and every backward edge is expensive - and checks that the search still finds
    a genuinely better order rather than being fooled into reversing one.
    """
    n = 6
    pair = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                pair[i][j] = 1.0 if j == i + 1 else (2.0 if j > i else 50.0)
    arc = [[0.0] * n for _ in range(n)]

    scrambled = [3, 1, 4, 0, 5, 2]
    improved = _improve(scrambled, pair, arc)
    assert sorted(improved) == list(range(n))
    assert _total(improved, pair, arc) <= _total(scrambled, pair, arc)
    # The forward chain is optimal; the search should find it or come close.
    assert _total(improved, pair, arc) <= _total(list(range(n)), pair, arc) + 1e-9


def test_total_cost_charges_every_position():
    pair = [[0.0, 1.0], [2.0, 0.0]]
    arc = [[10.0, 20.0], [30.0, 40.0]]
    # order [0, 1]: arc[0][0] + pair[0][1] + arc[1][1]
    assert _total([0, 1], pair, arc) == pytest.approx(10.0 + 1.0 + 40.0)
    # order [1, 0]: arc[1][0] + pair[1][0] + arc[0][1]
    assert _total([1, 0], pair, arc) == pytest.approx(30.0 + 2.0 + 20.0)


def test_energy_arc_rises_then_falls():
    curve = [target_energy(p / 20.0) for p in range(21)]
    peak = curve.index(max(curve))
    assert 0 < peak < 20, "the peak should be inside the set, not at either end"
    assert curve[0] < curve[peak] and curve[-1] < curve[peak]
    assert all(b >= a - 1e-9 for a, b in zip(curve[:peak], curve[1:peak + 1]))


def test_flat_arc_disables_the_term():
    assert target_energy(0.5, shape="flat") < 0


def test_weights_roundtrip():
    w = Weights(tempo=2.0, harmonic=0.5)
    again = Weights.from_dict(w.to_dict())
    assert again == w
    assert Weights.from_dict({"tempo": 3.0}).tempo == 3.0
    assert Weights.from_dict(None) == Weights()
    assert Weights.from_dict({"nonsense": 1.0}) == Weights()
