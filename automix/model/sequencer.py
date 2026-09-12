"""Ordering a set: a minimum-cost path through the tracks.

This is a Hamiltonian path problem - visit every track once, as cheaply as
possible - with two wrinkles that rule out the textbook TSP machinery.

**The cost is asymmetric.** ``a -> b`` compares the first track's outro against
the second's intro, which is simply not the same comparison as ``b -> a``.

**The cost depends on position.** The arc term asks whether a track is the right
intensity for *this point* in the set, so the same pair costs different amounts
in different places.

Together those two facts kill 2-opt, the usual local-search move, because 2-opt
works by reversing a segment: it assumes reversing an edge costs what the
original edge cost, which is false here. So the local search uses **Or-opt**
(relocate a short run) and **swap** (exchange two tracks) instead. Neither ever
reverses an edge, so both stay correct under an asymmetric cost.

The cost decomposes cleanly into a pair term and a per-position term, which is
what lets the exact solver stay exact: in a Held-Karp state the number of tracks
already placed *is* the position, so the arc term needs no extra state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..analysis.track import Track
from .cost import ENERGY_SCALE, TransitionCost, Weights, target_energy, transition_cost

# Above this, exact search is no longer affordable: Held-Karp is O(2^n * n^2),
# so each extra track roughly doubles the work.
EXACT_LIMIT = 12

MAX_PASSES = 40
OR_OPT_MAX_RUN = 3


@dataclass(slots=True)
class SetOrder:
    """A chosen ordering, with the cost that justified it."""

    order: list[int]
    total_cost: float
    transitions: list[TransitionCost]
    exact: bool

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "total_cost": round(self.total_cost, 4),
            "exact": self.exact,
            "transitions": [t.to_dict() for t in self.transitions],
        }


def _arc_costs(tracks: list[Track], weights: Weights, arc_shape: str) -> list[list[float]]:
    """``arc[track][position]`` - how well each track suits each slot in the set."""
    n = len(tracks)
    if n <= 1 or arc_shape == "flat":
        return [[0.0] * n for _ in range(n)]

    wants = [target_energy(p / (n - 1), shape=arc_shape) for p in range(n)]
    return [[weights.arc * abs(t.energy_mean - w) * ENERGY_SCALE for w in wants]
            for t in tracks]


def _pair_costs(tracks: list[Track], weights: Weights, pitch_preserving: bool):
    """``pair[a][b]`` - position-independent cost, and the full breakdowns."""
    n = len(tracks)
    costs = [[math.inf] * n for _ in range(n)]
    detail: list[list[TransitionCost | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tc = transition_cost(tracks[i], tracks[j], weights, position=None,
                                 pitch_preserving=pitch_preserving)
            costs[i][j] = tc.total
            detail[i][j] = tc
    return costs, detail


def _total(order: list[int], pair, arc) -> float:
    if not order:
        return 0.0
    total = arc[order[0]][0]
    for position in range(1, len(order)):
        total += pair[order[position - 1]][order[position]] + arc[order[position]][position]
    return total


def _held_karp(n: int, pair, arc, start: int | None):
    """Exact minimum-cost Hamiltonian path by dynamic programming over subsets.

    ``best[mask][last]`` is the cheapest way to have played exactly the tracks in
    ``mask``, finishing on ``last``. Because every state knows how many tracks it
    has placed, the arc term for the incoming track can be charged as it is
    added - no extra dimension needed.
    """
    size = 1 << n
    best = [[math.inf] * n for _ in range(size)]
    prev = [[-1] * n for _ in range(size)]

    starts = range(n) if start is None else [start]
    for i in starts:
        best[1 << i][i] = arc[i][0]

    for mask in range(size):
        row = best[mask]
        placed = bin(mask).count("1")
        for last in range(n):
            base = row[last]
            if base == math.inf or not (mask >> last) & 1:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                candidate = base + pair[last][nxt] + arc[nxt][placed]
                target = mask | (1 << nxt)
                if candidate < best[target][nxt]:
                    best[target][nxt] = candidate
                    prev[target][nxt] = last

    full = size - 1
    end = min(range(n), key=lambda i: best[full][i])
    if best[full][end] == math.inf:
        return list(range(n)), math.inf

    order = []
    mask, node = full, end
    while node != -1:
        order.append(node)
        node, mask = prev[mask][node], mask ^ (1 << node)
    order.reverse()
    return order, best[full][end]


def _greedy(n: int, pair, arc, start: int) -> list[int]:
    """Nearest-neighbour from a fixed start, charging the arc term as we go."""
    order = [start]
    unused = set(range(n)) - {start}
    while unused:
        last, position = order[-1], len(order)
        nxt = min(unused, key=lambda j: pair[last][j] + arc[j][position])
        order.append(nxt)
        unused.discard(nxt)
    return order


def _improve(order: list[int], pair, arc, max_passes: int = MAX_PASSES) -> list[int]:
    """Local search over relocate and swap moves.

    Both move types are chosen specifically because they never reverse an edge,
    which keeps them valid for an asymmetric cost. The cost of a candidate is
    recomputed in full rather than by a delta: positions shift when a track
    moves, so every downstream arc term can change, and a delta that assumed
    otherwise would silently drift.
    """
    n = len(order)
    if n < 3:
        return order

    current = list(order)
    best_cost = _total(current, pair, arc)

    for _ in range(max_passes):
        improved = False

        for run in range(1, min(OR_OPT_MAX_RUN, n - 1) + 1):
            for i in range(n - run + 1):
                segment = current[i:i + run]
                rest = current[:i] + current[i + run:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    candidate = rest[:j] + segment + rest[j:]
                    cost = _total(candidate, pair, arc)
                    if cost < best_cost - 1e-12:
                        current, best_cost, improved = candidate, cost, True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = list(current)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    cost = _total(candidate, pair, arc)
                    if cost < best_cost - 1e-12:
                        current, best_cost, improved = candidate, cost, True
                        break
                if improved:
                    break

        if not improved:
            break

    return current


def sequence(tracks: list[Track], weights: Weights | None = None,
             arc_shape: str = "peak", pitch_preserving: bool = True,
             start: int | None = None, exact_limit: int = EXACT_LIMIT) -> SetOrder:
    """Order `tracks` into the smoothest set that also follows the energy arc.

    Small sets are solved exactly. Larger ones use nearest-neighbour from every
    possible start, refined by local search - which is not guaranteed optimal,
    but the gap is small and the result is reported honestly via `exact`.
    """
    n = len(tracks)
    w = weights or Weights()
    if n == 0:
        return SetOrder([], 0.0, [], True)
    if n == 1:
        return SetOrder([0], 0.0, [], True)

    pair, detail = _pair_costs(tracks, w, pitch_preserving)
    arc = _arc_costs(tracks, w, arc_shape)

    if n <= exact_limit:
        order, total = _held_karp(n, pair, arc, start)
        exact = True
    else:
        starts = [start] if start is not None else range(n)
        best_order, best_cost = None, math.inf
        for s in starts:
            candidate = _improve(_greedy(n, pair, arc, s), pair, arc)
            cost = _total(candidate, pair, arc)
            if cost < best_cost:
                best_order, best_cost = candidate, cost
        order, total, exact = best_order or list(range(n)), best_cost, False

    transitions = []
    for position in range(1, len(order)):
        a, b = order[position - 1], order[position]
        tc = transition_cost(tracks[a], tracks[b], w,
                             position=position / max(1, n - 1),
                             arc_shape=arc_shape, pitch_preserving=pitch_preserving)
        transitions.append(tc)

    return SetOrder(order, total, transitions, exact)
