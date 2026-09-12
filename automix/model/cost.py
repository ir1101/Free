"""What makes two tracks fit together, expressed as a cost.

Five components, each answering a different question about a candidate join:

- **tempo**     can they be beat-matched without audible artefacts?
- **harmonic**  do the keys agree?
- **energy**    does the incoming track pick up where the outgoing one left off?
- **timbre**    do they sound like they belong to the same set?
- **arc**       is this track the right intensity for this point in the set?

The first four are properties of the *pair*. The fifth is a property of the
track and its position, which is what lets the sequencer shape a set rather than
just chain compatible records together.

Everything returns a breakdown rather than a single number, because the useful
question when a transition sounds wrong is always which of these caused it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ..analysis.energy import timbre_distance
from ..analysis.track import Track
from . import camelot

# Beyond this, a phase vocoder starts to smear transients and a turntable-style
# pitch shift becomes obvious. Inside it, a tempo change is essentially free.
COMFORTABLE_STRETCH = 0.06

# Half- and double-time mixing is legitimate but changes the feel of the set, so
# it carries a small standing cost even when the maths works out perfectly.
OCTAVE_PENALTY = 0.35

# Energy differences are on a 0-1 scale; this brings them to the same rough
# magnitude as the other components.
ENERGY_SCALE = 4.0

COMPONENTS = ("tempo", "harmonic", "energy", "timbre", "arc")


@dataclass(slots=True)
class Weights:
    """Relative importance of each component. Surfaced in the UI."""

    tempo: float = 1.0
    harmonic: float = 1.0
    energy: float = 1.0
    timbre: float = 0.5
    arc: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Weights":
        d = d or {}
        return cls(**{f: float(d[f]) for f in COMPONENTS if f in d})


@dataclass(slots=True)
class TransitionCost:
    """A scored join between two tracks, with the reasoning kept alongside."""

    total: float
    tempo: float
    harmonic: float
    energy: float
    timbre: float
    arc: float

    tempo_ratio: float = 1.0        # multiply B's playback rate by this
    stretch_percent: float = 0.0
    octave: float = 1.0             # 1.0, 0.5 or 2.0 - how B's tempo was read
    semitone_shift: float = 0.0     # side effect of rate-based tempo matching
    harmonic_label: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def target_energy(position: float, shape: str = "peak",
                  low: float = 0.35, high: float = 0.95,
                  peak_at: float = 0.68, close_at: float = 0.55) -> float:
    """The intensity a set wants at `position`, 0 at the start and 1 at the end.

    A DJ set is not a flat sequence of equally intense records. It opens below
    its ceiling, builds, peaks around two thirds through, and comes down. Placing
    the peak later leaves no room to land; placing it earlier makes the back half
    sag.

    ``shape="flat"`` returns -1 to mean "no target", which switches the arc term
    off and leaves pure track-to-track smoothness.
    """
    if shape == "flat":
        return -1.0

    def smoothstep(x: float) -> float:
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    if position <= peak_at:
        return low + (high - low) * smoothstep(position / max(1e-9, peak_at))
    tail = (position - peak_at) / max(1e-9, 1.0 - peak_at)
    return high - (high - close_at) * smoothstep(tail)


def tempo_match(bpm_a: float, bpm_b: float):
    """Best way to read B's tempo against A's.

    Returns ``(cost, ratio, stretch_fraction, octave)``. Half- and double-time
    are considered because an 85 BPM track and a 170 BPM track share a pulse, and
    refusing to see that would rule out perfectly good transitions.
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 10.0, 1.0, 0.0, 1.0

    best = None
    for octave in (1.0, 0.5, 2.0):
        ratio = bpm_a / (bpm_b * octave)
        stretch = ratio - 1.0
        cost = (abs(stretch) / COMFORTABLE_STRETCH) ** 2
        if octave != 1.0:
            cost += OCTAVE_PENALTY
        if best is None or cost < best[0]:
            best = (cost, ratio, stretch, octave)
    return best


def harmonic_match(track_a: Track, track_b: Track, ratio: float,
                   pitch_preserving: bool = True):
    """Key compatibility, accounting for pitch shift when rate-matching.

    Matching tempo by changing playback rate moves pitch with it, exactly as a
    turntable does. A 6% speed-up is very nearly a full semitone, enough to turn
    a perfect key match into a clash - so when the renderer is not
    pitch-preserving, B is transposed by however far the rate change moves it
    before compatibility is judged.
    """
    semitones = 12.0 * math.log2(ratio) if ratio > 0 else 0.0
    key_b = track_b.key
    if not pitch_preserving and abs(semitones) >= 0.5:
        key_b = key_b.transposed(int(round(semitones)))
    cost, label = camelot.harmonic_distance(track_a.key, key_b)
    return cost, label, semitones


def transition_cost(track_a: Track, track_b: Track, weights: Weights | None = None,
                    position: float | None = None, arc_shape: str = "peak",
                    pitch_preserving: bool = True) -> TransitionCost:
    """Score playing `track_b` after `track_a`.

    `position` is where B would sit in the set, 0 to 1. Pass ``None`` to judge
    the pair on its own merits with no arc term.
    """
    w = weights or Weights()
    notes: list[str] = []

    tempo_c, ratio, stretch, octave = tempo_match(track_a.bpm, track_b.bpm)
    if octave != 1.0:
        which = "double" if octave == 2.0 else "half"
        notes.append(which + "-time match")
    if abs(stretch) > COMFORTABLE_STRETCH:
        notes.append("needs %.1f%% stretch - artefacts likely" % (abs(stretch) * 100.0))

    harmonic_c, harmonic_label, semitones = harmonic_match(
        track_a, track_b, ratio, pitch_preserving=pitch_preserving
    )
    if not pitch_preserving and abs(semitones) >= 0.5:
        notes.append("rate match shifts B by %+.2f semitones" % semitones)

    # The join is between A's tail and B's head, so those are what get compared.
    energy_c = abs(track_a.energy_out - track_b.energy_in) * ENERGY_SCALE
    timbre_c = timbre_distance(track_a.timbre_out, track_b.timbre_in)

    arc_c = 0.0
    if position is not None:
        want = target_energy(position, shape=arc_shape)
        if want >= 0.0:
            arc_c = abs(track_b.energy_mean - want) * ENERGY_SCALE

    total = (w.tempo * tempo_c + w.harmonic * harmonic_c + w.energy * energy_c
             + w.timbre * timbre_c + w.arc * arc_c)

    return TransitionCost(
        total=total, tempo=tempo_c, harmonic=harmonic_c, energy=energy_c,
        timbre=timbre_c, arc=arc_c, tempo_ratio=ratio,
        stretch_percent=stretch * 100.0, octave=octave, semitone_shift=semitones,
        harmonic_label=harmonic_label, notes=notes,
    )


def cost_matrix(tracks: list[Track], weights: Weights | None = None,
                pitch_preserving: bool = True) -> list[list[float]]:
    """Pairwise costs with no arc term, as ``matrix[a][b]``.

    Deliberately asymmetric: a to b compares the first track's outro against the
    second track's intro, which is a different comparison from b to a. Anything
    consuming this has to respect that.
    """
    n = len(tracks)
    matrix = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = transition_cost(
                    tracks[i], tracks[j], weights, position=None,
                    pitch_preserving=pitch_preserving,
                ).total
    return matrix
