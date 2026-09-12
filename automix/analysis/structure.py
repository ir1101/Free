"""Structural segmentation and mix cue points.

A transition needs two decisions per track: where it is safe to start bringing
the track in, and where it should start going out. Mixing into a bare intro or
out of a final fade sounds wrong no matter how good the crossfade is.

Sections are found by clustering beat-synchronous features, which catches real
arrangement changes. Energy is then used to decide *which* boundaries are the
ones worth cueing to, because a boundary between two equally busy sections is
not a useful place to start a mix, whereas the moment the drums arrive is.

Both cue points are snapped to downbeats. A cue that is musically right but half
a bar off is still wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .beats import BeatGrid, BEATS_PER_BAR
from .energy import EnergyProfile

HOP = 512

# A track should not be cued so early that there is nothing playing, nor so late
# that there is no room left to mix.
MIN_INTRO_BARS = 4
MAX_INTRO_FRACTION = 0.35
MIN_OUTRO_FRACTION = 0.55

# How many consecutive bars must be busy before a point counts as "the track is
# properly running" - long enough to skip a one-bar fill.
SUSTAIN_BARS = 4


@dataclass(slots=True)
class Structure:
    """Section boundaries and the two cue points that matter for mixing."""

    boundaries: list[float]  # section start times, seconds
    mix_in: float            # bring the track in here
    mix_out: float           # start taking it out here

    def to_dict(self) -> dict:
        return {
            "boundaries": [round(b, 4) for b in self.boundaries],
            "mix_in": round(self.mix_in, 4),
            "mix_out": round(self.mix_out, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Structure":
        return cls(list(d["boundaries"]), d["mix_in"], d["mix_out"])


def _beat_sync_features(y: np.ndarray, sr: int, grid: BeatGrid) -> np.ndarray:
    """Chroma + MFCC averaged within each beat.

    Beat-synchronous features make segmentation tempo-invariant and massively
    cheaper, and they stop a boundary landing mid-beat.
    """
    beat_frames = librosa.time_to_frames(grid.beat_times(), sr=sr, hop_length=HOP)
    beat_frames = np.unique(np.clip(beat_frames, 0, None))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    n = min(chroma.shape[1], mfcc.shape[1])
    beat_frames = beat_frames[beat_frames < n]
    if len(beat_frames) < 4:
        return np.zeros((25, 0))

    feats = np.vstack([chroma[:, :n], mfcc[1:, :n]])
    sync = librosa.util.sync(feats, beat_frames, aggregate=np.mean)
    # Normalise each dimension so MFCC magnitudes do not swamp chroma.
    mu, sd = sync.mean(axis=1, keepdims=True), sync.std(axis=1, keepdims=True)
    return (sync - mu) / (sd + 1e-9)


def find_boundaries(y: np.ndarray, sr: int, grid: BeatGrid, n_sections: int = 8) -> list[float]:
    """Section boundary times, snapped to downbeats."""
    feats = _beat_sync_features(y, sr, grid)
    if feats.shape[1] < 8:
        return []

    k = int(np.clip(n_sections, 2, feats.shape[1] // 2))
    try:
        bounds = librosa.segment.agglomerative(feats, k)
    except Exception:
        return []

    times = [grid.beat_time(int(b)) for b in bounds]
    snapped = sorted({round(grid.snap_to_bar(t), 4) for t in times})
    return [t for t in snapped if t >= 0]


def _bar_activity(profile: EnergyProfile, grid: BeatGrid, threshold: float) -> np.ndarray:
    """Per-downbeat flag: is this bar above `threshold`?"""
    bar = grid.seconds_per_bar
    return np.array([profile.between(t, t + bar) >= threshold
                     for t in grid.downbeat_times()], dtype=bool)


def _sustained(active: np.ndarray, forward: bool) -> np.ndarray:
    """Require activity to hold across SUSTAIN_BARS, looking one way or the other.

    The direction is not cosmetic. Coming *in*, the question is "does the track
    keep going from here", so the window looks forward and the answer is the
    first bar of a run. Going *out*, the question is "has the track been running
    up to here", so the window looks back and the answer is the last bar of a
    run. Using a forward window for both places the outro cue SUSTAIN_BARS too
    early - the last bar that still has a full window ahead of it is several bars
    before the track actually stops.
    """
    n = len(active)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        window = active[i:i + SUSTAIN_BARS] if forward else active[max(0, i - SUSTAIN_BARS + 1):i + 1]
        out[i] = bool(len(window)) and bool(window.all())
    return out


def analyze_structure(y: np.ndarray, sr: int, grid: BeatGrid,
                      profile: EnergyProfile) -> Structure:
    """Find section boundaries and pick the two cue points."""
    duration = len(y) / sr
    boundaries = find_boundaries(y, sr, grid)
    downbeats = grid.downbeat_times()
    if len(downbeats) < 4:
        return Structure(boundaries, 0.0, max(0.0, duration * 0.75))

    # Threshold relative to the track's own busy sections, so this works for a
    # quiet ambient record as well as a loud one.
    busy = float(np.percentile(profile.energy, 75)) if len(profile.energy) else 0.0
    quiet = float(np.percentile(profile.energy, 20)) if len(profile.energy) else 0.0
    threshold = quiet + 0.45 * (busy - quiet)

    active = _bar_activity(profile, grid, threshold)
    starts = _sustained(active, forward=True)
    ends = _sustained(active, forward=False)

    earliest = MIN_INTRO_BARS
    latest_in = max(earliest + 1, int(len(downbeats) * MAX_INTRO_FRACTION))
    mix_in_idx = next((i for i in range(earliest, min(latest_in, len(starts)))
                       if starts[i]), earliest)

    earliest_out = max(mix_in_idx + 1, int(len(downbeats) * MIN_OUTRO_FRACTION))
    # Walk back from the end to the last bar that still had the track running.
    mix_out_idx = next((i for i in range(len(ends) - 1, earliest_out - 1, -1)
                        if ends[i]), None)
    if mix_out_idx is None:
        mix_out_idx = min(len(downbeats) - 1, max(earliest_out, len(downbeats) - 8))

    mix_in = float(downbeats[min(mix_in_idx, len(downbeats) - 1)])
    # `mix_out_idx` is the last bar the track was still running; the outro starts
    # on the bar after it, and that boundary is where the mix should begin.
    mix_out = float(downbeats[min(mix_out_idx + 1, len(downbeats) - 1)])

    # Prefer a real section boundary when one sits nearby - arrangement changes
    # are better mix points than an energy threshold crossing.
    for cue, name in ((mix_in, "in"), (mix_out, "out")):
        near = [b for b in boundaries if abs(b - cue) <= grid.seconds_per_bar * 2]
        if near:
            best = min(near, key=lambda b: abs(b - cue))
            if name == "in":
                mix_in = best
            else:
                mix_out = best

    if mix_out <= mix_in:
        mix_out = float(downbeats[min(len(downbeats) - 1, mix_in_idx + BEATS_PER_BAR)])
    return Structure(boundaries, mix_in, mix_out)
