"""Crossfade DSP: the gain law and the EQ bass swap.

Two ideas carry most of the perceived smoothness:

1. **Equal-power gain.** Two uncorrelated tracks sum in *power*, not amplitude.
   A linear crossfade therefore sags ~3 dB in the middle - the audible dip that
   makes an amateur mix sound like an amateur mix. Using cos/sin quarter-cycles
   keeps ``a**2 + b**2 == 1`` throughout, so loudness stays put.

2. **Bass swap.** Two kick drums and two basslines at once is mud, and it is the
   other half of why overlapping tracks sound bad. The incoming track's low band
   is held down and handed over at the midpoint, so only one track owns the
   bottom end at any moment.

Automation is represented as ``Curve`` - an explicit list of breakpoints sampled
over a time span. The same arrays are handed to the browser engine
(``setValueCurveAtTime``) and to the offline renderer, which is what keeps the
two from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# Resolution of exported automation curves. Web Audio interpolates linearly
# between points, so this only needs to be fine enough that the curvature is
# smooth - 128 points over a ~30 s fade is far more than enough.
CURVE_POINTS = 128

# Crossover for the bass swap. Below this the two tracks fight; above it they
# blend acceptably.
BASS_CROSSOVER_HZ = 200.0

# How far the incoming track's low band is pulled down at the start of the fade.
BASS_CUT_DB = -24.0


@dataclass(slots=True)
class Curve:
    """An automation curve: ``values`` spread evenly across ``duration``.

    Linear interpolation between points, matching Web Audio's
    ``setValueCurveAtTime`` so both renderers agree exactly.
    """

    start: float
    duration: float
    values: list[float] = field(default_factory=list)

    def sample(self, n: int) -> np.ndarray:
        """Resample to ``n`` evenly spaced points across the span."""
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        src = np.linspace(0.0, 1.0, len(self.values))
        dst = np.linspace(0.0, 1.0, n)
        return np.interp(dst, src, np.asarray(self.values, dtype=np.float64))

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 6),
            "duration": round(self.duration, 6),
            "values": [round(float(v), 6) for v in self.values],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Curve":
        return cls(start=d["start"], duration=d["duration"], values=list(d["values"]))


def equal_power(n: int = CURVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power crossfade gains: outgoing (1 -> 0) and incoming (0 -> 1).

    Satisfies ``out**2 + inc**2 == 1`` at every point, so total power is constant.
    """
    t = np.linspace(0.0, 1.0, n)
    return np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def bass_swap(n: int = CURVE_POINTS, cut_db: float = BASS_CUT_DB):
    """Low-band gain multipliers for the outgoing and incoming tracks.

    The incoming track arrives with its bass cut and recovers to full by the
    midpoint; the outgoing track is the mirror, losing its bass from the midpoint
    on. Exactly one track owns the low end at any time, with a short handover in
    the middle. Smoothstepped so the handover has no corner in it.
    """
    t = np.linspace(0.0, 1.0, n)
    floor = 10.0 ** (cut_db / 20.0)
    inc = floor + (1.0 - floor) * _smoothstep(np.clip(t * 2.0, 0.0, 1.0))
    out = floor + (1.0 - floor) * _smoothstep(np.clip(2.0 - t * 2.0, 0.0, 1.0))
    return out, inc


def split_bands(audio: np.ndarray, sr: int, crossover: float = BASS_CROSSOVER_HZ):
    """Split into (low, high) bands that sum back to the original exactly.

    The high band is derived by subtraction rather than a second filter, which
    makes reconstruction perfect regardless of the filter's phase response - so
    a bass swap at unity gain is a true no-op.
    """
    sos = signal.butter(4, crossover, btype="low", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, audio, axis=0)
    return low, audio - low


def apply_bass_gain(audio: np.ndarray, sr: int, gain: np.ndarray,
                    crossover: float = BASS_CROSSOVER_HZ) -> np.ndarray:
    """Apply a per-sample gain to the low band only.

    ``gain`` must already be sample-aligned with ``audio``. Implemented as a gain
    on a split band rather than a time-varying shelf filter, so there is no
    filter state to glitch and no zipper noise.
    """
    low, high = split_bands(audio, sr, crossover)
    g = gain[:, None] if audio.ndim == 2 else gain
    return high + low * g


def highpass_sweep(n: int = CURVE_POINTS, start_hz: float = 20.0,
                   end_hz: float = 400.0, begin_at: float = 0.75) -> np.ndarray:
    """Cutoff automation for an optional filter sweep on the outgoing track.

    Flat until ``begin_at`` through the fade, then sweeping up so the old track
    thins out and steps aside rather than simply getting quieter. Swept in
    log-frequency; a linear sweep sounds like nothing and then everything.
    """
    t = np.linspace(0.0, 1.0, n)
    prog = np.clip((t - begin_at) / max(1e-9, 1.0 - begin_at), 0.0, 1.0)
    return start_hz * (end_hz / start_hz) ** prog


def apply_highpass_sweep(audio: np.ndarray, sr: int, cutoff: np.ndarray,
                         block: int = 512) -> np.ndarray:
    """Time-varying high-pass, updated per block with filter state carried over.

    Block-wise coefficient updates are the standard way to automate a filter;
    512 samples at 44.1 kHz is ~12 ms, fast enough that the sweep sounds
    continuous.
    """
    out = np.empty_like(audio, dtype=np.float64)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    zi = None
    for start in range(0, len(audio), block):
        stop = min(start + block, len(audio))
        hz = float(np.mean(cutoff[start:stop]))
        if hz <= 25.0:  # effectively bypassed; don't waste a filter on it
            out[start:stop] = audio[start:stop]
            zi = None
            continue
        sos = signal.butter(2, min(hz, sr * 0.45), btype="high", fs=sr, output="sos")
        if zi is None:
            shape = (sos.shape[0], 2) if channels == 1 else (sos.shape[0], 2, channels)
            zi = np.zeros(shape)
        chunk, zi = signal.sosfilt(sos, audio[start:stop], axis=0, zi=zi)
        out[start:stop] = chunk
    return out
