"""Energy and timbre profiling.

Two tracks can share a tempo and a key and still collide, because "energy" in a
mix is not just loudness - it is loudness, brightness and busyness together. A
sparse dub track and a peak-time banger at the same RMS are not interchangeable.

Everything here is measured on absolute scales rather than normalised per track.
That matters: the whole point is to compare *across* tracks, and a per-track
normalisation would rate the loudest moment of a quiet ambient piece the same as
the loudest moment of a club record.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

HOP = 512

# Absolute ranges the three components are mapped onto. Chosen to span quiet
# intro to full-tilt drop across typical produced music.
LOUDNESS_RANGE = (-40.0, -6.0)     # dBFS
CENTROID_RANGE = (200.0, 4000.0)   # Hz, log-scaled
ONSET_RATE_RANGE = (0.0, 8.0)      # onsets per second

# Loudness dominates: it is what a listener reads as intensity first.
WEIGHTS = {"loudness": 0.5, "brightness": 0.25, "busyness": 0.25}

N_MFCC = 13


def _scale(values: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if log:
        v = np.log(np.maximum(v, 1e-6))
        lo, hi = np.log(max(lo, 1e-6)), np.log(max(hi, 1e-6))
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


@dataclass(slots=True)
class EnergyProfile:
    """Frame-wise energy components plus a combined curve."""

    times: np.ndarray
    loudness: np.ndarray    # dBFS
    brightness: np.ndarray  # 0-1
    busyness: np.ndarray    # 0-1
    energy: np.ndarray      # 0-1 combined

    def between(self, start: float, end: float) -> float:
        """Mean combined energy over a time span."""
        if len(self.times) == 0:
            return 0.0
        sel = (self.times >= start) & (self.times < end)
        if not sel.any():
            idx = int(np.clip(np.searchsorted(self.times, start), 0, len(self.times) - 1))
            return float(self.energy[idx])
        return float(np.mean(self.energy[sel]))

    def curve(self, n: int = 128) -> list[float]:
        """Downsampled energy curve for display and for the set-level arc."""
        if len(self.energy) == 0:
            return [0.0] * n
        src = np.linspace(0.0, 1.0, len(self.energy))
        return [round(float(v), 4) for v in np.interp(np.linspace(0, 1, n), src, self.energy)]


def analyze_energy(y: np.ndarray, sr: int, onset_env: np.ndarray | None = None) -> EnergyProfile:
    """Frame-wise loudness, brightness and busyness for a signal."""
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    loudness_db = 20.0 * np.log10(np.maximum(rms, 1e-6))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    n = min(len(loudness_db), len(centroid), len(onset_env))
    loudness_db, centroid, onset_env = loudness_db[:n], centroid[:n], onset_env[:n]

    # Busyness: local onset activity, smoothed over about a second so it tracks
    # arrangement density rather than individual hits.
    from scipy.ndimage import uniform_filter1d

    width = max(3, int(round(1.0 * sr / HOP)))
    rate = uniform_filter1d(onset_env, size=width, mode="nearest") * (sr / HOP)
    peak = float(np.percentile(rate, 99)) if len(rate) else 0.0
    if peak > 1e-9:
        rate = rate / peak * ONSET_RATE_RANGE[1] * 0.75

    loud = _scale(loudness_db, *LOUDNESS_RANGE)
    bright = _scale(centroid, *CENTROID_RANGE, log=True)
    busy = _scale(rate, *ONSET_RATE_RANGE)

    energy = (WEIGHTS["loudness"] * loud
              + WEIGHTS["brightness"] * bright
              + WEIGHTS["busyness"] * busy)

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)
    return EnergyProfile(times, loudness_db, bright, busy, energy)


def timbre_vector(y: np.ndarray, sr: int, start: float = 0.0,
                  end: float | None = None) -> np.ndarray:
    """Mean MFCC vector over a region - a compact description of texture.

    MFCC 0 tracks overall level, which energy already covers, so it is dropped;
    what is left describes spectral shape, which is what makes two tracks sound
    like they belong to the same record or not.
    """
    end = len(y) / sr if end is None else end
    a, b = int(max(0.0, start) * sr), int(min(len(y) / sr, end) * sr)
    segment = y[a:b] if b - a > sr // 4 else y
    if len(segment) < HOP * 2:
        return np.zeros(N_MFCC - 1)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    return np.mean(mfcc[1:], axis=1)


def timbre_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two timbre vectors, 0 (identical) to 1."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(np.clip(1.0 - (a @ b) / (na * nb), 0.0, 2.0)) / 2.0
