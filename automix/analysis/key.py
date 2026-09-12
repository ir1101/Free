"""Musical key detection via Krumhansl-Schmuckler profile matching.

Chroma folds the spectrum into twelve pitch classes, discarding octave. Averaged
over a track that gives a histogram of how much each pitch class is used, and
comparing it against the twenty-four major and minor key profiles - correlating
each against all twelve rotations - identifies the key.

The profiles are Krumhansl and Kessler's, from listening experiments measuring
how well each pitch class fits a given key. They encode what actually makes a
key sound like itself: the tonic and dominant dominate, the leading note is
distinctive, and the notes outside the scale are rare.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..model.camelot import Key, from_pitch_class

HOP = 512

# Chroma is taken from the chord register only. The default CQT range starts at
# C1, which hands the kick drum and bass fundamentals a vote - and a kick tuned
# near F#1 will tilt every track on the record towards F#. Measured over 24
# synthetic tracks of known key, starting at C1 scored 14/24; starting at C3
# with the harmonic component separated scored 20/24 exact and 23/24 on the
# Camelot number, which is the part mixing actually depends on.
#
# Pushing fmin higher still scored better on the number alone but lost mode
# accuracy, and tuning that finely against synthetic material would be fitting
# to this generator's kick rather than to music. C3 is the principled choice:
# above the rhythm section, across the range chords are voiced in.
CHROMA_FMIN_NOTE = "C3"
CHROMA_OCTAVES = 5

# Krumhansl-Kessler key profiles, starting on the tonic.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def chroma_profile(y: np.ndarray, sr: int, harmonic: bool = True) -> np.ndarray:
    """Average chroma vector for a signal, normalised to sum to 1.

    Percussion smears energy across every pitch class at once, which flattens the
    histogram and blunts the key estimate, so the harmonic component is separated
    out first. Frames are weighted by their own strength so that loud, clearly
    pitched passages count for more than near-silence.
    """
    if harmonic:
        y = librosa.effects.harmonic(y, margin=3.0)

    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=HOP,
        fmin=librosa.note_to_hz(CHROMA_FMIN_NOTE), n_octaves=CHROMA_OCTAVES,
    )
    weights = chroma.sum(axis=0)
    total = float(weights.sum())
    if total <= 1e-9:
        return np.full(12, 1.0 / 12.0)

    profile = (chroma * weights).sum(axis=1) / total
    s = float(profile.sum())
    return profile / s if s > 1e-9 else np.full(12, 1.0 / 12.0)


def _correlate(profile: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation of `profile` against all 12 rotations of `template`."""
    p = profile - profile.mean()
    p_norm = float(np.linalg.norm(p))
    scores = np.empty(12)
    for tonic in range(12):
        t = np.roll(template, tonic)
        t = t - t.mean()
        denom = p_norm * float(np.linalg.norm(t))
        scores[tonic] = float(p @ t) / denom if denom > 1e-12 else 0.0
    return scores


def detect_key(y: np.ndarray, sr: int, harmonic: bool = True):
    """Detect the key of a signal.

    Returns ``(Key, confidence)``. Confidence is the margin between the best key
    and the best *unrelated* alternative, so a track that is genuinely ambiguous
    between a key and its relative minor is not reported as uncertain - those two
    mix interchangeably anyway.
    """
    profile = chroma_profile(y, sr, harmonic=harmonic)
    major = _correlate(profile, MAJOR_PROFILE)
    minor = _correlate(profile, MINOR_PROFILE)

    scores = np.concatenate([major, minor])
    best = int(np.argmax(scores))
    tonic, is_major = best % 12, best < 12
    key = from_pitch_class(tonic, is_major)

    # Ignore the relative major/minor when measuring the margin: it shares a
    # Camelot number, so confusing the two costs nothing in a mix.
    relative = (tonic - 3) % 12 + 12 if is_major else (tonic + 3) % 12
    mask = np.ones(24, dtype=bool)
    mask[best] = False
    mask[relative] = False
    runner_up = float(np.max(scores[mask]))
    confidence = float(np.clip(scores[best] - runner_up, 0.0, 1.0))
    return key, confidence


def detect_key_over(y: np.ndarray, sr: int, start: float, end: float, **kw):
    """Key of one region, for checking whether a track modulates before its outro."""
    a, b = int(max(0.0, start) * sr), int(min(len(y) / sr, end) * sr)
    if b - a < sr:
        return detect_key(y, sr, **kw)
    return detect_key(y[a:b], sr, **kw)
