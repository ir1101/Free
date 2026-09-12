"""Synthetic audio with known BPM, key and structure.

The analysis code has to be checkable against ground truth: if a test feeds in a
track that is 128 BPM in A minor by construction, the detector's answer is either
right or wrong, with no argument. Real music can only ever be spot-checked.

Tracks are built with a real arrangement - intro, body, outro at different
energies - so structural cue-point detection has something to actually find.
"""

from __future__ import annotations

import numpy as np

SR = 44100

MAJOR_TRIAD = [0, 4, 7]
MINOR_TRIAD = [0, 3, 7]

# A four-chord loop per mode, as (root offset in semitones, triad quality).
#
# The quality has to vary with the degree. Building every chord of a minor
# progression as a minor triad puts notes outside the scale into the track - the
# major-VI chord would contribute a raised sixth that natural minor does not
# contain - and the result is not really in the key it claims to be. That makes
# it useless as ground truth for a key detector, which would be marked wrong for
# correctly describing what it was actually given.
# The loops also have to be modally *unambiguous*, which rules out the obvious
# pop progressions. C-G-Am-F and Am-F-C-G contain exactly the same four chords;
# nothing in the pitch content distinguishes C major from A minor, so either
# answer is defensible and neither can serve as ground truth for mode.
#
# The minor loop therefore ends on a major V, whose leading tone (G# in A minor)
# does not belong to the relative major at all. The major loop leans on I-IV-V
# with the tonic on both ends.
MAJOR_PROGRESSION = [  # I - IV - V - I
    (0, MAJOR_TRIAD), (5, MAJOR_TRIAD), (7, MAJOR_TRIAD), (0, MAJOR_TRIAD),
]
MINOR_PROGRESSION = [  # i - VI - III - V (major V: the leading tone fixes the mode)
    (0, MINOR_TRIAD), (8, MAJOR_TRIAD), (3, MAJOR_TRIAD), (7, MAJOR_TRIAD),
]


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _env(n: int, attack: int, decay: float) -> np.ndarray:
    """Percussive envelope: fast attack, exponential decay."""
    e = np.exp(-np.arange(n) / max(1.0, decay))
    e[:attack] *= np.linspace(0.0, 1.0, attack)
    return e


def kick(sr: int = SR, dur: float = 0.28) -> np.ndarray:
    """Sine pitched down 120 -> 45 Hz. Gives beat trackers a strong onset."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = 45.0 + 75.0 * np.exp(-t * 32.0)
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, 32, sr * 0.02)


def hat(sr: int = SR, dur: float = 0.06) -> np.ndarray:
    """Hi-hat: noise burst with the low end genuinely removed.

    The filter matters. A real hi-hat has essentially no energy below a few kHz,
    which is what lets a beat tracker tell it apart from a kick. A lazy 6 dB/oct
    tilt leaves plenty of low-frequency content behind and produces a "hat" that
    is really just broadband noise - which would make this an unrealistically
    hard test signal rather than a representative one.
    """
    from scipy import signal as _sig

    n = int(sr * dur)
    rng = np.random.default_rng(7)
    sos = _sig.butter(6, 5000.0, btype="high", fs=sr, output="sos")
    noise = _sig.sosfilt(sos, rng.standard_normal(n))
    return noise * _env(n, 8, sr * 0.004) * 0.5


def _tone(freqs: list[float], n: int, sr: int, harmonics: int = 4) -> np.ndarray:
    """Sum of mildly harmonic tones - richer than a sine, so chroma sees it."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for f in freqs:
        for h in range(1, harmonics + 1):
            if f * h < sr / 2:
                out += np.sin(2 * np.pi * f * h * t) / (h * h)
    return out / max(1e-9, np.max(np.abs(out)))


def make_track(
    bpm: float = 128.0,
    tonic: int = 9,           # pitch class, C = 0 (default A)
    is_major: bool = False,
    bars: int = 32,
    sr: int = SR,
    intro_bars: int = 4,
    outro_bars: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Render a track and return ``(audio, ground_truth)``.

    The arrangement is intro (drums only) -> body (full) -> outro (drums thinning),
    so ``mix_in`` should land at the end of the intro and ``mix_out`` at the start
    of the outro.
    """
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm                      # seconds per beat
    beats = bars * 4
    total = int(np.ceil(beats * spb * sr)) + sr
    audio = np.zeros(total)

    k, h = kick(sr), hat(sr)
    progression = MAJOR_PROGRESSION if is_major else MINOR_PROGRESSION

    body_start, body_end = intro_bars, bars - outro_bars

    for bar in range(bars):
        bar_start = int(bar * 4 * spb * sr)
        in_body = body_start <= bar < body_end
        in_outro = bar >= body_end
        level = 1.0 if in_body else (0.55 if bar < body_start else 0.4)

        # Drums run throughout; the outro drops the off-beat hats.
        for beat in range(4):
            pos = bar_start + int(beat * spb * sr)
            audio[pos:pos + len(k)] += k * level
            if not in_outro:
                off = pos + int(spb * sr / 2)
                audio[off:off + len(h)] += h * level

        # Harmony only in the intro/body; the outro strips back to drums.
        if in_outro:
            continue

        degree, triad = progression[bar % len(progression)]
        root = tonic + degree
        n = int(4 * spb * sr)

        # Bass: root, two octaves below middle.
        bass_midi = 36 + root
        audio[bar_start:bar_start + n] += _tone([midi_to_hz(bass_midi)], n, sr, 3) * 0.45 * level

        # Pad: triad on the chord root, softly enveloped so it doesn't smear onsets.
        chord = [midi_to_hz(60 + root + iv) for iv in triad]
        pad = _tone(chord, n, sr, 4) * 0.22 * level
        edge = int(sr * 0.05)
        pad[:edge] *= np.linspace(0, 1, edge)
        pad[-edge:] *= np.linspace(1, 0, edge)
        audio[bar_start:bar_start + n] += pad

    audio += rng.standard_normal(total) * 1e-4  # dither; keeps analysis off exact zeros
    audio /= max(1e-9, np.max(np.abs(audio)) * 1.05)

    truth = {
        "bpm": bpm,
        "tonic": tonic % 12,
        "is_major": is_major,
        "bars": bars,
        "beat_times": [i * spb for i in range(beats)],
        "downbeat_times": [b * 4 * spb for b in range(bars)],
        "mix_in": intro_bars * 4 * spb,
        "mix_out": (bars - outro_bars) * 4 * spb,
        "duration": beats * spb,
    }
    return audio.astype(np.float32), truth


def click_track(bpm: float = 128.0, bars: int = 16, sr: int = SR):
    """Bare accented click track - the cleanest possible beat-tracking target."""
    spb = 60.0 / bpm
    beats = bars * 4
    audio = np.zeros(int(beats * spb * sr) + sr)
    k = kick(sr)
    for i in range(beats):
        pos = int(i * spb * sr)
        audio[pos:pos + len(k)] += k * (1.0 if i % 4 == 0 else 0.6)
    truth = {"bpm": bpm, "beat_times": [i * spb for i in range(beats)],
             "downbeat_times": [i * spb for i in range(0, beats, 4)]}
    return (audio / max(1e-9, np.max(np.abs(audio)) * 1.05)).astype(np.float32), truth
