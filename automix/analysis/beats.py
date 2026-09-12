"""Tempo, beat grid and downbeat detection.

Everything downstream mixes on *bars*, so this module has to deliver three
things: an accurate tempo, a phase-locked grid, and which beat of the bar is
beat one. Getting the last one wrong is the most common way a technically
correct crossfade still sounds wrong.

A note on accuracy: ``librosa.beat.beat_track`` reports tempo from a tempogram
whose bins are coarse - it will happily call 128 BPM "129.2". Over a 32-bar
transition a 1% tempo error accumulates into roughly a whole beat of drift,
which is very audible. So the reported tempo is treated as a starting guess
only, and the real tempo is recovered by least-squares fitting a regular grid to
the detected beat *times*, which are far more precise than the tempo estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

HOP = 512
ANALYSIS_SR = 22050
BEATS_PER_BAR = 4

# Tempo range we fold octave errors into. Beat trackers routinely latch onto
# half or double time.
MIN_BPM = 70.0
MAX_BPM = 185.0

# Spectral flux compares each frame against the previous one, so an onset is
# first *reported* on the frame after it actually begins. Measured against
# synthetic tracks of known timing this is a flat ~25 ms at every tempo - one
# hop - so it is a fixed reporting lag, not a tempo-dependent effect, and it is
# subtracted back off when a beat is placed.
#
# It would largely cancel between two tracks in a mix, since both grids carry
# it, but cue points are also reported to the user and rendered against the
# original audio, so it is corrected at the source instead.
FLUX_LAG_FRAMES = 1.0


@dataclass(slots=True)
class BeatGrid:
    """A regular bar-aligned grid fitted to the track."""

    bpm: float
    phase: float          # time of grid beat 0, seconds
    downbeat: int         # which grid index is beat one (0-3)
    n_beats: int
    confidence: float     # how tightly detected beats hug the fitted grid, 0-1

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * BEATS_PER_BAR

    def beat_time(self, i: int) -> float:
        return self.phase + i * self.seconds_per_beat

    def beat_times(self) -> np.ndarray:
        return self.phase + np.arange(self.n_beats) * self.seconds_per_beat

    def downbeat_times(self) -> np.ndarray:
        idx = np.arange(self.downbeat, self.n_beats, BEATS_PER_BAR)
        return self.phase + idx * self.seconds_per_beat

    def nearest_downbeat(self, t: float) -> float:
        db = self.downbeat_times()
        if len(db) == 0:
            return t
        return float(db[int(np.argmin(np.abs(db - t)))])

    def snap_to_bar(self, t: float, prefer: str = "nearest") -> float:
        """Snap `t` to a downbeat, optionally biased forward or back."""
        db = self.downbeat_times()
        if len(db) == 0:
            return t
        if prefer == "before":
            earlier = db[db <= t + 1e-6]
            return float(earlier[-1]) if len(earlier) else float(db[0])
        if prefer == "after":
            later = db[db >= t - 1e-6]
            return float(later[0]) if len(later) else float(db[-1])
        return self.nearest_downbeat(t)

    def to_dict(self) -> dict:
        return {
            "bpm": round(self.bpm, 4),
            "phase": round(self.phase, 6),
            "downbeat": self.downbeat,
            "n_beats": self.n_beats,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        return cls(d["bpm"], d["phase"], d["downbeat"], d["n_beats"], d["confidence"])


def _fit_grid(times: np.ndarray, spb_guess: float):
    """Least-squares fit a regular grid to detected beat times.

    Returns ``(seconds_per_beat, phase, confidence)``.

    Beat times come back quantised to the analysis frame grid (~23 ms), so
    consecutive gaps jitter either side of the true beat length even when the
    tempo is dead steady. Averaged over a whole track that jitter cancels, which
    is why fitting recovers a far better tempo than the tempogram reports.

    Indices are seeded from *consecutive differences*, not by extrapolating the
    initial tempo guess across the track. That distinction matters: extrapolating
    a guess that is off by 2% misassigns indices near the end, and the fit then
    happily reproduces the very error it was meant to correct. Local gaps are
    each about one beat, so rounding them individually stays correct no matter
    how poor the starting guess is.
    """
    if len(times) < 4:
        return spb_guess, float(times[0]) if len(times) else 0.0, 0.0

    steps = np.maximum(1, np.round(np.diff(times) / spb_guess))
    idx = np.concatenate([[0.0], np.cumsum(steps)])

    spb, phase = spb_guess, float(times[0])
    for _ in range(8):
        if len(np.unique(idx)) < 2:
            break
        spb, phase = np.polyfit(idx, times, 1)
        new_idx = np.round((times - phase) / spb)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx

    residual = times - (phase + idx * spb)
    # Residuals as a fraction of a beat: a perfectly steady grid scores 1.0,
    # smearing of a quarter-beat or worse scores 0.
    rms = float(np.sqrt(np.mean(residual**2)))
    confidence = float(np.clip(1.0 - (rms / (spb * 0.25)), 0.0, 1.0))
    return float(spb), float(phase), confidence


def percussive_onset(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset envelope weighted towards the low end, where the beat actually is.

    Plain spectral flux responds to brightness, so a hi-hat - spread across many
    mel bands - out-peaks a kick drum that occupies only a few. Snapping a grid
    to that envelope drags it onto the off-beats, half a beat out, which is
    exactly wrong: the kick *is* the beat in most dance music.

    Restricting to the low bands puts the kick back on top. The full-band
    envelope is still mixed in at lower weight so that material with no kick at
    all (acoustic, ambient) still has something to lock onto.
    """
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmax=250, n_mels=24)
    full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    n = min(len(low), len(full))
    low, full = low[:n], full[:n]

    def unit(v):
        peak = float(np.max(v)) if len(v) else 0.0
        return v / peak if peak > 1e-9 else v

    return 2.0 * unit(low) + unit(full)


def _align_phase(onset: np.ndarray, sr: int, spb: float, phase: float,
                 duration: float, n: int = 96) -> float:
    """Globally align the grid's phase by sweeping a whole beat period.

    Local peak-snapping can only correct a beat position by less than its search
    window, so it cannot rescue a grid that starts out half a beat wrong - the
    true onset simply is not inside the window, and the fit settles confidently
    onto the off-beats. Beat trackers land there often enough that this has to be
    handled explicitly rather than hoped away.

    Sweeping every phase across one full beat and scoring each against the onset
    envelope makes the choice global: there is nowhere for a better alignment to
    hide. Cheap, too - one beat's worth of offsets, scored by array indexing.
    """
    n_beats = max(4, int((duration - phase) / spb) + 1)
    idx = np.arange(n_beats)
    offsets = np.linspace(0.0, spb, n, endpoint=False)

    best_score, best_offset = -np.inf, 0.0
    for off in offsets:
        frames = np.round((phase + off + idx * spb) * sr / HOP).astype(int)
        frames = frames[(frames >= 0) & (frames < len(onset))]
        if len(frames) < 4:
            continue
        score = float(np.mean(onset[frames]))
        if score > best_score:
            best_score, best_offset = score, float(off)

    aligned = phase + best_offset
    # Keep the grid starting at or before the first audio.
    while aligned - spb > 0:
        aligned -= spb
    return aligned


def _peak_snap(onset: np.ndarray, sr: int, times: np.ndarray, spb: float,
               search: float = 0.25):
    """Move each predicted beat onto the nearest real onset peak.

    Returns ``(snapped_times, kept_mask)``.

    The tempo estimate is only ever as good as the beat positions it is fitted
    to, and a beat tracker's picks scatter far more than the frame grid does.
    Searching a window around each predicted beat for the actual onset maximum
    replaces those noisy picks with the evidence in the signal, and a parabolic
    fit through the peak recovers position to well under one frame.

    Beats whose window holds nothing convincing are dropped rather than
    contributing a bogus position - quiet passages and breakdowns should not
    drag the grid around.
    """
    half = max(1, int(round(search * spb * sr / HOP)))
    floor = float(np.median(onset)) + 1e-9

    snapped = np.empty(len(times))
    keep = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        centre = int(round(t * sr / HOP))
        lo, hi = max(0, centre - half), min(len(onset), centre + half + 1)
        if hi - lo < 3:
            snapped[i] = t
            continue
        window = onset[lo:hi]
        k = int(np.argmax(window))
        if window[k] <= floor:  # nothing here worth trusting
            snapped[i] = t
            continue
        j = lo + k
        # Parabolic interpolation for sub-frame resolution.
        if 0 < j < len(onset) - 1:
            a, b, c = onset[j - 1], onset[j], onset[j + 1]
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            delta = 0.0
        snapped[i] = (j + delta - FLUX_LAG_FRAMES) * HOP / sr
        keep[i] = True
    return snapped, keep


def _refine_grid(onset: np.ndarray, sr: int, spb: float, phase: float,
                 duration: float, passes: int = 4):
    """Alternate between snapping beats to onsets and refitting the grid.

    Each pass predicts beats from the current grid, snaps them onto real onsets,
    then least-squares refits. Two or three passes are usually enough; the loop
    exits early once the tempo stops moving.

    Residual trimming guards the fit against the handful of beats that snap onto
    a syncopation or a fill instead of the beat itself.

    The search window starts wide and narrows. It has to: a starting tempo that
    is off by even half a percent accumulates most of a beat of drift over a few
    hundred beats, so a tight window would place its late beats past the real
    ones, snap them onto the wrong onsets, and fit a confidently wrong line. A
    first pass spanning a full beat period always brackets the right onset; once
    the tempo is close, narrowing keeps the fit from wandering onto syncopations.

    Returns ``(seconds_per_beat, phase, confidence)``.
    """
    confidence = 0.0
    windows = [0.5, 0.35, 0.25, 0.25]
    for p in range(passes):
        n = max(4, int((duration - phase) / spb) + 1)
        predicted = phase + np.arange(n) * spb
        snapped, keep = _peak_snap(onset, sr, predicted, spb,
                                   search=windows[min(p, len(windows) - 1)])
        if keep.sum() < 4:
            break

        idx = np.arange(n)[keep]
        obs = snapped[keep]

        # Trim outliers: fit once, discard the worst offenders, fit again.
        slope, intercept = np.polyfit(idx, obs, 1)
        residual = obs - (intercept + idx * slope)
        limit = max(2.5 * float(np.std(residual)), 0.02 * spb)
        good = np.abs(residual) <= limit
        if good.sum() >= 4:
            slope, intercept = np.polyfit(idx[good], obs[good], 1)

        converged = abs(slope - spb) < 1e-9
        spb, phase = float(slope), float(intercept)
        confidence = float(np.clip(1.0 - (float(np.sqrt(np.mean(residual[good]**2)))
                                          / (spb * 0.25)), 0.0, 1.0)) if good.sum() >= 4 else 0.0
        if converged:
            break
    return spb, phase, confidence


def _resolve_octave(onset: np.ndarray, sr: int, spb: float, phase: float,
                    duration: float, ratio: float = 0.62) -> float:
    """Decide between a tempo and half or double that tempo, using evidence.

    Beat trackers routinely report half or double the real tempo, and clamping
    to a plausible BPM range cannot separate the two: 90 and 180 are both
    perfectly ordinary tempos, so a range check has no way to tell a 90 BPM track
    from a 180 BPM one that was detected at half speed.

    What does tell them apart is what sits between the beats. Interleave a beat
    halfway between each pair and look at the onsets there: if those midpoints
    are struck about as hard as the beats themselves, they *are* beats and the
    tempo should be doubled. If they are much weaker - off-beat hats, or nothing
    - the slower reading was right.

    This works because the envelope it reads is weighted towards the low end,
    where kicks live: a hi-hat between two kicks will not clear the threshold,
    but a kick will.
    """
    def strength(period: float, offset: float) -> float:
        times = np.arange(phase + offset, duration, period)
        if len(times) < 4:
            return 0.0
        frames = np.round(times * sr / HOP).astype(int)
        frames = frames[(frames >= 0) & (frames < len(onset))]
        return float(np.mean(onset[frames])) if len(frames) >= 4 else 0.0

    on_beat = strength(spb, 0.0)
    if on_beat <= 1e-9:
        return spb

    # Double time: are the midpoints between beats also being hit?
    midpoints = strength(spb, spb / 2.0)
    if midpoints >= ratio * on_beat and 60.0 / (spb / 2.0) <= MAX_BPM:
        return spb / 2.0

    # Half time: are every-other beats much weaker, meaning we counted twice?
    alternate = strength(spb * 2.0, spb)
    if alternate < ratio * strength(spb * 2.0, 0.0) and 60.0 / (spb * 2.0) >= MIN_BPM:
        return spb * 2.0

    return spb


def _estimate_downbeat(onset_env: np.ndarray, sr: int, spb: float, phase: float,
                       n_beats: int) -> int:
    """Pick which of the 4 beat positions carries beat one.

    Assumes 4/4 and scores each candidate phase by mean onset strength across
    the beats it would make downbeats. Downbeats normally carry the strongest
    accent, so the winning phase is the one with the loudest onsets.
    """
    times = phase + np.arange(n_beats) * spb
    frames = np.clip(librosa.time_to_frames(times, sr=sr, hop_length=HOP),
                     0, len(onset_env) - 1)
    strength = onset_env[frames]
    scores = []
    for p in range(BEATS_PER_BAR):
        sel = strength[p::BEATS_PER_BAR]
        scores.append(float(np.mean(sel)) if len(sel) else 0.0)
    return int(np.argmax(scores))


def analyze_beats(y: np.ndarray, sr: int = ANALYSIS_SR):
    """Detect tempo, fit a bar grid, return ``(BeatGrid, onset_envelope)``.

    The onset envelope is handed back because structure detection reuses it -
    it is the most expensive thing computed here.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    duration = len(y) / sr

    if len(beat_times) < 4 or tempo <= 0:
        # Nothing rhythmic to lock onto; return a well-formed nominal grid so
        # the rest of the pipeline still has something to work with.
        return BeatGrid(120.0, 0.0, 0, max(1, int(duration / 0.5)), 0.0), onset_env

    spb, phase, confidence = _fit_grid(beat_times, 60.0 / tempo)
    beat_env = percussive_onset(y, sr)

    # The coarse fit lands within a fraction of a percent. That is not good
    # enough: 0.3% is a sixth of a beat of drift across a 32-bar transition, and
    # audible as sloppiness. Snapping to onsets closes the gap.
    phase = _align_phase(beat_env, sr, spb, phase, duration)
    spb, phase, confidence = _refine_grid(beat_env, sr, spb, phase, duration)

    spb = _resolve_octave(beat_env, sr, spb, phase, duration)

    # Only after the evidence-based choice, force anything still out of range
    # back into it.
    while 60.0 / spb < MIN_BPM:
        spb /= 2.0
    while 60.0 / spb > MAX_BPM:
        spb *= 2.0

    n_beats = max(1, int((duration - phase) / spb) + 1)
    downbeat = _estimate_downbeat(beat_env, sr, spb, phase, n_beats)


    # Pull the phase back so the grid starts at or before the first audio,
    # keeping the downbeat pointing at the same musical position.
    while phase - spb > 0:
        phase -= spb
        downbeat = (downbeat + 1) % BEATS_PER_BAR
        n_beats += 1

    return BeatGrid(60.0 / spb, phase, downbeat, n_beats, confidence), onset_env
