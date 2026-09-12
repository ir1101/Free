"""Rendering a SetPlan to audio.

The renderer walks the plan's timeline, resamples each track according to its
rate schedule, applies the plan's automation curves, and sums the result.

**Why resampling rather than a phase vocoder.** Matching tempo by resampling
moves pitch with speed, exactly as a turntable does; a phase vocoder would hold
pitch but smears transients. The deciding factor is not which artefact is
prettier - it is that the browser engine tempo-matches with ``playbackRate``,
which *is* resampling. Rendering the same plan a different way would produce a
file that does not match what you auditioned, and the whole point of a shared
plan is that the two agree. The cost model already keeps stretches inside about
6%, where the pitch shift is under a semitone and reads as a DJ nudge rather than
a mistake.

Linear interpolation is used for the resample. Within a few percent of unity its
error sits far below the noise floor of any real recording; it would be the wrong
choice for a large ratio, which is precisely what the cost model avoids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio_io import Audio, load_for_render, peak_normalise
from ..model.plan import Segment, SetPlan
from .crossfade import apply_bass_gain, apply_highpass_sweep


def _resample_segment(source: np.ndarray, sr: int, seg: Segment, n_out: int) -> np.ndarray:
    """Read `n_out` output samples from `source` following the segment's rate schedule.

    Output sample k corresponds to set time ``start_time + k / sr``; the plan
    turns that into a source position. Sampling the source at those positions is
    the whole resampler.
    """
    # `source_at` is a cheap closed form, but calling it once per output sample
    # dominates the render. The rate schedule is piecewise-smooth and almost
    # everywhere constant, so evaluating it on a coarse grid and interpolating
    # between is exact to far under one sample.
    if n_out > 4096:
        coarse_n = max(64, n_out // 512)
        coarse_t = seg.start_time + np.linspace(0, (n_out - 1) / sr, coarse_n)
        coarse_pos = np.array([seg.source_at(x) for x in coarse_t]) * sr
        positions = np.interp(np.arange(n_out), np.linspace(0, n_out - 1, coarse_n), coarse_pos)
    else:
        t = seg.start_time + np.arange(n_out) / sr
        positions = np.array([seg.source_at(x) for x in t]) * sr

    positions = np.clip(positions, 0, len(source) - 1)
    index = np.arange(len(source))
    if source.ndim == 1:
        return np.interp(positions, index, source)
    return np.stack([np.interp(positions, index, source[:, c])
                     for c in range(source.shape[1])], axis=1)


def _apply_curve(block: np.ndarray, values: np.ndarray) -> np.ndarray:
    return block * (values[:, None] if block.ndim == 2 else values)


def render(plan: SetPlan, sources: dict[str, Audio] | None = None,
           filter_sweep: bool = False, normalise: bool = True,
           progress=None) -> np.ndarray:
    """Render `plan` to a stereo array at the plan's sample rate."""
    sr = plan.sample_rate
    total = int(np.ceil(plan.duration * sr)) + sr
    mix = np.zeros((total, 2), dtype=np.float64)

    # Which transition does each segment play the outgoing / incoming part in?
    outgoing = {t.from_index: t for t in plan.transitions}
    incoming = {t.to_index: t for t in plan.transitions}

    for seg in plan.segments:
        if progress:
            progress(seg.index, len(plan.segments), seg.title)

        audio = (sources or {}).get(seg.path)
        if audio is None:
            audio = load_for_render(Path(seg.path), sr=sr)
        source = audio.samples
        if source.ndim == 1:
            source = np.repeat(source[:, None], 2, axis=1)

        start = int(round(seg.start_time * sr))
        n_out = max(0, int(round((seg.end_time - seg.start_time) * sr)))
        if n_out <= 0:
            continue
        n_out = min(n_out, total - start)
        if n_out <= 0:
            continue

        block = _resample_segment(source, sr, seg, n_out)

        # Incoming half of the transition this track enters on.
        tr = incoming.get(seg.index)
        if tr is not None:
            a = 0
            b = min(n_out, int(round(tr.duration * sr)))
            if b > a:
                block[a:b] = apply_bass_gain(block[a:b], sr, tr.bass_in.sample(b - a))
                block[a:b] = _apply_curve(block[a:b], tr.gain_in.sample(b - a))

        # Outgoing half of the transition this track hands over on.
        tr = outgoing.get(seg.index)
        if tr is not None:
            a = int(round((tr.start_time - seg.start_time) * sr))
            b = min(n_out, a + int(round(tr.duration * sr)))
            a = max(0, a)
            if b > a:
                span = b - a
                if filter_sweep:
                    block[a:b] = apply_highpass_sweep(block[a:b], sr,
                                                      tr.filter_out.sample(span))
                block[a:b] = apply_bass_gain(block[a:b], sr, tr.bass_out.sample(span))
                block[a:b] = _apply_curve(block[a:b], tr.gain_out.sample(span))
            # Nothing of this track should be heard after the handover completes.
            if b < n_out:
                block[b:] = 0.0

        mix[start:start + n_out] += block

    # Trim the tail of silence the padding left behind.
    loud = np.flatnonzero(np.max(np.abs(mix), axis=1) > 1e-5)
    if len(loud):
        mix = mix[: loud[-1] + 1]

    return peak_normalise(mix) if normalise else mix


def render_transition(plan: SetPlan, index: int, sources: dict[str, Audio] | None = None,
                      lead_bars: float = 2.0, tail_bars: float = 2.0,
                      filter_sweep: bool = False) -> tuple[np.ndarray, float]:
    """Render one transition with a little context either side.

    Auditioning a single join is the fastest way to tell whether the analysis got
    a track right, and far quicker than playing a whole set to reach it.
    Returns ``(audio, offset_seconds)`` where the offset is where the transition
    itself begins within the excerpt.
    """
    if not 0 <= index < len(plan.transitions):
        raise IndexError(f"no transition {index}")

    tr = plan.transitions[index]
    seg = next(s for s in plan.segments if s.index == tr.from_index)
    bar = 60.0 / max(1e-9, seg.bpm) * 4.0
    lead, tail = lead_bars * bar, tail_bars * bar

    full = render(plan, sources=sources, filter_sweep=filter_sweep, normalise=False)
    sr = plan.sample_rate
    a = max(0, int((tr.start_time - lead) * sr))
    b = min(len(full), int((tr.end_time + tail) * sr))
    return peak_normalise(full[a:b]), min(lead, tr.start_time)
