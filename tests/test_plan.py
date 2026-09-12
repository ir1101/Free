"""Timeline construction and rendering.

The properties asserted here are the ones a listener would notice: fades land
where the arrangement expects them, the rate schedule and the position integral
agree, and nothing overlaps that should not.
"""

import numpy as np
import pytest

from automix.analysis.track import analyze_signal
from automix.model.cost import Weights
from automix.model.plan import SetPlan, build_plan
from automix.model.sequencer import sequence
from automix.render.mixdown import render
from automix.synth import SR, make_track

from .conftest import ANALYSIS_SR, to_analysis

SPECS = [(124.0, 9, False), (126.0, 4, False), (122.0, 11, True)]


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    from automix import audio_io

    root = tmp_path_factory.mktemp("plan")
    tracks = []
    for i, (bpm, tonic, is_major) in enumerate(SPECS):
        audio, _ = make_track(bpm=bpm, tonic=tonic, is_major=is_major, bars=32)
        path = root / f"t{i}.wav"
        audio_io.write(path, audio, SR)
        track = analyze_signal(to_analysis(audio), ANALYSIS_SR, path=path, title=f"t{i}")
        tracks.append(track)

    order = sequence(tracks, Weights(), arc_shape="peak")
    plan = build_plan(tracks, order, sample_rate=22050)
    return tracks, plan


def test_plan_has_a_transition_between_every_pair(planned):
    tracks, plan = planned
    assert len(plan.segments) == len(tracks)
    assert len(plan.transitions) == len(tracks) - 1
    for i, tr in enumerate(plan.transitions):
        assert tr.from_index == i and tr.to_index == i + 1


def test_segments_run_in_order_and_overlap_only_during_transitions(planned):
    _, plan = planned
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert b.start_time > a.start_time
        # The next track starts before this one ends - that overlap is the mix.
        assert b.start_time < a.end_time
        # ...and only by the length of the crossfade.
        assert a.end_time == pytest.approx(b.start_time + plan.transitions[a.index].duration,
                                           abs=1e-6)


def test_crossfade_ends_at_the_outgoing_mix_out_cue(planned):
    """The fade must finish as the arrangement thins out, not begin there.

    Starting on the cue leaves the first half of the fade with a stripped-back
    outgoing track and a turned-down incoming one, which measures as a hole in
    the middle of the mix.
    """
    tracks, plan = planned
    by_id = {t.track_id: t for t in tracks}
    for tr in plan.transitions:
        seg = plan.segments[tr.from_index]
        track = by_id[seg.track_id]
        source_at_end = seg.source_at(tr.end_time)
        assert source_at_end == pytest.approx(track.structure.mix_out, abs=0.05)


def test_incoming_track_starts_at_its_mix_in_cue(planned):
    tracks, plan = planned
    by_id = {t.track_id: t for t in tracks}
    for seg in plan.segments[1:]:
        assert seg.source_start == pytest.approx(by_id[seg.track_id].structure.mix_in, abs=1e-6)


def test_first_track_plays_from_the_top(planned):
    _, plan = planned
    assert plan.segments[0].source_start == 0.0
    assert plan.segments[0].playback_rate == 1.0


def test_incoming_rate_matches_the_outgoing_tempo(planned):
    _, plan = planned
    for tr in plan.transitions:
        a, b = plan.segments[tr.from_index], plan.segments[tr.to_index]
        # Played at this rate, B's tempo equals A's for the length of the overlap.
        assert b.bpm * b.playback_rate == pytest.approx(a.bpm, rel=1e-6)


def test_rate_returns_to_native_after_the_overlap(planned):
    _, plan = planned
    for seg in plan.segments[1:]:
        assert seg.rate_ramp_end > seg.rate_ramp_start
        # Well past the ramp, one second of set time is one second of source.
        t = seg.rate_ramp_end + 5.0
        assert seg.source_at(t + 1.0) - seg.source_at(t) == pytest.approx(1.0, abs=1e-6)


def test_source_position_is_the_integral_of_the_rate(planned):
    """Web Audio advances a source node by integrating playbackRate.

    If this closed form and that integral disagreed, the browser and the offline
    render would drift apart - so check it numerically.
    """
    _, plan = planned
    seg = plan.segments[1]
    times = np.linspace(seg.start_time, min(seg.end_time, seg.rate_ramp_end + 10), 4000)
    rates = []
    for t in times:
        if t <= seg.rate_ramp_start:
            rates.append(seg.playback_rate)
        elif t >= seg.rate_ramp_end:
            rates.append(1.0)
        else:
            frac = (t - seg.rate_ramp_start) / (seg.rate_ramp_end - seg.rate_ramp_start)
            rates.append(seg.playback_rate + (1.0 - seg.playback_rate) * frac)
    integrated = seg.source_start + np.concatenate([[0], np.cumsum(
        np.diff(times) * (np.array(rates)[:-1] + np.array(rates)[1:]) / 2)])
    closed_form = np.array([seg.source_at(t) for t in times])
    assert np.max(np.abs(integrated - closed_form)) < 1e-3


def test_overlap_shrinks_rather_than_overrunning_a_short_track():
    """A track with a short body should get a shorter fade, not a broken one."""
    from automix import audio_io  # noqa: F401  (import kept local to the test)

    audio, _ = make_track(bpm=128.0, bars=12, intro_bars=2, outro_bars=2)
    short = analyze_signal(to_analysis(audio), ANALYSIS_SR, title="short")
    audio2, _ = make_track(bpm=128.0, tonic=2, bars=32)
    long = analyze_signal(to_analysis(audio2), ANALYSIS_SR, title="long")

    order = sequence([short, long], Weights(), arc_shape="flat", start=0)
    plan = build_plan([short, long], order, sample_rate=22050)
    tr = plan.transitions[0]
    body = short.structure.mix_out - short.structure.mix_in
    assert 0 < tr.duration <= body * 0.5 + 1e-6
    assert tr.bars >= 1
    assert plan.segments[0].start_time < tr.start_time


def test_plan_roundtrips_through_json(planned, tmp_path):
    _, plan = planned
    path = plan.save(tmp_path / "set.json")
    again = SetPlan.load(path)
    assert again.duration == pytest.approx(plan.duration, abs=1e-5)
    assert len(again.segments) == len(plan.segments)
    for a, b in zip(plan.segments, again.segments):
        assert a.start_time == pytest.approx(b.start_time, abs=1e-5)
        assert a.source_at(a.start_time + 12.0) == pytest.approx(
            b.source_at(b.start_time + 12.0), abs=1e-4)
    for a, b in zip(plan.transitions, again.transitions):
        assert np.allclose(a.gain_in.values, b.gain_in.values, atol=1e-5)


def test_render_produces_audio_without_clipping_or_gaps(planned):
    _, plan = planned
    mix = render(plan)
    assert mix.ndim == 2 and mix.shape[1] == 2
    assert np.max(np.abs(mix)) <= 1.0
    assert len(mix) > plan.sample_rate * 30

    # No silent gap anywhere in the body of the set.
    sr = plan.sample_rate
    mono = mix.mean(axis=1)
    window = sr // 2
    trimmed = mono[sr: len(mono) - 2 * sr]
    usable = len(trimmed) - len(trimmed) % window
    frames = trimmed[:usable].reshape(-1, window)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    assert float(np.min(rms)) > float(np.median(rms)) * 0.05


def test_loudness_does_not_sag_through_a_crossfade(planned):
    """The equal-power payoff, measured on rendered audio.

    A linear crossfade drops about 3 dB at its midpoint. This asserts the
    rendered mix stays well inside that.
    """
    _, plan = planned
    mix = render(plan).mean(axis=1)
    sr = plan.sample_rate

    for tr in plan.transitions:
        a, b = int(tr.start_time * sr), int(tr.end_time * sr)
        window = int(sr * 0.4)
        power = np.concatenate([[0.0], np.cumsum(mix[a:b].astype(np.float64) ** 2)])
        rms = np.sqrt((power[window:] - power[:-window]) / window)
        quietest = 20 * np.log10(max(float(np.min(rms)), 1e-9))

        before = mix[max(0, a - 3 * sr):a]
        after = mix[b:b + 3 * sr]
        reference = 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(
            np.concatenate([before, after]) ** 2)))))
        assert quietest - reference > -3.0, "crossfade sags like a linear fade"


def test_render_is_deterministic(planned):
    _, plan = planned
    assert np.array_equal(render(plan), render(plan))
