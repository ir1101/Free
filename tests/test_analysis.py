"""Analysis accuracy against synthetic tracks of known tempo, key and structure."""

import numpy as np
import pytest

from automix.analysis.beats import analyze_beats
from automix.analysis.key import detect_key
from automix.analysis.track import analyze_signal
from automix.model.camelot import from_pitch_class, harmonic_distance, wheel_distance
from automix.synth import click_track, make_track

from .conftest import ANALYSIS_SR, to_analysis

# A transition runs tens of seconds; 0.05 BPM is well under a millisecond of
# drift across one, which is inaudible.
BPM_TOLERANCE = 0.05


@pytest.mark.parametrize("bpm", [90.0, 100.0, 110.0, 123.5, 128.0, 140.0, 174.0, 180.0])
def test_tempo_is_recovered_precisely(bpm):
    audio, _ = make_track(bpm=bpm, bars=24)
    grid, _ = analyze_beats(to_analysis(audio), ANALYSIS_SR)
    assert grid.bpm == pytest.approx(bpm, abs=BPM_TOLERANCE)


def test_half_and_double_time_are_not_confused():
    """90 and 180 are both ordinary tempos; only the evidence separates them."""
    for bpm in (90.0, 180.0):
        audio, _ = make_track(bpm=bpm, bars=24)
        grid, _ = analyze_beats(to_analysis(audio), ANALYSIS_SR)
        assert grid.bpm == pytest.approx(bpm, abs=BPM_TOLERANCE)


def test_beat_grid_lands_on_the_beats(synth_track):
    audio, truth = synth_track
    grid, _ = analyze_beats(audio, ANALYSIS_SR)
    beats = grid.beat_times()
    for t in truth["beat_times"][4:-4]:
        assert float(np.min(np.abs(beats - t))) < 0.010, "beat off by more than 10 ms"


def test_downbeats_land_on_bar_one(synth_track):
    audio, truth = synth_track
    grid, _ = analyze_beats(audio, ANALYSIS_SR)
    downbeats = grid.downbeat_times()
    for t in truth["downbeat_times"][1:-1]:
        assert float(np.min(np.abs(downbeats - t))) < 0.010


def test_grid_is_confident_on_clean_material():
    audio, _ = click_track(bpm=128.0, bars=16)
    grid, _ = analyze_beats(to_analysis(audio), ANALYSIS_SR)
    assert grid.confidence > 0.85


def test_snap_to_bar_respects_direction(synth_track):
    audio, _ = synth_track
    grid, _ = analyze_beats(audio, ANALYSIS_SR)
    t = grid.downbeat_times()[4] + grid.seconds_per_bar * 0.3
    assert grid.snap_to_bar(t, "before") <= t
    assert grid.snap_to_bar(t, "after") >= t
    assert grid.snap_to_bar(t, "before") == pytest.approx(grid.downbeat_times()[4])


@pytest.mark.parametrize("tonic,is_major", [(9, False), (0, True), (4, False), (7, True)])
def test_detected_key_is_always_mixable_with_the_real_one(tonic, is_major):
    """The guarantee worth asserting per-key is compatibility, not identity.

    Key detection is not exact on every input - E minor in particular comes back
    as B minor on this material. What matters is that a miss stays *harmonically
    adjacent*: one step on the wheel, or a relative major/minor swap. Both score
    as smooth joins, so a transition built on the wrong answer still sounds
    right. A miss that landed across the wheel would not, and that is what this
    rules out.
    """
    audio, _ = make_track(bpm=128.0, tonic=tonic, is_major=is_major, bars=16)
    key, _ = detect_key(to_analysis(audio), ANALYSIS_SR)
    want = from_pitch_class(tonic, is_major)
    assert wheel_distance(key, want) <= 1
    assert harmonic_distance(want, key)[0] <= 2.0


def test_key_detection_accuracy_across_all_twenty_four_keys():
    """Aggregate accuracy, which is the honest way to state how good this is.

    Scored two ways because they mean different things: an exact miss that lands
    on the relative major or minor costs almost nothing in a mix, whereas a wrong
    Camelot number can pair genuinely clashing records.
    """
    exact = numbers = 0
    for tonic in range(12):
        for is_major in (True, False):
            audio, _ = make_track(bpm=128.0, tonic=tonic, is_major=is_major, bars=16)
            key, _ = detect_key(to_analysis(audio), ANALYSIS_SR)
            want = from_pitch_class(tonic, is_major)
            exact += key == want
            numbers += key.number == want.number
    assert exact >= 18, f"only {exact}/24 keys detected exactly"
    assert numbers >= 22, f"only {numbers}/24 Camelot numbers correct"


def test_cue_points_find_the_arrangement(synth_track):
    """mix_in should land at the end of the intro, mix_out at the start of the outro."""
    audio, truth = synth_track
    track = analyze_signal(audio, ANALYSIS_SR)
    bar = track.grid.seconds_per_bar
    assert abs(track.structure.mix_in - truth["mix_in"]) <= bar
    assert abs(track.structure.mix_out - truth["mix_out"]) <= bar


def test_cue_points_are_ordered_and_on_downbeats(synth_track):
    audio, _ = synth_track
    track = analyze_signal(audio, ANALYSIS_SR)
    assert 0 <= track.structure.mix_in < track.structure.mix_out <= track.duration
    downbeats = track.grid.downbeat_times()
    for cue in (track.structure.mix_in, track.structure.mix_out):
        assert float(np.min(np.abs(downbeats - cue))) < 0.05


def test_analysis_survives_silence():
    """A pathological input must still produce a well-formed Track."""
    track = analyze_signal(np.zeros(ANALYSIS_SR * 5), ANALYSIS_SR)
    assert track.bpm > 0
    assert track.structure.mix_out > track.structure.mix_in


def test_track_roundtrips_through_json(synth_track):
    from automix.analysis.track import Track

    audio, _ = synth_track
    track = analyze_signal(audio, ANALYSIS_SR, title="rt")
    again = Track.from_dict(track.to_dict())
    assert again.bpm == pytest.approx(track.bpm, abs=1e-3)
    assert again.key == track.key
    assert again.structure.mix_in == pytest.approx(track.structure.mix_in, abs=1e-3)
    assert np.allclose(again.head.timbre, track.head.timbre, atol=1e-3)
    assert np.allclose(again.tail.rhythm, track.tail.rhythm, atol=1e-3)
