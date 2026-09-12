"""The edge features the expanded cost model is built on.

Each of these is asserted against material constructed to differ in exactly one
way, so a feature that responds to the wrong thing shows up immediately.
"""

import numpy as np
import pytest
from scipy import signal as ss

from automix.analysis.beats import analyze_beats
from automix.analysis.energy import analyze_energy
from automix.analysis.features import (
    RHYTHM_SLOTS, build_context, edge_features, rhythm_distance, rhythm_pattern,
)
from automix.synth import SR, make_track

from .conftest import ANALYSIS_SR, to_analysis


def describe(audio):
    y = to_analysis(audio)
    ctx = build_context(y, ANALYSIS_SR)
    grid, _ = analyze_beats(y, ANALYSIS_SR)
    profile = analyze_energy(y, ANALYSIS_SR, onset_env=ctx.onset)
    return edge_features(ctx, 2.0, min(len(y) / ANALYSIS_SR, 20.0), grid, profile), ctx, grid


@pytest.fixture(scope="module")
def plain():
    audio, _ = make_track(bpm=128.0, bars=24)
    return describe(audio)


def test_features_are_in_range(plain):
    f, _, _ = plain
    for name in ("loudness", "brightness", "bass", "voice", "crest", "pulse",
                 "density", "energy"):
        value = getattr(f, name)
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
    assert len(f.rhythm) == RHYTHM_SLOTS
    assert f.loudness_db < 0.0


def test_brightness_tracks_the_spectrum(plain):
    """A high-passed copy must read as brighter than the original."""
    base, _, _ = plain
    audio, _ = make_track(bpm=128.0, bars=24)
    sos = ss.butter(4, 3000.0, btype="high", fs=SR, output="sos")
    bright, _, _ = describe(ss.sosfilt(sos, audio).astype(np.float32))
    assert bright.brightness > base.brightness + 0.05


def test_bass_measures_low_end_share(plain):
    """Removing the low end must drop the bass reading."""
    base, _, _ = plain
    audio, _ = make_track(bpm=128.0, bars=24)
    sos = ss.butter(4, 400.0, btype="high", fs=SR, output="sos")
    thin, _, _ = describe(ss.sosfilt(sos, audio).astype(np.float32))
    assert thin.bass < base.bass * 0.5


def test_loudness_follows_level(plain):
    base, _, _ = plain
    audio, _ = make_track(bpm=128.0, bars=24)
    quiet, _, _ = describe((audio * 0.1).astype(np.float32))
    assert quiet.loudness_db < base.loudness_db - 15.0
    assert quiet.loudness < base.loudness


def test_pulse_is_high_for_a_steady_beat_and_low_for_noise(plain):
    beat, _, _ = plain
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 20) * 0.2).astype(np.float32)
    vague, _, _ = describe(noise)
    assert beat.pulse > vague.pulse
    assert beat.pulse > 0.1


def test_rhythm_pattern_is_a_normalised_shape(plain):
    f, _, _ = plain
    assert len(f.rhythm) == RHYTHM_SLOTS
    assert f.rhythm.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(f.rhythm >= 0.0)


def test_kick_pattern_finds_four_to_the_floor(plain):
    """The low-band pattern must accent slots 0, 4, 8, 12 - the kick.

    The full-band pattern deliberately does not: spectral flux follows
    brightness, so on this material the off-beat hats take about three quarters
    of the weight. That is why the kick gets a band of its own rather than being
    inferred from the overall groove - two records whose kicks disagree would
    otherwise look compatible.
    """
    f, _, _ = plain
    on_beat = f.kick[[0, 4, 8, 12]].sum()
    assert on_beat > f.kick.sum() - on_beat
    assert on_beat > 0.6

    hats = f.rhythm[[2, 6, 10, 14]].sum()
    assert hats > f.rhythm[[0, 4, 8, 12]].sum(), "full band should be hat-led here"


def test_both_patterns_are_normalised_shapes(plain):
    f, _, _ = plain
    for pattern in (f.rhythm, f.kick):
        assert len(pattern) == RHYTHM_SLOTS
        assert pattern.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(pattern >= 0.0)


def test_rhythm_distance_bounds_and_identity():
    a = np.zeros(RHYTHM_SLOTS)
    a[[0, 4, 8, 12]] = 0.25
    b = np.zeros(RHYTHM_SLOTS)
    b[[2, 6, 10, 14]] = 0.25
    assert rhythm_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert rhythm_distance(a, b) > 0.4
    assert 0.0 <= rhythm_distance(a, b) <= 1.0


def test_rhythm_distance_is_neutral_when_there_is_nothing_to_compare():
    """Silence should not read as either a match or a clash."""
    assert rhythm_distance(np.zeros(RHYTHM_SLOTS), np.ones(RHYTHM_SLOTS) / RHYTHM_SLOTS) == 0.5


def test_rhythm_ignores_level(plain):
    """The same groove played quietly is still the same groove."""
    audio, _ = make_track(bpm=128.0, bars=24)
    loud, _, _ = describe(audio)
    quiet, _, _ = describe((audio * 0.25).astype(np.float32))
    assert rhythm_distance(loud.rhythm, quiet.rhythm) < 0.05


def test_rhythm_pattern_survives_an_empty_region(plain):
    _, ctx, grid = plain
    empty = rhythm_pattern(ctx.onset, ANALYSIS_SR, 1000.0, 1001.0, grid)
    assert len(empty) == RHYTHM_SLOTS
    assert np.all(np.isfinite(empty))


def test_edge_features_survive_a_tiny_region(plain):
    _, ctx, grid = plain
    from automix.analysis.energy import analyze_energy as ae

    profile = ae(ctx.y, ANALYSIS_SR, onset_env=ctx.onset)
    f = edge_features(ctx, 1.0, 1.0001, grid, profile)
    assert np.all(np.isfinite(f.rhythm))
    assert 0.0 <= f.bass <= 1.0


def test_edge_features_roundtrip(plain):
    from automix.analysis.features import EdgeFeatures

    f, _, _ = plain
    again = EdgeFeatures.from_dict(f.to_dict())
    assert again.loudness == pytest.approx(f.loudness, abs=1e-3)
    assert np.allclose(again.rhythm, f.rhythm, atol=1e-3)
    assert np.allclose(again.kick, f.kick, atol=1e-3)
    assert np.allclose(again.timbre, f.timbre, atol=1e-3)
