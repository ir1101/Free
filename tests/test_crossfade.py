"""The crossfade's smoothness guarantees, asserted directly."""

import numpy as np
import pytest
from scipy import signal as ss

from automix.render.crossfade import (
    Curve, apply_bass_gain, bass_swap, equal_power, highpass_sweep, split_bands,
)


def test_equal_power_holds_constant_power():
    """The whole point of a cos/sin fade: a^2 + b^2 stays at 1 throughout."""
    out, inc = equal_power(512)
    assert np.allclose(out**2 + inc**2, 1.0, atol=1e-12)


def test_equal_power_beats_a_linear_fade_at_the_midpoint():
    """A linear crossfade sags ~3 dB in the middle; this is why we don't use one."""
    out, inc = equal_power(513)
    mid = 513 // 2
    equal_power_db = 20 * np.log10(np.hypot(out[mid], inc[mid]))
    linear = np.linspace(1, 0, 513), np.linspace(0, 1, 513)
    linear_db = 20 * np.log10(np.hypot(linear[0][mid], linear[1][mid]))
    assert abs(equal_power_db) < 0.01
    assert linear_db < -2.9


def test_equal_power_endpoints():
    out, inc = equal_power(64)
    assert out[0] == pytest.approx(1.0) and out[-1] == pytest.approx(0.0, abs=1e-12)
    assert inc[0] == pytest.approx(0.0, abs=1e-12) and inc[-1] == pytest.approx(1.0)


def test_bass_swap_hands_the_low_end_over_once():
    """Incoming starts cut and recovers by the midpoint; outgoing is the mirror."""
    out, inc = bass_swap(512)
    mid = 256
    assert inc[0] < 0.1 and inc[mid] == pytest.approx(1.0, abs=1e-3) and inc[-1] == pytest.approx(1.0, abs=1e-3)
    assert out[0] == pytest.approx(1.0, abs=1e-3) and out[-1] < 0.1
    assert np.all(np.diff(inc) >= -1e-9), "incoming bass must never dip back down"
    assert np.all(np.diff(out) <= 1e-9), "outgoing bass must never come back up"


def test_split_bands_reconstructs_exactly():
    """High band is derived by subtraction, so the split is lossless."""
    sr = 44100
    x = np.random.default_rng(0).standard_normal(sr) * 0.1
    low, high = split_bands(x, sr)
    assert np.allclose(low + high, x, atol=1e-12)


def test_bass_gain_at_unity_is_a_no_op():
    sr = 44100
    x = np.random.default_rng(1).standard_normal(sr) * 0.1
    assert np.allclose(apply_bass_gain(x, sr, np.ones(sr)), x, atol=1e-12)


def test_bass_gain_removes_lows_and_leaves_highs():
    sr = 44100
    x = np.random.default_rng(2).standard_normal(sr) * 0.1
    y = apply_bass_gain(x, sr, np.zeros(sr))
    f, p0 = ss.welch(x, sr, nperseg=4096)
    _, p1 = ss.welch(y, sr, nperseg=4096)
    low, high = (f > 40) & (f < 120), (f > 2000) & (f < 8000)
    assert 10 * np.log10(p1[low].mean() / p0[low].mean()) < -20
    assert abs(10 * np.log10(p1[high].mean() / p0[high].mean())) < 0.5


def test_bass_gain_works_on_stereo():
    sr = 22050
    x = np.random.default_rng(3).standard_normal((sr, 2)) * 0.1
    assert apply_bass_gain(x, sr, np.ones(sr)).shape == x.shape


def test_highpass_sweep_is_flat_then_rises():
    c = highpass_sweep(128, start_hz=20.0, end_hz=400.0, begin_at=0.75)
    assert c[0] == pytest.approx(20.0)
    assert c[:90].max() == pytest.approx(20.0, rel=1e-6)
    assert c[-1] == pytest.approx(400.0)
    assert np.all(np.diff(c) >= -1e-9)


def test_curve_roundtrips_and_interpolates_linearly():
    c = Curve(start=1.0, duration=2.0, values=[0.0, 1.0])
    assert np.allclose(c.sample(3), [0.0, 0.5, 1.0])
    again = Curve.from_dict(c.to_dict())
    assert again.start == c.start and again.duration == c.duration
    assert np.allclose(again.values, c.values)
