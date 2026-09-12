"""The streaming renderer and its limiter."""

import struct

import numpy as np
import pytest

from automix.analysis.track import analyze_signal
from automix.model.cost import Weights
from automix.model.plan import build_plan
from automix.model.sequencer import sequence
from automix.render.mixdown import render
from automix.render.stream import CEILING, Limiter, MixStream, wav_header
from automix.synth import SR, make_track

from .conftest import ANALYSIS_SR, to_analysis

SPECS = [(124.0, 9, False), (126.0, 4, False), (122.0, 11, True)]


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    from automix import audio_io

    root = tmp_path_factory.mktemp("stream")
    tracks = []
    for i, (bpm, tonic, is_major) in enumerate(SPECS):
        audio, _ = make_track(bpm=bpm, tonic=tonic, is_major=is_major, bars=32)
        path = root / f"t{i}.wav"
        audio_io.write(path, audio, SR)
        tracks.append(analyze_signal(to_analysis(audio), ANALYSIS_SR,
                                     path=path, title=f"t{i}"))
    order = sequence(tracks, Weights(), arc_shape="peak")
    return build_plan(tracks, order, sample_rate=22050)


def test_wav_header_declares_real_sizes():
    header = wav_header(1000, 44100)
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert struct.unpack("<I", header[40:44])[0] == 1000 * 4
    assert struct.unpack("<I", header[4:8])[0] == 36 + 1000 * 4
    channels, rate = struct.unpack("<HI", header[22:28])
    assert channels == 2 and rate == 44100


def test_stream_matches_the_offline_render_sample_for_sample(planned):
    """Streaming must be a scheduling change, not a different mix."""
    streamed = np.concatenate(list(MixStream(planned).frames(1.0)))
    offline = render(planned, normalise=False)
    n = min(len(streamed), len(offline))
    assert n > planned.sample_rate * 30
    assert np.max(np.abs(streamed[:n] - offline[:n])) < 1e-12


def test_chunk_size_does_not_change_the_output(planned):
    """Whole-segment rendering means boundaries leave no seam."""
    a = np.concatenate(list(MixStream(planned).frames(0.5)))
    b = np.concatenate(list(MixStream(planned).frames(3.0)))
    n = min(len(a), len(b))
    assert np.max(np.abs(a[:n] - b[:n])) < 1e-12


def test_stream_holds_only_a_few_segments_at_once(planned):
    """Memory has to stay flat however long the set is.

    Three, not two: for the moment where one crossfade ends and the next begins,
    the outgoing track's last samples, the track between them and the incoming
    one all fall inside the same chunk. What matters is that the number is a
    small constant rather than growing with the length of the set.
    """
    stream = MixStream(planned)
    worst = 0
    for _ in stream.frames(1.0):
        worst = max(worst, len(stream._live))
    assert worst <= 3
    # And everything is released by the end.
    assert len(stream._live) <= 1


def test_wav_output_is_the_promised_length(planned):
    chunks = list(MixStream(planned).wav(2.0))
    header, body = chunks[0], b"".join(chunks[1:])
    declared = struct.unpack("<I", header[40:44])[0]
    assert len(body) == declared


def test_stream_never_clips(planned):
    """A stream cannot peak-normalise, so the limiter has to guarantee this."""
    chunks = list(MixStream(planned).wav(2.0))
    body = b"".join(chunks[1:])
    samples = np.frombuffer(body, dtype="<i2").astype(np.float64) / 32767.0
    assert np.max(np.abs(samples)) <= 1.0
    assert np.max(np.abs(samples)) <= CEILING + 0.02


def test_limiter_pulls_peaks_under_the_ceiling():
    sr = 22050
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal((sr * 3, 2)) * 0.8)
    loud[sr: sr + 500] *= 3.0  # a burst well over full scale

    limiter = Limiter(sr)
    out = np.concatenate([limiter.process(loud[i:i + 4096])
                          for i in range(0, len(loud), 4096)]
                         + [limiter.process(np.zeros((0, 2)), final=True)])
    assert np.max(np.abs(out)) <= CEILING + 1e-6
    assert len(out) == len(loud)


def test_limiter_leaves_quiet_material_alone():
    """Nothing below the ceiling should be touched."""
    sr = 22050
    quiet = np.tile(np.linspace(-0.3, 0.3, sr)[:, None], (1, 2))
    limiter = Limiter(sr)
    out = np.concatenate([limiter.process(quiet), limiter.process(np.zeros((0, 2)), final=True)])
    assert np.allclose(out, quiet, atol=1e-9)


def test_limiter_is_seamless_across_chunk_sizes():
    """Chunking is an implementation detail; it must not colour the sound."""
    sr = 22050
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((sr * 2, 2)) * 1.4

    def run(size):
        limiter = Limiter(sr)
        parts = [limiter.process(signal[i:i + size]) for i in range(0, len(signal), size)]
        parts.append(limiter.process(np.zeros((0, 2)), final=True))
        return np.concatenate(parts)

    a, b = run(1024), run(7777)
    assert len(a) == len(b) == len(signal)
    assert np.max(np.abs(a - b)) < 1e-12
