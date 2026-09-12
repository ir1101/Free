import warnings

import librosa
import numpy as np
import pytest

from automix.synth import SR, make_track

warnings.filterwarnings("ignore")

ANALYSIS_SR = 22050


def to_analysis(audio):
    return librosa.resample(np.asarray(audio, dtype=np.float64), orig_sr=SR, target_sr=ANALYSIS_SR)


@pytest.fixture(scope="session")
def synth_track():
    """A 128 BPM A-minor track with a known arrangement."""
    audio, truth = make_track(bpm=128.0, tonic=9, is_major=False, bars=32)
    return to_analysis(audio), truth
