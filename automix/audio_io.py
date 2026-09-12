"""Loading, decoding and caching audio.

Analysis and playback want different things from the same file. Analysis wants
mono at a low sample rate, because every feature extractor is happier and faster
that way. Rendering wants the original stereo at full rate, because that is what
ends up in your ears. So the two are loaded separately and never confused.

Decoding is the slowest step in the pipeline, so results are keyed by a fast
fingerprint of path, size and mtime rather than by hashing the audio itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ANALYSIS_SR = 22050
RENDER_SR = 44100

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}


@dataclass(slots=True)
class Audio:
    """Decoded audio plus where it came from."""

    samples: np.ndarray  # (n,) mono or (n, channels)
    sr: int
    path: Path | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    def mono(self) -> np.ndarray:
        return self.samples if self.samples.ndim == 1 else self.samples.mean(axis=1)


def fingerprint(path: Path) -> str:
    """Cheap identity for a file: path, size and mtime.

    Hashing the audio would be more principled but means reading every byte of
    every file on a rescan, which defeats the point of caching in the first
    place. Editing a file changes its mtime, so this catches real changes.

    Tracks analysed straight from a signal have no file behind them, so fall back
    to hashing the name alone. An identity that raises is worse than one that is
    merely weaker - callers only need it to be stable and unique.
    """
    try:
        st = path.stat()
        raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except (OSError, ValueError):
        raw = str(path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def scan(folder: Path) -> list[Path]:
    """All audio files under `folder`, sorted, recursing into subfolders."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"no such folder: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and is_audio(p))


def _read(path: Path, dtype: str = "float32"):
    """Decode a file, falling back to librosa's readers for exotic formats.

    ``soundfile`` bundles libsndfile 1.2+, which covers wav, flac, ogg and mp3 -
    so the common cases need no external tools. Anything it refuses (m4a, wma)
    goes through librosa, which will use ffmpeg if it is installed.
    """
    try:
        data, sr = sf.read(str(path), dtype=dtype, always_2d=False)
        return data, int(sr)
    except Exception:
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=False)
        if data.ndim == 2:
            data = data.T  # librosa gives (channels, n); match soundfile's layout
        return data.astype(np.float32), int(sr)


def load_for_analysis(path: Path, sr: int = ANALYSIS_SR) -> Audio:
    """Mono, downsampled - what every feature extractor here wants."""
    import librosa

    data, native_sr = _read(path)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    mono = np.asarray(mono, dtype=np.float64)
    if native_sr != sr:
        mono = librosa.resample(mono, orig_sr=native_sr, target_sr=sr)
    return Audio(mono, sr, Path(path))


def load_for_render(path: Path, sr: int = RENDER_SR) -> Audio:
    """Full-rate stereo, resampled to a common rate so tracks can be summed.

    Mono sources are widened to stereo so the mixdown has one consistent shape
    rather than branching on channel count at every step.
    """
    import librosa

    data, native_sr = _read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if native_sr != sr:
        data = librosa.resample(data.T, orig_sr=native_sr, target_sr=sr).T
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return Audio(data, sr, Path(path))


def peak_normalise(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at `headroom_db`, leaving clipping room."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-9:
        return audio
    return audio * (10.0 ** (headroom_db / 20.0) / peak)


def write(path: Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16" if path.suffix.lower() == ".wav" else None
    sf.write(str(path), audio, sr, subtype=subtype)
    return path
