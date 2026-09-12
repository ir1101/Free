"""The Track record: everything the mixer needs to know about one song.

Analysis is the slow part of the pipeline - a few seconds per track - so results
are cached to disk keyed by a fingerprint of the file. A rescan of an already
analysed library is then effectively instant, which is what makes iterating on
the sequencer weights bearable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import audio_io
from ..model.camelot import Key, parse as parse_key
from .beats import BeatGrid, analyze_beats
from .energy import analyze_energy, timbre_vector
from .key import detect_key
from .structure import Structure, analyze_structure

CACHE_VERSION = 3
CACHE_DIR = Path(".automix/cache")

# How much of a track counts as its "edges" for compatibility purposes. What
# matters when joining two tracks is how one *ends* and how the other *starts*,
# not what either averages to across four minutes.
EDGE_BARS = 16


@dataclass(slots=True)
class Track:
    """One analysed song."""

    path: Path
    title: str
    duration: float
    grid: BeatGrid
    key: Key
    key_confidence: float
    structure: Structure

    # Edge descriptors: measured on the outgoing tail and incoming head.
    energy_in: float
    energy_out: float
    energy_mean: float
    timbre_in: np.ndarray
    timbre_out: np.ndarray
    energy_curve: list[float] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        return self.grid.bpm

    @property
    def track_id(self) -> str:
        return audio_io.fingerprint(self.path)

    def to_dict(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "path": str(self.path),
            "title": self.title,
            "duration": round(self.duration, 4),
            "grid": self.grid.to_dict(),
            "key": self.key.code,
            "key_confidence": round(self.key_confidence, 4),
            "structure": self.structure.to_dict(),
            "energy_in": round(self.energy_in, 4),
            "energy_out": round(self.energy_out, 4),
            "energy_mean": round(self.energy_mean, 4),
            "timbre_in": [round(float(v), 4) for v in self.timbre_in],
            "timbre_out": [round(float(v), 4) for v in self.timbre_out],
            "energy_curve": self.energy_curve,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        return cls(
            path=Path(d["path"]),
            title=d["title"],
            duration=d["duration"],
            grid=BeatGrid.from_dict(d["grid"]),
            key=parse_key(d["key"]),
            key_confidence=d["key_confidence"],
            structure=Structure.from_dict(d["structure"]),
            energy_in=d["energy_in"],
            energy_out=d["energy_out"],
            energy_mean=d["energy_mean"],
            timbre_in=np.array(d["timbre_in"]),
            timbre_out=np.array(d["timbre_out"]),
            energy_curve=d.get("energy_curve", []),
        )

    def summary(self) -> str:
        return (f"{self.title[:38]:38s} {self.bpm:6.1f} BPM  {self.key.code:>3s}  "
                f"energy {self.energy_mean:.2f}  "
                f"mix {self.structure.mix_in:6.1f}s -> {self.structure.mix_out:6.1f}s")


def _cache_path(path: Path, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{audio_io.fingerprint(path)}.json"


def analyze_signal(y: np.ndarray, sr: int, path: Path | None = None,
                   title: str | None = None) -> Track:
    """Run the full analysis chain over an already-decoded mono signal."""
    grid, onset_env = analyze_beats(y, sr)
    profile = analyze_energy(y, sr, onset_env=onset_env)
    structure = analyze_structure(y, sr, grid, profile)
    key, key_conf = detect_key(y, sr)

    duration = len(y) / sr
    edge = EDGE_BARS * grid.seconds_per_bar

    # The head starts at the mix-in cue, not at t=0: the first seconds are often
    # an intro that says nothing about how the track sounds once it is running.
    head_start = structure.mix_in
    head_end = min(duration, head_start + edge)
    tail_end = structure.mix_out
    tail_start = max(0.0, tail_end - edge)

    return Track(
        path=Path(path) if path else Path("<memory>"),
        title=title or (Path(path).stem if path else "untitled"),
        duration=duration,
        grid=grid,
        key=key,
        key_confidence=key_conf,
        structure=structure,
        energy_in=profile.between(head_start, head_end),
        energy_out=profile.between(tail_start, tail_end),
        energy_mean=profile.between(structure.mix_in, structure.mix_out),
        timbre_in=timbre_vector(y, sr, head_start, head_end),
        timbre_out=timbre_vector(y, sr, tail_start, tail_end),
        energy_curve=profile.curve(),
    )


def analyze_file(path: Path, cache_dir: Path = CACHE_DIR, use_cache: bool = True) -> Track:
    """Analyse one file, reading from and writing to the on-disk cache."""
    path = Path(path)
    cache_file = _cache_path(path, cache_dir)

    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("version") == CACHE_VERSION:
                # The cache is keyed by the old path; trust the current one so a
                # moved library still resolves.
                data["path"] = str(path)
                return Track.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt or stale entry; just recompute

    audio = audio_io.load_for_analysis(path)
    track = analyze_signal(audio.mono(), audio.sr, path=path, title=path.stem)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(track.to_dict(), indent=1), encoding="utf-8")
    return track


def analyze_folder(folder: Path, cache_dir: Path = CACHE_DIR, use_cache: bool = True,
                   progress=None) -> list[Track]:
    """Analyse every audio file under `folder`.

    A file that fails to decode is reported and skipped rather than aborting the
    run - one bad download should not cost you the whole library.
    """
    paths = audio_io.scan(Path(folder))
    tracks: list[Track] = []
    for i, path in enumerate(paths):
        if progress:
            progress(i, len(paths), path)
        try:
            tracks.append(analyze_file(path, cache_dir=cache_dir, use_cache=use_cache))
        except Exception as exc:  # noqa: BLE001 - report and carry on
            if progress:
                progress(i, len(paths), path, error=exc)
    return tracks
