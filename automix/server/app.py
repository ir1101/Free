"""FastAPI backend for the AutoMix web app.

The server owns the slow, stateful half - decoding, analysis, sequencing - and
hands the browser a SetPlan. The browser owns playback. Audio files are streamed
as-is; browsers decode mp3, wav, flac and m4a natively, so there is no transcode
step and no reason to invent one.
"""

from __future__ import annotations

import mimetypes
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..analysis.track import CACHE_DIR, Track, analyze_file
from ..audio_io import scan
from ..model.cost import Weights, target_energy, transition_cost
from ..model.plan import DEFAULT_OVERLAP_BARS, build_plan
from ..model.sequencer import sequence

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


class _RevalidatingStatics(StaticFiles):
    """Static files that must be revalidated rather than served blind from cache.

    Without an explicit directive a browser is free to reuse a cached response
    without asking, and it will: edit the page, reload, and the old ES module is
    still running. ``no-cache`` does not disable caching - it requires a
    conditional request first, so unchanged files still come back as a 304 and
    cost nothing, while changed ones actually arrive.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Library:
    """In-memory view of the analysed collection.

    Analysis runs on a worker thread so the UI can show progress instead of
    freezing for the minute or two a first scan takes. The lock guards the
    dictionary; results are only ever added, never mutated in place.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.tracks: dict[str, Track] = {}
        self.folder: Path | None = None
        self.lock = threading.Lock()
        self.state = {"running": False, "done": 0, "total": 0,
                      "current": "", "errors": [], "finished": False}

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state, count=len(self.tracks))

    def ordered(self, ids: list[str] | None = None) -> list[Track]:
        with self.lock:
            if ids:
                return [self.tracks[i] for i in ids if i in self.tracks]
            return sorted(self.tracks.values(), key=lambda t: t.title.lower())

    def start_scan(self, folder: Path, use_cache: bool = True) -> dict:
        with self.lock:
            if self.state["running"]:
                return dict(self.state)
            paths = scan(folder)
            self.folder = folder
            self.state.update(running=True, done=0, total=len(paths),
                              current="", errors=[], finished=False)

        def work():
            for i, path in enumerate(paths):
                with self.lock:
                    self.state.update(done=i, current=path.name)
                try:
                    track = analyze_file(path, cache_dir=self.cache_dir, use_cache=use_cache)
                    with self.lock:
                        self.tracks[track.track_id] = track
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
                    with self.lock:
                        self.state["errors"].append(f"{path.name}: {exc}")
            with self.lock:
                self.state.update(running=False, done=len(paths), current="", finished=True)

        threading.Thread(target=work, daemon=True).start()
        return self.snapshot()


class ScanRequest(BaseModel):
    folder: str
    refresh: bool = False


class PlanRequest(BaseModel):
    track_ids: list[str] | None = None
    weights: dict | None = None
    arc: str = "peak"
    bars: int = DEFAULT_OVERLAP_BARS
    pitch_preserving: bool = False


def track_payload(t: Track) -> dict:
    return {
        "id": t.track_id, "title": t.title, "path": str(t.path),
        "duration": round(t.duration, 3), "bpm": round(t.bpm, 2),
        "key": t.key.code, "key_name": t.key.name,
        "key_confidence": round(t.key_confidence, 3),
        "grid_confidence": round(t.grid.confidence, 3),
        "energy": round(t.energy_mean, 3),
        "energy_in": round(t.energy_in, 3), "energy_out": round(t.energy_out, 3),
        "energy_curve": t.energy_curve,
        "mix_in": round(t.structure.mix_in, 3), "mix_out": round(t.structure.mix_out, 3),
        "boundaries": t.structure.boundaries,
    }


def create_app(folder: Path | None = None, cache_dir: Path = CACHE_DIR) -> FastAPI:
    app = FastAPI(title="AutoMix")
    library = Library(cache_dir=cache_dir)
    if folder:
        library.start_scan(Path(folder))

    @app.get("/api/state")
    def get_state():
        return {**library.snapshot(),
                "folder": str(library.folder) if library.folder else None}

    @app.post("/api/library/scan")
    def post_scan(req: ScanRequest):
        path = Path(req.folder).expanduser()
        if not path.exists():
            raise HTTPException(404, f"no such folder: {path}")
        return library.start_scan(path, use_cache=not req.refresh)

    @app.get("/api/tracks")
    def get_tracks():
        return {"tracks": [track_payload(t) for t in library.ordered()]}

    @app.get("/api/arc")
    def get_arc(shape: str = "peak", n: int = 64):
        """The target energy curve, so the UI can draw what the set is aiming at."""
        return {"shape": shape,
                "curve": [round(target_energy(i / max(1, n - 1), shape=shape), 4)
                          for i in range(n)]}

    @app.post("/api/set/plan")
    def post_plan(req: PlanRequest):
        tracks = library.ordered(req.track_ids)
        if len(tracks) < 2:
            raise HTTPException(400, "need at least two analysed tracks to build a set")

        weights = Weights.from_dict(req.weights)
        order = sequence(tracks, weights, arc_shape=req.arc,
                         pitch_preserving=req.pitch_preserving)
        plan = build_plan(tracks, order, overlap_bars=req.bars, arc_shape=req.arc)
        ordered = [tracks[i] for i in order.order]
        return {
            "plan": plan.to_dict(),
            "tracks": [track_payload(t) for t in ordered],
            "total_cost": round(order.total_cost, 3),
            "exact": order.exact,
        }

    @app.post("/api/set/compare")
    def post_compare(req: PlanRequest):
        """Every pairwise cost, for the compatibility matrix in the UI."""
        tracks = library.ordered(req.track_ids)
        weights = Weights.from_dict(req.weights)
        matrix = []
        for a in tracks:
            row = []
            for b in tracks:
                if a.track_id == b.track_id:
                    row.append(None)
                else:
                    row.append(transition_cost(
                        a, b, weights, position=None,
                        pitch_preserving=req.pitch_preserving).to_dict())
            matrix.append(row)
        return {"ids": [t.track_id for t in tracks], "matrix": matrix}

    @app.get("/api/audio/{track_id}")
    def get_audio(track_id: str, request: Request):
        with library.lock:
            track = library.tracks.get(track_id)
        if track is None:
            raise HTTPException(404, "unknown track")

        path = Path(track.path)
        if not path.exists():
            raise HTTPException(404, f"file has moved: {path}")

        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = request.headers.get("range")
        if not range_header:
            return FileResponse(path, media_type=media_type,
                                headers={"Accept-Ranges": "bytes"})

        # Range support: not needed by decodeAudioData, which fetches whole
        # files, but it is what lets a plain <audio> element seek.
        try:
            units, _, spec = range_header.partition("=")
            start_s, _, end_s = spec.partition("-")
            if units.strip() != "bytes":
                raise ValueError
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            raise HTTPException(400, "malformed Range header")

        start = max(0, start)
        end = min(end, size - 1)
        if start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)
        return Response(data, status_code=206, media_type=media_type, headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
        })

    @app.exception_handler(FileNotFoundError)
    def missing(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    if WEB_DIR.exists():
        app.mount("/", _RevalidatingStatics(directory=str(WEB_DIR), html=True), name="web")

    return app
