"""The server: preparing a set, noticing folder changes, and serving the stream.

The folder-change tests exist because this was originally missed: the set was
built once at startup and nothing ever looked at the folder again, so tracks
added later were silently invisible.
"""

import struct
import time

import pytest
from fastapi.testclient import TestClient

from automix import audio_io
from automix.server.app import create_app
from automix.synth import SR, make_track


def write_track(folder, name, bpm, tonic, is_major=False, bars=12):
    audio, _ = make_track(bpm=bpm, tonic=tonic, is_major=is_major, bars=bars)
    return audio_io.write(folder / f"{name}.wav", audio, SR)


def wait_ready(client, timeout=180):
    """Preparation runs on a worker thread; give it time to land."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/api/status").json()
        if body["error"]:
            pytest.fail(f"preparation failed: {body['error']}")
        if body["ready"] and not body["busy"]:
            return body
        time.sleep(0.4)
    pytest.fail("set was never ready")


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    folder = tmp_path_factory.mktemp("library")
    write_track(folder, "one", 124.0, 9)
    write_track(folder, "two", 126.0, 4)
    return folder


@pytest.fixture(scope="module")
def client(library, tmp_path_factory):
    app = create_app(library, cache_dir=tmp_path_factory.mktemp("cache"))
    with TestClient(app) as c:
        wait_ready(c)
        yield c


def test_status_reports_a_ready_set(client):
    body = client.get("/api/status").json()
    assert body["ready"] and not body["busy"]
    assert body["tracks"] == 2
    assert body["stale"] is False
    assert body["duration"] != "0:00"


def test_plan_lists_the_running_order(client):
    plan = client.get("/api/plan").json()
    assert plan["ready"] and len(plan["tracks"]) == 2
    assert len(plan["transitions"]) == 1
    first, second = plan["tracks"]
    assert first["cue"] == 0.0
    assert second["cue"] > first["cue"]
    for track in plan["tracks"]:
        assert track["bpm"] > 0 and track["key"]


def test_now_playing_follows_the_clock(client):
    plan = client.get("/api/plan").json()
    start = client.get("/api/now", params={"t": 0}).json()
    assert start["title"] == plan["tracks"][0]["title"]

    during = client.get("/api/now", params={"t": plan["transitions"][0]["start"] + 1}).json()
    assert "mixing into" in during["detail"]

    end = client.get("/api/now", params={"t": plan["duration"] - 1}).json()
    assert end["title"] == plan["tracks"][-1]["title"]


def test_stream_is_a_wav_of_the_declared_length(client):
    with client.stream("GET", "/stream") as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("audio/wav")
        declared = int(res.headers["content-length"])
        head = next(res.iter_bytes())
        assert head[:4] == b"RIFF" and head[8:12] == b"WAVE"
        assert struct.unpack("<I", head[40:44])[0] == declared - 44


def test_starting_part_way_in_returns_a_shorter_stream(client):
    """Skipping tracks relies on this: a start offset is a whole new stream."""
    plan = client.get("/api/plan").json()
    with client.stream("GET", "/stream") as whole:
        full = int(whole.headers["content-length"])
    with client.stream("GET", "/stream", params={"start": plan["duration"] / 2}) as part:
        half = int(part.headers["content-length"])
    assert 0 < half < full
    # Four bytes per frame, so the shortfall should match the time skipped.
    skipped = (full - half) / 4 / 44100
    assert skipped == pytest.approx(plan["duration"] / 2, rel=0.02)


def test_a_start_beyond_the_end_is_clamped(client):
    with client.stream("GET", "/stream", params={"start": 1e9}) as res:
        assert res.status_code == 200
        assert int(res.headers["content-length"]) > 44


def test_adding_a_file_marks_the_set_stale(client, library):
    """The bug this suite exists for: new files used to be invisible for ever."""
    assert client.get("/api/status").json()["stale"] is False

    write_track(library, "three", 122.0, 11, is_major=True)
    # Staleness is rechecked on a short interval rather than every call.
    deadline = time.time() + 6
    while time.time() < deadline:
        body = client.get("/api/status").json()
        if body["stale"]:
            break
        time.sleep(0.5)
    assert body["stale"] is True
    assert body["found"] == 3
    assert body["tracks"] == 2, "the built set should not change until it is rebuilt"


def test_rescan_rebuilds_with_the_new_file(client, library):
    client.post("/api/rescan")
    body = wait_ready(client)
    assert body["tracks"] == 3
    assert body["stale"] is False
    assert len(client.get("/api/plan").json()["tracks"]) == 3


def test_removing_a_file_is_noticed_too(client, library):
    (library / "three.wav").unlink()
    deadline = time.time() + 6
    while time.time() < deadline:
        body = client.get("/api/status").json()
        if body["stale"]:
            break
        time.sleep(0.5)
    assert body["stale"] is True

    client.post("/api/rescan")
    assert wait_ready(client)["tracks"] == 2


def test_an_empty_folder_reports_an_error_rather_than_hanging(tmp_path):
    app = create_app(tmp_path / "nothing-here", cache_dir=tmp_path / "cache")
    with TestClient(app) as c:
        deadline = time.time() + 20
        while time.time() < deadline:
            body = c.get("/api/status").json()
            if body["error"]:
                break
            time.sleep(0.3)
        assert body["error"], "a missing folder must surface as an error"
        assert body["busy"] is False, "a failed run must not leave the server busy"
        assert c.get("/stream").status_code == 409
