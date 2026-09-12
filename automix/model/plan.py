"""The SetPlan: one description of a mix, rendered by two different engines.

This is the spine of the system. Sequencing produces a plan, and both the browser
engine and the offline renderer consume that same plan. It carries absolute times
and explicit automation curves rather than parameters each side re-derives,
because two engines re-deriving "the same" timing from the same inputs is exactly
how they end up a few milliseconds apart.

**How tempo is handled.** Each track plays at its own natural tempo for its whole
body. Only the *incoming* track is bent: it is rate-matched to the outgoing track
for the length of the overlap, so the two are beat-locked while they are both
audible, and then eased back to its own tempo over the following few bars, once
it is alone and nothing can drift against it.

That choice matters. The obvious alternative - lock everything to the first
track - compounds: by the fifth track you are asking for a 30% stretch and it
sounds like a tape machine dying. Bending only the incoming track keeps every
stretch as small as the two neighbouring tempos require and no larger, and it
leaves each track's outro, the part you are actually listening to during a
transition, completely untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.track import Track
from ..render.crossfade import CURVE_POINTS, Curve, bass_swap, equal_power, highpass_sweep
from .cost import TransitionCost
from .sequencer import SetOrder

PLAN_VERSION = 1

# Length of the overlap. 16 bars is about 30 seconds at house tempo - long enough
# to feel like a blend rather than a cut, short enough to hold interest.
DEFAULT_OVERLAP_BARS = 16

# How long the incoming track takes to ease back to its own tempo once it is
# alone. Spread over several bars the change is felt rather than heard.
TEMPO_RESET_BARS = 8


@dataclass(slots=True)
class Segment:
    """One track's placement on the set timeline."""

    index: int
    track_id: str
    title: str
    path: str

    start_time: float      # set time when this track starts sounding
    end_time: float        # set time when it stops
    source_start: float    # where in the source file playback begins
    playback_rate: float   # rate during the overlap it enters on
    rate_ramp_start: float  # set time the ease back to native tempo begins
    rate_ramp_end: float   # set time it finishes; rate is 1.0 from here on

    bpm: float
    key: str

    def to_dict(self) -> dict:
        return {
            "index": self.index, "track_id": self.track_id, "title": self.title,
            "path": self.path,
            "start_time": round(self.start_time, 6), "end_time": round(self.end_time, 6),
            "source_start": round(self.source_start, 6),
            "playback_rate": round(self.playback_rate, 8),
            "rate_ramp_start": round(self.rate_ramp_start, 6),
            "rate_ramp_end": round(self.rate_ramp_end, 6),
            "bpm": round(self.bpm, 3), "key": self.key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**d)

    def source_at(self, t: float) -> float:
        """Where in the source file we are at set time `t`.

        Position is the integral of playback rate, which is also how Web Audio
        advances a source node - so this and the browser agree by construction
        rather than by coincidence.
        """
        if t <= self.start_time:
            return self.source_start
        pos = self.source_start
        ramp_start = max(self.start_time, self.rate_ramp_start)
        flat = min(t, ramp_start) - self.start_time
        pos += flat * self.playback_rate
        if t > ramp_start:
            span = max(1e-9, self.rate_ramp_end - ramp_start)
            elapsed = min(t, self.rate_ramp_end) - ramp_start
            # Linear rate ramp: the distance covered is the average rate times time.
            rate_at = self.playback_rate + (1.0 - self.playback_rate) * (elapsed / span)
            pos += elapsed * (self.playback_rate + rate_at) / 2.0
            if t > self.rate_ramp_end:
                pos += t - self.rate_ramp_end
        return pos


@dataclass(slots=True)
class Transition:
    """One crossfade, with every automation curve spelled out."""

    from_index: int
    to_index: int
    start_time: float
    duration: float
    bars: int

    gain_out: Curve
    gain_in: Curve
    bass_out: Curve
    bass_in: Curve
    filter_out: Curve

    cost: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def to_dict(self) -> dict:
        return {
            "from_index": self.from_index, "to_index": self.to_index,
            "start_time": round(self.start_time, 6),
            "duration": round(self.duration, 6), "bars": self.bars,
            "gain_out": self.gain_out.to_dict(), "gain_in": self.gain_in.to_dict(),
            "bass_out": self.bass_out.to_dict(), "bass_in": self.bass_in.to_dict(),
            "filter_out": self.filter_out.to_dict(),
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transition":
        return cls(
            from_index=d["from_index"], to_index=d["to_index"],
            start_time=d["start_time"], duration=d["duration"], bars=d["bars"],
            gain_out=Curve.from_dict(d["gain_out"]), gain_in=Curve.from_dict(d["gain_in"]),
            bass_out=Curve.from_dict(d["bass_out"]), bass_in=Curve.from_dict(d["bass_in"]),
            filter_out=Curve.from_dict(d["filter_out"]), cost=d.get("cost", {}),
        )


@dataclass(slots=True)
class SetPlan:
    """A complete mix: what plays, when, at what rate, through which curves."""

    segments: list[Segment]
    transitions: list[Transition]
    duration: float
    sample_rate: int = 44100
    total_cost: float = 0.0
    exact: bool = True
    arc_shape: str = "peak"

    def to_dict(self) -> dict:
        return {
            "version": PLAN_VERSION,
            "sample_rate": self.sample_rate,
            "duration": round(self.duration, 6),
            "total_cost": round(self.total_cost, 4),
            "exact": self.exact,
            "arc_shape": self.arc_shape,
            "segments": [s.to_dict() for s in self.segments],
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SetPlan":
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            transitions=[Transition.from_dict(t) for t in d["transitions"]],
            duration=d["duration"], sample_rate=d.get("sample_rate", 44100),
            total_cost=d.get("total_cost", 0.0), exact=d.get("exact", True),
            arc_shape=d.get("arc_shape", "peak"),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SetPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def cue_sheet(self) -> str:
        """Human-readable running order with timings."""
        lines = []
        for s in self.segments:
            mins, secs = divmod(int(s.start_time), 60)
            rate = "" if abs(s.playback_rate - 1.0) < 1e-4 else \
                f"  ({(s.playback_rate - 1.0) * 100:+.1f}% to enter)"
            lines.append(f"{mins:02d}:{secs:02d}  {s.title[:44]:44s} "
                         f"{s.bpm:6.1f} BPM  {s.key:>3s}{rate}")
        return "\n".join(lines)


def _curves(start: float, duration: float, sweep: bool):
    """Build the automation for one crossfade."""
    out_gain, in_gain = equal_power(CURVE_POINTS)
    out_bass, in_bass = bass_swap(CURVE_POINTS)
    cutoff = highpass_sweep(CURVE_POINTS) if sweep else np.full(CURVE_POINTS, 20.0)
    mk = lambda v: Curve(start, duration, [float(x) for x in v])  # noqa: E731
    return mk(out_gain), mk(in_gain), mk(out_bass), mk(in_bass), mk(cutoff)


def build_plan(tracks: list[Track], order: SetOrder, sample_rate: int = 44100,
               overlap_bars: int = DEFAULT_OVERLAP_BARS,
               reset_bars: int = TEMPO_RESET_BARS,
               filter_sweep: bool = False, arc_shape: str = "peak") -> SetPlan:
    """Lay an ordered set out on a timeline.

    Each track runs from its mix-in cue to its mix-out cue; the next track starts
    coming in when that mix-out is reached. Overlap length is measured in bars at
    the outgoing track's tempo, so a transition is the same musical length
    whatever the record.
    """
    if not order.order:
        return SetPlan([], [], 0.0, sample_rate)

    ordered = [tracks[i] for i in order.order]
    segments: list[Segment] = []
    transitions: list[Transition] = []

    # The first track plays from its beginning: an intro is a fine way to open a
    # set, even though it is a bad place to mix into.
    first = ordered[0]
    segments.append(Segment(
        index=0, track_id=first.track_id, title=first.title, path=str(first.path),
        start_time=0.0, end_time=first.duration, source_start=0.0,
        playback_rate=1.0, rate_ramp_start=0.0, rate_ramp_end=0.0,
        bpm=first.bpm, key=first.key.code,
    ))

    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        prev_seg = segments[-1]

        bar_seconds = prev.grid.seconds_per_bar  # prev plays at native tempo here

        # The crossfade *ends* at the outgoing track's mix-out cue rather than
        # starting there. That cue marks where the arrangement thins out into an
        # outro, so a fade beginning on it spends its first half with the
        # outgoing track already stripped back and the incoming one still turned
        # down - an audible hole in the middle of what is supposed to be the
        # seamless part. Ending on it keeps both tracks in their full sections
        # for the whole overlap, and hands over exactly as the old one drops away.
        body = max(0.0, prev.structure.mix_out - prev.structure.mix_in)
        # Never spend more than half the usable body on one fade, and keep the
        # overlap a whole number of bars so it stays musical.
        usable = min(overlap_bars * bar_seconds, body * 0.5)
        bars_used = max(1, int(round(usable / bar_seconds)))
        overlap = bars_used * bar_seconds

        # Anchor the fade to its *end* and work backwards. Deriving the start
        # from a source position instead would drift whenever the outgoing track
        # is still easing back to its own tempo during the fade - which happens
        # on short tracks, where the tempo reset runs into the next transition.
        # Then a fixed span of set time no longer covers the same span of source
        # time, and the fade stops landing on the cue. Anchoring the end makes
        # the guarantee hold by construction, at any rate schedule.
        fade_end = prev_seg.start_time + _set_time_to_source(prev_seg, prev.structure.mix_out)
        handover = fade_end - overlap

        # Never start fading out before this track has finished fading in.
        earliest = max(prev_seg.start_time, prev_seg.rate_ramp_start)
        if handover < earliest:
            handover = min(earliest, max(prev_seg.start_time, fade_end - bar_seconds))
            overlap = max(bar_seconds, fade_end - handover)
            bars_used = max(1, int(round(overlap / bar_seconds)))

        rate = prev.bpm / cur.bpm if cur.bpm > 0 else 1.0
        reset = reset_bars * bar_seconds

        prev_seg.end_time = handover + overlap

        ramp_start = handover + overlap
        ramp_end = ramp_start + reset
        seg = Segment(
            index=i, track_id=cur.track_id, title=cur.title, path=str(cur.path),
            start_time=handover, end_time=handover, source_start=cur.structure.mix_in,
            playback_rate=rate, rate_ramp_start=ramp_start, rate_ramp_end=ramp_end,
            bpm=cur.bpm, key=cur.key.code,
        )
        # Provisional end; corrected on the next iteration or after the loop.
        seg.end_time = handover + (cur.duration - cur.structure.mix_in)
        segments.append(seg)

        g_out, g_in, b_out, b_in, filt = _curves(handover, overlap, filter_sweep)
        cost = order.transitions[i - 1] if i - 1 < len(order.transitions) else None
        transitions.append(Transition(
            from_index=i - 1, to_index=i, start_time=handover, duration=overlap,
            bars=bars_used, gain_out=g_out, gain_in=g_in,
            bass_out=b_out, bass_in=b_in, filter_out=filt,
            cost=cost.to_dict() if isinstance(cost, TransitionCost) else {},
        ))


    # The last track runs to the end of the file.
    last_seg, last_track = segments[-1], ordered[-1]
    last_seg.end_time = last_seg.start_time + _source_to_set_time(
        last_seg, last_track.duration
    )

    duration = max(s.end_time for s in segments)
    return SetPlan(segments, transitions, duration, sample_rate,
                   total_cost=order.total_cost, exact=order.exact, arc_shape=arc_shape)


def _set_time_to_source(seg: Segment, source_target: float) -> float:
    """How long after `seg` starts until its source reaches `source_target`.

    Inverts the rate schedule. Needed because the timeline is built forwards in
    set time but cue points are defined in source time.
    """
    covered = source_target - seg.source_start
    if covered <= 0:
        return 0.0

    flat_span = max(0.0, seg.rate_ramp_start - seg.start_time)
    flat_distance = flat_span * seg.playback_rate
    if covered <= flat_distance or seg.playback_rate <= 0:
        return covered / max(1e-9, seg.playback_rate)

    covered -= flat_distance
    ramp_span = max(0.0, seg.rate_ramp_end - seg.rate_ramp_start)
    ramp_distance = ramp_span * (seg.playback_rate + 1.0) / 2.0
    if covered <= ramp_distance:
        # Constant acceleration: solve the quadratic for elapsed ramp time.
        r0, r1 = seg.playback_rate, 1.0
        a = (r1 - r0) / (2.0 * max(1e-9, ramp_span))
        if abs(a) < 1e-12:
            return flat_span + covered / max(1e-9, r0)
        disc = max(0.0, r0 * r0 + 4.0 * a * covered)
        return flat_span + (-r0 + disc ** 0.5) / (2.0 * a)

    covered -= ramp_distance
    return flat_span + ramp_span + covered


def _source_to_set_time(seg: Segment, source_target: float) -> float:
    return _set_time_to_source(seg, source_target)
