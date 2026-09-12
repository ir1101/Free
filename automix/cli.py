"""Command line interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import audio_io
from .analysis.track import CACHE_DIR, analyze_folder
from .model.cost import COMPONENTS, Weights
from .model.plan import DEFAULT_OVERLAP_BARS, build_plan
from .model.sequencer import sequence


def _progress(i, n, path, error=None):
    if error:
        print(f"  [{i + 1}/{n}] SKIPPED {Path(path).name}: {error}", file=sys.stderr)
    else:
        print(f"  [{i + 1}/{n}] {Path(path).name}", file=sys.stderr)


def _weights(args) -> Weights:
    return Weights(**{c: getattr(args, f"w_{c}") for c in COMPONENTS
                      if getattr(args, f"w_{c}", None) is not None})


def _load(args):
    started = time.time()
    tracks = analyze_folder(Path(args.folder), cache_dir=Path(args.cache),
                            use_cache=not args.no_cache, progress=_progress)
    if not tracks:
        raise SystemExit(f"no audio files found under {args.folder}")
    print(f"analysed {len(tracks)} tracks in {time.time() - started:.1f}s\n", file=sys.stderr)
    return tracks


def cmd_analyze(args) -> int:
    tracks = _load(args)
    print(f"{'title':38s} {'bpm':>6s}  {'key':>3s}  {'energy':>6s}  cue in/out")
    print("-" * 92)
    for t in sorted(tracks, key=lambda x: x.bpm):
        print(t.summary())
    return 0


def cmd_plan(args) -> int:
    tracks = _load(args)
    order = sequence(tracks, _weights(args), arc_shape=args.arc)
    plan = build_plan(tracks, order, overlap_bars=args.bars, arc_shape=args.arc)

    print(plan.cue_sheet())
    print(f"\n{len(plan.transitions)} transitions, "
          f"{plan.duration / 60:.1f} min, total cost {order.total_cost:.2f}"
          f"{'' if order.exact else ' (heuristic order - too many tracks for exact search)'}\n")

    print(f"{'#':>2s}  {'at':>7s}  {'key move':22s} {'tempo':>8s}  {'cost':>6s}  notes")
    print("-" * 92)
    for i, tr in enumerate(plan.transitions):
        c = tr.cost
        mins, secs = divmod(int(tr.start_time), 60)
        print("%2d  %3d:%02d  %-22s %+7.2f%%  %6.2f  %s"
              % (i, mins, secs, c.get("harmonic_label", ""), c.get("stretch_percent", 0.0),
                 c.get("total", 0.0), "; ".join(c.get("notes", []))))

    if args.out:
        print("\nwrote", plan.save(Path(args.out)), file=sys.stderr)
    return 0


def cmd_render(args) -> int:
    from .render.mixdown import render

    tracks = _load(args)
    order = sequence(tracks, _weights(args), arc_shape=args.arc)
    plan = build_plan(tracks, order, overlap_bars=args.bars, arc_shape=args.arc)
    print(plan.cue_sheet(), file=sys.stderr)

    started = time.time()
    mix = render(plan, filter_sweep=args.filter_sweep,
                 progress=lambda i, n, title: print(f"  [{i + 1}/{n}] {title}", file=sys.stderr))
    out = audio_io.write(Path(args.out), mix, plan.sample_rate)
    print(f"\nrendered {len(mix) / plan.sample_rate / 60:.1f} min in "
          f"{time.time() - started:.1f}s -> {out}", file=sys.stderr)

    plan.save(Path(args.out).with_suffix(".json"))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server.app import create_app

    app = create_app(Path(args.folder) if args.folder else None, cache_dir=Path(args.cache))
    print(f"\n  AutoMix running at http://{args.host}:{args.port}\n", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="automix",
        description="Analyse, sequence and beat-match local music into a continuous mix.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, needs_folder=True):
        p.add_argument("folder", nargs="?" if not needs_folder else None,
                       help="folder of audio files")
        p.add_argument("--cache", default=str(CACHE_DIR), help="analysis cache directory")
        p.add_argument("--no-cache", action="store_true", help="re-analyse everything")

    def planning(p):
        p.add_argument("--arc", choices=["peak", "flat"], default="peak",
                       help="peak: warm up, build, come down. flat: pure smoothness")
        p.add_argument("--bars", type=int, default=DEFAULT_OVERLAP_BARS,
                       help="crossfade length in bars")
        for c in COMPONENTS:
            p.add_argument(f"--w-{c}", type=float, default=None,
                           dest=f"w_{c}", help=f"weight for the {c} component")

    p = sub.add_parser("analyze", help="analyse a folder and print what was found")
    common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("plan", help="order a set and show every transition")
    common(p)
    planning(p)
    p.add_argument("--out", help="write the SetPlan JSON here")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("render", help="render the mix to an audio file")
    common(p)
    planning(p)
    p.add_argument("--out", default="set.wav", help="output audio file")
    p.add_argument("--filter-sweep", action="store_true",
                   help="sweep a high-pass over the outgoing track")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("serve", help="run the web app")
    common(p, needs_folder=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, SystemExit) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
