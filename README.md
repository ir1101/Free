# AutoMix

Analyses a folder of music, works out the order that flows best, and beat-matches
the tracks into one continuous mix — in the browser, live, or rendered to a file.

"Smooth" here is not a fade. It is four things happening at once:

| | |
|---|---|
| **Tempos matched** | the incoming track is rate-matched to the outgoing one for the length of the overlap |
| **Downbeats aligned** | everything is cut and joined on bars, never on arbitrary seconds |
| **Keys compatible** | ordering uses the Camelot wheel, so consecutive tracks are harmonically related |
| **Loudness held** | an equal-power crossfade, so the middle of the blend doesn't sag |

---

## Quick start

```bash
pip install -e .
```

```bash
automix serve
```

Then open <http://127.0.0.1:8420>, paste in a folder of audio files, and press
Analyse. When it finishes, press **Build the set** — then **transition ▶** to jump
straight to a join and hear whether it works.

Or from the command line:

```bash
automix analyze "C:/Users/you/Music"
```

```bash
automix render "C:/Users/you/Music" --out set.wav
```

First analysis costs a few seconds per track. Results are cached under
`.automix/cache`, so everything after the first pass is instant.

---

## How it works

### 1. Analysis — `automix/analysis/`

**Beat grid.** `librosa`'s beat tracker gives a rough tempo, but its tempogram is
coarse — it will call 128 BPM "129.2", and a 1% tempo error accumulates into a
whole beat of drift across a 32-bar transition. So the reported tempo is only a
starting guess. The real grid comes from snapping predicted beats onto onset
peaks and least-squares fitting a regular grid through them, which lands within
**0.015 BPM** on material of known tempo.

Three details in there matter more than they look:

- The onset envelope used for snapping is weighted towards **low frequencies**.
  Plain spectral flux responds to brightness, so a hi-hat out-peaks a kick drum
  and the grid locks half a beat off, onto the off-beats.
- The phase is aligned by sweeping a **whole beat period** before any local
  refinement. Local snapping can only correct a beat by less than its search
  window, so it cannot rescue a grid that starts out half a beat wrong.
- Half- and double-time are separated **by evidence**, not by a plausible-BPM
  range: 90 and 180 are both ordinary tempos. The decider is whether the
  midpoints between beats are struck as hard as the beats.

**Key.** Chroma matched against the Krumhansl-Kessler profiles, then mapped to
Camelot codes. Chroma is taken from the **chord register only** (from C3 up) —
the default range starts low enough to let the kick drum vote, and a kick tuned
near F♯ will tilt every track on the record towards F♯.

**Cue points.** Beat-synchronous features are clustered into sections, and energy
decides which boundaries are worth cueing to: `mix_in` where the track gets
going, `mix_out` where it thins out. Both snapped to downbeats.

### 2. The model — `automix/model/`

Each candidate join scores five components:

| component | question |
|---|---|
| `tempo` | can they be beat-matched without artefacts? (half/double time considered) |
| `harmonic` | do the keys agree, on the Camelot wheel? |
| `energy` | does the incoming track pick up where the outgoing one left off? |
| `timbre` | do they sound like the same set? |
| `arc` | is this track the right intensity for *this point* in the set? |

Energy and timbre compare **A's outro against B's intro**, not the two track
averages. A record that ends quiet wants following with one that starts quiet,
whatever either averages to.

Ordering is a minimum-cost Hamiltonian path — exact via Held-Karp up to 12
tracks, then nearest-neighbour plus local search. The cost is **asymmetric**
(A→B is not B→A) and **position-dependent** (the arc), which rules out 2-opt:
it works by reversing a segment, and a reversed edge does not cost what the
original did. The local search uses Or-opt and swap instead, neither of which
ever reverses an edge.

### 3. Transitions — `automix/render/`

- **Equal-power gain**, `cos`/`sin` quarter-cycles, so `a² + b² = 1` throughout.
  A linear crossfade drops 3 dB at its midpoint; that sag is most of what makes
  an amateur mix sound amateur.
- **Bass swap.** Two kicks and two basslines at once is mud. The incoming track's
  low band is held down and handed over at the midpoint, so one track owns the
  bottom end at any moment. Implemented as a gain on a split band rather than a
  time-varying shelf, so at unity it is exactly a no-op.
- **The fade ends on `mix_out`, it does not start there.** That cue marks where
  the arrangement thins into an outro; a fade beginning on it spends its first
  half with the outgoing track already stripped back and the incoming one still
  turned down — a measurable hole in the middle of the blend.

### 4. One plan, two engines — `automix/model/plan.py`

Sequencing produces a `SetPlan` carrying absolute times and explicit automation
curves. Both the browser and the offline renderer consume that same plan, rather
than each re-deriving timing from the same inputs and drifting apart.

Tempo matching is by **playback rate** on both sides — resampling offline,
`playbackRate` in the browser — so pitch moves with speed, exactly as a turntable
does. A phase vocoder would hold pitch, but then the exported file would not
match what you auditioned, and the cost model already keeps stretches inside
about 6%, where the shift is under a semitone.

Each track plays at its own tempo for its whole body; only the **incoming** track
is bent, and only for the overlap, easing back once it is alone. Locking the
whole set to the first track compounds — by the fifth record you are asking for a
30% stretch.

---

## Verified

Against synthetic tracks of known tempo, key and arrangement:

| | |
|---|---|
| Tempo | within **0.015 BPM** across 90–180 BPM |
| Downbeats | within **3 ms** |
| Cue points | exact to the bar |
| Key | 20/24 exact, 23/24 correct Camelot number |
| Crossfade | `a² + b² = 1` to machine precision |

And on a rendered mix, measured from the audio itself: loudness dips **1.2 dB**
through each crossfade (a linear fade would be 3.0), the beat stays coherent to
**0.02 BPM** with 0.97 grid confidence, and the bass does not double up.

```bash
python -m pytest
```

---

## Commands

| | |
|---|---|
| `automix analyze <folder>` | analyse and print what was found |
| `automix plan <folder>` | order a set, show every transition and why it scored as it did |
| `automix render <folder> --out set.wav` | render the mix to a file |
| `automix serve` | run the web app |

`plan` and `render` take `--arc peak\|flat`, `--bars 8\|16\|32`, and a weight per
component (`--w-harmonic 2.0` to insist on key matching, and so on).

---

## Notes

- **No ffmpeg needed.** `soundfile` bundles libsndfile, which reads wav, flac,
  ogg and mp3. Exotic formats fall back to librosa, which will use ffmpeg if it
  is installed.
- **Key detection is not exact on every track.** Misses land on the relative
  major/minor or one step around the wheel — both of which still mix — but it is
  worth a glance at the confidence bars in the library view.
- **This project lives in a OneDrive folder.** `.automix/` (cache and renders) is
  in `.gitignore`; consider excluding it from sync too, or every render kicks off
  a large upload.
