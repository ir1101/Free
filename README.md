# AutoMix

Point it at a folder of music. It works out the order the tracks flow best in,
beat-matches them, and plays the whole thing as one continuous mix.

Analysis, ordering, crossfading and rendering all happen on the server; the
browser receives one continuous audio stream and draws it.

The page is a single view: a nav bar, a full-window wave field that moves with
the music, and a transport with previous / play-pause / next, a seek bar and a
tracklist.

---

## Run it

```bash
pip install -e .
```

Drop audio into `music/` (or pass a path), then:

```bash
automix serve
```

Open <http://127.0.0.1:8420> and press play.

```bash
automix serve "C:/Users/you/Music"
```

First analysis costs ~10-25 s per track depending on length. Results are cached
under `.automix/cache`, so everything after the first pass starts immediately.

**Adding music while it is running** — drop files in and the page notices within
a few seconds. If nothing is playing it rebuilds the set straight away; if you
are mid-track it offers the rebuild in the nav rather than cutting the music off.
Only the new files cost analysis time; the rest come from cache. Removing or
replacing a file is picked up the same way.

---

## What "a good transition" means here

Twelve things are measured for every candidate pair, always comparing the
**outgoing track's outro against the incoming track's intro** — never the two
track averages. A record that ends sparse and quiet wants following with one
that starts sparse and quiet, whatever either averages to across four minutes.

| | |
|---|---|
| `tempo` | can they be beat-matched without artefacts (half/double time included) |
| `harmonic` | do the keys agree, on the Camelot wheel |
| `rhythm` | do the grooves agree — where the hits land inside the bar |
| `energy` | does the incoming track pick up where the outgoing one left off |
| `loudness` | will there be an audible step in level |
| `brightness` | does the tone balance jump |
| `timbre` | do they sound like the same set |
| `bass` | will two heavy low ends collide |
| `voice` | will two vocals talk over each other |
| `crest` | does punchy meet squashed |
| `pulse` | is there a clear enough beat on both sides to lock to |
| `arc` | is this track the right intensity for *this point* in the set |

Three of these deserve a note:

- **`rhythm` is measured twice**, full-band and low-band. Spectral flux follows
  brightness, so on a plain four-to-the-floor loop the off-beat hats take ~76% of
  the weight and the kick only 20%. A groove described that way is really a
  description of the hats, and two records with clashing kicks would look
  compatible. The kick gets its own band and the larger share of the vote.
- **`bass` and `voice` are products, not differences.** Two sparse basslines
  overlap happily; so do a heavy one and a sparse one — the bass swap exists to
  hand the low end over. Only two dense low ends at once is unrecoverable.
- **`harmonic` is scaled by detector confidence.** Key detection is not reliable
  on every record, and enforcing a harmonic rule derived from a coin-flip reading
  is worse than letting the other eleven components decide.

Weights are tunable per component: `automix serve --w-voice 2.0 --w-harmonic 0.5`.

---

## How it works

### Analysis — `automix/analysis/`

Every expensive step (STFT, harmonic separation, chroma, MFCC) is computed once
into a shared `AnalysisContext` and read by all the extractors, rather than each
module recomputing its own.

**Beat grid.** `librosa`'s tempogram is coarse — it will call 128 BPM "129.2",
and a 1% error accumulates into a whole beat of drift across a 32-bar
transition. So its estimate is only a starting guess; the real grid comes from
snapping predicted beats onto onset peaks and least-squares fitting. That lands
within **0.015 BPM**.

Three details matter more than they look:

- The onset envelope used for snapping is weighted towards **low frequencies**,
  or a hi-hat out-peaks the kick and the grid locks half a beat off.
- Phase is aligned by sweeping a **whole beat period** first. Local snapping can
  only correct by less than its window, so it cannot rescue a grid that starts
  half a beat wrong.
- Half- and double-time are separated **by evidence**, not by a plausible-BPM
  range: 90 and 180 are both ordinary tempos. The decider is whether the
  midpoints between beats are struck as hard as the beats.

**Key.** Chroma against the Krumhansl-Kessler profiles, mapped to Camelot codes.
Taken from the **chord register only** (C3 up) — the default range lets the kick
drum vote, and a kick tuned near F♯ tilts the whole record towards F♯.

**Cue points.** Beat-synchronous features clustered into sections; energy picks
which boundaries are worth cueing to. Both snapped to downbeats.

### Ordering — `automix/model/`

A minimum-cost Hamiltonian path: exact via Held-Karp up to 12 tracks, then
nearest-neighbour plus local search.

The cost is **asymmetric** (A→B is not B→A) and **position-dependent** (the arc),
which rules out 2-opt — it works by reversing a segment, and a reversed edge does
not cost what the original did. The local search uses Or-opt and swap instead,
neither of which ever reverses an edge.

### Transitions — `automix/render/`

- **Equal-power gain**, so `a² + b² = 1` throughout. A linear crossfade drops
  3 dB at its midpoint; that sag is most of what makes an amateur mix sound
  amateur.
- **Bass swap.** The incoming track's low band is held down and handed over at
  the midpoint, so one track owns the bottom end at any moment. Implemented as a
  gain on a split band, so at unity it is exactly a no-op.
- **The fade ends on `mix_out`, it does not start there.** That cue marks where
  the arrangement thins into an outro; a fade beginning on it spends its first
  half with the outgoing track already stripped back and the incoming one still
  turned down — a measurable hole in the middle of the blend.
- Tempo is matched by **playback rate**, so pitch moves with speed like a
  turntable. Only the *incoming* track is bent, and only for the overlap; locking
  the whole set to the first track compounds into a 30% stretch by the fifth
  record.

### The page — `web/`

`index.html` and `field.js`, no build step.

The **wave field** is a bundle of thin lines sharing one travelling waveform,
each offset slightly from the next. Where that offset is squeezed the lines
converge into a dark seam; where it opens they fan into a ribbon. Two such
ribbons run at different rates so their seams drift in and out of phase and
cross — one bundle alone reads as a plain wavy band.

Audio drives it through three bands rather than one level, because they want to
do different things: bass moves the slow body of the wave, mids open and close
the bundle, treble adds ripple along each strand. A single amplitude would just
make the whole thing pump. The cursor drags the field toward it, and clicking
sends a ring outwards.

Skipping tracks is a `from=` on the stream, not a seek: the mix is generated on
demand, so starting elsewhere is simply a new request. Jumping to a track lands
at the *start of the crossfade into it*, so you hear it mix in rather than
arriving mid-track.

### Streaming — `automix/render/stream.py`

The mix is generated as it is consumed, at ~100× realtime, so there is no wait
for a render and no giant temporary file.

- **Whole segments, not windows.** The bass swap's zero-phase filter would leave
  a click at every window boundary. Only ~3 segments are ever live at once, so
  memory stays flat however long the set is.
- **A look-ahead limiter**, because a stream cannot peak-normalise — it has not
  seen the end yet. Two equal-power crossfades summing at once reached 1.17 of
  full scale on the test set.
- **A real `Content-Length`**, since the plan fixes the duration in advance.
  Without it the player reports an infinite duration.

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
| Stream vs offline render | identical, **0.00e+00** over 12.4M frames |
| Loudness through a crossfade | dips **1.2 dB** (a linear fade dips 3.0) |
| Beat coherence through a fade | 126.02 vs 126.00 BPM, 0.97 grid confidence |

```bash
python -m pytest
```

111 tests.

---

## Commands

| | |
|---|---|
| `automix serve [folder]` | play the folder as a continuous mix (default: `music/`) |

Keyboard: space toggles playback, escape closes the tracklist, arrow keys seek
when the bar has focus.
| `automix analyze <folder>` | analyse and print what was found |
| `automix plan <folder>` | show the running order and why each join scored as it did |
| `automix render <folder> --out set.wav` | render the mix to a file |

`plan`, `render` and `serve` take `--arc peak|flat`, `--bars 8|16|32`, and a
weight per component.

---

## Notes

- **No ffmpeg needed.** `soundfile` reads wav, flac, ogg and mp3. Exotic formats
  fall back to librosa, which uses ffmpeg if present.
- **Key detection is not exact on every track.** Misses land on the relative
  major/minor or one step around the wheel — both of which still mix.
- **`music/` and `.automix/` are gitignored.** This project sits in a OneDrive
  folder; consider excluding them from sync too.
