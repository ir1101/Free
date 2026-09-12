/**
 * Live mixing engine.
 *
 * Plays a SetPlan through Web Audio. Every fade, filter move and tempo change is
 * scheduled on the AudioContext clock ahead of time rather than driven from a
 * timer. That is the whole trick: the audio thread runs the automation at sample
 * accuracy, so a busy main thread can stutter the UI without ever touching the
 * sound. Driving fades from requestAnimationFrame would make every dropped frame
 * audible.
 *
 * The signal path per track mirrors the offline renderer exactly:
 *
 *     source -> lowshelf (bass swap) -> gain (equal-power fade) -> master
 *
 * Tempo matching uses playbackRate, which moves pitch along with speed, the same
 * way a turntable does. The offline renderer resamples for the same reason, so
 * what you hear here is what gets exported.
 */

const BASS_CROSSOVER_HZ = 200;

export class MixEngine {
  constructor() {
    this.ctx = null;
    this.buffers = new Map(); // track id -> AudioBuffer
    this.plan = null;
    this.nodes = [];
    this.master = null;
    this.startedAt = 0; // ctx time when the current playback began
    this.offset = 0; // set time that corresponds to startedAt
    this.playing = false;
  }

  ensureContext() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      this.master.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  get duration() {
    return this.plan ? this.plan.duration : 0;
  }

  /** Where we are in set time. */
  get currentTime() {
    if (!this.ctx || !this.playing) return this.offset;
    return Math.min(this.duration, this.offset + (this.ctx.currentTime - this.startedAt));
  }

  /**
   * Adopt a plan without decoding anything yet.
   *
   * Kept separate from `load` so the transport can show the set's length and
   * scrub position the moment it is built, rather than staying at 0:00 until
   * the audio has finished decoding - which for a long set is a while to sit
   * looking at a dead clock.
   */
  setPlan(plan) {
    if (this.plan !== plan) {
      this.pause();
      this.plan = plan;
      this.offset = 0;
    }
  }

  /**
   * Decode every track the plan needs.
   *
   * Decoded audio is kept keyed by track id and reused, so re-planning the same
   * library - which is the main thing you do while tuning weights - does not
   * decode anything twice.
   */
  async load(plan, tracks, onProgress) {
    this.ensureContext();
    this.plan = plan;
    const needed = [...new Set(plan.segments.map((s) => s.track_id))];
    let done = 0;
    for (const id of needed) {
      if (!this.buffers.has(id)) {
        const meta = tracks.find((t) => t.id === id);
        if (onProgress) onProgress(done, needed.length, meta ? meta.title : id);
        const res = await fetch(`/api/audio/${id}`);
        if (!res.ok) throw new Error(`could not load audio for ${meta ? meta.title : id}`);
        const bytes = await res.arrayBuffer();
        this.buffers.set(id, await this.ctx.decodeAudioData(bytes));
      }
      done += 1;
      if (onProgress) onProgress(done, needed.length, '');
    }
  }

  /**
   * Where in the source file a segment is at a given set time.
   *
   * Mirrors Segment.source_at on the server: position is the integral of
   * playback rate, which is also how Web Audio advances a source node. Both
   * sides computing the same integral is what keeps them in step.
   */
  sourceAt(seg, t) {
    if (t <= seg.start_time) return seg.source_start;
    let pos = seg.source_start;
    const rampStart = Math.max(seg.start_time, seg.rate_ramp_start);
    const flat = Math.min(t, rampStart) - seg.start_time;
    pos += flat * seg.playback_rate;
    if (t > rampStart) {
      const span = Math.max(1e-9, seg.rate_ramp_end - rampStart);
      const elapsed = Math.min(t, seg.rate_ramp_end) - rampStart;
      const rateAt = seg.playback_rate + (1 - seg.playback_rate) * (elapsed / span);
      pos += (elapsed * (seg.playback_rate + rateAt)) / 2;
      if (t > seg.rate_ramp_end) pos += t - seg.rate_ramp_end;
    }
    return pos;
  }

  /**
   * The part of a curve still ahead of `from`, resampled onto its own grid.
   *
   * Needed because you can seek into the middle of a crossfade, and
   * setValueCurveAtTime always plays a curve from its first value: handing it
   * the whole curve late would restart the fade rather than continue it.
   */
  remainingCurve(curve, from) {
    const end = curve.start + curve.duration;
    if (from >= end) return null;
    const t0 = Math.max(from, curve.start);
    const frac = (t0 - curve.start) / curve.duration;
    const n = curve.values.length;
    const first = frac * (n - 1);
    const out = new Float32Array(Math.max(2, Math.ceil(n * (1 - frac))));
    for (let i = 0; i < out.length; i += 1) {
      const x = first + ((n - 1 - first) * i) / (out.length - 1);
      const lo = Math.floor(x);
      const hi = Math.min(n - 1, lo + 1);
      out[i] = curve.values[lo] + (curve.values[hi] - curve.values[lo]) * (x - lo);
    }
    return { values: out, start: t0, duration: end - t0 };
  }

  toDb(gain) {
    return 20 * Math.log10(Math.max(1e-4, gain));
  }

  stopNodes() {
    for (const n of this.nodes) {
      try {
        n.source.stop();
      } catch (e) {
        /* already stopped */
      }
      n.source.disconnect();
      n.shelf.disconnect();
      n.gain.disconnect();
    }
    this.nodes = [];
  }

  /** Schedule everything from set time `from` onwards. */
  schedule(from) {
    const ctx = this.ensureContext();
    const base = ctx.currentTime + 0.12; // a beat of headroom to get it all queued
    const plan = this.plan;
    const at = (setTime) => base + Math.max(0, setTime - from);

    const outgoing = new Map(plan.transitions.map((t) => [t.from_index, t]));
    const incoming = new Map(plan.transitions.map((t) => [t.to_index, t]));

    for (const seg of plan.segments) {
      if (seg.end_time <= from) continue;
      const buffer = this.buffers.get(seg.track_id);
      if (!buffer) continue;

      const source = ctx.createBufferSource();
      source.buffer = buffer;

      const shelf = ctx.createBiquadFilter();
      shelf.type = 'lowshelf';
      shelf.frequency.value = BASS_CROSSOVER_HZ;
      shelf.gain.value = 0;

      const gain = ctx.createGain();
      gain.gain.value = 1;

      source.connect(shelf).connect(gain).connect(this.master);

      const startSet = Math.max(seg.start_time, from);
      const sourceOffset = this.sourceAt(seg, startSet);

      // Tempo: hold the matched rate through the overlap, then ease back to the
      // track's own tempo. Web Audio integrates the rate to find its read
      // position, exactly as sourceAt does.
      source.playbackRate.setValueAtTime(
        startSet >= seg.rate_ramp_end ? 1 : seg.playback_rate,
        at(startSet),
      );
      if (seg.rate_ramp_end > startSet && seg.rate_ramp_end > seg.rate_ramp_start) {
        source.playbackRate.setValueAtTime(
          seg.playback_rate,
          at(Math.max(startSet, seg.rate_ramp_start)),
        );
        source.playbackRate.linearRampToValueAtTime(1, at(seg.rate_ramp_end));
      }

      const applyCurves = (tr, gainCurve, bassCurve) => {
        if (!tr) return;
        const g = this.remainingCurve(gainCurve, from);
        if (g) gain.gain.setValueCurveAtTime(g.values, at(g.start), g.duration);
        const b = this.remainingCurve(bassCurve, from);
        if (b) {
          const db = new Float32Array(b.values.length);
          for (let i = 0; i < db.length; i += 1) db[i] = this.toDb(b.values[i]);
          shelf.gain.setValueCurveAtTime(db, at(b.start), b.duration);
        }
      };

      applyCurves(incoming.get(seg.index), incoming.get(seg.index)?.gain_in,
        incoming.get(seg.index)?.bass_in);
      applyCurves(outgoing.get(seg.index), outgoing.get(seg.index)?.gain_out,
        outgoing.get(seg.index)?.bass_out);

      const out = outgoing.get(seg.index);
      const stopAt = out ? out.start_time + out.duration : seg.end_time;
      source.start(at(startSet), Math.max(0, sourceOffset));
      source.stop(at(Math.min(stopAt, plan.duration + 1)));

      this.nodes.push({ source, shelf, gain, index: seg.index });
    }
  }

  async play(from = null) {
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') await ctx.resume();
    const start = from === null ? this.currentTime : from;
    this.stopNodes();
    this.offset = Math.max(0, Math.min(start, this.duration));
    this.startedAt = ctx.currentTime + 0.12;
    this.schedule(this.offset);
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.offset = this.currentTime;
    this.playing = false;
    this.stopNodes();
  }

  async seek(t) {
    const wasPlaying = this.playing;
    this.pause();
    this.offset = Math.max(0, Math.min(t, this.duration));
    if (wasPlaying) await this.play(this.offset);
  }

  setVolume(v) {
    if (this.master) this.master.gain.value = v;
  }

  /** Set time just before transition `i`, for the jump-to-transition button. */
  transitionCue(i, lead = 4) {
    const tr = this.plan?.transitions[i];
    return tr ? Math.max(0, tr.start_time - lead) : 0;
  }
}
