/**
 * The wave field.
 *
 * A bundle of thin strands sharing one travelling waveform, each offset a little
 * from the next. Where that offset is squeezed the strands converge into a bright
 * seam; where it opens they fan into a ribbon. That pinch-and-fan is the
 * structure - it is not many independent curves, it is one curve seen at many
 * depths. Two such ribbons run at different rates so their seams drift in and out
 * of phase and cross.
 *
 * **Colour is a channel split.** Every strand is drawn three times, in red, green
 * and blue, each nudged a little way off the others and composited additively.
 * Where the three line up they sum to white; where the curve is steep, or the
 * separation opens, they pull apart into colour fringes. The colour is therefore
 * not painted onto the drawing - it falls out of the same geometry, which is why
 * it stays coherent as the shape moves.
 *
 * Each channel is lit by a different part of the spectrum: bass reddens, mids
 * green, treble blues. The palette then reports what the music is doing rather
 * than cycling on a timer.
 *
 * Efficiency note: each strand's path is built once and stroked three times under
 * a translate. Building it three times over would triple the per-point maths to
 * produce three copies that differ only by a shift.
 */

const STEP = 4;                 // px between sampled points along a strand
const BACKGROUND = '#0a0a0c';

/**
 * Ceiling on the backing store, in device pixels.
 *
 * Drawing cost here is pixel-bound: additive blending of a few hundred
 * antialiased strokes scales with area, not with the number of strands. At a
 * normal window it takes about 3 ms a frame, but a 4K display at devicePixelRatio
 * 2 is roughly nine times the pixels, which would put it past the frame budget.
 * Capping the resolution costs almost nothing visually for hairlines and keeps
 * the frame rate flat across screen sizes.
 */
const MAX_PIXELS = 2600 * 1600;

/**
 * The three channels: how far each is displaced, and which band lights it.
 *
 * Displacement is mostly vertical with a slight lateral lean, so the fringes
 * favour a direction and the split reads as an optical effect rather than a
 * symmetrical smear. Green sits near zero and carries the mids, which holds the
 * bright core of the image steady while red and blue swing either side of it.
 */
const CHANNELS = [
  { rgb: [255, 48, 92], dx: -0.4, dy: -1.0, band: 'bass', gain: 1.15 },
  { rgb: [56, 255, 150], dx: 0.0, dy: 0.15, band: 'mid', gain: 1.0 },
  { rgb: [72, 128, 255], dx: 0.4, dy: 1.0, band: 'treble', gain: 1.35 },
];

/**
 * Two ribbons.
 *
 * A single bundle reads as a plain wavy band; overlapping systems running at
 * different rates give the dense knots and open weave. The frequency ratios are
 * deliberately not simple multiples - 7.6 against 14.2 rather than against 15.2 -
 * so the pattern takes a very long time to repeat and never looks like a loop.
 */
const RIBBONS = [
  { lines: 26, freq: [7.6, 14.2, 25.1], drift: [0.29, -0.18, 0.42],
    pinchFreq: 4.1, pinchDrift: -0.23, offset: -0.06, bob: 0.11, band: 'bass',
    weight: 1.0 },
  { lines: 22, freq: [5.9, 11.4, 21.3], drift: [-0.21, 0.26, -0.35],
    pinchFreq: 3.3, pinchDrift: 0.19, offset: 0.07, bob: -0.08, band: 'mid',
    weight: 0.88 },
];

/** Exponential smoothing - band levels are jumpy frame to frame. */
const approach = (value, target, rate) => value + (target - value) * rate;

export class WaveField {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.analyser = null;
    this.bins = null;

    this.time = 0;
    this.active = false;
    this.bands = { bass: 0, mid: 0, treble: 0 };
    this.kick = 0;                       // transient swell on track change

    this.resize = this.resize.bind(this);
    this.frame = this.frame.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const wanted = window.devicePixelRatio || 1;
    const dpr = Math.min(wanted, Math.sqrt(MAX_PIXELS / Math.max(1, w * h)));

    // Backing store in device pixels, layout size in CSS pixels: set only one
    // and the canvas lays out at the wrong height on a high-DPI screen.
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
  }

  listenTo(analyser) {
    this.analyser = analyser;
    this.bins = new Uint8Array(analyser.frequencyBinCount);
  }

  setActive(on) { this.active = on; }

  /** A brief swell, used to mark a track change. */
  bump() { this.kick = 1; }

  /** Average a slice of the spectrum, normalised to 0-1. */
  band(from, to) {
    let sum = 0;
    for (let i = from; i < to; i += 1) sum += this.bins[i];
    return sum / ((to - from) * 255);
  }

  readAudio() {
    if (!this.analyser || !this.active) {
      // Idle: keep breathing gently so the page never looks frozen.
      const idle = 0.5 + 0.5 * Math.sin(this.time * 0.4);
      this.bands.bass = approach(this.bands.bass, 0.11 + 0.05 * idle, 0.02);
      this.bands.mid = approach(this.bands.mid, 0.13, 0.02);
      this.bands.treble = approach(this.bands.treble, 0.05, 0.02);
      return;
    }
    this.analyser.getByteFrequencyData(this.bins);
    const n = this.bins.length;
    // Different rates per band: bass should follow a kick quickly and fall away
    // slowly, treble is noisy and needs heavier smoothing or it flickers.
    this.bands.bass = approach(this.bands.bass, this.band(1, Math.floor(n * 0.02)), 0.22);
    this.bands.mid = approach(this.bands.mid,
      this.band(Math.floor(n * 0.02), Math.floor(n * 0.14)), 0.15);
    this.bands.treble = approach(this.bands.treble,
      this.band(Math.floor(n * 0.14), Math.floor(n * 0.55)), 0.10);
  }

  start() {
    this.last = performance.now();
    requestAnimationFrame(this.frame);
  }

  frame(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.time += dt;
    this.kick = Math.max(0, this.kick - dt * 1.6);
    this.readAudio();
    this.draw();
    requestAnimationFrame(this.frame);
  }

  draw() {
    const { ctx, w, h } = this;
    const t = this.time;
    const { bass, mid, treble } = this.bands;
    const level = { bass, mid, treble };

    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, w, h);
    // Additive from here on: overlapping channels sum towards white, which is
    // what makes the aligned cores luminous and leaves the edges coloured.
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineJoin = 'round';

    const cy = h * 0.5;
    // Scaled to the window so the drawing keeps its proportions instead of
    // filling a tall screen and vanishing on a short one.
    const unit = Math.min(h * 0.5, w * 0.26);
    const swell = 1 + this.kick * 0.6;
    const energy = Math.min(1, bass + mid * 0.7 + treble * 0.5);

    // How far the channels pull apart. Small at rest, so a quiet passage reads
    // as near-white line-work and opens into full colour as the track lifts.
    const split = (0.9 + energy * 7.5) * swell;

    for (const ribbon of RIBBONS) {
      const drive = ribbon.band === 'bass' ? bass : mid;
      const k = ribbon.weight * swell;

      const a1 = unit * (0.17 + drive * 0.42) * k;
      const a2 = unit * (0.10 + mid * 0.24) * k;
      const a3 = unit * (0.045 + treble * 0.15);
      const spread = unit * (0.40 + drive * 0.44) * k;
      // A slow vertical wander, so the ribbon is never quite settled.
      const base = cy + unit * ribbon.offset + unit * 0.11 * Math.sin(t * ribbon.bob);

      for (let i = 0; i < ribbon.lines; i += 1) {
        const u = i / (ribbon.lines - 1) - 0.5;   // -0.5 .. 0.5 across the bundle
        const path = new Path2D();

        for (let x = 0; x <= w + STEP; x += STEP) {
          const p = x / w;

          // One shared travelling waveform: this is what makes the strands read
          // as a single object rather than a stack of curves.
          let y = a1 * Math.sin(p * ribbon.freq[0] + t * ribbon.drift[0])
                + a2 * Math.sin(p * ribbon.freq[1] + t * ribbon.drift[1] + 1.7)
                + a3 * Math.sin(p * ribbon.freq[2] + t * ribbon.drift[2] + 3.1);

          // Bundle width, itself waving along x, raised to a high power so the
          // collapses are narrow and the fans broad. That contrast is what makes
          // the crossing seams instead of an evenly thick band.
          const pinch = 0.05 + 0.95 * Math.pow(
            0.5 + 0.5 * Math.sin(p * ribbon.pinchFreq + t * ribbon.pinchDrift), 2.2);
          y += u * spread * pinch
               * (1 + 0.6 * Math.sin(p * 8.7 + t * 0.15 + u * 2.6));

          // Per-strand shimmer, so strands stay distinguishable in the fans.
          y += a3 * 0.75 * Math.sin(p * 58 + t * 1.15 + i * 0.61);

          const py = base + y;
          if (x === 0) path.moveTo(x, py); else path.lineTo(x, py);
        }

        // Depth: strands near the middle of the bundle sit brighter and heavier
        // than those at its edges, which gives the ribbon body rather than the
        // flat look of a stack of equally weighted lines.
        const depth = 1 - Math.min(1, Math.abs(u) * 1.7);
        const weight = 0.55 + 0.45 * depth;

        for (const ch of CHANNELS) {
          // A floor under the band term so a channel never drops out entirely -
          // a quiet passage should thin the colour, not lose it.
          const lit = 0.42 + level[ch.band] * 1.5;
          const alpha = 0.075 * ribbon.weight * weight * lit * ch.gain;
          if (alpha < 0.002) continue;
          ctx.strokeStyle = `rgba(${ch.rgb[0]}, ${ch.rgb[1]}, ${ch.rgb[2]}, ${alpha.toFixed(4)})`;
          ctx.lineWidth = 0.7 + 0.5 * depth;
          ctx.save();
          ctx.translate(ch.dx * split, ch.dy * split);
          ctx.stroke(path);
          ctx.restore();
        }
      }
    }

    ctx.globalCompositeOperation = 'source-over';
  }
}
