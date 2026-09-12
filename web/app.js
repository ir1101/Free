import { MixEngine } from './engine.js';

const WEIGHTS = [
  ['tempo', 'Tempo', 'How closely the BPMs must match'],
  ['harmonic', 'Key', 'How strictly keys must agree'],
  ['energy', 'Energy', 'Matching the outgoing tail to the incoming head'],
  ['timbre', 'Timbre', 'Keeping the texture of the set consistent'],
  ['arc', 'Arc', 'How hard to push tracks towards the target curve'],
];
const DEFAULTS = { tempo: 1, harmonic: 1, energy: 1, timbre: 0.5, arc: 1 };

const $ = (id) => document.getElementById(id);
const engine = new MixEngine();

const state = {
  tracks: [],
  selected: new Set(),
  plan: null,
  ordered: [],
  targetArc: [],
  inspecting: 0,
  loaded: false,
};

/* ---------------------------------------------------------------- helpers */

const fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

const setStatus = (el, msg, isError = false) => {
  el.textContent = msg;
  el.classList.toggle('error', isError);
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

const post = (path, body) =>
  api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

/* ---------------------------------------------------------------- library */

function renderWeights() {
  $('weights').innerHTML = WEIGHTS.map(([key, label, hint]) => `
    <div class="weight" title="${hint}">
      <label for="w-${key}">${label}</label>
      <span class="val" id="wv-${key}">${DEFAULTS[key].toFixed(1)}</span>
      <input id="w-${key}" type="range" min="0" max="3" step="0.1" value="${DEFAULTS[key]}">
    </div>`).join('');
  for (const [key] of WEIGHTS) {
    $(`w-${key}`).addEventListener('input', (e) => {
      $(`wv-${key}`).textContent = Number(e.target.value).toFixed(1);
    });
  }
}

const weights = () =>
  Object.fromEntries(WEIGHTS.map(([k]) => [k, Number($(`w-${k}`).value)]));

function renderLibrary() {
  const body = document.querySelector('#library tbody');
  body.innerHTML = state.tracks.map((t) => {
    const conf = Math.min(t.grid_confidence, t.key_confidence * 4 + 0.4);
    return `
      <tr data-id="${t.id}">
        <td class="tick"><input type="checkbox" ${state.selected.has(t.id) ? 'checked' : ''}></td>
        <td class="title" title="${t.path}">${t.title}</td>
        <td class="num">${t.bpm.toFixed(1)}</td>
        <td><span class="key" title="${t.key_name}">${t.key}</span></td>
        <td class="num">${t.energy.toFixed(2)}</td>
        <td><div class="bar ${conf < 0.7 ? 'low' : ''}" title="beat grid ${(t.grid_confidence * 100).toFixed(0)}%">
              <i style="width:${Math.round(Math.max(0, Math.min(1, conf)) * 100)}%"></i></div></td>
      </tr>`;
  }).join('');

  body.querySelectorAll('tr').forEach((row) => {
    const id = row.dataset.id;
    row.querySelector('input').addEventListener('change', (e) => {
      if (e.target.checked) state.selected.add(id);
      else state.selected.delete(id);
      updateBuildButton();
    });
  });

  $('library-count').textContent = `${state.tracks.length} tracks`;
  $('library-panel').hidden = state.tracks.length === 0;
  $('controls-panel').hidden = state.tracks.length < 2;
  updateBuildButton();
}

function updateBuildButton() {
  $('build').disabled = state.selected.size < 2;
  $('build').textContent = state.selected.size < 2
    ? 'Select at least two tracks'
    : `Build a set from ${state.selected.size} tracks`;
}

async function refreshTracks() {
  const data = await api('/api/tracks');
  state.tracks = data.tracks;
  if (state.selected.size === 0) state.tracks.forEach((t) => state.selected.add(t.id));
  renderLibrary();
}

async function pollScan() {
  const s = await api('/api/state');
  if (s.running) {
    setStatus($('scan-status'),
      `Analysing ${s.done + 1} of ${s.total} — ${s.current}  (a few seconds per track, cached after the first pass)`);
    setTimeout(pollScan, 500);
  } else {
    const errors = s.errors && s.errors.length ? `  ${s.errors.length} file(s) skipped.` : '';
    setStatus($('scan-status'), s.finished ? `Analysed ${s.count} tracks.${errors}` : '');
    $('scan').disabled = false;
    await refreshTracks();
  }
}

/* ------------------------------------------------------------------- plan */

async function buildSet() {
  $('build').disabled = true;
  setStatus($('build-status'), 'Scoring every pair and searching for the best order…');
  try {
    const body = {
      track_ids: state.tracks.filter((t) => state.selected.has(t.id)).map((t) => t.id),
      weights: weights(),
      arc: $('arc').value,
      bars: Number($('bars').value),
      pitch_preserving: false,
    };
    const data = await post('/api/set/plan', body);
    state.plan = data.plan;
    state.ordered = data.tracks;
    state.targetArc = (await api(`/api/arc?shape=${body.arc}&n=64`)).curve;
    state.loaded = false;
    engine.setPlan(state.plan);

    $('set-panel').hidden = false;
    $('inspector-panel').hidden = false;
    $('set-summary').textContent =
      `${state.plan.segments.length} tracks · ${fmtTime(state.plan.duration)} · cost ${data.total_cost.toFixed(2)}`
      + (data.exact ? ' · exact order' : ' · heuristic order');

    invalidate();
    drawTimeline();
    renderInspectorNav();
    renderInspector(0);
    setStatus($('build-status'), 'Set ready. Press Play, or jump straight to a transition.');
    $('play').textContent = 'Play';
  } catch (err) {
    setStatus($('build-status'), err.message, true);
  } finally {
    updateBuildButton();
  }
}

/* --------------------------------------------------------------- timeline */

function drawTimeline() {
  const canvas = $('timeline');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 150;
  // Backing store in device pixels, layout size in CSS pixels. Without the
  // explicit CSS height the element takes its height from the attribute, which
  // is now in device pixels - so on a high-DPI screen the canvas lays out at
  // twice the intended height and the drawing sits in the top half.
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const plan = state.plan;
  if (!plan) return;
  const x = (t) => (t / plan.duration) * w;

  // Track lanes, alternating rows so overlaps are visible.
  plan.segments.forEach((seg, i) => {
    const y = 14 + (i % 2) * 26;
    g.fillStyle = '#3d5a80';
    g.fillRect(x(seg.start_time), y, Math.max(2, x(seg.end_time) - x(seg.start_time)), 20);
    g.fillStyle = '#cdd6e4';
    g.font = '11px ui-sans-serif, system-ui, sans-serif';
    const label = `${seg.title}  ${seg.bpm.toFixed(0)} ${seg.key}`;
    const avail = x(seg.end_time) - x(seg.start_time) - 8;
    if (avail > 40) g.fillText(trim(g, label, avail), x(seg.start_time) + 4, y + 14);
  });

  // Crossfade regions.
  g.fillStyle = 'rgba(88,200,176,0.35)';
  plan.transitions.forEach((tr) => {
    g.fillRect(x(tr.start_time), 10, Math.max(2, x(tr.duration) - x(0)), 66);
  });

  // Energy: what the set actually does, against what the arc wanted.
  const top = 88;
  const height = 46;
  if (state.targetArc.length) {
    g.strokeStyle = '#55607a';
    g.setLineDash([4, 4]);
    g.beginPath();
    state.targetArc.forEach((v, i) => {
      const px = (i / (state.targetArc.length - 1)) * w;
      const py = top + height - v * height;
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    });
    g.stroke();
    g.setLineDash([]);
  }

  g.strokeStyle = '#e0af68';
  g.lineWidth = 2;
  g.beginPath();
  let first = true;
  plan.segments.forEach((seg) => {
    const meta = state.ordered.find((t) => t.id === seg.track_id);
    const curve = meta && meta.energy_curve && meta.energy_curve.length
      ? meta.energy_curve : [meta ? meta.energy : 0.5];
    const span = seg.end_time - seg.start_time;
    curve.forEach((v, i) => {
      const px = x(seg.start_time + (i / Math.max(1, curve.length - 1)) * span);
      const py = top + height - v * height;
      if (first) { g.moveTo(px, py); first = false; } else g.lineTo(px, py);
    });
  });
  g.stroke();
  g.lineWidth = 1;

  // Playhead.
  const t = engine.currentTime;
  if (t > 0) {
    g.strokeStyle = '#f7768e';
    g.beginPath();
    g.moveTo(x(t), 6);
    g.lineTo(x(t), h - 6);
    g.stroke();
  }
}

function trim(g, text, maxWidth) {
  if (g.measureText(text).width <= maxWidth) return text;
  let s = text;
  while (s.length > 1 && g.measureText(`${s}…`).width > maxWidth) s = s.slice(0, -1);
  return `${s}…`;
}

/* -------------------------------------------------------------- inspector */

function renderInspectorNav() {
  $('inspector-nav').innerHTML = state.plan.transitions
    .map((_, i) => `<button class="chip ${i === state.inspecting ? 'active' : ''}" data-i="${i}">${i + 1}</button>`)
    .join('');
  $('inspector-nav').querySelectorAll('.chip').forEach((b) =>
    b.addEventListener('click', () => renderInspector(Number(b.dataset.i))));
}

function costClass(total) {
  if (total < 2.5) return 'good';
  if (total < 6) return 'warn';
  return 'bad';
}

function renderInspector(i) {
  state.inspecting = i;
  renderInspectorNav();
  const tr = state.plan.transitions[i];
  if (!tr) return;
  const c = tr.cost || {};
  const from = state.plan.segments[tr.from_index];
  const to = state.plan.segments[tr.to_index];

  const parts = [
    ['tempo', '#7aa2f7'], ['harmonic', '#58c8b0'], ['energy', '#e0af68'],
    ['timbre', '#bb9af7'], ['arc', '#f7768e'],
  ];
  const total = Math.max(1e-6, c.total || 0);
  const bar = parts
    .map(([k, col]) => `<i style="width:${((c[k] || 0) / total) * 100}%;background:${col}"></i>`)
    .join('');

  $('inspector').innerHTML = `
    <div class="insp-grid">
      <div class="insp-card">
        <h3>Overall</h3>
        <div class="big ${costClass(c.total || 0)}">${(c.total || 0).toFixed(2)}</div>
        <div class="cost-bar">${bar}</div>
        ${parts.map(([k, col]) =>
          `<div class="kv"><span><i class="swatch" style="background:${col}"></i> ${k}</span>
           <span>${(c[k] || 0).toFixed(2)}</span></div>`).join('')}
      </div>

      <div class="insp-card">
        <h3>Key</h3>
        <div class="big">${from.key} &rarr; ${to.key}</div>
        <div class="kv"><span>relationship</span><span>${c.harmonic_label || '—'}</span></div>
        ${Math.abs(c.semitone_shift || 0) >= 0.05
          ? `<div class="kv"><span>rate-match shift</span><span>${(c.semitone_shift).toFixed(2)} st</span></div>` : ''}
      </div>

      <div class="insp-card">
        <h3>Tempo</h3>
        <div class="big">${from.bpm.toFixed(1)} &rarr; ${to.bpm.toFixed(1)}</div>
        <div class="kv"><span>stretch applied to incoming</span>
          <span>${(c.stretch_percent || 0) >= 0 ? '+' : ''}${(c.stretch_percent || 0).toFixed(2)}%</span></div>
        <div class="kv"><span>read as</span><span>${c.octave === 1 ? 'same time' : c.octave === 2 ? 'double time' : 'half time'}</span></div>
        <div class="kv"><span>overlap</span><span>${tr.bars} bars · ${tr.duration.toFixed(1)}s</span></div>
      </div>

      <div class="insp-card">
        <h3>Crossfade curves</h3>
        <canvas id="curve-${i}" height="110"></canvas>
        <div class="legend">
          <span><i class="swatch" style="background:#f7768e"></i>out</span>
          <span><i class="swatch" style="background:#58c8b0"></i>in</span>
          <span><i class="swatch" style="background:#e0af68"></i>bass</span>
        </div>
      </div>
    </div>
    ${(c.notes && c.notes.length)
      ? `<ul class="notes">${c.notes.map((n) => `<li>${n}</li>`).join('')}</ul>` : ''}
  `;
  drawCurves(`curve-${i}`, tr);
}

function drawCurves(canvasId, tr) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 110;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const pad = 8;
  const plot = (values, colour, dashed = false) => {
    g.strokeStyle = colour;
    g.lineWidth = 2;
    g.setLineDash(dashed ? [3, 3] : []);
    g.beginPath();
    values.forEach((v, i) => {
      const px = pad + (i / (values.length - 1)) * (w - 2 * pad);
      const py = h - pad - v * (h - 2 * pad);
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    });
    g.stroke();
    g.setLineDash([]);
  };

  // Constant total power is the property worth seeing, so draw it.
  const out = tr.gain_out.values;
  const inc = tr.gain_in.values;
  const power = out.map((v, i) => Math.sqrt(v * v + inc[i] * inc[i]));

  g.strokeStyle = '#2a2f3a';
  g.beginPath();
  g.moveTo(pad, h - pad - (h - 2 * pad));
  g.lineTo(w - pad, h - pad - (h - 2 * pad));
  g.stroke();

  plot(out, '#f7768e');
  plot(inc, '#58c8b0');
  plot(tr.bass_in.values, '#e0af68', true);
  plot(power, '#8b93a3');

  g.fillStyle = '#8b93a3';
  g.font = '10px ui-sans-serif, system-ui, sans-serif';
  g.fillText('total power stays flat', pad + 4, h - pad - (h - 2 * pad) - 4);
}

/* -------------------------------------------------------------- transport */

async function ensureLoaded() {
  if (state.loaded) return;
  setStatus($('play-status'), 'Decoding audio…');
  await engine.load(state.plan, state.ordered, (done, total, title) => {
    setStatus($('play-status'), `Decoding ${done + 1}/${total} ${title}`);
  });
  state.loaded = true;
  setStatus($('play-status'), '');
}

async function togglePlay() {
  try {
    if (engine.playing) {
      engine.pause();
      $('play').textContent = 'Play';
    } else {
      await ensureLoaded();
      await engine.play();
      $('play').textContent = 'Pause';
    }
  } catch (err) {
    setStatus($('play-status'), err.message, true);
  }
}

async function jumpTransition(delta) {
  if (!state.plan) return;
  const now = engine.currentTime;
  const times = state.plan.transitions.map((t) => t.start_time);
  let target = delta > 0
    ? times.find((t) => t > now + 1)
    : [...times].reverse().find((t) => t < now - 1);
  if (target === undefined) target = delta > 0 ? times[0] : times[times.length - 1];
  if (target === undefined) return;

  const i = times.indexOf(target);
  renderInspector(i);
  try {
    await ensureLoaded();
    await engine.seek(Math.max(0, target - 4));
    if (!engine.playing) {
      await engine.play(Math.max(0, target - 4));
      $('play').textContent = 'Pause';
    }
    setStatus($('play-status'), `Transition ${i + 1}: ${state.plan.segments[i].title} → ${state.plan.segments[i + 1].title}`);
  } catch (err) {
    setStatus($('play-status'), err.message, true);
  }
}

let lastDrawn = -1;

/**
 * Repaint the playhead while the set is playing.
 *
 * Only redraws when the visible time has actually moved. A loop that repainted
 * every frame regardless would burn a core while paused, for a canvas that has
 * not changed - and it keeps the page from ever going idle.
 */
function tick() {
  if (state.plan) {
    const t = engine.currentTime;
    if (Math.abs(t - lastDrawn) > 0.05) {
      lastDrawn = t;
      $('clock').textContent = `${fmtTime(t)} / ${fmtTime(engine.duration)}`;
      if (!$('seek').matches(':active')) {
        $('seek').value = String(Math.round((t / Math.max(1e-6, engine.duration)) * 1000));
      }
      drawTimeline();
    }
  }
  requestAnimationFrame(tick);
}

/** Force the next tick to repaint, after anything that changes the picture. */
function invalidate() {
  lastDrawn = -1;
}

/* ------------------------------------------------------------------ setup */

renderWeights();
$('scan').addEventListener('click', async () => {
  const folder = $('folder').value.trim();
  if (!folder) return setStatus($('scan-status'), 'Give me a folder path first.', true);
  $('scan').disabled = true;
  try {
    await post('/api/library/scan', { folder, refresh: $('refresh').checked });
    pollScan();
  } catch (err) {
    setStatus($('scan-status'), err.message, true);
    $('scan').disabled = false;
  }
});
$('folder').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('scan').click(); });
$('select-all').addEventListener('click', () => {
  state.tracks.forEach((t) => state.selected.add(t.id));
  renderLibrary();
});
$('select-none').addEventListener('click', () => {
  state.selected.clear();
  renderLibrary();
});
$('build').addEventListener('click', buildSet);
$('play').addEventListener('click', togglePlay);
$('next-tr').addEventListener('click', () => jumpTransition(1));
$('prev-tr').addEventListener('click', () => jumpTransition(-1));
$('seek').addEventListener('change', async (e) => {
  if (!state.plan) return;
  await ensureLoaded();
  await engine.seek((Number(e.target.value) / 1000) * engine.duration);
});
$('volume').addEventListener('input', (e) => engine.setVolume(Number(e.target.value) / 100));
window.addEventListener('resize', () => { invalidate(); drawTimeline(); renderInspector(state.inspecting); });

// Exposed for debugging from the console: inspect the plan, the engine's clock,
// and what is currently scheduled.
window.__automix = { engine, state };

api('/api/state').then((s) => {
  if (s.folder) $('folder').value = s.folder;
  if (s.running) { $('scan').disabled = true; pollScan(); }
  else if (s.count) refreshTracks();
}).catch(() => {});

tick();
