/**
 * Plan demo: Qwen3-VL-2B answers a list question in the visitor's browser, and the page draws where
 * two groups of attention heads look while it does.
 *
 * Live path (photos): transformers.js on WebGPU with the q4f16 weights of the Gaze Heads demo model. The decoder
 * graph is our own copy (model/decoder_model_merged_q4f16.onnx): the same graph plus one output,
 * gaze_attn (28 layers, 16 heads, 1, all tokens), the attention of every head at the last processed token.
 * The image encoder never runs here; each image ships its features (images/<name>.f16.bin).
 *
 * Recorded path: the page plays runs/<image>__<question>.json. For a photo this is a recording of the same
 * live run, shown when the browser cannot run the model. Strips of panels are always played from recordings
 * of the full-precision model: the compressed copy that runs in the browser does not read the panels.
 *
 * A run is a list of events: prompt, bind (binding heads at the question's key word), then for every
 * written token a look (gaze heads just before the token) and the token itself. The model writes its whole
 * answer first, which takes a second or two, and the page then shows the steps at reading speed.
 */
import {
  AutoModelForImageTextToText,
  AutoProcessor,
  Tensor,
  TextStreamer,
  env,
  load_image,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

const MODEL_ID = 'baulab/Qwen3-VL-2B-Instruct-GazeHeads-AllLayers-ONNX';
const GRAPH = 'onnx/decoder_model_merged_q4f16.onnx';
const GRAPH_VERSION = 1;                 // bump when model/decoder_model_merged_q4f16.onnx changes
const N_LAYERS = 28, N_HEADS = 16;
const PACE_MS = 900;                     // time the heat map stays up before the token is written
const MAX_NEW = { strip: 14, photo: 16 };

// Head sets of the 2B, found with the paper's search on strips of panels (layer, head).
const HEADS = {
  binding: [[13, 6], [11, 13], [12, 5], [13, 14], [11, 11], [14, 15], [11, 2], [14, 14]],
  gaze: [[22, 12], [21, 13], [22, 9], [19, 5], [17, 12], [17, 8], [24, 5], [17, 1], [24, 7], [21, 1], [26, 0],
    [21, 11], [25, 8], [19, 6], [25, 5], [17, 9]],
};
const COLOR = { binding: [0, 114, 178], gaze: [0, 158, 115] };
const BRIGHT = { binding: [40, 160, 255], gaze: [0, 235, 160] };   // the same two hues, lighter, for maps drawn over photos

const ASSETS_VERSION = 2;                // bump when files under images/ or runs/ change, so browsers drop old copies
const asset = (p) => new URL(p + (p.includes('?') ? '&' : '?') + 'a=' + ASSETS_VERSION, import.meta.url).href;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------------------------------------------------------------------- small helpers
function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const x = u16[i], s = (x & 0x8000) ? -1 : 1, e = (x >> 10) & 0x1f, f = x & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}
function packMap(map) {                   // Float32Array -> { m: max, d: base64 of bytes }
  let max = 0;
  for (const v of map) if (v > max) max = v;
  let bin = '';
  for (const v of map) bin += String.fromCharCode(max ? Math.round(255 * v / max) : 0);
  return { m: +max.toPrecision(4), d: btoa(bin) };
}
function unpackMap(p) {
  const bin = atob(p.d), out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) / 255 * p.m;
  return out;
}
/** Mean attention of a head set to each image token, from gaze_attn (28, 16, 1, T). */
function headMap(attn, T, heads, imgStart, nImg) {
  const map = new Float32Array(nImg);
  for (const [l, h] of heads) {
    const base = (l * N_HEADS + h) * T + imgStart;
    for (let j = 0; j < nImg; j++) map[j] += attn[base + j];
  }
  for (let j = 0; j < nImg; j++) map[j] /= heads.length;
  return map;
}

// ---------------------------------------------------------------------------- the live model
class Live {
  constructor() { this.attn = null; this.onStep = null; }

  async load(onProgress) {
    // Our decoder graph replaces the published one; the weight file it points to is unchanged.
    const hfFetch = env.fetch;
    env.fetch = (url, opts) => {
      const u = typeof url === 'string' ? url : (url?.url ?? String(url));
      if (u.includes(MODEL_ID) && u.endsWith(GRAPH)) return fetch(asset(`model/decoder_model_merged_q4f16.onnx?v=${GRAPH_VERSION}`), opts);
      return hfFetch(url, opts);
    };
    try {                                 // never reuse a cached copy of the graph under the published name
      const cache = await caches.open(env.cacheKey ?? 'transformers-cache');
      for (const k of await cache.keys()) if (k.url.includes(MODEL_ID) && k.url.endsWith(GRAPH)) await cache.delete(k);
    } catch { /* no Cache API: nothing to clear */ }

    this.processor = await AutoProcessor.from_pretrained(MODEL_ID);
    this.config = await (await fetch(`https://huggingface.co/${MODEL_ID}/resolve/main/config.json`)).json();
    const files = new Map();
    this.model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
      dtype: { embed_tokens: 'q4f16', vision_encoder: 'q4f16', decoder_model_merged: 'q4f16' },
      device: 'webgpu',
      progress_callback: (p) => {
        if (p.status !== 'progress' || !p.file || !p.total) return;
        files.set(p.file, p);
        let loaded = 0, total = 0;
        for (const f of files.values()) { loaded += f.loaded ?? 0; total += f.total ?? 0; }
        onProgress(loaded, total);
      },
    });
    const decoder = Object.values(this.model.sessions).find((s) => s.inputNames?.includes?.('gaze_sign'));
    if (!decoder || !decoder.outputNames?.includes?.('gaze_attn')) throw new Error('the decoder has no attention output');
    this.attach(decoder);
    this.model.encode_image = async () => new Tensor('float32', this.embeds, this.embedsDims);
    window.addEventListener('beforeunload', () => { try { this.model.dispose(); } catch { /* closing */ } });
  }

  /** Feed the two unused steering inputs as zeros, and take the attention output of every decoder run. */
  attach(session) {
    const hidden = ['gaze_head_mask', 'gaze_sign'];
    const names = [...session.inputNames];
    Object.defineProperty(session, 'inputNames', { get: () => names.filter((n) => !hidden.includes(n)), configurable: true });
    const run = session.run.bind(session);
    const self = this;
    session.run = async function (feeds, ...rest) {
      const am = feeds.attention_mask, T = Number(am.dims[am.dims.length - 1]), Ctor = am.constructor;
      const out = await run({
        ...feeds,
        gaze_head_mask: new Ctor('float16', new Uint16Array(N_LAYERS * N_HEADS), [N_LAYERS, N_HEADS]),
        gaze_sign: new Ctor('float16', new Uint16Array(T), [T]),
      }, ...rest);
      const a = out.gaze_attn;
      if (a) {
        const data = a.location === 'gpu-buffer' ? await a.getData() : a.data;
        delete out.gaze_attn;
        self.attn = { data, T };
        try { a.dispose?.(); } catch { /* released */ }
        if (self.onStep) await self.onStep(self.attn);
      }
      return out;
    };
  }

  /** Run one question on one image. Calls emit(event) for every step, as fast as the model goes; stops when stop() is true. */
  async run(item, question, emit, stop) {
    const { processor, model } = this;
    const buf = await (await fetch(asset(`images/${item.embeds.file}`))).arrayBuffer();
    this.embeds = f16ToF32(new Uint16Array(buf));
    this.embedsDims = item.embeds.dims;
    const image = await load_image(asset(`images/${item.file}`));
    const text = processor.apply_chat_template(
      [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: question.text }] }],
      { add_generation_prompt: true }) + 'Answer:';
    const inputs = await processor(text, image);
    const ids = Array.from(inputs.input_ids.data, Number), N = ids.length;
    const tk = processor.tokenizer;
    const piece = (id) => tk.decode([id], { skip_special_tokens: false });

    const imageId = Number(this.config.image_token_id);
    const imgStart = ids.indexOf(imageId), nImg = ids.lastIndexOf(imageId) - imgStart + 1;
    if (nImg !== item.embeds.dims[0]) throw new Error(`image tokens ${nImg} do not match the saved features ${item.embeds.dims[0]}`);

    // The question's tokens sit between the image and the end of the user turn.
    const qStart = imgStart + nImg + 1;
    let qEnd = qStart;
    while (qEnd < N && piece(ids[qEnd]) !== '<|im_end|>') qEnd += 1;
    const qTokens = ids.slice(qStart, qEnd).map(piece);
    // The key word (the shape, or the kind of object): the token where its last mention ends.
    const joined = qTokens.join(''), at = joined.toLowerCase().lastIndexOf(question.word.toLowerCase());
    let word = -1, seen = 0;
    for (let i = 0; i < qTokens.length && at >= 0; i++) {
      seen += qTokens[i].length;
      if (seen >= at + question.word.length) { word = i; break; }
    }
    if (word < 0) throw new Error('the key word is not in the question');
    emit({ type: 'prompt', tokens: [...qTokens, 'Answer', ':'], word });

    const cut = (n) => ({
      input_ids: inputs.input_ids.slice(null, [0, n]),
      attention_mask: inputs.attention_mask.slice(null, [0, n]),
      image_grid_thw: inputs.image_grid_thw,
    });
    const quiet = { max_new_tokens: 1, do_sample: false, return_dict_in_generate: true };

    // 1. Read the picture and the question up to the key word. Binding heads act at that token.
    this.onStep = null;
    const k = qStart + word;
    const a = await model.generate({ ...cut(k + 1), pixel_values: inputs.pixel_values, ...quiet });
    emit({ type: 'bind', map: headMap(this.attn.data, this.attn.T, HEADS.binding, imgStart, nImg) });
    if (stop()) return;

    // 2. Read the rest of the prompt, up to the token before the colon.
    // These tokens are all text, so the step that places image features into the prompt is skipped.
    const merge = model._merge_input_ids_with_image_features;
    model._merge_input_ids_with_image_features = ({ inputs_embeds, attention_mask }) => ({ inputs_embeds, attention_mask });
    let b;
    try {
      b = await model.generate({ ...cut(N - 1), past_key_values: a.past_key_values, ...quiet });
    } finally {
      model._merge_input_ids_with_image_features = merge;
    }

    // 3. Write the answer. Before every token, show where the gaze heads look.
    this.onStep = ({ data, T }) => {
      emit({ type: 'look', map: headMap(data, T, HEADS.gaze, imgStart, nImg) });
    };
    class Tokens extends TextStreamer {
      put(value) {
        if (!this.started) { this.started = true; return; }       // the first call carries the prompt
        if (stop()) throw new Error('__stopped__');
        const t = tk.decode([Number(value[0][0])], { skip_special_tokens: true });
        emit({ type: 'token', text: t });
        if (t.includes('\n')) throw new Error('__stopped__');
      }
      end() {}
    }
    try {
      const res = await model.generate({
        ...inputs, past_key_values: b.past_key_values,
        max_new_tokens: MAX_NEW[item.kind], do_sample: false,
        streamer: new Tokens(tk, { skip_prompt: true }), return_dict_in_generate: true,
      });
      try { res.past_key_values?.dispose?.(); } catch { /* best effort */ }
    } catch (err) {
      if (err.message !== '__stopped__') throw err;
    } finally {
      this.onStep = null;
    }
    emit({ type: 'end' });
  }
}

// ---------------------------------------------------------------------------- recorded runs
async function loadRecorded(item, qi) {
  const rec = await (await fetch(asset(`runs/${item.name}__${qi}.json`))).json();
  return rec.events.map((e) => (e.map ? { ...e, map: unpackMap(e.map) } : e));
}
/**
 * On a photo the gaze heads spread over the whole picture, and a few image tokens draw attention at every
 * step. So for photos the page draws, at each step, the attention above the average of all steps of this
 * answer: what is special about this step. Strips are drawn as they are.
 */
function aboveAverage(events) {
  const looks = events.filter((e) => e.type === 'look');
  if (looks.length < 2) return;
  const n = looks[0].map.length, mean = new Float32Array(n);
  for (const e of looks) for (let j = 0; j < n; j++) mean[j] += e.map[j] / looks.length;
  for (const e of looks) e.rel = e.map.map((v, j) => Math.max(0, v - mean[j]));
}

// ---------------------------------------------------------------------------- the page
const STYLES = `
.plandemo { text-align: left; }
.plandemo .pd-controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 6px 0 12px; }
.plandemo select, .plandemo button { font: inherit; font-size: 15px; padding: 6px 12px; border-radius: 6px;
  border: 1.5px solid #888; background: #f5f5f5; color: #111; }
.plandemo button { cursor: pointer; }
.plandemo button:disabled, .plandemo select:disabled { opacity: .5; cursor: default; }
.plandemo .pd-run { background: #0072B2; border-color: #0072B2; color: #fff; font-weight: 700; margin-left: auto; }
.plandemo .pd-stage { position: relative; margin: 0 auto; border: 1px solid #bbb; border-radius: 6px;
  overflow: hidden; background: #fff; max-width: 100%; }
.plandemo .pd-stage img { display: block; width: 100%; filter: grayscale(.75) brightness(1.06); }
.plandemo .pd-stage.pd-photo img { filter: grayscale(1) brightness(.78); }   /* a dim gray photo, so the only color is the map */
.plandemo .pd-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none;
  transition: opacity .25s; }
.plandemo .pd-say { min-height: 3.2em; margin: 10px 0 6px; font-size: 15px; line-height: 1.45; }
.plandemo .pd-rowlabel { font-size: 12px; color: #666; letter-spacing: .04em; text-transform: uppercase; margin-top: 8px; }
.plandemo .pd-toks { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0;
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 14px; }
.plandemo .pd-tok { border: 1px solid #b9b9b9; border-radius: 4px; background: #f7f7f7; padding: 2px 7px;
  white-space: pre; color: #111; }
.plandemo .pd-tok.pd-has { cursor: pointer; }
.plandemo .pd-tok.pd-bind { border-color: #0072B2; background: #dcecf6; font-weight: 700; }
.plandemo .pd-tok.pd-gaze { border-color: #009E73; background: #dcf1ea; font-weight: 700; }
.plandemo .pd-tok.pd-on { outline: 2.5px solid #111; outline-offset: 1px; }
.plandemo .pd-tok.pd-wait { border-style: dashed; color: #888; background: #fff; }
.plandemo .pd-note { font-size: 13px; color: #555; margin: 8px 0 0; }
.plandemo .pd-credit { font-size: 12px; color: #777; margin-top: 4px; min-height: 1.2em; }
.plandemo .pd-load { margin: 10px 0; font-size: 14px; color: #333; }
.plandemo .pd-load progress { width: min(380px, 60vw); height: 14px; vertical-align: middle; accent-color: #0072B2; }
@media (max-width: 600px) { .plandemo .pd-run { margin-left: 0; } }
`;

const TEMPLATE = `
<div class="plandemo">
  <div class="pd-load" hidden><span class="pd-loadmsg"></span> <progress max="100"></progress> <span class="pd-loadnum"></span></div>
  <div class="pd-controls">
    <label>Image <select class="pd-image"></select></label>
    <label>Ask for <select class="pd-question"></select></label>
    <button class="pd-run" disabled>&#9654;&nbsp; Run</button>
  </div>
  <div class="pd-stage"><img class="pd-img" alt="the image given to the model" /><canvas class="pd-heat"></canvas></div>
  <div class="pd-credit"></div>
  <div class="pd-say"></div>
  <div class="pd-rowlabel">Question tokens</div>
  <div class="pd-toks pd-q"></div>
  <div class="pd-rowlabel">Tokens the model writes</div>
  <div class="pd-toks pd-a"></div>
  <p class="pd-note"></p>
</div>`;

export async function start(container, opts = {}) {
  if (!document.getElementById('plandemo-styles')) {
    const st = document.createElement('style');
    st.id = 'plandemo-styles';
    st.textContent = STYLES;
    document.head.appendChild(st);
  }
  const mount = document.createElement('div');
  mount.innerHTML = TEMPLATE;
  container.appendChild(mount);
  const $ = (c) => mount.querySelector('.' + c);

  const index = await (await fetch(asset('images/index.json'))).json();
  const items = {};
  const state = { live: null, item: null, running: false, stopFlag: false, events: [], all: [], shown: null };

  $('pd-image').innerHTML = index.map((it) => `<option value="${it.name}">${esc(it.label)}</option>`).join('');

  // ---- drawing
  function drawHeat(map, kind) {
    const cv = $('pd-heat'), it = state.item;
    if (!map) { cv.style.opacity = 0; return; }
    const [, gh, gw] = it.embeds.grid_thw, h = gh / 2, w = gw / 2;
    cv.width = w; cv.height = h;
    const photo = it.kind === 'photo';
    const ctx = cv.getContext('2d'), img = ctx.createImageData(w, h), [r, g, b] = (photo ? BRIGHT : COLOR)[kind];
    let max = 0;
    for (const v of map) if (v > max) max = v;
    for (let j = 0; j < map.length; j++) {
      const v = max ? map[j] / max : 0;
      // Over a photo the map is drawn stronger: values from about half the peak up are close to solid color.
      const a = photo ? 0.88 * Math.pow(Math.min(1, 1.8 * v), 0.6) : Math.pow(v, 0.5);
      img.data.set([r, g, b, Math.round(255 * a)], 4 * j);
    }
    ctx.putImageData(img, 0, 0);
    // One image token is one cell. A blur of about one cell turns single tokens into soft spots.
    cv.style.filter = `blur(${($('pd-stage').clientWidth / w * 0.55).toFixed(1)}px) saturate(1.6)`;
    cv.style.opacity = 1;
  }
  /** Share of a map on the image, and the region with the highest attention per image token. */
  function shares(map) {
    const it = state.item, [, gh, gw] = it.embeds.grid_thw, h = gh / 2, w = gw / 2;
    let total = 0;
    for (const v of map) total += v;
    const by = new Map();                 // region name -> set of grid cells (an object can have several boxes)
    it.regions.forEach((reg, p) => {
      const key = it.kind === 'strip' ? `panel ${p + 1}` : `the ${reg.name}`;
      const cells = by.get(key) ?? by.set(key, new Set()).get(key);
      const [x0, y0, x1, y1] = reg.box;
      for (let rr = 0; rr < h; rr++) {
        const cy = (rr + 0.5) * it.height / h;
        if (cy < y0 || cy > y1) continue;
        for (let cc = 0; cc < w; cc++) {
          const cx = (cc + 0.5) * it.width / w;
          if (cx >= x0 && cx <= x1) cells.add(rr * w + cc);
        }
      }
    });
    let top = null, best = -1;
    for (const [key, cells] of by) {
      if (!cells.size) continue;
      let s = 0;
      for (const c of cells) s += map[c];
      if (s / cells.size > best) { best = s / cells.size; top = key; }
    }
    return { total, top };
  }
  const pct = (v) => `${Math.round(100 * v)}%`;

  function show(i) {                      // show the map recorded with event i (a bind or a look)
    const ev = state.events[i];
    if (!ev || !ev.map) return;
    state.shown = i;
    mount.querySelectorAll('.pd-tok.pd-on').forEach((el) => el.classList.remove('pd-on'));
    mount.querySelector(`.pd-tok[data-ev="${i}"]`)?.classList.add('pd-on');
    const shown = ev.rel ?? ev.map;
    const { total } = shares(ev.map), { top } = shares(shown);
    const where = !top || !total ? '' : ev.rel ? ` Green marks where they look more than their average over this answer, most of all on <b>${top}</b>.`
      : ` It is highest on <b>${top}</b>.`;
    if (ev.type === 'bind') {
      drawHeat(ev.map, 'binding');
      $('pd-say').innerHTML = `<b style="color:#0072B2">Bind.</b> The model has read the question up to <b>${esc(state.word.trim())}</b>. `
        + `The ${HEADS.binding.length} binding heads put ${pct(total)} of their attention on the image.${where}`;
    } else {
      drawHeat(shown, 'gaze');
      const next = ev.next === undefined ? 'the next token' : `<b>${esc(ev.next.trim() || ev.next)}</b>`;
      $('pd-say').innerHTML = `<b style="color:#009E73">Gaze.</b> Before the model writes ${next}, `
        + `the ${HEADS.gaze.length} gaze heads put ${pct(total)} of their attention on the image.${where}`;
    }
  }

  function handle(ev) {
    const i = state.events.push(ev) - 1;
    if (ev.type === 'prompt') {
      state.word = ev.tokens[ev.word];
      $('pd-q').innerHTML = ev.tokens.map((t, j) => `<span class="pd-tok" data-q="${j}">${esc(t.trim() || t)}</span>`).join('');
    } else if (ev.type === 'bind') {
      const el = mount.querySelector(`.pd-tok[data-q="${state.events[0].word}"]`);
      el.classList.add('pd-bind', 'pd-has');
      el.dataset.ev = i;
      show(i);
    } else if (ev.type === 'look') {
      const el = document.createElement('span');
      el.className = 'pd-tok pd-wait';
      el.textContent = '?';
      el.dataset.ev = i;
      $('pd-a').appendChild(el);
      show(i);
    } else if (ev.type === 'token') {
      const look = state.events.findLastIndex((e) => e.type === 'look' && e.next === undefined);
      if (look < 0) return;
      state.events[look].next = ev.text;
      const el = mount.querySelector(`.pd-a .pd-tok[data-ev="${look}"]`);
      el.classList.remove('pd-wait');
      el.classList.add('pd-gaze', 'pd-has');
      el.textContent = ev.text.trim() || ev.text;
      if (state.shown === look) show(look);
    }
  }
  mount.addEventListener('click', (e) => {
    const el = e.target.closest('.pd-tok.pd-has');
    if (el && !state.running) show(Number(el.dataset.ev));
  });

  // ---- choosing an image and a question
  async function choose(name) {
    if (!items[name]) items[name] = await (await fetch(asset(`images/${name}.json`))).json();
    const it = state.item = items[name];
    $('pd-img').src = asset(`images/${it.file}`);
    $('pd-stage').classList.toggle('pd-photo', it.kind === 'photo');
    $('pd-stage').style.width = it.kind === 'photo' ? `min(100%, ${Math.round(380 * it.width / it.height)}px)` : '100%';
    $('pd-question').innerHTML = it.questions.map((q, i) => `<option value="${i}">${esc(q.label)}</option>`).join('');
    $('pd-credit').innerHTML = it.credit
      ? `Photo from ${esc(it.credit.source)} (<a href="${it.credit.url}" target="_blank" rel="noopener">Flickr</a>, ${esc(it.credit.license)}).`
      : 'This strip is a recording of the full-size model. The compressed copy that runs in a browser reads photos, but not these panels.';
    reset();
  }
  function reset() {
    state.events = []; state.shown = null;
    drawHeat(null);
    $('pd-q').innerHTML = ''; $('pd-a').innerHTML = '';
    $('pd-say').textContent = 'Press Run. The model reads the image and the question, then writes its answer one token at a time.';
  }
  $('pd-image').onchange = async () => { state.stopFlag = true; await idle(); await choose($('pd-image').value); };
  $('pd-question').onchange = async () => { state.stopFlag = true; await idle(); reset(); };
  const idle = async () => { while (state.running) await sleep(50); };

  async function run() {
    if (state.running) { state.stopFlag = true; return; }
    reset();
    state.running = true; state.stopFlag = false;
    $('pd-run').innerHTML = '&#9632;&nbsp; Stop';
    const it = state.item, qi = Number($('pd-question').value), stop = () => state.stopFlag;
    try {
      let events = [];
      if (state.live && it.embeds.file) {            // the model answers first; the page then shows its steps slowly
        $('pd-say').textContent = 'The model is reading the image and writing its answer…';
        await state.live.run(it, it.questions[qi], (e) => events.push(e), stop);
      } else {
        events = await loadRecorded(it, qi);
      }
      if (it.kind === 'photo') aboveAverage(events);
      state.all = events;                            // the full run, kept for recordAll
      // The last step is the model's stop token, which has no text. It is not shown.
      const last = events.findLastIndex((e) => e.type === 'token');
      const steps = last > 0 && events[last].text === '' ? events.filter((_, i) => i !== last && i !== last - 1) : events;
      for (const ev of steps) {
        if (stop()) break;
        handle(ev);
        if (ev.type === 'bind') await sleep(PACE_MS * 1.6);
        if (ev.type === 'look') await sleep(PACE_MS);
      }
    } catch (err) {
      console.error(err);
      $('pd-say').textContent = 'The run failed: ' + err.message;
    } finally {
      state.running = false;
      $('pd-run').innerHTML = '&#9654;&nbsp; Run';
    }
  }
  $('pd-run').onclick = run;

  // Used once, on a local copy of the site, to save the recordings (dev_server.py accepts the POST).
  async function recordAll(only) {
    for (const it0 of index) {
      if (it0.kind !== 'photo' || (only && !only.includes(it0.name))) continue;   // strips are recorded on a lab machine
      $('pd-image').value = it0.name;
      await choose(it0.name);
      for (let qi = 0; qi < state.item.questions.length; qi++) {
        $('pd-question').value = qi;
        await run();
        const events = state.all.map((e) => {
          const o = { ...e };
          if (o.map) o.map = packMap(o.map);
          delete o.next; delete o.rel;
          return o;
        });
        const answer = state.all.filter((e) => e.type === 'token').map((e) => e.text).join('');
        await fetch(`/__save/runs/${it0.name}__${qi}.json`, { method: 'POST', body: JSON.stringify({ image: it0.name, question: state.item.questions[qi].text, answer, events }) });
        console.log('[plandemo] saved', it0.name, qi, JSON.stringify(answer));
      }
    }
    return 'done';
  }
  window.__plandemo = { state, run, recordAll, show };

  // ---- start: everything plays at once from recordings; the model for the photos loads behind when the browser can run it
  await choose(index[0].name);
  $('pd-run').disabled = false;
  const STRIPS = 'The strips are recordings of the full-size model.';
  const recorded = () => {                 // no model here: every run plays from its recording
    state.live = null;
    $('pd-load').hidden = true;
    $('pd-note').textContent = '';
  };
  const phone = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  let adapter = null;
  try { adapter = opts.recorded || phone ? null : await navigator.gpu?.requestAdapter(); } catch { /* no WebGPU */ }
  if (!adapter) {
    recorded();
  } else {
    const live = new Live();
    $('pd-load').hidden = false;
    $('pd-loadmsg').textContent = 'Preparing the model for the photos…';
    $('pd-note').textContent = `Until the model is ready, the photos play as recordings. ${STRIPS}`;
    try {
      await live.load((loaded, total) => {
        $('pd-load').querySelector('progress').value = 100 * loaded / total;
        $('pd-loadmsg').textContent = 'Downloading the model for the photos, one time only:';
        $('pd-loadnum').textContent = `${(loaded / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`;
      });
      state.live = live;
      $('pd-load').hidden = true;
      $('pd-note').textContent = `On photos, the model runs in this browser tab. Nothing is sent to a server. ${STRIPS}`;
    } catch (err) {
      console.error(err);
      recorded();
    }
  }
}
