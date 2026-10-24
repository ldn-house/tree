// Tree effects viewer: three.js scene driven by effects/ running in Pyodide
// (see worker.js). Frames go through effects/_output.py exactly as on the
// device, so the power-cap indicator matches the firmware and render.py.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const N = 500;
const $ = (id) => document.getElementById(id);

// ─── Worker RPC ───────────────────────────────────────────────────────────────

const worker = new Worker("worker.js");
let nextId = 1;
const pending = new Map();
worker.onmessage = (e) => {
  const { id, ok, result, error } = e.data;
  const p = pending.get(id);
  pending.delete(id);
  ok ? p.resolve(result) : p.reject(new Error(error));
};
function call(type, args = {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...args });
  });
}

// ─── Scene ────────────────────────────────────────────────────────────────────

const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.enableDamping = true;

const VIEWS = {
  front: [0, 0.5, 2.1], // camera on +z, matching render.py's front panel
  side: [2.1, 0.5, 0], // camera on +x
  top: [0, 2.4, 0.001],
};
function setView(name) {
  camera.position.set(...VIEWS[name]);
  controls.target.set(0, name === "top" ? 0 : 0.5, 0);
  controls.update();
}
setView("front");

// Floor ring and trunk axis for orientation.
const ring = new THREE.Mesh(
  new THREE.RingGeometry(0.34, 0.345, 96),
  new THREE.MeshBasicMaterial({ color: 0x1a2230, side: THREE.DoubleSide })
);
ring.rotation.x = -Math.PI / 2;
scene.add(ring);
const axis = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]),
  new THREE.LineBasicMaterial({ color: 0x1a2230 })
);
scene.add(axis);

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.15, "rgba(255,255,255,0.8)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.18)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

const positions = new Float32Array(N * 3);
const colors = new Float32Array(N * 3);
const geom = new THREE.BufferGeometry();
geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
const glow = new THREE.Points(
  geom,
  new THREE.PointsMaterial({
    size: 0.12, map: glowTexture(), vertexColors: true, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
  })
);
scene.add(glow);
// Dim bulbs so unlit LEDs and the tree's shape stay visible.
const bulbs = new THREE.Points(
  geom.clone().deleteAttribute("color"),
  new THREE.PointsMaterial({ size: 0.008, color: 0x2a3140, sizeAttenuation: true })
);
scene.add(bulbs);

function setPositions(list) {
  for (let i = 0; i < N; i++) {
    let p = list && list[i];
    if (!p) {
      // Stand-in spiral for LEDs without coordinates (same as render.py).
      const h = i / N, a = i * 2.39996, r = 0.3 * (1 - h);
      p = [r * Math.cos(a), h, r * Math.sin(a)];
    }
    positions.set(p, i * 3);
  }
  geom.attributes.position.needsUpdate = true;
  bulbs.geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
}

// Output values are linear PWM; lift them like render.py does (1/2.2) so dim
// LEDs read the way they do to the eye.
const LIFT = new Float32Array(256).map((_, v) => Math.pow(v / 255, 1 / 2.2));
function setColors(bytes) {
  for (let i = 0; i < N * 3; i++) colors[i] = LIFT[bytes[i]];
  geom.attributes.color.needsUpdate = true;
}

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ─── State and UI ─────────────────────────────────────────────────────────────

const state = {
  effect: null, t0: performance.now(), brightness: 255, floorPct: 0, userFloorPct: 0, colourPct: 0, showRaw: false, spin: false,
  frames: 0, fpsT: performance.now(), fps: 0, ready: false, busy: false,
};
window.viewer = state; // handy for automation: viewer.last holds the latest frame stats

function showError(err) {
  $("error").textContent = err ? String(err.message || err) : "";
  if (err) console.error(err);
}

function paramControl(key, [def, lo, hi], value) {
  const isInt = [def, lo, hi].every(Number.isInteger);
  const step = isInt ? 1 : Math.pow(10, Math.floor(Math.log10((hi - lo) / 100)));
  const label = document.createElement("label");
  label.innerHTML = `${key} <output></output><input type="range" min="${lo}" max="${hi}" step="${step}">`;
  const input = label.querySelector("input");
  const out = label.querySelector("output");
  input.value = value;
  out.textContent = value;
  input.addEventListener("input", async () => {
    const v = isInt ? parseInt(input.value, 10) : parseFloat(input.value);
    out.textContent = isInt ? v : v.toFixed(2);
    await call("param", { key, value: v }).catch(showError);
  });
  return label;
}

async function selectEffect(name) {
  state.ready = false;
  showError(null);
  try {
    const { info } = await call("select", { name, seed: parseInt($("seed").value, 10) || 0 });
    state.effect = name;
    state.info = info;
    state.t0 = performance.now();
    const box = $("params");
    box.innerHTML = "";
    for (const [k, spec] of Object.entries(info.params)) box.append(paramControl(k, spec, info.values[k]));
    if (!Object.keys(info.params).length) box.innerHTML = '<div class="note">No params.</div>';
    const u = new URL(location);
    u.searchParams.set("effect", name);
    history.replaceState(null, "", u);
    state.ready = true;
  } catch (err) {
    showError(err);
  }
}

$("effect").addEventListener("change", (e) => selectEffect(e.target.value));
$("restart").addEventListener("click", () => selectEffect(state.effect));
$("raw").addEventListener("click", (e) => {
  state.showRaw = !state.showRaw;
  e.target.classList.toggle("on", state.showRaw);
});
$("spin").addEventListener("click", (e) => {
  state.spin = !state.spin;
  controls.autoRotate = state.spin;
  e.target.classList.toggle("on", state.spin);
});
for (const b of document.querySelectorAll("[data-view]")) b.addEventListener("click", () => setView(b.dataset.view));
// Non-zero brightness maps onto floor..100% in the output stage
// (effects/_output.py), as on the tree; show both. The floor is the higher
// of the brightness floor and the effect's colour floor.
function showBrightness() {
  const b = state.brightness;
  const f = state.floorPct;
  // 3 is Home Assistant's 1%: it and below give the floor.
  const level = b <= 0 ? 0 : f === 0 ? (b * 100) / 255 : b <= 3 ? f : f + ((b - 3) * (100 - f)) / 252;
  const why = state.colourPct > state.userFloorPct ? `colour floor; brightness floor ${state.userFloorPct}%` : "brightness floor";
  $("bri-out").textContent = `${b} (${Math.round((b / 255) * 100)}%; ${level.toFixed(1)}% with the ${f.toFixed(1)}% ${why})`;
}
$("bri").addEventListener("input", (e) => {
  state.brightness = parseInt(e.target.value, 10);
  showBrightness();
});
$("limit").addEventListener("input", async (e) => {
  $("limit-out").textContent = `${parseFloat(e.target.value).toFixed(1)} A`;
  await call("limit", { amps: parseFloat(e.target.value) }).catch(showError);
});
$("acc").addEventListener("input", async (e) => {
  const v = parseFloat(e.target.value);
  $("acc-out").textContent = v > 0 ? v.toFixed(3) : "off";
  await call("accuracy", { value: v }).catch(showError);
});
$("bri").dispatchEvent(new Event("input"));
$("limit-out").textContent = `${parseFloat($("limit").value).toFixed(1)} A`;

// ─── Frame loop ───────────────────────────────────────────────────────────────

async function requestFrame() {
  if (!state.ready || state.busy) return;
  state.busy = true;
  try {
    const t = performance.now() - state.t0;
    const f = await call("frame", { t, brightness: state.brightness });
    state.last = { t: Math.round(t), capped: f.capped, factor: f.factor, rawTotal: f.rawTotal, outTotal: f.outTotal, budget: f.budget, amps: f.amps, limitA: f.limitA };
    setColors(state.showRaw ? f.raw : f.out);
    if (f.effectivePct !== state.floorPct || f.colourPct !== state.colourPct) {
      state.floorPct = f.effectivePct;
      state.colourPct = f.colourPct;
      showBrightness();
    }
    const meter = $("meter");
    meter.firstElementChild.style.width = `${Math.min(100, (100 * f.amps) / f.limitA)}%`;
    meter.classList.toggle("capped", f.capped);
    $("cap").className = f.capped ? "on" : "off";
    $("power").textContent = `${f.amps.toFixed(2)} A of ${f.limitA.toFixed(2)} A` +
      (f.capped ? ` · scaled ×${f.factor.toFixed(2)}` : "");
    state.frames++;
  } catch (err) {
    state.ready = false;
    showError(err);
  } finally {
    state.busy = false;
  }
}

function animate(now) {
  requestAnimationFrame(animate);
  requestFrame();
  controls.update();
  renderer.render(scene, camera);
  if (now - state.fpsT > 1000) {
    state.fps = (state.frames * 1000) / (now - state.fpsT);
    state.frames = 0;
    state.fpsT = now;
    if (state.effect)
      $("status").textContent = `${state.effect}  t=${((now - state.t0) / 1000).toFixed(1)}s  ${state.fps.toFixed(0)} fps` +
        (state.showRaw ? "  (raw frame)" : "");
  }
}
requestAnimationFrame(animate);

// ─── Hot reload ───────────────────────────────────────────────────────────────
// serve.py exposes /api/mtimes; when a file in effects/ changes, re-sync the
// package into Pyodide and restart the current effect.

let mtimes = null;
async function pollChanges() {
  try {
    const r = await fetch("../api/mtimes", { cache: "no-store" });
    if (!r.ok) return;
    const next = await r.json();
    const changed = mtimes && Object.keys(next).filter((k) => next[k] !== mtimes[k]);
    if (!mtimes) $("files").textContent = `Watching ${Object.keys(next).length} files in effects/ (hot reload on)`;
    mtimes = next;
    if (changed && changed.length) {
      $("files").textContent = `Reloaded ${changed.join(", ")} at ${new Date().toLocaleTimeString()}`;
      const { names } = await call("reload");
      fillPicker(names);
      await selectEffect(names.includes(state.effect) ? state.effect : names[0]);
    }
  } catch {
    /* static hosting: no hot reload */
  } finally {
    setTimeout(pollChanges, 700);
  }
}

function fillPicker(names) {
  const sel = $("effect");
  const current = sel.value;
  sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  sel.value = names.includes(current) ? current : sel.value;
  sel.disabled = false;
}

(async () => {
  try {
    const { names, coordsPath, positions: pos, limitA, maPerChannel, floorPct, accuracy } = await call("init", { n: N });
    state.floorPct = floorPct;
    state.userFloorPct = floorPct;
    showBrightness();
    $("acc").value = accuracy;
    $("acc-out").textContent = accuracy > 0 ? accuracy.toFixed(3) : "off";
    $("limit").value = limitA;
    $("limit-out").textContent = `${limitA.toFixed(1)} A`;
    $("limit").title = `Estimated at ${maPerChannel} mA per channel (effects/_output.py)`;
    setPositions(pos);
    $("status").textContent = coordsPath ? `coords: ${coordsPath}` : "no coordinates; showing a stand-in spiral";
    fillPicker(names);
    const wanted = new URL(location).searchParams.get("effect");
    const first = names.includes(wanted) ? wanted : names[0];
    $("effect").value = first;
    await selectEffect(first);
    pollChanges();
  } catch (err) {
    showError(err);
    $("status").textContent = "failed to start";
  }
})();
