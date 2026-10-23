// Runs effects/ in Pyodide, off the main thread. Loads the package files
// straight from the server and calls them through the effect API: no mocks
// and no source rewriting.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");

const ROOT = new URL("../", self.location.href);
let py = null;
let host = null;

const HOST_PY = `
import sys, random
import effects
from effects import _coords, _output

class Host:
    def __init__(self, n, coords_text, legacy):
        self.n = n
        coords = None
        if coords_text:
            coords = _coords.parse(coords_text)
            if legacy:
                coords = _coords.adapt_legacy(coords)
            coords = _coords.pad(coords, n)
        self.coords = coords
        self.raw = bytearray(3 * n)
        self.out = bytearray(3 * n)
        self.stage = _output.Output(n)
        self.effect = None
        self.params = {}

    def names(self):
        return list(effects.EFFECTS)

    def select(self, name, seed, params):
        mod = effects.load(name)
        self.params = effects.defaults(mod)
        self.params.update(params)
        ctx = effects.Ctx(self.n, self.coords, self.params, random.Random(seed))
        self.effect = mod.Effect(ctx)
        self.name = name
        spec = {k: list(v) for k, v in getattr(mod, "PARAMS", {}).items()}
        return {"params": spec, "values": dict(self.params),
                "needs_coords": bool(getattr(mod, "NEEDS_COORDS", False)),
                "duration_ms": getattr(mod, "DURATION_MS", 0)}

    def set_param(self, key, value):
        self.params[key] = value

    def limits(self):
        return [_output.LIMIT_A, _output.MA_PER_CHANNEL, self.stage.floor]

    def set_limit(self, limit_a):
        self.stage.budget = _output.budget_for_current(limit_a)

    def frame(self, t_ms, brightness):
        self.effect.render(int(t_ms), self.raw)
        self.stage.process(self.raw, self.out, int(brightness))
        out_total = sum(self.out)
        # Estimated current from the bytes actually output (after gamma,
        # white balance and dithering), as the firmware reports it.
        amps = _output.current_a(out_total)
        limit = _output.current_a(self.stage.budget)
        return (bytes(self.out), bytes(self.raw), self.stage.capped, self.stage.factor,
                self.stage.raw_total, out_total, self.stage.budget, amps, limit)

    def reload(self):
        for k in [k for k in sys.modules if k == "effects" or k.startswith("effects.")]:
            del sys.modules[k]
        global effects, _coords, _output
        import effects
        from effects import _coords, _output
        self.stage = _output.Output(self.n, budget=self.stage.budget)
`;

async function fetchText(path) {
  let r;
  try {
    r = await fetch(new URL(path, ROOT), { cache: "no-store" });
  } catch (err) {
    throw new Error(`${path}: ${err.message}`);
  }
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.text();
}

// Write effects/__init__.py, the shared modules and every registered effect
// into Pyodide's filesystem. Returns {path: source} for change detection.
async function syncEffects() {
  py.FS.mkdirTree("/home/pyodide/effects");
  const init = await fetchText("effects/__init__.py");
  const names = py.runPython(
    `ns = {}\nexec(compile(${JSON.stringify(init)}, "effects/__init__.py", "exec"), ns)\nlist(ns["EFFECTS"])`
  ).toJs();
  const files = ["__init__", "_util", "_output", "_coords", ...names];
  const sources = {};
  await Promise.all(
    files.map(async (f) => {
      sources[f] = await fetchText(`effects/${f}.py`);
    })
  );
  for (const f of files) py.FS.writeFile(`/home/pyodide/effects/${f}.py`, sources[f]);
  return names;
}

async function loadCoords() {
  for (const [path, legacy] of [
    ["coordinates/coords.txt", false],
    ["coordinates/coords_compact.txt", true],
  ]) {
    try {
      return { text: await fetchText(path), legacy, path };
    } catch (e) {
      /* try the next one */
    }
  }
  return { text: "", legacy: false, path: null };
}

function reply(id, payload, transfer) {
  self.postMessage({ id, ...payload }, transfer || []);
}

const handlers = {
  async init({ n }) {
    py = await loadPyodide();
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    const names = await syncEffects();
    const coords = await loadCoords();
    py.runPython(HOST_PY);
    host = py.globals.get("Host")(n, coords.text, coords.legacy);
    py.globals.set("H", host);
    const pos = py.runPython("None if H.coords is None else [list(c) if c else None for c in H.coords]");
    const limits = host.limits().toJs();
    return { names, coordsPath: coords.path, positions: pos ? pos.toJs() : null, limitA: limits[0], maPerChannel: limits[1], floorPct: limits[2] };
  },
  async select({ name, seed, params }) {
    const proxy = host.select(name, seed, py.toPy(params || {}));
    const info = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy();
    return { info };
  },
  async param({ key, value }) {
    host.set_param(key, value);
    return {};
  },
  async limit({ amps }) {
    host.set_limit(amps);
    return {};
  },
  async frame({ t, brightness }) {
    const proxy = host.frame(t, brightness);
    const r = proxy.toJs();
    proxy.destroy();
    const out = new Uint8Array(r[0]);
    const raw = new Uint8Array(r[1]);
    return {
      out, raw, capped: r[2], factor: r[3] / 4096, rawTotal: r[4], outTotal: r[5], budget: r[6], amps: r[7], limitA: r[8],
      __transfer: [out.buffer, raw.buffer],
    };
  },
  async reload() {
    const names = await syncEffects();
    host.reload();
    return { names };
  },
};

self.onmessage = async (e) => {
  const { id, type, ...args } = e.data;
  try {
    const result = await handlers[type](args);
    const transfer = result.__transfer;
    delete result.__transfer;
    reply(id, { ok: true, result }, transfer);
  } catch (err) {
    reply(id, { ok: false, error: String(err && err.message ? err.message : err) });
  }
};
