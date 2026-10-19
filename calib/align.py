"""Recover which LED each still frame shows when the capture loop raced.

The sequential captures were taken by setting an LED over HTTP, waiting a
fixed time and reading the webcam. The firmware applies changes on a 100 ms
poll and the webcam buffers frames, so a frame often shows an earlier LED: it
repeats the previous state (lag grows) or misses one (lag shrinks), and each
view drifts differently. Up to 20 LEDs of drift by the end of a capture was
measured on the real data.

Geometry fixes the relative drift. For each view we look for a monotone
mapping frame -> LED with steps of +1 (normal), 0 (repeat) or +2/+3 (skip)
that best agrees with the projections of the current 3D solution (Viterbi),
then re-solve the 3D with the new correspondences, growing the range of LEDs
along the wire as the solution firms up. Frame 0 is dark in every real view,
which pins the absolute start at lag 1.
"""

from dataclasses import dataclass

import numpy as np

from .solve import Solution, Solver, ViewObs, project


@dataclass
class FrameDets:
    name: str
    width: int
    height: int
    uv: np.ndarray     # (F, 2) detection per frame, NaN if none
    conf: np.ndarray   # (F,)


STEP_COST = {0: 3.0, 1: 0.0, 2: 4.0, 3: 8.0}


def viterbi(frames: FrameDets, pred: np.ndarray, start_state: int, sigma_px: float = 6.0,
            cap: float = 9.0, min_conf: float = 0.3, unknown_cost: float = 1.5) -> np.ndarray:
    """Most likely LED per frame. pred: (n, 2) predicted LED positions (NaN = unknown).

    State s = LED index + 1; s = 0 means the frame shows nothing (before LED 0).
    Returns (F,) LED index per frame (-1 = none).
    """
    F, n = len(frames.uv), len(pred)
    S = n + 1
    det = np.isfinite(frames.uv).all(1) & (frames.conf >= min_conf)
    w = np.clip(frames.conf, 0, 1)
    emis = np.zeros((F, S))
    known = np.isfinite(pred).all(1)
    for f in np.nonzero(det)[0]:
        d2 = ((pred - frames.uv[f]) ** 2).sum(1) / sigma_px ** 2
        e = np.where(known, np.minimum(d2, cap), unknown_cost)
        emis[f, 1:] = w[f] * e
        emis[f, 0] = w[f] * cap
    # Identical consecutive detections are strong evidence of a repeated frame.
    same = np.zeros(F, bool)
    same[1:] = det[1:] & det[:-1] & (np.linalg.norm(frames.uv[1:] - frames.uv[:-1], axis=1) < 1.5)
    INF = 1e18
    D = np.full(S, INF)
    D[start_state] = emis[0, start_state]
    back = np.zeros((F, S), np.int8)
    for f in range(1, F):
        best = np.full(S, INF)
        arg = np.zeros(S, np.int8)
        for step, c in STEP_COST.items():
            c = c - 3.0 if (same[f] and step == 0) else c + (2.0 if same[f] else 0.0)
            cand = np.full(S, INF)
            cand[step:] = D[:S - step] + c
            better = cand < best
            best[better] = cand[better]
            arg[better] = step
        D = best + emis[f]
        back[f] = arg
    s = int(np.argmin(D))
    states = np.zeros(F, int)
    for f in range(F - 1, -1, -1):
        states[f] = s
        s -= int(back[f, s])
    return states - 1


def observations(frames: list[FrameDets], mapping: list[np.ndarray], n: int, max_led: int) -> list[ViewObs]:
    views = []
    for fd, m in zip(frames, mapping):
        uv = np.full((n, 2), np.nan)
        conf = np.zeros(n)
        for f, led in enumerate(m):
            if 0 <= led < min(n, max_led) and np.isfinite(fd.uv[f]).all() and fd.conf[f] > conf[led]:
                uv[led] = fd.uv[f]
                conf[led] = fd.conf[f]
        views.append(ViewObs(fd.name, fd.width, fd.height, uv, conf))
    return views


def predict(sol: Solution, views: list[ViewObs]) -> list[np.ndarray]:
    out = []
    for i, v in enumerate(views):
        f = sol.focal[v.device]
        k1, k2 = sol.dist[v.device]
        X = sol.points
        ok = np.isfinite(X).all(1)
        uv = np.full((len(X), 2), np.nan)
        if ok.any():
            p, z = project(X[ok], np.array(sol.rvecs[i]), np.array(sol.tvecs[i]), f, k1, k2, v.width, v.height)
            p[z <= 0] = np.nan
            uv[ok] = p
        out.append(uv)
    return out


def align_and_solve(frames: list[FrameDets], n: int = 500, start_lag: int = 1, first: int = 120,
                    grow: int = 80, final_rounds: int = 3, verbose: bool = False,
                    wire_prior: bool = True, init_mapping: list[np.ndarray] | None = None
                    ) -> tuple[Solution, list[np.ndarray], list[str]]:
    """Alternate Viterbi index alignment and bundle adjustment along the wire.

    With init_mapping (e.g. from initial_mapping), start from it over the
    whole string; otherwise start from f - start_lag and grow along the wire.
    """
    log = []
    F = len(frames[0].uv)
    mapping = init_mapping if init_mapping is not None else [np.arange(F) - start_lag for _ in frames]
    if init_mapping is not None:
        first = n
    start_state = 1 - start_lag
    sol = None
    K = first
    rounds_at_full = 0
    while True:
        views = observations(frames, mapping, n, K)
        sol = Solver(views).solve(wire_prior=False, init=sol)
        preds = predict(sol, views)
        for p in preds:
            p[K:] = np.nan  # beyond the solved range the projection is not trusted
        new = [viterbi(fd, p, start_state) for fd, p in zip(frames, preds)]
        changed = sum(int((a != b).sum()) for a, b in zip(new, mapping))
        mapping = new
        msg = (f"LEDs < {K}: median reproj {sol.cost_px:.2f}px, mapping changed on {changed} frames, "
               f"final offsets {[int(F - 1 - start_lag - m[-1]) for m in mapping]}")
        log.append(msg)
        if verbose:
            print("  " + msg)
        if K >= n:
            rounds_at_full += 1
            if rounds_at_full >= final_rounds or changed == 0:
                break
        K = min(n, K + grow)
    views = observations(frames, mapping, n, n)
    sol = Solver(views).solve(wire_prior=wire_prior, init=sol)
    log.append(f"final: median reproj {sol.cost_px:.2f}px")
    return sol, mapping, log


def dtw_heights(y_k: np.ndarray, y_r: np.ndarray, cap_px: float = 40.0, iters: int = 4,
                band: int = 40) -> np.ndarray:
    """Monotone frame correspondence k -> r from per-frame image heights alone.

    Heights of the same LED in two views are close to an affine function of
    each other. Steps: (1, 1) normal, (1, 0) repeat in k, (1, 2) skip in r,
    each with a small penalty. NaN heights cost a constant. Returns, for each
    frame of k, the matched frame of r.
    """
    F = len(y_k)
    a, b = 1.0, 0.0
    match = np.arange(F)
    for _ in range(iters):
        pred = a * y_r + b
        cost = np.abs(y_k[:, None] - pred[None, :])
        cost = np.where(np.isfinite(cost), np.minimum(cost, cap_px) / cap_px, 0.35)
        ii, jj = np.indices(cost.shape)
        cost[np.abs(ii - jj) > band] = 1e3
        INF = 1e18
        D = np.full((F, F), INF)
        back = np.zeros((F, F), np.int8)
        D[0, :band] = cost[0, :band]
        for i in range(1, F):
            c1 = np.concatenate([[INF], D[i - 1, :-1]]) + 0.0    # (1, 1)
            c0 = D[i - 1] + 0.3                                   # (1, 0)
            c2 = np.concatenate([[INF, INF], D[i - 1, :-2]]) + 0.4  # (1, 2)
            stack = np.stack([c0, c1, c2])
            arg = np.argmin(stack, 0)
            D[i] = stack[arg, np.arange(F)] + cost[i]
            back[i] = arg
        j = int(np.argmin(D[-1]))
        match = np.zeros(F, int)
        for i in range(F - 1, -1, -1):
            match[i] = j
            j -= int(back[i, j])
        ok = np.isfinite(y_k) & np.isfinite(y_r[match])
        res = y_k[ok] - (a * y_r[match][ok] + b)
        keep = np.abs(res) < 2.5 * 1.4826 * np.median(np.abs(res)) + 1
        A = np.vstack([y_r[match][ok][keep], np.ones(keep.sum())]).T
        a, b = np.linalg.lstsq(A, y_k[ok][keep], rcond=None)[0]
    return match


def initial_mapping(frames: list[FrameDets], ref: int = 0, start_lag: int = 1,
                    min_conf: float = 0.5) -> list[np.ndarray]:
    """Frame -> LED for every view, relative to the reference view's f - start_lag."""
    def heights(fd):
        y = fd.uv[:, 1].copy()
        y[~(fd.conf >= min_conf)] = np.nan
        return y
    y_r = heights(frames[ref])
    F = len(y_r)
    out = []
    for k, fd in enumerate(frames):
        if k == ref:
            out.append(np.arange(F) - start_lag)
        else:
            out.append(dtw_heights(heights(fd), y_r) - start_lag)
    return out
