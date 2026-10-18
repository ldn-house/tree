"""Multi-view LED triangulation with self-calibrated perspective cameras.

Inputs are per-view 2D detections keyed by LED index (NaN = not seen) with a
confidence weight. Because the LED IDs give point correspondences across
views, this is structure-from-motion with known matches:

1. Initialise: essential matrix on the best-overlapping pair, PnP for the
   other views, multi-view DLT for the points. Several focal-length guesses are
   tried and the lowest robust cost wins.
2. Bundle-adjust camera poses, a shared focal length and radial distortion per
   device, and all points, with a robust loss weighted by confidence.
3. Reject observations with large reprojection error and repeat.
4. Add the wire prior: consecutive LEDs are at most one pitch apart (hinge),
   plus a weak pull towards the midpoint of the neighbours. This places LEDs
   seen in one view on their ray and interpolates LEDs seen in none.

A perspective reconstruction has no mirror ambiguity, so the result's
handedness is fixed by the images (the orthographic method could not tell
the 90 and 270 degree views apart).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass
class ViewObs:
    name: str
    width: int
    height: int
    uv: np.ndarray            # (n, 2), NaN where not detected
    weight: np.ndarray        # (n,) confidence in [0, 1]
    device: int = 0           # views sharing intrinsics


class PlacementError(RuntimeError):
    """A view could not be placed against the others (PnP failed)."""

    def __init__(self, view: str):
        super().__init__(f"PnP failed for {view}")
        self.view = view


@dataclass
class Solution:
    points: np.ndarray                    # (n, 3) world, arbitrary scale; NaN if unsolved
    views_used: np.ndarray                # (n,) number of inlier observations
    reproj_px: np.ndarray                 # (n,) RMS reprojection error of inliers
    rvecs: list
    tvecs: list
    focal: list                           # per device
    dist: list                            # per device (k1, k2)
    inliers: list                         # per view (n,) bool
    cost_px: float                        # median inlier reprojection error
    log: list = field(default_factory=list)
    views: list = field(default_factory=list)    # names of the views actually used, in order
    dropped: list = field(default_factory=list)  # views that could not be placed

    def camera_centres(self) -> np.ndarray:
        return np.array([-cv2.Rodrigues(r)[0].T @ t for r, t in zip(self.rvecs, self.tvecs)])

    def camera_up(self) -> np.ndarray:
        return np.array([-cv2.Rodrigues(r)[0][1] for r in self.rvecs])


def _K(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


def _rodrigues(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def project(X, rv, t, f, k1, k2, w, h):
    pc = X @ _rodrigues(rv).T + t
    z = np.where(np.abs(pc[:, 2]) < 1e-6, 1e-6, pc[:, 2])
    x, y = pc[:, 0] / z, pc[:, 1] / z
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * x * d + w / 2, f * y * d + h / 2], 1), pc[:, 2]


def triangulate_dlt(Ps: list[np.ndarray], uvs: list[np.ndarray]) -> np.ndarray:
    A = []
    for P, (u, v) in zip(Ps, uvs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _undistort(uv, f, k, w, h):
    K = _K(f, w, h)
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, np.array([k[0], k[1], 0, 0, 0]))
    return und.reshape(-1, 2) * f + np.array([w / 2, h / 2])


class Solver:
    def __init__(self, views: list[ViewObs], min_weight: float = 0.5, huber_px: float = 2.0,
                 verbose: bool = False):
        self.views = views
        self.n = len(views[0].uv)
        self.min_weight = min_weight
        self.huber = huber_px
        self.verbose = verbose
        self.n_dev = max(v.device for v in views) + 1
        self.mask = [np.isfinite(v.uv).all(1) & (v.weight >= min_weight) for v in views]
        self.log = []

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print("  " + msg)

    # ---- initialisation -------------------------------------------------
    def pair_stats(self):
        """Shared LEDs and median image shift (fraction of the diagonal) for every view pair."""
        V = len(self.views)
        out = {}
        for a in range(V):
            for b in range(a + 1, V):
                m = self.mask[a] & self.mask[b]
                if m.sum() == 0:
                    continue
                diag = np.hypot(self.views[a].width, self.views[a].height)
                shift = float(np.median(np.linalg.norm(self.views[a].uv[m] - self.views[b].uv[m], axis=1)) / diag)
                out[(a, b)] = (int(m.sum()), shift)
        return out

    def duplicate_viewpoints(self, max_shift: float = 0.03, min_shared: int = 30):
        """Pairs of views filmed from (nearly) the same place: they add almost no 3D information."""
        return [(self.views[a].name, self.views[b].name, sh, n) for (a, b), (n, sh) in self.pair_stats().items()
                if n >= min_shared and sh < max_shift]

    def _init(self, f0: float):
        """Two-view start, then resection. Tries pairs by shared LEDs x parallax.

        A pair filmed from nearly the same place shares many LEDs but pins down
        no depth, so pairs are ranked by overlap times image shift (capped), and
        a pair whose triangulation angle is too small is skipped. A view that
        cannot be placed from any starting pair is dropped with a warning.
        """
        stats = self.pair_stats()
        ranked = sorted(((n * min(1.0, sh / 0.08), a, b) for (a, b), (n, sh) in stats.items() if n >= 15),
                        reverse=True)
        if not ranked:
            raise RuntimeError("not enough shared LEDs between any two views")
        last = None
        for _score, a, b in ranked:
            try:
                return self._init_from(f0, a, b)
            except RuntimeError as e:
                last = e
                self._say(f"f0={f0:.0f}: start pair {self.views[a].name}+{self.views[b].name} failed ({e})")
                self.failures.append(getattr(e, "view", None))
        raise last

    def _init_from(self, f0: float, a: int, b: int):
        V, n = len(self.views), self.n
        rv = [None] * V
        tv = [None] * V
        X = np.full((n, 3), np.nan)
        va, vb = self.views[a], self.views[b]
        m = self.mask[a] & self.mask[b]
        Ka, Kb = _K(f0, va.width, va.height), _K(f0, vb.width, vb.height)
        pa = cv2.undistortPoints(va.uv[m].reshape(-1, 1, 2), Ka, None).reshape(-1, 2)
        pb = cv2.undistortPoints(vb.uv[m].reshape(-1, 1, 2), Kb, None).reshape(-1, 2)
        E, inl = cv2.findEssentialMat(pa, pb, np.eye(3), cv2.RANSAC, 0.999, 2.0 / f0)
        if E is None:
            raise RuntimeError("essential matrix failed")
        E = E[:3]
        _, R, t, inl2 = cv2.recoverPose(E, pa, pb, np.eye(3), mask=inl)
        # Triangulation angle: rays from the two centres to the shared points.
        C_b = (-R.T @ t).ravel()
        Xab = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, t]), pa.T, pb.T)
        Xab = (Xab[:3] / Xab[3]).T
        ra, rb = Xab, Xab - C_b
        cosang = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
        angle = float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1)))))
        if angle < 3.0:
            raise RuntimeError(f"triangulation angle only {angle:.1f} degrees")
        rv[a], tv[a] = np.zeros(3), np.zeros(3)
        rv[b], tv[b] = cv2.Rodrigues(R)[0].ravel(), t.ravel()
        self._triangulate(X, rv, tv, [f0] * self.n_dev, [(0, 0)] * self.n_dev)
        for _ in range(V):
            todo = [k for k in range(V) if rv[k] is None]
            if not todo:
                break
            best = max(todo, key=lambda k: (self.mask[k] & np.isfinite(X).all(1)).sum())
            vk = self.views[best]
            mk = self.mask[best] & np.isfinite(X).all(1)
            ok = False
            if mk.sum() >= 8:
                ok, r, t_, _inl = cv2.solvePnPRansac(X[mk], vk.uv[mk], _K(f0, vk.width, vk.height), None,
                                                     reprojectionError=6.0, iterationsCount=500)
            if not ok:
                raise PlacementError(vk.name)
            rv[best], tv[best] = r.ravel(), t_.ravel()
            self._triangulate(X, rv, tv, [f0] * self.n_dev, [(0, 0)] * self.n_dev)
        return rv, tv, X

    def _triangulate(self, X, rv, tv, f, k, inliers=None):
        Ps, posed = {}, [i for i in range(len(self.views)) if rv[i] is not None]
        for i in posed:
            v = self.views[i]
            R = _rodrigues(rv[i])
            Ps[i] = _K(f[v.device], v.width, v.height) @ np.hstack([R, tv[i][:, None]])
        und = {i: _undistort(np.nan_to_num(self.views[i].uv), f[self.views[i].device], k[self.views[i].device],
                             self.views[i].width, self.views[i].height) for i in posed}
        for p in range(self.n):
            vs = [i for i in posed if self.mask[i][p] and (inliers is None or inliers[i][p])]
            if len(vs) >= 2:
                X[p] = triangulate_dlt([Ps[i] for i in vs], [und[i][p] for i in vs])
        return X

    # ---- bundle adjustment ----------------------------------------------
    def _pack(self, rv, tv, f, k, X):
        return np.concatenate([np.ravel(rv), np.ravel(tv), f, np.ravel(k), X.ravel()])

    def _unpack(self, x):
        V, D, n = len(self.views), self.n_dev, self.n
        rv = x[:3 * V].reshape(V, 3)
        tv = x[3 * V:6 * V].reshape(V, 3)
        f = x[6 * V:6 * V + D]
        k = x[6 * V + D:6 * V + 3 * D].reshape(D, 2)
        X = x[6 * V + 3 * D:].reshape(n, 3)
        return rv, tv, f, k, X

    def _bundle(self, rv, tv, f, k, X, inliers, wire: dict | None = None, iters: int = 60):
        if wire is None:
            return self._bundle_lm(rv, tv, f, k, X, inliers)
        V, D, n = len(self.views), self.n_dev, self.n
        solved = np.isfinite(X).all(1)
        if wire is None:
            X = np.where(solved[:, None], X, 0.0)
        obs = []  # (view, point, u, v, sqrt-weight)
        for i, v in enumerate(self.views):
            idx = np.nonzero(inliers[i] & (solved if wire is None else np.ones(n, bool)))[0]
            obs.append((i, idx, v.uv[idx], np.sqrt(np.clip(v.weight[idx], 0.05, 1))))
        n_obs = sum(len(o[1]) for o in obs)
        pairs = np.arange(n - 1) if wire is not None else np.zeros(0, int)
        n_res = 2 * n_obs + (len(pairs) * 4 if wire is not None else 0)
        # Focal length is optimised as log(f / f0) so it is dimensionless, and every
        # block gets an explicit typical scale. With x_scale="jac" a focal length
        # in pixels barely moved in 60 iterations, so it stayed at its guess.
        f0 = np.array(f, float)
        x0 = self._pack(rv, tv, np.zeros(D), k, X)
        n_par = len(x0)
        fin = np.isfinite(X).all(1) & (np.abs(X).sum(1) > 0)
        scene = float(np.median(np.linalg.norm(X[fin] - np.median(X[fin], 0), axis=1))) if fin.sum() > 3 else 1.0
        xs = np.concatenate([np.full(3 * V, 1e-2), np.full(3 * V, 1e-2 * scene), np.full(D, 1e-2),
                             np.full(2 * D, 1e-2), np.full(3 * n, 1e-2 * scene)])
        free_pts = np.ones(n, bool) if wire is not None else solved
        S = lil_matrix((n_res, n_par), dtype=np.uint8)
        row = 0
        for i, idx, _uv, _w in obs:
            dev = self.views[i].device
            for j, p in enumerate(idx):
                for r in (row + 2 * j, row + 2 * j + 1):
                    S[r, 3 * i:3 * i + 3] = 1
                    S[r, 3 * V + 3 * i:3 * V + 3 * i + 3] = 1
                    S[r, 6 * V + dev] = 1
                    S[r, 6 * V + D + 2 * dev:6 * V + D + 2 * dev + 2] = 1
                    S[r, 6 * V + 3 * D + 3 * p:6 * V + 3 * D + 3 * p + 3] = 1
            row += 2 * len(idx)
        base = 6 * V + 3 * D
        if wire is not None:
            for j, p in enumerate(pairs):
                S[row + j, base + 3 * p:base + 3 * p + 6] = 1  # hinge on (p, p+1)
            row += len(pairs)
            for j in range(len(pairs)):
                # Midpoint row j constrains point j + 1 against points j and j + 2.
                lo, hi = j, min(j + 3, n)
                for c in range(3):
                    S[row + 3 * j + c, base + 3 * lo:base + 3 * hi] = 1
            row += 3 * len(pairs)

        def resid(x):
            rv_, tv_, lf_, k_, X_ = self._unpack(x)
            f_ = f0 * np.exp(lf_)
            out = []
            for i, idx, uv, w in obs:
                v = self.views[i]
                pr, _ = project(X_[idx], rv_[i], tv_[i], f_[v.device], k_[v.device, 0], k_[v.device, 1],
                                v.width, v.height)
                out.append(((pr - uv) * w[:, None]).ravel())
            if wire is not None:
                pitch, w_h, w_m, unit = wire["pitch"], wire["w_hinge"], wire["w_mid"], wire["px_per_unit"]
                d = np.linalg.norm(X_[1:] - X_[:-1], axis=1)
                out.append(w_h * unit * np.maximum(d - pitch, 0))
                mid = np.zeros((n - 1, 3))
                inner = np.arange(1, n - 1)
                mid[inner - 1] = X_[inner] - 0.5 * (X_[inner - 1] + X_[inner + 1])
                wm = w_m[pairs] if np.ndim(w_m) else np.full(len(pairs), w_m)
                out.append((mid * (wm * unit)[:, None]).ravel())
            return np.concatenate(out)

        # Fix the gauge loosely: freeze nothing, but keep point columns of
        # unsolved LEDs out of the problem when there is no wire prior.
        res = least_squares(resid, x0, jac_sparsity=S.tocsr(), loss="huber", f_scale=self.huber,
                            x_scale=xs, max_nfev=iters, method="trf", verbose=0)
        rv, tv, lf, k, Xn = self._unpack(res.x)
        f = f0 * np.exp(lf)
        if wire is None:
            Xn = np.where(solved[:, None], Xn, np.nan)
        return list(rv), list(tv), list(f), [tuple(kk) for kk in k], Xn

    def _bundle_lm(self, rv, tv, f, k, X, inliers, iters: int = 40):
        """Cameras, intrinsics and points by Levenberg-Marquardt (calib.ba)."""
        from .ba import Problem, bundle_adjust
        solved = np.isfinite(X).all(1)
        vi, pi, uv, w = [], [], [], []
        for i, v in enumerate(self.views):
            idx = np.nonzero(inliers[i] & solved & self.mask[i])[0]
            vi.append(np.full(len(idx), i))
            pi.append(idx)
            uv.append(v.uv[idx])
            w.append(np.sqrt(np.clip(v.weight[idx], 0.05, 1)))
        prob = Problem(np.concatenate(vi), np.concatenate(pi), np.concatenate(uv), np.concatenate(w),
                       [v.device for v in self.views], [(v.width, v.height) for v in self.views])
        rv, tv, f, k, Xn, _rms = bundle_adjust(prob, rv, tv, f, k, np.nan_to_num(X), solved, iters=iters,
                                               huber=self.huber)
        Xn = np.where(solved[:, None], Xn, np.nan)
        return rv, [np.asarray(t) for t in tv], f, k, Xn

    def _errors(self, rv, tv, f, k, X):
        errs = []
        for i, v in enumerate(self.views):
            pr, z = project(np.nan_to_num(X), rv[i], tv[i], f[v.device], k[v.device][0], k[v.device][1],
                            v.width, v.height)
            e = np.linalg.norm(pr - np.nan_to_num(v.uv), axis=1)
            e[~self.mask[i] | ~np.isfinite(X).all(1) | (z <= 0)] = np.nan
            errs.append(e)
        return errs

    def solve(self, f_guesses=(0.6, 0.85, 1.2, 1.6), wire_prior: bool = True,
              reject_px: float = 6.0, init: "Solution | None" = None) -> Solution:
        """Solve from scratch, or warm-start from a previous solution's cameras."""
        if init is not None:
            rv, tv = [np.array(r) for r in init.rvecs], [np.array(t) for t in init.tvecs]
            f, k = list(init.focal), list(init.dist)
            X = self._triangulate(np.full((self.n, 3), np.nan), rv, tv, f, k)
            rv, tv, f, k, X = self._bundle(rv, tv, f, k, X, [m.copy() for m in self.mask])
            return self._finish(rv, tv, f, k, X, wire_prior, reject_px)
        best = None
        self.failures = []
        long_side = max(max(v.width, v.height) for v in self.views)
        for g in f_guesses:
            f0 = g * long_side
            try:
                rv, tv, X = self._init(f0)
            except RuntimeError as e:
                self._say(f"f0={f0:.0f}: init failed ({e})")
                continue
            f, k = [f0] * self.n_dev, [(0.0, 0.0)] * self.n_dev
            inl = [m.copy() for m in self.mask]
            rv, tv, f, k, X = self._bundle(rv, tv, f, k, X, inl)
            errs = self._errors(rv, tv, f, k, X)
            med = float(np.nanmedian(np.concatenate(errs)))
            self._say(f"f0={f0:.0f}: focal {np.round(f, 0)}, median reproj {med:.2f}px")
            if best is None or med < best[0]:
                best = (med, rv, tv, f, k, X)
        if best is None:
            # Last resort: no start pair and focal guess placed every view. Drop the
            # view that failed most often and solve the rest.
            names = [v for v in self.failures if v]
            if names and len(self.views) > 2:
                worst = max(set(names), key=names.count)
                self._say(f"WARNING: dropping {worst}: it could not be placed against the other views")
                keep = [v for v in self.views if v.name != worst]
                sub = Solver(keep, self.min_weight, self.huber, self.verbose)
                sub.log = self.log
                sol = sub.solve(f_guesses, wire_prior, reject_px)
                sol.dropped = [worst] + sol.dropped
                return sol
            raise RuntimeError("all initialisations failed")
        _, rv, tv, f, k, X = best
        return self._finish(rv, tv, f, k, X, wire_prior, reject_px)

    def _finish(self, rv, tv, f, k, X, wire_prior, reject_px) -> Solution:
        inl = [m.copy() for m in self.mask]
        for rnd in range(3):
            errs = self._errors(rv, tv, f, k, X)
            allv = np.concatenate(errs)
            thr = max(reject_px, 4 * 1.4826 * np.nanmedian(allv))
            inl = [self.mask[i] & (np.nan_to_num(errs[i], nan=1e9) <= thr) for i in range(len(self.views))]
            X = self._triangulate(np.full((self.n, 3), np.nan), rv, tv, f, k, inl)
            rv, tv, f, k, X = self._bundle(rv, tv, f, k, X, inl)
            self._say(f"reject round {rnd}: threshold {thr:.1f}px, inliers {sum(m.sum() for m in inl)}")
        errs = self._errors(rv, tv, f, k, X)
        if wire_prior:
            X = self._wire(rv, tv, f, k, X, inl, errs)
            errs = self._errors(rv, tv, f, k, X)
        n_used = np.zeros(self.n, int)
        sq = np.zeros(self.n)
        for i in range(len(self.views)):
            e = np.where(inl[i], errs[i], np.nan)
            ok = np.isfinite(e)
            n_used += ok
            sq += np.where(ok, e ** 2, 0)
        rms = np.where(n_used > 0, np.sqrt(sq / np.maximum(n_used, 1)), np.nan)
        all_inl = np.concatenate([np.where(inl[i], errs[i], np.nan) for i in range(len(self.views))])
        return Solution(X, n_used, rms, rv, tv, f, k, inl, float(np.nanmedian(all_inl)), self.log,
                        [v.name for v in self.views], [])

    def ray_angles(self, rv, tv, X, inl):
        """Widest angle (degrees) between any two inlier rays to each point."""
        C = np.array([-_rodrigues(r).T @ t for r, t in zip(rv, tv)])
        best = np.zeros(self.n)
        ok = np.isfinite(X).all(1)
        for a in range(len(C)):
            for b in range(a + 1, len(C)):
                both = ok & inl[a] & inl[b]
                if not both.any():
                    continue
                ra, rb = X[both] - C[a], X[both] - C[b]
                cos = (ra * rb).sum(1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1) + 1e-12)
                best[both] = np.maximum(best[both], np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return best

    def _wire(self, rv, tv, f, k, X, inl, errs, min_angle: float = 3.0):
        n = self.n
        # A point seen only from (nearly) the same place has an ill-defined depth:
        # treat it like a single-view point (on its ray, placed by the wire).
        angle = self.ray_angles(rv, tv, X, inl)
        weak = np.isfinite(X).all(1) & (angle < min_angle)
        if weak.any():
            self._say(f"{int(weak.sum())} LEDs triangulated at under {min_angle:g} degrees "
                      f"(near-duplicate viewpoints): placed along their rays by the wire prior")
        X = np.where(weak[:, None], np.nan, X)
        multi = np.isfinite(X).all(1)
        d = np.linalg.norm(np.diff(X, axis=0), axis=1)
        d = d[np.isfinite(d)]
        if len(d) < 10 or multi.sum() < 10:
            self._say("wire prior skipped: too few triangulated neighbours")
            return X
        pitch = float(np.percentile(d, 85))
        # Initialise unsolved points: single-view points on their ray at the depth
        # of the interpolated neighbours; others by interpolation along the wire.
        Xi = X.copy()
        idx = np.arange(n)
        good = np.nonzero(multi)[0]
        for c in range(3):
            Xi[:, c] = np.interp(idx, good, X[good, c])
        for p in np.nonzero(~multi)[0]:
            seen = [i for i in range(len(self.views)) if self.mask[i][p]]
            if not seen:
                continue
            i = seen[0]
            v = self.views[i]
            R = _rodrigues(rv[i])
            C = -R.T @ tv[i]
            und = _undistort(v.uv[p:p + 1], f[v.device], k[v.device], v.width, v.height)[0]
            ray = R.T @ np.array([(und[0] - v.width / 2) / f[v.device], (und[1] - v.height / 2) / f[v.device], 1])
            ray /= np.linalg.norm(ray)
            Xi[p] = C + ray * np.dot(Xi[p] - C, ray)
            inl[i][p] = True
        # px_per_unit converts world distances to pixels so the priors are
        # commensurate with reprojection residuals.
        Cs = np.array([-_rodrigues(r).T @ t for r, t in zip(rv, tv)])
        depth = np.median(np.linalg.norm(Cs[:, None] - Xi[good][None], axis=-1))
        px_per_unit = float(np.mean(f) / depth)
        n_views = np.array([sum(self.mask[i][p] and inl[i][p] for i in range(len(self.views))) for p in range(n)])
        n_views = np.where(weak, 1, n_views)  # near-duplicate rays count as one view
        # Midpoint row j is for point j + 1; pull harder where the images say little.
        w_mid = np.where(n_views[1:] >= 2, 0.05, 0.5)
        wire = {"pitch": pitch, "w_hinge": 1.0, "w_mid": w_mid, "px_per_unit": px_per_unit}
        self._say(f"wire prior: pitch {pitch:.4f} units ({pitch * px_per_unit:.1f}px), "
                  f"{(~multi).sum()} LEDs placed by single view or interpolation")
        _, _, _, _, Xn = self._bundle(rv, tv, f, k, Xi, inl, wire=wire, iters=40)
        return Xn


# ---- output frame ---------------------------------------------------------

def to_tree_frame(points: np.ndarray, sol: Solution, front_view: int = 0,
                  height_m: float | None = None) -> tuple[np.ndarray, dict]:
    """Map world points to the shared coordinate contract.

    y up with y = 0 at the lowest LED and 1 at the highest; x and z in the same
    units; origin on the trunk axis; +x towards the front camera; right-handed.
    Also returns metres if height_m (lowest to highest LED) is given.
    """
    up = sol.camera_up().mean(0)
    up /= np.linalg.norm(up)
    # Rotate so up -> +y.
    v = np.cross(up, [0, 1, 0])
    s, c = np.linalg.norm(v), np.dot(up, [0, 1, 0])
    if s < 1e-9:
        R1 = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    P = points @ R1.T
    C = sol.camera_centres() @ R1.T
    ok = np.isfinite(P).all(1)
    y = P[ok, 1]
    # Trunk axis: median of per-height-slice centroids.
    edges = np.quantile(y, np.linspace(0, 1, 11))
    cents = [P[ok][(y >= a) & (y <= b)][:, [0, 2]].mean(0) for a, b in zip(edges[:-1], edges[1:])]
    axis = np.median(np.array(cents), axis=0)
    P[:, 0] -= axis[0]
    P[:, 2] -= axis[1]
    C[:, 0] -= axis[0]
    C[:, 2] -= axis[1]
    ang = np.arctan2(C[front_view, 2], C[front_view, 0])
    ca, sa = np.cos(ang), np.sin(ang)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])  # rotates the front camera onto +x
    P = P @ Ry.T
    C = C @ Ry.T
    y0, y1 = np.nanmin(P[:, 1]), np.nanmax(P[:, 1])
    unit = y1 - y0
    P[:, 1] -= y0
    C[:, 1] -= y0
    norm = P / unit
    info = {"unit_world": float(unit), "cameras": (C / unit).tolist()}
    if height_m:
        info["metres"] = (norm * height_m).tolist()
    return norm, info
