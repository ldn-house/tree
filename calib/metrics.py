"""Scoring reconstructed coordinates against ground truth."""

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def align(est: np.ndarray, truth: np.ndarray, allow_reflection: bool = False, robust: bool = True):
    """Align est to truth (both (n, 3), NaN allowed). Returns aligned est and the transform."""
    ok = np.isfinite(est).all(1) & np.isfinite(truth).all(1) & (np.abs(est) < 1e6).all(1)
    if ok.sum() < 4:
        raise ValueError("fewer than 4 points to align")
    keep = ok.copy()
    for _ in range(4 if robust else 1):
        s, R, t = umeyama(est[keep], truth[keep], allow_reflection)
        al = est @ (s * R).T + t
        e = np.linalg.norm(al - truth, axis=1)
        thr = np.nanpercentile(e[ok], 80) * 2.5
        keep = ok & (e <= thr)
        if keep.sum() < max(4, 0.5 * ok.sum()):
            keep = ok
    return est @ (s * R).T + t, (s, R, t)


def position_errors(est: np.ndarray, truth: np.ndarray, pitch: float, allow_reflection: bool = False) -> dict:
    al, (s, R, t) = align(est, truth, allow_reflection)
    ok = np.isfinite(al).all(1)
    e = np.linalg.norm(al - truth, axis=1)
    e_ok = e[ok]
    # "ID error" in 3D: the reconstructed LED is closer to another true LED
    # than to its own, by more than half a pitch.
    d_all = np.linalg.norm(al[ok][:, None] - truth[None], axis=-1)
    nearest = d_all.argmin(1)
    own = np.nonzero(ok)[0]
    swapped = (nearest != own) & (e_ok > pitch / 2)
    return {
        "solved": int(ok.sum()),
        "median_cm": float(np.median(e_ok) * 100),
        "mean_cm": float(e_ok.mean() * 100),
        "p90_cm": float(np.percentile(e_ok, 90) * 100),
        "max_cm": float(e_ok.max() * 100),
        "within_2cm": float((e_ok < 0.02).mean()),
        "within_5cm": float((e_ok < 0.05).mean()),
        "over_10cm": int((e_ok > 0.10).sum()),
        "id_swaps": int(swapped.sum()),
        "reflected": bool(np.linalg.det(R) < 0),
        "scale": float(s),
    }
