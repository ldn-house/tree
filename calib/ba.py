"""Bundle adjustment: Levenberg-Marquardt with analytic Jacobians.

Unknowns: per view a rotation (left-multiplied small-angle update) and a
translation; per device a log focal length and radial distortion k1, k2; per
point its 3D position. Observations are robustified with Huber weights
(iteratively reweighted). The normal equations are small enough to solve
directly (a few hundred points, a handful of cameras).

scipy's generic least_squares with finite differences converged very slowly
along the focal-length/depth direction, which left the focal length at its
initial guess on real and synthetic data.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix


def _rodrigues(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _log_rot(R):
    import cv2
    return cv2.Rodrigues(R)[0].ravel()


def _skew(v):
    """(m, 3) -> (m, 3, 3) cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


class Problem:
    def __init__(self, vi, pi, uv, w, dev_of_view, sizes):
        self.vi = np.asarray(vi, int)          # (m,) view of each observation
        self.pi = np.asarray(pi, int)          # (m,) point of each observation
        self.uv = np.asarray(uv, float)        # (m, 2)
        self.w = np.asarray(w, float)          # (m,) sqrt weights (confidence)
        self.dev = np.asarray(dev_of_view, int)
        self.cxcy = np.array([(w_ / 2, h_ / 2) for w_, h_ in sizes], float)  # per view
        self.V = len(dev_of_view)
        self.D = int(self.dev.max()) + 1

    def project(self, Rs, ts, f, k, X, with_jac=False):
        vi, pi = self.vi, self.pi
        R = Rs[vi]                                     # (m, 3, 3)
        Xw = X[pi]
        RX = np.einsum("mij,mj->mi", R, Xw)
        Xc = RX + ts[vi]
        z = Xc[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
        d = self.dev[vi]
        fv, k1, k2 = f[d], k[d, 0], k[d, 1]
        r2 = x * x + y * y
        dist = 1 + k1 * r2 + k2 * r2 * r2
        c = self.cxcy[vi]
        u = fv * x * dist + c[:, 0]
        v = fv * y * dist + c[:, 1]
        pred = np.stack([u, v], 1)
        if not with_jac:
            return pred, z
        dd = k1 + 2 * k2 * r2                          # d(dist)/d(r2)
        # d(u, v)/d(x, y)
        Juv = np.empty((len(x), 2, 2))
        Juv[:, 0, 0] = fv * (dist + 2 * x * x * dd)
        Juv[:, 0, 1] = fv * 2 * x * y * dd
        Juv[:, 1, 0] = Juv[:, 0, 1]
        Juv[:, 1, 1] = fv * (dist + 2 * y * y * dd)
        # d(x, y)/d(Xc)
        Jxy = np.zeros((len(x), 2, 3))
        Jxy[:, 0, 0] = 1 / zs
        Jxy[:, 0, 2] = -x / zs
        Jxy[:, 1, 1] = 1 / zs
        Jxy[:, 1, 2] = -y / zs
        JXc = Juv @ Jxy                                # (m, 2, 3)
        J = {
            "rot": JXc @ (-_skew(RX)),                 # left perturbation of R
            "t": JXc,
            "pt": JXc @ R,
            "logf": np.stack([fv * x * dist, fv * y * dist], 1)[:, :, None],
            "k": np.stack([np.stack([fv * x * r2, fv * x * r2 * r2], 1),
                           np.stack([fv * y * r2, fv * y * r2 * r2], 1)], 1),  # (m, 2, 2)
        }
        return pred, z, J


def bundle_adjust(prob: Problem, rvecs, tvecs, f, k, X, active, iters=40, huber=2.0,
                  fix_intrinsics=False, fix_distortion=False, fix_k2=True, verbose=False):
    """Refine cameras, intrinsics and active points. Returns updated arrays and RMS error.

    By default only k1 is estimated: phone video is already distortion-corrected,
    and k1 and k2 together trade off against each other and fit noise.
    """
    V, D = prob.V, prob.D
    Rs = np.stack([_rodrigues(np.asarray(r, float)) for r in rvecs])
    ts = np.array(tvecs, float)
    f = np.array(f, float)
    k = np.array(k, float).reshape(D, 2)
    X = np.array(X, float)
    act = np.nonzero(active)[0]
    col_of_point = -np.ones(len(X), int)
    col_of_point[act] = np.arange(len(act))
    keep = active[prob.pi]
    sub = Problem(prob.vi[keep], prob.pi[keep], prob.uv[keep], prob.w[keep], prob.dev,
                  [(c[0] * 2, c[1] * 2) for c in prob.cxcy])
    nk = 0 if fix_distortion else (1 if fix_k2 else 2)
    n_int = 0 if fix_intrinsics else D * (1 + nk)
    G = 6 * V + n_int
    P = 3 * len(act)

    def robust_cost(pred):
        e = np.linalg.norm(pred - sub.uv, axis=1) * sub.w
        hub = np.where(e <= huber, 0.5 * e * e, huber * (e - 0.5 * huber))
        return float(hub.sum()), e

    pred, _ = sub.project(Rs, ts, f, k, X)
    cost, e = robust_cost(pred)
    lam = 1e-3
    m = len(sub.vi)
    rel = 1.0
    for it in range(iters):
        pred, z, J = sub.project(Rs, ts, f, k, X, with_jac=True)
        res = (pred - sub.uv)
        e = np.linalg.norm(res, axis=1) * sub.w
        hw = np.where(e <= huber, 1.0, huber / np.maximum(e, 1e-9))   # Huber IRLS weight
        sw = (sub.w * np.sqrt(hw))[:, None]
        r = (res * sw).ravel()
        # Assemble the sparse Jacobian: columns [views 6V | intrinsics | points 3P]
        blocks_r, blocks_c, blocks_v = [], [], []

        def add(Jb, cols):  # Jb (m, 2, q), cols (m, q)
            q = Jb.shape[2]
            Jb = Jb * sw[:, :, None]
            rr = np.repeat(np.arange(2 * m).reshape(m, 2, 1), q, 2)
            cc = np.repeat(cols[:, None, :], 2, 1)
            blocks_r.append(rr.ravel())
            blocks_c.append(cc.ravel())
            blocks_v.append(Jb.ravel())

        vi = sub.vi
        add(J["rot"], 6 * vi[:, None] + np.arange(3)[None])
        add(J["t"], 6 * vi[:, None] + 3 + np.arange(3)[None])
        dv = sub.dev[vi]
        if not fix_intrinsics:
            add(J["logf"], 6 * V + dv[:, None])
            if nk:
                add(J["k"][:, :, :nk], 6 * V + D + nk * dv[:, None] + np.arange(nk)[None])
        pc = col_of_point[sub.pi]
        add(J["pt"], G + 3 * pc[:, None] + np.arange(3)[None])
        Jm = coo_matrix((np.concatenate(blocks_v), (np.concatenate(blocks_r), np.concatenate(blocks_c))),
                        shape=(2 * m, G + P)).tocsr()
        H = (Jm.T @ Jm).toarray()
        g = Jm.T @ r
        diag = np.diag(H).copy()
        improved = False
        for _try in range(8):
            A = H + np.diag(lam * np.maximum(diag, 1e-9) + 1e-9)
            try:
                delta = -cho_solve(cho_factor(A), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            Rn = Rs.copy()
            for i in range(V):
                Rn[i] = _rodrigues(delta[6 * i:6 * i + 3]) @ Rs[i]
            tn = ts + delta[:6 * V].reshape(V, 6)[:, 3:]
            fn, kn = f.copy(), k.copy()
            if not fix_intrinsics:
                fn = f * np.exp(delta[6 * V:6 * V + D])
                if nk:
                    kn[:, :nk] = k[:, :nk] + delta[6 * V + D:6 * V + D + nk * D].reshape(D, nk)
            Xn = X.copy()
            Xn[act] += delta[G:].reshape(-1, 3)
            pn, _ = sub.project(Rn, tn, fn, kn, Xn)
            cn, _ = robust_cost(pn)
            if cn < cost:
                rel = (cost - cn) / max(cost, 1e-12)
                Rs, ts, f, k, X, cost = Rn, tn, fn, kn, Xn, cn
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 4
        if verbose:
            print(f"  ba iter {it}: cost {cost:.2f} lambda {lam:.1e} f {np.round(f, 1)}")
        if not improved or rel < 1e-7:
            break
    pred, _ = sub.project(Rs, ts, f, k, X)
    rms = float(np.sqrt(np.mean(np.sum((pred - sub.uv) ** 2, 1))))
    return [_log_rot(R) for R in Rs], list(ts), list(f), [tuple(kk) for kk in k], X, rms
