"""Pinhole camera with OpenCV-style radial/tangential distortion.

World frame: metres, +y up. Camera frame (OpenCV): +x right, +y down, +z
forward. X_cam = R @ X_world + t.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    name: str = ""

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def rvec(self) -> np.ndarray:
        return cv2.Rodrigues(self.R)[0].ravel()

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R.T + self.t

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (uv (n, 2), depth (n,)). Points behind the camera get NaN."""
        pc = self.to_camera(pts)
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3).astype(np.float64), self.rvec,
                                  self.t.astype(np.float64), self.K, self.dist)
        uv = uv.reshape(-1, 2)
        uv[pc[:, 2] <= 0.05] = np.nan
        return uv, pc[:, 2]

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit world-frame ray directions through pixel coordinates."""
        und = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist).reshape(-1, 2)
        d = np.hstack([und, np.ones((len(und), 1))]) @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "height": self.height,
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "R": self.R.tolist(), "t": self.t.tolist(), "dist": list(map(float, self.dist))}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(d["width"], d["height"], d["fx"], d["fy"], d["cx"], d["cy"],
                   np.array(d["R"]), np.array(d["t"]), np.array(d["dist"]), d.get("name", ""))


def look_at(position, target, up=(0, 1, 0), roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation for a camera at position looking at target."""
    position = np.asarray(position, float)
    fwd = np.asarray(target, float) - position
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    if roll_deg:
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R, -R @ position


def phone_camera(yaw_deg: float, distance: float, height: float, target_y: float,
                 width: int = 1080, height_px: int = 1920, hfov_short_deg: float = 45.6,
                 roll_deg: float = 0.0, k1: float = 0.0, k2: float = 0.0,
                 name: str = "") -> Camera:
    """Portrait phone camera on a circle around the trunk.

    The camera sits at (d cos yaw, h, d sin yaw) looking at the trunk, so yaw 0
    is the front camera on +x. Increasing yaw turns from +x towards +z, which is
    clockwise seen from above in this right-handed, y-up frame.
    """
    th = np.radians(yaw_deg)
    pos = np.array([distance * np.cos(th), height, distance * np.sin(th)])
    R, t = look_at(pos, [0, target_y, 0], roll_deg=roll_deg)
    fx = (width / 2) / np.tan(np.radians(hfov_short_deg) / 2)
    return Camera(width, height_px, fx, fx, width / 2, height_px / 2, R, t,
                  np.array([k1, k2, 0, 0, 0], float), name)
