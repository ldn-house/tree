#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "numpy",
# ]
# ///
"""
3D Triangulation - Convert 2D positions from multiple angles to 3D coordinates.

Usage:
    # Basic triangulation from two angles
    ./triangulate.py --angles front,side

    # With visualization
    ./triangulate.py --angles front,side --visualize

    # Custom output format
    ./triangulate.py --angles front,side --format csv
    ./triangulate.py --angles front,side --format json
"""

import argparse
import csv
import cv2
import json
import numpy as np
from pathlib import Path

LED_COUNT = 500
POSITIONS_DIR = Path("positions")
OUTPUT_DIR = Path("coordinates")


def load_positions(angle: str) -> dict:
    """Load 2D positions from JSON file."""
    path = POSITIONS_DIR / f"{angle}_positions.json"
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")

    with open(path) as f:
        return json.load(f)


def auto_align_vertical(coords: dict[int, tuple[float, float, float]]
                        ) -> dict[int, tuple[float, float, float]]:
    """
    Use PCA to find the tree's principal axis and rotate to align with Y.

    The tree should be taller than it is wide, so the longest principal
    component should be the vertical axis.
    """
    if len(coords) < 10:
        return coords

    # Convert to numpy array
    indices = sorted(coords.keys())
    points = np.array([coords[i] for i in indices])

    # Center the points
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered)

    # Principal components (rows of Vt)
    # S contains singular values in descending order
    # The first component (Vt[0]) is the direction of maximum variance - should be vertical
    principal_axis = Vt[0]

    print(f"  Principal axis direction: [{principal_axis[0]:.3f}, {principal_axis[1]:.3f}, {principal_axis[2]:.3f}]")
    print(f"  Variance ratios: {S/S.sum()}")

    # We want the principal axis to align with Y (0, 1, 0)
    # But we need to check if it should be +Y or -Y (tree could be detected upside down)
    target = np.array([0, 1, 0])

    # If principal axis points mostly downward, flip it
    if principal_axis[1] < 0:
        principal_axis = -principal_axis

    # Calculate rotation matrix to align principal_axis with target
    # Using Rodrigues' rotation formula
    v = np.cross(principal_axis, target)
    c = np.dot(principal_axis, target)

    if np.linalg.norm(v) < 1e-6:
        # Already aligned (or anti-aligned)
        if c < 0:
            # Anti-aligned: rotate 180° around X axis
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            R = np.eye(3)
    else:
        # Rodrigues formula
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

    # Apply rotation
    rotated = (R @ centered.T).T

    # Re-center so Y=0 is at bottom
    y_min = rotated[:, 1].min()
    rotated[:, 1] -= y_min

    # Build result dict
    result = {}
    for i, idx in enumerate(indices):
        result[idx] = tuple(rotated[i])

    # Calculate how much we rotated
    angle = np.arccos(np.clip(c, -1, 1)) * 180 / np.pi
    print(f"  Rotated {angle:.1f}° to align with vertical")

    return result


def fix_outliers(coords: dict[int, tuple[float, float, float]],
                 threshold: float = 3.0,
                 max_passes: int = 5) -> dict[int, tuple[float, float, float]]:
    """
    Fix outlier points that are too far from their neighbors.

    LEDs are on a string, so each LED should be close to its neighbors.
    Points that are far from both neighbors are likely detection errors.
    Based on Matt Parker's approach.

    Runs multiple passes to catch chains of outliers.
    """
    if len(coords) < 3:
        return coords

    result = dict(coords)
    total_fixed = 0

    for pass_num in range(max_passes):
        # Calculate distances to neighbors for all points
        sorted_indices = sorted(result.keys())
        distances = {}

        for i, idx in enumerate(sorted_indices):
            prev_idx = sorted_indices[i - 1] if i > 0 else None
            next_idx = sorted_indices[i + 1] if i < len(sorted_indices) - 1 else None

            p = np.array(result[idx])

            dist_prev = np.linalg.norm(p - np.array(result[prev_idx])) if prev_idx is not None else None
            dist_next = np.linalg.norm(p - np.array(result[next_idx])) if next_idx is not None else None

            distances[idx] = (dist_prev, dist_next)

        # Calculate median neighbor distance
        all_dists = [d for idx in distances for d in distances[idx] if d is not None]
        median_dist = np.median(all_dists)

        if pass_num == 0:
            print(f"  Median neighbor distance: {median_dist:.4f}")

        # Find outliers: points where BOTH neighbors are too far
        outliers = []
        for idx in sorted_indices:
            dist_prev, dist_next = distances[idx]

            # Check if both neighbors are too far
            prev_bad = dist_prev is not None and dist_prev > threshold * median_dist
            next_bad = dist_next is not None and dist_next > threshold * median_dist

            # Mark as outlier if BOTH neighbors are far
            if prev_bad and next_bad:
                outliers.append(idx)

        if not outliers:
            break

        total_fixed += len(outliers)

        # Fix outliers by interpolating from nearest valid neighbors
        outlier_set = set(outliers)
        for idx in outliers:
            # Find nearest valid previous neighbor
            prev_valid = None
            for i in range(idx - 1, -1, -1):
                if i in result and i not in outlier_set:
                    prev_valid = i
                    break

            # Find nearest valid next neighbor
            next_valid = None
            for i in range(idx + 1, LED_COUNT):
                if i in result and i not in outlier_set:
                    next_valid = i
                    break

            # Interpolate
            if prev_valid is not None and next_valid is not None:
                t = (idx - prev_valid) / (next_valid - prev_valid)
                p1 = np.array(result[prev_valid])
                p2 = np.array(result[next_valid])
                result[idx] = tuple(p1 + t * (p2 - p1))
            elif prev_valid is not None:
                result[idx] = result[prev_valid]
            elif next_valid is not None:
                result[idx] = result[next_valid]

    print(f"  Fixed {total_fixed} outliers in {pass_num + 1} passes (>{threshold}x median distance)")

    # Also fix points outside expected cone shape
    # Tree should be roughly conical: wider at bottom (low Y), narrower at top (high Y)
    y_vals = [c[1] for c in result.values()]
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min if y_max > y_min else 1

    cone_outliers = []
    for idx, (x, y, z) in result.items():
        # Normalized height (0 = bottom, 1 = top)
        height_ratio = (y - y_min) / y_range
        # Expected max radius decreases with height (cone shape)
        # At bottom (height=0): max_radius = 1.0, at top (height=1): max_radius = 0.1
        max_radius = 1.0 - 0.9 * height_ratio
        actual_radius = np.sqrt(x**2 + z**2)

        if actual_radius > max_radius * 1.5:  # 50% tolerance
            cone_outliers.append(idx)

    if cone_outliers:
        print(f"  Found {len(cone_outliers)} points outside cone shape, clamping...")
        for idx in cone_outliers:
            x, y, z = result[idx]
            height_ratio = (y - y_min) / y_range
            max_radius = 1.0 - 0.9 * height_ratio
            actual_radius = np.sqrt(x**2 + z**2)

            if actual_radius > 0:
                # Scale down to fit within cone
                scale = max_radius / actual_radius
                result[idx] = (x * scale, y, z * scale)

    return result


def simple_triangulate(pos_a: dict, pos_b: dict) -> dict[int, tuple[float, float, float]]:
    """
    Simple triangulation assuming orthographic projection and cameras at ~90°.

    This assumes:
    - Camera A (e.g., "front") gives us (X, Y) in image -> maps to (tree_X, tree_Y)
    - Camera B (e.g., "side") gives us (X, Y) in image -> maps to (tree_Z, tree_Y)
    - Y coordinate should be similar from both views (both at same height)

    Returns normalized coordinates in range [-1, 1] with origin at tree center-bottom.
    """
    coords_3d = {}

    positions_a = pos_a["positions"]
    positions_b = pos_b["positions"]
    img_size_a = pos_a["image_size"]
    img_size_b = pos_b["image_size"]

    # Find common LEDs
    common_leds = set(positions_a.keys()) & set(positions_b.keys())
    print(f"Common LEDs found in both angles: {len(common_leds)}")

    # First pass: collect raw coordinates
    raw_coords = {}
    for led_str in common_leds:
        led_idx = int(led_str)
        pa = positions_a[led_str]
        pb = positions_b[led_str]

        # Normalized coordinates (0 to 1, origin at top-left)
        xa_norm = pa["x_norm"]
        ya_norm = pa["y_norm"]
        xb_norm = pb["x_norm"]
        yb_norm = pb["y_norm"]

        # Convert to centered coordinates (-0.5 to 0.5)
        # X: center of image = 0
        # Y: bottom of image = 0, top = 1 (flip Y so up is positive)
        xa_centered = xa_norm - 0.5
        ya_centered = 1.0 - ya_norm  # Flip Y
        xb_centered = xb_norm - 0.5
        yb_centered = 1.0 - yb_norm

        # Map to 3D:
        # - Front view X -> tree X
        # - Side view X -> tree Z
        # - Average of Y values -> tree Y
        tree_x = xa_centered
        tree_z = xb_centered
        tree_y = (ya_centered + yb_centered) / 2

        raw_coords[led_idx] = (tree_x, tree_y, tree_z)

    if not raw_coords:
        return {}

    # Normalize to [-1, 1] range based on actual extent
    all_x = [c[0] for c in raw_coords.values()]
    all_y = [c[1] for c in raw_coords.values()]
    all_z = [c[2] for c in raw_coords.values()]

    # Find bounds
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    # Calculate ranges
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    z_range = z_max - z_min if z_max > z_min else 1

    # Assume circular cross-section: normalize X and Z by the same factor
    # Use average of X and Z ranges for horizontal scaling
    xz_range = (x_range + z_range) / 2
    print(f"  X range: {x_range:.3f}, Z range: {z_range:.3f}, using avg: {xz_range:.3f}")

    # Scale factor: use max of horizontal (xz) and vertical (y) for overall scaling
    max_range = max(xz_range, y_range)

    # Center coordinates
    x_center = (x_max + x_min) / 2
    z_center = (z_max + z_min) / 2

    # Normalize: center X and Z with equal scaling, keep Y with bottom at 0
    for led_idx, (x, y, z) in raw_coords.items():
        # Scale X and Z by the same factor to preserve circular cross-section
        norm_x = (x - x_center) / xz_range * 2
        norm_y = (y - y_min) / max_range * 2  # Bottom at 0
        norm_z = (z - z_center) / xz_range * 2
        coords_3d[led_idx] = (norm_x, norm_y, norm_z)

    return coords_3d


def opencv_triangulate(pos_a: dict, pos_b: dict,
                       focal_length: float = 1000) -> dict[int, tuple[float, float, float]]:
    """
    More sophisticated triangulation using OpenCV.

    Uses the fundamental matrix to estimate camera poses, then triangulates.
    This handles non-perpendicular camera angles.
    """
    positions_a = pos_a["positions"]
    positions_b = pos_b["positions"]
    img_size = (pos_a["image_size"]["width"], pos_a["image_size"]["height"])

    # Find common LEDs and build point arrays
    common_leds = sorted(set(positions_a.keys()) & set(positions_b.keys()), key=int)
    print(f"Common LEDs: {len(common_leds)}")

    if len(common_leds) < 8:
        print("Need at least 8 common points for fundamental matrix estimation")
        return {}

    points_a = np.array([[positions_a[k]["x"], positions_a[k]["y"]] for k in common_leds], dtype=np.float64)
    points_b = np.array([[positions_b[k]["x"], positions_b[k]["y"]] for k in common_leds], dtype=np.float64)

    # Estimate camera intrinsic matrix (approximate)
    cx, cy = img_size[0] / 2, img_size[1] / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Find fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 3.0, 0.99)

    if F is None:
        print("Failed to find fundamental matrix")
        return simple_triangulate(pos_a, pos_b)

    inliers = mask.ravel().astype(bool)
    print(f"Fundamental matrix inliers: {inliers.sum()}/{len(common_leds)}")

    # Essential matrix from fundamental
    E = K.T @ F @ K

    # Recover pose (rotation and translation)
    _, R, t, pose_mask = cv2.recoverPose(E, points_a[inliers], points_b[inliers], K)

    # Camera projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Triangulate all points
    points_4d = cv2.triangulatePoints(P1, P2, points_a.T, points_b.T)
    points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous

    # Build result dict
    coords_3d = {}
    for i, led_str in enumerate(common_leds):
        led_idx = int(led_str)
        x, y, z = points_3d[:, i]
        coords_3d[led_idx] = (float(x), float(y), float(z))

    # Normalize to [-1, 1]
    coords_3d = normalize_coordinates(coords_3d)

    return coords_3d


def normalize_coordinates(coords: dict[int, tuple[float, float, float]]
                         ) -> dict[int, tuple[float, float, float]]:
    """Normalize coordinates to [-1, 1] range with Y=0 at bottom."""
    if not coords:
        return {}

    all_x = [c[0] for c in coords.values()]
    all_y = [c[1] for c in coords.values()]
    all_z = [c[2] for c in coords.values()]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)

    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    z_range = z_max - z_min if z_max > z_min else 1
    max_range = max(x_range, y_range, z_range)

    x_center = (x_max + x_min) / 2
    z_center = (z_max + z_min) / 2

    normalized = {}
    for led_idx, (x, y, z) in coords.items():
        norm_x = (x - x_center) / max_range * 2
        norm_y = (y - y_min) / max_range * 2
        norm_z = (z - z_center) / max_range * 2
        normalized[led_idx] = (norm_x, norm_y, norm_z)

    return normalized


def interpolate_missing(coords: dict[int, tuple[float, float, float]],
                        total_leds: int = LED_COUNT) -> dict[int, tuple[float, float, float]]:
    """
    Interpolate positions for missing LEDs based on neighbors.
    LEDs are in a string, so missing ones should be between their neighbors.
    """
    if len(coords) == total_leds:
        return coords

    result = dict(coords)
    missing = set(range(total_leds)) - set(coords.keys())
    print(f"Interpolating {len(missing)} missing LEDs...")

    for led in missing:
        # Find nearest neighbors that we have
        prev_led = led - 1
        next_led = led + 1

        while prev_led >= 0 and prev_led not in result:
            prev_led -= 1
        while next_led < total_leds and next_led not in result:
            next_led += 1

        if prev_led >= 0 and next_led < total_leds:
            # Interpolate between neighbors
            p1 = np.array(result[prev_led])
            p2 = np.array(result[next_led])
            t = (led - prev_led) / (next_led - prev_led)
            interp = p1 + t * (p2 - p1)
            result[led] = tuple(interp)
        elif prev_led >= 0:
            # Extrapolate from previous
            result[led] = result[prev_led]
        elif next_led < total_leds:
            # Extrapolate from next
            result[led] = result[next_led]

    return result


def save_coordinates_csv(coords: dict[int, tuple[float, float, float]], path: Path):
    """Save coordinates in CSV format (extended GIFT-like)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "z", "confidence"])

        for i in range(LED_COUNT):
            if i in coords:
                x, y, z = coords[i]
                conf = 1.0
            else:
                x, y, z = 0, 0, 0
                conf = 0.0
            writer.writerow([i, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{conf:.2f}"])

    print(f"Saved CSV: {path}")


def save_coordinates_json(coords: dict[int, tuple[float, float, float]], path: Path):
    """Save coordinates in JSON format."""
    data = {
        "led_count": LED_COUNT,
        "found_count": len(coords),
        "coordinate_system": {
            "x": "left-right",
            "y": "up (0 = bottom)",
            "z": "front-back"
        },
        "range": "[-1, 1] normalized",
        "coordinates": [
            {"x": coords[i][0], "y": coords[i][1], "z": coords[i][2]}
            if i in coords else None
            for i in range(LED_COUNT)
        ]
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {path}")


def save_coordinates_compact(coords: dict[int, tuple[float, float, float]], path: Path):
    """Save coordinates in compact format for Pico (one line per LED)."""
    with open(path, "w") as f:
        for i in range(LED_COUNT):
            if i in coords:
                x, y, z = coords[i]
                f.write(f"{x:.4f},{y:.4f},{z:.4f}\n")
            else:
                f.write("null\n")

    print(f"Saved compact: {path}")


def visualize_3d(coords: dict[int, tuple[float, float, float]]):
    """Simple ASCII visualization of 3D coordinates (top-down view)."""
    if not coords:
        return

    # Create a simple grid
    grid_size = 40
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

    for led_idx, (x, y, z) in coords.items():
        # Map X, Z to grid (top-down view)
        gx = int((x + 1) / 2 * (grid_size - 1))
        gz = int((z + 1) / 2 * (grid_size - 1))
        gx = max(0, min(grid_size - 1, gx))
        gz = max(0, min(grid_size - 1, gz))

        # Mark position (later LEDs overwrite earlier)
        grid[gz][gx] = '.'

    print("\nTop-down view (X horizontal, Z vertical):")
    print("+" + "-" * grid_size + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_size + "+")


def main():
    parser = argparse.ArgumentParser(description="Triangulate 3D LED positions")
    parser.add_argument("--angles", default="front,side",
                        help="Comma-separated angle names (need at least 2)")
    parser.add_argument("--method", choices=["simple", "opencv"], default="simple",
                        help="Triangulation method")
    parser.add_argument("--format", choices=["csv", "json", "both"], default="both",
                        help="Output format")
    parser.add_argument("--interpolate", action="store_true",
                        help="Interpolate missing LED positions")
    parser.add_argument("--visualize", action="store_true",
                        help="Show ASCII visualization")
    parser.add_argument("--flip-y", action="store_true",
                        help="Flip Y axis (if tree appears upside down)")
    parser.add_argument("--fix-outliers", action="store_true", default=True,
                        help="Fix outlier points far from neighbors (default: on)")
    parser.add_argument("--no-fix-outliers", action="store_false", dest="fix_outliers",
                        help="Disable outlier fixing")
    parser.add_argument("--outlier-threshold", type=float, default=3.0,
                        help="Outlier threshold (multiple of median neighbor distance)")
    parser.add_argument("--tree-width", type=float, default=None,
                        help="Expected tree width at base (any unit, e.g. 4 for 4ft)")
    parser.add_argument("--tree-height", type=float, default=None,
                        help="Expected tree height (same unit as width, e.g. 6 for 6ft)")
    parser.add_argument("--auto-align", action="store_true", default=False,
                        help="Auto-align tree vertical using PCA (experimental)")
    parser.add_argument("--no-auto-align", action="store_false", dest="auto_align",
                        help="Disable auto vertical alignment (default)")

    args = parser.parse_args()

    angles = [a.strip() for a in args.angles.split(",")]
    if len(angles) < 2:
        print("Need at least 2 angles for triangulation")
        return

    # Load positions
    print(f"Loading positions for angles: {angles}")
    try:
        pos_data = [load_positions(angle) for angle in angles]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run analyze.py first to generate position files")
        return

    for i, data in enumerate(pos_data):
        print(f"  {angles[i]}: {data['found_count']} LEDs found")

    # Triangulate (using first two angles)
    print(f"\nTriangulating using {args.method} method...")
    if args.method == "opencv":
        coords = opencv_triangulate(pos_data[0], pos_data[1])
    else:
        coords = simple_triangulate(pos_data[0], pos_data[1])

    print(f"Triangulated {len(coords)} LED positions")

    if not coords:
        print("No coordinates computed!")
        return

    # Auto-align vertical axis using PCA
    if args.auto_align:
        print("\nAuto-aligning vertical axis...")
        coords = auto_align_vertical(coords)

    # Fix outliers (before other processing)
    if args.fix_outliers:
        print("\nFixing outliers...")
        coords = fix_outliers(coords, threshold=args.outlier_threshold)

    # Apply tree aspect ratio correction
    if args.tree_width and args.tree_height:
        expected_ratio = args.tree_width / args.tree_height  # e.g., 4/6 = 0.67
        # Current coords have X, Z in some range and Y in some range
        # We want the X/Z span to be (expected_ratio) times the Y span
        y_vals = [c[1] for c in coords.values()]
        x_vals = [c[0] for c in coords.values()]
        z_vals = [c[2] for c in coords.values()]

        current_y_span = max(y_vals) - min(y_vals)
        current_xz_span = max(max(x_vals) - min(x_vals), max(z_vals) - min(z_vals))

        if current_y_span > 0 and current_xz_span > 0:
            # Scale X and Z so that xz_span / y_span = expected_ratio
            target_xz_span = current_y_span * expected_ratio
            scale_factor = target_xz_span / current_xz_span
            print(f"Applying aspect ratio correction: {args.tree_width}:{args.tree_height} = {expected_ratio:.2f}")
            print(f"  Scaling X/Z by {scale_factor:.2f}")
            coords = {idx: (x * scale_factor, y, z * scale_factor) for idx, (x, y, z) in coords.items()}

    # Flip Y if requested
    if args.flip_y:
        print("Flipping Y axis...")
        max_y = max(c[1] for c in coords.values())
        coords = {idx: (x, max_y - y, z) for idx, (x, y, z) in coords.items()}

    # Interpolate missing
    if args.interpolate:
        coords = interpolate_missing(coords)
        print(f"After interpolation: {len(coords)} LED positions")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.format in ["csv", "both"]:
        save_coordinates_csv(coords, OUTPUT_DIR / "coords.csv")

    if args.format in ["json", "both"]:
        save_coordinates_json(coords, OUTPUT_DIR / "coords.json")

    # Always save compact format for Pico
    save_coordinates_compact(coords, OUTPUT_DIR / "coords_compact.txt")

    # Visualize
    if args.visualize:
        visualize_3d(coords)

    print("\nDone! Coordinate files saved to:", OUTPUT_DIR)
    print("\nUpload to Pico:")
    print('  curl -X POST --data-binary @coordinates/coords_compact.txt "http://192.168.2.149:8080/update?file=coords_compact.txt"')


if __name__ == "__main__":
    main()
