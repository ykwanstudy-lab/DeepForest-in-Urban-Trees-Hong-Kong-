#!/usr/bin/env python3
"""Turn tree inventory points + panorama positions into a labelling manifest.

Street-level species / symptom models are limited by *labels*, not by images. Hong Kong
already publishes tree points with species (HyD, ArchSD). If you know where a tree is and
where each panorama was taken, you can crop that tree out of every frame that sees it and
inherit the inventory label -- no manual annotation for the species task.

This module does the geometry only (no image I/O), so it runs anywhere:

    bearing_distance()   compass bearing and ground distance between two points
    pano_pixel()         equirectangular pixel column/row for a bearing and elevation
    crop_window()        pixel box around a target of a given size at a given distance
    build_manifest()     one row per (tree, panorama) pair worth labelling

Output columns are directly usable by PIL / OpenCV / torchvision cropping:

    tree_id,species,pano_id,heading_deg,bearing_deg,distance_m,elevation_deg,
    u,v,crop_w,crop_h,u0,u1,v0,v1,wrap

Usage
-----
    python3 pano_label_manifest.py --panos panos.csv --trees trees.csv --out manifest.csv
    python3 pano_label_manifest.py --self-test

``panos.csv``: pano_id, lat, lng, heading_deg, width, height
``trees.csv``: tree_id, lat, lng, species [, crown_diameter_m] [, crown_height_m]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass

EARTH_RADIUS_M = 6_371_008.8
DEFAULT_CAMERA_HEIGHT_M = 2.5


def bearing_distance(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> tuple[float, float]:
    """Compass bearing (0-360, clockwise from north) and distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    distance = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing, distance


def pano_pixel(
    bearing_deg: float,
    heading_deg: float,
    width: int,
    height: int,
    elevation_deg: float = 0.0,
) -> tuple[float, float]:
    """Pixel column/row of a target in a levelled equirectangular panorama.

    Column 0 is 180 degrees left of ``heading_deg``, the centre column is straight ahead.
    Row 0 is +90 degrees elevation (zenith), the middle row is the horizon.
    """
    azimuth = (bearing_deg - heading_deg + 180.0) % 360.0
    u = azimuth / 360.0 * width
    v = (0.5 - elevation_deg / 180.0) * height
    return u, v


def crop_window(
    u: float,
    v: float,
    width: int,
    height: int,
    target_size_m: float,
    distance_m: float,
    padding: float = 0.25,
    max_fov_deg: float = 60.0,
) -> dict:
    """Pixel box that contains a target of ``target_size_m`` seen from ``distance_m``.

    The angular size is ``2*atan(size/2/distance)``; it is padded and capped at
    ``max_fov_deg`` so a close tree does not swallow the whole panorama.
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if target_size_m <= 0:
        raise ValueError("target_size_m must be > 0")
    angular = math.degrees(2 * math.atan(target_size_m / (2 * distance_m))) * (1 + padding)
    angular = min(angular, max_fov_deg)
    crop_w = angular / 360.0 * width
    crop_h = angular / 180.0 * height
    u0, u1 = u - crop_w / 2, u + crop_w / 2
    v0, v1 = v - crop_h / 2, v + crop_h / 2
    return {
        "crop_w": round(crop_w, 1),
        "crop_h": round(crop_h, 1),
        "u0": round(max(u0, 0.0), 1),
        "u1": round(min(u1, float(width)), 1),
        "v0": round(max(v0, 0.0), 1),
        "v1": round(min(v1, float(height)), 1),
        "wrap": u0 < 0 or u1 > width,
    }


@dataclass
class Panorama:
    pano_id: str
    lat: float
    lng: float
    heading_deg: float
    width: int = 8000
    height: int = 4000


@dataclass
class Tree:
    tree_id: str
    lat: float
    lng: float
    species: str = ""
    crown_diameter_m: float = 6.0
    crown_height_m: float = 8.0


def build_manifest(
    panoramas: list[Panorama],
    trees: list[Tree],
    per_tree: int = 3,
    min_distance_m: float = 3.0,
    max_distance_m: float = 25.0,
    ideal_distance_m: float = 10.0,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
) -> list[dict]:
    """One row per (tree, panorama) pair worth cropping for labelling."""
    if per_tree <= 0:
        raise ValueError("per_tree must be > 0")
    if min_distance_m <= 0 or max_distance_m <= min_distance_m:
        raise ValueError("need 0 < min_distance_m < max_distance_m")

    rows: list[dict] = []
    for tree in trees:
        candidates = []
        for pano in panoramas:
            bearing, distance = bearing_distance(pano.lat, pano.lng, tree.lat, tree.lng)
            if not (min_distance_m <= distance <= max_distance_m):
                continue
            candidates.append((abs(distance - ideal_distance_m), pano, bearing, distance))
        candidates.sort(key=lambda item: item[0])
        for _, pano, bearing, distance in candidates[:per_tree]:
            elevation = math.degrees(
                math.atan2(tree.crown_height_m - camera_height_m, distance)
            )
            u, v = pano_pixel(bearing, pano.heading_deg, pano.width, pano.height, elevation)
            box = crop_window(
                u, v, pano.width, pano.height, tree.crown_diameter_m, distance
            )
            rows.append(
                {
                    "tree_id": tree.tree_id,
                    "species": tree.species,
                    "pano_id": pano.pano_id,
                    "heading_deg": round(pano.heading_deg, 1),
                    "bearing_deg": round(bearing, 1),
                    "distance_m": round(distance, 1),
                    "elevation_deg": round(elevation, 1),
                    "u": round(u, 1),
                    "v": round(v, 1),
                    **box,
                }
            )
    return rows


def load_panoramas(path: str) -> list[Panorama]:
    out: list[Panorama] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("lat") or not row.get("lng"):
                continue
            out.append(
                Panorama(
                    pano_id=row.get("pano_id") or f"pano{len(out)}",
                    lat=float(row["lat"]),
                    lng=float(row["lng"]),
                    heading_deg=float(row.get("heading_deg") or 0.0),
                    width=int(float(row.get("width") or 8000)),
                    height=int(float(row.get("height") or 4000)),
                )
            )
    if not out:
        raise ValueError(f"{path}: no panorama rows")
    return out


def load_trees(path: str) -> list[Tree]:
    out: list[Tree] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("lat") or not row.get("lng"):
                continue
            out.append(
                Tree(
                    tree_id=str(row.get("tree_id") or f"T{len(out)}"),
                    lat=float(row["lat"]),
                    lng=float(row["lng"]),
                    species=(row.get("species") or "").strip(),
                    crown_diameter_m=float(row.get("crown_diameter_m") or 6.0),
                    crown_height_m=float(row.get("crown_height_m") or 8.0),
                )
            )
    if not out:
        raise ValueError(f"{path}: no tree rows")
    return out



FIELD_ORDER = [
    "tree_id", "species", "pano_id", "heading_deg", "bearing_deg", "distance_m",
    "elevation_deg", "u", "v", "crop_w", "crop_h", "u0", "u1", "v0", "v1", "wrap",
]


def _self_test() -> int:
    """Geometry checks; the numbers are hand-computed, not fitted."""
    lat0, lng0 = 22.3193, 114.1694
    checks: list[tuple[str, object, object]] = []

    def approx(actual: float, expected: float, tol: float = 0.5) -> bool:
        return abs(actual - expected) <= tol

    for name, (dlat, dlng), expected in [
        ("bearing north", (0.001, 0.0), 0.0),
        ("bearing east", (0.0, 0.001), 90.0),
        ("bearing south", (-0.001, 0.0), 180.0),
        ("bearing west", (0.0, -0.001), 270.0),
    ]:
        bearing, _ = bearing_distance(lat0, lng0, lat0 + dlat, lng0 + dlng)
        checks.append((name, approx(bearing, expected), True))

    _, dist = bearing_distance(lat0, lng0, lat0 + 0.001, lng0)
    checks.append(("111 m per 0.001 deg lat", approx(dist, 111.2, 2.0), True))

    W, H = 8000, 4000
    u, v = pano_pixel(90, 90, W, H)
    checks.append(("east tree in east-facing pano", approx(u, W / 2, 1), True))
    u, _ = pano_pixel(0, 90, W, H)
    checks.append(("north tree left of east heading", approx(u, W / 4, 1), True))
    _, v = pano_pixel(0, 0, W, H, elevation_deg=0)
    checks.append(("horizon at mid row", approx(v, H / 2, 1), True))
    _, v = pano_pixel(0, 0, W, H, elevation_deg=45)
    checks.append(("45 deg up at quarter row", approx(v, H / 4, 1), True))

    near = crop_window(W / 2, H / 2, W, H, 6.0, 10.0)
    far = crop_window(W / 2, H / 2, W, H, 6.0, 20.0)
    checks.append(("crop shrinks with distance", near["crop_w"] > far["crop_w"], True))
    checks.append(("crop capped at 60 deg", crop_window(W / 2, H / 2, W, H, 6.0, 1.0)["crop_w"], 1333.3))
    checks.append(("wrap flagged behind camera", crop_window(0.0, H / 2, W, H, 6.0, 10.0)["wrap"], True))

    panos = [Panorama("p1", lat0, lng0, 0.0, W, H)]
    trees = [Tree("T1", lat0 + 0.00009, lng0, "Ficus microcarpa"),      # ~10 m north
             Tree("T2", lat0 + 0.00036, lng0, "Bauhinia"),              # ~40 m north
             Tree("T3", lat0 - 0.00009, lng0, "Ficus benjamina")]       # ~10 m south
    rows = build_manifest(panos, trees, per_tree=1, max_distance_m=25)
    checks.append(("manifest keeps 2 of 3", len(rows), 2))
    checks.append(("ideal distance preferred", {r["tree_id"] for r in rows}, {"T1", "T3"}))
    checks.append(("T1 bearing ~0", round(rows[0]["bearing_deg"]), 0))
    checks.append(("T1 above horizon", rows[0]["v"] < H / 2, True))

    failures = 0
    for name, actual, expected in checks:
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name:<34} {actual!r}")
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a crop/label manifest from panoramas and inventory tree points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--panos", help="CSV: pano_id, lat, lng, heading_deg [, width, height]")
    parser.add_argument("--trees", help="CSV: tree_id, lat, lng, species [, crown_diameter_m, crown_height_m]")
    parser.add_argument("--out", help="output CSV (default: stdout)")
    parser.add_argument("--per-tree", type=int, default=3, help="views per tree (default 3)")
    parser.add_argument("--min-m", type=float, default=3.0, help="minimum distance in m (default 3)")
    parser.add_argument("--max-m", type=float, default=25.0, help="maximum distance in m (default 25)")
    parser.add_argument("--ideal-m", type=float, default=10.0, help="preferred distance in m (default 10)")
    parser.add_argument("--camera-height-m", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of CSV")
    parser.add_argument("--self-test", action="store_true", help="run geometry checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not (args.panos and args.trees):
        parser.error("--panos and --trees are required (or use --self-test)")

    try:
        rows = build_manifest(
            load_panoramas(args.panos),
            load_trees(args.trees),
            per_tree=args.per_tree,
            min_distance_m=args.min_m,
            max_distance_m=args.max_m,
            ideal_distance_m=args.ideal_m,
            camera_height_m=args.camera_height_m,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {"rows": rows, "count": len(rows)}
        print(json.dumps(payload, indent=2))
        return 0

    target = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(target, fieldnames=FIELD_ORDER)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            target.close()

    if args.out:
        trees = {row["tree_id"] for row in rows}
        print(
            f"wrote {len(rows):,} crops for {len(trees):,} trees to {args.out}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

