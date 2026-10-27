#!/usr/bin/env python3
"""Two-source tree-detection fusion, after Rodríguez-Puerta et al. (2022).

The paper ("Mapping Tree Canopy in Urban Environments Using Point Clouds from Airborne
Laser Scanning and Street Level Imagery", Sensors 22(9):3269) does not rebuild geometry
from street-level imagery. It runs two independent *detections* -- one from ALS canopy
height models, one from street-level images -- and then uses each source to remove the
other's false positives, merging candidates that fall within 4 m and scoring against the
municipal inventory with a 5 m tolerance.

This module implements exactly those operations, dependency-free, so the same architecture
can be applied with legal inputs: DeepForest crowns vs LiDAR CHM treetops vs the HyD / ArchSD
inventory points.

Functions
---------
    haversine_m(a, b)                         great-circle distance in metres
    dedupe(points, radius_m)                  merge candidates closer than radius_m
    match_points(detected, reference, tol_m)  one-to-one matching -> p / r / F1
    cross_validate(primary, secondary, r_m)   keep primary points supported by secondary

Usage
-----
    python3 tree_detection_fusion.py --detected lidar_tops.csv --reference inventory.csv
    python3 tree_detection_fusion.py --detected lidar_tops.csv --reference inventory.csv \\
        --cross-check deepforest.csv --cross-radius 4 --dedupe 4
    python3 tree_detection_fusion.py --self-test

CSV files need lat/lng-like columns (``lat``/``lng``, ``latitude``/``longitude`` or ``y``/``x``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two ``(lat, lng)`` points, in metres."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


class _Grid:
    """Tiny lat/lng grid index so matching stays fast on city-sized point sets."""

    def __init__(self, points: list[tuple[float, float]], cell_m: float):
        self.points = points
        self.cell_m = max(cell_m, 1.0)
        mean_lat = sum(p[0] for p in points) / len(points) if points else 0.0
        self.dlat = self.cell_m / 111_320.0
        self.dlon = self.cell_m / (111_320.0 * max(math.cos(math.radians(mean_lat)), 1e-6))
        self.cells: dict[tuple[int, int], list[int]] = {}
        for index, (lat, lng) in enumerate(points):
            self.cells.setdefault(self._key(lat, lng), []).append(index)

    def _key(self, lat: float, lng: float) -> tuple[int, int]:
        return int(math.floor(lat / self.dlat)), int(math.floor(lng / self.dlon))

    def neighbours(self, point: tuple[float, float], radius_m: float):
        """Yield ``(index, distance_m)`` for stored points within ``radius_m``."""
        base_lat, base_lng = self._key(*point)
        rings = max(1, int(math.ceil(radius_m / self.cell_m)))
        for i in range(base_lat - rings, base_lat + rings + 1):
            for j in range(base_lng - rings, base_lng + rings + 1):
                for index in self.cells.get((i, j), ()):
                    distance = haversine_m(point, self.points[index])
                    if distance <= radius_m:
                        yield index, distance


def load_points_csv(path: str) -> list[tuple[float, float]]:
    """Read ``(lat, lng)`` pairs from a CSV with lat/lng-like headers."""
    aliases = (("lat", "latitude", "y"), ("lng", "lon", "long", "longitude", "x"))
    points: list[tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        lat_key = next((fields[a] for a in aliases[0] if a in fields), None)
        lng_key = next((fields[a] for a in aliases[1] if a in fields), None)
        if lat_key is None or lng_key is None:
            raise ValueError(f"{path}: need lat/lng-like columns, found {reader.fieldnames}")
        for row in reader:
            if not row.get(lat_key) or not row.get(lng_key):
                continue
            points.append((float(row[lat_key]), float(row[lng_key])))
    if not points:
        raise ValueError(f"{path}: no coordinate rows found")
    return points


@dataclass
class DedupeResult:
    """Output of :func:`dedupe`."""

    centroids: list[tuple[float, float]]
    cluster_sizes: list[int]

    @property
    def merged(self) -> int:
        return sum(size - 1 for size in self.cluster_sizes)


@dataclass
class MatchReport:
    """Precision / recall / F1 of detections against a reference set."""

    detected: int
    reference: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    tolerance_m: float


def dedupe(points: list[tuple[float, float]], radius_m: float = 4.0) -> DedupeResult:
    """Merge candidates closer than ``radius_m`` into their centroid.

    Mirrors the paper's step 2.3.5 (they buffer, polygonise and take centroids at 4 m).
    Greedy single pass: a point joins the nearest existing cluster centroid within the
    radius, otherwise it seeds a new cluster.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    clusters: list[list[float]] = []  # [sum_lat, sum_lng, count]
    grid: dict[tuple[int, int], list[int]] = {}
    cell_m = radius_m
    dlat = cell_m / 111_320.0
    mean_lat = sum(p[0] for p in points) / len(points) if points else 0.0
    dlon = cell_m / (111_320.0 * max(math.cos(math.radians(mean_lat)), 1e-6))

    def key(lat: float, lng: float) -> tuple[int, int]:
        return int(math.floor(lat / dlat)), int(math.floor(lng / dlon))

    for lat, lng in points:
        base = key(lat, lng)
        best_index, best_distance = None, radius_m
        for i in (base[0] - 1, base[0], base[0] + 1):
            for j in (base[1] - 1, base[1], base[1] + 1):
                for cluster_index in grid.get((i, j), ()):
                    sum_lat, sum_lng, count = clusters[cluster_index]
                    centroid = (sum_lat / count, sum_lng / count)
                    distance = haversine_m((lat, lng), centroid)
                    if distance <= best_distance:
                        best_index, best_distance = cluster_index, distance
        if best_index is None:
            clusters.append([lat, lng, 1])
            grid.setdefault(base, []).append(len(clusters) - 1)
        else:
            clusters[best_index][0] += lat
            clusters[best_index][1] += lng
            clusters[best_index][2] += 1

    centroids = [(c[0] / c[2], c[1] / c[2]) for c in clusters]
    return DedupeResult(centroids=centroids, cluster_sizes=[c[2] for c in clusters])


def match_points(
    detected: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    tolerance_m: float = 5.0,
) -> MatchReport:
    """One-to-one match of detections to reference trees (the paper's accuracy rule).

    A detection counts as a true positive when it is within ``tolerance_m`` of a reference
    point and is the closest free detection to it -- exactly the criterion used against the
    Pamplona municipal inventory at 5 m.
    """
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be > 0")
    grid = _Grid(detected, cell_m=max(tolerance_m, 1.0))
    used: set[int] = set()
    for reference_point in reference:
        best_index, best_distance = None, tolerance_m
        for index, distance in grid.neighbours(reference_point, tolerance_m):
            if index in used:
                continue
            if distance <= best_distance:
                best_index, best_distance = index, distance
        if best_index is not None:
            used.add(best_index)

    tp = len(used)
    fp = len(detected) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return MatchReport(
        detected=len(detected),
        reference=len(reference),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(100 * precision, 2),
        recall=round(100 * recall, 2),
        f1=round(100 * f1, 2),
        tolerance_m=tolerance_m,
    )


def cross_validate(
    primary: list[tuple[float, float]],
    secondary: list[tuple[float, float]],
    radius_m: float = 4.0,
) -> tuple[list[tuple[float, float]], int]:
    """Keep primary points supported by at least one secondary point.

    This is the paper's "ALS -> GSV debugging" stage: the second, independent source only
    votes on *existence*. Returns ``(kept_points, rejected_count)``.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    grid = _Grid(secondary, cell_m=max(radius_m, 1.0))
    kept = [p for p in primary if any(True for _ in grid.neighbours(p, radius_m))]
    return kept, len(primary) - len(kept)



def _fmt_report(label: str, report: MatchReport) -> str:
    return (
        f"{label:<22}: TP {report.true_positives:,}  FP {report.false_positives:,}  "
        f"FN {report.false_negatives:,}  p {report.precision:.1f}%  "
        f"r {report.recall:.1f}%  F1 {report.f1:.1f}%"
    )


def _self_test() -> int:
    """Synthetic checks for dedupe, matching and cross-validation."""
    lat0, lng0 = 22.3193, 114.1694
    coslat = math.cos(math.radians(lat0))

    def pt(north_m: float, east_m: float) -> tuple[float, float]:
        return (lat0 + north_m / 111_320.0, lng0 + east_m / (111_320.0 * coslat))

    truth = [pt(0, 0), pt(0, 10), pt(0, 20), pt(10, 0), pt(10, 10)]
    checks: list[tuple[str, object, object]] = []

    perfect = match_points(list(truth), truth, 5.0)
    checks.append(("perfect precision", perfect.precision, 100.0))
    checks.append(("perfect recall", perfect.recall, 100.0))

    noisy = [
        pt(0.5, 0.5), pt(0.0, 0.8), pt(0.3, 10.4), pt(0.0, 20.2),
        pt(10.2, 0.1), pt(10.0, 10.3), pt(30.0, 30.0),
    ]
    report = match_points(noisy, truth, 5.0)
    checks.append(("noisy TP", report.true_positives, 5))
    checks.append(("noisy FP", report.false_positives, 2))
    checks.append(("noisy FN", report.false_negatives, 0))

    merged = dedupe(noisy, 4.0)
    checks.append(("dedupe clusters", len(merged.centroids), 6))
    checks.append(("dedupe merged", merged.merged, 1))

    kept, rejected = cross_validate(truth + [pt(40, 40)], truth, 4.0)
    checks.append(("cross-validate kept", len(kept), 5))
    checks.append(("cross-validate rejected", rejected, 1))

    failures = 0
    for name, actual, expected in checks:
        ok = actual == expected
        failures += 0 if ok else 1
        print(f"[{'ok' if ok else 'FAIL'}] {name:<24} {actual!r} (expected {expected!r})")
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fuse two independent tree detections (paper: ALS x street-level imagery).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detected", help="CSV of detections (e.g. LiDAR CHM treetops)")
    parser.add_argument("--reference", help="CSV of reference trees (e.g. municipal inventory)")
    parser.add_argument("--cross-check", help="CSV of an independent source used to filter detections")
    parser.add_argument("--tolerance", type=float, default=5.0, help="match tolerance in m (default 5)")
    parser.add_argument("--cross-radius", type=float, default=4.0, help="cross-check radius in m (default 4)")
    parser.add_argument("--dedupe", type=float, default=0.0, help="merge detections within N m (0 = off)")
    parser.add_argument("--json", action="store_true", help="emit machine readable JSON")
    parser.add_argument("--self-test", action="store_true", help="run synthetic checks")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.detected:
        parser.error("--detected is required (or use --self-test)")

    try:
        detections = load_points_csv(args.detected)
        reference = load_points_csv(args.reference) if args.reference else None
        secondary = load_points_csv(args.cross_check) if args.cross_check else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    raw = match_points(detections, reference, args.tolerance) if reference else None
    working, merged_count, rejected_count = detections, 0, 0
    if args.dedupe > 0:
        result = dedupe(working, args.dedupe)
        merged_count, working = result.merged, result.centroids
    if secondary is not None:
        working, rejected_count = cross_validate(working, secondary, args.cross_radius)
    fused = match_points(working, reference, args.tolerance) if reference else None

    if args.json:
        payload = {
            "detections": len(detections),
            "after_dedupe": len(detections) - merged_count,
            "after_cross_check": len(working),
            "merged": merged_count,
            "rejected": rejected_count,
            "raw": asdict(raw) if raw else None,
            "fused": asdict(fused) if fused else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("Tree detection fusion")
    print("-" * 22)
    print(f"{'detections':<22}: {len(detections):,}")
    if args.dedupe > 0:
        label = f"dedupe @ {args.dedupe:g} m"
        print(f"{label:<22}: -> {len(detections) - merged_count:,} ({merged_count:,} merged)")
    if secondary is not None:
        label = f"cross-check @ {args.cross_radius:g} m"
        print(f"{label:<22}: -> {len(working):,} ({rejected_count:,} rejected)")
    if reference:
        print(f"{'reference trees':<22}: {len(reference):,}")
        print("")
        if raw:
            print(_fmt_report("raw vs reference", raw))
        if fused:
            print(_fmt_report("fused vs reference", fused))
        print("")
        print("Reporting matches Rodríguez-Puerta et al. (2022): tolerance "
              f"{args.tolerance:g} m, merge radius {args.cross_radius:g} m.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

