#!/usr/bin/env python3
"""Estimate Google Street View Static API volume, coverage and cost.

Offline by default. With ``--probe`` (and ``GOOGLE_MAPS_API_KEY`` set) it queries the
*free* Street View metadata endpoint only: no image requests are made and nothing is
downloaded or stored. See ``docs/street_view_api_and_3d.md`` for the licensing notes --
notably Google Maps Platform ToS 3.2.3(a)(ii), (b), (c)(v) and (c)(vii), which rule out
bulk image downloads, caching, city-scale tree indexing and ML training on GSV content.

Pricing source: Google Maps Platform global price list, SKU ``Static Street View``
(9BD0-A2EE-44C3), checked 2026-10-27. Metadata SKU is free and unlimited.

Examples
--------
    python3 gsv_survey_planner.py --length-km 40 --spacing-m 10
    python3 gsv_survey_planner.py --length-km 2100 --spacing-m 10 --headings 4
    python3 gsv_survey_planner.py --length-km 2100 --spacing-m 10 --months 6 --json
    python3 gsv_survey_planner.py --points survey_points.csv --probe --limit 100
    python3 gsv_survey_planner.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

# Static Street View SKU (9BD0-A2EE-44C3), USD per 1,000 requests.
# (upper bound of the band, price per 1,000); None = no upper bound.
PRICE_TIERS = (
    (100_000, 7.00),
    (500_000, 5.60),
    (1_000_000, 4.20),
    (5_000_000, 2.10),
    (None, 0.53),
)
FREE_CALLS_PER_MONTH = 10_000
MAX_IMAGE_PX = 640
MAX_FOV = 120
MAX_RADIUS_M = 50


def cost_usd(requests_per_month: int) -> float:
    """Monthly cost in USD for ``requests_per_month`` Static Street View calls.

    The 10,000 free calls are consumed from the cheapest band first, which is how the
    published free usage cap behaves in practice.
    """
    if requests_per_month < 0:
        raise ValueError("requests_per_month must be >= 0")
    billable = max(0, requests_per_month - FREE_CALLS_PER_MONTH)
    cost = 0.0
    lower = 0
    for upper, price_per_thousand in PRICE_TIERS:
        if billable <= lower:
            break
        top = billable if upper is None else min(billable, upper)
        cost += (top - lower) * price_per_thousand / 1000.0
        lower = top
    return round(cost, 2)


def panoramas_for_length(length_km: float, spacing_m: float) -> int:
    """Number of capture points along ``length_km`` at ``spacing_m`` metres."""
    if length_km <= 0:
        raise ValueError("length_km must be > 0")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")
    return int(math.floor(length_km * 1000.0 / spacing_m)) + 1


@dataclass
class SurveyPlan:
    """Volume and cost of a Street View survey."""

    length_km: float | None
    spacing_m: float | None
    locations: int
    headings: int
    image_requests: int
    months: int
    requests_per_month: int
    free_calls_per_month: int
    billable_per_month: int
    cost_per_month_usd: float
    total_cost_usd: float


def build_plan(
    locations: int,
    headings: int = 1,
    months: int = 1,
    length_km: float | None = None,
    spacing_m: float | None = None,
) -> SurveyPlan:
    """Combine locations, headings and months into a :class:`SurveyPlan`."""
    if locations <= 0:
        raise ValueError("locations must be > 0")
    if headings <= 0:
        raise ValueError("headings must be > 0")
    if months <= 0:
        raise ValueError("months must be > 0")

    requests = locations * headings
    per_month = int(math.ceil(requests / months))
    monthly_cost = cost_usd(per_month)
    return SurveyPlan(
        length_km=length_km,
        spacing_m=spacing_m,
        locations=locations,
        headings=headings,
        image_requests=requests,
        months=months,
        requests_per_month=per_month,
        free_calls_per_month=FREE_CALLS_PER_MONTH,
        billable_per_month=max(0, per_month - FREE_CALLS_PER_MONTH),
        cost_per_month_usd=monthly_cost,
        total_cost_usd=round(monthly_cost * months, 2),
    )



def load_points_csv(path: str) -> list[tuple[float, float]]:
    """Read ``lat``/``lng`` pairs from a CSV.

    Accepts ``lat``/``lng``, ``latitude``/``longitude`` or ``y``/``x`` as headers.
    """
    aliases = (
        ("lat", "latitude", "y"),
        ("lng", "lon", "long", "longitude", "x"),
    )
    points: list[tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        lat_key = next((fields[a] for a in aliases[0] if a in fields), None)
        lng_key = next((fields[a] for a in aliases[1] if a in fields), None)
        if lat_key is None or lng_key is None:
            raise ValueError(
                f"{path}: need lat/lng-like columns, found {reader.fieldnames}"
            )
        for row in reader:
            if not row.get(lat_key) or not row.get(lng_key):
                continue
            points.append((float(row[lat_key]), float(row[lng_key])))
    if not points:
        raise ValueError(f"{path}: no coordinate rows found")
    return points


def probe_coverage(
    points: list[tuple[float, float]],
    api_key: str,
    radius: int = MAX_RADIUS_M,
    timeout: float = 15.0,
    pause: float = 0.05,
) -> list[dict]:
    """Check panorama availability with the free metadata endpoint.

    Free and unlimited per the price list, but keep ``pause`` polite. Only ``pano_id``
    may be retained under the Maps Service Specific Terms; treat the rest as transient.
    """
    results: list[dict] = []
    for index, (lat, lng) in enumerate(points, start=1):
        query = urllib.parse.urlencode(
            {"location": f"{lat},{lng}", "radius": radius, "key": api_key}
        )
        record = {"lat": lat, "lng": lng, "status": "ERROR", "pano_id": None, "date": None}
        try:
            with urllib.request.urlopen(f"{METADATA_URL}?{query}", timeout=timeout) as resp:
                payload = json.load(resp)
            record["status"] = payload.get("status", "UNKNOWN")
            record["pano_id"] = payload.get("pano_id")
            record["date"] = payload.get("date")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            record["error"] = str(exc)
        results.append(record)
        if pause and index < len(points):
            time.sleep(pause)
    return results


def summarise_probe(results: list[dict]) -> dict:
    """Coverage summary for a metadata probe run."""
    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    hit = counts.get("OK", 0)
    return {
        "points_checked": len(results),
        "status_counts": dict(sorted(counts.items())),
        "coverage_pct": round(100.0 * hit / len(results), 1) if results else 0.0,
    }


def format_plan(plan: SurveyPlan) -> str:
    """Human readable summary of a :class:`SurveyPlan`."""
    lines = ["Street View Static API survey plan", "-" * 34]
    if plan.length_km:
        lines.append(f"survey length          : {plan.length_km:,.1f} km")
        lines.append(f"capture spacing        : {plan.spacing_m:,.0f} m")
    lines += [
        f"locations              : {plan.locations:,}",
        f"headings per location  : {plan.headings}",
        f"image requests         : {plan.image_requests:,}",
        f"months                 : {plan.months} (<= {plan.requests_per_month:,}/month)",
        f"free allowance/month   : {plan.free_calls_per_month:,}",
        f"billable per month     : {plan.billable_per_month:,}",
        f"cost per month         : ${plan.cost_per_month_usd:,.2f}",
        f"total cost             : ${plan.total_cost_usd:,.2f}",
        "",
        f"Image limit per request: {MAX_IMAGE_PX}x{MAX_IMAGE_PX} px, "
        f"fov <= {MAX_FOV} deg, radius <= {MAX_RADIUS_M} m.",
        "Metadata calls (coverage checks) are free and uncapped.",
    ]
    return "\n".join(lines)


def _self_test() -> int:
    """Assert the worked examples quoted in docs/street_view_api_and_3d.md."""
    cases = [
        (panoramas_for_length(40, 10), 1, 0.00),
        (panoramas_for_length(200, 10), 1, 70.01),
        (panoramas_for_length(200, 10), 4, 490.03),
        (panoramas_for_length(2_100, 10), 1, 1_260.01),
        (panoramas_for_length(2_100, 10), 4, 4_326.02),
    ]
    failures = 0
    for locations, headings, expected in cases:
        plan = build_plan(locations, headings)
        ok = plan.total_cost_usd == expected
        failures += 0 if ok else 1
        print(
            f"[{'ok' if ok else 'FAIL'}] {locations:>7,} loc x {headings} heading(s) -> "
            f"${plan.total_cost_usd:>9,.2f} (expected ${expected:,.2f})"
        )
    if cost_usd(FREE_CALLS_PER_MONTH) != 0.0:
        failures += 1
        print("[FAIL] the free allowance is not free")
    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan and price a Google Street View Static API survey (offline by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--length-km", type=float, help="length of the survey corridor in km")
    parser.add_argument(
        "--spacing-m", type=float, default=10.0, help="capture spacing in metres (default: 10)"
    )
    parser.add_argument(
        "--headings", type=int, default=1, help="image requests per location, e.g. 4 for 360 deg"
    )
    parser.add_argument("--months", type=int, default=1, help="spread the survey over N months")
    parser.add_argument("--points", help="CSV of lat/lng points that define the survey")
    parser.add_argument(
        "--probe", action="store_true",
        help="check coverage with the free metadata endpoint",
    )
    parser.add_argument("--limit", type=int, default=200, help="max points to probe (default: 200)")
    parser.add_argument(
        "--radius", type=int, default=MAX_RADIUS_M, help="panorama search radius in metres"
    )
    parser.add_argument("--json", action="store_true", help="emit machine readable JSON")
    parser.add_argument(
        "--self-test", action="store_true", help="verify the documented price examples"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    points: list[tuple[float, float]] | None = None
    if args.points:
        try:
            points = load_points_csv(args.points)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if points is not None:
        locations = len(points)
        length_km, spacing_m = args.length_km, args.spacing_m
    elif args.length_km:
        locations = panoramas_for_length(args.length_km, args.spacing_m)
        length_km, spacing_m = args.length_km, args.spacing_m
    else:
        parser.error("provide --length-km or --points (or --self-test)")

    try:
        plan = build_plan(
            locations,
            headings=args.headings,
            months=args.months,
            length_km=length_km,
            spacing_m=spacing_m,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    coverage = None
    if args.probe:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            print(
                "error: --probe needs GOOGLE_MAPS_API_KEY in the environment "
                "(metadata calls are free, but still require a key)",
                file=sys.stderr,
            )
            return 2
        if points is None:
            print("error: --probe needs --points", file=sys.stderr)
            return 2
        coverage = summarise_probe(
            probe_coverage(points[: max(1, args.limit)], api_key, radius=args.radius)
        )

    if args.json:
        payload = asdict(plan)
        payload["pricing"] = {
            "sku": "Static Street View 9BD0-A2EE-44C3",
            "free_calls_per_month": FREE_CALLS_PER_MONTH,
            "tiers_usd_per_1000": [
                {"up_to": upper, "price": price} for upper, price in PRICE_TIERS
            ],
        }
        if coverage is not None:
            payload["coverage"] = coverage
        print(json.dumps(payload, indent=2))
        return 0

    print(format_plan(plan))
    if coverage is not None:
        print("")
        print("Coverage probe (free metadata endpoint)")
        print("-" * 34)
        print(f"points checked         : {coverage['points_checked']:,}")
        print(f"coverage               : {coverage['coverage_pct']}%")
        for status, count in coverage["status_counts"].items():
            print(f"  {status:<22}: {count:,}")
        print("Only pano_id may be retained; do not store images or derived geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

