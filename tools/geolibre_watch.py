#!/usr/bin/env python3
"""Check whether the pinned GeoLibre versions are behind the published ones.

GeoLibre (https://github.com/opengeos/GeoLibre) cuts a release roughly every
four days, and every channel is published at the same tag, so one number tracks
the whole ecosystem. This script compares the versions pinned in
``tools/geolibre_versions.json`` against GitHub releases, PyPI, npm, and
conda-forge, and reports what moved.

Usage:
    python3 tools/geolibre_watch.py            # report drift, exit 1 if behind
    python3 tools/geolibre_watch.py --update   # re-pin to the published versions
    python3 tools/geolibre_watch.py --json     # machine-readable output

Exit codes:
    0  everything pinned matches what is published
    1  at least one pinned version is behind
    2  a source could not be reached (network or upstream error)

Only the Python standard library is used, so this runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "opengeos/GeoLibre"
PINS_FILE = Path(__file__).with_name("geolibre_versions.json")
TIMEOUT = 25
USER_AGENT = "geolibre-watch/1.0 (+https://github.com/opengeos/GeoLibre)"

# label -> (channel, URL, dotted path into the JSON response)
SOURCES = {
    "github_release": ("github", f"https://api.github.com/repos/{REPO}/releases/latest", "tag_name"),
    "pypi": ("pypi", "https://pypi.org/pypi/geolibre/json", "info.version"),
    "conda_forge": ("conda", "https://api.anaconda.org/package/conda-forge/geolibre", "latest_version"),
    "npm_core": ("npm", "https://registry.npmjs.org/@geolibre/core", "dist-tags.latest"),
    "npm_map": ("npm", "https://registry.npmjs.org/@geolibre/map", "dist-tags.latest"),
    "npm_embed": ("npm", "https://registry.npmjs.org/@geolibre/embed", "dist-tags.latest"),
}

RELEASES_FEED = f"https://github.com/{REPO}/releases.atom"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def dig(payload: dict, dotted: str):
    node = payload
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def version_key(value: str):
    """Sortable key for a version string such as ``3.4.0`` or ``v3.4.0``."""
    numbers = [int(part) for part in re.findall(r"\d+", value or "")]
    return tuple(numbers) if numbers else (0,)


def collect():
    """Return (published, errors) for every source."""
    published, errors = {}, {}
    for label, (_channel, url, dotted) in SOURCES.items():
        try:
            published[label] = dig(fetch_json(url), dotted)
            if not published[label]:
                errors[label] = f"no version field at {dotted!r}"
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors[label] = f"{type(exc).__name__}: {exc}"
    return published, errors


def recent_releases(limit=None):
    """Latest release tags from the releases atom feed, newest first."""
    try:
        request = urllib.request.Request(RELEASES_FEED, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            feed = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    titles = re.findall(r"<title>([^<]+)</title>", feed)
    updated = re.findall(r"<updated>([^<]+)</updated>", feed)
    entries = list(zip(titles[1:], updated[1:]))  # skip the feed title
    return (entries[:limit] if limit else entries), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="re-pin to the published versions")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--quiet", action="store_true", help="skip the release feed listing")
    parser.add_argument(
        "--installed",
        metavar="VERSION",
        help="record the GeoLibre Desktop build actually deployed (e.g. 2.8.0)",
    )
    args = parser.parse_args()

    pins, installed = {}, {}
    if PINS_FILE.exists():
        stored = json.loads(PINS_FILE.read_text(encoding="utf-8"))
        pins = stored.get("pinned", {})
        installed = stored.get("installed", {})
    if args.installed:
        installed["geolibre_desktop"] = args.installed.lstrip("v")

    published, errors = collect()
    if not published:
        print("Could not reach any GeoLibre source.", file=sys.stderr)
        for label, message in errors.items():
            print(f"  {label}: {message}", file=sys.stderr)
        return 2

    behind, results = [], {}
    for label, version in published.items():
        pinned = pins.get(label)
        status = "unpinned"
        if pinned:
            if version_key(pinned) < version_key(version):
                status = "behind"
                behind.append(label)
            elif version_key(pinned) > version_key(version):
                status = "ahead"
            else:
                status = "current"
        results[label] = {"pinned": pinned, "published": version, "status": status}

    releases, feed_error = recent_releases()
    latest_tag = published.get("github_release", "")
    deployed = installed.get("geolibre_desktop")
    skipped = []
    if deployed and latest_tag:
        skipped = [
            tag
            for tag, _when in releases
            if version_key(deployed) < version_key(tag) <= version_key(latest_tag)
        ]
        skipped.reverse()  # oldest first

    if args.update:
        PINS_FILE.write_text(
            json.dumps(
                {
                    "comment": "Pinned GeoLibre versions. Refresh with tools/geolibre_watch.py --update.",
                    "repo": f"https://github.com/{REPO}",
                    "pinned": {label: info["published"] for label, info in sorted(results.items())},
                    "installed": installed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {PINS_FILE}")
        return 0

    if args.json:
        print(
            json.dumps(
                {
                    "results": results,
                    "errors": errors,
                    "behind": behind,
                    "installed": installed,
                    "skipped_releases": skipped,
                },
                indent=2,
            )
        )
        return 1 if behind else 0

    width = max(len(label) for label in results)
    for label, info in sorted(results.items()):
        marker = {"behind": "!!", "ahead": " ?", "current": "  ", "unpinned": " -"}[info["status"]]
        pinned = info["pinned"] or "(unpinned)"
        print(f"{marker} {label:<{width}}  pinned {pinned:<10} published {info['published']}")

    if errors:
        print("\nUnreachable sources:")
        for label, message in sorted(errors.items()):
            print(f"    {label}: {message}")

    if deployed and latest_tag:
        state = "behind" if version_key(deployed) < version_key(latest_tag) else "current"
        print(f"\nDeployed GeoLibre Desktop {deployed} vs published {latest_tag}: {state}")
        if skipped:
            print(f"    {len(skipped)} release(s) skipped: {', '.join(skipped)}")

    if releases and not args.quiet:
        print("\nRecent releases:")
        for title, when in releases[:5]:
            print(f"    {title:<10} {when}")
    elif feed_error:
        print(f"\n(Release feed unavailable: {feed_error})")

    if behind:
        print(f"\n{len(behind)} channel(s) behind: {', '.join(behind)}")
        print("Read the release notes, then re-pin with: geolibre_watch.py --update")
        return 1
    print("\nAll pinned channels are current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
