#!/usr/bin/env python3
"""Check the Arbotic Twin GeoLibre plugin: manifest, bundle, and installed copy.

GeoLibre Desktop loads external plugins from its app-data ``plugins/`` directory
at startup, while ``deploy/start_arbotic.bat`` copies the plugin from
``apps/map-plugin/`` into that directory. Two copies therefore exist, and when
they disagree the plugin silently fails to load: the loader requires the entry
bundle's exported ``id``, ``name`` and ``version`` to match ``plugin.json``, and
it requires a single self-contained bundle (relative imports inside the plugin
directory are not resolved).

This script checks both copies and reports drift, so a stale installed copy is
caught before it costs an afternoon.

Usage:
    python3 tools/arbotic_plugin_doctor.py
    python3 tools/arbotic_plugin_doctor.py --installed "C:/Users/me/AppData/.../arbotic-twin"
    python3 tools/arbotic_plugin_doctor.py --json

Exit codes:
    0  plugin is consistent and the installed copy matches the workspace
    1  a manifest/bundle problem, or the installed copy differs
    2  nothing to check (workspace plugin or manifest not found)

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DEFAULT = REPO_ROOT / "apps" / "map-plugin"
PLUGIN_ID = "arbotic-twin"
REQUIRED_KEYS = ("id", "name", "version")
SELF_CONTAINED_BAD = (
    re.compile(r"""from\s+["']\.{1,2}/"""),
    re.compile(r"""import\s*\(\s*["']\.{1,2}/"""),
    re.compile(r"""require\s*\(\s*["']\.{1,2}/"""),
)


def default_installed_dir(plugin_id: str = PLUGIN_ID) -> Path | None:
    """The GeoLibre Desktop app-data plugins directory for this platform."""
    if os.name == "nt" or sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if not base:
            return None
        return Path(base) / "org.geolibre.desktop" / "plugins" / plugin_id
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library/Application Support/org.geolibre.desktop/plugins"
            / plugin_id
        )
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "org.geolibre.desktop" / "plugins" / plugin_id


def load_manifest(plugin_dir: Path):
    """Return (manifest, error)."""
    path = plugin_dir / "plugin.json"
    if not path.exists():
        return None, f"no plugin.json in {plugin_dir}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"plugin.json is unreadable: {type(exc).__name__}: {exc}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_files(plugin_dir: Path):
    """Every regular file under the plugin directory, as relative POSIX paths."""
    if not plugin_dir.is_dir():
        return set()
    return {
        item.relative_to(plugin_dir).as_posix()
        for item in plugin_dir.rglob("*")
        if item.is_file()
    }


def check_bundle(plugin_dir: Path, manifest: dict):
    """Validate the entry bundle against the manifest. Returns a problem list."""
    problems = []
    entry_name = manifest.get("entry") or "plugin.js"
    entry = plugin_dir / entry_name
    if not entry.exists():
        problems.append(f"entry file {entry_name!r} declared in plugin.json is missing")
        return problems, None

    try:
        source = entry.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        problems.append(f"entry {entry_name} is unreadable: {exc}")
        return problems, None

    for pattern in SELF_CONTAINED_BAD:
        match = pattern.search(source)
        if match:
            problems.append(
                f"entry {entry_name} has a relative import ({match.group(0)!r}); "
                "the loader does not resolve relative modules inside a plugin directory"
            )
            break

    if len(source) < 200:
        problems.append(f"entry {entry_name} is only {len(source)} bytes — is it built?")

    for key in REQUIRED_KEYS:
        value = manifest.get(key)
        if not value:
            problems.append(f"plugin.json is missing {key!r}")
            continue
        if not re.search(rf"""["']?{key}["']?\s*[:=]\s*["']{re.escape(str(value))}["']""", source):
            problems.append(
                f"entry does not declare {key}: {value!r} — the exported plugin "
                f"{key} must match plugin.json"
            )
    return problems, entry


def compare(workspace: Path, installed: Path):
    """Return a list of drift lines between the two copies."""
    drift = []
    ws_manifest, ws_err = load_manifest(workspace)
    in_manifest, in_err = load_manifest(installed)

    if ws_err or in_err:
        return [f"workspace: {ws_err}" if ws_err else f"installed: {in_err}"]

    for key in ("id", "name", "version"):
        if ws_manifest.get(key) != in_manifest.get(key):
            drift.append(
                f"{key}: workspace {ws_manifest.get(key)!r} != installed {in_manifest.get(key)!r}"
            )

    ws_files = bundle_files(workspace)
    in_files = bundle_files(installed)
    only_ws = sorted(ws_files - in_files)
    only_in = sorted(in_files - ws_files)
    if only_ws:
        drift.append(f"only in workspace: {', '.join(only_ws)}")
    if only_in:
        drift.append(f"only in installed: {', '.join(only_in)}")

    for name in sorted(ws_files & in_files):
        ws_hash = sha256(workspace / name)
        in_hash = sha256(installed / name)
        if ws_hash != in_hash:
            drift.append(f"{name}: content differs (workspace {ws_hash[:8]}, installed {in_hash[:8]})")
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, default=WORKSPACE_DEFAULT,
                        help=f"workspace plugin directory (default: {WORKSPACE_DEFAULT})")
    parser.add_argument("--installed", type=Path, default=None,
                        help="installed plugin directory (default: the GeoLibre app-data path)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"Workspace plugin directory not found: {workspace}", file=sys.stderr)
        print("Pass --workspace <dir> if the plugin lives elsewhere.", file=sys.stderr)
        return 2

    manifest, manifest_err = load_manifest(workspace)
    if manifest_err:
        print(f"Cannot read the workspace plugin: {manifest_err}", file=sys.stderr)
        return 2

    problems, entry = check_bundle(workspace, manifest)

    installed = args.installed or default_installed_dir(str(manifest.get("id") or PLUGIN_ID))
    installed_present = bool(installed and Path(installed).is_dir())
    drift = compare(workspace, Path(installed)) if installed_present else []

    if args.json:
        print(json.dumps({
            "workspace": str(workspace),
            "manifest": manifest,
            "problems": problems,
            "installed": str(installed) if installed else None,
            "installed_present": installed_present,
            "drift": drift,
        }, indent=2))
        return 1 if (problems or drift) else 0

    print(f"Workspace plugin : {workspace}")
    print(f"  id {manifest.get('id')!r}  name {manifest.get('name')!r}  "
          f"version {manifest.get('version')!r}")
    if entry:
        print(f"  entry {entry.name} ({entry.stat().st_size:,} bytes)")

    if problems:
        print("\nManifest / bundle problems:")
        for problem in problems:
            print(f"  !! {problem}")
    else:
        print("  manifest and bundle: OK")

    print(f"\nInstalled plugin : {installed}")
    if not installed_present:
        print("  not found — GeoLibre has no copy to load, so run deploy\\start_arbotic.bat")
    elif drift:
        print("  DRIFT — the copy GeoLibre actually loads differs from the workspace:")
        for line in drift:
            print(f"    !! {line}")
        print("  Re-run the launcher to re-sync, then re-check.")
    else:
        print("  in sync with the workspace")

    return 1 if (problems or drift) else 0


if __name__ == "__main__":
    sys.exit(main())

