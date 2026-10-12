# Keeping Arbotic Twin in step with GeoLibre

Snapshot taken **2026-10-12** (UTC), read from the upstream repo and the public
package registries. Re-run [`tools/geolibre_watch.py`](../tools/geolibre_watch.py)
to refresh the numbers.

- Upstream: <https://github.com/opengeos/GeoLibre> (MIT, maintained by
  Qiusheng Wu / `opengeos`)
- Docs: <https://geolibre.app> · Roadmap: <https://geolibre.app/roadmap/>
- Latest release: **v3.4.0** (2026-10-11)
- **Arbotic Twin is pinned to GeoLibre Desktop 2.8.x** — six releases, roughly
  six weeks, behind.

> **Finding: "2.8.1" is not an upstream release.** The tag list goes
> `v2.8.0` → `v2.9.0`, and the desktop artifacts are named
> `geolibre-desktop-2.8.0-x64-portable.zip`, `GeoLibre.Desktop_2.8.0_x64-setup.exe`,
> and so on — the app version equals the release tag. Check what the installed
> app actually reports (Help/About, or the Diagnostics panel) before treating
> 2.8.1 as the baseline; if it is really 2.8.0, then `minGeoLibreVersion` checks
> and any compatibility note written against "2.8.1" are pinned to a version
> that does not exist.


Snapshot taken **2026-10-12** (UTC). Everything below was read from the upstream
repo and the public package registries on that date — re-run
[`tools/geolibre_watch.py`](../tools/geolibre_watch.py) to refresh the numbers.

- Upstream: <https://github.com/opengeos/GeoLibre> (MIT, maintained by
  Qiusheng Wu / `opengeos`)
- Docs: <https://geolibre.app> · Roadmap: <https://geolibre.app/roadmap/>
- Latest release at snapshot time: **v3.4.0**, published 2026-10-11

---

## 1. Where GeoLibre is right now

| Signal | Value (2026-10-12) |
| --- | --- |
| Latest release | **v3.4.0** (2026-10-11) |
| Tagged releases | 35, from v0.1.0 (2026-05-28) to v3.4.0 |
| Release cadence | ~1 release every 4 days (11 releases between 2026-08-06 and 2026-10-11) |
| Commits | active daily (last commit 2026-10-12) |
| Contributors | 71 |
| Stars / forks / watchers | ~7.9k / ~880 / 50 |
| Open issues | 18 (triaged fast) |
| Discussions | ~200 threads |
| License / citation | MIT; Zenodo DOI per release (concept DOI `10.5281/zenodo.20785400`) |

It is a **TypeScript monorepo** (`apps/`, `packages/`, `python/`, `backend/`,
`workers/`, `skills/`) shipping one codebase to the web, desktop (Tauri v2),
Android, iOS, and Jupyter. Core stack: React + TypeScript, MapLibre GL JS,
DuckDB-WASM Spatial, deck.gl, CesiumJS, ArcGIS Maps SDK.

### Release train, in themes

| Version | Date | Theme |
| --- | --- | --- |
| v1.0 | 2026-06-10 | Processing pipelines, external plugin system |
| v1.3 | 2026-06-15 → 07 | Collaboration, story maps, SQL workspace, AI assistant |
| v2.0 | 2026-07-11 | 3D globe, planetary mapping, editable source layers |
| v2.1 | 2026-07-14 | Browser object detection, map recording, Browser panel |
| v2.5 | 2026-08-06 | QGIS/ArcGIS Pro import, DGGS, hyperspectral, self-hosted server |
| v2.6 | 2026-08-14 | **MCP server**, deep links, COG terrain |
| v2.8 | 2026-08-27 | Popup designer, quick filters, npm packages |
| v3.0 | 2026-09-14 | Cesium as first-class engine, ArcGIS write-back |
| v3.1 | 2026-09-25 | Mapbox + ArcGIS engines, voice AI assistant |
| v3.2 | 2026-10-01 | **Point cloud annotation**, private S3, OS keychain secrets |
| v3.3 | 2026-10-05 | Line of sight, MGRS, SQL Server, plugin marketplace UI |
| v3.4 | 2026-10-11 | **OBIA workbench (eCognition import)**, **ICESat-2 / GEDI**, turn-by-turn navigation |

Next up per the roadmap: plugin-marketplace phases 2–3 (version checks,
update/remove flows, third-party submission workflow).

---

## 2. What actually matters for a tree / canopy system

The features that overlap with the DeepForest-HK pipeline and an Arbotic digital
tree platform:

- **AI Segmentation (SamGeo / SAM 3)** — text prompts such as `"trees"` or
  automatic mask generation, results returned as GeoJSON layers with a
  per-feature `score`. Runs via a separate `samgeo-api` model server, and the
  SamGeo *plugin* works in the browser build too.
- **In-browser object detection** — ONNX/YOLO models executed in the webview,
  no server and no Python, over map imagery **or an imported geotagged photo
  layer** (added in v3.4). This is the natural home for a tree-crown detector
  exported to ONNX.
- **Object-Based Image Analysis (v3.4)** — segmentation → object metrics →
  multilevel hierarchy → training samples → random-forest / ruleset
  classification → accuracy assessment, with eCognition rule-set import. Closest
  upstream analogue to crown delineation + inventory attribution.
- **LiDAR** — LAS/LAZ/COPC/EPT point clouds, 65 browser-side LiDAR tools, point
  cloud annotation with ASPRS classes, polygon-to-point lifting (SAM masks onto
  the cloud), oriented 3D boxes, and export to labelled LAS/LAZ, NumPy, or
  Segments.ai cuboids. Directly usable for height filtering of crowns.
- **Spaceborne LiDAR (v3.4)** — ICESat-2 ATL06/ATL08 and GEDI L2A/L2B/L4A as
  footprint layers, ATL03 photons streamed for the view, along-track profiles.
- **Field + reporting** — mobile field collection with custom forms and GPS,

---

## 2b. The six releases you have skipped, plugin-relevant changes

Arbotic's only coupling to GeoLibre is `apps/map-plugin/` — the plugin *is* the
integration. These are the upstream changes between 2.9 and 3.4 that touch a
plugin host, newest first:

| Release | Date | What could affect the Arbotic plugin |
| --- | --- | --- |
| v3.4 | 2026-10-11 | Plugin panels run on the Cesium globe through a recording control facade; plugin panels + point clouds + Street View on the globe |
| v3.3 | 2026-10-05 | **Plugin ecosystem hardens**: registry installs verified against the published bundle hash, registry blocklist honored, `?plugin=` deep link behind a trust prompt, external plugins grouped under Plugins → Installed, plugins can add items to built-in toolbar menus, keep project state through credential stripping, use host proj4/WFS APIs, get a native fetch with a cookie jar on desktop, and read the active map tool |
| v3.2 | 2026-10-01 | `app.credentials` gives plugins the OS keychain (macOS Keychain / Windows Credential Manager / Linux Secret Service); MapLibre-control plugins run on the ArcGIS renderer |
| v3.1 | 2026-09-25 | Mapbox and ArcGIS join the rendering engines — a plugin that only declares `maplibre` is *suspended* when another engine is active |
| v3.0 | 2026-09-14 | Cesium becomes a primary engine; the app drives the map through a renderer-neutral `MapEngine`; **"plugins are gated on what the active engine can actually do rather than silently doing nothing"**; plugins declare supported engines and switching renderers suspends unsupported ones while keeping settings |
| v2.9 | 2026-09-03 | A capability model in `@geolibre/core` gates processing, sidecar tools, project editing, data loading, **plugins**, and settings independently; Web Services plugins move into a shared dockable panel host |

Two practical consequences for a 2.8-era plugin:

1. **The renderer gate is new.** A plugin whose manifest omits `engines` defaults
   to `["maplibre"]`. On 2.8 that was the only engine; on 3.x, switching to
   Cesium/Mapbox/ArcGIS greys the plugin out (hover for the reason) or suspends
   it. If the Arbotic popup/3D viewer should survive an engine switch, declare
   `"engines": ["maplibre"]` explicitly and decide deliberately whether the
   plugin is engine-neutral.
2. **Deployment policy can refuse to load your plugin.** `deployment.json`
   (v2.9+) governs plugin precedence, and `sideload: false` suppresses
   *project manifest URL* trust. A plugin dropped into the app data `plugins/`
   directory is still an external plugin, so policy — not just the folder —
   decides whether it loads. See `docs/deployment-policy.md#plugin-precedence`
   upstream if you ever ship a locked-down build.

## 2c. The plugin contract as of 3.4

What the loader requires, which is what your `plugin.json` and bundle must satisfy:

- **Install forms**: a `.zip` with a root `plugin.json`, an unpacked directory
  with a root `plugin.json`, or an HTTPS `plugin.json` manifest URL. Desktop
  loads external plugins from the **app data `plugins/` directory** at startup —
  `%APPDATA%\org.geolibre.desktop\plugins\` on Windows, which is where
  `deploy/start_arbotic.bat` copies `arbotic-twin\`.
- **Dev directories win.** Settings → Plugins can add local development
  directories; those are scanned *before* the app data `plugins/` directory, so
  a dev copy overrides an installed plugin with the same id. **Built-in plugins
  take precedence over all external plugins.**
- **The manifest and the bundle must agree.** The `entry` file must export a
  `GeoLibrePlugin` as the default export or a named `plugin` export, and its
  `id`, `name`, and `version` **must match `plugin.json`**.
- **The entry must be self-contained** — a single `.js`/`.mjs` bundle, because
  relative module imports inside a zip are not resolved by the loader.
- **Optional manifest fields**: `activeByDefault` (honored only for bundled
  drop-ins, not for user/project plugins), `engines`, `exclusiveGroup`,
  `urlParameterNames`.
- **Registry entries** (only relevant if you ever publish) carry `id`, `name`,
  `version`, `manifestUrl` (required) plus `minGeoLibreVersion`,
  `publishableSettings`, and so on. `minGeoLibreVersion` gates *installation*;
  `publishableSettings` lists the plugin project-state keys allowed to survive
  "Strip credentials" and shared/exported projects — by default an external
  plugin's whole state is dropped there and treated as credential-bearing.
- **CSP on desktop**: `connect-src` allows any `https:`/`http:` host, so your
  plugin can reach the Flask engine on `127.0.0.1:8000`. `script-src` only
  allows the app itself, `blob:` URLs, and version-pinned CDN paths — so the
  bundle must not inject `<script>` or `import()` from another host. Bundle the
  code (and three.js) into the plugin, or serve it from the engine as data.

Start from <https://github.com/opengeos/geolibre-plugin-template> — its
`npm run package:geolibre` produces the zip layout Desktop expects, which is a
cheaper upgrade path than hand-maintaining `plugin.js` + `plugin.json`.


## 3. How to keep up (in priority order)

### a. Subscribe to releases, not to `main`

- Watch the repo with **Releases only**: GitHub → Watch → Custom → Releases.
- Atom feed (no account needed):
  `https://github.com/opengeos/GeoLibre/releases.atom`
- CLI equivalent:

  ```bash
  gh release list -R opengeos/GeoLibre --limit 10
  gh release view -R opengeos/GeoLibre            # latest, with full notes
  gh api repos/opengeos/GeoLibre/releases/latest --jq '.tag_name'
  ```

- There is no `CHANGELOG.md`; the release notes **are** the changelog. The
  roadmap page records the same history grouped by theme, and marks the current
  version.

### b. Pin the versioned artifacts you depend on

All channels move in lockstep at the release tag, so one number tracks everything:

| Channel | Artifact | At snapshot |
| --- | --- | --- |
| npm | `@geolibre/core`, `@geolibre/map`, `@geolibre/embed` | 3.4.0 |
| PyPI | `geolibre` (`pip install geolibre`) | 3.4.0 |
| conda-forge | `geolibre` | 3.4.0 |
| Desktop/mobile | installers, Mac App Store, Microsoft Store, Google Play, App Store | 3.4.0 |
| Docker | browser build | 3.4.0 |

Pin exactly (`"@geolibre/core": "3.4.0"`, `geolibre==3.4.0`) and bump
deliberately — a caret range will drift weekly.

Arbotic's real dependency is the **Desktop build**, which is not a
package-manager artifact. `tools/geolibre_versions.json` therefore records both
the `pinned` set (what has been validated against) and the `installed` Desktop
build (what is deployed today). `tools/geolibre_watch.py` reports the gap and
names every release skipped.

### c. Keep the plugin on the sanctioned surface

The plugin API is the right layer for Arbotic — the Flask engine stays the
science, the plugin stays the client. Two rules keep that safe across upgrades:

- Call the optional host helpers with **optional chaining**
  (`app.addTileLayer?.(...)`, `app.getLayerFeatures?.(...)`): upstream types
  them optional precisely so a plugin keeps working against host variants that
  lack a helper.
- Declare what you support (`engines`, `urlParameterNames`,
  `exclusiveGroup`) rather than relying on defaults, so the host gates the
  plugin instead of failing at runtime.

Avoid: vendoring GeoLibre internals, importing the app's own modules from a
plugin bundle, or bundling a second copy of a library the host already
provides (deck.gl, `@carbonplan/zarr-layer`). Upstream's
`docs/maintenance.md` is a catalogue of how those seams break *silently*.

### d. Track your own drift with the included script

```bash
python3 tools/geolibre_watch.py           # compare pinned vs published
python3 tools/geolibre_watch.py --update  # re-pin after a deliberate bump
```

It reads `tools/geolibre_versions.json` (your pinned set), queries GitHub, PyPI,
npm, and conda-forge, and exits non-zero when something is behind — so it drops
straight into CI or a weekly cron job.

### e. Optional, if you want to contribute back

- `CONTRIBUTING.md` + `docs/contributing.md`, `docs/maintenance.md`,
  `CLAUDE.md` for the repo rules and quality gate.
- Discussions (≈200 threads) for design questions; issues are triaged quickly
  (18 open at snapshot).
- Note the v3.4 direction: spaceborne LiDAR, OBIA, and the plugin marketplace —
  a tree-crown/forestry plugin would land in a receptive area.

---

## 4. The five-minute check, for every new release

1. `python3 tools/geolibre_watch.py` — see what moved, and which releases you
   skipped.
2. Read the release notes for anything touching **project format**, the
   **plugin API**, or the **rendering-engine gate**. Everything else is additive.
3. `python3 tools/arbotic_plugin_doctor.py` — confirm the workspace plugin and
   the installed copy still agree.
4. Decide: adopt now, or stay pinned. Staying pinned is a legitimate answer —
   record *why* in `MEMORY.md` so the next session does not re-litigate it.
5. `--update` the watch file and commit it, so the next diff is meaningful.

The one-off migration from 2.8.x to 3.x is in section 5.

---

## 5. Upgrade runbook: Desktop 2.8.x to 3.4.x

Six releases of accumulated change, one of them (v3.0) a rendering-engine
re-architecture. Treat it as a migration, not an update.

**Before touching anything**

1. Keep the 2.8.0 installer (`geolibre-desktop-2.8.0-x64-portable.zip` or
   `GeoLibre.Desktop_2.8.0_x64-setup.exe`) as the rollback path.
2. Record the real installed version (Help/About, or Diagnostics) — see the
   *2.8.1 does not exist* finding at the top of this file.
3. Snapshot `var/`, `services/api/tra/tra.db`, and the current
   `%APPDATA%\org.geolibre.desktop\plugins\arbotic-twin\` copy.
4. Run `python3 tools/arbotic_plugin_doctor.py` and keep the output as the
   pre-upgrade baseline.

**Install and re-point**

5. Install 3.4.x, then let `deploy\start_arbotic.bat` re-copy the plugin. The
   launcher overwrites the installed copy on every run, which is the correct
   behaviour — the failure mode to watch is a *stale* installed copy that no
   longer matches the workspace.
6. Confirm the engine is reachable: `curl http://127.0.0.1:8000/api/health`.

**Verify, in this order**

7. Does the plugin appear at all? Plugins menu → checkmark next to the entry.
   A greyed-out entry means the active rendering engine is not in the plugin's
   `engines` list — hover for the reason. A plugin that fails to start is
   reported as a notification and in **Diagnostics** under `runtime`.
8. Does the popup still open, with risk badge, metrics, cylinders and the
   three.js twin?
9. Do the layers still render — `/api/layers/hk/<z>/<x>/<y>.mvt`, the PMTiles
   layer, the pilot layers, tree icons?
10. Re-run the repo's own gates: `verify\verify_live_http.py`,
    `verify\verify_pmtiles_layer.py`, `verify\verify_species_facets.py`, and
    `services\api\tra\cli.py --test`. Preserve the documented baselines
    (682 facet rows, filter total 12,275, about 15,705 Kowloon hits, 1,549 KP
    features, z13 tile 200 / ocean 404, Range 206).

**Then**

11. `python3 tools/geolibre_watch.py --update` and commit the pins, so the next
    diff starts from the new baseline.
12. Update the "GeoLibre Desktop 2.8.1" line in the intro, the environment
    reference table, and `MEMORY.md`.

## 6. Where Arbotic should plug in next

The plugin is the only GeoLibre-aware component, so upstream features are only
usable to the extent the plugin surfaces them. Highest value, lowest risk first:

- **C4's `POST /api/qsm/process`** — keep the enqueue-and-poll shape the
  architecture already promises; the plugin renders progress from the job id.
  The same job surface is what a future MCP/agent path would drive.
- **Adopt the plugin template and zip packaging**
  (<https://github.com/opengeos/geolibre-plugin-template>).
  `npm run package:geolibre` emits the layout Desktop expects and gives you
  `engines` / `minGeoLibreVersion` metadata for free, instead of a hand-built
  `plugin.js` + `plugin.json` pair that the loader must be kept in sync with.
- **Declare `engines` and `minGeoLibreVersion` explicitly.** The engine list
  stops a renderer switch from silently disabling the plugin; the minimum
  version stops an older Desktop from loading a plugin it cannot run.
- **Features worth exposing through the plugin** as the platform grows: the
  LiDAR point-cloud annotation workbench (v3.2) for scan QA, AI Segmentation /
  SamGeo for crown candidates, the Print Layout composer for TRA Form 2 output,
  and mobile field collection for inspection notes.

## 7. Licensing

- GeoLibre Desktop is **MIT**. A proprietary plugin that talks to it over the
  plugin API is ordinary use, and MIT imposes no copyleft on the plugin. If you
  *redistribute* GeoLibre (for example bundled in an Arbotic installer), keep
  the MIT licence text and copyright notice, and do not imply endorsement.
- The GPL components Arbotic already lists — PyTLidar (GPL-3.0, bundled in
  `engine/`) and R/tReeTraits (GPL-2/GPL-3) — are a separate question from
  GeoLibre and are unaffected by a GeoLibre upgrade. Distribution obligations
  there are worth a lawyer's read before any release.
- Publishing the plugin to the GeoLibre marketplace (`plugins.geolibre.app`)
  would make it downloadable by anyone, which sits badly with the "proprietary,
  all rights reserved" stance. Keep it private: install from a local directory
  or a self-hosted manifest URL instead.

## 8. Tooling in this repo

| Command | Purpose |
| --- | --- |
| `python3 tools/geolibre_watch.py` | Published vs pinned versions across GitHub, PyPI, npm, conda-forge; lists the releases you have skipped |
| `python3 tools/geolibre_watch.py --update` | Re-pin after a deliberate bump |
| `python3 tools/arbotic_plugin_doctor.py` | Validate `apps/map-plugin/plugin.json` and its entry bundle, and diff the workspace copy against the installed GeoLibre copy |
