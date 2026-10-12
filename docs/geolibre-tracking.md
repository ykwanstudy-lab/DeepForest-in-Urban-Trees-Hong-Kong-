# Keeping the Arbotic / DeepForest stack in step with GeoLibre

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

### c. Prefer the stable integration surfaces over forking internals

Ranked by how much upstream churn they insulate you from:

1. **Project format** — `.geolibre` (legacy `.geolibre.json` still opens). The
   app writes format `version: "0.2.0"`; the Python package writes `"0.1.0"`.
   Reading older files is supported; forward compatibility is not guaranteed, so
   validate on load. Helpers live in `@geolibre/core`:
   `createEmptyProject`, `parseProject`, `redactCredentials`, `serializeProject`
   (see `docs/project-format.md` upstream).
2. **Plugin API** — a `plugin.json` manifest plus a `GeoLibrePlugin` interface
   (`id`, `name`, `version`, `activeByDefault`, renderer support, `app.*`
   helpers). This is the sanctioned way to add a panel, a map control, a
   toolbar item, or a custom layer. Bundled drop-ins (`public/plugins/<id>/`)
   and external HTTPS manifests both work, and the marketplace registry lives at
   `plugins.geolibre.app` (source: `opengeos/geolibre-plugins`) with
   `minGeoLibreVersion` compatibility checks.
3. **MCP server / agent skill** — `pip install "geolibre[mcp]"` then
   `geolibre-mcp --root ~/maps`; writes real projects headlessly and can drive a
   running desktop map through `live_*` tools. `skills/geolibre/` teaches an
   external agent how to use it.
4. **Embed API** — `@geolibre/embed` + the versioned postMessage protocol for
   hosting a GeoLibre map inside another app.

Vendoring or patching GeoLibre internals is the option that costs the most to
maintain: upstream documents its own hand-written mirrors of unexported
upstream constants in `docs/maintenance.md`, and those drift **silently**.

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

## 4. Upgrade checklist when a new version lands

1. `python3 tools/geolibre_watch.py` — see what moved.
2. Read the release notes for **breaking changes** and for anything in
   *project format*, *plugin API*, or *embed protocol*.
3. If you ship a `.geolibre` file to users: open an existing file in the new
   build before upgrading production, and re-save to migrate the format version.
4. Re-run your own pipeline against the pinned artifact, then bump pins.
5. `--update` the watch file and commit it, so the next diff is meaningful.

---

## 5. Note on the current Arbotic repo

`Arbotic-Twin_Prototypes-` (React 19 + Vite + three/R3F + `@mlc-ai/web-llm`)
does **not** currently depend on any GeoLibre package — there is no
`@geolibre/*` entry in its `package.json`. So nothing needs upgrading today;
the question is which upstream surface to adopt when the two are wired
together. For that repo the cheapest path is the **embed API** or a
**`.geolibre` project export**, rather than importing `@geolibre/core` and
re-implementing the map. `DeepForest-in-Urban-Trees-Hong-Kong-` is a separate
Python pipeline pinned to the `deepforest` 1.4.x API
(`model.predict_tile(raster_path=...)`) and is not affected by GeoLibre
releases at all.

  offline areas, print layout with atlas, story maps.
- **Automation surfaces** — Python package (`geolibre`), R package, MCP server
  (`geolibre-mcp`), agent skill, plugin API, embed API, self-hostable server.
