# GeoLibre features Arbotic Twin can actually use

Beyond version tracking: which upstream capabilities map onto this platform, and
what each one would replace or add. Verified against the upstream docs on
**2026-10-12** (GeoLibre v3.4.0).

Read it as a shopping list. Tier 1 features replace something Arbotic has
already hand-built; Tier 2 is capability Arbotic does not have at all.

> **Constraint to keep in mind.** The architecture promises engines that run
> **offline, no network**. GeoLibre honours that for its WASM toolboxes, LiDAR,
> 3D Tiles, styling, SQL (after first load), and field collection, but a few
> features need a remote host: GDAL export, ONNX/YOLO object detection, "Segment
> Everything", story-map HTML export, and Pyodide without a configured mirror.
> Its `GEOLIBRE_NO_EXTERNAL_CDN=1` build is the honest mode for an air-gapped
> deployment — those features then report that they need a host instead of
> failing mid-run. Anything marked **needs network** below is on that list.

---

## Tier 1 — replaces something already built

| Feature | Replaces | Upstream |
| --- | --- | --- |
| Native LiDAR layers (LAS/LAZ/COPC/EPT) + Point Cloud Annotation | The `three.js` iframe at `/viewer/<tree_id>` | v3.2 |
| Native 3D Tiles | Your own `/api/3dtiles/...` client work | v1.10+ |
| Print Layout + Atlas | Server-rendered bilingual TRA HTML reports | v2.2, v3.0 |
| GPS Tracking + Field Collection | Manual `/api/field-trees` entry | v1.4, v3.4 |
| Attribute table + SQL Workspace | Ad-hoc pandas over `hk_trees.gpkg` | v0.4+ |

### 1. Native LiDAR layers and point-cloud annotation

GeoLibre reads **LAS, LAZ, COPC and EPT** point clouds natively, streams COPC by
level of detail, and the Point Cloud Annotation plugin (v3.2) lets an operator
label points with ASPRS or custom classes, group points into instances, lift
polygon layers (SAM masks, building footprints) onto the cloud, pre-label with
Whitebox LiDAR classifiers in the browser, and export a labelled **LAS/LAZ
rewrite**, NumPy arrays, GeoJSON, or Segments.ai cuboids.

Why it matters here: your QSM runner already produces a per-tree cloud, and the
3.8M-point case is a QSM workload, not a map workload. Annotation turns scan QA
into a visual, auditable step — mark noise, reclassify a stem, flag a defect
before the metrics are trusted — and it is saved with the project.

**Caveat:** the LiDAR reader does **not** take PLY, which is what
`/api/trees/<id>/cloud` serves today. Serve LAZ or COPC alongside it (the source
LAS is already there) and the layer drops straight in.

### 2. Native 3D Tiles

`add_3d_tiles` takes a `tileset.json` URL (or a Cesium Ion asset id, globe only),
with optional request headers and an altitude offset. Your `/api/3dtiles/...`
route is already the right shape — the difference is that GeoLibre draws it in
the main map, in the same project and layer stack as the inventory, instead of a
separate viewer. On the Cesium engine it is a real globe scene.

### 3. Print Layout and Atlas

A layout composer with scale bar, north arrow, legend and data blocks, exporting
**PNG, PDF or SVG**; tick "Generate a multi-page map series" and it becomes an
atlas — **one page per feature**, with `{atlas.attr:FIELD}`, `{atlas.pagenumber}`
and `{atlas.total}` tokens, exported as a multi-page PDF or a ZIP of PNGs.

That is a per-tree TRA sheet generator: drive the atlas from the tree layer,
tokens pulling tree id, species, DBH and risk band onto each sheet. SVG is the
format to choose when the output goes to print or into a document. It complements
rather than replaces the HTML report — the HTML stays the interactive record,
the atlas becomes the filing copy.

### 4. GPS Tracking and Field Collection

**Controls → GPS Tracking** streams a live position (including NMEA receivers
with RTK fixed/float status) and records a track; **Field Collection** captures
point/line/polygon observations through a form you define, with optional photos.
Both write **ordinary GeoJSON layers** saved with the project, exportable to
GeoJSON/GPX — there is no server sync step, which suits your
`var/user_points.json` model: collect in the field, export, ingest through the
existing API. There are native Android and iOS builds of the same app.

### 5. Attribute table and SQL Workspace

The attribute table gives filtering, a field calculator, virtual fields,
persistent joins, charts and statistics. The SQL Workspace runs **DuckDB Spatial**
(default, offline after first use), **PostGIS** (PGlite) or **Apache Sedona**
in-app, with `ST_*` available in all three, and can add a result as a layer, keep
a live re-runnable query layer, and export CSV or GeoParquet.

For Arbotic: join TRA outputs onto the 543,536-tree layer, or slice canopy by
district without writing a Flask route or a pandas script.

---

## Tier 2 — capability Arbotic does not have

| Feature | What it buys | Upstream |
| --- | --- | --- |
| Rule-based styling, expression labels, clustering, heatmaps | Risk badge as a real cartographic symbol on all 543,536 trees | v1.1, v2.2 |
| Symbology interchange (SLD / QML / Mapbox GL) | Move styles between QGIS and GeoLibre | v2.0 |
| H3 and DGGS grids | Aggregate risk per hexagon cell; tree-equity and heat analysis | v2.3, v2.5 |
| Historical Imagery, Timelapse, Layer Swipe | 2024 vs 2025 canopy comparison, side by side | v2.x, v3.3 |
| AI Segmentation (SamGeo / SAM 3) | Crown delineation from a text prompt ("trees") | v1.3, v2.8 |
| In-browser ONNX/YOLO detection | Crown or defect detection with no Python — **needs network** | v2.1, v3.4 |
| Object-Based Image Analysis workbench | Segment, classify with random forest or rulesets, assess accuracy, import eCognition rule sets | v3.4 |
| Elevation Profile, line of sight, MGRS | Fall-zone and sightline checks; grid coordinates in the field | v2.x, v3.3 |
| Story maps, standalone HTML export, map recording, camera tours | Stakeholder and public-facing deliverables | v1.3, v1.8 |
| Docked Jupyter Notebook panel | Run `qsm_runner.py` and the pipelines beside the live map | v1.4 |
| Real-time collaboration, share links | Multi-operator review (relevant once more than one person works a survey) | v1.3 |

Two of these deserve a note:

**Styling is the cheapest big win.** You already compute a risk band, a species
and a DBH per tree. A rule-based renderer with scale-dependent visibility and
data-defined labels turns that into a map where the *city* reads at z11 and the
*individual tree* reads at z17 — the current popup-only presentation hides the
pattern. `importLayerStyle` accepts OGC SLD, QGIS QML or Mapbox GL JSON, so an
existing QGIS style can be reused instead of rewritten.

**Historical Imagery and Layer Swipe speak directly to the 2024/2025 demo
images** in this repo's README: a swipe bar over two canopy layers, or the Esri
Wayback basemap, is a one-session comparison rather than a scripted diff.

## Tier 3 — for the web deployment plan

- **Deep links** — `?lat=&lon=&zoom=`, `?data=<url>` (GeoJSON, GeoParquet,
  PMTiles, COG, LAS/LAZ/COPC/EPT, STAC), `?style=`, `?tool=<tool-id>`,
  `?plugin=<id>`, and a `maponly` parameter for embeds. A TRA report can link
  straight to its tree on the map.
- **Embed API** — `@geolibre/embed` plus the versioned postMessage protocol
  (`ready`, `projectLoaded`, `selectionChanged`, `viewChanged`, `toolCompleted`)
  to host a GeoLibre map inside another page.
- **Self-hostable server and Docker** — the browser build, the sidecar, and
  `deployment.json` policy (including plugin precedence and `sideload`), which is
  the piece that decides whether an external plugin like `arbotic-twin` loads at
  all in a locked-down deployment.
- **Chrome extension** — opens data from a page into GeoLibre, with a
  configurable GeoLibre base URL (so it can point at your own deployment).

## Tier 4 — pipeline side (the `geolibre` Python package)

The pipelines are Python, and the package is a leafmap-style API over the same
project builders the app uses, so a script can author a real `.geolibre` project:

```python
from geolibre import Map

m = Map(center=(114.17, 22.32), zoom=13)

# The city inventory, straight off your own Flask routes
m.add_pmtiles("http://127.0.0.1:8000/api/layers/hk_trees.pmtiles", name="HK trees")

# A single scanned tree's cloud (serve LAZ/COPC, not PLY)
m.add_lidar("http://127.0.0.1:8000/api/trees/tree_scaled_2/cloud.laz", name="QSM cloud")

# Your existing 3D Tiles route, drawn natively
m.add_3d_tiles("http://127.0.0.1:8000/api/3dtiles/kp/tileset.json", name="Kowloon Park")

# Custom classes for the annotator (codes 19-255)
m.set_point_cloud_classes([
    {"code": 64, "name": "Stem", "color": "#8b5a2b"},
    {"code": 65, "name": "Defect", "color": "#e11d48"},
])
m
```

The same builders back the **MCP server** (`geolibre-mcp`), which can author
projects headlessly or drive a map already open in GeoLibre Desktop through its
`live_*` tools — the automation path for a C4 job queue. `geolibre.authoring` is
the widget-free module underneath both, so a pipeline can build a project dict
with no notebook involved.

## Wiring summary

| Arbotic asset (exists today) | GeoLibre surface to use |
| --- | --- |
| `/api/layers/hk_trees.pmtiles` (20.7 MB, z11-15) | `add_pmtiles`, or the PMTiles layer type |
| `/api/layers/hk/<z>/<x>/<y>.mvt` | Vector tile layer (plugin `addTileLayer`) |
| `/api/trees/<id>/cloud` (PLY today) | Add LAZ/COPC output, then a native LiDAR layer |
| `/api/3dtiles/...` | 3D Tiles layer, `add_3d_tiles` |
| `/api/trees/<id>/{metrics,cylinders}` | Popup/tooltip designer, attribute table, expression labels |
| `/tra-report/<tree_id>` (HTML) | Print Layout atlas for the filing copy |
| `/api/field-trees` | Field Collection + GPS Tracking, exported as GeoJSON |
| `data/hk_trees.gpkg` (read-only) | SQL Workspace (DuckDB Spatial), attribute joins |
| `var/results/<tree_id>/` | Project layers plus plugin state |
| `pipelines/`, `engine/qsm_runner.py` | Docked Notebook panel, `geolibre` package, MCP `live_*` |

## Honest limits

- **PLY is not a format GeoLibre's LiDAR reader takes.** Adding a LAZ or COPC
  export to `/api/trees/<id>/cloud` is a prerequisite for the point-cloud items.
- **Offline vs CDN.** ONNX object detection, SamGeo, GDAL export and story-map
  HTML export need a host; DuckDB-WASM needs one download on first use. Match the
  build to the deployment (`GEOLIBRE_NO_EXTERNAL_CDN=1` for air-gapped sites) and
  expect those features to say so rather than fail mid-run.
- **SamGeo needs a GPU `samgeo-api` model server.** In-browser segmentation is
  the fallback, and it has the network dependency above.
- **GeoLibre is not a backend.** It has no notion of the TRA workflow, the DEVB
  Form 2 matrices, or the system of record in `tra.db`. Everything above is
  presentation, capture and analysis *around* the engine — the science stays in
  Flask and the pipelines, which is exactly the coupling the architecture
  diagram already promises.
- **Version gate.** Every item here is from 3.x. On the pinned 2.8.x build,
  expect the LiDAR annotation workbench, OBIA, GCP/ICESat-2/GEDI, the atlas
  improvements and the engine gate to be missing entirely.

---

<sub>繁中摘要：除版本追蹤外，Arbotic Twin 可直接使用 GeoLibre 的原生 LiDAR 與 3D Tiles 圖層、點雲標註、Print Layout Atlas（每棵樹一頁）、GPS 與野外採集、SQL Workspace（DuckDB Spatial）、規則式風險分級與雙語標籤、H3 網格、歷史影像對比、OBIA 與 SamGeo 分割，以及 Python 套件與 MCP 自動化。注意：LiDAR 圖層不支援 PLY（需改出 LAZ/COPC），ONNX 偵測、SamGeo、GDAL 匯出等則需要網絡；而以上多數功能要 3.x 才有。</sub>

