# Google Street View (GSV) API — usage, cost, and whether it can be used for 3D modelling

*Notes compiled 2026-10-27 against the live Google Maps Platform documentation, price list and
Terms of Service. Prices are USD from the **global** price list; Hong Kong bills from the global
list (India has a separate list). Always re-check the
[price list](https://developers.google.com/maps/billing-and-pricing/pricing) and the
[Terms](https://cloud.google.com/maps-platform/terms) before committing budget or writing code.*

---

## TL;DR

1. **How to use it** — one HTTPS GET returns a perspective JPEG crop of a panorama
   (`/maps/api/streetview`), and a second free endpoint returns panorama metadata
   (`/maps/api/streetview/metadata`: `pano_id`, date, copyright, status). The hard limit is
   **640 x 640 px max** and `fov <= 120`.
2. **Cost** — SKU `Static Street View` = **$7.00 / 1,000 panoramas** (falling to $5.60, $4.20, $2.10,
   $0.53 at higher volume) with **10,000 free calls per SKU per month**. The metadata SKU is
   **free and unlimited**.
3. **Can I do 3D modelling from GSV?** — **Legally no, and technically not to survey grade.**
   Google's ToS §3.2.3(c) forbids creating content based on Google Maps Content, and it names our
   exact use case: *"(v) construct an index of tree locations within a city from Street View
   imagery"*. §3.2.3(c)(vii) also forbids using the content to train/validate ML models, and
   §3.2.3(a)(ii) forbids bulk downloading Street View images. See
   [What Google's terms actually forbid](#3-what-googles-terms-actually-forbid).
4. **For Hong Kong, better legal 3D sources already exist** — the Lands Department **Mobile Mapping
   System** (street-level LAS + 7,000 x 3,500 panoramas, 4–6 cm point density) and the
   **3D Visualisation Map** (whole territory, includes a *vegetation* class), plus free LiDAR for
   canopy height. See [Alternatives](#6-what-to-use-instead-for-hk-street-level--tree-3d).

---

## 1. How the API works

Two endpoints, both simple GETs, both requiring an API key on a billing-enabled Google Cloud
project.

### 1.1 Image request

```bash
curl -o pano.jpg \
  "https://maps.googleapis.com/maps/api/streetview?size=640x640\
&location=22.3193,114.1694&heading=90&pitch=0&fov=90&key=$GOOGLE_MAPS_API_KEY"
```

| Parameter | Default | Notes |
|---|---|---|
| `location` | — | `lat,lng` (or an address). Mutually exclusive with `pano` |
| `pano` | — | Panorama ID, obtained from the metadata endpoint |
| `size` | `600x300` | **Max 640 x 640.** This is the binding constraint for any reconstruction work |
| `heading` | auto | 0–360, `0`/`360` = North, `90` = East |
| `pitch` | `0` | −90 (straight down) … +90 (straight up) |
| `fov` | `90` | **Max 120** (horizontal degrees). Smaller `fov` = more zoom = fewer pixels/metre at the edges |
| `radius` | `50` | Search radius in metres when snapping `location` to the closest panorama |
| `source` | `default` | `google` = official only; `outdoor` excludes indoor collections |
| `return_error_code` | `false` | `true` returns HTTP 404 instead of the grey "no imagery" placeholder |
| `signature` | — | Digital signature (recommended; required above 25,000 requests/day) |

If there is no imagery you get a grey placeholder image, so always check metadata first (free) or
set `return_error_code=true`.

### 1.2 Metadata request (free, unlimited)

```bash
curl "https://maps.googleapis.com/maps/api/streetview/metadata\
?location=22.3193,114.1694&key=$GOOGLE_MAPS_API_KEY"
```

```json
{
  "status": "OK",
  "copyright": "© 2024 Google",
  "date": "2024-03",
  "location": { "lat": 22.319288, "lng": 114.169434 },
  "pano_id": "CAoSLEFGMVFpcE1..."
}
```

`status` is the cheap coverage check: `OK`, `ZERO_RESULTS`, `NOT_FOUND`, `OVER_QUERY_LIMIT`,
`REQUEST_DENIED`, `INVALID_REQUEST`, `UNKNOWN_ERROR`.

### 1.3 Minimal Python client (stdlib only)

```python
import json, os, urllib.parse, urllib.request

KEY = os.environ["GOOGLE_MAPS_API_KEY"]
BASE = "https://maps.googleapis.com/maps/api/streetview"

def metadata(lat, lng, radius=50):
    q = urllib.parse.urlencode({"location": f"{lat},{lng}", "radius": radius, "key": KEY})
    with urllib.request.urlopen(f"{BASE}/metadata?{q}", timeout=30) as r:
        return json.load(r)

def image(lat, lng, path, heading=0, pitch=0, fov=90, size="640x640"):
    q = urllib.parse.urlencode({
        "location": f"{lat},{lng}", "size": size, "heading": heading,
        "pitch": pitch, "fov": fov, "return_error_code": "true", "key": KEY,
    })
    with urllib.request.urlopen(f"{BASE}?{q}", timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())
```

The reusable, offline cost/coverage planner that ships with this repo is
[`gsv_survey_planner.py`](../gsv_survey_planner.py).

### 1.4 Other Street View surfaces (and what they cost)

| Product | What it gives you | Free / month | Price per 1,000 |
|---|---|---|---|
| Street View Static API (`Static Street View`) | one 640x640 max JPEG per call | 10,000 | $7.00 → $5.60 → $4.20 → $2.10 → $0.53 |
| Street View Static API (`Street View Metadata`) | `pano_id`, date, copyright, status | unlimited | free |
| Maps JavaScript `Dynamic Street View` | interactive viewer | 5,000 | $14.00 |
| Map Tiles API `Street View Tiles` | raw 512 px equirectangular tiles | 100,000 | $2.00 |
| Map Tiles API `Photorealistic 3D Tiles` | Google's own pre-built 3D mesh | 1,000 | $6.00 |
| Aerial View API | fly-around video of a POI | 5,000 | $16.00 |

`Street View Tiles` is the "high resolution" route: a full equirectangular panorama is assembled
from many 512 px tiles, which is exactly the bulk-download pattern prohibited by §3.2.3(a)(ii).
`Photorealistic 3D Tiles` is the *legitimate* way to get Google 3D geometry, but it is
render-only: you may not export, redistribute or derive datasets from the mesh.

---

## 2. What it costs in practice

`Static Street View` is billed **per panorama request**, not per pixel and not per axis. Asking for
4 headings at one location = **4 billed requests**.

Worked examples (global list, one calendar month, one viewing direction unless stated):

| Survey | Requests | Billable after 10k free | Cost |
|---|---|---|---|
| 40 km of road, one frame every 10 m | 4,001 | 0 | **$0.00** |
| 200 km, every 10 m | 20,001 | 10,001 | **$70.01** |
| 200 km, every 10 m, 4 headings (360° coverage) | 80,004 | 70,004 | **$490.03** |
| Territory-wide HK road network (~2,100 km), every 10 m | 210,001 | 200,001 | **$1,260.01** |
| Same, 4 headings | 840,004 | 830,004 | **$4,326.02** |

Two things fall out of that table:

* A single-heading, low-density sweep of a district is essentially free.
* Anything that looks like a 3D capture (multi-heading, high density, repeat visits) costs
  thousands of dollars per month — and is still capped at 640 px per frame.

Also note that the volume bands are **per calendar month**, so spreading one large survey over
several months usually costs *more*, not less: the same 840,004 requests spread over 6 months
(140,001/month) bill at $5,208.06 instead of $4,326.02. Free 10k calls do not compensate for
losing the volume discount. `gsv_survey_planner.py --months N` shows this directly.

The free allowance is **per SKU per month**, not a $200 credit (that model ended on
1 March 2025). Metadata calls are free and unlimited, so always probe coverage with the metadata
endpoint before spending image requests.

### 2.1 Billing setup in five minutes

1. Google Cloud Console → create/select a project → **enable billing**.
2. **APIs & Services → Library → "Street View Static API" → Enable**
   (enable `Map Tiles API` too if you want photorealistic 3D).
3. **Credentials → Create credentials → API key**. Restrict it to the Street View Static API and
   to your IPs/referrers — an unrestricted key in a public repo is how people wake up to a
   four-figure bill.
4. Set **quotas** (Cloud Console → Google Maps Platform → Quotas) to a hard ceiling, e.g.
   20,000/day, so a runaway loop stops instead of billing.
5. Set a **budget alert** at $10 / $50 / $100.

> Never commit the key. Use `GOOGLE_MAPS_API_KEY` (as the snippets above do) and keep it out of
> notebooks, CSVs and screenshots.

---

## 3. What Google's terms actually forbid

The governing document is the
[Google Maps Platform Terms of Service](https://cloud.google.com/maps-platform/terms), §3.2.3
*Restrictions Against Misusing the Services*. Street View imagery is "Google Maps Content", so
every clause below applies to it. Quoted verbatim (emphasis added):

> **§3.2.3(a) No Scraping.** Customer will not export, extract, or otherwise scrape Google Maps
> Content for use outside the Services. For example, Customer will not: (i) pre-fetch, index,
> store, reshare, or rehost Google Maps Content outside the services; **(ii) bulk download Google
> Maps tiles, Street View images, geocodes, directions, …**

> **§3.2.3(b) No Caching.** Customer will not cache Google Maps Content except as expressly
> permitted under the Maps Service Specific Terms.

> **§3.2.3(c) No Creating Content From Google Maps Content.** Customer will not create content
> based on Google Maps Content. For example, Customer will not: (i) trace or digitize roadways,
> building outlines, utility posts, or electrical lines …; (ii) create 3D building models from 45°
> Imagery …; (iii) build terrain models based on elevation values …; (iv) …; **(v) construct an
> index of tree locations within a city from Street View imagery**; … **(vii) use Google Maps
> Content to improve machine learning and artificial intelligence models, including to train,
> test, validate or fine-tune the models.**

> **§3.2.3(e) No Use With Non-Google Maps.** … Customer will not … **(ii) display Street View
> imagery and non-Google Maps on the same screen**, …

Two operational details that matter for engineering:

* **The only Street View field you may persist is `pano_id`.** The
  [Maps Service Specific Terms](https://cloud.google.com/maps-platform/terms/maps-service-terms) §3
  allows caching of "Google ID values … for example … `pano_ID`, from Street View Static API".
  Lat/lng, dates, images and derived products are all temporary-at-best.
* **There is no research/academic exemption** in the Terms. Papers that scrape GSV for training
  data are, strictly speaking, in breach; they are usually tolerated because they are low-volume
  and non-commercial, which is not a defence you want to rely on for a public repo or a
  government-facing deliverable.

### Why this matters for *this* project specifically

`DeepForest-HK` builds a city-scale tree index and trains/validates a detector — clauses
**§3.2.3(c)(v)** and **§3.2.3(c)(vii)** name both activities almost word for word. A GSV-based
tree inventory for Hong Kong is therefore **outside** what the API licence permits, regardless of
whether you pay for the calls. The same conclusion applies to:
* assembling panoramas from `Street View Tiles` (that is §3.2.3(a)(ii) bulk download);
* mosaicking 640 px crops into a pseudo-panorama (derivative work, §3.2.3(c));
* displaying results over an OSM / HK1980-Grid basemap in the same window (§3.2.3(e)(ii)).

---

## 4. Can GSV be used for 3D modelling?

### 4.1 Legal answer

No. See §3. Paying per request buys you the right to *display* imagery inside a Google-Maps-based
application — it does not buy rights to reconstruct geometry, to store the result, or to publish
it. The one sanctioned Google 3D product is **Photorealistic 3D Tiles**, and it is render-only.

### 4.2 Technical answer (what you would get even if licensing were not an issue)

**What the data actually is.** A GSV panorama is an equirectangular mosaic stitched from a
roof-mounted rig roughly 2.5–3 m above the road. The API gives you a *perspective crop* of it,
max 640 x 640 px, with `fov <= 120`. Metadata is limited to `pano_id`, capture date, copyright,
lat/lng snapped to the panorama, and heading inferred from the pano graph — **no roll/pitch, no
lens model, no exposure, no baseline.**

| Constraint | Consequence for reconstruction |
|---|---|
| 640 px cap, ~90° fov | ≈ 1.4 mrad/px. At 10 m range ≈ 1.4 cm/px; at 30 m ≈ 4 cm/px; at the crop edges it degrades further. A full 13K-wide panorama would take ~150–250 tile requests — i.e. the bulk download that §3.2.3(a)(ii) prohibits |
| Monocular rig, captures months/years apart | Matching must bridge different lighting, seasons, foliage, parked cars and construction. There is no synchronised stereo pair |
| Stitching, blending, tone-mapping | Seams and ghosting become false 3D structure; photogrammetric residuals blow up exactly at foliage and thin structures |
| Pose = lat/lng + heading only | No 6-DoF refinement, no camera intrinsics → absolute accuracy is metres, not centimetres; global drift along a street |
| Urban occlusion | Parked vehicles, signboards, footbridges, dense canopy hide both trunk bases and facades; the capture vehicle cannot enter parks, slopes or villages — where most HK trees actually are |
| Moving objects, blurring | Pedestrians/vehicles are removed or blurred, which *removes* the texture MVS needs |

**Trees are the worst possible target for image-based reconstruction.** Foliage is self-similar
(every leaf looks like every other leaf), semi-transparent, non-rigid and moves in the wind. SfM/MVS
converges to a noisy "fluff" envelope rather than branch geometry, which is why quantitative
structure models (QSM: [`TreeQSM`](https://github.com/InverseTampere/TreeQSM)) are built from
terrestrial/mobile **laser scans** at mm–cm density, not from street photos.

**What the existing projects actually achieve.** The `streetview-to-3d` style repos produce a
walkable corridor point cloud — good for previews and navigation, not measurable canopy volume.
`HoliCity` gets its 3D labels from *existing city 3D models* plus GSV images, not from
multi-view stereo of GSV. That is the honest state of the art.

### 4.3 What GSV *is* legitimately good for

* **2D street-level semantics**: green view index (GVI), visible-sky ratio, tree/trunk detection,
  facade and street-furniture segmentation — i.e. what
  [`Helsinki_GreenView`](https://github.com/geoporttishare/Helsinki_GreenView) and
  [`Green-View-Index-for-QGIS`](https://github.com/kowalski93/Green-View-Index-for-QGIS) do.
  Note that even a city-wide *tree index* built this way collides with §3.2.3(c)(v).
* **Manual visual QA** of a handful of locations (a few dozen images a month, not stored).
* **Monocular relative depth** for illustration (`360MonoDepth`, Depth-Anything style models) —
  relative, ungeoreferenced, not a survey product.

---

## 5. Who is already doing GSV → 3D (repo landscape)

Star counts and activity as of 2026-10-27. Nothing here changes the licensing conclusion above;
they are listed so you can see what is technically achievable and which pieces are reusable
*with imagery you are allowed to use* (own capture, Mapillary, licensed research datasets).

### 5.1 Street-level capture → 3D

| Repo | ★ | What it does | Licence | Notes |
|---|---|---|---|---|
| [`mapillary/OpenSfM`](https://github.com/mapillary/OpenSfM) | 3,812 | Structure-from-Motion pipeline built for street-level sequences | BSD-2 | The production-grade choice for your **own** car/360 capture; handles equirect + GPS priors |
| [`amir32002/3D_Street_View`](https://github.com/amir32002/3D_Street_View) | 510 | Stanford dataset: 25 M GSV images with 6-DoF pose + 3D models of 8 cities | research | The reference dataset for pose estimation / matching. Request access, don't scrape |
| [`zhou13/holicity`](https://github.com/zhou13/holicity) | 97 | London city-scale GSV dataset with holistic 3D structures | research (custom) | Good template for "GSV + city 3D model" fusion |
| [`THU-LYJ-Lab/SS3DM-Benchmark`](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark) | 15 | NeurIPS'24 benchmark: street-view surface reconstruction against synthetic 3D mesh | none stated | Useful evaluation protocol if you build a reconstruction model |
| [`YellowO2/streetview-to-3d`](https://github.com/YellowO2/streetview-to-3d) | 14 | Corridor point cloud from GSV / Apple Look Around panoramas | MIT | Shows the realistic output quality: visualization-grade, not survey-grade |
| [`manurare/360monodepth`](https://github.com/manurare/360monodepth) | 209 | Monocular depth from high-res 360 panoramas | research | Depth for illustration; not metric without control points |
| [`rbejar/StreetView3DOverlay`](https://github.com/rbejar/StreetView3DOverlay) | 58 | Overlay your own 3D models on GSV panos with three.js | — | *Display* only — the legal way to mix your 3D data with GSV |
| [`google-deepmind/streetlearn`](https://github.com/google-deepmind/streetlearn) | 326 | Navigation agents on GSV graph | Apache-2.0 | Research infrastructure |

### 5.2 Downloaders (know the risk)

| Repo | ★ | Notes |
|---|---|---|
| [`sk-zk/streetlevel`](https://github.com/sk-zk/streetlevel) | 244 | MIT. Downloads panoramas + metadata from Street View, **Apple Look Around**, Baidu, Yandex. Handy, but GSV bulk download is exactly §3.2.3(a)(ii); Look Around has its own terms |
| [`robolyst/streetview`](https://github.com/robolyst/streetview) | 548 | Python package for current/historical GSV photos — same caveat |
| [`spite/GSVPano.js`](https://github.com/spite/GSVPano.js) | 140 | Assembles full panoramas from tiles in the browser — the tile-assembly pattern §3.2.3(a)(ii) forbids |

### 5.3 Tree / greenery side (reusable with legal imagery)

| Repo | ★ | What it does |
|---|---|---|
| [`weecology/DeepForest`](https://github.com/weecology/DeepForest) | 785 | The crown detector this project already uses (aerial RGB) |
| [`InverseTampere/TreeQSM`](https://github.com/InverseTampere/TreeQSM) | 233 | Quantitative structure models (trunk/branch cylinders) from **laser** point clouds |
| [`geoporttishare/Helsinki_GreenView`](https://github.com/geoporttishare/Helsinki_GreenView) | 23 | End-to-end GVI pipeline from GSV images (scripts + methodology) |
| [`kowalski93/Green-View-Index-for-QGIS`](https://github.com/kowalski93/Green-View-Index-for-QGIS) | 15 | QGIS plugin for GVI over an AOI |

---

## 6. What to use instead for HK street-level / tree 3D

Hong Kong is unusually well served here — better data than GSV exists, and it is licensable.

### 6.1 Lands Department Mobile Mapping System (MMS) — the direct GSV substitute

The Lands Department runs its own vehicle-based MMS and sells the data
([3D Mapping page](https://www.landsd.gov.hk/en/survey-mapping/mapping/3d-mapping.html)):

| Item | Value |
|---|---|
| Coverage | Hong Kong Island, Kowloon, Lantau, New Territories |
| Collection | Continuous updating from 2021 onward |
| Point cloud density | 4–6 cm |
| Imagery | 7,000 x 3,500 px or better, 2:1 equirectangular (vs GSV's 640 x 640 crop) |
| Formats | Point cloud `.las` (ASPRS), images `.jpg` |
| Absolute accuracy | 0.3 m horizontal, 0.5 m vertical; ~5 cm relative |
| Cost | Charged on a cost-recovery basis (quotation); **1 GB free sample** for testing |
| Conditions | Government copyright — **must not be redistributed** to third parties |
| Contact | `mmsdata.services@landsd.gov.hk`; index in Shapefile; browser preview in **Open3Dhk → Streetscape 360** |

For street-tree work this is strictly better than GSV on every axis (resolution, density, accuracy,
licence) — and it is the same vehicle viewpoint, so the same algorithms apply.

### 6.2 Lands Department 3D Visualisation Map — city-scale 3D, including vegetation

Launched territory-wide in March 2025
([3D Digital Map](https://www.landsd.gov.hk/en/survey-mapping/mapping/3d-mapping.html)):

* **3D Visualisation Map (individualised models)** — geometry + texture for building,
  infrastructure, **vegetation**, waterbody, terrain and generic objects, whole territory.
* **Non-textured models** — geometry only (useful as a city model / occlusion mask).
* **Tile-based mesh** — photogrammetric mesh from oblique aerial imagery, whole territory.
* **3D Spatial Data series (3D-BIT00)**, **3D Pedestrian Network**, **3D Indoor Map**.

Download through the **CSDI Portal**, browse on **Open3Dhk**. If your deliverable is a "3D urban
forest layer", the vegetation class is already there — enrich it with DeepForest crown polygons +
LiDAR heights instead of reconstructing anything from street images.

### 6.3 Free LiDAR for canopy height (already on this repo's roadmap)

CEDD LiDAR is distributed through the CSDI Portal
([DATA.GOV.HK listing](https://data.gov.hk/en-data/dataset/hk-cedd-csu-lidar)). Standard recipe:

```
LiDAR point cloud → ground classification → DEM
                  → DSM (first returns)    → CHM = DSM − DEM
DeepForest crowns (aerial ortho) ∩ CHM → per-tree height, crown volume (voxel), canopy closure
```

That gives *measurable* 3D metrics with fully licensed inputs and no photogrammetry at all.

### 6.4 Open street-level imagery: Mapillary / Panoramax

* **Mapillary** (Meta) — free API with a client token; rate limits 60,000 entity req/min,
  10,000 search req/min, 50,000 tile req/day; user imagery is **CC BY-SA 4.0**, so derived
  datasets (including 3D reconstructions) are allowed provided you attribute and ShareAlike.
  Process with `mapillary_tools` + `OpenSfM`. Check HK coverage first via the vector tiles.
* **Panoramax** — federated, open-source, self-hostable street-level imagery (code on GitLab),
  OSM-native. Good if you want your own capture campaign without a vendor.

### 6.5 Capture it yourself

An Insta360-class 360 camera + RTK GNSS on a car/bike/backpack, processed with **OpenSfM** or
COLMAP/Meshroom, is a weekend project and gives you clean rights. For trunk and branch structure,
a terrestrial laser scanner (or even phone LiDAR for coarse stems) still beats any camera.

### 6.6 Comparison

| Source | Street-level view | 3D geometry | HK coverage | Cost | Can you publish derived 3D data? |
|---|---|---|---|---|---|
| GSV Static API | yes (640 px crop) | no (and forbidden) | good | $7 / 1,000 | **No** |
| Google Photorealistic 3D Tiles | no | yes (render only) | major areas | $6 / 1,000 tile requests | **No** (no export) |
| LandsD MMS | yes (7K pano + LAS) | yes, 4–6 cm | territory-wide | quotation | Within your organisation only — no redistribution |
| LandsD 3D Visualisation Map | no | yes (incl. vegetation) | territory-wide | free download via CSDI | Per CSDI terms (generally reusable with attribution) |
| CEDD LiDAR | no | yes (point cloud) | territory-wide | free | Yes, per data.gov.hk terms |
| Mapillary | yes | yes (you reconstruct) | patchy, check tiles | free | **Yes**, CC BY-SA 4.0 |
| Own 360 capture | yes | yes | wherever you drive | hardware + time | Yes |

---

## 7. Recommendation for this repo

1. **Do not build the tree inventory on GSV.** §3.2.3(c)(v) and (vii) describe exactly what
   `DeepForest-HK` does; a paid key does not change that. Keep GSV to manual visual QA of a few
   dozen locations at most.
2. **Finish the LiDAR branch first** (already "In Progress" in the README). CHM + DeepForest crowns
   gives height and crown volume — the 3D metrics that actually matter for risk and heat work.
3. **Use the LandsD 3D Visualisation Map** for the city-scale 3D context layer; use its
   *vegetation* class as the baseline canopy shell to enrich.
4. **Buy/evaluate the free 1 GB MMS sample** for stem-level work on a pilot street; if it fits,
   quote a district-sized extract instead of building a GSV pipeline.
5. **For a public demo**, Mapillary + `OpenSfM` is the only route that lets you show street-level
   3D *and* publish it.
6. Document the licence of every input in the README so downstream users (and any government
   partner) can see the pipeline is clean.

---

## 8. Sources

* Street View Static API — usage and billing: <https://developers.google.com/maps/documentation/streetview/usage-and-billing>
* Street View Static API — request parameters: <https://developers.google.com/maps/documentation/streetview/request-streetview>
* Street View Static API — image metadata: <https://developers.google.com/maps/documentation/streetview/metadata>
* Google Maps Platform price list (global): <https://developers.google.com/maps/billing-and-pricing/pricing>
* Google Maps Platform Terms of Service §3.2.3: <https://cloud.google.com/maps-platform/terms>
* Maps Service Specific Terms §3 (Google ID caching): <https://cloud.google.com/maps-platform/terms/maps-service-terms>
* Photorealistic 3D Tiles: <https://developers.google.com/maps/documentation/tile/3d-tiles>
* Street View Publish API (upload your own 360 imagery): <https://developers.google.com/streetview/publish>
* Mapillary API documentation: <https://www.mapillary.com/developer/api-documentation>
* Mapillary Terms of Use (CC BY-SA 4.0 user content): <https://www.mapillary.com/terms>
* HK Lands Department 3D Mapping / Mobile Mapping System: <https://www.landsd.gov.hk/en/survey-mapping/mapping/3d-mapping.html>
* HK LiDAR dataset: <https://data.gov.hk/en-data/dataset/hk-cedd-csu-lidar>



