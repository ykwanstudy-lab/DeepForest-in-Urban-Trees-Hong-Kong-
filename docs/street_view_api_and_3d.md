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

### 3.1 "But this published paper did it" — breach of contract is not the same as breaking the law

Three separate layers, routinely conflated:

| Layer | What it is | What can happen | Is it "illegal"? |
|---|---|---|---|
| **Contract (ToS)** | You accepted the Google Maps Platform Terms when you created the billing account / used the key | Google can suspend or terminate the project, revoke the key, stop billing, demand deletion, and in principle seek damages or an injunction (§5 Suspension, §10 Termination) | No — it is a **contractual** breach, not a criminal offence |
| **Copyright** | GSV imagery is copyrighted by Google and its imagery providers | A rights holder can send a takedown notice, request removal from GitHub, or bring a civil claim for infringement if you redistribute the imagery or publish it as your own dataset | Civil wrong; criminal only in narrow commercial-scale cases |
| **Practical enforcement** | What Google actually does | Individual researchers are almost never pursued (cost >> benefit). Enforcement is aimed at commercial scale, resale, or public repos that ship a downloader | Risk is **low for private research, high for public/government deliverables** |

So the honest summary: a paper like this is **in breach of the Terms** (and possibly of copyright if
imagery is redistributed), but it is **not "breaking the law"** in the criminal sense, and peer
review does not check licensing. Journals review the science; MDPI is not going to audit your API
key. That is why these papers exist and keep being published — **publication is not a licence**.

Risk by context, for this project:

* private experiment, nothing published or shared → low practical risk, still a breach;
* public repo + downloader script + city-scale tree index → medium (takedown, institutional IT
  policy, reviewer questions);
* tender, government delivery, audit → high (procurement requires clean provenance; "a paper did
  it" is not an acceptable answer in a data-licence register).

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

### 5.4 Case study — Rodríguez-Puerta et al. (2022), *Sensors* 22(9):3269

> Rodríguez-Puerta, F.; Barrera, C.; García, B.; Pérez-Rodríguez, F.; García-Pedrero, A.M.
> *Mapping Tree Canopy in Urban Environments Using Point Clouds from Airborne Laser Scanning and
> Street Level Imagery.* Sensors 2022, 22, 3269. <https://doi.org/10.3390/s22093269> (CC BY 4.0)

This is the closest published precedent to the idea "use GSV to improve the LiDAR inventory", so
it is worth reading precisely.

**What they actually did** (Pamplona, Spain; 8 circles of 150 m radius, 5 of them
vehicle-traversable):

| Stage | Input | Method |
|---|---|---|
| 1A | **3,102 GSV images** (77 in 2015, 37 in 2017, 1,100 in 2018, 1,888 in 2019) downloaded via the GSV API | Mask R-CNN (Detectron2), fine-tuned on 2,000 LabelMe-masked images, detects *stem and crown*; each detection is geolocated by trigonometry from the panorama's heading/pitch/FOV → one point per tree |
| 1B | ALS from IGN, Sep–Nov 2017, LEICA SPL100, **14 first returns/m²** | FUSION `CanopyModel` CHM (0.25 m cells, smooth 3), building mask, then `ForestTools` local maxima → one point per treetop |
| 2A | GSV again | For every candidate, pull the 3 nearest panoramas; if all three trig positions agree within 4 m → one tree, placed at the centroid |
| 2B | RGB/NIR orthophotos (2014, 2017) | Random forest / SVM on 50 cm buffer zonal statistics, trained on 15,766 labelled points, to classify TREE / NOT TREE |
| 3 | — | Cluster detections within **4 m** (buffer → polygon cloud → centroid); validate against the municipal inventory with a **5 m** true-positive tolerance |

**Results** (mean of the 5 vehicle-traversable zones; p = precision, r = recall):

| Method | p | r | F1 |
|---|---|---|---|
| GSV only (stage 1A) | 41.1% | **85.1%** | 55.4% |
| ALS only (stage 1B) | 51.0% | **86.4%** | 64.1% |
| GSV → GSV debugging | 68.2% | 77.4% | 72.5% |
| GSV → ML debugging | 66.4% | 78.2% | 71.8% |
| **ALS → GSV debugging** | 76.2% | 76.8% | **76.5%** |
| ALS → ML debugging | **81.0%** | 71.4% | 75.9% |

**What that actually means**

* Both sources reach the same *recall* (~85–86%) but GSV produces far more false positives, so
  the useful role of GSV is **cross-validation**: it lifts ALS precision from 51% to 76% at the
  cost of ~10 points of recall. GSV is a **filter / second opinion**, not a geometry source.
* **No point cloud is reconstructed from GSV.** The paper's "point clouds from … street level
  imagery" are *sets of detected tree points* (the authors literally write "the point cloud was
  transformed into a polygon cloud"). LiDAR points are never modified, densified or corrected by
  GSV. There is no crown volume, no mesh, no DBH.
* Their own numbers show why GSV alone is weak for inventory: 41% precision, i.e. more than half
  of the street-level detections are not real separate trees (duplicates across images, trig
  geolocation error, stems vs crowns).
* Scope limit, stated by the authors: *"This methodology, being based on GSV street-level imagery,
  was only possible on vehicle-traversable streets."* Parks, slopes, villages and pedestrian
  zones — most of Hong Kong's canopy — are out of reach of the GSV stage.
* Licensing: 3,102 images were bulk-downloaded through the GSV API and used to fine-tune a neural
  network. Under today's Google Maps Platform ToS that is §3.2.3(a)(ii) + (c)(v) + (c)(vii). The
  paper is a scientific precedent, **not** a licence, and it does not make the approach reusable
  for a public repo or a government deliverable.
* **Timeline check.** The
  [archived Terms of Service of 1 April 2022](https://web.archive.org/web/20220401212655/https://cloud.google.com/maps-platform/terms/)
  — three weeks before the paper appeared — already contained, verbatim:
  *"(ii) bulk download Google Maps tiles, Street View images, …"* (§3.2.3(a)) and
  *"(v) construct an index of tree locations within a city from Street View imagery"* (§3.2.3(c)).
  So the two clauses that matter most were **in force at the time of the study**; the paper was not
  written under an older, looser regime. The one clause that did *not* exist yet is (c)(vii)
  (training ML models): in 2022 that activity fell under the general "will not create content based
  on Google Maps Content", rather than a clause naming machine learning explicitly.
* The authors themselves make the escape hatch explicit: **"Although GSV images have been used in
  this study, any street-level image is valid to replicate the proposed methodology."** In Hong
  Kong that means LandsD MMS panoramas (7,000 x 3,500, same geometry, licensed, 4–6 cm LAS
  alongside) instead of GSV — strictly better inputs for the identical pipeline.

**What is worth copying for this repo**

1. The **two-source fusion architecture**: independent detection from aerial/ALS + independent
   detection from street level, then mutual filtering. Our equivalent, with fully legal inputs:
   DeepForest crowns (aerial ortho) ↔ LiDAR CHM treetops ↔ municipal inventory points.
2. The **4 m merge radius** and **5 m true-positive tolerance** as starting parameters for dense
   HK street-tree spacing (paper: tree spacing > point spacing, so 4 m was chosen between the two).
3. Using the **municipal inventory as the validation reference** (their ground truth was Pamplona's
   official tree database) — exactly the HyD / ArchSD datasets already planned in this repo.
4. Precision/recall/F1 reporting at the *tree-point* level rather than only IoU on crowns, which
   is what an inventory user actually cares about.

A small, dependency-free implementation of points 1–2 (dedupe, one-to-one matching, precision /
recall / F1, cross-validation filter) is included as
[`tree_detection_fusion.py`](../tree_detection_fusion.py).

### 5.5 Is Mapillary enough to *train* a LiDAR-segmentation refinement model?

**Short answer: enough images, not enough labels — and for this task you should not train from
scratch at all.** What "enough" means depends on which part of the LiDAR product you are refining.

| What you want to fix in the LiDAR product | What the street view must supply | Supervision actually needed | Is Mapillary enough? |
|---|---|---|---|
| False treetops from CHM local maxima (poles, roofs, walls) | binary evidence "is there a tree at this XY?" | **none to train** — a pretrained detector zero-shot, plus 200–500 checked locations to *measure* accuracy | Yes, where coverage exists |
| Merged / split crowns, crown boundary | instance masks of individual crowns | instance labels; Vistas only has a `vegetation` *stuff* class, so you label it yourself | Images yes, labels no |
| Trunk position, crown base height, clearance | close-range images with visible stems | keypoints / lines for a few hundred trees | Images yes (flat 1.5–2 m images beat a 640 px GSV crop), labels no |
| Point-cloud semantic classes (tree vs building vs furniture) | semantic masks + camera poses | fine-tune on 200–2,000 labelled HK images | Yes, once you label them |

**What Mapillary actually provides**

* 2 billion images globally (milestone reached August 2023), free API with a client token.
* Rate limits: 60,000 entity requests/min, 10,000 search requests/min, 50,000 tile requests/day.
* Mostly **flat forward-facing images at ~1.5–2 m** plus 360 sequences — better for trunks than GSV's
  640 px perspective crop, and every image carries `captured_at`, `quality_score` and geometry.
* User imagery is **CC BY-SA 4.0**: derived datasets (including 3D) are allowed with attribution and
  ShareAlike.
* **Mapillary Vistas v2.0** is the only large labelled set: **25,000 images, 124 classes (70 of them
  instance-annotated)**, 29.3 GB, licence **CC BY-NC-SA 4.0 — research / non-commercial only**.
  It annotates `vegetation` as a stuff class, **not individual trees**, so it cannot teach tree
  instances or trunks.

**Label budget reality**

| Approach | Labelled data | Effort | Comment |
|---|---|---|---|
| Zero-shot pretrained segmentation (Mask2Former / SegFormer on Vistas or Cityscapes) | 0 | ~1 day | fine for "tree vs not-tree" masks; watch planters, aerial roots, bamboo scaffolds |
| Fine-tune for Hong Kong | 200–2,000 images | 1–3 weeks labelling | Rodríguez-Puerta used 2,000; 200–500 is often enough for a domain shift |
| Train from scratch | Vistas 25k + your data | months + GPU | not worth it for a refinement step |

⚠️ **Check the pretrained weights' licence, not just the imagery.** Vistas and Cityscapes are both
non-commercial. If this work feeds a tender or an operational government product, start from
permissively licensed weights (e.g. COCO/OpenImages-trained backbones, DeepForest's MIT weights) or
train on your own labels.

**Hong Kong coverage is the empirical question** — and it is the one thing Mapillary cannot promise.
A 0.01° cell (~1.1 km) with fewer than ~50 images should be treated as "no coverage". Probe before
you plan anything:

```python
import json, os, urllib.parse, urllib.request

TOKEN = os.environ["MAPILLARY_TOKEN"]          # client token from mapillary.com/dashboard/developers

def images_near(lat, lng, d=0.005):            # bbox must stay < 0.01 deg square
    query = urllib.parse.urlencode({
        "access_token": TOKEN, "bbox": f"{lng-d},{lat-d},{lng+d},{lat+d}",
        "fields": "id,captured_at,is_pano,width,height,quality_score", "limit": 2000,
    })
    with urllib.request.urlopen(f"https://graph.mapillary.com/images?{query}", timeout=30) as r:
        return json.load(r)["data"]
```

Run it on a 200 m grid over your AOI and plot image count per cell; that map is your training-set
size. If the AOI is blank, Mapillary cannot help — but the Lands Department **MMS sample (1 GB,
free)** covers the whole territory at 7,000 x 3,500 px, so it removes the coverage risk entirely.

**Verdict for this repo**

* Do **not** plan a "Mapillary-trained" model for a city-wide HK product. Coverage is patchy and
  unlabelled; you would spend the budget on labels anyway.
* Use a **pretrained segmentation/detection model** and fine-tune it on a few hundred **MMS-derived**
  HK images (guaranteed coverage, better resolution, licensed for internal use).
* Use Mapillary for what it is uniquely good at: free, legal, georeferenced imagery for spot checks,
  active-learning candidates, and before/after comparison where sequences exist.
* Use Vistas only for research prototypes, never in a commercial or government deliverable.

### 5.6 Species, pest and disease from street-level imagery: what is actually trainable

Source must be MMS / Mapillary / own capture — not GSV (§3). With an 8,000 x 4,000 MMS panorama a
tree at 10 m is roughly 1,200 px across, which is enough for crown-scale judgement and *not* enough
for leaf-scale diagnosis. Split the ambition accordingly.

| Target | What the camera sees at 5–15 m | Verdict |
|---|---|---|
| Growth form (palm / broadleaf / conifer / bamboo) | whole silhouette | **Easy.** 100–200 crops/class → >0.9 F1 |
| Species, top-20 HK street species | crown shape, leaf mass, trunk habit, flowers/fruit when present | **Plausible.** 0.6–0.8 top-1, ~0.85 top-3 with inventory labels + multi-view. Use top-3 for triage |
| Crown dieback / defoliation / dead tree | bare branches, thinning, brown crown | **Good.** Ordinal 0–3; quadratic weighted kappa 0.6–0.75 is realistic |
| Lean, root-plate lift, uprooting | whole trunk line vs horizon | **Good** — and directly useful for typhoon risk |
| Cavities, wounds, cracks, deadwood | visible with an 8K panorama at 5–10 m | **Workable** as flags, needs close passes and human confirmation |
| Bracket fungi / conks / cankers | 5–10 cm structures | **Marginal.** Rare positives → anomaly detection, not a 20-class classifier |
| Brown root rot (*Phellinus noxius*) | crown thinning, yellowing, dieback; the diagnostic root-collar crust does **not** show | **Suspect-level only.** Screen crowns, then ground inspection + lab confirmation |
| Red palm weevil | crown wilting / collapse yes; 1–2 cm entry holes barely | **Suspect-level only** |
| Polyphagous shot hole borer (*Euwallacea fornicatus*) | 1 mm entry holes | **No.** Needs <2 m and macro optics, not a moving vehicle |
| Leaf spots, galls, sap-sucking insects, scale | sub-centimetre symptoms | **No.** Leaf close-ups only |
| Epiphytes / mistletoe / vines | yes | **Good**, and useful for HK banyans |

**Frame the product as VTA pre-screening, not diagnosis.** Hong Kong runs a statutory-ish tree risk
regime (GLTMS / Tree Risk Assessment) where a qualified person signs off. A model that flags
"crown dieback 3/3, lean 12°, possible cavity" and routes the tree to an inspector is defensible;
a model that outputs "infected with PSHB" is not, and will not survive scrutiny after a failure.

**Where the labels come from (the part that decides the project)**

1. **Inventory as free labels.** HyD / ArchSD publish species per tree point. Combine with
   [`pano_label_manifest.py`](../pano_label_manifest.py) to crop that tree out of every panorama that
   sees it and inherit the species label — zero manual annotation for the species task.
2. **Multi-view aggregation.** The same tree appears in 3–10 frames. Average the logits (or vote) per
   tree; this is usually a bigger gain than a bigger backbone, and it also fixes occlusion.
3. **Split by tree, not by image.** Multi-view crops of the same tree in train and test is the
   classic way to publish an 0.95 that collapses in the field.
4. **Ordinal, not one-hot, for severity.** Dieback 0–3 has order; use QWK and a confusion matrix.
5. **Rare classes → anomaly detection + active learning.** Do not force 20 balanced classes out of 15
   positives; train a one-class / OOD score on healthy crowns and send the top-k to a human.
6. **Pretraining data caveat.** PlantVillage and similar leaf-close-up datasets are the wrong domain —
   they will not transfer. iNaturalist is useful for a coarse species prior, but check each photo's
   licence (CC BY-NC photos cannot go into a commercial product).

**Realistic label budget**

| Task | Labels | Expected at that budget |
|---|---|---|
| Growth form, 3–4 classes | 100–200 crops/class | F1 > 0.9 |
| Top-20 species | 150–300 trees/species × 3 views | top-1 0.6–0.8, top-3 ~0.85 |
| Crown condition, 4 levels | 300–500 crowns, balanced | QWK 0.6–0.75 |
| Defect flags (binary) | 100–200 positives per flag | recall 0.6–0.8 at precision ~0.5 → triage, not verdict |

**Geometry you need from LandsD.** The manifest above assumes a levelled panorama, the vehicle
heading from the trajectory, and a 2.5 m camera height. Ask for **camera exterior orientation**
(and ideally the IMU trajectory) in the MMS quotation (§6.1.2) — without it your bearing-to-pixel
mapping carries a few metres of error, which matters for a 6 m crown but not for a 15 m one.

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

### 6.1.1 What the free sample actually contains (verified 2026-10-27)

The
[sample point cloud data](<https://www.landsd.gov.hk/doc/en/survey-mapping/mapping/3d-mapping/sample%20point%20cloud%20data.zip>)
archive is **873 MB / 231 files**, one short trajectory:

| Content | Detail |
|---|---|
| `pointcloud/merge_001.laz` | 1 file, **418 MB** compressed LAS |
| `panorama1/original/*.jpg` | **209 images**, 1.7–2.8 MB each (~472 MB), sequential capture along the run |
| Readme / licence | none inside the archive — terms come with the quotation |

Resolution check: the first JPEG in the archive decodes to **8,000 x 4,000 px** (2:1), i.e. better
than the published "7,000 x 3,500 or better". That is ~11x the linear resolution of a 640 px GSV
crop, which is what makes species and crown-condition work plausible at all.

The companion **MMS trajectory index** (shapefile, 129 MB zip) shows where the vehicle actually
drove, so you can check coverage of your AOI *before* paying:
<https://www.landsd.gov.hk/doc/en/survey-mapping/mapping/3d-mapping/VMMS%20trajectory%20index.zip>

### 6.1.2 How to apply, and what it costs

There is **no published unit rate**. MMS is not in the SMO
[price list](https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/doc/pricelist.pdf)
(February 2026), which only lists the catalogue products: digital maps, orthophotos and
`3D-BIT00` at **HK$0**, and four digital aerial photo series at **HK$325–460 per sheet**. MMS is
handled as a separate paid service "subject to a fee which aims to recover the cost of providing
the service" — i.e. **a quotation per request**.

Procedure:

1. **Download the free sample and the trajectory index** (links above). Run your pipeline on it
   first; it is the only way to know whether the data fits your tree workflow.
2. **Define your AOI tightly.** A list of road names, a shapefile polygon, or HK1980 Grid
   coordinates. Cost scales with trajectory length and corridor area, so a 2–3 km pilot is far
   cheaper than "Kowloon".
3. **Email `mmsdata.services@landsd.gov.hk`** with: AOI, data required (LAS/LAZ and/or JPEG,
   whether you need the original 7,000 x 3,500 panoramas), capture period, intended use,
   organisation, and delivery preference (download link vs physical media).
4. LandsD replies with **quotation + terms and conditions + order form**; you pay, then they
   release the extract (volumes are large — plan storage and transfer).
5. Ask for **written confirmation of the licence for derived products** (can you publish maps,
   statistics, model outputs? can you share the data with a project partner?) before signing.

What to ask explicitly in the quotation request, because it drives the price:

* unit basis — per km of trajectory, per km² of corridor, per GB, or per sheet?
* minimum charge, and whether panorama images and point cloud can be bought separately;
* LAS or LAZ, and whether GPS/IMU trajectory + camera exterior orientation are included
  (you need these to co-register imagery with the LiDAR);
* delivery date for a newly captured area vs an existing archive run;
* whether the T&C allow use inside a government-funded project with external partners.

**Volume estimate.** The sample's 209 panoramas at 1.7–2.8 MB each means ~450 MB of JPEG per
kilometre *if* they were captured at ~5 m spacing (assumption — LandsD does not publish the MMS
spacing); add the LAZ for the same corridor (418 MB for that one tile). So budget roughly
**1 GB per km of street** for imagery + point cloud, and multiply by the number of passes.
Treat this as an order-of-magnitude figure for planning storage, not as a price.

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

Mapillary supplies *images*, not tree labels: its only large labelled set (Vistas, 25k images) is
CC BY-NC-SA 4.0 and annotates `vegetation` as a stuff class, not individual trees — see §5.5.

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
7. **Adopt the two-source fusion architecture from Rodríguez-Puerta et al. (2022)** (§5.4) with
   legal inputs — DeepForest crowns, LiDAR CHM treetops and the HyD/ArchSD inventory — and report
   tree-level precision/recall/F1 with `tree_detection_fusion.py`. That reproduces the paper's
   76% → 81% precision gain over a single source without touching GSV.
8. **Do not budget for "training on Mapillary"** (§5.5). Probe HK coverage first; if you need labels,
   fine-tune a pretrained model on 200–500 MMS-derived images instead. Check the *weights* licence
   too — Vistas and Cityscapes are non-commercial, which disqualifies them from a tender.
9. **If you pursue species / pest / disease ID** (§5.6): start with inventory-labelled pano crops
   (`pano_label_manifest.py`), growth form and crown-condition severity. Treat pest and disease
   outputs as VTA *suspect flags* — screening for a human inspection, never a diagnosis.

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
* Mapillary Vistas 2.0 release notes (25,000 images, 124 classes, 70 instance-annotated):
  <https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html>
* Mapillary 10 years / 2 billion images: <https://blog.mapillary.com/update/2024/01/12/Mapillary-10-Years-2-Billion-Images.html>
* Rodríguez-Puerta et al. (2022), *Sensors* 22(9):3269 — GSV × ALS tree detection fusion:
  <https://doi.org/10.3390/s22093269> (open access, CC BY 4.0)
* Google Maps Platform Terms of Service as at 1 April 2022 (Wayback Machine), used to confirm which
  clauses were already in force when that paper was published:
  <https://web.archive.org/web/20220401212655/https://cloud.google.com/maps-platform/terms/>
* HK Lands Department 3D Mapping / Mobile Mapping System: <https://www.landsd.gov.hk/en/survey-mapping/mapping/3d-mapping.html>
* SMO price list (February 2026) — catalogue products; 3D-BIT00 free, DAP HK$325–460/sheet, no MMS line item:
  <https://www.landsd.gov.hk/doc/en/mapping/digital-map/common/doc/pricelist.pdf>
* Digital map product information paper (formats, e.g. 3D-BIT00 in 3ds / 3D Max / FBX / VRML):
  <https://www.landsd.gov.hk/doc/en/mapping/digital-map/product.pdf>
* HK LiDAR dataset: <https://data.gov.hk/en-data/dataset/hk-cedd-csu-lidar>



