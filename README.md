# DeepForest-HK: Urban Canopy Detection & Municipal Inventory Mapping

## Overview
An automated pipeline applying the `DeepForest` machine learning model to high-density urban environments in the Asia-Pacific region, starting with Hong Kong. This project bridges the gap between deep learning-based crown delineation and open-source municipal tree inventories (e.g., Hong Kong HyD and ArchSD datasets).

Inspired by the "Auto Arborist" project, this prototype demonstrates how computer vision can be scaled in subtropical urban forests to support smart city management, digital twins, and urban heat mitigation.

## The Problem
While individual tree-crown detection models like `DeepForest` excel in natural ecosystems (like the NEON network), high-density, multi-layered urban environments present unique challenges. Furthermore, cities often possess massive datasets of street and slope trees with rich biological and risk-assessment data, but these exist as discrete spatial points rather than mapped crown polygons.

## The Solution (Planned)
This pipeline performs three core tasks:
1. **Crown Delineation:** Uses the pre-trained `DeepForest` neural network to predict bounding boxes for tree crowns from high-resolution Hong Kong aerial imagery.
2. **Spatial Data Fusion:** Uses geospatial joins to automatically match predicted crown polygons with municipal ground-truth point data (species, DBH, risk level).
3. **LiDAR Integration (In Progress):** Extracts Z-axis data from annual Hong Kong LiDAR to assign tree height constraints, filtering out non-canopy urban infrastructure.

## Future Applications
* **Tree Risk Monitoring:** Cross-referencing crown size and height against municipal risk-assessment flags for typhoons.
* **Tree Equity & Heat Mitigation:** Mapping canopy volume accurately to optimize urban cooling initiatives.
* **Digital Twin Integration:** Generating 3D urban forest layers for smart city planning.

## Demo result in an Urban Park
![MASKED_2024_E220473C](https://github.com/user-attachments/assets/752db3dd-fc49-4d85-9ac8-5b028e23edec)

![MASKED_2025_E259476C](https://github.com/user-attachments/assets/bc4795c9-064f-416e-8a6b-545b398a15c1)

## Street-level imagery & 3D: what is (and is not) allowed

A common question is whether Google Street View (GSV) can supply the 3D layer. Short answer:
not for this use case.

* **How/price** — Street View Static API returns one perspective crop per request, max
  **640 x 640 px**, at **$7.00 per 1,000 panoramas** with 10,000 free calls per month
  (metadata calls are free and uncapped).
* **Licence** — Google Maps Platform ToS §3.2.3 prohibits bulk Street View downloads (a)(ii),
  caching (b), creating content from Maps Content (c) — which explicitly includes
  *"construct an index of tree locations within a city from Street View imagery"* (c)(v) —
  and using Maps Content to train or validate ML models (c)(vii).
* **Therefore** this pipeline stays on aerial imagery + LiDAR, and where street-level detail is
  genuinely needed it uses the Lands Department Mobile Mapping System (4–6 cm LAS + 7,000 x 3,500
  panoramas) or Mapillary (CC BY-SA 4.0) instead.
* **Closest published precedent** — Rodríguez-Puerta et al. (2022), *Sensors* 22(9):3269, uses GSV
  only to *cross-validate* ALS treetop detections (precision 51% → 76%, recall 86% → 77%). It never
  rebuilds geometry from GSV; its "point clouds from street level imagery" are sets of detected tree
  points. The same fusion is implemented here in `tree_detection_fusion.py`.
* **Species / pest / disease** — feasible at crown scale (growth form, dieback, lean, deadwood),
  not at leaf scale (1 mm borer holes, leaf spots). Inventory species labels + `pano_label_manifest.py`
  give you training data without manual annotation; see §5.6 of the write-up for the honest
  per-target verdict.

Full write-up, worked cost examples, technical limits of GSV photogrammetry, the Hong Kong
alternatives and the related open-source projects:
**[docs/street_view_api_and_3d.md](docs/street_view_api_and_3d.md)**.

Plan a survey's volume and cost offline with:

```bash
python3 gsv_survey_planner.py --length-km 40 --spacing-m 10
python3 gsv_survey_planner.py --length-km 2100 --spacing-m 10 --headings 4 --json
python3 gsv_survey_planner.py --self-test   # checks the documented price examples

python3 tree_detection_fusion.py --detected lidar_tops.csv --reference inventory.csv \
    --cross-check deepforest.csv --dedupe 4
python3 tree_detection_fusion.py --self-test

python3 pano_label_manifest.py --panos panos.csv --trees trees.csv --out manifest.csv
python3 pano_label_manifest.py --self-test
```
