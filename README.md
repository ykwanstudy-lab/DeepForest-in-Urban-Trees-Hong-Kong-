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

## Keeping in step with GeoLibre

The map tier runs on [GeoLibre](https://github.com/opengeos/GeoLibre) Desktop,
which releases roughly every four days; this repo is pinned to 2.8.x while
upstream is at 3.4.x. [`docs/geolibre-tracking.md`](docs/geolibre-tracking.md)
holds the status snapshot, the plugin-relevant changes in the releases skipped,
the plugin contract, an upgrade runbook, and the licensing notes.

```bash
python3 tools/geolibre_watch.py            # exit 1 when a pinned channel is behind
python3 tools/geolibre_watch.py --update   # re-pin after a deliberate bump
python3 tools/arbotic_plugin_doctor.py     # workspace plugin vs the installed copy
```

## Demo result in an Urban Park
![MASKED_2024_E220473C](https://github.com/user-attachments/assets/752db3dd-fc49-4d85-9ac8-5b028e23edec)

![MASKED_2025_E259476C](https://github.com/user-attachments/assets/bc4795c9-064f-416e-8a6b-545b398a15c1)
