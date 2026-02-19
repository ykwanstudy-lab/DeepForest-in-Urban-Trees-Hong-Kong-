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


