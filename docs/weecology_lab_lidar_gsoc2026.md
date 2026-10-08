# Weecology lab × LiDAR × GSoC 2026 研究筆記

檢查日期：2026-10-08
對象：`weecology`（University of Florida, Dept. of Wildlife Ecology & Conservation）

## 1. Lab focus

兩個創辦人：**Ethan White**、**Morgan Ernest**（Professor of Wildlife Ecology and Conservation, UF）。

官網 `weecology.org/research` 列出 6 個項目：

| 項目 | 內容 |
|---|---|
| Portal Project | Chihuahuan Desert 長期（30+ 年）小型哺乳類／植物／蟻／天氣實驗 |
| Community Forecasting | 生態預測（forecasting），Portal 長期實驗預測 |
| Ecological Dynamics | regime shift、時間序列動態 |
| Everglades Wading Birds | 佛州大沼澤涉禽長期族群監測 + UAS 影像 |
| **Forest Remote Sensing** | **DeepForest**、MillionTrees、NeonTreeEvaluation |
| AI for Bird Monitoring | 飛機／無人機影像雀鳥偵測（DeepForest bird、Everglades） |

Lab DNA：open science + long-term data + benchmark-driven + ecoinformatics。
**唔係商業 lab** — 出方法、開放數據、benchmark，唔出 product／服務。

核心人物：
- **Ben Weinstein** — Research Scientist，DeepForest 主腦
- **Josh Veitch-Michaelis** — Research Scientist，DeepForest 開發
- **Henry Senyondo** — Software Developer
- **Glenda Yenni** — Research Scientist / Project Manager
- 資金：近期由 World Resources Institute (WRI) 支持，早期由 NSF 支持

## 2. 同 LiDAR 嘅 intersection（高，但方向唔同）

### (a) DeepForest 個 base model 本身係 LiDAR 造出來嘅
- Repo `weecology/deepforest-pretrain`（舊名 **DeepLidar Pretrain**）：
  「imagery and annotations are created from NEON data, with initial boxes created
  from **LIDAR canopy height maps**」→ 用 LiDAR CHM 做無監督樹冠偵測產生弱標註，
  再 tile 成 DeepForest 預訓練數據（40M+ tree locations，27 個 NEON sites）。
- 即係模式係：**LiDAR 做 label，RGB 做 inference**。

### (b) MillionTrees benchmark（2026 仍活躍）入面嘅 LiDAR / TLS 來源
- **Weinstein et al. 2018** — NEON coregistered LiDAR + RGB，弱監督，40M+ locations
- **Šrollerů et al. 2025** — LiDAR-derived crown bounding boxes projected into orthoimagery
- **Allen et al. 2025**（Joensuu/Tajo）、**Frey et al. 2026**（EcoSense, Central Europe）—
  **TLS（terrestrial LiDAR）樹冠多邊形做 validation-only ground truth**，
  明文規定唔可以做 hyperparameter tuning
- **Dubrovin et al. 2024** — Kaggle UAV LiDAR point clouds + RGB orthophotos（Perm Krai, Russia）
- 規模：2,323,333 人手標註 / 59,500 影像 / 51 個來源；另有 6,864,993 弱標註

### (c) 純 LiDAR 樹冠分割 pipeline
- Repo `weecology/TreeSegmentation`（R + `lidR`）：CHM watershed / 無監督分類做
  LiDAR-based crown delineation，配 `DeepLidar`（Python，已 archive，功能搬入 DeepForest）
- Repo `weecology/NeonTreeEvaluation`：RGB + **hyperspectral + LIDAR** 樹木偵測 benchmark
- Repo `weecology/DeepTreeAttention`：hyperspectral 樹種分類

### (d) 關鍵分歧 — DeepForest 唔食 LiDAR
- DeepForest 模型 input 係 **3-band RGB**，無 LiDAR ／ 深度 channel，無 RGB+LiDAR fusion。
- 舊 issue「Custom backbone for multiple inputs」(DeepForest_demos #3) 一直冇做。
- 所以如果要 LiDAR-first / point cloud 做 inference（DBH、trunk tilt、canopy volume、
  點雲去噪、TRAQ risk）→ weecology 完全冇 coverage（除咗 `allometry` /
  `WestFall_Allometries` 用 FIA field data 做 DBH→biomass allometry）。
- 結論：**互補多過重疊**。weecology 用 LiDAR 做 label / 驗證；Arbotic 用 LiDAR 做量測。

### (e) 可以點接
1. City-wide canopy mapping 用 DeepForest/MillionTrees 模型，再用自家 LiDAR/TLS 做
   獨立驗證 → 直接跟 Allen et al. 2025 個 TLS validation 思路（學術上可發表）
2. LiDAR CHM crown segmentation 可以對標 `TreeSegmentation` 個 lidR pipeline
3. 反向貢獻：HK urban LiDAR/TLS 數據入 MillionTrees（UrbanLondon 已經係 urban 先例，
   佢地公開歡迎數據貢獻）— 最實際嘅 intersection

## 3. GSoC 2026（DeepForest under NumFOCUS）

Issue #1251（2025-12-28 開）原本寫「唔預期有 GSoC 2026 project」，後來 update 改口話會提供。
最後 2 個學生（weecology people page 只有 2 位 GSoC student）：

### (1) Muhammad Saqlain — Recovering Forest Damage Annotations from Aerial Imagery
- Repo：`weecology/forest-damage-segmentation`
- Mentors：Ethan White、Henry Senyondo、Josh Veitch-Michaelis、Ben Weinstein
- 問題：USFS Aerial Detection Survey (ADS) 嘅 ~48,000 個森林損害 polygon
  係從飛機上手繪，位置錯 50–500 m，加上 NAIP 影像同 survey 差 1–3 年 → 唔可以直接做 training label
- 結果（負面結果先行）：假設「係位移」→ 錯，係 reshape 唔係 shift；
  classical hybrid energy alignment（70% inside-outside contrast + 20% contour distance
  + 10% out-of-bounds penalty）→ mean IoU 0.082 → 0.231；最後 pixel-level segmentation
  （TreeFinder NeurIPS 2025 pretraining）→ mean IoU ~0.30 vs ADS 自己 0.115
- 三個誠實評估規則：fold 按地理位置分、同一 tile 嘅 crop 唔可以跨 train/test、
  epoch 同 threshold 喺 15% hold-out 揀（唔用 test set，否則虛高 0.056）
- 有做 confidence ranking（rho = +0.62）但結論係唔需要做 gate
- 時間：coding period 2026-05-27 → 08-25（12 週）

### (2) Vicky Sharma — Recovering computer vision annotations from historical aerial imagery
- Repo：`weecology/recovering-computer-vision-annotations`
- 問題：2010–2021 Gulf of Mexico 航空照片，調查員用 point-counting 工具點雀鳥，
  工具只存 screenshot 冇存坐標 → 18,304 張 screenshot、2.81M 個雀鳥點焗死喺 pixel 裏面
- 做法：從 screenshot 反推每個點嘅坐標、讀圖例判斷物種、map 返原相
- 結果：registration 96.7%（0.38 px median error）、dot placement 0.65 px median、
  分類 0.781、匯出 118,270 boxes / 413 張相、fine-tune bird model mAP@50 0.036 → 0.087、
  176 tests passing；約 48% 影像通過檢查但覆蓋 ~72% 嘅點（~2M annotations）

### 共通點（重要）
兩個 GSoC 2026 項目都係 **「annotation archaeology」** — 從舊有、唔對齊、唔可用嘅
標註裏面救返可用訓練數據，**完全冇改 model architecture**。
再一次印證 lab 立場：「data, not architecture, is the limiting factor」。

### 對 HK 項目嘅啟示
HK 政府樹木記錄（HyD / ArchSD 點位、TRAQ 報告）本身就係同一類 "coarse / misaligned labels"。
呢兩個 GSoC 個方法論（geographic split、confidence ranking、negative results 照記錄、
label provenance 檢查）可以直接照抄落去。

## 4. 參考

- Lab: https://www.weecology.org/research/ , https://www.weecology.org/people/
- deepforest-pretrain (DeepLidar Pretrain): https://github.com/weecology/deepforest-pretrain
- MillionTrees: https://github.com/weecology/MillionTrees , https://milliontrees.idtrees.org
- TreeSegmentation (lidR): https://github.com/weecology/TreeSegmentation
- NeonTreeEvaluation: https://github.com/weecology/NeonTreeEvaluation
- GSoC 2026 issue: https://github.com/weecology/DeepForest/issues/1251
- GSoC project 1: https://github.com/weecology/forest-damage-segmentation
- GSoC project 2: https://github.com/weecology/recovering-computer-vision-annotations
- Saqlain blog: https://musaqlain.dev/blog/gsoc-2026-community-bonding/
- Arbotic GeoAI: https://linkedin.com/company/arbotic-geoai (HK, Cyberport CCMF, LiDAR → DBH / trunk tilt / canopy volume / TRAQ / digital twin)
