# DeepForest 上游狀態備忘 (research notes)

檢查日期：2026-10-08（sandbox UTC）
來源：PyPI、GitHub releases / milestones、readthedocs 2.1.0、Hugging Face Hub、上游 source code

## 1. 直接答問題：係咪仲用緊 YOLOv8？

**唔係，而且 DeepForest 從來都無用過 YOLO（包括 YOLOv8 / YOLO11）。**

- 核心 detection backbone 係 **torchvision RetinaNet + ResNet-50 FPN**，
  即上游 `src/deepforest/conf/config.yaml` 裏面 `architecture: 'retinanet'`（default）。
- 預訓練權重 `weecology/deepforest-tree`（32.3M params，safetensors）最後更新
  2024-09-20，仍然係同一個 RetinaNet checkpoint（NEON 訓練），
  即 2.x 嘅 detection 精度同 1.x release model 基本一樣，變嘅係工程/生態。
- 2023 年有人開 issue #540「yolov8-ultralytics integration」，maintainer bw4sz 回覆
  （立場至今未變）：**"data, not architecture, is the limiting factor"**，
  願意支援任何模型但唔會內建 ultralytics。該 issue 已 closed，無 YOLO 整合。
- 想用 YOLO 嘅話，2.x 有 plug-in 路徑：喺 `src/deepforest/models/` 加 module
  + implement `create_model()`，只要 input/output 跟 torchvision 格式
  （`boxes` / `labels` / `scores`）就得。

## 2. 版本 timeline

| 版本 | 日期 | 重點 |
|---|---|---|
| 2.1.0 | 2026-02-25 | Kornia 取代 Albumentations；CLI 拆 sub-command；evaluation 改用 torchmetrics；支援 numpy 2.x；CropModel 加 macro-precision / `cropmodel.expand` |
| 2.0.0 | 2025-11-04 | 大型重寫（見下） |
| 1.5.2 | 2025-02-06 | 最後一個 1.x |
| 1.0.0 | 2021-06-07 | TensorFlow → PyTorch |

依賴（2.1.0）：Python >=3.10,<3.15、`torch>=2.2`、`torchvision>=0.17`、
`numpy>=2.0`、`pytorch-lightning<=2.6.1`、`hydra-core`、`kornia>=0.8.2`、`supervision`。

## 3. 2.0 有咩大 update（2025-11）

1. **Model hub 搬去 Hugging Face**：`use_release()` → `load_model(model_name="weecology/deepforest-tree")`
2. **Hydra 做 config 管理**：`deepforest(config_args={...})`、`deepforest --show-config`
3. **Albumentations → Kornia**（2.0 開始，2.1 完成）
4. **Visualization 改用 Roboflow `supervision`**：`plot_predictions()` → `plot_results()`
5. **CLI**：`deepforest train` / `predict` / `evaluate`，支援 config override
6. **CropModel 概念**：detection model + 第二階段 classification model（ResNet18/50）
7. **Point / keypoint 模型**（TreeFormer，PVTv2 density map，10 cm/px 訓練）

### 2.0 移除嘅 1.x API（我哋個 demo 中招）

- `use_release()` / `use_bird_release()` → `load_model(...)`
- `predict_tile(raster_path=...)` → `predict_tile(path=...)`
- `plot_predictions` / `draw_predictions` / `plot_points` / `draw_points` → `plot_results`
- `boxes_to_shapefile()` / `project_boxes()` → `image_to_geo_coordinates()`
- `augment=` → `augmentations=`
- `deepforest(num_classes=, label_dict=)` → 用 config

## 4. 現時支援嘅架構（上游 `src/deepforest/models/`）

| module | architecture 值 | 用途 |
|---|---|---|
| `retinanet.py` | `retinanet`（default） | bounding box detection |
| `DeformableDetr.py` | `DeformableDetr` | transformers Deformable DETR wrapper（測試覆蓋，`tests/test_detr.py`） |
| `treeformer.py` | `treeformer`（`conf/point.yaml`） | point / density map（TreeFormer, PVTv2 backbone） |

CropModel backbones：`resnet18`、`resnet50`。

## 5. 官方預訓練模型（Hugging Face `weecology/*`）

- `deepforest-tree`（default，box）
- `deepforest-bird`、`deepforest-livestock`、`deepforest-marine-biodiversity`
- `deepforest-tree-point`（TreeFormer，2026-04 上載）
- `cropmodel-tree-species`（148 species）、`cropmodel-tree-genus`（54 genus）
- `cropmodel-deadtrees`（alive/dead，95.8% acc）
- `everglades-nest-detection`、`everglades-bird-species-detector`

## 6. 開發中（milestone DeepForest 2.2，2 個 open issue）

- #758 **Polygon model support**（polygon workflow 支援 code 已喺 2026-08-30 merge，PR #1419）
- #460 Integrate SAM（Segment Anything）做 bbox → polygon 後處理

→ 對「crown polygon delineation」呢個項目係直接相關，值得跟。

## 7. 對我哋個 repo 嘅行動項

1. `Deepforest demo.py` 同 `hk_tree (pseudocode).py` 用嘅係 1.x API，要升級：
   - `model.use_release()` → `model.load_model(model_name="weecology/deepforest-tree")`
   - `predict_tile(raster_path=...)` → `predict_tile(path=...)`
   - `model.config["score_thresh"] = 0.5` → `main.deepforest(config_args={"score_thresh": 0.5})`
2. `hk_tree (pseudocode).py` 有 typo：`to_crs(epsG=2326)` → `to_crs(epsg=2326)`
3. 精度現實：prebuilt model 係美國 NEON 溫帶森林訓練，HK 高密度亞熱帶城市樹冠
   （密冠層、重疊樹冠）recall 會偏低 → 需要本地 fine-tune。
4. HK 正射影像多數係 10 cm/px，同 `deepforest-tree-point` 嘅訓練解析度吻合，
   可以試 point model 做樹冠中心點（比 bbox 更適合密林）。

## 8. 參考

- Changelog: https://deepforest.readthedocs.io/en/latest/whatsnew/history.html
- Prebuilt models: https://deepforest.readthedocs.io/en/latest/user_guide/02_prebuilt.html
- DeepForest 2.0 blog: https://jabberwocky.weecology.org/2025/11/04/deepforest-2-0/
- PyPI: https://pypi.org/project/deepforest/
- HF: https://huggingface.co/weecology/deepforest-tree
- YOLO issue #540: https://github.com/weecology/DeepForest/issues/540
- Milestone 2.2: https://github.com/weecology/DeepForest/milestone/6
