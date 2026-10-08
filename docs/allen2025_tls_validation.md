# Allen et al. 2025 — TLS 驗證方法論（DeepForest 效能真相）

論文：**Manual Labelling Artificially Inflates Deep Learning-Based Segmentation
Performance on RGB Images of Closed Canopy: Validation Using TLS**
作者：M.J. Allen, H.J.F. Owen, S.W.D. Grieve, E.R. Lines
（Cambridge Dept. of Geography；Grieve 在 QMUL）
期刊：Remote Sensing of Environment；arXiv:2503.14273v2（2025-03-19），CC BY 4.0

一句總結：**用人工手繪標註去評估樹冠分割模型，會嚴重高估效能。**
換成 co-located TLS 做獨立 ground truth，AP50 由 0.670 跌到 0.094。

## 1. 三個場地

| 場地 | 生態系 | 標註方式 | 樹冠數 | GSD |
|---|---|---|---|---|
| Joensuu, Finland | boreal 混交林（15 plots） | **TLS 衍生** | 1,271 | ~1.0–1.5 cm/px |
| Alto Tajo, Spain | Mediterranean | **TLS 衍生** | 1,387 | ~1.1–1.7 cm/px |
| Almorox, Spain | Mediterranean（同 Alto Tajo 相似） | **人工手繪**（QGIS 3.30） | — | — |

Almorox 就係對照組：同一個生態系，但用傳統人手標註。

## 2. 核心方法：由 TLS 點雲造出「由上面望落去」嘅樹冠多邊形

論文 Algorithm 1（重點步驟）：

1. 每個獨立分割好嘅 TLS 樹（`TLS_Segs`），轉去同一個 CRS
2. 逐棵樹用 **PDAL** 建 DEM（**resolution 0.02 m, window 1 px**）→ 即係每棵樹嘅冠頂高度柵格
3. 由所有樹嘅 min/max x,y 算出整個 plot 嘅 extent
4. 用 **GDAL** 建一個空 VRT（對齊 extent + DEM 解析度），讀成 array，兩個 band：
   - **Band 0 = DEM**（該 pixel 嘅冠頂高度）
   - **Band 1 = 該 pixel 最高嘅樹嘅 index**
5. 逐棵樹再建 VRT 對齊 pixel，更新 DEM 同「最高樹」band
6. **Polygonise Band 1** → 每個 polygon 屬於一棵樹
7. 過濾：每棵樹只保留面積最大嘅 polygon、移除內部幾何

→ 得出**唔重疊、由上空可見**嘅樹冠 footprint。
之後再同正射影像對齊（保持樹與樹之間相對間距），因為正射影像有輕微幾何變形，
要人手微調；偶爾造成極輕微重疊。

**Appendix D 的實務提醒**（想複製呢個 pipeline 就要睇）：
正射影像對 TLS 嘅對齊，關鍵係 **用 GNSS 量測嘅 Ground Control Point (GCP) 精度**。

## 3. 評估設定

- 兩個模型：**DeepForest (RetinaNet)** vs **Detectree2 (Mask R-CNN)**
- 為了可比性，**只比較 bounding box 表現**（Detectree2 本身可出非矩形）
- **Gridsearch**：tile size × NMS IoU 聯合搜尋，tile 之間 relative overlap = 0.5
- 指標：**AP50、AP75**，以及最佳 AP50 那點嘅 best F1（在 confidence threshold 上取最大）
- **Canopy tree 定義：plot 內最大高度嘅 ≥ 75%**
- 預測要算 canopy 指標時，要先幫預測**指派高度**：
  當預測面積 > 50% 被某個 GT label 覆蓋就指派該樹高度
  （因為 GT 本身設計成幾乎唔重疊，所以一個預測只可能指派到一個高度）；
  指派唔到高度嘅預測**直接丟棄**，唔計入 canopy 指標。

## 4. 結果（殘酷）

| 場地 | 模型 | AP50 (all / canopy) | AP75 (all / canopy) | F1 (all / canopy) |
|---|---|---|---|---|
| Almorox（**人工**） | DeepForest | **0.385** / – | 0.036 / – | 0.523 / – |
| Almorox（**人工**） | Detectree2 | **0.670** / – | 0.375 / – | 0.674 / – |
| Alto Tajo（**TLS**） | DeepForest | 0.050 / **0.161** | 0.002 / 0.005 | 0.196 / 0.361 |
| Alto Tajo（**TLS**） | Detectree2 | 0.094 / **0.365** | 0.011 / 0.051 | 0.227 / – |
| Joensuu（**TLS**, boreal） | DeepForest | 0.142 / 0.257 | 0.011 / – | – |
| Joensuu（**TLS**, boreal） | Detectree2 | 0.105 / 0.308 | 0.004 / – | – |

三個結論：
1. **人工標註會虛高**：同生態系下 AP50 0.670 → 0.094
2. **限制在 canopy**（只計主冠層樹）差距大幅收窄（0.094 → 0.365），但依然遠低於人工標註
3. **AP75 幾乎全滅**（max 0.051）→ 唔係「detect 唔到」，係**定位精度根本唔夠**；
   論文指呢個同 aerial LiDAR 研究觀察一致，係密閉冠層嘅本質限制

## 5. 對 Arbotic（HK urban LiDAR）嘅直接應用

### (a) 立即可抄嘅嘢
- **驗證策略**：LiDAR/TLS 標註要設為 **validation-only**，明文禁止用嚟調 hyperparameter
  （MillionTrees 都係咁寫）。呢個係學術上站得住嘅做法。
- **Algorithm 1 直接可用**：PDAL 逐棵樹建 0.02 m DEM → GDAL VRT 疊加 →
  「每 pixel 最高樹」band → polygonise。呢個就係一個**由點雲出樹冠多邊形**嘅 pipeline。
- **報 AP50 + AP75**：只報 AP50 會掩蓋定位問題。HK 街樹／公園樹比密閉林冠孤立，
  AP75 應該會明顯高過 0.05 —— 呢個係你個系統可以量化嘅優勢。
- **分層評估**：論文用「≥75% 最大高度」；你哋可以改用 DBH 級距 / 冠幅級距 /
  TRAQ 風險等級分層，出一組更貼近樹木管理嘅指標。

### (b) 你哋同論文嘅關鍵差異（要小心）
- 論文係 **密閉林冠 + 由上而下 UAV RGB**；你哋係 **城市街樹 + 點雲為主**。
  城市環境反而**更有利**：樹冠重疊少、有 GCP、結構明確。
- 論文嘅「LiDAR」係 **TLS（地面掃描）**，你哋可能係 ALS/MLS → 遮擋模式唔同，
  TLS 對樹幹/DBH 友善，ALS 對冠層友善。做驗證時要講清楚邊種。
- 論文兩個模型都**唔食 LiDAR**，只食 RGB。你哋如果係 LiDAR-first，
  要另外設計 RGB 分支做對照，否則無可比性。

### (c) 反向機會（最值錢）
論文正正指出「用自己 pipeline 產生嘅 label 去評估自己 pipeline」係循環論證。
你哋有 **LiDAR/TLS + 真實樹木管理記錄（DBH、樹種、TRAQ）**，
係全球都缺嘅 urban ground truth → 可以：
1. 用 Algorithm 1 造 HK 樹冠多邊形 → 貢獻入 MillionTrees（UrbanLondon 已有先例）
2. 出一篇「urban street tree 上 DeepForest/Detectree2 嘅 TLS 驗證」——
   直接填補 Allen et al. 只做 forest（非 urban）嘅空白

## 6. 延伸閱讀
- arXiv: https://arxiv.org/abs/2503.14273
- MillionTrees TLS 來源（Allen 2025 / Frey 2026 為 validation-only）:
  https://milliontrees.idtrees.org/en/latest/datasets.html
- Detectree2: https://github.com/PatBall1/detectree2
- PDAL: https://pdal.io ｜ GDAL VRT: https://gdal.org
