# 報 weecology / DeepForest GSoC 2027 — 作戰手冊

研究日期：2026-10-08

## 0. 最重要嘅三件事（先睇呢度）

1. **入口唔係 DeepForest repo，係 retriever wiki**。
   weecology 嘅 GSoC 專案 ideas page 放喺：
   `https://github.com/weecology/retriever/wiki/GSoC-<年份>-Project-Ideas`
   （2026 年版最後編輯 2026-03-29，共 12 個 revision）。
   NumFOCUS 嘅 sub-org 名單上，weecology 係以 **Data Retriever** 出現；
   2026 兩位學生（Vicky Sharma、Muhammad Saqlain）喺 NumFOCUS blogs 頁都係掛 Data Retriever 名下。
2. **唔好發 intro email**。NumFOCUS 明文寫：
   *"GSoC is hands-on: rather than sending introductory emails, begin by triaging
   or solving issues to get to know the team."*
   所有問題要用 **project repo 嘅 issue** 問。
3. **要有 PR**。NumFOCUS 評分表裏面「有冇喺現有 code 開 PR」同「有冇同 mentor 溝通」
   各佔 5 分（最高分項），而且係最容易拎嘅分。

## 1. GSoC 官方時間線（2026 實際日期 → 2027 推算）

| 階段 | 2026 日期 | 2027 預期 |
|---|---|---|
| Mentoring org 申請 | 1/19 – 2/3 18:00 UTC | 1 月中 – 2 月初 |
| 公佈 accepted orgs | 2/19 | 2 月中／下旬 |
| 同 org 討論 ideas | 2/19 – 3/15 | 2 月中 – 3 月中 |
| **Contributor 申請期** | **3/16 – 3/31 18:00 UTC**（15 日） | **3 月中 – 3 月底** |
| Org 提交 proposal 排名 | 4/21 | 4 月中／下旬 |
| 公佈 accepted projects | 4/30 | 4 月尾 |
| Community bonding | 5/1 – 5/24（3 週） | 5 月 |
| **Coding 開始** | 5/25 | 5 月尾／6 月頭 |
| Midterm evaluation | 7/6 – 7/10 | 7 月 |
| 最後提交 | 8/17 – 8/24（standard 12 週） | 8 月尾 |
| Extended timeline 最後提交 | 11/2 | 11 月 |

注意：DeepForest 2026 個 project 係 **Intermediate, long = 350 小時**，
即係要當一份接近全職嘅 summer job，proposal 裏面一定要寫明你有時間。

⚠️ 風險：2026 年 DeepForest 原本 2025-12 話「唔預期有 project」，之後先改口。
所以要 **Plan B**：同一 umbrella 下其他 NumFOCUS sub-org，或 weecology 嘅
Data Retriever 本體（retriever 都係 weecology 出）。

## 2. NumFOCUS 官方評分表（照住呢個 checklist 寫 proposal）

| 分 | 項目 |
|---|---|
| 5 | 有冇同 org 嘅 mentor 溝通 |
| 5 | 有冇同 community 溝通 |
| 5 | 有冇引用自己寫過嘅 project（附 repo 連結或 code） |
| 5 | 有冇提供多種聯絡方式（email / phone / chat…） |
| 5 | **有冇喺現有 codebase 開 PR** |
| 5 | **有冇持續溝通直到 accepted contributors 公佈** |
| 3 | 有冇初步 project plan（before / during / after GSoC） |
| 3 | 有冇講明申請邊個 project、點解你覺得自己做得完 |
| 3 | 有冇講明你有時間（並列出其他 commitments） |
| 1 | 有冇一個 link 連去你所有申請文件（GitHub / Dropbox） |
| 0 | 誠實（"only universal Karma points"） |


## 3. 2026 年官方 project ideas（2027 嘅最佳預測）

兩個都係 `Source Code: DeepForest`、Difficulty: Intermediate long (350 h)、
Skills: Deep learning / Git-GitHub / ML / Software testing / Python + package deployment、
Mentors: **@bw4sz（Ben）、@jveitchmichaelis（Josh）、@henrysenyondo（Henry）、@ethanwhite（Ethan）**。

**Proposal 1**：Recovering computer vision annotations from historical airborne imagery
for biodiversity monitoring（→ Vicky Sharma 中）
- 2010 年 Deepwater Horizon 漏油後 Gulf of Mexico 航空鳥類調查，只存低解析 screenshot
- 用 CV + AI + forensic data analysis 反推鳥類點位；提及 **Segment Anything 3**
- 產出：recovered dataset、CV model、**PR 入 DeepForest**、blog

**Proposal 2**：Recovering historical image data using automated ortho-registration
and image-matching（→ Muhammad Saqlain 中）
- USFS Aerial Detection Survey 森林健康 polygon 同 NAIP 影像有 50–500 m 偏差 + 1–3 年時差
- 用 template matching + orthorectification 對齊；Oregon 30 cm NAIP；
  train 一個預測 affine shift 嘅模型；用「隨機位移再學對齊」做弱增強
- 產出：DeepForest ↔ NAIP map server 連接、weak dataset、forest health model、blog

歷史 pattern（2024/2025 嘅 ideas）：bird nest detection、bird detection/classification、
modernizing tree detection、unique image detection、**active learning module**、
airborne wildlife benchmark、LandingAI vision agent。
→ 規律：**數據復原 / benchmark / 工具整合** 為主，好少直接叫你換 architecture。

## 4. 2027 可能出現嘅方向（由 open issues 推斷）

- **#758 Polygon model support**（milestone 2.2）：將 torchvision **mask-rcnn** 加入
  `models/`、寫 polygon dataset class、令 `preprocess.split_raster` /
  visualization 支援 polygon、出一份 train polygon model 嘅 notebook
- **#1038 SAM2 for object tracking after deepforest prediction**（label: Google Summer of Code，open）：
  做 `deepforest[sam2]`，用 DeepForest 嘅 box/point 做 prompt 去 segment + track
- **#460 Integrate Segment Anything for post-model bounding box to polygons**

## 5. 你嘅 6 個月準備計劃（2026-10 → 2027-03）

### Phase 0 — 2026 年 10–12 月：技術底 + 第一個 PR
- [ ] 環境升到 DeepForest 2.1（Python ≥3.10、torch ≥2.2、numpy ≥2.0）
- [ ] 用 `pip install -e .` 裝 dev 版，跑一次 `pytest`（CONTRIBUTING.md 有寫）
- [ ] 讀 `src/deepforest/`：`main.py`、`model.py`、`models/`、`predict.py`、`datasets/`、`conf/`
- [ ] 揀 1–2 個 **good first issue** 開 PR（現時 open 嘅）：
  - #1274 `PadIfNeeded` augmentation 命名錯誤（會 crop 又會 pad）
  - #889 分開 predict / train 嘅 batch size config
  - #797 `predict_file` 應該收 dataframe 唔止 csv
  - #999 修「Working with deepforest data」docs
  - #404 Profile `predict_tile`
  - #1064 自動 log 幾張 train/val 圖
- [ ] 讀 code of conduct（GSoC wiki 寫明「first read」）

### Phase 1 — 2026 年 12 月–2027 年 1 月：卡位
- [ ] 定期睇 `weecology/DeepForest` issue #1251（GSoC thread）嘅更新
- [ ] 定期 check `retriever/wiki` 有冇 GSoC-2027 page
- [ ] 開一個 GitHub issue（唔係 email）自我介紹 + 問 2027 方向，附你已 merge 嘅 PR
- [ ] 開始寫 public blog（跟 Vicky / Saqlain 咁，mentor 會睇你點寫技術決策）
- [ ] 準備一個 **prototype**：2026 兩位學生都係靠「申請前已經有 prototype」勝出

### Phase 2 — 2027 年 2–3 月：Proposal
- [ ] 2/19 左右 org 公佈後即刻對應 ideas page
- [ ] 寫 proposal，逐項對住 §2 評分表自我打分
- [ ] 內容要有：目標、方法、milestones（3 週 bonding + 12 週 coding 分週）、
      deliverables（**包括一個 PR 入 DeepForest**）、風險與替代方案、
      你嘅時間承諾、聯絡方法、AI 使用聲明
- [ ] 3/31 18:00 UTC 前提交（HK 時間 = 4/1 凌晨 2 點，唔好最後一日先做）
- [ ] **提交後繼續開 PR、繼續溝通到 4/30** —— 呢項佔 5 分，2026 得主就係咁做

## 6. AI 使用規定（唔跟會死）

- NumFOCUS：contributor **必須清楚列明用過咩 AI 工具**；每個 code 部分你都要理解
- weecology wiki 原文警告：*"We have intentionally selected projects that require
  creativity, thought and problem-solving. This is not the kind of project that a
  student can drop into Cursor/Claude and get a solution... blindly following them
  will yield very little success."*
- 對策：proposal 同 PR 裏面寫清楚「我用 AI 做咗 X，但我自己決定咗 Y、驗證咗 Z」

## 7. 你嘅獨特優勢（要寫入 proposal）

1. **真實 LiDAR / 點雲 + 城市樹木管理場景（HK）** —— 全球都缺 urban tree ground truth
2. 有 **TRAQ / DBH / 樹種** 等 domain label，可以接 MillionTrees 嘅 data contribution
3. HK 正射影像 10 cm/px，同 `deepforest-tree-point`（TreeFormer）訓練解析度吻合
4. 已有 DeepForest-HK pipeline（crown detection + municipal inventory spatial join）
   —— 呢個係「referenced projects with links」嘅 5 分
5. 可以提議 **LiDAR/TLS validation** 角度（呼應 Allen et al. 2025 +
   MillionTrees validation-only TLS 來源）—— 但記住：**idea 由 org 定，唔係你定**；
   你嘅角色係將佢哋嘅 idea 用你嘅 domain 經驗做得更好

## 8. 參考
- 2026 ideas page: https://github.com/weecology/retriever/wiki/GSoC-2026-Project-Ideas
- NumFOCUS GSoC repo: https://github.com/numfocus/gsoc
- Contributor guide: https://github.com/numfocus/gsoc/blob/master/CONTRIBUTING-contributors.md
- GSoC timeline: https://developers.google.com/open-source/gsoc/timeline
- DeepForest CONTRIBUTING: https://github.com/weecology/DeepForest/blob/main/CONTRIBUTING.md
- 2026 得主心得：https://vickysharma.hashnode.dev/gsoc-2026-weecology
- 2026 得主心得 2: https://musaqlain.dev/blog/gsoc-2026-community-bonding/


## 9. 錢、時間、面試、CV（FAQ）

### (a) 有幾錢？GSoC 2026 stipend（PPP 調整，按 project size + 居住國）

| Size | 工時 | **香港 (USD)** | 全球範圍 |
|---|---|---|---|
| Small | ~90 h | **$1,200** | $750 – $1,650 |
| Medium | ~175 h | **$2,400** | $1,500 – $3,300 |
| **Large** | **~350 h** | **$4,800** | $3,000 – $6,600 |

- DeepForest 2026 條 project = **Large (350 h)** → 香港居民 **US$4,800**（≈ HK$37,400）
- 計法：base $6,000（large）再按國家 PPP 調整（HK 係 0.8 倍）
- **分兩期**：第一期 **45%**（midterm evaluation 通過後，約 7/11）、
  第二期 **55%**（final evaluation 通過後，約 9/1）
- **付款基於通過 evaluation，唔係基於 code 有冇被採用**
- 派錢用 Payoneer（190+ 國家）；通過後要交 **tax form + proof of residency**
- ⚠️ 唔係僱傭、唔係實習 → 唔可以寫 Google 做僱主；
  完成後正確寫法：*"Google Summer of Code 2026 contributor, NumFOCUS/weecology"*

### (b) 做幾個月？

- Community bonding **3 週**（2026: 5/1–5/24）—— 呢段已經要開始投入
- Coding period 標準 **12 週**（2026: 5/25 開始 → 8/17–8/24 最後提交）
- Google 容許 **8–22 週**；350 h 可以攤到 22 週（extended deadline 11/2）
- 350 h ÷ 12 週 ≈ **每週 ~29 小時** → 半職到接近全職
- **全程 ≈ 5 月中到 8 月尾 = 3.5 個月**
- 硬性要求（NumFOCUS 明文）：
  - **summer 完結前至少有一個 commit merge 入 development branch**，否則過唔到 final evaluation
  - 每 **2 星期寫一次 blog**
  - 每個 evaluation 前至少有一個 commit 經 mentor review

### (c) 睇唔睇 CV？

- **官方 proposal 冇 resume/CV 一欄**。NumFOCUS 個 proposal template 只有
  **"Development Experience"**（原句：*"Do you have code on github? Can you show
  previous contributions to other projects?"*）同 "Other Experiences"。
- 2015–2017 年代好多 NumFOCUS proposal 自己附 CV link → **optional 但常見**。
- 評分表 6 個 5 分項**冇一項係學歷/履歷**。
- 2026 得主 Vicky Sharma 係 **一年級本科生**（Delhi University, Mathematics），
  打贏 Masters/PhD 背景申請者；佢原話：mentors
  *"valued depth of thinking, experimentation, and problem-solving ability more than
  credentials alone."*
- **結論：你嘅 GitHub 就係你嘅 CV。** 多一個 merged PR 值錢過多一頁 CV。

### (d) 使唔使 interview？

- **官方文件冇寫 interview 程序**（NumFOCUS repo 全文搜 "interview" = 零結果）。
- **但 2026 實際係有**：
  - Muhammad Saqlain：*"I had 4 interviews with the NumFOCUS team totaling 6+ hours,
    where we discussed everything from PRs and prototypes to debugging, learning new
    computer vision frameworks in a very short time, and presenting our findings.
    All of these things were evaluated in a points-based approach by NumFOCUS senior
    administrators."* 佢仲話 mentors 喺官方 selection 前 **36 小時**先 interview 佢。
  - Vicky Sharma：提及 *"multiple discussion calls with mentors"*。
- **所以：預咗要視像傾 2–4 次，而且係技術討論，唔係 HR 面試。**
  準備方向：
  1. 能逐步講清楚你每個 PR 嘅技術決定（點解咁寫、試過咩、失敗咗咩）
  2. 帶一個 prototype demo
  3. 即場 debug / 睇新 framework 嘅能力（Saqlain 話佢哋真係考呢樣）
  4. present 你嘅發現（有數據、有圖）

### (e) 資格（唔好睇漏）

- 18 歲以上、可以喺居住地合法工作、唔住美國禁運國家
- **Student 或 "open source beginner"**、之前接受過 GSoC **唔多過一次**
- **"Beginner" 定義**：open source 經驗極少 —— 個人/課堂 project、單一機構內部 project、
  開過 **少過 10 個** issue/PR 都仍然算 beginner
  ⚠️ 如果你已經有大量 open source 貢獻紀錄，可能唔符合，要先自己評估
- 競爭：2026 年 131 個國家共 **23,371 份 proposal**，整體錄取率 **4.88%**，NumFOCUS 更低

### (f) NumFOCUS 官方 proposal template（照抄結構）

```
# Title
## Abstract          （最多 10 句，唔可以照抄 ideas page）
## Technical Details （必須寫齊 library、同 mentor 傾過嘅內容、相關 code/literature 連結）
## Schedule of Deliverables
   ### Community Bonding Period
   ### Phase 1 / Phase 2
   ### Final Week
## Development Experience  （GitHub link、之前嘅貢獻、課程/專案）
## Other Experiences
## Why this project?
## Appendix
```
- 最後要交 **PDF**，檔名開頭要加 **`[sub-org-name]`**（即 `[Data Retriever]` 或 `[weecology]`）
- 最多可交 3 份 proposal，但只可以接受 1 個

- 2025 提過但未做：**active learning module**、airborne wildlife benchmark
