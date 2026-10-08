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
- 派錢用 Payoneer（190+ 國家）；**美國**稅務居民／喺美國境內做過工才要交 tax form（見 §12）
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


## 10. Live coding？用 AI 得唔得？（實話實說）

### (a) 要唔要 live coding？

- **GSoC 官方冇 live coding 環節**。官方 mentor guide 講 selection 時只提
  *"Interact with the GSoC contributors during the proposal period... Finding out that a
  GSoC contributor will not interact, or cannot interact well, is absolutely crucial."*
  —— 全文冇 interview、冇 live coding、冇筆試。
- **但 NumFOCUS/weecology 2026 實際有多次視像對談**（見 §9d）。Saqlain 講明對談內容係
  *"PRs and prototypes to debugging, learning new computer vision frameworks in a very
  short time, and presenting our findings"* —— 即係**考你解釋 + 快速學 + debug**，
  唔係考你背演算法。
- **實務結論：預備「live debugging + 講解」，唔係預備「白板寫 algorithm」。**
  真正會發生嘅情境：
  1. 開 screen share，叫你講你個 PR 改咗咩、點解咁改
  2. 即場跑 `pytest`，出 error，睇你點定位
  3. 叫你睇一段陌生 code 然後講出你理解幾多
  4. 叫你 demo prototype，然後解釋你嘅設計取捨
- 所以「手寫 code 慢」唔係致命傷；**「講唔出自己 code 點解咁寫」才係致命傷**。

### (b) 用 AI 得唔得？—— 可以，但有硬規矩

**Google 官方 2026 指引（原文要點）：**
- 每個 org 政策唔同：*"Some organizations do not allow any use of AI tooling, including
  in the writing of proposals. Others won't allow any code generated from LLMs into
  their code base."* → 一定要先讀 org 自己嘅指引
- **最重要一條**：*"Always Validate and Fully Understand the Code... The human
  contributor retains 100% responsibility for the work, which necessitates complete
  understanding and verification. If you don't understand it or are not sure, don't use
  it until you are able to figure it out."*
- **AI 用喺邊**：研究/學習/理解新領域（推薦）；boilerplate、imports、寫 test、debug
  （推薦，但 *"User needs to define the test scope"*）
- **AI 唔應該用喺**：core logic、最重要嘅部分
- Mentor 最擔心：妨礙學習、盲信唔驗證、code quality 差增加 maintainer 工作量、
  licensing/copyright、AI 喺新問題上失效（*"AI is terrible at writing anything other
  than simple code in a limited context"*）

**NumFOCUS 規定**：*"Contributors must clearly cite any AI tools used. Make sure you
understand each part of the code."* + 要讀 Google 嘅 AI tooling 指引。

**weecology 規定**（GSoC wiki 原文）：*"We have intentionally selected projects that
require creativity, thought and problem-solving. This is not the kind of project that a
student can drop into Cursor/Claude and get a solution. You are welcome to use AI agents
to help build concepts and speed up development, but blindly following them will yield
very little success."*

### (c) 真實後果：DeepForest repo 有標籤 `AI_Generated_NA`

repo 有個 label 叫 **`AI_Generated_NA`**，2026 年有 7 個項目被標上（5 個 PR）。
抽查結果（全部由同一 contributor 提交）：

| PR | 標題 | 結果 |
|---|---|---|
| #1306 | Security hardening: replace unsafe `eval()` with AST evaluator | **closed, 冇 merge** |
| #1335 | Fix data dimension swapping and axis errors | **closed, 冇 merge** |
| #1336 | Fix: syntax errors in utility function validations | **closed, 冇 merge** |
| #1337 | Fix dictionary key type mismatch causing KeyError | **closed, 冇 merge** |

## 11. Toronto 居民 stipend + 其他可以報嘅 org

### (a) 住 Toronto 拎幾多？

GSoC stipend = **project size × 居住國 PPP**，而 *"Your location is determined by the
country where you are residing during the GSoC coding period."*（即 5–8 月實際住邊）

| Size | **加拿大 (USD)** | 香港 (USD) | 差額 |
|---|---|---|---|
| Small ~90 h | $1,500 | $1,200 | +$300 |
| Medium ~175 h | $3,000 | $2,400 | +$600 |
| **Large ~350 h** | **$6,000** | $4,800 | **+$1,200** |

- **加拿大係最高一級之一**（同美國、英國、瑞士、澳洲、紐西蘭、以色列同級；
  全球最高 $6,600 = 瑞士／澳洲／紐西蘭／以色列）
- DeepForest 條 project 係 Large → **Toronto 居民 = US$6,000（≈ HK$46,800）**
- ⚠️ **唯一槓桿就係居住地**。冇 needs-based 加成、冇「多啲補貼」申請。
  350 h 已經係最大 size，冇得再升。
- ⚠️ 如果你 coding period 返香港住 → 變返 HK rate（$4,800）
- ⚠️ **稅務：Google 唔會預扣稅，加拿大居民亦唔需要填美國稅表**（官方講明只有美國稅務
  居民／resident alien，或者喺美國境內做過一日工嘅 contributor 才要交 tax form）
  → 你會**全額收到**，稅係自己報 T1 時計。詳見 §12。
- ⚠️ Quebec 因法規問題被 Payoneer 排除（Toronto 冇影響）
- ⚠️ 稅務：stipend **唔係免稅**，只係 Google 唔預扣；加拿大要自己報 T1 —— 見 §12

### (b) 其他可以報嘅 org（2026 有參加、同 LiDAR／遙感／空間數據相關）

**第一梯隊（最貼你 profile）**

| Org | 點解適合 | 2026 狀況 |
|---|---|---|
| **NumFOCUS** → sub-org **GRASS GIS** | `r.in.pdal` / `v.in.lidar` / `r.in.lidar`，LiDAR 光達處理最強嘅開源 GIS；2026 已由 OSGeo 轉去 **NumFOCUS** 做 fiscal sponsor | 2026 收 2 人（r.proj 平行化、GUI 時空數據） |
| **NumFOCUS** → **Data Retriever / weecology** | 即係 DeepForest（見前文） | 2026 收 2 人 |
| **NumFOCUS** → **PySAL** | Python 空間分析（libpysal, spopt, esda）—— 你個 spatial join / 空間統計部分 | 2026 有參加 |
| **OSGeo** | 地理空間 umbrella：**QGIS（有點雲支援，PDAL backend）**、pgRouting、istSOS、ZOO-Project | 2026 收 2 人（都係 GRASS 出身嘅 mentor 圈子） |
| **52°North Spatial Information Research GmbH** | **德國研究機構**（唔止基金會），做 sensor web、web geoprocessing、**Earth observation** | 2026 有參加 |
| **PEcAn Project** | 生態系模型（Boston University Dietze lab 出身）—— 學術 lab、做遙感 + 碳循環數據融合 | 2026 有參加 |

**第二梯隊（技術相鄰）**

| Org | 點解適合 |
|---|---|
| **CGAL Project** | 3D 計算幾何：point set processing、surface reconstruction、shape detection —— 點雲處理核心算法 |
| **Kornia** | 2026 起主力做 **kornia-rs**：Rust 3D CV + spatial AI，edge／robotics 部署（DeepForest 依賴 Kornia！） |
| **Open Robotics (ROS)** | 點雲感知、PCL 整合、octomap |
| **OpenVINO Toolkit** | 模型 edge 部署（如果你要將 DeepForest 落去現場設備） |
| **MLLAM** | AI 天氣預報（neural-lam），DMI／MET Norway 等國立氣象機構 |
| **IOOS** | NOAA 海洋觀測（遙感 + 開放數據） |
| **ML4SCI** | 機器學習 + 科學（有 Earth observation 方向） |
| **ArduPilot / JdeRobot** | 無人機／機器人（如果你做 UAV LiDAR） |
| **OpenStreetMap** | 開放地理數據 |

**2026 年總共有 183 個 org**；完整名單用 API 拎得到：
`https://summerofcode.withgoogle.com/api/program/2026/organizations/`
（2027 版 2 月 19 日左右出）

### (c) 實戰建議

1. **你可以交最多 3 份 proposal**，而且可以交去唔同 org → 唔好只賭 DeepForest。
2. **同一 umbrella 有優勢**：NumFOCUS 一份申請流程，可以同時考慮
   Data Retriever / GRASS / PySAL。GRASS 個 AI 政策寫得好直白：
   *"AI-generated 'slop' ... is easy to spot and will hurt your application.
   We evaluate applications primarily on GitHub contributions and communication with
   the GRASS community, not just proposal polish."*
   → 再次印證：**PR + 溝通 > proposal 文筆**
3. **GRASS 係 LiDAR 角度最好嘅 second choice**：佢有 `r.in.pdal`、
   point cloud 資料類型、CHM 相關工具，你嘅 LiDAR 經驗直接對口。
4. **52°North 同 PEcAn 係「真 lab」**（研究機構／大學 lab），
   如果你想要學術路線而唔止係開源工程，值得優先睇。
5. **注意組織穩定性**：2026 年 OSGeo 嘅 ideas page 只剩 4 個子項目（GRASS 已搬去
   NumFOCUS），所以每年 2 月一定要重新 check 當年名單。
6. 用同一個方法驗證任何新 org：GitHub 活躍度、mentor 回應速度、
   ideas page 具體程度、有冇 AI policy。


發生咗咩事：
- maintainer（henrykironde）review 之後 request changes：
  *"Please provide script to reproduce the bug"*
- 2026 GSoC contributor（vickysharma-prog）公開指出：
  三個 PR 嘅 description 幾乎一模一樣，問係咪刻意拆分 → 作者自己承認
  *"the descriptions accidentally matching across those 3 PRs was just a mistake on my
  end"*
- 結果：全部 closed，冇一個 merge

**教訓（直接影響你）：**
1. **每個 PR 要有 reproduction script / test** —— 呢個係 maintainer 第一個要求
2. **唔可以批量生成似樣嘅 PR** —— description 重複會即刻被認出
3. **AI 生嘅 code 會被標籤** —— 唔申報係不誠實，申報係正常但會更嚴格被檢視
4. 相反：**有 reproduction + test + 你自己講得清嘅 PR 會 merge** ——
   2026 兩位 GSoC 得主就係靠一串 merged PR 入選

### (d) 「手寫 code 好渣」點算？—— 5 個具體對策

1. **記住 NumFOCUS 原句**（可以喺 proposal 直接引用）：
   *"We value creativity, intelligence and enthusiasm above specific knowledge of the
   libraries or algorithms we use. We think that an interested and motivated contributor
   who is willing to learn is more valuable than anything else."*
   → 佢哋評分表 6 個 5 分項**冇一項係寫 code 能力**
2. **練「講」多過練「寫」**：每次開 PR 前，用 3–5 分鐘錄音自己講一次
   「我改咗咩、點解、驗證咗咩、試過咩失敗」。呢個直接對應 interview。
3. **練 live debugging**：故意喺自己環境整壞一個 test，然後開 screen share 錄住自己
   點用 traceback → 睇 code → 加 print → 修好。練 5 次就會自然。
4. **AI 用喺正確位置**：用 AI 讀懂陌生 code、寫 test scaffold、解釋 error；
   **core logic 自己寫**，因為你一定會被問到。
5. **用你嘅 domain 做護城河**：你識 LiDAR、點雲、TRAQ、HK 樹木管理 —— 呢啲係
   mentor 唔識嘅。將你嘅 proposal 定位成「我提供 domain 判斷，engineering 我邊做邊學」。

### (e) Interview 準備 checklist

- [ ] 準備好講 3 個自己嘅 PR（背景 → 改動 → 點驗證 → 有咩 trade-off）
- [ ] 準備一個 5 分鐘 prototype demo（可以係 DeepForest-HK 個 pipeline）
- [ ] 練一次「共用 screen 跑 pytest 然後 debug」
- [ ] 準備 3 條你想問 mentor 嘅技術問題（顯示你真係讀過 code）
- [ ] 準備好講明你嘅時間承諾（350 h / 12 週，每週幾多個鐘）
- [ ] 準備 AI 使用聲明：你用咗咩、用喺邊、你點驗證

- 2025 提過但未做：**active learning module**、airborne wildlife benchmark

## 12. 稅務實算：Toronto US$6,000 實際袋幾多？

匯率參考：**USD/CAD = 1.4243**（2026-10-08）→ US$6,000 = **CAD $8,546**

### (a) Google 會唔會預扣稅？→ **唔會**

Google 官方 Tax Form Instructions 原文：
- *"All U.S. residents (or resident aliens) or any contributor coding in the U.S. for any
  length of time during the GSoC program will need to complete a tax form."*
- *"Only contributors who complete a W-9 and are U.S. residents or resident aliens will
  receive a 1099-NEC."*

→ **全程喺加拿大做嘢嘅話：唔需要填美國稅表、冇預扣、冇 1099**。
你會全額收到 US$6,000，稅係自己報加拿大 T1 時計。
（Google 亦明講：*"Google can not provide you with tax advice"*，有事要搵會計師。）

### (b) 加拿大 2026 稅率（Ontario）

- 聯邦最低稅率 **14%**（首 $58,523）；聯邦 BPA **$16,452**
  → 即係約 **$16,452 應稅收入以下，聯邦稅 = $0**
- Ontario 最低 5.05%（首 $53,891）；Ontario BPA 約 $12,990
- **合併邊際稅率（2026，Ontario，其他收入）**：

| 應稅收入 | 合併邊際率 |
|---|---|
| 首 $53,891 | 19.05% |
| $53,891 – $58,523 | 23.15% |
| **$58,523 – $94,907** | **29.65%** |
| $94,907 – $107,785 | 31.48% |
| $117,045 – $150,000 | 43.41% |

### (c) 兩個情境

**情境 A：GSoC 係你 2026 年唯一收入（例如全職學生）**
- 應稅收入 CAD $8,546
- 聯邦：稅 14% × 8,546 = $1,196，但 BPA 抵免 14% × 16,452 = $2,303 → **$0**
- Ontario：稅 5.05% × 8,546 = $432，BPA 抵免 5.05% × 12,990 = $656 → **$0**
- 亦唔使交 CPP / EI（唔係僱傭收入）
- **→ 稅 ≈ CAD $0，實收 ≈ CAD $8,546（≈ US$6,000）**

**情境 B：你本身有全職工作（例如 CAD $60,000 年薪）**
- GSoC 嗰 CAD $8,546 疊上去，行 **29.65%** 邊際率
- 稅 = 29.65% × 8,546 ≈ **CAD $2,534**
- **→ 實收 ≈ CAD $6,012 ≈ US$4,220** ← **你估嘅 $4.5k 就係呢個情境**
- 如果稅局當佢係**自僱／生意收入**（T2125），仲要交 CPP：
  11.9% × (8,546 − 3,500 基本豁免) ≈ **$600** → 實收再跌到 ≈ US$3,800

### (d) 一句總結

> **你嘅 $4.5k 估算，只有在你本身有其他收入時才成立。**
> 如果 GSoC 係你 2026 年唯一收入，實際稅 ≈ **$0**，實收 ≈ **US$6,000**。
> 差別可以係 **US$1,800**，所以值得搞清楚。

⚠️ 我唔係會計師。以上係按 CRA 2026 稅率／BPA 同 Ontario 稅率計算嘅**估算**，
未計其他抵免、退稅（GST/HST credit）同你個人狀況。正式申報前搵會計師確認，
特別係要問清楚「GSoC stipend 應該報 line 13000 other income，
定係 T2125 自僱收入，定係有其他處理」。

### (e) 值唔值？（誠實計法）

- US$6,000 ÷ 350 小時 = **US$17.1/小時**（稅前）
- 折合 **CAD $24.4/小時** —— 高過 Ontario 最低工資（約 CAD $17.6），
  但低過一般 junior developer（CAD $30–40/h）
- **所以唔應該當佢係一份工嚟計錢。** 真正價值係：
  1. 一段有學術 mentor 嘅研究經歷（Ben Weinstein / Josh Veitch-Michaelis 級數）
  2. 可以掛上 CV 嘅 *"Google Summer of Code contributor, NumFOCUS"*
  3. 你嘅 HK 樹冠數據有機會入 MillionTrees benchmark → 變成可引用成果
  4. 一條通往論文／學術合作嘅路（Allen et al. 2025 條路線）
  5. 對 Arbotic 嚟講係「團隊做過 GSoC」嘅技術背書
- **如果只係為錢：唔值。如果係為咗上面 5 樣：超值。**

## 13. 「我住邊都可以」點算？→ 申報 **加拿大**，而且要住喺 Toronto

### (a) 規則原文

- GSoC：*"Your location is determined by the country where you are **residing during the
  GSoC coding period**"*；並且 *"Accepted participants must provide tax forms and
  **proof of residency**"*
- 資格要求：*"Eligible to work in your country of residence"*
- 官方 student guide 警告：*"if you are in a country on a student visa or another type of
  visa you could have restrictions on the number of hours you can participate"*

→ 即係唔可以「唔報」或者「報個高 PPP 嘅國家」。要揀**一個**國家，而且要證明得到。

### (b) 結論：揀加拿大。三個原因

1. **加拿大本身就係接近最高 tier**（Large = US$6,000，全球最高 $6,600）
   → 報加拿大**冇蝕底**，唔需要冒險報其他國家
2. **最證明得到**：家人喺 Toronto、屋企地址、安省證件、銀行戶口、報稅紀錄
3. **時區贏晒**：Toronto 同 University of Florida 同一個 Eastern Time
   → 同 mentor 開會唔需要捱夜，而「溝通」喺 NumFOCUS 評分表佔 **5 分 × 2 項**
   （如果喺 HK 就要 12–13 鐘時差，好蝕）

### (c) 稅務上：你大概率係 CRA 講嘅「factual resident」

CRA 原文：
> *"You are a factual resident of Canada for income tax purposes if you keep
> **significant residential ties** in Canada while living or travelling outside the country."*

CRA 列明嘅情況包括：*working temporarily outside Canada*、*attending school in another
country*、*vacationing outside Canada*、*spending part of the year in the U.S.*

Factual resident 嘅後果：
- **要報全世界收入**（inside and outside Canada）
- 繼續享有聯邦／省抵免
- 按你**保持住宅連繫嘅省份**（即 Ontario）繳省稅
- 幾時會斷：*decide to stay permanently*、*sell your house in Canada*、
  *move your spouse/common-law partner and dependent children with you*

→ **家人喺 Toronto + 你唔係永久搬走 = 大概率仍然係加拿大稅務居民**，
所以 GSoC stipend 要報 T1（但見 §12：如果係你唯一收入，稅 ≈ $0）。

### (d) 要準備嘅「proof of residency」

Payoneer 開戶 + Google 要驗證身份／地址，實務上要有：
- [ ] 政府證件（加拿大護照／PR 卡／安省車牌／OHIP 卡）
- [ ] 地址證明（銀行月結單、水電費、租約、家人住址）
- [ ] 加拿大銀行戶口（收 Payoneer 提款）
- [ ] （如有）CRA Notice of Assessment —— 呢個係最硬嘅稅務居民證明

**一致性最重要**：GSoC dashboard、Payoneer、稅表三個地方要寫同一個國家／地址。

### (e) 三個實務提醒

1. **5–8 月真係住喺 Toronto** —— 一次過解決稅務居民身份、proof of residency、
   時區、網絡穩定性四個問題。GSoC guide 明講：
   *"if you are not sure you will have good Internet connectivity continuously over the
   summer, GSoC is not for you."*
2. **OHIP 有居住要求**（一般 12 個月內要喺安省實際逗留 153 日）。
   如果你長期四圍走，健康卡資格要自己 check 清楚。
3. **「Eligible to work in your country of residence」**：如果你係加拿大公民／PR 就冇問題；
   如果你只係訪客身份，呢一項要小心。

### (f) 唔好做嘅事

- ❌ 報瑞士／澳洲／紐西蘭／以色列（$6,600，多 $600）—— 證明唔到，
  而且 Google 明文要 proof of residency，被查到係失去 stipend 級別嘅風險
- ❌ 一時報加拿大一時報香港 —— 影響信譽，而 mentor 係會睇你點溝通
- ❌ 申報期間搬國家 —— 會令 Payoneer 驗證同稅務變複雜

**一句總結：加拿大 = 最高一級（91% of max）+ 最好證明 + 同 mentor 同時區。
呢個係唯一合理答案，唔需要諗其他。**
