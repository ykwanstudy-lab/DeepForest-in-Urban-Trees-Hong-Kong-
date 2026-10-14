# OpenCode free tier × Hermes (research note, 2026-10-14)

## TL;DR
- The Zen/Console **free tier** ("Big Pickle", "Space Bunny Free", `*-free` ids, "Muse Spark 1.3 Contributor Free", ...) is gated **server-side to the official OpenCode client**. There is **no sanctioned way** to consume it from Hermes or any other third-party harness.
- Officially supported routes only: **(1)** paid OpenCode Go / Console API key (Hermes is a *validated* Go client), **(2)** drive the user's own unmodified `opencode` binary local server (`opencode serve`), **(3)** don't use OpenCode's quota at all — BYOK another provider.
- Header spoofing is both **against the ToS** and **no longer sufficient**: a byte-faithful replay of a genuine request with genuine headers still returns 403 (#49621), so the check goes deeper than headers.

> Repo note: `sst/opencode` now redirects to **`anomalyco/opencode`** (`ANOMALY INNOVATIONS, INC.`), so old links in blog posts resolve there.

## 1. Official statement (in-repo, by a maintainer)
`@rekram1-node` [COLLABORATOR], 2026-09-18, in #49580 (repeated in #49621):

> "We've been tightening our logic to fight abuse, we did temporarily have an issue where compaction was
> impacted and marked as not from OpenCode it has been resolved. **You cannot use the free tier in other
> harnesses (this is only a limitation for the free tier nothing else).** The logic in place may not work
> properly if you have a very custom setup."

- https://github.com/anomalyco/opencode/issues/49580
- https://github.com/anomalyco/opencode/issues/49621

Hermes side, `@teknium1` [COLLABORATOR], 2026-09-21, in **NousResearch/hermes-agent#117056**:

> "the free tier is restricted server-side to the OpenCode client, so **Hermes (or any third-party client)
> cannot use it**. Paid OpenCode Zen/Go credentials work through `hermes auth add opencode-zen`.
> Closing as not something Hermes can change."

- https://github.com/NousResearch/hermes-agent/issues/117056

## 2. Live probe I ran on 2026-10-14 (`POST https://opencode.ai/zen/v1/chat/completions`, no credential, honest UA `hermes-agent/0.22.0`)
| model | result |
|---|---|
| `space-bunny-free` | **403** `{"type":"error","error":{"type":"FreeTierError","message":"OpenCode's free tier can only be used from within OpenCode. Download it at https://opencode.ai and keep building."}}` |
| `big-pickle` | **403** FreeTierError (same message) |
| `mimo-v2.5-free` | **410** ModelDeprecated -> "Use mimo-v2.6-flash-free instead" |
| `GET /zen/v1/models` | **200**, public, lists the `-free` ids (readable but useless without the client gate) |

Key finding: **`space-bunny-free` is now 403 too.** Hermes PR #133168 (2026-10-05) documented it as the
*only* keyless `-free` id answering 200 (`cost: "0"`) while 9 others 403'd. That last gap is closed as of
today's probe.

## 3. What the gate actually checks (public analysis, #49433)
- `User-Agent` must look like `opencode/<recent semver>` (case-insensitive; `opencode/latest`, bare
  `opencode`, `curl/*`, `pi (...)` all 403).
- `x-opencode-session` must be `ses_` + 12 hex + 14 base62 (an `Identifier.create` id), otherwise 403 --
  a 400 `MissingSessionID` variant exists ("OpenCode's free tier can only be used in OpenCode").
- https://github.com/anomalyco/opencode/issues/49433

Even with both correct, #49621's elimination matrix shows node/.NET/Python/Bun/raw-TLS clients **and a
byte-faithful replay** of a captured genuine request (same header order, UA, `ses_`, `msg_`,
`prompt_cache_key`, tools, HTTP/1.1 *and* HTTP/2) all get **403**, while the genuine binary and a local
`opencode serve` call get **200**. => additional fingerprinting beyond headers; spoofing is not a path.
https://github.com/anomalyco/opencode/issues/49621

## 4. Terms of Use (effective 2026-08-15) -- https://opencode.ai/legal/terms-of-service
> "You will only use the Services for **your own internal use**, and **not on behalf of or for the benefit
> of any third party** ..."

Restricted conduct includes accounts used "to **circumvent usage limits, access restrictions**, billing
obligations, promotions, suspensions, **or any other restriction or policy applicable to the Services**"
and anything that "**automatically or programmatically extracts data or Output**".

Note: the ToS page itself does **not** contain the words "must be used from within OpenCode", and the
demand letter posted at hermes-agent#116494 quoted "Terms of Use **[add section/link]**" and was replied
to with "Reported for impersonation." -- treat that specific notice as unverified; the maintainer
statements in section 1 are first-party.

## 5. Docs wording
- Zen/Console "Free models" section, per model: *"... is available on **OpenCode** for a limited time"* /
  *"is a stealth model that's **free on OpenCode** for a limited time"*
  -> https://opencode.ai/docs/zen/ (section `Free models`)
- The free-tier client restriction is **not documented** anywhere; users filed #49858 ("Looks like the doc
  is now a lie") and #52917 ("docs: misleading free-tier auth error has no documented fix").
- By contrast, **Go** docs *do* document third-party use, and list **Hermes** under "Validated clients":
  *"Hermes -- Builds containing PR #101864 send the header on main and auxiliary OpenCode requests. The fix
  was merged after v0.21.0"* -> https://opencode.ai/docs/go

## 6. GitHub reports (verified via `gh`)
OpenCode repo -- 105+ results for `"can only be used from within OpenCode"`, 20 for `FreeTierError`:
- **#49144 [open]** Policy on third-party open-source apps using Zen free-tier models -- answered by
  neither side; contains a **Hermes user (@edge513) asking exactly this question** and a follow-up saying
  the plan is to bundle the unmodified official binary instead of spoofing headers.
- #49433 [closed] root-cause thread (UA + session gate)
- #49580 [closed] MonoCode third-party frontend -> maintainer statement
- #49621 [closed] "Free-tier Zen 403 for all third-party stacks despite valid session + key"
- #49858 [open] "Looks like the doc is now a lie" (Pi agent)
- #52917 [open] misleading error; no documented fix
- #49207 [closed] "may a third-party desktop app launch the user's own `opencode serve` ...?" -> auto-closed
  by the issue bot for template reasons; **no written yes/no exists**
- #52178 [open] CORS headers only on `/zen/v1/models`; every inference endpoint 404s on preflight
- Version-gate variants: #50581, #50451, #55430 ("OpenCode 1.18.0 or newer is required", 426 on betas)
- False positives hitting *legit* OpenCode users (agents with denied tools): #51315, #51241, #50627,
  #50806, #49723
- Fresh wave on 2026-10-14: #55426, #55428, #55431, #55432, #55435 (official Desktop 2.0.29 users)

Hermes repo:
- #117056 [closed] maintainer: not something Hermes can change
- #110014 [open] built-in keyless `opencode-free` provider: all `-free` models rate-limited / 401
- #116494 [open] removal demand for the keyless `opencode-free` tier (authenticity disputed in-thread)
- #133168 [open PR] restore `space-bunny-free` as keyless route -- **now stale per section 2 probe**
- #136413 [open] Go-served `-free` slugs (`step-5-preview-free`, `longcat-2.5-preview-free`) return 200 on
  `/zen/go/v1` **with a Go key** -- the one way "free"-named models still work from Hermes today
- #101864 [merged] every Hermes OpenCode request carries `x-opencode-session` (Go validated-client fix)

Hermes removed its keyless `opencode-free` provider in `998f614c7` when the relay began 403ing anonymous
traffic; `opencode-zen` is now credential-mandatory.

## 7. Options, cheapest-legit first
1. **Drive the user's own genuine binary** -- `opencode serve` + your own UI over the documented local
   HTTP/SSE API (the T3 Code / MonoCode / "Lift" pattern). Technically works (local `opencode serve` = 200
   in #49621) because the genuine process issues the request. Policy never answered in writing (#49207
   auto-closed, #49144 silent) => not blessed, could break later; never modify the binary or hold the
   credential, and keep tool execution denied.
2. **Pay for Go ($10/mo) or Console** and use it from Hermes -- the supported route; `hermes auth add
   opencode-zen`. Go even serves some `-free`-suffixed models (#136413).
3. **Skip OpenCode for free quota** -- BYOK any other provider into Hermes (e.g. OpenRouter `:free` ids,
   per hermes-agent#106348) instead of fighting the gate.

Not viable: UA/session spoofing (sections 3-4) -- deeper fingerprinting + circumvention clause + account risk.


---

# Part 2 (2026-10-14): where to actually get free / near-free quota instead

## OpenRouter: free list is small, and has NO DeepSeek/MiMo/Kimi/GLM/Qwen
`curl https://openrouter.ai/api/v1/models` -> only **14** ids end in `:free` and price 0/0:

| id | ctx |
|---|---|
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 1,000,000 |
| `nvidia/nemotron-3-super-120b-a12b:free` | 262,144 |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256,000 |
| `nvidia/nemotron-3.5-lightning:free` | 1,000,000 |
| `nvidia/nemotron-3.5-content-safety:free` | 128,000 |
| `google/gemma-4-31b-it:free` | 262,144 |
| `google/gemma-4-26b-a4b-it:free` | 262,144 |
| `cohere/north-mini-code:free` | 256,000 |
| `poolside/laguna-s-2.1:free` | 262,144 |
| `poolside/laguna-xs-2.1:free` | 262,144 |
| `thinkingmachines/inkling:free` | 1,048,576 |
| `thinkingmachines/inkling-small:free` | 1,048,576 |
| `dots-studio/dots-3-note-preview:free` | 512,000 |
| `liquid/lfm-2.5-2.6b:free` | 65,536 |
| `openrouter/free` (router, picks a random free model) | 200,000 |

Rate limits (from the docs JS constants): `FREE_MODEL_RATE_LIMIT_RPM=20`,
`FREE_MODEL_NO_CREDITS_RPD=50`, `FREE_MODEL_HAS_CREDITS_RPD=1000`,
`FREE_MODEL_CREDITS_THRESHOLD=10` -> 20 req/min; 50 req/day if you have bought <10 credits
all-time; 1,000 req/day once you have bought >=10 credits.

**Near-free instead of free** (cheapest paid OpenRouter ids, $/1M input tokens):
`deepseek/deepseek-v4-flash-0731` $0.016, `deepseek/deepseek-v4.1-flash` $0.0202,
`z-ai/glm-5.3-flash` $0.036, `qwen/qwen3.7-flash` $0.03, `xiaomi/mimo-v2.6-flash` $0.14,
`moonshotai/kimi-k3` $0.70. I.e. a full 1M-token DeepSeek V4.1 Flash pass costs ~2 US cents.

## Providers that are genuinely free and decent
| Provider | Free thing | Notes |
|---|---|---|
| **Z.ai / GLM** | `GLM-4.7-Flash`, `GLM-4.5-Flash`, `GLM-4.6V-Flash` listed as **Free / Free / Free / Free** on the official pricing page | best sanctioned "actually free" option; Hermes has a native `zai` provider (`GLM_API_KEY`) |
| **NVIDIA NIM** | "Free Endpoint Available" for `nvidia/nemotron-3-ultra-550b-a55b` (1M ctx, agentic coding), `moonshotai/kimi-k3`, `deepseek-v4-pro-0813` | call `https://integrate.api.nvidia.com/v1` with an `NVIDIA_API_KEY`; note the **kimi-k3 free endpoint is deprecated 2026-10-14 / unsupported after 10-15** |
| **Alibaba Model Studio (DashScope)** | new-user free quota, **~1,000,000 tokens per model, valid 90 days**, Singapore region, real-time inference only | covers `qwen3.8-max`, `qwen3.8-flash`, `deepseek-v4-pro-0813`, `glm-5.3`, `kimi-k3`, `MiniMax-M2.5`; Hermes has a native `alibaba` provider (`DASHSCOPE_API_KEY`); enable "Free Quota Only" to avoid surprise billing |
| **Groq** | Free plan (e.g. `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` ~30 RPM / 1K RPD / 8K TPM / 200K TPD) | Hermes community plugin `groq-provider` |
| **Cloudflare Workers AI** | **10,000 Neurons/day free** on the Workers Free plan | some models (kimi-k2.6/k2.7...) require paid billing; Hermes plugin `cloudflare-workers-ai-provider` |
| **Google AI Studio / Gemini API** | free tier exists (per-model RPM/RPD/TPM shown in AI Studio for your project) | rate-limit table is rendered client-side; check AI Studio |
| **Nous Portal** | Hermes' own OAuth provider, subscription-based | the "Nemotron 3 Ultra free" guide was a **2-week promo (June 4-18)**, expired |

Dead ends: **Cerebras** has *no* permanent free tier (only $5 of credits expiring in 30 days);
**GitHub Models was fully retired on 2026-07-30**.

## The one community attempt at OpenCode-free-in-Hermes
- `oc-free-provider` — community plugin by **anpicasso** (`github.com/anpicasso/hermes-oc-free-provider`),
  v3.0.0, 9 stars, listed Sep 21 2026, **last pushed 2026-09-21 — the exact day the Zen gate started 403ing**.
  `hermes plugins install oc-free-provider`. Its own README says: *"This plugin's current public
  compatibility transport is not documented by OpenCode as a supported Hermes integration and **may be
  changed or disabled at any time**"* and *"Do not use this project to ... **bypass an access restriction**.
  A disclaimer cannot make a prohibited use compliant."*
- Same pattern elsewhere: `diegosouzapw/OmniRoute` has an open PR (2026-10-11) *"remove the OpenCode Free
  provider and its free-tier cl..."*; `decolua/9router` issue (2026-10-13) *"fix(opencode): 400
  missing_required_parameter 'messages[N].reasoning_c..."*; `haoyiyin/9router` advertises
  "OpenCode Free oc/ models" for Pi.
  => every third-party bridge keeps breaking as the gate moves. Treat as unsupported.


---

# Part 3 (2026-10-14): OpenCode Go economics, and where GPT Luna sits

## How Go's limits actually work (from https://opencode.ai/docs/go)
- Plans: **Go $10/month**, **Go Plus $40/month** ("2x - Limited time").
- "Usage limits are defined as **monthly dollar amounts**" per model. Each model has a monthly
  dollar allowance, and three rolling caps: **5-hour = 20% of monthly, weekly = 50%, monthly = 100%**.
- Over-limit: you can enable **"Use balance"** in the console so Go falls back to your pay-as-you-go
  Console balance instead of blocking.
- Why allowances differ: *"For most models, we make this work through bulk discounts and reserved GPU
  capacity... For some models, we haven't had the opportunity to negotiate a discount... For these models,
  you still get a little more than if you paid the model providers directly; this is why their included
  monthly usage is lower."*

## GPT 6 Luna in the OpenAI lineup (models.dev + Console)
| model | released | $/1M in | $/1M out |
|---|---|---|---|
| GPT 6 Astra | 2026-09-04 | $10 | $50 |
| GPT 6.1 Sol | 2026-09-29 | $2 | $10 |
| GPT 6 Sol | 2026-09-22 | $2 | $10 |
| **GPT 6 Luna** | 2026-09-22 | **$0.10** | **$0.50** |
| GPT 5.6 Luna | 2026-07-09 | $0.20 | $1.20 |

- `gpt-6-luna` = *"OpenAI's most efficient model for focused, high-volume tasks"*; 1.05M ctx, 128K output,
  tool calling + reasoning. `gpt-6-luna-pro` = **same underlying model** with `reasoning.mode=pro` for
  higher quality on complex tasks, **same price**.
- Go lists Luna at `$0.10 / $0.50` (<=272K) and `$0.20 / $0.75` (>272K), **monthly limit $15**
  -> **4,230 requests / 5h, 10,560 / week, 21,130 / month**.

## Go included-requests table (selected, requests per month)
| model | req/5h | req/week | req/month | monthly $ limit | $/1M in-out |
|---|---|---|---|---|---|
| Muse Spark 1.3 Contributor | 45,300 | 113,300 | **226,600** | $60 | 0.10 / 0.20 |
| DeepSeek V4.1 Flash | 34,670 | 86,670 | **173,330** | $80 | 0.15 / 0.60 (off-peak) |
| MiMo-V2.6-Flash | 30,100 | 75,200 | **150,400** | $60 | 0.14 / 0.28 |
| LongCat-2.0 | 11,400 | 28,600 | 57,200 | $60 | 0.30 / 1.20 |
| GLM-5.3-Flash | 6,320 | 15,790 | 31,580 | $60 | 0.15 / 0.50 |
| Qwen3.8 Flash | 5,400 | 13,500 | 27,000 | $30 | 0.15 / 0.47 |
| Hy3 | 4,300 | 10,750 | 21,500 | $60 | 0.14 / 0.58 |
| **GPT 6 Luna** | **4,230** | **10,560** | **21,130** | **$15** | **0.10 / 0.50** |
| Claude Haiku 5.5 | 3,850 | 9,620 | 19,230 | $15 | 0.10 / 0.50 |
| MiMo-V2.6-Pro | 3,250 | 8,150 | 16,300 | $15 | 0.435 / 0.87 |
| Space Bunny | 3,130 | 7,810 | 15,630 | $30 | 0.15 / 0.60 |
| MiniMax M3 | 3,200 | 8,000 | 16,000 | $60 | 0.30 / 1.20 |
| GPT 5.6 Luna | 2,050 | 5,100 | 10,250 | $15 | 0.20 / 1.20 |
| Kimi K2.7 Code | 1,350 | 3,380 | 6,750 | $60 | 0.95 / 4.00 |
| DeepSeek V4 Pro | 1,050 | 2,600 | 5,200 | $15 | 0.66-1.32 / 1.98-3.96 |
| GLM-5.3 | 220 | 540 | 1,080 | $15 | 1.40 / 4.40 |
| Qwen3.8 Max | 160 | 400 | 810 | $15 | 2.00 / 6.00 |
| Grok 4.7 | 169 | 423 | 845 | $15 | 2.00 / 6.00 |
| Kimi K3 | 110 | 250 | 490 | $15 | 3.00 / 15.00 |
| LongCat 2.5 Preview Free | Unlimited | Unlimited | Unlimited | Free | 0 / 0 |
| Step 5 Preview Free | Unlimited | Unlimited | Unlimited | Free | 0 / 0 |
| Cat Jam Free | Unlimited | Unlimited | Unlimited | Free | 0 / 0 |


---

# Part 4 (2026-10-14): GPT-6 Luna vs the rest, on real benchmarks

Source: Artificial Analysis leaderboard (`artificialanalysis.ai/leaderboards/models`, scraped 2026-10-14)
+ AA article "GPT-6 Sol and Luna push the cost efficiency frontier" (2026-09-22).
Columns: AA **Intelligence Index** (higher better), **cost per task** (USD), output speed (tokens/s).

| Model | ctx | Intelligence | $/task | t/s |
|---|---|---|---|---|
| Claude Opus 5.5 (max) | 1M | 58 | $5.98 | 95 |
| Claude Sonnet 5.5 (max) | 1M | 56 | $5.46 | 137 |
| GPT-6 Astra (max) | 1M | 53 | $3.26 | 61 |
| Gemini 4 Argon (high) | 1M | 53 | $1.99 | - |
| GPT-6.1 Sol (max) | 1M | 52 | $0.72 | 59 |
| GPT-6.1 Sol (medium) | 1M | 48 | $0.21 | 55 |
| Muse Spark 1.3 (max) | 1M | **48** | $1.60 | 128 |
| MiMo-V2.6-Pro | 1M | **46** | **$0.13** | 47 |
| Grok 4.7 (xhigh) | 500k | 46 | $3.74 | 86 |
| GLM-5.3 (max) | 1M | 45 | $2.01 | 62 |
| Kimi K3 (max) | 1.05M | 44 | $2.00 | 45 |
| Claude Haiku 5.5 (max) | 1M | 43 | $0.21 | **237** |
| GLM-5.3-Flash | 1M | 42 | $0.28 | 67 |
| DeepSeek V4.1 Flash (max) | 1M | 39 | $0.27 | 187 |
| MiMo-V2.6-Flash | 1M | 38 | **$0.06** | 65 |
| **GPT-6 Luna (max)** | 1M | **38** | **$0.07** | 135 |
| LongCat 2.0 | 1M | 19 | $0.15 | 68 |

Coding Agent Index (AA, Codex harness): **GPT-6 Luna (max) = 41**, a 2-point *regression* from GPT-5.6 Luna
(max) = 43, at ~60% lower cost per task ($0.07 vs $0.18 on the Intelligence Index). GPT-6 Sol (max) = 57.
AA-Omniscience: Luna hallucination 93% -> 77%, accuracy 44% (vs 43%), index -10 -> +1. GDPval-AA v2.1
drops ~75 Elo; AA-Briefcase v1.1 drops ~45 Elo. Luna emits more output tokens per task (51k vs 41k).

## Verdict for a Go subscriber
- Luna's index (38) equals **MiMo-V2.6-Flash** (38) and sits *below* DeepSeek V4.1 Flash (39),
  GLM-5.3-Flash (42), Haiku 5.5 (43), GLM-5.3 (45), MiMo-V2.6-Pro (46), Muse Spark 1.3 (48).
- On Go's own allowance table Luna is also the *worst value* of that group:
  Luna $15 -> 21,130 req/mo, vs MiMo-V2.6-Flash $60 -> 150,400, DeepSeek V4.1 Flash $80 -> 173,330,
  Muse Spark 1.3 Contributor $60 -> 226,600, GLM-5.3-Flash $60 -> 31,580.
- Muse Spark 1.3 Contributor is index 48 at $0.10/$0.20 -- the standout, but its prompts/completions
  **may be used to train future Meta models** and availability is region-limited.
- Luna is only the right pick when you specifically want an OpenAI-family small model / its tool-calling
  behaviour, 1M context, or high concurrency at low cost per call.

## Read of it
- Value multiple = monthly allowance / $10 sub: Luna and the other un-discounted frontier models are
  **1.5x** ($15/$10); MiMo-V2.6-Flash, GLM-5.3-Flash, LongCat-2.0, Hy3, MiniMax are **6x** ($60);
  DeepSeek V4.1 Flash is **8x** ($80). Luna's public OpenAI price ($0.10/$0.50) equals Go's listed price,
  so Go is 1.5x cheaper than paying OpenAI direct for Luna.
- Same price outside Go: OpenRouter / OpenAI / Azure all list `gpt-6-luna` at **$0.10/$0.50** (Databricks
  $1/$6, AWS Mantle GPT-5.6 Luna $0.22/$1.32) -> you can pay-per-use Luna without the $10 subscription.
- Cheapest sane mix for a Hermes user on Go: run everyday turns on the **3 unlimited free stealth models**
  (LongCat 2.5 Preview Free / Step 5 Preview Free / Cat Jam Free) or the 6x-value Flash models, and reserve
  Luna/Haiku for tasks where you specifically want an OpenAI/Anthropic-family small model.

