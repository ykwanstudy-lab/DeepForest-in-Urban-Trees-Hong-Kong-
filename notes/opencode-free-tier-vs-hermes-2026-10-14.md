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

