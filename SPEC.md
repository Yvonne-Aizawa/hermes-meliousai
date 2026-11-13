# Spec: Melious-aware model provider for Hermes (installable plugin)

Status:        IMPLEMENTED (2026-11-13) — see §10 for the record and the two
               corrections building it forced. Source: this repository.
               Installs to: <hermes home>/plugins/model-providers/melious/
Author:        Lumina, 2026-11-13, for Yvonne
Goal:          DONE. This document is now the design record; the plugin repo above is
               the living artifact. Do NOT re-derive the facts — they were measured.

Revision:      2 (2026-11-13). Rev 1 wrongly claimed Hermes has no LLM-provider
               plugin interface; Rev 2 corrects that (see §0) — the interface
               exists and a community plugin (`mistral-vibe`) proves it.

---

## 0. TL;DR

Hermes HAS a model-provider plugin interface. `~/.hermes/plugins/model-providers/<name>/`
plugins register a `ProviderProfile` (`providers/base.py`) that can supply its
own client transport (`create_client`), request-field translation
(`build_api_kwargs_extras`), error classification (`classify_api_error`),
model catalog (`fetch_models`) and custom auth — no core edits required.
Official guide: `website/docs/developer-guide/model-provider-plugin.md` in the
checkout (also at hermes-agent.nousresearch.com/docs). Reference
implementation: `https://github.com/nebneo/hermes-mistral-vibe` (files:
`plugin.yaml`, `vibe_provider.py`, `install.py`; installs to
`~/.hermes/plugins/model-providers/mistral-vibe/`).

**Recommended build: a native model-provider plugin `melious`** (mistral-vibe
pattern) that wraps api.melious.ai with the Melious-aware policy below. The
plugin is picked up for the EXISTING `custom:melious` provider automatically
via base-url prefix match (`_profile_for_base_url` in
`agent/agent_runtime_helpers.py`: a profile whose `base_url` is a prefix of the
configured one is used), so `model.provider` does not even need to change.

Fallback options if the plugin client cannot express something:
- **B-sidecar** — local OpenAI-compatible proxy owned by the plugin (kokoro-tts
  sidecar pattern). Only needed if we must intercept traffic transparently for
  callers we cannot teach about a new provider id (aux models, MCP, cron).
- **C-upstream** — nothing new needed upstream; at most a marketplace listing.

## 1. Why this exists (measured facts, do not re-litigate)

Melious (api.melious.ai) is Yvonne's primary LLM provider, OpenAI-wire, model
family glm-5.3-flash / glm-5.3 / deepseek-v4*. Failure behavior measured in
production on 2026-11-13:

1. **Long prompts go to a separate queue.** Sessions routinely send 80–171K
   tokens (`est_tokens` via `estimate_request_context_tokens`). At peak the
   long queue is slow for some models. Confirmed by Melious support.
2. **The dominant failure:** stream opens (HTTP 200), zero chunks arrive, then
   at ~240 s Melious sends an SSE error frame `server_error: "The model did not
   start responding in time. Please try again."` (their new stream-start
   timeout). Observed silences 180–300 s; 240.2 s measured on the canonical
   record. Hermes classifies it retryable → without tuning, one message can
   burn 12+ minutes (3 attempts × ~4.5 min + post-exhaustion recovery cycles).
3. **The error envelope carries a templated `retry_after: 30`**, but Melious
   support explicitly recommends a SHORT backoff so the retry reuses the
   prompt prefix cache and lands in the short queue.
4. **Request correlation:** the request id exists ONLY in the `X-Request-ID`
   response header (UUID-shaped, e.g. `3f2504e0-7f9a-4d61-bc7c-1a2b3c4d5e6f`).
   Error bodies carry none by design. No response ⇒ no id client-side. Melious
   accepts a client-supplied `request_id` body param (docs) — Hermes core does
   not send one, which is the one gap only a provider plugin can close.
5. **Cache rule:** a retry that resends a byte-identical prefix (and follow-up
   turns within ~10 min) can be served from prefix cache and enter the short
   queue. Zero-chunk failures already retry identically — preserve that.
6. **Peak-time escape hatch:** support suggests `deepseek-v4.1-flash` (undated)
   — verified present on `/v1/models` and in the melious `custom_providers`
   allowlist.

Canonical ledger record (the shape a support ticket needs; values illustrative):

```json
{"ts": "2026-11-13T17:27:18", "reason": "provider_stream_error",
 "session_id": "20260101_000000_00000000", "provider": "custom:melious",
 "model": "glm-5.3-flash", "base_url": "https://api.melious.ai/v1",
 "api_mode": "chat_completions", "error_type": "APIError",
 "error_code": "server_error", "status": 200,
 "provider_request_id": "3f2504e0-7f9a-4d61-bc7c-1a2b3c4d5e6f",
 "retry_after": 30, "messages": 102, "est_tokens": 86024,
 "stream": {"http_status": 200, "chunks": 0, "ttfb_s": null,
            "no_output_s": 240.2, "open_headers": {"x-request-id": "3f2504e0-7f9a-4d61-bc7c-1a2b3c4d5e6f", "server": "nginx"}}}
```

## 2. Reference material

The plugin is STANDALONE: nothing below is required for it to build, install or
run. These are the references it was designed against.

| Artifact | Location | Notes |
|---|---|---|
| Provider-plugin guide | `website/docs/developer-guide/model-provider-plugin.md` in the Hermes checkout (also at hermes-agent.nousresearch.com/docs) | authoritative interface doc |
| Reference impl | GitHub `nebneo/hermes-mistral-vibe` | the pattern this plugin mirrors (`plugin.yaml` + provider module + `install.py`) |
| Melious API docs | https://melious.ai/docs/llms.mdx/reference/chat-completions | request_id param, X-Request-ID header, error shapes |
| Host-side failure ledger (optional) | `<hermes home>/provider-failures.log` | this plugin writes its own lines on a stock Hermes; a host that also carries a `provider-request-id-and-failure-ledger` core patch additionally gets core-written lines for every stream failure and terminal dump |

A host that installs such a core patch keeps it as the safety net for traffic
NOT routed through this plugin (other providers, MCP, auxiliary calls) — the
plugin does not depend on it.

## 3. Recommended core settings (A) — the policy the plugin embodies

```yaml
providers:
  "custom:melious":
    request_timeout_seconds: 90     # becomes stream read timeout too
    stale_timeout_seconds: 90       # operator deadline, beats context floors
    retry_after_cap_seconds: 10     # caps templated retry_after:30 (non-429)
agent:
  api_max_retries: 2
  auto_recovery_cycles: 0           # kills the 12-minute resurrection loop
  stream_retries: 0                 # one stream attempt per API call
fallback_providers:
  - provider: custom:melious
    model: deepseek-v4.1-flash
```

Trade-off baked in: 90 s stale PREEMPTS Melious's 240 s error, so they see
fewer of their own errors from us. If their prefix cache only populates after
prefill completes, retries may still hit the long queue — expose a knob to go
back to waiting for the 240 s error (§6 open questions).

## 4. Target architecture — native model-provider plugin `melious`

### 4.1 Component layout (proposal, mirror mistral-vibe)

```
~/.hermes/plugins/model-providers/melious/
  __init__.py              # discovery entry; exposes `profile`
  plugin.yaml              # manifest (name/version/description/…)
  melious_provider.py      # ProviderProfile subclass + profile instance
  install.py               # copy MANAGED_FILES into place (mistral-vibe pattern)
  README.md
  tests/                   # offline suite: disposable HERMES_HOME, socket-blocked,
                           # httpx fixtures (mirror mistral-vibe's test approach)
```

`plugin.yaml` keys to check against the guide: name, version, manifest_version,
description, author, license, requires_hermes. No `provides_tools`/hooks needed
(provider plugins register via the profile, discovered by
`hermes_cli/plugins_discovery.py` and mirrored into `PROVIDER_REGISTRY` by
`hermes_cli/auth_plugin_providers.sync_plugin_provider_registry`).

### 4.2 Interface mapping (which hook does what)

| ProviderProfile hook | Melious behavior |
|---|---|
| `name` | `"melious"` (aliases optional; the existing id `custom:melious` keeps working via base-url match) |
| `base_url` | `https://api.melious.ai/v1` — prefix match makes this serve `custom:melious` automatically |
| `auth_type` / `env_vars` | `api_key`; `(HERMES_CUSTOM_API_MELIOUS_AI_API_KEY,)` — reuses the existing key, no new auth flow needed |
| `create_client(**kwargs)` | `MeliousClient`: see §4.3. MUST tolerate unknown kwargs (core adds fields over time). Return a standard-shaped OpenAI client object so core streaming/tool machinery is untouched. |
| `build_api_kwargs_extras` | Inject top-level `request_id` (client-generated UUID per logical request, stable across its retries) — mirrors how mistral-vibe injects top-level `reasoning_effort` |
| `classify_api_error` | Precise classification of Melious errors: `server_error`/"did not start responding" → retryable-short; `RATE_LIMIT_EXCEEDED` → rate-limit; return None for anything unknown |
| `fetch_models` | Live catalog from `/v1/models` (mistral-vibe's implementation is a template; reuse the melious models endpoint + key from env) |
| `supports_health_check` / `supports_model_listing` | True / True |

### 4.3 `MeliousClient` behavior (acceptance tests hang off this)

**P0 — pass-through fidelity.** Must behave exactly like the standard client:
chat.completions streaming + non-streaming, tool calling, usage chunk,
`stream_options`/`include_usage`, `response_format`, vision parts. Keep
`max_retries=0` (retries are owned by Hermes core + this client's policy, NOT
the OpenAI SDK — mistral-vibe does the same). Forward Melious's `X-Request-ID`
header to Hermes (core ledger reads it from the stream response headers).

**P1 — client-side correlation.** Generate `request_id` (UUID) per logical
request; send it as the Melious `request_id` body param via
`build_api_kwargs_extras`. Log it in the ledger line even when no response
arrives (this closes the one gap core cannot: "no response ⇒ no id").

**P2 — queue classification.** Classify each request by `est_tokens`
(reuse `estimate_request_context_tokens`): SHORT (< 32K) vs LONG (≥ 32K —
threshold is a config knob; validate against Melious, §6). Stamp the class
into the ledger line and log it at debug level.

**P3 — retry policy (the 240 s error).** Design decision: PREFER core-driven
retries (already tuned in §3: 90 s stale preempts the 240 s error, 2 attempts,
10 s backoff cap) over duplicating a retry loop inside the client. The client's
job is then: send `request_id`, expose the error to core cleanly, and make sure
the ledger gets every attempt. Implement an INTERNAL short-backoff retry ONLY
if core behavior proves insufficient in testing (documented in the failure
matrix below as the escalation path). Byte-identical prefix across retries is
guaranteed by core for zero-chunk failures — never append anything.

**P4 — failover.** Prefer core `fallback_providers` (already configured:
`deepseek-v4.1-flash`). If we want same-call failover inside the client,
implement carefully: never after partial output; ledger `attempt_model` per
attempt. Recommend core-driven first; internal only if the UX demands it.

**P5 — prefix stability across turns.** Never mutate prior turn prefixes, no
system-prompt rewrites. Keep the `request_id` param stable across retries of
the SAME logical request (it participates in the retry being cacheable —
verify with Melious which headers/fields are in the cache key, §6).

**P6 — ledger + metrics.** A host carrying the `provider-request-id-and-failure-ledger`
core patch writes every stream failure to `<hermes home>/provider-failures.log`
with the §1 shape (including `est_tokens` and `provider_request_id`); on a stock
Hermes nothing writes there. Either way the plugin is standalone: its client ADDs
its own line (same JSON shape, marked `source: melious-plugin`) for the one case
core structurally cannot cover — a connect timeout, where no response arrives and
the plugin's generated `request_id` is the only id that exists. Keep ONE ledger
and document who writes what (README, "One ledger, two writers"). No separate
reader ships; `jq` on the JSONL covers it.

**P7 — support-ticket one-liner.** On every failure, emit a WARNING line in
`~/.hermes/logs/errors.log` (or the plugin's own log) with: ts, id, model,
est_tokens, no_output_s, error_code — the exact text someone can paste.

### 4.4 Failure matrix (encode as tests)

| Upstream event | Expected behavior |
|---|---|
| HTTP 2xx, chunks flow | stream through unchanged |
| HTTP 2xx, zero chunks > abort_silent_after_s (90) | core stale detector kills it → core retries (10 s cap) → ledger line with id |
| SSE error frame server_error "did not start responding" (~240 s) | same path; classified retryable-short by `classify_api_error` |
| HTTP 4xx (auth/format/billing) | pass through, ledger once, no retry |
| HTTP 429 with Retry-After | honor full Retry-After (cap does NOT apply), ledger, retry |
| 5xx before any chunk | core short-backoff retry, then fallback model |
| 5xx after partial output | no retry; Hermes core partial-delivery stub path |
| Connect timeout (no response at all) | client logs its `request_id` + `no_output_s` (the ONLY place an id exists) |
| All attempts failed | last error to Hermes; ledger `reason: exhausted` listing attempt ids |
| Fallback model succeeded | ledger `attempt_model: deepseek-v4.1-flash`, `reason: recovered_via_fallback` |

## 5. Config schema (plugin `config_schema`)

```yaml
melious_provider:
  short_threshold_tokens: 32000
  abort_silent_after_s: 90          # mirrors providers."custom:melious".stale_timeout_seconds
  retry_backoff_s: 10               # mirrors retry_after_cap_seconds
  max_attempts: 2                   # mirrors api_max_retries
  honor_retry_after: false          # for the templated 30s; 429s always honored
  fallback_model: deepseek-v4.1-flash
  ledger: "<hermes home>/provider-failures.log"   # shared with core
  log_level: info
```

The plugin should INSTALL these values into config.yaml on setup (idempotent),
so it is the single source of truth and uninstall restores previous values.

## 6. Open questions — status (docs + live probes, 2026-11-13)

Answered from https://melious.ai/docs (rate-limits, errors, streaming, models concepts)
and two live probes through the plugin's own client (glm-5.3-flash); no support ticket
needed for these.

1. **Queue split threshold — still OPEN.** The docs never mention queues; support's
   "separate long queue" claim has no published threshold. `short_threshold_tokens`
   stays a heuristic knob.
2. **Prefix-cache key — ANSWERED (probe): changing `request_id` does not evict.**
   Identical prompt resent immediately with a fresh client `request_id`:
   5 632 / 5 695 prompt tokens served from cache; the same again with no id.
   Docs: caching is transparent and PER-PROVIDER, reported as `usage.cached_tokens`
   at a cheaper cache-read rate; explicit `cache_control` markers are dropped.
3. **Prefill caching — ANSWERED (probe): an abort during prefill still caches.**
   A 72 125-token request was killed at 6.1 s (timeout, no response, no output); the
   identical retry was served with 28 160 cached tokens — the prefix processed before
   the abort survived. The 90 s stale-abort + short-backoff retry strategy is therefore
   measured-good; a cache miss on retry is usually a provider switch, not a broken
   prefix.
4. **`deepseek-v4.1-flash` — present in the catalog** (1 M context, hybrid reasoning,
   same window as glm-5.3 / glm-5.3-flash). The "recommended at peak-time escape"
   part remains support advice, not documentation.
5. **Account/stream limits — ANSWERED (docs).** Rate limits are per account and per
   endpoint path in a sliding 60 s window: 120/min base, 240/min with a credit balance
   ≥ 10, a per-key limit overrides, and concurrency plans add in-flight + daily caps.
   The account limiter's 429 is a flat body `{"code": "RATE_LIMIT_EXCEEDED",
   "details": {"retry_after": N}}` with no `type` — classify on `code`, honor
   `Retry-After` (which can be `0`; a concurrency cap always says `5`, a daily cap the
   seconds to midnight UTC). `X-RateLimit-*` headers ride on every response.

Also from the docs, corroborating this design: Melious retries the same request on the
NEXT PROVIDER internally when the first fails before any output (one response, one
charge — which is likely what the ~240 s stream-start timeout spans, and why an internal
failover also loses the prefix cache); their guidance is to set the SDK's
`max_retries=0` and retry yourself on `type`/`code` (never on message text), treat
503 `server_error`/`provider_error`/`timeout` as transient-with-backoff, retry 500
once and then quote `X-Request-ID`, and never auto-resend a stream after output has
started (billed, and it may duplicate tool calls).

## 7. Install/verify checklist (what "done" means)

- [ ] `python3 install.py --home ~/.hermes` places the plugin in
      `plugins/model-providers/melious/`; `hermes` discovers it
      (`hermes model` shows a `melious` provider; `hermes auth status`
      lists it).
- [ ] `provider: custom:melious` still works and the profile's
      `create_client` is actually used (log line or a sentinel header check;
      do NOT trust "it works" — assert the custom client was constructed).
- [ ] A normal chat turn streams through `MeliousClient` unchanged; tool calls
      work; usage chunk arrives.
- [ ] Fake 240 s stall (test harness/monkeypatched upstream) produces: ledger
      line with client + upstream request ids, `est_tokens`, `no_output_s`,
      short-backoff retry with byte-identical body, then success or failover.
- [ ] The new ledger lines are readable (`jq` over `<hermes home>/provider-failures.log`,
      plugin lines carry `source: melious-plugin`).
- [ ] Uninstall restores previous config and Hermes works directly with
      Melious again.
- [ ] `hermes update` does not break the plugin (it lives under the Hermes
      home, outside the checkout).

## 8. Acceptance criteria (user-visible)

1. A stalled turn surfaces an error in ≤ ~3.5 min worst case (2 attempts ×
   90 s + backoff), vs 12+ min today.
2. Every failure produces a support-ready record: id (client-generated, so it
   exists even on connect timeouts), model, timestamp, est_tokens, no_output_s.
3. At peak, repeated stalls on glm-5.3-flash automatically fail over to
   deepseek-v4.1-flash with a ledger trail.
4. Uninstalling the plugin restores the previous direct-to-Melious setup.

## 9. Out of scope / notes

- Core patch (A) stays as the safety net for non-plugin traffic; do not rip it
  out when the plugin ships. Revisit once the plugin has weeks of production
  data.
- Marketplace listing for the finished plugin (marketplace pages live at
  hermes-agent.nousresearch.com/marketplace — the mistral-vibe page is the
  format template). Note: community plugin, not affiliated with Melious —
  include the same kind of disclosure mistral-vibe carries.
- Revisit `short_threshold_tokens` and the cache/backoff knobs once Melious
  answers §6 — the answers tune, they do not re-architect.
- Rev 1's sidecar-proxy design (local OpenAI-compatible server) is preserved
  conceptually in §0 as option B-sidecar; do not build it unless the plugin
  client hits a hard wall.

## 10. Implementation record (2026-11-13)

Shipped as this repository. Verified at release time:

- Offline suite 28/28 green (`run_tests.py`, runtime interpreter auto-detected):
  the profile's hooks, request-id injection and uniqueness, `extra_body`
  preservation, streaming tool-call pass-through with the `.response` object
  intact, `max_retries=0`, both failure-ledger paths, the catalog cache, and
  install/update/uninstall/verify/doctor (including `--update` surviving the
  `__pycache__` that Hermes writes into the plugin dir).
- `tools/wire_check.py` (local echo server) pins the exact wire bytes: top-level
  `request_id` body field, sentinel header `X-Hermes-Provider-Plugin`,
  streaming, and the stream's `.response` (core's X-Request-ID source) intact.
- A live completion through the installed client returned 200 with `request_id`
  in the body — Melious accepts the field. A running Hermes picks the plugin up
  on its next provider lookup; no restart needed.
- Kill switch `HERMES_MELIOUS_PLUGIN_DISABLE=1` returns the standard client.

Corrections this build forced (both measured, not opinions):

1. **§0 is wrong about the base-url match serving `custom:melious`.**
   `get_provider_profile("custom:melious")` resolves the bundled GENERIC
   `custom` profile ("named custom routes share the generic wire policy unless
   a plugin explicitly registered that route") — name lookup SUCCEEDS, so
   `_provider_supplied_client`'s base-url fallback is never reached, and the
   transport's profile resolution (`get_provider_profile(agent.provider)`)
   bypasses the plugin too. Fix: the profile declares `aliases=
   ("custom:melious",)`, which claims the route. `model.provider` is still
   unchanged.
2. **§4.2's "top-level request_id" cannot be a top-level kwarg.** The OpenAI
   SDK's `create()` has a fixed signature; it goes in `extra_body` (which the
   SDK flattens into the JSON body — so it IS a top-level wire field, which is
   what Melious documents). Verified on the wire by tools/wire_check.py.

Residual gap (documented, deliberately not closed): agents whose provider
string is literally `custom` (not `custom:<name>`) keep the standard client —
core resolves those to the bundled generic `custom` profile, and this plugin
does not claim that shared name. Options if that ever matters: point
`model.provider` at `melious` (re-key any tuned `providers."custom:<name>"`
settings accordingly) or override the bundled `custom` profile with an
endpoint-gated client.

Two structural facts worth keeping (also in the skill reference):

- Core hoists `provider_request_id` from the error's response headers ONLY —
  `_api_error_debug_info` sets it inside `if response_obj is not None`. So the
  connect-timeout case needs the plugin's own ledger line (P6's second option),
  which is what shipped: one ledger, two writers, `source` marks the plugin's.
- `ssl_verify` / `ssl_ca_cert` are popped from client_kwargs BEFORE the provider
  seam, so no plugin client can honor them; `SSL_CERT_FILE` (via httpx
  trust_env) is the workaround.

§6 open questions remain open — ask Melious; the answers only tune
`short_threshold_tokens` and the backoff knobs, not the architecture.

### Patch retirement (2026-11-13, later the same day)

The core patch is now RETIRED on the host this was built on (reverted from the checkout,
the `.patch` file renamed to `.patch.retired` so the reapply script skips it). The
plugin absorbed its jobs; 32/32 offline tests cover the new paths:

- **In-stream SSE error frames** (the ~240 s stream-start timeout — the patch's
  `_attach_stream_response_headers` + stream-error ledger): the client now wraps the
  returned Stream in a transparent `MeliousStreamProxy`. Core iterates it unchanged
  (`_iter_provider_stream_chunks` is `yield from`), keeps reading `.response`, and reaches
  `get_final_message()`/`close()` through the attribute passthrough. On an iteration error
  the proxy is the only thing that still holds the response object: it records BOTH ids
  (client `request_id` + upstream `X-Request-ID`), stamps the upstream id onto the
  exception where stock core's `_api_error_debug_info` reads `request_id`, writes the
  `provider_stream_error` ledger line (chunks / ttfb_s / no_output_s / open headers), and
  clamps the templated `retry_after`. Partial-output failures ledger with `ttfb_s` and
  no auto-resend (a resend after output is separately billed).
- **The backoff cap** (the patch's `turn_recovery.py` change): `clamp_retry_after` caps a
  non-rate-limit body `retry_after` at `melious_provider.retry_backoff_s` — stock core
  honors a body `retry_after` up to 120 s (`agent.retry_utils.provider_retry_after_seconds`
  reads top-level or `error`-nested). Rate limits keep their real cooldown.
- **`agent.stream_retries`** (patch-only config key) → `HERMES_STREAM_RETRIES=0` in
  `<home>/.env`; stock core reads the env var per call, and the dotenv loader puts it in
  `os.environ` for every Hermes process (gateway, dashboard, one-shots, cron).
- **`providers.<id>.retry_after_cap_seconds`** (patch-only) → removed; the plugin
  enforces the cap, so `config_drift` no longer mirrors it.
- **Intentionally dropped**: core-written ledger lines for OTHER providers' failures, and
  the request-dump `provider_request_id` top-level hoist (the plugin's `request_id` stamp
  reaches stock dumps through the same-named attribute).
- **Provider id**: a model switch landed `model.provider: melious` (the profile's
  canonical name), so the tuned keys were re-keyed to `providers."melious"`;
  `custom:melious` remains a working alias and `config_drift` mirrors whichever spelling
  is keyed. The 90 s fast-fail needed no move at all — `request_timeout_seconds` /
  `stale_timeout_seconds` are stock Hermes config.

### Vision facts from the live catalog (2026-11-13, evening)

`vision_analyze` returned "I don't see any image": Hermes routed the analysis to the main
chat model because models.dev marks the GLM-5.3 family multimodal — but Melious serves
those ids text-only (their own catalog reports `capabilities.vision` absent; only
`mistral-small-3.2-24b-instruct` carries it). Fix, split by ownership:

- **the model CHOICE** — `auxiliary.vision: {provider, model}` in `config.yaml`, the
  user-level knob (verified working: a live image test through it described the image).
- **the FACTS** — the plugin queries `GET /v1/models?include_meta=true` (free to poll,
  cached 1 h, keyed by home+key fingerprint): `default_vision_model()` discovers the
  vision model live (the documented hook for "no hardcoded id can rot"; the verified
  2026-11-13 id is the unreachable-catalog fallback), `live_context_length()` answers
  from `_meta.context_length`, and every fetch folds the catalog's per-model
  `vision`/`function_calling`/`context_length` into `model_capabilities` in place
  (Hermes reads that dict fresh on every lookup). A small static block stays as the
  pre-fetch floor — it is what corrects models.dev's false multimodal belief in a fresh
  process before any catalog call has run, e.g. for native image embeds.