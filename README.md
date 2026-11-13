# Melious provider plugin for Hermes Agent

An **unofficial community** model-provider plugin that adds Melious-aware behaviour to
[api.melious.ai](https://api.melious.ai) in [Hermes Agent](https://github.com/NousResearch/hermes-agent).
It is not affiliated with Melious or Nous Research.

Hermes keeps its agent loop, tools and UI; this plugin changes only how the Melious route
is served.

## What it adds

| Behaviour | Why |
|---|---|
| **Client-generated `request_id` on every call** | Melious returns its request id *only* in the `X-Request-ID` response header, so a request that gets no response has no id anywhere. The plugin generates one (UUID) per API call, sends it as the documented `request_id` body field, and records it on the one failure shape where nothing else can. |
| **A ledger line for every failed request** | No-response failures (connect timeouts) carry the client-generated id; an in-stream SSE error frame (the ~240 s stream-start timeout) carries **both** ids — the client id and Melious' upstream `X-Request-ID`, read off the response object the stream wrapper still holds when the SDK raises with none. Same file, same JSON shape, `source: melious-plugin`; works on a stock Hermes with no core patch. |
| **Long/short queue classification** | Melious serves long prompts from a separate queue that is slow for some models at peak. Each request is classified by Hermes' own token estimate and stamped into the log line / ledger record. |
| **Melious-specific error classification** | The stream-start timeout arrives as an SSE `server_error` frame inside an HTTP 200 stream, and the rate limit is `RATE_LIMIT_EXCEEDED`. Both are classified explicitly (`server_error` retryable / `rate_limit`) instead of falling through the generic pipeline. |
| **A backoff cap, plugin-enforced** | Melious templates `retry_after: 30` into its stream-start error while recommending a short backoff; stock Hermes honors a body `retry_after` up to 120 s. The plugin clamps it to `retry_backoff_s` (rate limits keep their real cooldown) — the job a core patch used to do. |
| **What each model can do, from the live catalog** | `GET /v1/models?include_meta=true` (free to poll, cached 1 h) is the provider's own capability truth, and `_meta.input_modalities` is the image signal its request path checks — the `capabilities.vision` flag is sparse (one True vs ~19 image-accepting models on a tested account). The plugin discovers the account's vision model LIVE from that signal (image-generation rows excluded), feeds every model's real image/tools/context facts into Hermes' capability lookups, and answers `live_context_length()` from it. A static floor corrects exactly one trap: models.dev marks the GLM-5.3 family multimodal, but Melious serves **glm-5.3** text-only — without the correction, images route to a blind model and are silently dropped (glm-5.3-flash and deepseek-v4.1-flash DO accept images there). |
| **A support-ready one-liner per failure** | `melious provider failure: id=… upstream_id=… model=… status=… code=… est_tokens=… queue=… no_output_s=… error=…`, logged at WARNING (so it lands in `logs/errors.log`). Paste it into a support ticket. |
| **A kill switch** | `HERMES_MELIOUS_PLUGIN_DISABLE=1` returns the standard client, leaving the install intact. |

Retry policy is deliberately **not** the plugin's business: this client never retries
(`max_retries=0`) and never touches the message prefix, so a retry that Hermes core decides
on resends a byte-identical body and stays eligible for Melious' prefix cache. The tuned
fast-fail settings below are what bound latency.

## Install

```sh
git clone <this repo> melious-provider-plugin
cd melious-provider-plugin
python3 install.py --home "${HERMES_HOME:-$HOME/.hermes}"
```

Run it with the **Hermes Python interpreter** (the same one `hermes` uses) so the
`--doctor` / `--verify` checks can read Hermes' own modules.

The installer touches only:

```text
<Hermes home>/plugins/model-providers/melious/
  __init__.py
  plugin.yaml
  melious_provider.py
```

It never reads or writes `config.yaml`, `auth.json` or `.env`. An existing installation is
refused by default.

**Restart Hermes** (or start a new session) after installing or updating: provider plugin
code is imported at discovery time.

### Update / uninstall / inspect

```sh
python3 install.py --home "$HERMES_HOME" --update      # replace managed files
python3 install.py --home "$HERMES_HOME" --uninstall   # remove the plugin
python3 install.py --home "$HERMES_HOME" --verify      # assert the route uses this client
python3 install.py --home "$HERMES_HOME" --doctor      # is the core config still in step?
```

`--update` refuses to replace a directory containing files this plugin does not manage;
`--force` overrides that, after you have looked at them.

## Authentication

No new credential. The profile reuses the key the existing `custom:melious` provider
already uses:

```sh
# the env var Hermes derives for a custom provider
HERMES_CUSTOM_API_MELIOUS_AI_API_KEY=...
# or, as a manual alias
MELIOUS_API_KEY=...
```

Nothing is written to `auth.json`, and `hermes auth add melious` is not needed.

## Configuration

All keys are optional. The plugin **never writes `config.yaml`** — that file is
security-sensitive, and every value below already has a home. Put this block in
`config.yaml` if you want to override the defaults:

```yaml
melious_provider:
  short_threshold_tokens: 32000   # >= this goes to Melious' long queue
  retry_backoff_s: 10             # ENFORCED here: cap on the templated body retry_after
  abort_silent_after_s: 90        # mirrors providers.melious.stale_timeout_seconds
  max_attempts: 2                 # mirrors agent.api_max_retries
  honor_retry_after: false        # the templated retry_after:30 is capped, never honored in full
  fallback_model: deepseek-v4.1-flash
  log_level: info
```

`retry_backoff_s` is enforced by the plugin itself: stock Hermes honors a body
`retry_after` up to 120 s, and Melious templates `retry_after: 30` into its stream-start
error while recommending a SHORT backoff — so the plugin clamps it to this value on every
non-rate-limit error it catches (rate limits keep their real cooldown). The other
*mirrored* keys are read-only documentation of the policy the plugin expects from Hermes
core; `install.py --doctor` compares them against the core settings they mirror and prints
any drift, so the two cannot silently diverge.

The core settings this plugin's policy assumes — all of them stock Hermes, no core
patch needed:

```yaml
providers:
  "melious":                      # or "custom:melious" — both resolve to the plugin
    request_timeout_seconds: 90   # also becomes the stream read timeout
    stale_timeout_seconds: 90     # the operator deadline that kills a silent stream
agent:
  api_max_retries: 2
  auto_recovery_cycles: 0
fallback_providers:
  - provider: melious
    model: deepseek-v4.1-flash
```

One policy knob lives in the environment on a stock Hermes: the per-API-call stream
attempt count is read from the `HERMES_STREAM_RETRIES` env var (default 2), not from
`config.yaml`. The tuned fast-fail policy wants one attempt per call:

```
HERMES_STREAM_RETRIES=0    # in <hermes home>/.env — reaches every Hermes process
```

## The failure ledger

`<Hermes home>/provider-failures.log` is a JSONL file. On a stock Hermes this plugin is
its only writer — one line per failed Melious request, `"source": "melious-plugin"`:

| Failure shape | `reason` | What it carries |
|---|---|---|
| no HTTP response at all (connect timeout) | `provider_connect_timeout` | the client-generated `request_id` — the only id that exists anywhere |
| an in-stream SSE error frame (the ~240 s stream-start timeout arrives this way) | `provider_stream_error` | **both** ids: the client `request_id` and Melious' upstream `X-Request-ID` (the stream wrapper still holds the response object when the SDK raises with none), plus `chunks`, `ttfb_s`/`no_output_s` and the open-time headers |

A host that carries the `provider-request-id-and-failure-ledger` core patch adds its own
core-written lines for the same events (without a `source` marker) — redundant by design;
the plugin exists so that patch can be retired. Read with plain `jq` — e.g.
`jq -c 'select(.source=="melious-plugin")'` — and note the `est_tokens` field is enough
to derive the queue class after the fact (`>= short_threshold_tokens`).

## Coverage, and what the plugin does not do

Verified at release time by the offline suite, `tools/wire_check.py`, and a live
completion through the installed client. Three limits are worth stating plainly:

- **A route named `custom:<name>` is claimed via the profile's `aliases`.** Core resolves
  `custom:<name>` to the bundled *generic* `custom` profile unless a plugin explicitly
  registered that route, so an agent whose provider string is literally `custom` (some
  internal/older paths) still gets the standard client. Setting `model.provider: melious`
  resolves the profile directly by name — the canonical spelling a model switch lands on —
  and sidesteps that whole class of paths. If plain-`custom` coverage is ever wanted on
  top, the option is to override the bundled `custom` profile with an endpoint-gated
  client — not done here, deliberately.
- **`ssl_verify` / `ssl_ca_cert` cannot reach a plugin client.** Core pops those keys out of
  `client_kwargs` before it consults the provider seam, so a plugin structurally cannot see
  them. Use `SSL_CERT_FILE` / `SSL_CERT_DIR` (honored by httpx via `trust_env`) if a custom
  CA bundle is needed.
- **`abort_silent_after_s` and `max_attempts` are declared, not executed.** `retry_backoff_s`
  IS executed (the clamp above). The client never retries and never times out a stream on
  its own — the 90 s kill is core's stock `stale_timeout_seconds`, and `--doctor` reports
  when the declared numbers and the core settings drift.

## Two deliberate deviations from `SPEC.md`

Both were found by running the contract tests against the installed Hermes; neither changes
the intent of the design.

1. **`request_id` rides in `extra_body`, not as a top-level kwarg.** SPEC §4.2 says
   "top-level". It *is* a top-level field on the wire — the OpenAI SDK flattens
   `extra_body` into the JSON body — but it cannot be passed as a top-level Python kwarg:
   `create()` has a fixed signature and raises `TypeError` on unknown keywords.
2. **The profile claims `custom:melious` as an alias.** SPEC §0 assumed the base-url prefix
   match in `_provider_supplied_client` would serve the existing route. It does not:
   `get_provider_profile("custom:melious")` resolves the bundled **generic `custom`**
   profile first ("named custom routes share the generic wire policy unless a plugin
   explicitly registered that route"), and the base-url fallback is only reached when name
   lookup *fails*. Registering the route as an alias is what makes both the client seam and
   the request-field hooks apply — still with `model.provider` unchanged.

## Tests

```sh
python3 run_tests.py
```

Offline: disposable `HOME`/`HERMES_HOME`, sockets blocked, all HTTP synthetic. Runs under
the Hermes runtime interpreter (auto-detected; set `HERMES_SOURCE` if the checkout is not
importable). Covers the profile's hooks, request-id injection and uniqueness, `extra_body`
preservation, streaming tool-call pass-through (including the `.response` object core reads
the upstream id from), `max_retries=0`, both failure-ledger paths, the catalog cache, and
install/update/uninstall/verify/doctor.

## Known unknowns (for Melious support)

Mostly answered from their docs plus two live probes — details and sources in SPEC §6:

1. **Open — the queue split threshold.** The docs never mention queues; the
   `short_threshold_tokens` knob stays a heuristic until support names a number.
2. **Answered (probe)** — changing `request_id` does **not** evict the prefix cache
   (5 632/5 695 tokens served from cache on an immediate resend with a fresh id).
3. **Answered (probe)** — the prefix cache **populates during prefill**: a 72 k-token
   request aborted at 6.1 s still left 28 k tokens cached for the identical retry.
4. **Half-answered** — `deepseek-v4.1-flash` is in the catalog (1 M context, same window
   as the GLMs); "the recommended peak-time escape" remains verbal advice.
5. **Answered (docs)** — per-account per-endpoint rate limits: 120/min sliding 60 s base,
   240/min with ≥ 10 credits; concurrency plans add in-flight + daily caps; the account
   limiter's 429 is a flat body with `code: RATE_LIMIT_EXCEEDED`.

## Security and limits

- Fixed origins: `https://api.melious.ai/v1` for inference and the catalog. A route pointed
  at another host is not catalogued by this plugin.
- The SDK's retries are disabled; proxy/redirect/SSL behaviour is otherwise left exactly as
  the standard client has it.
- The catalog cache is keyed by provider + Hermes home + a non-reversible key fingerprint;
  the raw API key is never cached.
- Failure records contain no prompt text, only counts and ids.
- Community plugin: not affiliated with Melious or Nous Research. Endpoint behaviour can
  change without notice.

## License

MIT — see [LICENSE](LICENSE).
