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
| **One ledger line for no-response failures** | Hermes' failure-ledger writer hoists `provider_request_id` only when the error carries a *response* object. On a connect timeout the plugin's id is the only one that exists, so the plugin writes that line itself — same file, same JSON shape, `source: melious-plugin`. |
| **Long/short queue classification** | Melious serves long prompts from a separate queue that is slow for some models at peak. Each request is classified by Hermes' own token estimate and stamped into the log line / ledger record. |
| **Melious-specific error classification** | The stream-start timeout arrives as an SSE `server_error` frame inside an HTTP 200 stream, and the rate limit is `RATE_LIMIT_EXCEEDED`. Both are classified explicitly (`server_error` retryable / `rate_limit`) instead of falling through the generic pipeline. |
| **A support-ready one-liner per failure** | `melious provider failure: id=… model=… status=… code=… est_tokens=… queue=… no_output_s=… error=…`, logged at WARNING (so it lands in `logs/errors.log`). Paste it into a support ticket. |
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
  abort_silent_after_s: 90        # mirrors providers."custom:melious".stale_timeout_seconds
  retry_backoff_s: 10             # mirrors providers."custom:melious".retry_after_cap_seconds
  max_attempts: 2                 # mirrors agent.api_max_retries
  honor_retry_after: false        # the templated retry_after:30 is not honored
  fallback_model: deepseek-v4.1-flash
  log_level: info
```

The four *mirrored* keys are read-only documentation of the policy the plugin expects from
Hermes core. `install.py --doctor` compares them against the core settings they mirror and
prints any drift, so the two cannot silently diverge.

The core settings this plugin's policy assumes (recommended for the route):

```yaml
providers:
  "custom:melious":
    request_timeout_seconds: 90
    stale_timeout_seconds: 90
    retry_after_cap_seconds: 10
agent:
  api_max_retries: 2
  auto_recovery_cycles: 0
  stream_retries: 0
fallback_providers:
  - provider: custom:melious
    model: deepseek-v4.1-flash
```

## One ledger, two writers

`<Hermes home>/provider-failures.log` is a JSONL file. Two writers can feed it, and they
do not overlap:

| Writer | When | Notes |
|---|---|---|
| this plugin | a failure with **no HTTP response at all** | `"source": "melious-plugin"`, `"reason": "provider_connect_timeout"`; works on a stock Hermes, no core patch needed |
| Hermes core, when the host carries the `provider-request-id-and-failure-ledger` patch | every stream failure and terminal dump | carries Melious' own `X-Request-ID` when a response arrived; without that patch core writes nothing here |

The plugin is standalone: on a stock Hermes the ledger simply holds only the plugin's
lines. Read it with plain `jq` — e.g. `jq -c 'select(.source=="melious-plugin")'` — and
note the `est_tokens` field is enough to derive the queue class after the fact
(`>= short_threshold_tokens`).

## Coverage, and what the plugin does not do

Verified at release time by the offline suite, `tools/wire_check.py`, and a live
completion through the installed client. Three limits are worth stating plainly:

- **A route named `custom:<name>` is claimed via the profile's `aliases`.** Core resolves
  `custom:<name>` to the bundled *generic* `custom` profile unless a plugin explicitly
  registered that route, so an agent whose provider string is literally `custom` (some
  internal/older paths) still gets the standard client. If plain-`custom` coverage is
  ever wanted, the options are to target `model.provider: melious` directly or to override
  the bundled `custom` profile with an endpoint-gated client — neither is done here,
  deliberately.
- **`ssl_verify` / `ssl_ca_cert` cannot reach a plugin client.** Core pops those keys out of
  `client_kwargs` before it consults the provider seam, so a plugin structurally cannot see
  them. Use `SSL_CERT_FILE` / `SSL_CERT_DIR` (honored by httpx via `trust_env`) if a custom
  CA bundle is needed.
- **`abort_silent_after_s`, `retry_backoff_s`, `max_attempts` are declared, not executed.**
  The client never retries and never times out a stream on its own: those numbers describe
  the core settings the plugin expects, and `--doctor` reports when the two drift.

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

1. What exactly is the short/long queue split — input tokens or total? Can a request be
   classified on `prompt_tokens` alone?
2. Does the prefix-cache key include request fields (`request_id`, headers) or only the
   message/tool bytes? Is changing `request_id` between logical requests safe for the cache?
3. Does the prefix cache populate during prefill (so aborting at 90 s still caches the
   prefix for the retry) or only after the first token?
4. Is `deepseek-v4.1-flash` (undated) the recommended peak-time model, and does it share the
   long-queue behaviour?
5. Any per-account concurrency or stream limits the client should respect?

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
