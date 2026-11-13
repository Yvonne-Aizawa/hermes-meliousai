"""Melious-aware model provider plugin for Hermes Agent.

Community plugin — not affiliated with Melious or Nous Research.

What this adds over plain OpenAI-wire behaviour:

* **Client-side request correlation.** Melious returns its request id only in the
  ``X-Request-ID`` *response header*, so a request that never receives a response has no
  id anywhere. This plugin generates one per API call, sends it as the documented
  ``request_id`` body field, and stamps it onto request-time failures that carry no
  response at all — so Hermes' provider-failure ledger always has something quotable.
* **Queue classification.** Long prompts (``>= short_threshold_tokens``) are served from
  a separate queue on Melious. The class is logged with every request and included in the
  support one-liner; the ledger's own ``est_tokens`` makes it derivable after the fact.
* **Precise error classification.** Melious' stream-start timeout arrives as an SSE
  ``server_error`` frame inside an HTTP 200 stream, and its rate limit is
  ``RATE_LIMIT_EXCEEDED``; both are classified explicitly instead of falling through the
  generic pipeline.

Retry policy is deliberately left to Hermes core (``agent.api_max_retries``,
``agent.stream_retries``, ``providers."custom:melious".stale_timeout_seconds`` and
``retry_after_cap_seconds``). This client never retries on its own
(``max_retries=0`` — the SDK accepts no retry ownership here) and never touches the
message prefix, so a core retry resends a byte-identical body and stays eligible for
Melious' prefix cache.

Config block, all optional, read from ``config.yaml``::

    melious_provider:
      short_threshold_tokens: 32000   # >= this goes to Melious' long queue
      abort_silent_after_s: 90        # mirrors providers."custom:melious".stale_timeout_seconds
      retry_backoff_s: 10             # mirrors providers."custom:melious".retry_after_cap_seconds
      max_attempts: 2                 # mirrors agent.api_max_retries
      honor_retry_after: false        # templated retry_after:30 is NOT honored
      fallback_model: deepseek-v4.1-flash
      log_level: info

The mirrored settings are read-only documentation of the policy the plugin expects from
core: this plugin never writes ``config.yaml`` (see README). ``config_drift()`` reports
when the two disagree and ``install.py --doctor`` prints it.

Kill switch: ``HERMES_MELIOUS_PLUGIN_DISABLE=1`` makes ``create_client()`` return ``None``
so Hermes builds its standard OpenAI client again, leaving the rest of the install intact.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from providers.base import ProviderProfile

logger = logging.getLogger("melious_provider")

# ── Identity ──────────────────────────────────────────────────────────────────
SLUG = "melious"
BASE_URL = "https://api.melious.ai/v1"
DISPLAY_NAME = "Melious"
PLUGIN_VERSION = "1.0.0"

# A header the standard client never sends: it is how ``install.py --verify`` (and any
# log dig) proves the plugin's client is the one actually on the wire.
SENTINEL_HEADER = "X-Hermes-Provider-Plugin"
SENTINEL_VALUE = f"{SLUG}/{PLUGIN_VERSION}"

# Reuse the credential the existing ``custom:melious`` provider already uses (Hermes
# derives this name for a custom provider). MELIOUS_API_KEY is accepted as a manual alias.
API_KEY_ENV_VARS = ("HERMES_CUSTOM_API_MELIOUS_AI_API_KEY", "MELIOUS_API_KEY")

DISABLE_ENV = "HERMES_MELIOUS_PLUGIN_DISABLE"
LEDGER_ENV = "HERMES_PROVIDER_FAILURE_LEDGER"

FALLBACK_MODELS = ("glm-5.3-flash", "glm-5.3", "deepseek-v4.1-flash")

DEFAULT_SETTINGS: dict[str, Any] = {
    "short_threshold_tokens": 32000,
    "abort_silent_after_s": 90,
    "retry_backoff_s": 10,
    "max_attempts": 2,
    "honor_retry_after": False,
    "fallback_model": "deepseek-v4.1-flash",
    "log_level": "info",
}

_MODELS_CACHE_TTL_S = 3600.0
_models_cache: dict[tuple, tuple[float, tuple[str, ...]]] = {}

# Same rotation budget the core ledger writer uses, so this writer cannot grow the file.
_LEDGER_MAX_BYTES = 5 * 1024 * 1024


# ── Config / environment ──────────────────────────────────────────────────────
def _load_config() -> dict[str, Any]:
    """The live config.yaml, or ``{}`` when it cannot be read. Never raises."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        return config if isinstance(config, dict) else {}
    except Exception:
        logger.debug("melious_provider: config.yaml not readable", exc_info=True)
        return {}


def plugin_settings() -> dict[str, Any]:
    """The effective ``melious_provider`` block: defaults overlaid with config.yaml."""
    settings = dict(DEFAULT_SETTINGS)
    block = _load_config().get("melious_provider")
    if isinstance(block, dict):
        for key, value in block.items():
            if key in DEFAULT_SETTINGS and value is not None:
                settings[key] = value
    return settings


def plugin_disabled() -> bool:
    """True when the operator asked for the standard client via the kill switch."""
    return (os.environ.get(DISABLE_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def hermes_home() -> Path:
    """The active Hermes home ($HERMES_HOME when set, else ~/.hermes)."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def ledger_path() -> Path:
    """The ONE provider-failure ledger — the same file Hermes core appends to."""
    override = (os.environ.get(LEDGER_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return hermes_home() / "provider-failures.log"


def api_key_from_env() -> str:
    """The Melious key from the environment, first match wins ("" when unset)."""
    for name in API_KEY_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _as_number(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def config_drift(config: dict[str, Any] | None = None) -> list[str]:
    """Drift between the plugin's declared policy and the core keys it mirrors.

    The plugin never writes ``config.yaml``. This is how ``install.py --doctor`` tells the
    operator when ``melious_provider.*`` and the equivalent core settings disagree — the
    values must be kept in step by hand (that is the one operational cost of not writing
    a protected file).
    """
    if config is None:
        config = _load_config()
    if not isinstance(config, dict):
        return ["config.yaml could not be read; policy not checked"]

    def core(*path: str) -> Any:
        node: Any = config
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    block = config.get("melious_provider")
    declared: dict[str, Any] = dict(DEFAULT_SETTINGS)
    if isinstance(block, dict):
        declared.update({k: v for k, v in block.items() if k in DEFAULT_SETTINGS and v is not None})

    drift: list[str] = []
    mirrors = (
        ("abort_silent_after_s", 'providers."custom:melious".stale_timeout_seconds',
         core("providers", "custom:melious", "stale_timeout_seconds")),
        ("retry_backoff_s", 'providers."custom:melious".retry_after_cap_seconds',
         core("providers", "custom:melious", "retry_after_cap_seconds")),
        ("max_attempts", "agent.api_max_retries", core("agent", "api_max_retries")),
    )
    for name, label, actual in mirrors:
        if actual is None:
            drift.append(f"{name}: declared {declared[name]!r} but {label} is unset")
        elif _as_number(declared[name]) != _as_number(actual):
            drift.append(f"{name}: declared {declared[name]!r} vs {label} = {actual!r}")

    fallback = declared["fallback_model"]
    chain = core("fallback_providers")
    models = [row.get("model") for row in chain if isinstance(row, dict)] if isinstance(chain, list) else []
    if fallback not in models:
        drift.append(f"fallback_model: declared {fallback!r}, absent from fallback_providers {models!r}")
    return drift


# ── Request helpers ───────────────────────────────────────────────────────────
def new_request_id() -> str:
    """The client-generated correlation id (UUID, the shape Melious uses itself)."""
    return str(uuid.uuid4())


def _settings_threshold() -> int:
    try:
        return int(plugin_settings()["short_threshold_tokens"])
    except (TypeError, ValueError, KeyError):
        return int(DEFAULT_SETTINGS["short_threshold_tokens"])


def classify_queue(est_tokens: int | None, *, threshold: int | None = None) -> str:
    """Which Melious queue a request lands in: ``long``/``short``/``unknown``.

    Support confirmed a separate queue for long prompts that is slow for some models at
    peak (2026-11). The threshold is a config knob to validate against Melious — see the
    open questions in SPEC.md §6.
    """
    if est_tokens is None:
        return "unknown"
    limit = threshold if isinstance(threshold, int) and threshold > 0 else _settings_threshold()
    return "long" if est_tokens >= limit else "short"


def estimate_tokens(body: Any) -> int | None:
    """Hermes' own cheap context estimate, or None when it is unavailable."""
    try:
        from agent.chat_completion_helpers import estimate_request_context_tokens

        return int(estimate_request_context_tokens(body))
    except Exception:
        return None


def _message_count(kwargs: dict[str, Any]) -> int:
    messages = kwargs.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def ensure_request_id(kwargs: dict[str, Any]) -> str | None:
    """Return this call's request id, injecting it into the body when absent.

    Melious documents ``request_id`` as a top-level body field. It rides in ``extra_body``
    because the OpenAI SDK's ``create()`` has a fixed signature and rejects unknown
    top-level kwargs, while it flattens ``extra_body`` into the JSON body — which is
    exactly the top-level field Melious wants on the wire.

    Returns None when ``extra_body`` is present but not a mapping: the caller's body is
    left untouched, and no id is invented for a request we do not fully control.
    """
    body = kwargs.get("extra_body")
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return None
    existing = body.get("request_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    request_id = new_request_id()
    body["request_id"] = request_id
    kwargs["extra_body"] = body
    return request_id


def _error_code(exc: BaseException) -> str | None:
    """The provider's error code from an SDK exception, nested or flat."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        candidate = err if isinstance(err, dict) else body
        for key in ("code", "type"):
            value = candidate.get(key)
            if value:
                return str(value)
    value = getattr(exc, "code", None)
    return str(value) if value else None


def _stamp_failure(
    exc: BaseException,
    *,
    request_id: str | None,
    model: Any,
    est_tokens: int | None,
    queue: str,
    elapsed_s: float | None,
) -> None:
    """Record the client id on the failure and emit the support one-liner.

    Hermes' ledger writer hoists ``provider_request_id`` from the error's *response*
    headers, so on a connect timeout — no response at all — the id this plugin generated is
    the only one that exists and core structurally cannot record it. So this method stamps
    it on the exception AND writes one ledger line itself (:func:`append_client_failure_record`),
    but ONLY for the no-response case. A failure that carries a response keeps Melious' own
    ``X-Request-ID``, which is the id support can actually look up.
    """
    no_response = getattr(exc, "response", None) is None
    if request_id:
        try:
            if no_response and not getattr(exc, "request_id", None):
                setattr(exc, "request_id", request_id)
        except Exception:
            logger.debug("melious: could not stamp request id on failure", exc_info=True)
        try:
            setattr(exc, "_melious_client_request_id", request_id)
        except Exception:
            pass

    no_output_s = round(elapsed_s, 1) if elapsed_s is not None else None
    if request_id and no_response:
        append_client_failure_record(
            request_id=request_id,
            model=model,
            est_tokens=est_tokens,
            queue=queue,
            no_output_s=no_output_s,
            error_type=type(exc).__name__,
            error_code=_error_code(exc),
            status=getattr(exc, "status_code", None),
        )

    logger.warning(
        "melious provider failure: id=%s model=%s status=%s code=%s est_tokens=%s queue=%s no_output_s=%s error=%s",
        request_id or "-",
        model or "-",
        getattr(exc, "status_code", None) or "-",
        _error_code(exc) or "-",
        est_tokens if est_tokens is not None else "-",
        queue,
        no_output_s if no_output_s is not None else "-",
        type(exc).__name__,
    )


def append_client_failure_record(
    *,
    request_id: str,
    model: Any,
    est_tokens: int | None,
    queue: str,
    no_output_s: float | None,
    error_type: str,
    error_code: str | None = None,
    status: Any = None,
    base_url: str | None = None,
) -> None:
    """Append ONE ledger line for a failure that arrived with no HTTP response at all.

    Same file and same JSON shape as the core record (``provider-failures.log``), with
    ``source`` marking this writer: core still writes its own line for the same event, but
    without an id, because ``_api_error_debug_info`` only hoists ``provider_request_id``
    when the error carries a response object. One ledger, two writers, documented here and
    in the README. Best effort: never raises.
    """
    import json

    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "reason": "provider_connect_timeout",
        "source": f"{SLUG}-plugin",
        "request_id": request_id,
        "provider_request_id": request_id,
        "model": model,
        "base_url": base_url or BASE_URL,
        "error_type": error_type,
        "error_code": error_code,
        "status": status,
        "queue": queue,
        "est_tokens": est_tokens,
        "stream": {"http_status": None, "chunks": 0, "ttfb_s": None, "no_output_s": no_output_s},
    }
    try:
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _LEDGER_MAX_BYTES:
            path.replace(path.with_name(path.name + ".1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.debug("melious: could not append the client ledger record", exc_info=True)


def _fingerprint(key: str) -> str:
    import hashlib

    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()


# ── Client ────────────────────────────────────────────────────────────────────
class MeliousClient:
    """OpenAI-shaped client for Melious.

    Owns a real ``openai.OpenAI`` built from the kwargs Hermes would have used, minus its
    retry policy (``max_retries=0``: retries belong to Hermes core and to this plugin's
    tuned policy, never to the SDK). It injects a client-generated ``request_id`` when the
    caller supplied none and stamps that id onto request-time failures that carry no
    response. Everything else is pass-through: streaming, tool calling, usage chunks,
    ``stream_options``, ``response_format`` and vision parts behave exactly as they do with
    the standard client, and the returned stream keeps its ``.response`` (which is where
    core reads Melious' ``X-Request-ID`` header from).
    """

    # Hermes must not rebuild this as a plain AsyncOpenAI: ``create`` has a real async path.
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, **client_kwargs: Any) -> None:
        # Unknown kwargs are tolerated on purpose: core adds to this mapping over time.
        self._client_kwargs = dict(client_kwargs)
        self._client: Any = None
        self._client_fingerprint: tuple | None = None
        self.chat = SimpleNamespace(completions=self)
        self.last_request_id: str = ""

    # ── OpenAI-shaped surface ─────────────────────────────────────────────────
    def create(self, **kwargs: Any) -> Any:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._create(**kwargs)
        return self._async_create(**kwargs)

    async def _async_create(self, **kwargs: Any) -> Any:
        import asyncio

        result = await asyncio.to_thread(self._create, **kwargs)
        if not kwargs.get("stream"):
            return result

        async def _chunks():
            sentinel = object()
            iterator = iter(result)
            try:
                while True:
                    item = await asyncio.to_thread(next, iterator, sentinel)
                    if item is sentinel:
                        break
                    yield item
            finally:
                await asyncio.to_thread(result.close)

        return _chunks()

    def _create(self, **kwargs: Any) -> Any:
        client = self._ensure_client()
        request_id = ensure_request_id(kwargs)
        est_tokens = estimate_tokens(kwargs)
        queue = classify_queue(est_tokens)
        logger.debug(
            "melious request id=%s model=%s queue=%s est_tokens=%s messages=%s stream=%s",
            request_id or "-",
            kwargs.get("model"),
            queue,
            est_tokens if est_tokens is not None else "-",
            _message_count(kwargs),
            bool(kwargs.get("stream")),
        )
        if request_id:
            self.last_request_id = request_id
        started = time.monotonic()
        try:
            # Pass-through: retries, prefix and response shape are untouched.
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            _stamp_failure(
                exc,
                request_id=request_id,
                model=kwargs.get("model"),
                est_tokens=est_tokens,
                queue=queue,
                elapsed_s=time.monotonic() - started,
            )
            raise

    def close(self) -> None:
        self._close()

    # ── Internals ────────────────────────────────────────────────────────────
    def _ensure_client(self) -> Any:
        """Build (or rebuild after a credential/endpoint change) the real SDK client."""
        from openai import OpenAI

        source = self._client_kwargs
        api_key = str(source.get("api_key") or api_key_from_env() or "missing")
        base_url = str(source.get("base_url") or BASE_URL).rstrip("/")
        timeout = source.get("timeout")
        headers = dict(source.get("default_headers") or {})
        headers.setdefault(SENTINEL_HEADER, SENTINEL_VALUE)
        fingerprint = (api_key, base_url, repr(timeout), tuple(sorted(headers.items())))
        if self._client is not None and self._client_fingerprint == fingerprint:
            return self._client
        self._close()
        client_kwargs: dict[str, Any] = {
            # Preserve the standard client's proxy/redirect/SSL behaviour: only retries
            # ownership and the sentinel header differ.
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": 0,
            "default_headers": headers,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._client = OpenAI(**client_kwargs)
        self._client_fingerprint = fingerprint
        logger.debug("melious client built (base_url=%s, max_retries=0)", base_url)
        return self._client

    def _close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("melious client close failed", exc_info=True)
        self._client = None
        self._client_fingerprint = None


# ── Profile ───────────────────────────────────────────────────────────────────
def classify_melious_error(
    error: Any,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str = "",
    body: Any = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Melious-specific error classification; ``None`` declines to the built-in pipeline.

    Bound as ``ProviderProfile.classify_api_error`` — a dataclass FIELD, not a method: an
    override defined on the subclass would be shadowed by the instance attribute the
    dataclass sets in ``__init__``, so this must be passed to the constructor.
    """
    code = str(error_code or "").strip().lower()
    text = str(message or "").lower()
    status = status_code if isinstance(status_code, int) else None

    if status == 429 or code in {"rate_limit_exceeded", "rate_limited", "too_many_requests"}:
        # Rate limits stay retryable and reach the fallback chain after backoff.
        return {"reason": "rate_limit"}
    if code == "server_error" or "did not start responding in time" in text:
        # Melious' stream-start timeout: HTTP 200, zero chunks, an SSE error frame at
        # ~240s. Retryable — and deliberately SHORT, via the core retry_after cap.
        return {"reason": "server_error", "retryable": True}
    if status is not None and 500 <= status < 600:
        return {"reason": "server_error", "retryable": True}
    return None


class MeliousProfile(ProviderProfile):
    """Melious' behaviour on top of the generic OpenAI-wire profile."""

    def create_client(self, **client_kwargs: Any) -> Any:
        if plugin_disabled():
            logger.info("melious plugin disabled by %s; using the standard client", DISABLE_ENV)
            return None
        return MeliousClient(**client_kwargs)

    def build_api_kwargs_extras(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Add the client-generated ``request_id`` body field (extra_body side).

        Hermes builds an API call's kwargs once and reuses them for that call's stream
        retries, so the id is stable across retries of the same logical request — the
        resend stays byte-identical and cache-eligible — and fresh for the next call.
        :meth:`MeliousClient.create` repeats the injection, so requests whose kwargs never
        pass through here (auxiliary calls, a route that resolves no profile) still carry
        an id.
        """
        if plugin_disabled():
            return {}, {}
        return {"request_id": new_request_id()}, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live catalog from ``{base}/models``, cached per (home, key fingerprint)."""
        if not self.supports_model_listing:
            return None
        requested = str(base_url or "").rstrip("/")
        effective = requested or BASE_URL
        # A route pointed somewhere that is not Melious is not ours to catalogue.
        if not effective.startswith(BASE_URL):
            return None
        key = str(api_key or "").strip() or api_key_from_env()
        if not key:
            return None

        scope = (SLUG, str(hermes_home().resolve()), _fingerprint(key))
        cached = _models_cache.get(scope)
        if cached and time.monotonic() - cached[0] < _MODELS_CACHE_TTL_S:
            return list(cached[1])

        items = _fetch_model_items(effective, key, timeout)
        if items is None:
            return None
        models = _model_ids(items)
        if not models:
            return None
        _models_cache[scope] = (time.monotonic(), tuple(models))
        return list(models)


def _fetch_model_items(base_url: str, api_key: str, timeout: float) -> list[Any] | None:
    """GET {base}/models and return its row list, or None on any failure."""
    import json
    import urllib.request

    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", f"hermes-melious-plugin/{PLUGIN_VERSION}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except Exception as exc:
        logger.debug("melious fetch_models failed: %s", exc)
        return None
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    return None


def _model_ids(items: list[Any]) -> list[str]:
    """Chat-capable model ids from a `/models` payload, order preserved."""
    try:
        from hermes_cli.chat_catalog import chat_catalog_ids

        ids = [m for m in chat_catalog_ids(items) if isinstance(m, str) and m.strip()]
        if ids:
            return ids
    except Exception:
        logger.debug("melious: chat_catalog_ids unavailable", exc_info=True)
    ids = []
    for row in items:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip():
            model_id = row["id"].strip()
            if model_id not in ids:
                ids.append(model_id)
    return ids


profile = MeliousProfile(
    name=SLUG,
    # The one wire-level correction to SPEC §0: a `custom:<name>` route resolves to the
    # bundled generic `custom` profile FIRST (`get_provider_profile` -> "named custom
    # routes share the generic wire policy unless a plugin explicitly registered that
    # route"), so the base-url prefix match in `_provider_supplied_client` is never
    # reached for `custom:melious` — name lookup succeeds before it. Claiming the route as
    # an alias is what makes this plugin's client and request-field hooks serve the
    # EXISTING `custom:melious` provider without changing `model.provider`.
    aliases=("custom:melious",),
    display_name=DISPLAY_NAME,
    description="Melious (api.melious.ai) — OpenAI-wire, with client-side request correlation",
    signup_url="https://melious.ai",
    env_vars=API_KEY_ENV_VARS,
    base_url=BASE_URL,
    auth_type="api_key",
    api_mode="chat_completions",
    supports_health_check=True,
    supports_model_listing=True,
    fallback_models=FALLBACK_MODELS,
    default_aux_model="deepseek-v4.1-flash",
    default_headers={SENTINEL_HEADER: SENTINEL_VALUE},
    # A dataclass FIELD on ProviderProfile, so it must be bound here: a subclass method of
    # the same name would be shadowed by the None the dataclass writes in __init__.
    classify_api_error=classify_melious_error,
)
