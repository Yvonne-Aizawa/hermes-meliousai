"""Offline contract tests for the Melious provider plugin — synthetic HTTP only."""

import contextlib
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from melious_provider import SENTINEL_VALUE  # noqa: E402


def completion(model):
    return {
        "id": "cmpl", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@contextlib.contextmanager
def mock_transport(handler):
    """Route the SDK's HTTP through *handler*.

    The openai SDK builds its httpx transport from the *call-time* global
    ``httpx._client.HTTPTransport``, so patching ``httpx.HTTPTransport`` alone has no
    effect — the real network client is still constructed (and the offline socket guard
    raises instead of the handler being consulted).
    """
    import httpx

    with patch("httpx._client.HTTPTransport", return_value=httpx.MockTransport(handler)):
        yield


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)

        patcher = patch.dict(os.environ, {
            "HERMES_HOME": self.home.name,
            "HERMES_CUSTOM_API_MELIOUS_AI_API_KEY": "test-key",
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("HERMES_MELIOUS_PLUGIN_DISABLE", None)

    def _client(self):
        from melious_provider import BASE_URL, profile
        return profile.create_client(base_url=BASE_URL, api_key="test-key", timeout=90)


class ProfileTests(BaseCase):
    def test_profile_identity_and_hooks(self):
        from melious_provider import profile
        self.assertEqual(profile.name, "melious")
        self.assertEqual(profile.base_url, "https://api.melious.ai/v1")
        self.assertEqual(profile.api_mode, "chat_completions")
        self.assertEqual(profile.auth_type, "api_key")
        self.assertEqual(profile.env_vars[0], "HERMES_CUSTOM_API_MELIOUS_AI_API_KEY")
        self.assertEqual(profile.display_name, "Melious")
        self.assertIn("glm-5.3-flash", profile.fallback_models)
        self.assertTrue(profile.supports_model_listing)
        self.assertTrue(profile.supports_health_check)
        # classify_api_error is a dataclass FIELD: it must be bound at construction, not
        # overridden as a method (the dataclass would shadow it with None).
        self.assertTrue(callable(profile.classify_api_error))

    def test_build_api_kwargs_extras_injects_a_uuid_request_id(self):
        from melious_provider import profile
        first_extra, first_top = profile.build_api_kwargs_extras(model="glm-5.3-flash")
        second_extra, _ = profile.build_api_kwargs_extras(model="glm-5.3-flash")
        self.assertEqual(first_top, {})
        self.assertEqual(set(first_extra), {"request_id"})
        uuid.UUID(first_extra["request_id"])
        self.assertNotEqual(first_extra["request_id"], second_extra["request_id"])

    def test_build_api_kwargs_extras_defers_to_the_kill_switch(self):
        from melious_provider import DISABLE_ENV, profile
        with patch.dict(os.environ, {DISABLE_ENV: "1"}):
            self.assertEqual(profile.build_api_kwargs_extras(model="glm-5.3-flash"), ({}, {}))

    def test_queue_classification(self):
        from melious_provider import classify_queue
        self.assertEqual(classify_queue(None), "unknown")
        self.assertEqual(classify_queue(1000, threshold=32000), "short")
        self.assertEqual(classify_queue(32000, threshold=32000), "long")
        self.assertEqual(classify_queue(86024, threshold=32000), "long")

    def test_classify_api_error_maps_melious_shapes(self):
        from melious_provider import profile
        hook = profile.classify_api_error
        self.assertEqual(
            hook(None, status_code=200, error_code="server_error",
                 message='the model did not start responding in time. please try again.'),
            {"reason": "server_error", "retryable": True})
        self.assertEqual(
            hook(None, status_code=None, error_code=None,
                 message="The model did not start responding in time. Please try again."),
            {"reason": "server_error", "retryable": True})
        self.assertEqual(hook(None, status_code=429, error_code="RATE_LIMIT_EXCEEDED"),
                         {"reason": "rate_limit"})
        self.assertEqual(hook(None, status_code=429), {"reason": "rate_limit"})
        self.assertEqual(hook(None, status_code=503), {"reason": "server_error", "retryable": True})
        # Declines anything it does not own, leaving the built-in pipeline in charge.
        self.assertIsNone(hook(None, status_code=401, error_code="invalid_api_key"))
        self.assertIsNone(hook(None, status_code=400, error_code="invalid_request_error"))
        self.assertIsNone(hook(None, status_code=200, error_code="content_policy"))

    def test_config_drift_reports_mismatch_and_agreement(self):
        import melious_provider as mp
        agreeing = {
            "melious_provider": {"abort_silent_after_s": 90, "retry_backoff_s": 10, "max_attempts": 2},
            "providers": {"custom:melious": {"stale_timeout_seconds": 90, "retry_after_cap_seconds": 10}},
            "agent": {"api_max_retries": 2},
            "fallback_providers": [{"provider": "custom:melious", "model": "deepseek-v4.1-flash"}],
        }
        self.assertEqual(mp.config_drift(agreeing), [])
        # The canonical spelling (model.provider: melious) mirrors just as well.
        canonical = {
            "melious_provider": {"abort_silent_after_s": 90, "max_attempts": 2},
            "providers": {"melious": {"stale_timeout_seconds": 90}},
            "agent": {"api_max_retries": 2},
            "fallback_providers": [{"provider": "melious", "model": "deepseek-v4.1-flash"}],
        }
        self.assertEqual(mp.config_drift(canonical), [])

        drifting = {
            "melious_provider": {"abort_silent_after_s": 300},
            "providers": {"custom:melious": {"stale_timeout_seconds": 90}},
            "agent": {"api_max_retries": 2},
            "fallback_providers": [],
        }
        lines = mp.config_drift(drifting)
        self.assertTrue(any("stale_timeout_seconds" in line for line in lines), lines)
        self.assertTrue(any("fallback_model" in line for line in lines), lines)
        # retry_backoff_s is plugin-ENFORCED (the retry_after clamp), never mirrored.
        self.assertFalse(any("retry_backoff_s" in line for line in lines), lines)


class ClientTests(BaseCase):
    def test_request_id_is_sent_on_every_call_and_is_unique(self):
        import httpx
        bodies: list = []
        headers: list = []

        def fake(request):
            bodies.append(json.loads(request.content))
            headers.append(dict(request.headers))
            return httpx.Response(200, json=completion("glm-5.3-flash"))

        with mock_transport(fake):
            client = self._client()
            for _ in range(2):
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        self.assertEqual(len(bodies), 2)
        ids = [body.get("request_id") for body in bodies]
        for request_id in ids:
            uuid.UUID(request_id)
        self.assertNotEqual(ids[0], ids[1])
        # The sentinel header is what proves the plugin's client is on the wire.
        self.assertEqual(headers[0].get("x-hermes-provider-plugin"), SENTINEL_VALUE)
        self.assertEqual(str(headers[0].get("authorization")), "Bearer test-key")

    def test_caller_supplied_request_id_and_extra_body_are_preserved(self):
        import httpx
        bodies: list = []

        def fake(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=completion("glm-5.3-flash"))

        with mock_transport(fake):
            client = self._client()
            client.chat.completions.create(
                model="glm-5.3-flash", messages=[{"role": "user", "content": "hi"}],
                extra_body={"request_id": "fixed-id", "response_format": {"type": "json_object"}})
            client.close()

        self.assertEqual(bodies[0]["request_id"], "fixed-id")
        self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})

    def test_streaming_tool_calls_pass_through_and_keep_the_response_object(self):
        import httpx

        def fake(request):
            delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "local_echo", "arguments": '{"text":"bonjour"}'}}]}
            chunk = {"id": "cmpl", "object": "chat.completion.chunk", "created": 0,
                     "model": "glm-5.3-flash",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls"}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream", "x-request-id": "upstream-1"},
                                  content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")

        with mock_transport(fake):
            client = self._client()
            stream = client.chat.completions.create(
                model="glm-5.3-flash", messages=[{"role": "user", "content": "echo"}],
                tools=[{"type": "function", "function": {"name": "local_echo", "parameters": {"type": "object"}}}],
                stream=True, stream_options={"include_usage": True})
            # Core reads Melious' X-Request-ID off the stream's `.response`; it must survive.
            self.assertEqual(stream.response.headers.get("x-request-id"), "upstream-1")
            chunks = list(stream)
            client.close()

        call = chunks[0].choices[0].delta.tool_calls[0]
        self.assertEqual(call.function.name, "local_echo")
        self.assertEqual(json.loads(call.function.arguments)["text"], "bonjour")

    def test_sdk_retries_are_disabled_and_upstream_id_is_kept(self):
        import httpx
        calls: list = []

        def fake(request):
            calls.append(request)
            return httpx.Response(500, headers={"x-request-id": "upstream-abc"},
                                  json={"error": {"message": "boom", "code": "server_error"}})

        with mock_transport(fake):
            client = self._client()
            with self.assertRaises(Exception) as caught:
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        self.assertEqual(len(calls), 1, "the SDK must not retry; retries belong to core")
        self.assertEqual(caught.exception.status_code, 500)
        # A response DID arrive: Melious' own id is the one to keep.
        self.assertEqual(getattr(caught.exception, "request_id", None), "upstream-abc")
        self.assertNotEqual(caught.exception._melious_client_request_id, "upstream-abc")

    def test_no_plugin_ledger_line_when_a_response_arrived(self):
        import httpx
        import melious_provider as mp

        def fake(request):
            return httpx.Response(500, headers={"x-request-id": "upstream-abc"},
                                  json={"error": {"message": "boom"}})

        with mock_transport(fake):
            client = self._client()
            with self.assertRaises(Exception):
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        self.assertFalse(mp.ledger_path().exists(),
                         "core records response-bearing failures; the plugin must not duplicate them")

    def test_client_id_is_stamped_when_no_response_arrives(self):
        import httpx

        def boom(request):
            raise httpx.ConnectError("no route to host", request=request)

        with mock_transport(boom):
            client = self._client()
            with self.assertRaises(Exception) as caught:
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        error = caught.exception
        self.assertIsNone(getattr(error, "response", None))
        # The connect-timeout case: this is the only place an id can exist.
        uuid.UUID(error.request_id)
        self.assertEqual(error.request_id, error._melious_client_request_id)

    def test_connect_failure_writes_one_ledger_line_with_the_client_id(self):
        """SPEC §P6: the no-response case is the plugin's line to write."""
        import httpx
        import melious_provider as mp

        def boom(request):
            raise httpx.ConnectError("no route to host", request=request)

        with mock_transport(boom):
            client = self._client()
            with self.assertRaises(Exception) as caught:
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        records = [json.loads(line) for line in mp.ledger_path().read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["provider_request_id"], caught.exception.request_id)
        self.assertEqual(record["request_id"], caught.exception.request_id)
        self.assertEqual(record["source"], "melious-plugin")
        self.assertEqual(record["reason"], "provider_connect_timeout")
        self.assertEqual(record["model"], "glm-5.3-flash")
        self.assertEqual(record["queue"], "short")
        self.assertEqual(record["stream"]["chunks"], 0)
        self.assertIsNone(record["stream"]["http_status"])
        self.assertIsNotNone(record["stream"]["no_output_s"])
        uuid.UUID(record["provider_request_id"])

    def test_non_mapping_extra_body_is_left_untouched(self):
        from melious_provider import ensure_request_id
        kwargs = {"extra_body": "not-a-mapping"}
        self.assertIsNone(ensure_request_id(kwargs))
        self.assertEqual(kwargs["extra_body"], "not-a-mapping")


class StreamFailureTests(BaseCase):
    """The in-stream failure path the stream proxy owns (retired core-patch territory)."""

    def _client(self):
        from melious_provider import BASE_URL, profile
        return profile.create_client(base_url=BASE_URL, api_key="test-key", timeout=90)

    @staticmethod
    def _sse(*frames):
        return ("data: " + json.dumps(frame) + "\n\n" for frame in frames)

    def test_in_stream_error_ledgers_both_ids_and_clamps_retry_after(self):
        import httpx
        import melious_provider as mp

        def fake(request):
            frame = {"error": {"message": "The model did not start responding in time. Please try again.",
                               "type": "server_error", "code": "server_error", "retry_after": 30}}
            payload = ("data: " + json.dumps(frame) + "\n\ndata: [DONE]\n\n").encode()
            return httpx.Response(200, headers={"content-type": "text/event-stream",
                                                "x-request-id": "upstream-240s"}, content=payload)

        with mock_transport(fake):
            client = self._client()
            stream = client.chat.completions.create(
                model="glm-5.3-flash", messages=[{"role": "user", "content": "hi"}], stream=True)
            self.assertIsInstance(stream, mp.MeliousStreamProxy)
            # Core's header capture keeps working through the proxy.
            self.assertEqual(stream.response.headers.get("x-request-id"), "upstream-240s")
            with self.assertRaises(Exception) as caught:
                list(stream)
            client.close()

        exc = caught.exception
        # The upstream id is stamped where stock core's debug dump reads it.
        self.assertEqual(getattr(exc, "request_id", None), "upstream-240s")
        self.assertEqual(exc._melious_client_request_id, client.last_request_id)
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            carrier = body.get("error") if isinstance(body.get("error"), dict) else body
            if "retry_after" in carrier:
                self.assertEqual(float(carrier["retry_after"]), 10.0, "templated 30s must clamp to 10s")

        records = [json.loads(line) for line in mp.ledger_path().read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["reason"], "provider_stream_error")
        self.assertEqual(record["source"], "melious-plugin")
        self.assertEqual(record["provider_request_id"], "upstream-240s")
        self.assertEqual(record["request_id"], client.last_request_id)
        self.assertEqual(record["queue"], "short")
        self.assertEqual(record["model"], "glm-5.3-flash")
        self.assertEqual(record["stream"]["chunks"], 0)
        self.assertIsNone(record["stream"]["ttfb_s"])
        self.assertIsNotNone(record["stream"]["no_output_s"])
        self.assertEqual(record["stream"]["http_status"], 200)
        self.assertEqual(record["stream"]["open_headers"].get("x-request-id"), "upstream-240s")

    def test_partial_output_failure_records_ttfb_not_no_output(self):
        import httpx
        import melious_provider as mp
        good = {"id": "cmpl", "object": "chat.completion.chunk", "created": 0, "model": "glm-5.3-flash",
                "choices": [{"index": 0, "delta": {"content": "par"}, "finish_reason": None}]}
        bad = {"error": {"message": "mid-stream drop", "type": "server_error", "code": "server_error"}}

        def fake(request):
            frames = self._sse(good, bad)
            content = "".join(frames).encode() + b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream",
                                                "x-request-id": "upstream-mid"}, content=content)

        with mock_transport(fake):
            client = self._client()
            stream = client.chat.completions.create(
                model="glm-5.3-flash", messages=[{"role": "user", "content": "hi"}], stream=True)
            with self.assertRaises(Exception):
                list(stream)
            client.close()

        records = [json.loads(line) for line in mp.ledger_path().read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        stream_block = records[0]["stream"]
        self.assertEqual(stream_block["chunks"], 1)
        self.assertIsNotNone(stream_block["ttfb_s"])
        self.assertIsNone(stream_block["no_output_s"], "output WAS seen; there is no 'no output' span")

    def test_request_time_error_body_retry_after_is_clamped(self):
        import httpx
        import melious_provider as mp

        def fake(request):
            return httpx.Response(503, json={"error": {"message": "boom", "code": "provider_error",
                                                        "retry_after": 45}})

        with mock_transport(fake):
            client = self._client()
            with self.assertRaises(Exception) as caught:
                client.chat.completions.create(model="glm-5.3-flash",
                                               messages=[{"role": "user", "content": "hi"}])
            client.close()

        self.assertEqual(caught.exception.status_code, 503)
        # The SDK may keep the envelope nested or flatten it; the clamp handles both.
        body = caught.exception.body
        carrier = body.get("error") if isinstance(body.get("error"), dict) else body
        self.assertEqual(float(carrier["retry_after"]), 10.0, body)
        # A response DID arrive: no plugin ledger line for this one.
        self.assertFalse(mp.ledger_path().exists())

    def test_retry_after_clamp_shapes_and_exemptions(self):
        from melious_provider import clamp_retry_after

        class Err(Exception):
            pass

        nested = Err()
        nested.body = {"error": {"code": "server_error", "retry_after": 30}}
        clamp_retry_after(nested)
        self.assertEqual(nested.body["error"]["retry_after"], 10)

        flat = Err()
        flat.body = {"code": "server_error", "retry_after": 45}
        clamp_retry_after(flat)
        self.assertEqual(flat.body["retry_after"], 10)

        below = Err()
        below.body = {"error": {"code": "server_error", "retry_after": 5}}
        clamp_retry_after(below)
        self.assertEqual(below.body["error"]["retry_after"], 5, "never raised, only capped")

        rate = Err()
        rate.body = {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_exceeded", "retry_after": 30}}
        clamp_retry_after(rate)
        self.assertEqual(rate.body["error"]["retry_after"], 30, "rate limits keep the real cooldown")

        no_body = Err()
        clamp_retry_after(no_body)  # must not raise
        weird = Err()
        weird.body = "not-a-mapping"
        clamp_retry_after(weird)  # must not raise


    def test_vision_routing_facts(self):
        """The account's chat models are text-only on Melious; one vision model exists.

        models.dev marks the GLM-5.3 family multimodal, so stock Hermes routes image
        analysis (and native embeds) to the main chat model and the image is silently
        dropped. The plugin declares the provider's own catalog truth.
        """
        import melious_provider as mp

        # Catalog unreachable -> the verified pin is the fallback (live discovery is
        # covered by CatalogTests; here we pin the fallback contract).
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertEqual(mp.profile.default_vision_model(), mp.VISION_MODEL)
        self.assertEqual(mp.VISION_MODEL, "mistral-small-3.2-24b-instruct")
        caps = mp.MODEL_CAPABILITIES
        for blind in ("glm-5.3", "glm-5.3-flash", "deepseek-v4.1-flash"):
            self.assertIs(caps[blind]["supports_vision"], False, blind)
        self.assertIs(caps[mp.VISION_MODEL]["supports_vision"], True)
        self.assertEqual(caps[mp.VISION_MODEL]["context_window"], 131072)
        # The declarations ride on the profile (config model_overrides still win over them).
        self.assertIs(mp.profile.model_capabilities[mp.VISION_MODEL]["supports_vision"], True)


class CatalogTests(BaseCase):
    def setUp(self):
        super().setUp()
        import melious_provider as mp

        mp._models_cache.clear()
        mp._catalog_rows_cache.clear()
        snapshot = dict(mp.profile.model_capabilities)
        self.addCleanup(lambda: (mp.profile.model_capabilities.clear(),
                                 mp.profile.model_capabilities.update(snapshot)))

        def _drop_memo():
            if hasattr(mp.profile, "_vision_model_memo"):
                del mp.profile._vision_model_memo

        _drop_memo()
        self.addCleanup(_drop_memo)

    @staticmethod
    def _urlopen_response(payload):
        class Resp:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Resp()

    def test_fetch_models_reads_the_endpoint_and_caches_per_key(self):
        import melious_provider as mp
        seen: list = []

        def fake_urlopen(request, timeout=None):
            seen.append((str(request.full_url), dict(request.headers)))
            return self._urlopen_response({"data": [{"id": "glm-5.3-flash"}, {"id": "deepseek-v4.1-flash"}]})

        os.environ["HERMES_CUSTOM_API_MELIOUS_AI_API_KEY"] = "key-one"
        with patch("urllib.request.urlopen", fake_urlopen):
            first = mp.profile.fetch_models(base_url=mp.BASE_URL)
            second = mp.profile.fetch_models(base_url=mp.BASE_URL)
            os.environ["HERMES_CUSTOM_API_MELIOUS_AI_API_KEY"] = "key-two"
            third = mp.profile.fetch_models(base_url=mp.BASE_URL)

        self.assertEqual(first, ["glm-5.3-flash", "deepseek-v4.1-flash"])
        self.assertEqual(second, first)
        self.assertEqual(third, first)
        self.assertEqual(len(seen), 2, "the second call is served from cache")
        self.assertEqual(seen[0][0], mp.BASE_URL + "/models?include_meta=true")
        self.assertEqual(seen[0][1].get("Authorization"), "Bearer key-one")
        # The raw key never lands in the cache structure.
        self.assertNotIn("key-one", repr(mp._models_cache))
        self.assertNotIn("key-two", repr(mp._models_cache))

    def test_default_vision_model_is_discovered_live(self):
        import melious_provider as mp
        payload = {"data": [
            {"id": "glm-5.3", "_meta": {"capabilities": {"vision": False, "function_calling": True},
                                         "context_length": 1000000}},
            {"id": "qwen3-vl-8b", "_meta": {"capabilities": {"vision": True, "function_calling": True},
                                           "context_length": 262144}},
        ]}

        with patch("urllib.request.urlopen", lambda request, timeout=None: self._urlopen_response(payload)):
            discovered = mp.profile.default_vision_model()
        self.assertEqual(discovered, "qwen3-vl-8b", "must come from the LIVE catalog, not the static pin")
        # Memoized: a second call does not refetch.
        with patch("urllib.request.urlopen", side_effect=AssertionError("must be memoized")):
            self.assertEqual(mp.profile.default_vision_model(), "qwen3-vl-8b")
        # The same fetch folded the catalog truth into the capability declarations.
        self.assertIs(mp.profile.model_capabilities["glm-5.3"]["supports_vision"], False)
        self.assertEqual(mp.profile.model_capabilities["glm-5.3"]["context_window"], 1000000)
        self.assertIs(mp.profile.model_capabilities["qwen3-vl-8b"]["supports_vision"], True)
        self.assertEqual(mp.profile.model_capabilities["qwen3-vl-8b"]["context_window"], 262144)

    def test_default_vision_model_falls_back_when_the_catalog_is_unreachable(self):
        import melious_provider as mp
        with patch("urllib.request.urlopen", side_effect=OSError("listing down")):
            self.assertEqual(mp.profile.default_vision_model(), mp.VISION_MODEL)

    def test_live_context_length_comes_from_the_catalog(self):
        import melious_provider as mp
        payload = {"data": [
            {"id": "glm-5.3", "_meta": {"capabilities": {"vision": False}, "context_length": 1000000}},
            {"id": "no-window", "_meta": {"capabilities": {}}},
        ]}
        with patch("urllib.request.urlopen", lambda request, timeout=None: self._urlopen_response(payload)):
            self.assertEqual(mp.profile.live_context_length("glm-5.3"), 1000000)
            self.assertIsNone(mp.profile.live_context_length("no-window"))
            self.assertIsNone(mp.profile.live_context_length("not-in-catalog"))

    def test_fetch_models_declines_foreign_origins_and_failures(self):
        import melious_provider as mp
        mp._models_cache.clear()
        with patch("urllib.request.urlopen", side_effect=AssertionError("must not be called")):
            self.assertIsNone(mp.profile.fetch_models(base_url="https://evil.invalid/v1"))

        def broken(request, timeout=None):
            raise OSError("network down")

        with patch("urllib.request.urlopen", broken):
            self.assertIsNone(mp.profile.fetch_models(base_url=mp.BASE_URL))

    def test_fetch_models_returns_none_without_a_key(self):
        import melious_provider as mp
        os.environ["HERMES_CUSTOM_API_MELIOUS_AI_API_KEY"] = ""
        os.environ.pop("MELIOUS_API_KEY", None)
        self.assertIsNone(mp.profile.fetch_models(base_url=mp.BASE_URL))


if __name__ == "__main__":
    unittest.main()
