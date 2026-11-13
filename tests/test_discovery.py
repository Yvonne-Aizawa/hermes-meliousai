"""The registration seams Hermes actually consults, exercised against the real core."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)

        def isolate():
            patcher = patch.dict(os.environ, {
                "HERMES_HOME": self.home.name,
                "HERMES_CUSTOM_API_MELIOUS_AI_API_KEY": "test-key",
            })
            patcher.start()
            os.environ.pop("HERMES_MELIOUS_PLUGIN_DISABLE", None)
            self.addCleanup(patcher.stop)

        isolate()
        import melious_provider as mp
        from providers import register_provider

        register_provider(mp.profile)
        self.mp = mp

    def test_profile_resolves_by_name_and_by_base_url(self):
        from providers import get_provider_profile

        self.assertIs(get_provider_profile(self.mp.SLUG), self.mp.profile)
        # `custom:melious` resolves to the generic `custom` profile UNLESS this plugin
        # claims the route — which it does via aliases. This is the path that matters,
        # because model.provider is not changing.
        resolved = get_provider_profile("custom:melious")
        self.assertIs(resolved, self.mp.profile)
        self.assertEqual(str(resolved.base_url).rstrip("/"), self.mp.BASE_URL)

    def test_provider_client_seam_returns_the_plugin_client(self):
        from agent.agent_runtime_helpers import _provider_supplied_client

        kwargs = {"base_url": self.mp.BASE_URL, "api_key": "test-key", "timeout": 90}
        for label, agent in (
            (self.mp.SLUG, SimpleNamespace(provider=self.mp.SLUG)),
            ("custom:melious", SimpleNamespace(provider="custom:melious")),
        ):
            with self.subTest(provider=label):
                client = _provider_supplied_client(agent, dict(kwargs))
                self.assertIsNotNone(client, f"{label} fell through to the standard client")
                self.assertEqual(type(client).__name__, "MeliousClient")
                self.assertTrue(getattr(client, "HERMES_SKIP_ASYNC_WRAP", False))
                client.close()

    def test_kill_switch_restores_the_standard_client(self):
        from agent.agent_runtime_helpers import _provider_supplied_client

        kwargs = {"base_url": self.mp.BASE_URL, "api_key": "test-key", "timeout": 90}
        with patch.dict(os.environ, {self.mp.DISABLE_ENV: "1"}):
            self.assertIsNone(_provider_supplied_client(SimpleNamespace(provider=self.mp.SLUG), dict(kwargs)))
            self.assertIsNone(_provider_supplied_client(SimpleNamespace(provider="custom:melious"), dict(kwargs)))

    def test_registered_profile_is_mirrored_into_the_auth_registry(self):
        """A profile with a non-empty env_vars row is what `hermes auth`/`--provider` accept."""
        from providers import list_providers

        names = {getattr(p, "name", "") for p in list_providers()}
        self.assertIn(self.mp.SLUG, names)


if __name__ == "__main__":
    unittest.main()
