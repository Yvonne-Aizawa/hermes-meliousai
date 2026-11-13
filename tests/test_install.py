"""Offline tests for install.py: placement, idempotence, refusals, uninstall, doctor."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import install as installer  # noqa: E402

CONFIG_BODY = """\
model:
  provider: "custom:melious"
  default: "glm-5.3-flash"
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
"""


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.home_path = Path(self.home.name) / "hermes"
        self.config = self.home_path / "config.yaml"
        self.config.parent.mkdir(parents=True)
        self.config.write_text(CONFIG_BODY)
        self.target = self.home_path / "plugins" / "model-providers" / "melious"

    def test_install_places_managed_files_and_touches_nothing_else(self):
        before = self.config.read_text()
        installed = installer.install(ROOT, self.home_path, update=False, force=False)
        self.assertEqual(installed, self.target)
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), sorted(installer.MANAGED_FILES))
        self.assertEqual(self.config.read_text(), before)
        self.assertEqual(sorted(p.name for p in self.home_path.iterdir()), ["config.yaml", "plugins"])
        # No staging directories left behind.
        self.assertEqual([p.name for p in self.target.parent.iterdir()], ["melious"])

    def test_second_install_is_refused_without_update(self):
        installer.install(ROOT, self.home_path, update=False, force=False)
        with self.assertRaises(SystemExit):
            installer.install(ROOT, self.home_path, update=False, force=False)

    def test_update_replaces_managed_files_but_refuses_unexpected_ones(self):
        installer.install(ROOT, self.home_path, update=False, force=False)
        installer.install(ROOT, self.home_path, update=True, force=False)

        (self.target / "stray.py").write_text("unexpected\n")
        with self.assertRaises(SystemExit):
            installer.install(ROOT, self.home_path, update=True, force=False)
        self.assertTrue((self.target / "stray.py").exists())

        installer.install(ROOT, self.home_path, update=False, force=True)
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), sorted(installer.MANAGED_FILES))

    def test_update_survives_the_import_bytecode_directory(self):
        installer.install(ROOT, self.home_path, update=False, force=False)
        # Hermes imports the installed package, so CPython writes __pycache__ there.
        cache = self.target / "__pycache__"
        cache.mkdir()
        (cache / "melious_provider.cpython-314.pyc").write_bytes(b"\x00")
        installer.install(ROOT, self.home_path, update=True, force=False)
        self.assertEqual(sorted(p.name for p in self.target.iterdir()), sorted(installer.MANAGED_FILES))

    def test_uninstall_removes_only_the_plugin_and_is_idempotent(self):
        installer.install(ROOT, self.home_path, update=False, force=False)
        before = self.config.read_text()
        self.assertTrue(installer.uninstall(self.home_path, force=False))
        self.assertFalse(self.target.exists())
        self.assertEqual(self.config.read_text(), before)
        self.assertFalse(installer.uninstall(self.home_path, force=False))

    def test_doctor_reports_drift_and_agreement(self):
        import melious_provider as mp
        installer.install(ROOT, self.home_path, update=False, force=False)

        agreeing = {
            "melious_provider": {"abort_silent_after_s": 90, "retry_backoff_s": 10, "max_attempts": 2},
            "providers": {"custom:melious": {"stale_timeout_seconds": 90, "retry_after_cap_seconds": 10}},
            "agent": {"api_max_retries": 2},
            "fallback_providers": [{"provider": "custom:melious", "model": "deepseek-v4.1-flash"}],
        }
        with patch.object(mp, "_load_config", lambda: agreeing):
            self.assertEqual(installer.doctor(self.home_path), 0)

        with patch.object(mp, "_load_config", lambda: {"agent": {"api_max_retries": 2}}):
            self.assertEqual(installer.doctor(self.home_path), 1)

    def test_verify_flags_a_missing_install(self):
        self.assertEqual(installer.verify(self.home_path), 2)


if __name__ == "__main__":
    unittest.main()
