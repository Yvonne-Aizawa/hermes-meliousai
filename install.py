"""Install, update, verify or remove the Melious provider plugin in a Hermes home.

The plugin is three files placed at ``<home>/plugins/model-providers/melious/``, where
Hermes' provider discovery picks them up. This script never reads or writes
``config.yaml``, ``auth.json`` or ``.env``: the tuned core settings the plugin depends on
live in ``config.yaml`` and are checked, not written (see README.md, and ``--doctor``).

    python3 install.py --home "${HERMES_HOME:-$HOME/.hermes}"            # install
    python3 install.py --home ... --update                              # replace
    python3 install.py --home ... --uninstall                           # remove
    python3 install.py --home ... --doctor                              # policy vs core config
    python3 install.py --home ... --verify                              # plugin really in use
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path

MANAGED_FILES = ("__init__.py", "plugin.yaml", "melious_provider.py")
PLUGIN_DIR_NAME = "melious"

# Hermes imports the installed directory, so CPython writes bytecode there. It is not an
# operator's file: ignoring it keeps --update / --uninstall working after a normal import.
_IGNORED_ENTRIES = {"__pycache__"}


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _target(home: Path) -> Path:
    return home / "plugins" / "model-providers" / PLUGIN_DIR_NAME


def _unexpected(target: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in target.iterdir()
        if entry.name not in _IGNORED_ENTRIES
        and (entry.name not in MANAGED_FILES or not entry.is_file() or entry.is_symlink())
    )


def install(source: Path, home: Path, *, update: bool, force: bool) -> Path:
    """Copy MANAGED_FILES into <home>/plugins/model-providers/melious/ and return it."""
    target = _target(home)
    if target.exists() or target.is_symlink():
        if not (update or force):
            raise SystemExit(
                "plugin already exists; use --update after reviewing it, or --force to replace it"
            )
        if not force:
            if target.is_symlink() or not target.is_dir():
                raise SystemExit("existing plugin is not a regular directory; use --force to replace it")
            unexpected = _unexpected(target)
            if unexpected:
                raise SystemExit(
                    "existing plugin has unexpected files; refusing update: " + ", ".join(unexpected)
                )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{PLUGIN_DIR_NAME}.install-{uuid.uuid4().hex}"
    try:
        staging.mkdir(mode=0o700)
        for name in MANAGED_FILES:
            shutil.copy2(source / name, staging / name)
        if target.exists() or target.is_symlink():
            _remove(target)
        staging.replace(target)
    finally:
        _remove(staging)
    return target


def uninstall(home: Path, *, force: bool) -> bool:
    """Remove the installed plugin directory; True when something was removed."""
    target = _target(home)
    if not (target.exists() or target.is_symlink()):
        print(f"Nothing to uninstall at {target}")
        return False
    if target.is_symlink() or not target.is_dir():
        raise SystemExit("installed plugin is not a regular directory; inspect and remove it by hand")
    unexpected = _unexpected(target)
    if unexpected and not force:
        raise SystemExit("installed plugin has unexpected files; refusing to remove: " + ", ".join(unexpected))
    shutil.rmtree(target)
    print(f"Removed {target}")
    print("config.yaml was never touched by this plugin; Hermes talks to Melious directly again.")
    return True


def _load_plugin_module(home: Path):
    """Import the plugin module (needs the Hermes source tree on sys.path)."""
    os.environ["HERMES_HOME"] = str(home)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import melious_provider  # noqa: PLC0415 — deferred: needs the Hermes interpreter

    return melious_provider


def doctor(home: Path) -> int:
    """Print the plugin's declared policy and any drift from the mirrored core keys."""
    try:
        module = _load_plugin_module(home)
    except Exception as exc:
        print(f"cannot import the plugin ({exc})")
        print("run this with the Hermes Python interpreter (the one `hermes` uses)")
        return 2

    print(f"melious_provider policy (HERMES_HOME={home})")
    for key, value in module.plugin_settings().items():
        print(f"  {key}: {value}")

    drift = module.config_drift()
    if not drift:
        print("OK: the core settings the plugin mirrors match its declared policy")
        return 0
    print("drift — the plugin does NOT write config.yaml, so reconcile these by hand:")
    for line in drift:
        print(f"  - {line}")
    return 1


def verify(home: Path) -> int:
    """Assert Hermes really routes the Melious route through this plugin's client.

    Offline: no inference call is made. Checks that (a) the profile is registered and
    (b) the provider-client seam returns a MeliousClient both by provider name and by
    base-URL prefix match — the path ``custom:melious`` actually takes — and that the kill
    switch restores the standard client.
    """
    from types import SimpleNamespace

    try:
        module = _load_plugin_module(home)
    except Exception as exc:
        print(f"cannot import the plugin ({exc})")
        print("run this with the Hermes Python interpreter (the one `hermes` uses)")
        return 2

    target = _target(home)
    if not (target / "__init__.py").is_file():
        print(f"not installed at {target}; run install first")
        return 2

    try:
        from agent.agent_runtime_helpers import _provider_supplied_client
        from providers import get_provider_profile
    except Exception as exc:
        print(f"Hermes internals not importable ({exc}); run this with the Hermes interpreter")
        return 2

    failures: list[str] = []

    profile = get_provider_profile(module.SLUG)
    if profile is None:
        failures.append(f"profile {module.SLUG!r} is not registered (restart required after install?)")
    else:
        if profile.base_url != module.BASE_URL:
            failures.append(f"profile base_url is {profile.base_url!r}, expected {module.BASE_URL!r}")
        if profile.create_client.__func__ is __import__(
            "providers.base", fromlist=["ProviderProfile"]
        ).ProviderProfile.create_client:
            failures.append("profile does not override create_client()")

    kwargs = {"base_url": module.BASE_URL, "api_key": "offline-verify", "timeout": 90}
    for label, agent in (
        (f"provider={module.SLUG}", SimpleNamespace(provider=module.SLUG)),
        ("provider=custom:melious (base-url match)", SimpleNamespace(provider="custom:melious")),
    ):
        client = _provider_supplied_client(agent, dict(kwargs))
        if client is None or type(client).__name__ != "MeliousClient":
            failures.append(f"{label}: provider-client seam returned {client!r}, not a MeliousClient")
        else:
            print(f"OK  {label} -> {type(client).__name__}")
            client.close()

    os.environ[module.DISABLE_ENV] = "1"
    try:
        disabled = _provider_supplied_client(SimpleNamespace(provider=module.SLUG), dict(kwargs))
        if disabled is not None:
            failures.append(f"{module.DISABLE_ENV}=1 did not fall back to the standard client")
        else:
            print(f"OK  {module.DISABLE_ENV}=1 -> standard client")
    finally:
        os.environ.pop(module.DISABLE_ENV, None)

    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("OK: the Melious route is served by this plugin's client")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home", required=True, type=Path,
                        help='explicit Hermes home (for example "$HERMES_HOME")')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true",
                      help="replace an existing install containing only managed files")
    mode.add_argument("--force", action="store_true",
                      help="replace the plugin directory even if it contains unexpected files")
    mode.add_argument("--uninstall", action="store_true", help="remove the installed plugin")
    mode.add_argument("--doctor", action="store_true",
                      help="print the declared policy and any drift from the mirrored core keys")
    mode.add_argument("--verify", action="store_true",
                      help="assert Hermes routes the Melious route through this plugin's client")
    args = parser.parse_args()

    home = args.home.expanduser().resolve()
    if home.exists() and not home.is_dir():
        parser.error("--home must name a directory")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)

    if args.doctor:
        return doctor(home)
    if args.verify:
        return verify(home)
    if args.uninstall:
        uninstall(home, force=args.force)
        return 0

    source = Path(__file__).resolve().parent
    target = install(source, home, update=args.update, force=args.force)
    action = "Updated" if (args.update or args.force) else "Installed"
    print(f"{action} the Melious provider plugin in {target}")
    print("Restart Hermes (or start a new session) so the plugin code is loaded.")
    print("No config.yaml, auth.json or .env file was read or changed.")
    drift = None
    try:
        drift = _load_plugin_module(home).config_drift()
    except Exception:
        pass
    if drift:
        print("Note — the core settings this plugin mirrors need attention (see --doctor):")
        for line in drift:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
