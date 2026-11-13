"""Run the offline test suite: installed Hermes, disposable home, no live network.

Uses the Hermes *runtime* interpreter — the PM-managed dependency environment that has
``openai``/``httpx`` — rather than whichever python invoked this script. Point
``HERMES_SOURCE`` at the checkout when it is not importable from that interpreter.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def _source_tree() -> str:
    source = os.environ.get("HERMES_SOURCE")
    if source:
        return source
    spec = importlib.util.find_spec("providers")
    if spec is None:
        raise SystemExit("Use the Hermes Python interpreter or set HERMES_SOURCE to its source tree")
    return str(Path(spec.origin).resolve().parents[1])


def _runtime_interpreter(source: str) -> str:
    """The python that actually runs Hermes (PM's committed environment), if findable."""
    sys.path.insert(0, source)
    try:
        from pm import environments

        venv = environments.committed_venv(Path(source))
        if venv is not None:
            candidate = Path(venv) / "bin" / "python3"
            if candidate.exists():
                return str(candidate)
    except Exception as exc:
        print(f"note: could not resolve Hermes' dependency environment ({exc})", flush=True)
    return sys.executable


def main() -> int:
    source = _source_tree()
    interpreter = _runtime_interpreter(source)
    if Path(interpreter).resolve() != Path(sys.executable).resolve():
        # Re-exec under the runtime interpreter so openai/httpx are importable.
        os.execv(interpreter, [interpreter, str(Path(__file__)), *sys.argv[1:]])

    with tempfile.TemporaryDirectory(prefix=".offline-", dir=ROOT) as folder:
        home = Path(folder) / "user"
        home.mkdir()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "HERMES_HOME": str(home / "hermes"),
            "TMPDIR": folder,
            "PYTHONPATH": os.pathsep.join([source, str(ROOT)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_NO_UPDATE_CHECK": "1",
        }
        code = """
import socket, unittest, sys
from unittest.mock import patch
with patch.object(socket.socket, 'connect', side_effect=AssertionError('Live network forbidden in offline tests')):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
sys.exit(not result.wasSuccessful())
"""
        print("OFFLINE: synthetic HTTP only; isolated HOME/HERMES_HOME; installed Hermes contract tests", flush=True)
        return subprocess.call([interpreter, "-c", code], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
