"""L3 system tests for CLI entrypoint and parameter commands."""

from pathlib import Path
import os
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_cli(args, extra_env=None):
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "RELIC_SKIP_CALIBRATED": "1",
        "RELIC_SKIP_LEGACY": "1",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "relic.interfaces.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_main_help_system():
    result = _run_cli(["--help"])
    assert result.returncode == 0


def test_cli_params_list_system():
    result = _run_cli(["params", "list", "--namespace", "biosphere.phenology"])
    assert result.returncode == 0


def test_cli_params_get_system():
    result = _run_cli(["params", "get", "biosphere.photosynthesis.vcmax25"])
    assert result.returncode == 0
    combined = (result.stdout + result.stderr).lower()
    assert "vcmax25" in combined
