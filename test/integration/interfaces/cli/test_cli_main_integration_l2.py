"""L2 integration tests for CLI main entrypoint."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_cli(args):
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    env.pop("RELIC_CONFIG_PATH", None)
    return subprocess.run(
        [sys.executable, "-m", "relic.interfaces.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=60,
    )


def test_cli_main_help():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "usage" in combined.lower()
    assert "[N]" in combined
    assert "runtime_unbound" in combined
    assert "relic optimization calibrate --config" not in combined


def test_cli_subcommand_help():
    for subcmd in ["optimization", "stochastic", "params"]:
        result = _run_cli([subcmd, "--help"])
        assert result.returncode == 0


@pytest.mark.parametrize(
    ("command_args", "output_name"),
    [
        (["calibrate", "--method", "differential_evolution"], "calibrate-output"),
        (
            ["assimilate", "--method", "enkf", "--ensemble-size", "20", "--inflation-factor", "1.05"],
            "assimilate-output",
        ),
        (["sensitivity", "--method", "morris", "--n-trajectories", "10"], "sensitivity-output"),
        (
            ["uncertainty", "--method", "ensemble", "--n-samples", "64", "--sampling", "lhs"],
            "uncertainty-output",
        ),
    ],
)
def test_optimization_process_commands_fail_closed_without_outputs(
    command_args,
    output_name,
    tmp_path,
):
    output_path = tmp_path / output_name
    result = _run_cli(
        [
            "optimization",
            *command_args,
            "--output",
            str(output_path),
        ]
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "[N]" in combined
    assert "runtime_unbound" in combined
    assert not output_path.exists()

SDE_SECTION = (
    "sde:\n"
    "  process: stochastic_air_temperature\n"
    "  initial_value: 1.0\n"
    "  samples: 2\n"
    "  steps: 3\n"
    "  dt_seconds: 3600.0\n"
    "  seed: 7\n"
)


@pytest.mark.parametrize("runtime_config", [None, "examples/simulation_minimal.yaml"])
def test_stochastic_sde_reads_its_own_config_file(tmp_path, runtime_config):
    """`stochastic sde --config` names the sde file; a global --config stays the RELICConfig."""
    sde_config = tmp_path / "sde.yaml"
    sde_config.write_text(SDE_SECTION, encoding="utf-8")
    output = tmp_path / "sde_out"
    global_args = [] if runtime_config is None else ["--config", runtime_config]

    result = _run_cli(
        [*global_args, "stochastic", "--format", "json", "--output", str(output),
         "sde", "--config", str(sde_config)]
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (tmp_path / "sde_out.json").is_file()


def test_importing_cli_main_leaves_root_logging_unconfigured():
    """Importing the CLI module leaves the root logger level and handler count unchanged."""
    import json

    probe = "\n".join(
        (
            "import json, logging",
            "root = logging.getLogger()",
            "before = [root.level, len(root.handlers)]",
            "import relic.interfaces.cli.main",
            "after = [root.level, len(root.handlers)]",
            "print(json.dumps({'before': before, 'after': after}))",
        )
    )
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=120,
        check=True,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["after"] == observed["before"], (
        "Importing relic.interfaces.cli.main changed the root logger [level, handler count] "
        f"from {observed['before']} to {observed['after']}\nstderr:\n" + result.stderr[-2000:]
    )


def test_cli_main_configures_logging_when_a_command_runs(tmp_path):
    """main() configures the root logger once it runs a command: records are formatted."""
    missing = tmp_path / "missing_runtime.yaml"
    result = _run_cli(["--config", str(missing), "stochastic"])

    # A named configuration file that does not exist fails the command; the
    # formatted error record shows that main() configured the root logger.
    assert result.returncode == 1
    assert " - ERROR - Failed to load configuration: Configuration file not found" in (
        result.stderr
    ), result.stderr[-2000:]
    assert str(missing) in result.stderr, result.stderr[-2000:]
