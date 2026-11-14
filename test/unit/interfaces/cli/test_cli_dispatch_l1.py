"""L1 tests for CLI dispatch behavior."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pytest

from relic.interfaces.cli import optimization as opt_cli
from relic.interfaces.cli import stochastic as stoch_cli
from relic.interfaces.cli import data as data_cli


def test_optimization_execute_requires_subcommand(capsys):
    args = argparse.Namespace(optimization_command=None)
    assert opt_cli.execute(args, {}) is False
    output = capsys.readouterr().out
    assert "[N]" in output
    assert "runtime_unbound" in output
    assert "Available commands:" not in output


def test_stochastic_execute_requires_subcommand():
    args = argparse.Namespace(stochastic_command=None)
    assert stoch_cli.execute(args, {}) is False


def test_optimization_parser_smoke():
    parser = argparse.ArgumentParser()
    opt_cli.setup_parser(parser)
    args = parser.parse_args(["calibrate", "--method", "nelder_mead"])
    assert args.optimization_command == "calibrate"
    assert args.method == "nelder_mead"


def test_optimization_help_marks_runtime_binding_not_implemented(capsys):
    parser = argparse.ArgumentParser()
    opt_cli.setup_parser(parser)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["calibrate", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "[N]" in help_text
    assert "GraphIR/ExecutionPlan/GraphExecutor binding" in help_text


@pytest.mark.parametrize(
    "command_args",
    [
        ["calibrate", "--method", "differential_evolution"],
        ["assimilate", "--method", "enkf", "--ensemble-size", "20", "--inflation-factor", "1.05"],
        ["assimilate", "--method", "var", "--max-iterations", "50"],
        ["sensitivity", "--method", "morris", "--n-trajectories", "10"],
        ["sensitivity", "--method", "sobol", "--n-samples", "256"],
        ["uncertainty", "--method", "ensemble", "--n-samples", "64", "--sampling", "lhs"],
        [
            "uncertainty", "--method", "bootstrap", "--n-samples", "64", "--sampling", "random",
            "--n-bootstrap", "200", "--ci-method", "bca",
        ],
    ],
)
def test_optimization_commands_fail_closed_without_runtime_binding(
    command_args,
    caplog,
    monkeypatch,
    tmp_path,
):
    import relic.core.simulator as simulator_module
    import relic.core.state as state_module

    constructed = []

    def forbidden_owner(*_args, **_kwargs):
        constructed.append(True)
        raise RuntimeError("standalone optimization runtime is forbidden")

    monkeypatch.setattr(state_module, "StateManager", forbidden_owner)
    monkeypatch.setattr(simulator_module, "RELICSimulator", forbidden_owner)
    monkeypatch.setattr(simulator_module, "create_simulator", forbidden_owner)

    output_path = tmp_path / f"{command_args[0]}-output"
    parser = argparse.ArgumentParser()
    opt_cli.setup_parser(parser)
    args = parser.parse_args([*command_args, "--output", str(output_path)])

    assert opt_cli.execute(args, {}) is False
    assert constructed == []
    assert not output_path.exists()
    assert not hasattr(opt_cli, "OptimizationCLI")
    assert "[N]" in caplog.text
    assert "runtime_unbound" in caplog.text


def test_stochastic_parser_smoke():
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)
    args = parser.parse_args(
        ["weather", "--samples", "3", "--duration", "30", "--wet-day-temperature-offset", "-1.5"]
    )
    assert args.stochastic_command == "weather"
    assert (args.samples, args.duration, args.wet_day_temperature_offset) == (3, 30, -1.5)


def test_stochastic_weather_help_marks_runtime_binding_not_implemented(
    capsys,
):
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["weather", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "[N]" in help_text
    assert "GraphIR/Functor runtime binding" in help_text


def test_stochastic_sde_parser_rejects_canned_scientific_processes():
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["sde", "--process", "decomposition"])


def test_stochastic_sde_parser_accepts_only_bound_processes(capsys):
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    args = parser.parse_args(
        [
            "sde",
            "--process",
            "stochastic_air_temperature",
            "--samples",
            "2",
            "--steps",
            "3",
            "--dt",
            "3600",
            "--initial-value",
            "1.5",
        ]
    )
    assert args.stochastic_command == "sde"
    assert args.process == "stochastic_air_temperature"
    assert args.samples == 2 and args.steps == 3
    assert args.dt == 3600.0 and args.initial_value == 1.5

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["sde", "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "stochastic_air_temperature" in help_text
    assert "GraphIR/Functor" in help_text
    assert "[N]" not in help_text


def test_stochastic_gillespie_parser_accepts_only_bound_marked_jump_processes(
    capsys,
):
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    args = parser.parse_args(
        ["gillespie", "--process", "stochastic_rainfall", "--samples", "2", "--steps", "3", "--dt", "21600"]
    )
    assert args.stochastic_command == "gillespie"
    assert args.process == "stochastic_rainfall"
    assert args.samples == 2 and args.steps == 3 and args.dt == 21600.0

    with pytest.raises(SystemExit):
        parser.parse_args(["gillespie", "--process", "birth_death"])

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["gillespie", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "stochastic_rainfall" in help_text
    assert "GraphIR/Functor" in help_text
    assert "[N]" not in help_text


def test_stochastic_validation_parser_exposes_nominal_coverage():
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    args = parser.parse_args(
        [
            "validate",
            "--results",
            "results.nc",
            "--observations",
            "observations.nc",
            "--nominal-coverage",
            "0.8",
        ]
    )

    assert args.stochastic_command == "validate"
    assert args.nominal_coverage == 0.8


def test_stochastic_validation_parser_has_no_literal_nominal_coverage():
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    args = parser.parse_args(
        ["validate", "--results", "results.nc", "--observations", "observations.nc"]
    )

    assert args.nominal_coverage is None


def test_stochastic_validation_reads_the_nominal_coverage_from_the_parameter_database(monkeypatch):
    """Without --nominal-coverage the single source is validation.stochastic_scores.nominal_interval_coverage."""
    from relic.parameters import get_global_manager

    captured = {}
    monkeypatch.setattr(
        stoch_cli,
        "run_validation_analysis",
        lambda received: captured.setdefault("args", received),
    )
    args = argparse.Namespace(stochastic_command="validate", nominal_coverage=None)

    assert stoch_cli.execute(args, {}) is True
    expected = get_global_manager().get(
        "validation.stochastic_scores.nominal_interval_coverage", required=True
    )
    assert captured["args"].nominal_coverage == float(expected)


def test_stochastic_validation_parser_requires_observations(capsys):
    """run_validation_analysis rejects a run without observations, so argparse requires them."""
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["validate", "--results", "results.nc"])

    assert excinfo.value.code == 2
    assert "--observations" in capsys.readouterr().err


def test_stochastic_validation_help_marks_observations_required():
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    validate_parser = subparsers.choices["validate"]
    observations = next(
        action for action in validate_parser._actions if "--observations" in action.option_strings
    )

    assert observations.required is True
    assert "optional" not in observations.help.lower()
    assert "required" in observations.help.lower()


def test_stochastic_validation_execute_dispatches_configured_coverage(
    monkeypatch,
):
    captured = {}
    args = argparse.Namespace(
        stochastic_command="validate",
        nominal_coverage=0.75,
    )
    monkeypatch.setattr(
        stoch_cli,
        "run_validation_analysis",
        lambda received: captured.setdefault("args", received),
    )

    assert stoch_cli.execute(args, {}) is True
    assert captured["args"].nominal_coverage == 0.75


def test_data_quality_cli_names_the_bounded_cf_metadata_scope():
    parser = argparse.ArgumentParser()
    data_cli.setup_parser(parser)

    args = parser.parse_args(
        [
            "quality",
            "validate",
            "input.nc",
            "--cf-variable-metadata-check",
        ]
    )
    assert args.cf_variable_metadata_check is True
    assert not hasattr(args, "cf_check")

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["quality", "validate", "input.nc", "--cf-check"]
        )


def test_data_quality_cli_forwards_bounded_cf_metadata_check(monkeypatch):
    from relic.data.cli_ops import DataCLI
    import relic.data.io.reader as reader_module

    captured = {}

    class ReadConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DataReader:
        def __init__(self, _config):
            pass

        def read_dataset(self, _path):
            return argparse.Namespace(warnings=[])

    monkeypatch.setattr(reader_module, "ReadConfig", ReadConfig)
    monkeypatch.setattr(reader_module, "DataReader", DataReader)
    args = argparse.Namespace(
        input="input.nc",
        output=None,
        cf_variable_metadata_check=True,
    )

    assert DataCLI()._run_quality_validate(args) is True
    assert captured["cf_validate"] is True


def test_compute_benchmark_output_is_scoped_as_synthetic(
    tmp_path,
    monkeypatch,
):
    import time

    from relic.computation.cli_ops import run_benchmark

    captured = []
    perf_counter_ticks = iter((10.0, 10.001))
    monkeypatch.setattr(time, "time", lambda: 1.0)
    monkeypatch.setattr(time, "perf_counter", lambda: next(perf_counter_ticks))

    class ProgressReporter:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, *_args):
            pass

    io = {
        "print_success": lambda *_args, **_kwargs: None,
        "print_error": lambda *_args, **_kwargs: None,
        "print_warning": lambda *_args, **_kwargs: None,
        "print_info": lambda *_args, **_kwargs: None,
        "print_header": lambda *_args, **_kwargs: None,
        "save_output": lambda payload, *_args: captured.append(payload),
        "ProgressReporter": ProgressReporter,
        "check_experimental_feature": lambda *_args, **_kwargs: True,
    }
    args = argparse.Namespace(
        target="transport",
        device="cpu",
        jit=False,
        batch_size=1,
        n_points=8,
        output=tmp_path / "benchmark.json",
        format="json",
    )

    assert run_benchmark(args, {}, io) is True
    assert len(captured) == 1
    payload = captured[0]
    assert payload["benchmark_scope"] == "synthetic_single_environment_microbenchmark"
    assert payload["scientific_equivalence"] is False
    assert payload["scientific_runtime_owner"] is None
    assert payload["target_label"] == "transport"
    assert set(payload["modes"]) == {"off"}
    assert payload["modes"]["off"]["avg_time_s"] == pytest.approx(0.0001)
    assert payload["modes"]["off"]["throughput_samples_per_s"] == pytest.approx(
        10000.0
    )


def _option_dests(parser):
    """Yield (parser prog, dest) for every optional argument, recursing into subparsers."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _option_dests(subparser)
        elif action.option_strings and action.dest != "help":
            yield parser.prog, action.dest


def test_no_subcommand_option_shares_a_dest_with_a_global_option():
    """A subparser default would overwrite the global value of a shared dest."""
    from relic.interfaces.cli.main import create_main_parser

    parser = create_main_parser()
    global_dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and action.dest not in {"help", "version"}
    }
    clashes = sorted(
        (prog, dest)
        for prog, dest in _option_dests(parser)
        if prog != parser.prog and dest in global_dests
    )

    assert clashes == []


@pytest.mark.parametrize("command", ["uncertainty", "weather", "sde", "gillespie"])
def test_stochastic_config_is_separate_from_the_global_runtime_config(command):
    from relic.interfaces.cli.main import create_main_parser

    required = (
        ["--samples", "2", "--duration", "10", "--wet-day-temperature-offset", "0.0"]
        if command == "weather"
        else []
    )
    args = create_main_parser().parse_args(
        ["--config", "runtime.yaml", "stochastic", command, "--config", "stochastic.yaml", *required]
    )

    assert args.config == "runtime.yaml"
    assert args.stochastic_config == "stochastic.yaml"


def test_global_verbose_survives_the_compute_health_subcommand():
    from relic.interfaces.cli.main import create_main_parser

    args = create_main_parser().parse_args(["--verbose", "compute", "health"])

    assert args.verbose is True


def test_health_benchmark_passes_the_explicit_gpu_switch(monkeypatch):
    from relic.interfaces.cli import healthcheck as health_cli
    from relic.interfaces.cli.main import create_main_parser

    received = []

    class _RecordingChecker:
        def __init__(self, config):
            received.append(config)

        def check_all(self, detailed=False, benchmark=False):
            return {"overall_status": "HEALTHY"}

    monkeypatch.setattr(health_cli, "HealthChecker", _RecordingChecker)
    args = create_main_parser().parse_args(["health", "all", "--benchmark", "--no-gpu-benchmark"])

    assert health_cli.execute(args, {"simulation": {}}) is True
    assert received == [{"simulation": {}, "enable_gpu_benchmark": False}]


def _critical_validation_result():
    from relic.core.config.validator import (
        ConfigValidationResult,
        ConfigValidationStage,
        ValidationMessage,
        ValidationSeverity,
    )

    result = ConfigValidationResult()
    result.add_message(
        ValidationMessage(
            level=list(ConfigValidationStage)[0],
            severity=ValidationSeverity.CRITICAL,
            message="time_step must be positive",
        )
    )
    return result


def test_critical_configuration_errors_stop_the_cli(monkeypatch):
    from relic.interfaces.cli import main as cli_main

    monkeypatch.setattr(
        cli_main, "load_runtime_config", lambda path: ({}, _critical_validation_result(), None)
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_main._load_and_validate_config(argparse.Namespace(config=None))

    assert excinfo.value.code == 1


def test_configuration_load_failure_without_explicit_path_stops_the_cli(monkeypatch):
    from relic.interfaces.cli import main as cli_main

    def _broken_global_config(path):
        raise RuntimeError("global configuration could not be built")

    monkeypatch.setattr(cli_main, "load_runtime_config", _broken_global_config)

    with pytest.raises(SystemExit) as excinfo:
        cli_main._load_and_validate_config(argparse.Namespace(config=None))

    assert excinfo.value.code == 1


def _parser_defaults(module, *path):
    parser = argparse.ArgumentParser()
    module.setup_parser(parser)
    node = parser
    for name in path:
        action = next(a for a in node._actions if isinstance(a, argparse._SubParsersAction))
        node = action.choices[name]
    return {action.dest: action for action in node._actions}


@pytest.mark.parametrize(
    ("command", "dest"),
    [
        ("assimilate", "ensemble_size"),
        ("assimilate", "inflation_factor"),
        ("assimilate", "max_iterations"),
        ("sensitivity", "n_trajectories"),
        ("sensitivity", "n_samples"),
        ("uncertainty", "n_samples"),
        ("uncertainty", "n_bootstrap"),
        ("uncertainty", "ci_method"),
    ],
)
def test_optimization_method_parameters_have_no_literal_default(command, dest):
    assert _parser_defaults(opt_cli, command)[dest].default is None


@pytest.mark.parametrize(
    ("command", "dest"),
    [("calibrate", "method"), ("uncertainty", "sampling")],
)
def test_optimization_method_choices_are_required(command, dest):
    action = _parser_defaults(opt_cli, command)[dest]
    assert action.required is True
    assert action.default is None


@pytest.mark.parametrize(
    ("argv", "missing"),
    [
        (["assimilate", "--method", "enkf", "--ensemble-size", "20"], "--inflation-factor"),
        (["assimilate", "--method", "var"], "--max-iterations"),
        (["sensitivity", "--method", "morris"], "--n-trajectories"),
        (["sensitivity", "--method", "sobol"], "--n-samples"),
        (["uncertainty", "--method", "ensemble", "--sampling", "lhs"], "--n-samples"),
        (
            ["uncertainty", "--method", "bootstrap", "--n-samples", "64", "--sampling", "lhs", "--ci-method", "bca"],
            "--n-bootstrap",
        ),
    ],
)
def test_optimization_commands_reject_missing_method_parameters(argv, missing, capsys):
    parser = argparse.ArgumentParser()
    opt_cli.setup_parser(parser)
    args = parser.parse_args(argv)

    assert opt_cli.execute(args, {}) is False
    assert missing in capsys.readouterr().out


@pytest.mark.parametrize("dest", ["samples", "duration", "wet_day_temperature_offset"])
def test_stochastic_weather_parameters_are_required(dest):
    action = _parser_defaults(stoch_cli, "weather")[dest]
    assert action.required is True
    assert action.default is None


def test_compute_benchmark_target_and_splitting_step_are_required():
    from relic.interfaces.cli import compute as compute_cli

    assert _parser_defaults(compute_cli, "bench")["target"].required is True
    assert _parser_defaults(compute_cli, "splitting-sensitivity")["dt"].required is True


@pytest.mark.parametrize(
    ("command", "dest", "key"),
    [
        ("splitting-sensitivity", "n_permutations", "computation.diagnostics.splitting_sensitivity.max_permutations"),
        ("dynamical-benchmark", "series_length", "computation.diagnostics.dynamical_benchmark.series_length"),
        ("dynamical-benchmark", "dt", "computation.diagnostics.dynamical_benchmark.sampling_interval"),
        ("dynamical-benchmark", "frequency", "computation.diagnostics.dynamical_benchmark.dominant_frequency"),
    ],
)
def test_compute_diagnostic_settings_come_from_the_parameter_database(monkeypatch, command, dest, key):
    from relic.interfaces.cli import compute as compute_cli
    from relic.parameters import get_global_manager

    assert _parser_defaults(compute_cli, command)[dest].default is None
    parser = argparse.ArgumentParser()
    compute_cli.setup_parser(parser)
    argv = [command] + (["--dt", "30"] if command == "splitting-sensitivity" else [])
    args = parser.parse_args(argv)
    captured = {}
    monkeypatch.setattr(
        compute_cli._compute_cli_ops,
        "execute",
        lambda received, _config, _io: captured.setdefault("args", received) is not None,
    )

    assert compute_cli.execute(args, {}) is True
    assert getattr(captured["args"], dest) == get_global_manager().get(key, required=True)


def test_run_layered_start_time_is_required():
    from relic.interfaces.cli import run_layered

    parser = argparse.ArgumentParser()
    run_layered.setup_parser(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "cfg.json", "--output", "out.json"])
    with pytest.raises(TypeError, match="start_time"):
        run_layered.run_layered_simulation(config_path="cfg.json", output_path="out.json")


def test_health_output_format_is_read_from_the_parsed_option(tmp_path, monkeypatch):
    """No second copy of the argparse default: a namespace without format is a defect."""
    from relic.interfaces.cli import healthcheck as health_cli

    class Checker:
        def __init__(self, _config):
            pass

        def _check_compute_health(self, _detailed):
            return {"status": "HEALTHY"}

    monkeypatch.setattr(health_cli, "HealthChecker", Checker)
    args = argparse.Namespace(health_command="compute", detailed=False, output=str(tmp_path / "h.txt"))

    with pytest.raises(AttributeError, match="format"):
        health_cli.execute(args, {})


def test_data_stream_run_reports_not_implemented(capsys):
    from relic.data.cli_ops import DataCLI

    assert DataCLI()._run_stream_run(argparse.Namespace(stream_command="run", source="in")) is False
    assert "not implemented" in capsys.readouterr().out.lower()


def test_data_stream_run_has_no_unread_options(monkeypatch):
    monkeypatch.setenv("RELIC_EXPERIMENTAL_STREAMING", "1")
    dests = _parser_defaults(data_cli, "stream", "run")
    for dest in ("window_hours", "file_pattern", "format"):
        assert dest not in dests


def test_data_io_cli_reads_parsed_device_and_compression_level(monkeypatch):
    """data/cli_ops.py keeps no second copy of the argparse defaults."""
    from relic.data.cli_ops import DataCLI
    import relic.data.io.reader as reader_module

    class ReadConfig:
        def __init__(self, **kwargs):
            raise AssertionError("ReadConfig must not be built without a device")

    monkeypatch.setattr(reader_module, "ReadConfig", ReadConfig)
    args = argparse.Namespace(format="auto", variables=None, to_torch=False, dtype="float32", path="x.nc")

    with pytest.raises(AttributeError, match="device"):
        DataCLI()._run_io_read(args)


def test_health_subsystem_result_without_status_fails_the_command(monkeypatch, capsys):
    from relic.interfaces.cli import healthcheck as health_cli

    class Checker:
        def __init__(self, _config):
            pass

        def _check_compute_health(self, _detailed):
            return {"details": {}}

    monkeypatch.setattr(health_cli, "HealthChecker", Checker)
    args = argparse.Namespace(health_command="compute", detailed=False, output=None, format="text")

    assert health_cli.execute(args, {}) is False
    assert "status" in capsys.readouterr().out


def test_health_summary_requires_each_subsystem_status():
    from relic.interfaces.cli import healthcheck as health_cli

    checker = health_cli.HealthChecker({})
    checker.overall_status = "HEALTHY"
    checker.results = {"compute": {"details": {}}}

    with pytest.raises(KeyError, match="status"):
        checker._print_summary()


def test_stochastic_weather_has_no_correlation_option():
    """WeatherGenerator has no correlation parameter; the old option name is not kept as an alias."""
    parser = argparse.ArgumentParser()
    stoch_cli.setup_parser(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["weather", "--samples", "3", "--duration", "30", "--correlation", "0.2"])
    assert "correlation" not in _parser_defaults(stoch_cli, "weather")


def test_stochastic_weather_offset_help_states_unit_and_meaning():
    help_text = _parser_defaults(stoch_cli, "weather")["wet_day_temperature_offset"].help

    assert "[K]" in help_text
    assert "wet" in help_text.lower()
    assert "mean" in help_text.lower()


def test_optimization_sensitivity_has_no_second_order_option():
    """SobolConfig has no second-order switch, so the CLI offers none."""
    parser = argparse.ArgumentParser()
    opt_cli.setup_parser(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["sensitivity", "--method", "sobol", "--n-samples", "64", "--second-order"])
    assert "second_order" not in _parser_defaults(opt_cli, "sensitivity")
