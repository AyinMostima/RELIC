"""
Stochastic CLI Module

Command-line interface for RELIC stochastic processes and uncertainty
quantification. Thin wrapper around relic.stochastic.workflows.

Features:
- Parameter uncertainty sampling
- Weather generation is [N] until GraphIR/Functor forcing binding is available
- SDE execution runs bound continuous-SDE processes through the canonical
  GraphIR/Functor path; unbound process names fail closed
- Gillespie execution runs bound marked-jump processes through the canonical
  GraphIR/Functor path; arbitrary reaction networks fail closed
- Validation metrics (PIT, CRPS, coverage)
- CF-1.8 compliant NetCDF output

Authors: Yuchong Long
"""

import argparse
import logging
import traceback
from typing import Any

from relic.parameters import get_global_manager
from relic.stochastic.workflows import (
    BOUND_CONTINUOUS_SDE_PROCESSES,
    BOUND_MARKED_JUMP_PROCESSES,
    run_uncertainty_analysis,
    run_weather_generation,
    run_sde_simulation,
    run_gillespie_simulation,
    run_validation_analysis,
)

logger = logging.getLogger(__name__)

# Namespace attribute of the stochastic subcommand ``--config`` option.
STOCHASTIC_CONFIG_DEST = "stochastic_config"
# Subcommands that declare that option.
WORKFLOW_FILE_COMMANDS = ("uncertainty", "weather", "sde", "gillespie")
# Parameter-database key read when validate is run without --nominal-coverage.
NOMINAL_COVERAGE_KEY = "validation.stochastic_scores.nominal_interval_coverage"


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Setup argument parser for stochastic commands.

    The subcommand ``--config`` names the stochastic workflow file and is stored
    under ``STOCHASTIC_CONFIG_DEST``, never under ``config``: the main CLI uses
    ``config`` for the RELICConfig file, and a shared dest would let the subparser
    overwrite it (argparse copies subparser values onto the parent namespace).
    """
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", "-o", type=str, help="Output file path (without extension)")
    parser.add_argument("--format", choices=["netcdf", "json"], default="netcdf", help="Output format")

    subparsers = parser.add_subparsers(dest="stochastic_command", help="Stochastic operations")

    uncertainty_parser = subparsers.add_parser("uncertainty", help="Parameter uncertainty quantification")
    uncertainty_parser.add_argument(
        "--config", dest=STOCHASTIC_CONFIG_DEST, required=True, help="Configuration file (YAML/JSON)"
    )
    uncertainty_parser.add_argument("--samples", type=int, help="Number of parameter samples")

    weather_help = (
        "[N] Weather generation requires a declared weather-forcing operator "
        "and a validated GraphIR/Functor runtime binding"
    )
    weather_parser = subparsers.add_parser(
        "weather",
        help=weather_help,
        description=weather_help,
    )
    weather_parser.add_argument(
        "--config", dest=STOCHASTIC_CONFIG_DEST, help="Configuration file (YAML/JSON)"
    )
    weather_parser.add_argument("--samples", type=int, required=True, help="Number of weather realizations")
    weather_parser.add_argument("--duration", type=int, required=True, help="Simulation duration (days)")
    weather_parser.add_argument(
        "--wet-day-temperature-offset",
        type=float,
        required=True,
        help=(
            "Wet-day temperature offset [K]: shift of the mean daily temperature on wet days "
            "relative to dry days, T = T_OU + offset * S (Richardson 1981; SWAT 2005, 1:3.4.15-18)"
        ),
    )

    sde_help = (
        "Execute a bound continuous-SDE process through the canonical "
        "GraphIR/Functor path; unbound process names fail closed and drift/"
        "diffusion coefficients come from the ParameterManager database"
    )
    sde_parser = subparsers.add_parser("sde", help=sde_help, description=sde_help)
    sde_parser.add_argument(
        "--process",
        choices=sorted(BOUND_CONTINUOUS_SDE_PROCESSES),
        default=None,
        help="Bound continuous-SDE process name (sde.process in --config when omitted)",
    )
    sde_parser.add_argument(
        "--config",
        dest=STOCHASTIC_CONFIG_DEST,
        help="Configuration file (YAML/JSON) with an optional sde section",
    )
    sde_parser.add_argument(
        "--samples", type=int, default=None, help="Number of batched realizations (sde.samples)"
    )
    sde_parser.add_argument(
        "--steps", type=int, default=None, help="Number of fixed steps (sde.steps)"
    )
    sde_parser.add_argument(
        "--dt", type=float, default=None, help="Fixed simulator step in seconds (sde.dt_seconds)"
    )
    sde_parser.add_argument(
        "--initial-value",
        dest="initial_value",
        type=float,
        default=None,
        help="Initial state in the SCM state unit (sde.initial_value)",
    )

    gillespie_help = (
        "Execute a bound marked-jump process (direct-SSA event clock) through "
        "the canonical GraphIR/Functor path; arbitrary reaction networks and "
        "unbound process names fail closed"
    )
    gillespie_parser = subparsers.add_parser(
        "gillespie",
        help=gillespie_help,
        description=gillespie_help,
    )
    gillespie_parser.add_argument(
        "--process",
        choices=sorted(BOUND_MARKED_JUMP_PROCESSES),
        default=None,
        help="Bound marked-jump process name (gillespie.process in --config when omitted)",
    )
    gillespie_parser.add_argument(
        "--config",
        dest=STOCHASTIC_CONFIG_DEST,
        help="Configuration file (YAML/JSON) with an optional gillespie section",
    )
    gillespie_parser.add_argument(
        "--samples", type=int, default=None, help="Number of batched realizations (gillespie.samples)"
    )
    gillespie_parser.add_argument(
        "--steps", type=int, default=None, help="Number of fixed steps (gillespie.steps)"
    )
    gillespie_parser.add_argument(
        "--dt", type=float, default=None, help="Fixed simulator step in seconds (gillespie.dt_seconds)"
    )

    validation_parser = subparsers.add_parser("validate", help="Validation analysis of simulation results")
    validation_parser.add_argument("--results", required=True, help="Simulation results file")
    # run_validation_analysis compares the ensemble with observations and raises
    # without them, so the option is required at parse time.
    validation_parser.add_argument(
        "--observations",
        required=True,
        help="Observational data file (required; NetCDF or torch-saved mapping keyed like --results)",
    )
    validation_parser.add_argument(
        "--nominal-coverage",
        type=float,
        default=None,
        help=(
            "Nominal probability for the central ensemble interval "
            "(when omitted: validation.stochastic_scores.nominal_interval_coverage)"
        ),
    )


def execute(args: Any, config: Any) -> bool:
    """Execute stochastic command."""
    if not hasattr(args, "stochastic_command") or not args.stochastic_command:
        print("Error: stochastic command requires a subcommand")
        print("Available commands: uncertainty, weather, sde, gillespie, validate")
        return False
    if args.stochastic_command in WORKFLOW_FILE_COMMANDS:
        # relic.stochastic.workflows reads its own workflow file from ``config``.
        args = argparse.Namespace(**{**vars(args), "config": getattr(args, STOCHASTIC_CONFIG_DEST)})
    if args.stochastic_command == "validate" and args.nominal_coverage is None:
        # The parameter database is the single source of the nominal coverage.
        coverage = get_global_manager().get(
            NOMINAL_COVERAGE_KEY,
            units="dimensionless",
            required=True,
        )
        args = argparse.Namespace(**{**vars(args), "nominal_coverage": float(coverage)})

    try:
        if args.stochastic_command == "uncertainty":
            run_uncertainty_analysis(args)
        elif args.stochastic_command == "weather":
            run_weather_generation(args)
        elif args.stochastic_command == "sde":
            run_sde_simulation(args)
        elif args.stochastic_command == "gillespie":
            run_gillespie_simulation(args)
        elif args.stochastic_command == "validate":
            run_validation_analysis(args)
        else:
            logger.error("Unknown stochastic command: %s", args.stochastic_command)
            return False
        return True

    except (
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        KeyError,
    ) as exc:
        logger.error("Error executing stochastic command: %s", exc)
        if hasattr(args, "verbose") and args.verbose:
            traceback.print_exc()
        return False
