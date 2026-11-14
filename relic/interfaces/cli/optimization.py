"""
Optimization CLI Module

Command-line interface for RELIC optimization operations.

The command surface is retained for discovery, but execution is fail-closed
until it can receive a sealed canonical runtime binding.
"""

import logging

logger = logging.getLogger(__name__)

_RUNTIME_UNBOUND_MESSAGE = (
    "[N] optimization runtime_unbound: execution requires a caller-supplied "
    "sealed GraphIR/ExecutionPlan/GraphExecutor binding, canonical simulator "
    "lifecycle owners, explicit initial state and output projection, unit and "
    "observation contracts, and restart-safe RNG semantics; standalone runtime "
    "construction is forbidden"
)


# Options each (command, method) must state; argparse cannot express
# method-conditional requirements, so execute checks them before anything else.
_METHOD_REQUIRED_OPTIONS = {
    ("assimilate", "enkf"): ("--ensemble-size", "--inflation-factor"),
    ("assimilate", "var"): ("--max-iterations",),
    ("sensitivity", "morris"): ("--n-trajectories",),
    ("sensitivity", "sobol"): ("--n-samples",),
    ("uncertainty", "ensemble"): ("--n-samples",),
    ("uncertainty", "bootstrap"): ("--n-samples", "--n-bootstrap", "--ci-method"),
}


def setup_parser(parser):
    """Setup argument parser for optimization commands."""
    subparsers = parser.add_subparsers(dest="optimization_command", help="Optimization operations")

    calib_parser = subparsers.add_parser(
        "calibrate",
        help="[N] Parameter calibration requires canonical runtime binding",
        description=_RUNTIME_UNBOUND_MESSAGE,
    )
    calib_parser.add_argument("--parameters", type=str, help="Comma-separated parameter names")
    calib_parser.add_argument("--param-bounds", type=str, help="Parameter bounds YAML/JSON file")
    calib_parser.add_argument("--target-data", type=str, help="Target data file")
    calib_parser.add_argument("--method", type=str, required=True, help="Optimization method")
    calib_parser.add_argument("--output", type=str, help="Output directory")

    assim_parser = subparsers.add_parser(
        "assimilate",
        help="[N] Data assimilation requires canonical runtime binding",
        description=_RUNTIME_UNBOUND_MESSAGE,
    )
    assim_parser.add_argument("--method", type=str, required=True, choices=["enkf", "var"], help="Assimilation method")
    assim_parser.add_argument("--observations", type=str, help="Observations file")
    assim_parser.add_argument("--ensemble-size", type=int, help="Ensemble size (required for enkf)")
    assim_parser.add_argument(
        "--inflation-factor", type=float, help="Multiplicative covariance inflation (required for enkf)"
    )
    assim_parser.add_argument("--max-iterations", type=int, help="Max iterations (required for var)")
    assim_parser.add_argument("--output", type=str, help="Output directory")

    sens_parser = subparsers.add_parser(
        "sensitivity",
        help="[N] Sensitivity analysis requires canonical runtime binding",
        description=_RUNTIME_UNBOUND_MESSAGE,
    )
    sens_parser.add_argument("--method", type=str, required=True, choices=["morris", "sobol"], help="Sensitivity method")
    sens_parser.add_argument("--param-ranges", type=str, help="Parameter ranges YAML/JSON file")
    sens_parser.add_argument("--n-trajectories", type=int, help="Number of trajectories (required for morris)")
    sens_parser.add_argument("--n-samples", type=int, help="Number of samples (required for sobol)")
    sens_parser.add_argument("--output", type=str, help="Output directory")

    unc_parser = subparsers.add_parser(
        "uncertainty",
        help="[N] Uncertainty quantification requires canonical runtime binding",
        description=_RUNTIME_UNBOUND_MESSAGE,
    )
    unc_parser.add_argument("--method", type=str, required=True, choices=["ensemble", "bootstrap"], help="UQ method")
    unc_parser.add_argument("--uncertain-params", type=str, help="Uncertain parameters YAML/JSON file")
    unc_parser.add_argument("--n-samples", type=int, help="Number of samples/ensemble size (required)")
    unc_parser.add_argument("--n-bootstrap", type=int, help="Number of bootstrap samples (required for bootstrap)")
    unc_parser.add_argument(
        "--sampling", type=str, required=True, choices=["random", "lhs", "sobol"], help="Sampling strategy"
    )
    unc_parser.add_argument("--parallel", action="store_true", help="Use parallel evaluation")
    unc_parser.add_argument(
        "--ci-method", type=str, choices=["percentile", "bca", "basic"], help="CI method (required for bootstrap)"
    )
    unc_parser.add_argument("--output", type=str, help="Output directory")


def execute(args, config):
    """Execute optimization command."""
    if not hasattr(args, "optimization_command") or not args.optimization_command:
        print("Error: optimization command requires a subcommand")
        print(
            "Declared [N] commands (runtime_unbound): calibrate, assimilate, "
            "sensitivity, uncertainty"
        )
        return False

    if args.optimization_command not in {
        "calibrate",
        "assimilate",
        "sensitivity",
        "uncertainty",
    }:
        logger.error("Unknown optimization command: %s", args.optimization_command)
        return False

    # Method-specific design settings have no default: the command states them.
    required = _METHOD_REQUIRED_OPTIONS.get((args.optimization_command, args.method), ())
    missing = [option for option in required if getattr(args, option.lstrip("-").replace("-", "_")) is None]
    if missing:
        print(
            f"Error: optimization {args.optimization_command} --method {args.method} "
            f"requires {', '.join(missing)}"
        )
        return False

    del config
    logger.error(_RUNTIME_UNBOUND_MESSAGE)
    return False
