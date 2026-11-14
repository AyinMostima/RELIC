"""Stochastic output helpers with bounded CF variable metadata handling."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_path: str, format: str = "netcdf") -> None:
    """Save results without claiming dataset-level CF conformance."""
    output_file = Path(output_path)

    if format.lower() == "netcdf":
        from relic.data.io.writer import DataWriter, WriteConfig
        from relic.coupling.external.common.io import DataFormat, CompressionType

        dataset = _convert_results_to_dataset(results)

        config = WriteConfig(
            format=DataFormat.NETCDF4,
            compression=CompressionType.ZLIB,
            cf_sync=True,
            add_bounds=True,
            update_history=True,
            title="RELIC Stochastic Simulation Results",
            source="RELIC Stochastic Framework",
            comment="Results generated with bounded CF variable metadata handling",
        )

        writer = DataWriter(config)
        result = writer.write_dataset(dataset, output_file.with_suffix(".nc"))

        if not result.validation_passed:
            raise RuntimeError(
                "NetCDF output failed: " + "; ".join(result.warnings)
            )
        if result.warnings:
            for warning in result.warnings:
                logger.warning("NetCDF output warning: %s", warning)

        logger.info("Results saved to %s", output_file.with_suffix(".nc"))

    elif format.lower() == "json":
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value

        with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as handle:
            import json
            json.dump(json_results, handle, indent=2)
        logger.info("Results saved to %s", output_file.with_suffix(".json"))
    else:
        raise ValueError(f"Unsupported output format: {format}")


def _convert_results_to_dataset(results: Dict[str, Any]) -> xr.Dataset:
    """Convert stochastic results to CF-1.8 compliant xarray Dataset."""
    data_vars: Dict[str, Any] = {}
    coords: Dict[str, Any] = {}
    global_attrs: Dict[str, Any] = {}

    # The "time" dimension of a trajectory is a step index; elapsed simulation
    # time is carried by the data variable time_seconds on that dimension.
    # A CF "time" coordinate would require calendar "since" units, which a
    # relative simulation clock does not have, so no such coordinate is built.
    # Textual metadata (process names, owners, units) becomes global
    # attributes: CF data variables carry numeric or boolean payloads only.
    def _store_scalar(name: str, value: Any) -> None:
        if isinstance(value, str):
            global_attrs[name] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            global_attrs[name] = ",".join(value)
        elif isinstance(value, (list, tuple)):
            numpy_data = np.array(value)
            dims = [f"{name}_dim{i}" for i in range(numpy_data.ndim)]
            data_vars[name] = (dims, numpy_data, {"long_name": name})
        else:
            data_vars[name] = ([], np.array(value), {"long_name": name})

    # Every tensor is described by its producer: per-state variables by
    # results['state_units'] (units from the SSOT catalog), every other tensor
    # by results['variable_units'] and results['variable_dimensions']. A tensor
    # without a declaration raises; no units, standard name or dimension is
    # inferred from the variable name.
    declared_units = _declaration(results, "state_units")
    variable_units = _declaration(results, "variable_units")
    variable_dimensions = _declaration(results, "variable_dimensions")

    def _store_tensor(name: str, tensor: torch.Tensor) -> None:
        numpy_data = tensor.detach().cpu().numpy()
        declared = _declared_state_variable(name, declared_units, numpy_data.ndim)
        if declared is None:
            declared = _declared_variable(name, numpy_data.ndim, variable_units, variable_dimensions)
        dims, attrs = declared
        if "sample" in dims:
            coords["sample"] = np.arange(numpy_data.shape[dims.index("sample")])
        data_vars[name] = (list(dims), numpy_data, attrs)

    for key, value in results.items():
        if key in ("variable_units", "variable_dimensions"):
            continue
        if isinstance(value, torch.Tensor):
            _store_tensor(key, value)

        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                full_key = f"{key}_{sub_key}"
                if isinstance(sub_value, torch.Tensor):
                    _store_tensor(full_key, sub_value)
                else:
                    _store_scalar(full_key, sub_value)

        else:
            _store_scalar(key, value)

    dataset = xr.Dataset(data_vars, coords=coords)
    dataset.attrs.update(global_attrs)
    dataset.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": "RELIC Stochastic Simulation Results",
            "institution": "Yuchong Long",
            "source": "RELIC Stochastic Framework",
            "history": f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}: Generated by RELIC stochastic workflows",
            "references": "RELIC Documentation",
            "comment": "Stochastic simulation results with uncertainty quantification",
        }
    )

    return dataset


_STATE_VALUE_PREFIXES = ("trajectory_", "ensemble_mean_", "analytic_mean_")
_STATE_VARIANCE_PREFIXES = ("ensemble_variance_", "analytic_variance_")
_UNIT_FACTOR = re.compile(r"^([A-Za-z]+)(-?\d+)?$")


def _squared_units(units: str) -> str:
    """Square a UDUNITS product of factors such as 'kg m-2 s-1' -> 'kg2 m-4 s-2'.

    '1' (dimensionless) stays '1'. A unit string that is not a plain product
    of 'symbol[exponent]' factors is rejected instead of being guessed.
    """
    factors = units.split()
    if factors == ["1"]:
        return "1"
    squared = []
    for factor in factors:
        match = _UNIT_FACTOR.match(factor)
        if match is None:
            raise ValueError(f"Cannot square the unit factor {factor!r} of {units!r}")
        exponent = 2 * int(match.group(2) or 1)
        squared.append(f"{match.group(1)}{exponent}")
    if not squared:
        raise ValueError("Cannot square an empty unit string")
    return " ".join(squared)


def _declaration(results: Dict[str, Any], name: str) -> Dict[str, Any]:
    """A declaration mapping of the results (absent means empty)."""
    value = results.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"results['{name}'] must be a mapping")
    return value


def _declared_state_variable(
    key: str, declared_units: Dict[str, str], ndim: int
) -> Optional[Tuple[Tuple[str, ...], dict]]:
    """Dimensions and attributes of a per-state result variable, or None.

    trajectory_<state> is (sample, time); ensemble_/analytic_ means and
    variances are per time step. Values carry the declared state units and
    variances their square. No CF standard name is added: the SSOT catalog
    owns it and declares none for these states.
    """
    for prefix in _STATE_VALUE_PREFIXES + _STATE_VARIANCE_PREFIXES:
        state = key[len(prefix):]
        if not (key.startswith(prefix) and state in declared_units):
            continue
        dims = ("sample", "time") if prefix == "trajectory_" else ("time",)
        if ndim != len(dims):
            raise ValueError(
                f"Result variable '{key}' has {ndim} dimensions; the per-state layout "
                f"declares {dims}"
            )
        if prefix in _STATE_VARIANCE_PREFIXES:
            return dims, {
                "long_name": key,
                "units": _squared_units(declared_units[state]),
                "comment": f"Variance of the bound stochastic state {state}",
            }
        return dims, {
            "long_name": key,
            "units": declared_units[state],
            "comment": f"Bound stochastic state {state} (units declared by the SSOT catalog)",
        }
    return None


def _declared_variable(
    key: str,
    ndim: int,
    variable_units: Dict[str, str],
    variable_dimensions: Dict[str, Any],
) -> Tuple[Tuple[str, ...], dict]:
    """Dimensions and attributes of a declared result variable; undeclared raises."""
    if key not in variable_units:
        raise ValueError(
            f"Result variable '{key}' has no declared units: list it in "
            "results['variable_units'] (or results['state_units'] for a bound state); "
            "units are not inferred from the variable name"
        )
    if key not in variable_dimensions:
        raise ValueError(
            f"Result variable '{key}' has no declared dimensions in "
            "results['variable_dimensions']"
        )
    dims = tuple(str(name) for name in variable_dimensions[key])
    if len(dims) != ndim:
        raise ValueError(
            f"Result variable '{key}' has {ndim} dimensions but declares {list(dims)}"
        )
    return dims, {"long_name": key, "units": str(variable_units[key])}
