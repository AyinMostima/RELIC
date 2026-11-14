"""Minimal construction of every Functor class defined in ``relic`` (test helper).

One owner for "which Functor classes exist" and "how each is built with its
documented minimal arguments", shared by the dependency contract test
(``test/core/test_bridge_classes.py``) and the inventory collector
(``test/audit/collect_functor_inventory.py``).

* ``relic_functor_class_paths()`` finds the modules that define FunctorBase
  subclasses by a static scan of the class-inheritance graph under ``relic``,
  imports exactly those modules (an import failure raises), and returns the
  ``module:qualname`` of every FunctorBase subclass defined in ``relic``.
* ``MINIMAL_ARGUMENTS`` gives the documented minimal keyword arguments of
  classes that cannot be built with no arguments; every other class is built
  with ``cls()``.
* ``NOT_MINIMALLY_CONSTRUCTIBLE`` lists the classes that cannot be built
  minimally, each with the kind (``abstract``, ``fail_closed``,
  ``external_resource``, ``not_functor``), a marker the failure must contain,
  and the reason. Callers re-verify each entry, so no class is skipped silently.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELIC_ROOT = PROJECT_ROOT / "relic"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _functor_defining_modules() -> List[str]:
    """Static inheritance closure from FunctorBase over top-level classes in relic."""
    bases_of: Dict[Tuple[str, str], List[str]] = {}
    for path in sorted(RELIC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        module = _module_name(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    names.append(base.attr)
            bases_of[(module, node.name)] = names
    functor_names = {"FunctorBase"}
    changed = True
    while changed:
        changed = False
        for (_, class_name), names in bases_of.items():
            if class_name not in functor_names and any(n in functor_names for n in names):
                functor_names.add(class_name)
                changed = True
    return sorted(
        {module for (module, class_name) in bases_of if class_name in functor_names}
    )


def _all_subclasses(cls: type) -> set:
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def relic_functor_class_paths() -> List[str]:
    """Return ``module:qualname`` of every FunctorBase subclass defined in relic."""
    from relic.core.functor import FunctorBase

    for module in _functor_defining_modules():
        importlib.import_module(module)
    return sorted(
        f"{cls.__module__}:{cls.__qualname__}"
        for cls in _all_subclasses(FunctorBase)
        if cls.__module__.startswith("relic.") and "<locals>" not in cls.__qualname__
    )


def resolve_class(path: str) -> type:
    module_name, qualname = path.split(":")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


# ---------------------------------------------------------------------------
# Documented minimal arguments (from each constructor's fail-fast requirement).
# ---------------------------------------------------------------------------


def _regular_grid(n: int, extent: float = 4.0) -> Any:
    """n x n cartesian cells covering [0, extent]^2 with explicit cell bounds [m]."""
    from relic.coupling.scales.spatial.grid_utils import GridDescriptor

    edges = torch.linspace(0.0, extent, n + 1)
    bounds = torch.stack([edges[:-1], edges[1:]], dim=1)
    centers = bounds.mean(dim=1)
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(n, n),
        x_coords=centers,
        y_coords=centers.clone(),
        x_bounds=bounds,
        y_bounds=bounds.clone(),
        units={"x": "m", "y": "m"},
    )


def _hierarchy_index() -> Any:
    from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex

    topology = {
        "n_source_units": 2,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0},
        "source_areas": [1.0, 1.0],
        "target_areas": [2.0],
    }
    return HierarchyIndex(
        source_scale="patch",
        target_scale="hillslope",
        topology=topology,
        device=torch.device("cpu"),
    )


def _temporal_config(direction: str) -> Any:
    from relic.coupling.scales.temporal.types import (
        ConservationLevel,
        ScalingDirection,
        TemporalConfig,
    )

    if direction == "up":
        return TemporalConfig(
            source_frequency="hourly",
            target_frequency="daily",
            conservation_level=ConservationLevel.NONE,
        )
    return TemporalConfig(
        source_frequency="daily",
        target_frequency="hourly",
        direction=ScalingDirection.DOWNSCALE,
        conservation_level=ConservationLevel.NONE,
    )


# Documented minimal keyword arguments, built lazily (grids, indices, configs).
MINIMAL_ARGUMENTS: Dict[str, Callable[[], Dict[str, Any]]] = {
    # CESMConfig and CMIP6Config declare root_path (no adapter assumes the current directory).
    "relic.coupling.external.atmospheric_models.cesm_adapter:CESMAdapter": (
        lambda: dict(
            config=resolve_class("relic.coupling.external.atmospheric_models.cesm_adapter:CESMConfig")(
                root_path="."
            ),
        )
    ),
    "relic.coupling.external.climate_models.cmip6_adapter:CMIP6Adapter": (
        lambda: dict(
            config=resolve_class("relic.coupling.external.climate_models.cmip6_adapter:CMIP6Config")(
                root_path="."
            ),
        )
    ),
    # SWATConfig declares root_path and output_level (only the output.hru layout is read).
    "relic.coupling.external.hydrological_models.swat_adapter:SWATAdapter": (
        lambda: dict(
            config=resolve_class("relic.coupling.external.hydrological_models.swat_adapter:SWATConfig")(
                root_path=".", output_level="hru"
            ),
        )
    ),
    # The Functor-owned substream seed of the wildfire ignition draw is required.
    "relic.scientific.anthroposphere.management.fire_management:FireManagement": (
        lambda: dict(
            config={"seed": 0},
        )
    ),
    "relic.scientific.atmosphere.stochastic_rainfall:MarkedPoissonRainfallFunctor": (
        lambda: dict(
            config={"seed": 0},
        )
    ),
    "relic.scientific.atmosphere.stochastic_temperature:OrnsteinUhlenbeckTemperatureFunctor": (
        lambda: dict(
            config={"seed": 0},
        )
    ),
    "relic.scientific.biosphere.growth.phenology_growth:PhenologyGrowth": (
        lambda: dict(
            phenology_model_owner="static_phenology",
        )
    ),
    "relic.scientific.biosphere.phenology.phenology_functor:PhenologyFunctor": (
        lambda: dict(
            config={"canonical_model_owner": "static_phenology"},
        )
    ),
    # The chemistry process law (open mechanism or closed box) is a required selection.
    "relic.scientific.atmosphere.chemistry:ChemistryFunctor": (
        lambda: dict(
            physical_boundary="open_mechanism",
        )
    ),
    # The domain dimension and the stability scheme are required selections.
    "relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor:AtmosphericAdvectionDiffusionFunctor": (
        lambda: dict(
            spatial_dims=3,
            stability_scheme="monin-obukhov",
        )
    ),
    # The canopy resistance source is a required declaration of the
    # Penman-Monteith configuration ('state' reads surface_resistance).
    "relic.scientific.hydrosphere.evapotranspiration.penman_monteith:PenmanMonteithET": (
        lambda: dict(
            config={"canopy_resistance_source": "state"},
        )
    ),
    # The Medlyn law only implements the leaf_surface_equals_canopy_air substitution and
    # refuses an undeclared surface condition (Medlyn et al. 2011; CTSM tech note Eq. 2.9.1).
    "relic.scientific.biosphere.photosynthesis.stomatal:MedlynStomatalConductance": (
        lambda: dict(
            config={"surface_conditions": "leaf_surface_equals_canopy_air"},
        )
    ),
    # The temporal support is a required declaration of the leaf energy balance
    # ('daily': FAO-56 24-hour form, G = 0, dt = 86400 s, as in the global daily graph).
    "relic.scientific.biosphere.photosynthesis.leaf_energy_balance:LeafEnergyBalance": (
        lambda: dict(
            config={"temporal_support": "daily"},
        )
    ),
    "relic.coupling.internal.soil_vegetation:SoilVegetationCoupling": (
        lambda: dict(
            name="soil_vegetation",
            config={},
        )
    ),
    "relic.coupling.internal.water_carbon:WaterCarbonCoupling": (
        lambda: dict(
            name="water_carbon",
            config={},
        )
    ),
    "relic.coupling.observation.quality:PostObservationQualityOperator": (
        lambda: dict(
            config={
                "instrument_contract_parameter_prefix": (
                    "coupling.observation.tower_concentration.profile_004_synthetic_qc"
                )
            },
        )
    ),
    "relic.coupling.observation.state_feedback:ObservationStateFeedbackFunctor": (
        lambda: dict(
            config={
                "state_variable": "soil_moisture",
                "observation_variable": "observed_soil_moisture",
            },
        )
    ),
    # The bare scale-coupling bridge fabricates no scales: the caller declares
    # the scale pair and the coupling type (a missing one raises TypeError).
    "relic.core.functor.scale_coupling_bridge:ScaleCouplingFunctor": (
        lambda: dict(
            source_scale="fine",
            target_scale="coarse",
            coupling_type="spatial",
        )
    ),
    "relic.coupling.plugin.adapter:PluginFunctorAdapter": (
        lambda: dict(
            name="plugin_adapter_probe",
            plugin_path=str(PROJECT_ROOT / "unused_probe_plugin.dll"),
            # A plugin adapter declares the scales it couples (required config keys).
            config={
                "source_scale": "external_grid",
                "target_scale": "grid_cell",
                "coupling_type": "spatial",
            },
        )
    ),
    "relic.coupling.scales.hierarchical.aggregator:HierarchicalAggregator": (
        lambda: dict(
            index=_hierarchy_index(),
            weight_type="area",
            rule="conservative",
        )
    ),
    "relic.coupling.scales.spatial.factory:AutoRegridder": (
        lambda: dict(
            source_grid=_regular_grid(4),
            target_grid=_regular_grid(2),
        )
    ),
    "relic.coupling.scales.spatial.regrid_bilinear:BilinearRegridder": (
        lambda: dict(
            source_grid=_regular_grid(4),
            target_grid=_regular_grid(2),
        )
    ),
    "relic.coupling.scales.spatial.regrid_conservative:ConservativeRegridder": (
        lambda: dict(
            source_grid=_regular_grid(4),
            target_grid=_regular_grid(2),
            quantity_basis="intensive",
        )
    ),
    "relic.coupling.scales.spatial.regrid_nearest:NearestNeighborRegridder": (
        lambda: dict(
            source_grid=_regular_grid(4),
            target_grid=_regular_grid(2),
        )
    ),
    "relic.coupling.scales.spatial.spatial_scaling:SpatialScaling": (
        lambda: dict(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
        )
    ),
    "relic.coupling.scales.adapters:SpatialAggregation": (
        lambda: dict(
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
    ),
    "relic.coupling.scales.temporal.temporal_scaling:TemporalScaling": (
        lambda: dict(
            source_scale="source",
            target_scale="target",
            method="mean",
        )
    ),
    "relic.coupling.scales.temporal.aggregator:TemporalAggregator": (
        lambda: dict(
            config=_temporal_config("up"),
        )
    ),
    "relic.coupling.scales.temporal.disaggregator:TemporalDisaggregator": (
        lambda: dict(
            config=_temporal_config("down"),
        )
    ),
}

# Classes that cannot be constructed minimally: (kind, marker, reason). For
# ``fail_closed`` and ``external_resource`` the marker must appear in the error
# raised by a no-argument (or documented-minimal) construction.
NOT_MINIMALLY_CONSTRUCTIBLE: Dict[str, Tuple[str, str, str]] = {
    "relic.core.functor.transport_bridge:TransportMechanismFunctor": (
        "fail_closed",
        "TRANSPORT_TYPE",
        "the bridge declares no transport label; a concrete class declares TRANSPORT_TYPE",
    ),
    "relic.scientific.anthroposphere.emissions.coupling_to_atmosphere:AtmosphereCoupling": (
        "fail_closed",
        "runtime_dependency_unbound",
        "fails closed until the emission inventory and surface-flux providers are bound",
    ),
    "relic.scientific.anthroposphere.interface_mapping:AnthroposphereInterfaceMapping": (
        "fail_closed",
        "runtime_metadata_domain_unbound",
        "fails closed until exact SCM/CF metadata bindings exist",
    ),
    "relic.scientific.anthroposphere.inventory:EmissionInventory": (
        "fail_closed",
        "runtime_domain_unbound",
        "fails closed until a verified target-domain binding exists",
    ),
    "relic.coupling.external.hydrological_models.swat_adapter:SWATAdapter": (
        "external_resource",
        "needs a SWATConfig declaring root_path",
        "requires a declared SWAT output directory (root_path) and output_level",
    ),
    "relic.scientific.isotopes.functor:IsotopeHeartbeatFunctor": (
        "fail_closed",
        "runtime_unbound",
        "fails closed because governed isotope component updates are absent",
    ),
    "relic.orchestration.adapters:ExecutableAdapterFunctor": (
        "fail_closed",
        "requires adapter_type",
        "a graph adapter exists only for a compiled GraphIR edge (adapter type, source node and ports)",
    ),
    "relic.models.pytorch_surrogate:PyTorchSurrogate": (
        "external_resource",
        "model_path",
        "loads a trained model file at construction",
    ),
    "relic.models.surrogates:ONNXSurrogate": (
        "external_resource",
        "model_path",
        "loads a trained ONNX model file at construction",
    ),
    "relic.core.functor.physical_field_bridge:PhysicalFieldFunctor": (
        "abstract",
        "get_value_at",
        "the bridge supplies no coordinate query or forward; a concrete field "
        "declares SPATIAL_DIMS and implements both",
    ),
    "relic.scientific.biosphere.phenology.base:PhenologyBase": (
        "abstract",
        "compute_phenological_state",
        "abstract base; its concrete phenology models are probed instead",
    ),
    "relic.models.surrogates:SurrogateFunctor": (
        "abstract",
        "_load_model",
        "abstract surrogate base; concrete surrogates need a model file",
    ),
    "relic.coupling.external.common.base_adapter:ExternalAdapter": (
        "abstract",
        "read",
        "abstract external-model adapter base; concrete adapters are probed",
    ),
    "relic.coupling.scales.spatial.base_regridder:SpatialRegridderBase": (
        "abstract",
        "upscale",
        "abstract regridder base; concrete regridders are probed",
    ),
}


# Arguments that re-verify a non-constructible entry (default: no arguments).
_SURROGATE_PROBE = {
    "name": "surrogate_probe",
    "config": {"input_variables": ["x"], "output_variables": ["y"]},
}
FAILURE_PROBE_ARGUMENTS: Dict[str, Dict[str, Any]] = {
    "relic.models.pytorch_surrogate:PyTorchSurrogate": _SURROGATE_PROBE,
    "relic.models.surrogates:ONNXSurrogate": _SURROGATE_PROBE,
}


def verify_not_minimally_constructible(path: str) -> str:
    """Re-verify one NOT_MINIMALLY_CONSTRUCTIBLE entry and return its reason."""
    from relic.core.functor import FunctorBase

    kind, marker, reason = NOT_MINIMALLY_CONSTRUCTIBLE[path]
    cls = resolve_class(path)
    if kind == "abstract":
        if not inspect.isabstract(cls) or marker not in cls.__abstractmethods__:
            raise AssertionError(f"{path} is listed abstract ({marker}) but is not")
        return reason
    if kind == "not_functor":
        if isinstance(cls, type) and issubclass(cls, FunctorBase):
            raise AssertionError(f"{path} is listed as not a Functor but is one")
        return reason
    try:
        cls(**FAILURE_PROBE_ARGUMENTS.get(path, {}))
    except Exception as exc:  # the listed failure is the expected outcome
        if marker not in str(exc):
            raise AssertionError(
                f"{path} failed for another reason than {marker!r}: {exc}"
            ) from exc
        return reason
    raise AssertionError(f"{path} is now minimally constructible; remove it from the list")


def minimal_arguments(path: str) -> Dict[str, Any]:
    """Documented minimal keyword arguments of one constructible Functor class."""
    if path in NOT_MINIMALLY_CONSTRUCTIBLE:
        raise LookupError(f"{path} is not minimally constructible: "
                          f"{NOT_MINIMALLY_CONSTRUCTIBLE[path][2]}")
    builder = MINIMAL_ARGUMENTS.get(path)
    return builder() if builder is not None else {}


def construct_minimal(path: str) -> Any:
    """Build one Functor with its documented minimal arguments."""
    return resolve_class(path)(**minimal_arguments(path))
