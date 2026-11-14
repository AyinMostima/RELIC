"""One strict graph of a forest and its boundary layer, with an observed surface layer.

Photosynthesis, stomatal conductance, ozone stress and isoprene emission feed the boundary layer
through BoundaryLayerEmissionSource; dry deposition, eddy exchange and chemistry act on the same
atmospheric state, and ozone returns to the plant through ozone stress. Friction velocity, Obukhov
length and aerodynamic resistance are forcing, so no atmospheric_turbulence node exists. VPD and PPFD
are forcing too, so neither environmental_forcing nor atmospheric_radiation exists.
Units: mixing ratios mol mol-1, fluxes mol m-2 s-1, dt s.
"""

import pytest
import torch

from relic.core.state.manager import StateManager
from relic.orchestration import (
    GraphExecutor,
    GraphIR,
    NodeRegistry,
    NodeSpec,
    SSOTCatalog,
    compile_graph,
)
from relic.parameters import get_global_manager
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.transport.boundary_layer_emission_source import (
    BoundaryLayerEmissionSource,
)
from relic.scientific.atmosphere.transport.dry_deposition_aggregator import DryDepositionAggregator
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import EddyDiffusionBLMFunctor
from relic.scientific.biosphere.bvoc.megan_model import EmissionPathway, MEGANModel
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.photosynthesis.stomatal import MedlynStomatalConductance
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress

DT = 20.0  # s
SURFACE_LAYER = ("friction_velocity", "monin_obukhov_length", "aerodynamic_resistance")
FARQUHAR_CONFIG = {"enable_triose_phosphate_limitation": True}
FARQUHAR_SELECTION = {"validate_inputs": False, "environment_source": "external_forcing"}
MEDLYN_CONFIG = {"surface_conditions": "leaf_surface_equals_canopy_air"}
MEGAN_CONFIG = {
    "enabled_compounds": ["isoprene"],
    "ppfd_source": "state",
    "use_stress_functors": True,
    "stress_factors": ["ozone"],
}
FORCING_SELECTION = {"surface_layer_source": "external_forcing"}


def _type_id(cls) -> str:
    return f"Functor:{cls.__module__}.{cls.__name__}"


def _state() -> dict[str, torch.Tensor]:
    def tensor(first: float, second: float) -> torch.Tensor:
        return torch.tensor([first, second], dtype=torch.float64)

    return {
        "air_temperature": tensor(293.0, 300.0),
        "atmospheric_pressure": tensor(9.9e4, 1.0e5),
        "ppfd": tensor(600.0, 1400.0),
        "atmospheric_co2_ppm": tensor(415.0, 415.0),
        "vapor_pressure_deficit": tensor(800.0, 1600.0),
        "leaf_area_index": tensor(4.0, 5.0),
        "shortwave_down": tensor(300.0, 700.0),
        "cos_zenith": tensor(0.3, 0.7),
        "specific_humidity": tensor(0.008, 0.011),
        "co_mixing_ratio": tensor(1.2e-7, 1.2e-7),
        "ch4_mixing_ratio": tensor(1.9e-6, 1.9e-6),
        "pbl_height": tensor(800.0, 1400.0),
        "aerodynamic_resistance": tensor(25.0, 40.0),
        "friction_velocity": tensor(0.35, 0.6),
        "monin_obukhov_length": tensor(-150.0, -60.0),
        "o3_mixing_ratio": tensor(3.0e-8, 6.0e-8),
        "isoprene_mixing_ratio": tensor(2.0e-10, 6.0e-10),
        "no_mixing_ratio": tensor(1.5e-10, 2.5e-10),
        "no2_mixing_ratio": tensor(6.0e-10, 8.0e-10),
        "so2_mixing_ratio": tensor(5.0e-10, 5.0e-10),
    }


def _bound(names) -> dict:
    return {name: {"state": name} for name in names}


def _state_outputs(functor) -> list[str]:
    return [name for name in functor.metadata.output_variables if not name.startswith("diagnostics.")]


def _compiled():
    farquhar = FarquharPhotosynthesis(
        name="farquhar_photosynthesis", config=dict(FARQUHAR_CONFIG), **FARQUHAR_SELECTION
    )
    medlyn = MedlynStomatalConductance(name="medlyn_stomatal_conductance", config=dict(MEDLYN_CONFIG))
    stress = OzoneStress(name="ozone_stress")
    megan = MEGANModel(
        name="megan_bvoc_model", config=dict(MEGAN_CONFIG), emission_pathway=EmissionPathway.STANDARD
    )
    source = BoundaryLayerEmissionSource()
    deposition = DryDepositionAggregator(**FORCING_SELECTION)
    eddy = EddyDiffusionBLMFunctor(**FORCING_SELECTION)
    chemistry = ChemistryFunctor(name="atmospheric_chemistry", physical_boundary="open_mechanism")
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    for functor in (farquhar, medlyn, stress, megan, source, deposition, eddy, chemistry):
        registry.register_functor_instance(functor, node_type_id=_type_id(type(functor)))
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=list(_state())),
            "farquhar_photosynthesis": NodeSpec(
                class_type=_type_id(FarquharPhotosynthesis),
                config={**FARQUHAR_CONFIG, **FARQUHAR_SELECTION},
                inputs=_bound(
                    ["air_temperature", "atmospheric_co2_ppm", "ppfd", "vapor_pressure_deficit", "leaf_area_index"]
                ),
                outputs=["net_photosynthesis"],
            ),
            "medlyn_stomatal_conductance": NodeSpec(
                class_type=_type_id(MedlynStomatalConductance),
                config=dict(MEDLYN_CONFIG),
                inputs={
                    **_bound(["leaf_area_index", "ppfd", "atmospheric_co2_ppm", "vapor_pressure_deficit"]),
                    "net_photosynthesis": {"link": ["farquhar_photosynthesis", "net_photosynthesis"]},
                },
                outputs=["stomatal_conductance"],
            ),
            "ozone_stress": NodeSpec(
                class_type=_type_id(OzoneStress),
                config={},
                inputs={
                    "o3_mixing_ratio": {"state": "o3_mixing_ratio"},
                    "stomatal_conductance": {"link": ["medlyn_stomatal_conductance", "stomatal_conductance"]},
                },
                outputs=["ozone_stress_factor", "ozone_uptake", "antioxidant_depletion"],
            ),
            "megan_bvoc_model": NodeSpec(
                class_type=_type_id(MEGANModel),
                config={**MEGAN_CONFIG, "emission_pathway": "standard"},
                inputs={
                    **_bound(["air_temperature", "ppfd", "leaf_area_index", "atmospheric_co2_ppm"]),
                    "ozone_stress_factor": {"link": ["ozone_stress", "ozone_stress_factor"]},
                },
                outputs=["isoprene_emission"],
            ),
            "boundary_layer_emission_source": NodeSpec(
                class_type=_type_id(BoundaryLayerEmissionSource),
                config={},
                inputs={
                    "isoprene_emission": {"link": ["megan_bvoc_model", "isoprene_emission"]},
                    **_bound(["pbl_height", "air_temperature", "atmospheric_pressure"]),
                },
                outputs=["isoprene_mixing_ratio"],
            ),
            "dry_deposition_aggregator": NodeSpec(
                class_type=_type_id(DryDepositionAggregator),
                config=dict(FORCING_SELECTION),
                inputs=_bound(deposition.metadata.input_variables),
                outputs=_state_outputs(deposition),
            ),
            "eddy_diffusion_blm": NodeSpec(
                class_type=_type_id(EddyDiffusionBLMFunctor),
                config=dict(FORCING_SELECTION),
                inputs=_bound(eddy.metadata.input_variables),
                outputs=_state_outputs(eddy),
            ),
            "atmospheric_chemistry": NodeSpec(
                class_type=_type_id(ChemistryFunctor),
                config={"physical_boundary": "open_mechanism"},
                inputs=_bound(chemistry.metadata.input_variables),
                outputs=_state_outputs(chemistry),
            ),
        },
    )
    graph.links = graph._infer_links_from_inputs()
    return compile_graph(graph, registry, strict=True, auto_adapt=False), registry


def _executor(plan, registry) -> GraphExecutor:
    # The two forcing columns are independent cells and each patch fills its cell: identical source
    # and target descriptors map every cell onto itself, so no value is regridded.
    grid = {
        "grid_type": "regular",
        "coord_system": "cartesian",
        "shape": [2, 1],
        "spacing": [1.0, 1.0],
        "origin": [0.0, 0.0],
    }
    return GraphExecutor(
        plan,
        state_manager=StateManager(max_history_length=8, device=torch.device("cpu")),
        registry=registry,
        simulator_config={
            "device": "cpu",
            "require_gpu": False,
            "auto_report": False,
            "enable_auto_scale": True,
            "strict_scale_detection": True,
            "auto_scale": {"source_grid": dict(grid), "target_grid": dict(grid)},
        },
    )


@pytest.fixture
def transport_enabled():
    """The eddy process is opt-in: atmosphere.transport.enabled is false in the database."""
    manager = get_global_manager()
    original = manager.get("atmosphere.transport.enabled")
    manager.set_value("atmosphere.transport.enabled", True)
    yield
    manager.set_value("atmosphere.transport.enabled", original)


def test_fixed_model_compiles_strictly_without_a_turbulence_node(transport_enabled):
    plan, _ = _compiled()
    order = [step.node_id for step in plan.steps]
    for absent in ("atmospheric_turbulence", "environmental_forcing", "atmospheric_radiation"):
        assert absent not in order
    assert order.index("farquhar_photosynthesis") < order.index("medlyn_stomatal_conductance")
    assert plan.report.adapter_insertions == []
    assert plan.report.metadata["compile_run_witness"]["passed"] is True
    # Atmospheric ozone reaches the plant in the same step; emission then enters the air.
    assert order.index("atmospheric_chemistry") < order.index("ozone_stress")
    assert order.index("ozone_stress") < order.index("megan_bvoc_model")
    assert order.index("megan_bvoc_model") < order.index("boundary_layer_emission_source")


def test_two_governed_steps_keep_the_observed_surface_layer_and_write_every_species(transport_enabled):
    plan, registry = _compiled()
    state = _state()
    executor = _executor(plan, registry)
    with torch.no_grad():
        result = executor.execute({key: value.clone() for key, value in state.items()}, n_steps=2, dt=DT)

    assert executor.execution_summary["step_count"] == 2
    for name in SURFACE_LAYER:
        assert torch.equal(result[name], state[name]), name
    assert bool(torch.all(result["net_photosynthesis"] > 0.0))
    assert bool(torch.all(result["isoprene_emission"] > 0.0))
    assert bool(torch.all(result["ozone_stress_factor"] > 0.0))
    for name in ("o3_mixing_ratio", "isoprene_mixing_ratio", "no_mixing_ratio", "no2_mixing_ratio"):
        assert bool(torch.all(torch.isfinite(result[name]))) and not torch.equal(result[name], state[name])
    records = executor._simulator.get_variable_provenance("isoprene_mixing_ratio")
    assert {record["functor"] for record in records} == {
        "boundary_layer_emission_source",
        "eddy_diffusion_blm",
        "atmospheric_chemistry",
    }


def test_atmospheric_ozone_reaches_the_emission_of_the_same_graph(transport_enabled):
    plan, registry = _compiled()
    high = _state()
    low = {key: value.clone() for key, value in high.items()}
    # The stress factor sits at its floor of 0.1 at the ozone of _state, so the contrast lowers ozone.
    low["o3_mixing_ratio"] = low["o3_mixing_ratio"] * 0.1
    with torch.no_grad():
        result_low = _executor(plan, registry).execute(low, n_steps=2, dt=DT)
        result_high = _executor(plan, registry).execute(high, n_steps=2, dt=DT)
    for name in ("ozone_stress_factor", "isoprene_emission"):
        assert bool(torch.all(result_high[name] < result_low[name])), name
