from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
import torch
import xarray as xr
import yaml
import tutorial.case.run_case as run_case_module

from relic.core import device_policy
from relic.core.state import UpdateItem
from relic.contracts.contract_loader import ContractRegistry
from relic.core.functor import BiologicalProcessFunctor
from relic.models.pytorch_surrogate import ModelLoadError
from relic.parameters import get_global_manager, reset_global_manager
from relic.utils.physical_constants import get_physical_constant
from tutorial.case.functors.ml_surrogates import (
    CaseMEGANDiurnalActivitySurrogate,
    CaseIsopreneResidualSurrogate,
    CaseOzoneStressSurrogate,
    CaseSoilMoistureSurrogate,
    CaseStomatalSurrogate,
    CaseWaterStressSurrogate,
)
from tutorial.case.plot_results import _detect_ml_variables
from tutorial.case.plot_results import _load_parameter_change_context
from tutorial.case.run_case import (
    _convert_prediction_to_observation_units,
    _encode_bounded_factor_target,
    _persist_ml_surrogate_checkpoints,
    _get_registered_functor,
    _resolve_ml_training_device_for_functor,
    _restore_ml_surrogate_checkpoints_from_run,
    _simulator_has_ml_path,
    build_simulator,
    _resolve_case_runtime_device,
    compute_ml_path_diagnostics,
    load_optional_yaml,
    load_yaml,
    prepare_ml_surrogates,
    register_case_parameters,
    run_forward_pass,
)
from tutorial.case.utils.diagnostics_writer import DiagnosticsWriter
from tutorial.case.utils.netcdf_loader import CaseNetCDFLoader, LoaderConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _drop_case_parameter_sources(fresh_global_parameter_manager):
    """register_case_parameters adds the case parameter file (it overrides
    biosphere.bvoc.*, biosphere.photosynthesis.* and hydrosphere.* keys) as a
    source of the process-wide manager; every test works on a fresh manager and
    the manager that was global before is put back afterwards, so later test
    modules read the packaged database only."""
    yield


def _case_root() -> Path:
    return _repo_root() / "tutorial" / "case"


@contextmanager
def _workspace_tempdir(prefix: str):
    temp_root = _repo_root() / "tmp_test_artifacts"
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = temp_root / f"{prefix}{uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield work_dir
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _build_loader() -> CaseNetCDFLoader:
    case_root = _case_root()
    processed_dir = case_root / "data" / "processed"
    config_dir = case_root / "config"
    return CaseNetCDFLoader(
        LoaderConfig(
            case_root=case_root,
            forcing_path=processed_dir / "site_forcing.nc",
            obs_path=processed_dir / "real_obs.nc",
            case_parameter_path=config_dir / "case_parameters.yaml",
            device=torch.device("cpu"),
        )
    )


def _load_case_configs(model_root: Path, *, ml_enabled: bool) -> tuple[dict, dict]:
    # Each load registers the case parameter sources on a new process-wide
    # manager. The replaced object lives only in this test's slot: the autouse
    # fresh_global_parameter_manager fixture puts the pre-test manager back.
    reset_global_manager()
    case_root = _case_root()
    config_dir = case_root / "config"
    register_case_parameters(config_dir / "case_parameters.yaml")

    manager = get_global_manager()
    model_dir = model_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    manager.set_value(
        "tutorial.case.ml_surrogate.soil_moisture.model_path",
        str((model_dir / "sm_surrogate_test.pth").resolve()),
        validate=False,
    )
    manager.set_value(
        "tutorial.case.ml_surrogate.stomatal.model_path",
        str((model_dir / "gs_surrogate_test.pth").resolve()),
        validate=False,
    )
    manager.set_value(
        "tutorial.case.ml_surrogate.water_stress.model_path",
        str((model_dir / "water_stress_surrogate_test.pth").resolve()),
        validate=False,
    )
    manager.set_value(
        "tutorial.case.ml_surrogate.ozone_stress.model_path",
        str((model_dir / "ozone_stress_surrogate_test.pth").resolve()),
        validate=False,
    )
    manager.set_value(
        "tutorial.case.ml_surrogate.diurnal_activity.model_path",
        str((model_dir / "diurnal_activity_surrogate_test.pth").resolve()),
        validate=False,
    )
    manager.set_value(
        "tutorial.case.ml_surrogate.isoprene_residual.model_path",
        str((model_dir / "isoprene_residual_surrogate_test.pth").resolve()),
        validate=False,
    )

    simulation_cfg = copy.deepcopy(load_yaml(config_dir / "simulation.yaml"))
    ml_cfg = copy.deepcopy(load_optional_yaml(config_dir / "ml_surrogate.yaml"))
    simulation_cfg["simulation"]["device"] = "cpu"
    simulation_cfg["simulation"]["require_gpu"] = False
    simulation_cfg["simulation"]["ml_surrogate"]["enabled"] = ml_enabled
    simulation_cfg["simulation"]["ml_surrogate"]["soil_moisture"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["water_stress"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["stomatal"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["ozone_stress"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["diurnal_activity"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["isoprene_residual"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["train_from_physics"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["training_epochs"] = 1
    simulation_cfg["simulation"]["ml_surrogate"]["observation_guidance"]["enabled"] = True
    simulation_cfg["simulation"]["ml_surrogate"]["observation_guidance"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["soil_moisture"]["training"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["water_stress"]["training"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["stomatal"]["training"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["ozone_stress"]["training"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["diurnal_activity"]["training"]["epochs"] = 1
    ml_cfg["ml_surrogates"]["isoprene_residual"]["training"]["epochs"] = 1
    return simulation_cfg, ml_cfg


def test_loader_exposes_most_boundary_conditions_from_parameter_ssot() -> None:
    case_root = _case_root()
    register_case_parameters(case_root / "config" / "case_parameters.yaml")
    loader = _build_loader()
    forcing = loader.get_forcing_step(0)
    manager = get_global_manager()

    roughness = float(
        manager.get(
            "coupling.scales.biosphere_atmosphere.roughness_length_momentum",
            required=True,
        )
    )
    displacement_fraction = float(
        manager.get(
            "coupling.scales.biosphere_atmosphere.zero_displacement_height_fraction",
            required=True,
        )
    )
    canopy_height = float(manager.get("case.site.canopy_height", required=True))

    assert torch.allclose(
        forcing["roughness_length_momentum"],
        torch.full((1, 1), roughness, dtype=torch.float32),
    )
    assert torch.allclose(
        forcing["displacement_height"],
        torch.full(
            (1, 1),
            displacement_fraction * canopy_height,
            dtype=torch.float32,
        ),
    )
    for source_key, alias_key in (
        ("ppfd", "par"),
        ("precipitation_flux", "precipitation"),
        ("co2_concentration", "atmospheric_co2_ppm"),
        ("air_temperature", "temperature"),
    ):
        assert source_key in forcing
        assert alias_key in forcing
        assert torch.allclose(forcing[source_key], forcing[alias_key])
    assert {"relative_humidity", "net_radiation"}.issubset(forcing)
    # The forcing carries contract names only: o3_mixing_ratio [mol mol-1], the
    # OzoneStress input, and vapor_pressure_deficit [Pa], the Farquhar and
    # Penman-Monteith input. The kPa copy 'vpd' and the retired ozone names have
    # no reader and no declared units.
    assert {"o3_mixing_ratio", "vapor_pressure_deficit"}.issubset(forcing)
    for retired in ("vpd", "o3_concentration", "ozone_concentration"):
        assert retired not in forcing

    # References:
    # - Guenther et al. (2012), MEGAN2.1 Eq. (2) and Section 2.5.3.
    # Mathematical expression:
    # - gamma_i = C_CE * LAI * gamma_P,i * gamma_T,i * gamma_A,i
    #   * gamma_SM,i * gamma_C,i.
    # Parameter meanings:
    # - LAI is the canonical foliage driver; any mass-density conversion is
    #   derived only inside the production MEGAN process that consumes it.
    assert "foliar_density" not in forcing
    assert manager.get(
        "case.biosphere.foliar_density_factor",
        default=None,
        required=False,
    ) is None

    initial_state = loader.make_initial_state()
    expected_soil_moisture = loader.initial_soil_moisture.to(
        dtype=torch.float32,
        device=loader.device,
    ).reshape(1, 1)
    assert torch.allclose(initial_state["soil_moisture"], expected_soil_moisture)
    assert "foliar_density" not in initial_state


def test_isoprene_prediction_conversion_uses_model_molar_flux_units() -> None:
    case_root = _case_root()
    register_case_parameters(case_root / "config" / "case_parameters.yaml")
    loader = _build_loader()
    manager = get_global_manager()
    predicted_mol_flux = torch.tensor([1.0e-6], dtype=torch.float32)

    # References:
    # - Guenther et al. (2012), MEGAN species flux is represented as mol m-2 s-1.
    # Mathematical expression:
    # - E_ug = E_mol * MW / c_ug_to_g
    # Parameter meanings:
    # - E_mol: modeled isoprene molar flux [mol m-2 s-1]
    # - MW: isoprene molecular weight [g mol-1]
    # - c_ug_to_g: micrograms-to-grams conversion [g ug-1]
    molecular_weight = float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene"
        )
    )
    micrograms_to_grams = float(
        get_physical_constant("relic.physical.constants.conversions.micrograms_to_grams")
    )
    expected = predicted_mol_flux * (molecular_weight / micrograms_to_grams)

    actual = _convert_prediction_to_observation_units(
        loader,
        "isoprene_emission",
        predicted_mol_flux,
    )

    assert torch.allclose(actual, expected)


def test_bounded_factor_target_encoding_round_trips_through_sigmoid() -> None:
    target = torch.tensor([0.2, 0.8, 2.4], dtype=torch.float32)
    lower = 0.1
    upper = 3.0

    # References:
    # - Standard logistic link for bounded regression.
    # Mathematical expression:
    # - z = log((f - f_min) / (f_max - f))
    # - f = f_min + sigmoid(z) * (f_max - f_min)
    # Parameter meanings:
    # - f: bounded multiplicative correction factor [-]
    # - z: unconstrained network target [-]
    encoded = _encode_bounded_factor_target(target, lower=lower, upper=upper)
    decoded = lower + torch.sigmoid(encoded) * (upper - lower)

    assert torch.allclose(decoded, target, atol=1.0e-6, rtol=1.0e-6)


def test_case_physics_contracts_authorize_canonical_writers_and_units() -> None:
    registry = ContractRegistry(strict=False)
    registry.load_contracts_dir(_repo_root() / "relic" / "contracts" / "variables")

    expected = {
        "evapotranspiration_flux": ("penman_monteith_et", "flux", "kg m-2 s-1"),
        "canopy_storage": ("penman_monteith_et", "state", "mm"),
        "diagnostics.evapotranspiration": (
            "penman_monteith_et",
            "diagnostic",
            "dimensionless",
        ),
        "soil_moisture": ("bucket_soil_moisture", "state", "m3 m-3"),
        "surface_runoff_flux": ("bucket_soil_moisture", "flux", "kg m-2 s-1"),
        "drainage_flux": ("bucket_soil_moisture", "flux", "kg m-2 s-1"),
        "temperature_stress_factor": ("thermal_stress", "diagnostic", "dimensionless"),
        "heat_stress_factor": ("thermal_stress", "diagnostic", "dimensionless"),
        "cold_stress_factor": ("thermal_stress", "diagnostic", "dimensionless"),
        "electron_transport_rate": (
            "farquhar_photosynthesis",
            "flux",
            "umol e- m-2 s-1",
        ),
    }
    for variable, (writer, kind, units) in expected.items():
        contract = registry.get_contract(variable)
        assert contract is not None, variable
        assert registry.can_write(variable, writer), variable
        assert contract.kind == kind, variable
        assert contract.udunits == units, variable

    # The registered case bucket writes exactly its declared outputs; the
    # water storage and moisture change rate are entries of the bucket
    # diagnostics mapping (diagnostics.soil_moisture_bucket), not state writes.
    from tutorial.case.functors.case_relic_wrappers import CaseBucketSoilMoisture

    bucket = CaseBucketSoilMoisture(name="bucket_soil_moisture")
    outputs = list(bucket.metadata.output_variables)
    assert outputs == [
        "soil_moisture",
        "surface_runoff_flux",
        "drainage_flux",
        "diagnostics.soil_moisture_bucket",
    ]
    for variable in outputs:
        assert registry.can_write(variable, bucket.name), variable
    for diagnostic_entry in ("soil_moisture_change", "soil_water_storage"):
        assert diagnostic_entry not in outputs


def test_resolve_case_runtime_device_uses_device_policy(monkeypatch) -> None:
    simulation_cfg, _ = _load_case_configs(Path("."), ml_enabled=False)
    simulation_cfg["simulation"]["device"] = "auto"
    simulation_cfg["simulation"]["require_gpu"] = False
    captured: dict[str, object] = {}

    def _fake_resolve_device(*, explicit_device=None, default="cpu", require_gpu=False, **kwargs):
        captured["explicit_device"] = explicit_device
        captured["default"] = default
        captured["require_gpu"] = require_gpu
        captured["extra_kwargs"] = dict(kwargs)
        return torch.device("cpu")

    monkeypatch.setattr(device_policy, "resolve_device", _fake_resolve_device)
    resolved = _resolve_case_runtime_device(simulation_cfg)

    assert resolved == torch.device("cpu")
    assert captured["explicit_device"] == "auto"
    assert captured["default"] == "auto"
    assert captured["require_gpu"] is False


def test_ml_training_device_selector_uses_gpu_only_for_eligible_surrogates(monkeypatch) -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=True)
        simulation_cfg["simulation"]["ml_surrogate"]["training_acceleration"]["min_workload"] = 10
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)

        soil_functor = _get_registered_functor(simulator, "ml_soil_moisture")
        water_functor = _get_registered_functor(simulator, "ml_water_stress")
        assert isinstance(soil_functor, CaseSoilMoistureSurrogate)
        assert isinstance(water_functor, CaseWaterStressSurrogate)

        def _fake_resolve_device(*, explicit_device=None, default="cpu", require_gpu=False, **kwargs):
            del explicit_device, default, require_gpu, kwargs
            return torch.device("cuda:0")

        monkeypatch.setattr(device_policy, "resolve_device", _fake_resolve_device)

        soil_device = _resolve_ml_training_device_for_functor(
            sim_cfg=simulation_cfg,
            functor_name="ml_soil_moisture",
            functor=soil_functor,
            sample_count=100,
            epochs=50,
        )
        water_device = _resolve_ml_training_device_for_functor(
            sim_cfg=simulation_cfg,
            functor_name="ml_water_stress",
            functor=water_functor,
            sample_count=100,
            epochs=50,
        )

        assert soil_device.type == "cuda"
        assert water_device.type == "cpu"


def test_build_simulator_keeps_ml_surrogates_in_post_step_aux_path() -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=True)

        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)

        assert "atmospheric_turbulence" in simulator.functors
        assert "penman_monteith_et" in simulator.functors
        assert "water_stress" in simulator.functors
        assert "farquhar_photosynthesis" in simulator.functors
        assert "megan_bvoc_model" in simulator.functors
        assert "turbulence" not in simulator.functors
        assert "penman_monteith" not in simulator.functors
        assert "stress_adapter" not in simulator.functors
        assert "farquhar" not in simulator.functors
        assert "megan_bvoc" not in simulator.functors
        simulator.initialize_state(_build_loader().make_initial_state())
        assert simulator.run_specs is not None
        assert simulator.run_specs.default_mode == "replace"

        expected_aux_types = {
            "ml_soil_moisture": CaseSoilMoistureSurrogate,
            "ml_water_stress": CaseWaterStressSurrogate,
            "ml_stomatal": CaseStomatalSurrogate,
            "ml_ozone_stress": CaseOzoneStressSurrogate,
            "ml_diurnal_activity": CaseMEGANDiurnalActivitySurrogate,
            "ml_isoprene_residual": CaseIsopreneResidualSurrogate,
        }
        expected_architectures = {
            "ml_soil_moisture": "resnet",
            "ml_water_stress": "mlp",
            "ml_stomatal": "resnet",
            "ml_ozone_stress": "mlp",
            "ml_diurnal_activity": "mlp",
            "ml_isoprene_residual": "resnet",
        }
        for name, expected_type in expected_aux_types.items():
            assert name not in simulator.functors
            functor = _get_registered_functor(simulator, name)
            assert isinstance(functor, expected_type)
            assert functor.architecture_config["type"] == expected_architectures[name]

        ozone_inputs = list(
            _get_registered_functor(simulator, "ml_ozone_stress").metadata.input_variables
        )
        assert "o3_mixing_ratio" in ozone_inputs
        assert "ozone_concentration" not in ozone_inputs

        # Every surrogate input is a committed state variable with a declared unit.
        # MEGAN's environmental stress and CO2 inhibition factors are namespaced
        # diagnostics (diagnostics.bvoc.megan.*), never state: the residual
        # surrogate reads the committed temperature_stress_factor and the CO2
        # driver atmospheric_co2_ppm of the MEGAN CO2 inhibition instead.
        for name in expected_aux_types:
            functor = _get_registered_functor(simulator, name)
            functor._get_expected_input_units()
        residual_inputs = list(
            _get_registered_functor(simulator, "ml_isoprene_residual").input_variables
        )
        assert "environmental_stress_factor" not in residual_inputs
        assert "co2_inhibition_factor" not in residual_inputs
        assert {"temperature_stress_factor", "atmospheric_co2_ppm"}.issubset(residual_inputs)


def test_case_surrogates_forward_returns_exactly_their_declared_outputs() -> None:
    """Each case surrogate predicts its declared output from its declared inputs.

    The case consumes no per-output uncertainty entry, so no surrogate requests
    uncertainty_quantification (the SurrogateFunctor option that requires an
    _estimate_uncertainty estimator); the MC-dropout spread of uncertainty.method
    stays available through get_prediction_uncertainty()."""
    with _workspace_tempdir(prefix="relic_ml_surrogate_kernel_") as temp_dir:
        _load_case_configs(temp_dir, ml_enabled=True)
        surrogate_types = (
            CaseSoilMoistureSurrogate,
            CaseWaterStressSurrogate,
            CaseStomatalSurrogate,
            CaseOzoneStressSurrogate,
            CaseMEGANDiurnalActivitySurrogate,
            CaseIsopreneResidualSurrogate,
        )
        for surrogate_type in surrogate_types:
            surrogate = surrogate_type(model_dir=temp_dir / "kernel_models")
            assert surrogate.uncertainty_quantification is False
            state = {
                str(surrogate.input_mapping.get(name, name)): torch.full((2,), 0.5)
                for name in surrogate.input_variables
            }
            outputs = surrogate.forward(state, 0.0, 3600.0)
            assert set(outputs) == set(surrogate.output_variables), surrogate_type.__name__
            for name in surrogate.output_variables:
                assert bool(torch.all(torch.isfinite(outputs[name])))


def test_surrogate_checkpoint_loads_only_into_its_feature_definition() -> None:
    """A saved checkpoint records the ordered input and output names with the
    weights; loading it into a surrogate whose declared features differ (or from a
    file without the names) raises ModelLoadError naming the retraining path,
    never a silent fresh initialization or a silent use of mismatched weights."""
    with _workspace_tempdir(prefix="relic_ml_checkpoint_features_") as temp_dir:
        _load_case_configs(temp_dir, ml_enabled=True)
        model_dir = temp_dir / "feature_models"
        source = CaseOzoneStressSurrogate(model_dir=model_dir)
        checkpoint_path = model_dir / "ozone_stress_surrogate.pth"
        source.save_model(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert payload["input_variables"] == list(source.input_variables)
        assert payload["output_variables"] == list(source.output_variables)

        restored = CaseOzoneStressSurrogate(model_dir=model_dir)
        for name, tensor in source.model.state_dict().items():
            assert torch.equal(restored.model.state_dict()[name], tensor)

        stale = dict(payload)
        stale["input_variables"] = [
            "ozone_concentration" if name == "o3_mixing_ratio" else name
            for name in payload["input_variables"]
        ]
        torch.save(stale, checkpoint_path)
        with pytest.raises(ModelLoadError, match="retrain"):
            CaseOzoneStressSurrogate(model_dir=model_dir)

        unnamed = {key: value for key, value in payload.items() if key != "input_variables"}
        torch.save(unnamed, checkpoint_path)
        with pytest.raises(ModelLoadError, match="input features"):
            CaseOzoneStressSurrogate(model_dir=model_dir)


def test_shipped_case_checkpoints_match_the_current_surrogate_definitions() -> None:
    """The six checkpoints the case parameters point to (tutorial/case/models) were
    trained on the feature vectors the surrogates declare now."""
    with _workspace_tempdir(prefix="relic_ml_shipped_models_") as temp_dir:
        _load_case_configs(temp_dir, ml_enabled=True)
        model_dir = _case_root() / "models"
        for surrogate_type in (
            CaseSoilMoistureSurrogate,
            CaseWaterStressSurrogate,
            CaseStomatalSurrogate,
            CaseOzoneStressSurrogate,
            CaseMEGANDiurnalActivitySurrogate,
            CaseIsopreneResidualSurrogate,
        ):
            surrogate = surrogate_type(model_dir=model_dir)
            assert surrogate.is_trained is True, surrogate_type.__name__


def test_build_simulator_registers_ml_surrogates_as_bridge_functors(caplog) -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=True)

        with caplog.at_level("WARNING"):
            simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)

        for functor_name in (
            "ml_soil_moisture",
            "ml_water_stress",
            "ml_stomatal",
            "ml_ozone_stress",
            "ml_diurnal_activity",
            "ml_isoprene_residual",
        ):
            assert functor_name not in simulator.functors
            assert isinstance(
                _get_registered_functor(simulator, functor_name),
                BiologicalProcessFunctor,
            )

        assert not any(
            "Non-bridge functor" in record.getMessage() for record in caplog.records
        )


def test_build_simulator_skips_ml_surrogates_when_disabled() -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=False)

        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)

        assert "ml_soil_moisture" not in simulator.functors
        assert "ml_water_stress" not in simulator.functors
        assert "ml_stomatal" not in simulator.functors
        assert "ml_ozone_stress" not in simulator.functors
        assert "ml_isoprene_residual" not in simulator.functors
        assert _get_registered_functor(simulator, "ml_diurnal_activity") is None


def test_build_simulator_declares_case_topology_through_the_dependency_api() -> None:
    """The case composition declares dependencies only through
    set_dependencies/add_dependency (read back through get_dependencies); it
    writes no private _dependencies list and no optional_dependencies attribute."""
    import ast

    from relic.core.dependency_graph import normalize_dependency_declaration

    source = (_case_root() / "run_case.py").read_text(encoding="utf-8")
    written = {
        target.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }
    assert "_dependencies" not in written
    assert "optional_dependencies" not in written

    with _workspace_tempdir(prefix="relic_case_topology_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=False)
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)

    def declaration(name):
        functor = simulator.functors[name]
        return normalize_dependency_declaration(
            functor.get_dependencies(), owner=name
        ).canonical()

    assert "precip_adapter" in declaration("surface_runoff")["required"]
    assert declaration("precip_adapter")["required"] == []
    assert declaration("farquhar_photosynthesis")["optional"] == []


def test_physics_step_preserves_vpd_and_ppfd_contract_forcings() -> None:
    with _workspace_tempdir(prefix="relic_case_forcing_contract_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=False)
        loader = _build_loader()
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)
        simulator.initialize_state(loader.make_initial_state())
        simulator._compute_group_execution_orders()
        fast_order = list(simulator._order_fast)

        assert "infiltration_bridge" not in simulator.functors
        assert fast_order.index("penman_monteith_et") < fast_order.index(
            "bucket_soil_moisture"
        )
        assert fast_order.index("surface_runoff") < fast_order.index(
            "bucket_soil_moisture"
        )

        forcing = loader.get_forcing_step(15)
        simulator.state_manager.set_state(forcing, validate=False)
        simulator.step(float(simulation_cfg["simulation"]["dt_seconds"]))
        state = simulator.state_manager.get_state(clone=True)
        evapotranspiration = state["evapotranspiration_flux"]
        et_provenance = simulator.get_variable_provenance(
            "evapotranspiration_flux"
        )

        assert torch.allclose(
            state["vapor_pressure_deficit"],
            forcing["vapor_pressure_deficit"],
        )
        assert torch.allclose(state["ppfd"], forcing["ppfd"])
        assert bool(torch.all(torch.isfinite(evapotranspiration)))
        assert bool(torch.all(evapotranspiration >= 0.0))
        assert any(
            record["functor"] == "penman_monteith_et"
            and record["role"] == "final_writer"
            for record in et_provenance
        )
        assert float(state["net_photosynthesis"].detach().reshape(-1)[0]) > 0.0


def test_run_forward_pass_disables_ml_progress_and_outputs_when_ml_disabled(monkeypatch) -> None:
    with _workspace_tempdir(prefix="relic_ml_disabled_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=False)
        loader = _build_loader()
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)
        simulator.initialize_state(loader.make_initial_state())
        diagnostics = DiagnosticsWriter(temp_dir / "run")
        dt_seconds = float(simulation_cfg["simulation"]["dt_seconds"])
        created_bars: list[dict[str, object]] = []

        def _fake_create_progress_bar(*, enabled, total, desc, position, leave):
            created_bars.append(
                {
                    "enabled": enabled,
                    "total": total,
                    "desc": desc,
                    "position": position,
                    "leave": leave,
                }
            )
            return None

        monkeypatch.setattr(run_case_module, "_create_progress_bar", _fake_create_progress_bar)

        def reset_state() -> None:
            simulator.current_time = 0.0
            simulator.step_count = 0
            simulator.state_manager.reset()
            simulator.state_manager.set_state(loader.make_initial_state(), validate=False)

        assert _simulator_has_ml_path(simulator) is False
        run_forward_pass(
            simulator=simulator,
            loader=loader,
            diagnostics=diagnostics,
            steps=1,
            dt_seconds=dt_seconds,
            reset_state=reset_state,
            epoch_label=0,
            enable_grad=False,
            phase_name="simulate",
            log_fn=lambda _: None,
            monitoring_cfg={
                "enabled": True,
                "log_interval_steps": 1,
                "checkpoint_interval_steps": 1,
            },
            show_progress=True,
        )
        diagnostics.write_process_monitor()
        diagnostics.write_timeseries()

        assert len(created_bars) == 2
        assert created_bars[0]["desc"] == "simulate physics"
        assert created_bars[0]["enabled"] is True
        assert created_bars[1]["desc"] == "simulate ML"
        assert created_bars[1]["enabled"] is False
        ml_ds = xr.load_dataset(temp_dir / "run" / "timeseries" / "state_hourly_ML.nc")
        assert list(ml_ds.data_vars) == []


def test_detect_ml_variables_maps_only_existing_pairs() -> None:
    state_ds = xr.Dataset(
        data_vars={
            "soil_moisture": ("sample", [0.22, 0.24]),
            "soil_moisture_ML": ("sample", [0.25, 0.26]),
            "stomatal_conductance": ("sample", [0.18, 0.20]),
            "stomatal_conductance_ML": ("sample", [0.17, 0.19]),
            "orphan_ML": ("sample", [1.0, 2.0]),
        }
    )

    mapping = _detect_ml_variables(state_ds)

    assert mapping == {
        "soil_moisture": "soil_moisture_ML",
        "stomatal_conductance": "stomatal_conductance_ML",
    }


def test_parameter_change_context_prefers_actual_start_values_over_yaml_defaults() -> None:
    with _workspace_tempdir(prefix="relic_plot_params_") as temp_dir:
        run_dir = temp_dir / "run"
        input_snapshot = run_dir / "input_snapshot"
        optimization_dir = run_dir / "optimization"
        input_snapshot.mkdir(parents=True, exist_ok=True)
        optimization_dir.mkdir(parents=True, exist_ok=True)

        case_payload = {
            "parameters": {
                "hydrosphere.soil_moisture.field_capacity": {
                    "value": 0.4,
                    "units": "m3 m-3",
                },
                "biosphere.bvoc.default.isoprene": {
                    "value": 24.0,
                    "units": "ug g-1 DW h-1",
                },
            }
        }
        optimized_payload = {
            "hydrosphere.soil_moisture.field_capacity": 0.42,
            "biosphere.bvoc.default.isoprene": 23.5,
        }
        actual_start_payload = {
            "hydrosphere.soil_moisture.field_capacity": 0.41,
            "biosphere.bvoc.default.isoprene": 23.9,
        }

        (input_snapshot / "case_parameters.yaml").write_text(
            yaml.safe_dump(case_payload, sort_keys=True),
            encoding="utf-8",
        )
        (optimization_dir / "optimized_parameters.yaml").write_text(
            yaml.safe_dump(optimized_payload, sort_keys=True),
            encoding="utf-8",
        )
        (optimization_dir / "initial_trainable_parameters.yaml").write_text(
            yaml.safe_dump(actual_start_payload, sort_keys=True),
            encoding="utf-8",
        )

        init_vals, opt_vals, units, source_label = _load_parameter_change_context(
            run_dir,
            [
                ("hydrosphere.soil_moisture.field_capacity", "FC"),
                ("biosphere.bvoc.default.isoprene", "IsopEF"),
            ],
        )

        assert source_label == "start"
        assert list(init_vals) == [0.41, 23.9]
        assert list(opt_vals) == [0.42, 23.5]
        assert units == ["m3 m-3", "ug g-1 DW h-1"]


def test_prepare_ml_surrogates_reuses_one_physics_sample_collection(
    monkeypatch,
) -> None:
    fake_functor = SimpleNamespace(is_trained=True)
    payload = {"fine_tune_samples": [{"step": 0}]}
    collection_calls = 0
    fine_tune_samples = None

    def fake_collect(**_kwargs):
        nonlocal collection_calls
        collection_calls += 1
        return payload

    def fake_train(**kwargs):
        training_payload = kwargs.get("training_payload")
        if training_payload is None:
            training_payload = fake_collect()
        assert training_payload is payload
        return {"ml_soil_moisture": {"epochs_trained": 1}}

    def fake_fine_tune(**kwargs):
        nonlocal fine_tune_samples
        fine_tune_samples = kwargs["fine_tune_samples"]
        return {"epochs": 1}

    monkeypatch.setattr(
        run_case_module,
        "_collect_ml_surrogate_functors",
        lambda _simulator: {"ml_soil_moisture": fake_functor},
    )
    monkeypatch.setattr(
        run_case_module,
        "_get_registered_functor",
        lambda _simulator, _name: fake_functor,
    )
    monkeypatch.setattr(
        run_case_module,
        "_resolve_ml_runtime_settings",
        lambda _config: {
            "train_from_physics": True,
            "observation_guidance": {"enabled": True},
        },
    )
    monkeypatch.setattr(run_case_module, "_collect_ml_training_samples", fake_collect)
    monkeypatch.setattr(run_case_module, "train_ml_surrogates", fake_train)
    monkeypatch.setattr(
        run_case_module,
        "fine_tune_ml_surrogates_with_observations",
        fake_fine_tune,
    )

    summary = prepare_ml_surrogates(
        simulator=object(),
        sim_cfg={},
        ml_cfg=None,
        loader=object(),
        steps=4,
        dt_seconds=3600.0,
        log_fn=lambda _message: None,
    )

    assert collection_calls == 1
    assert fine_tune_samples is payload["fine_tune_samples"]
    assert summary["observation_guidance"]["epochs"] == 1


def test_ml_dual_path_training_forward_and_gradient_chain() -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=True)
        loader = _build_loader()
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)
        simulator.initialize_state(loader.make_initial_state())

        dt_seconds = float(simulation_cfg["simulation"]["dt_seconds"])
        training_summary = prepare_ml_surrogates(
            simulator=simulator,
            sim_cfg=simulation_cfg,
            ml_cfg=ml_cfg,
            loader=loader,
            steps=4,
            dt_seconds=dt_seconds,
            log_fn=lambda _: None,
        )

        assert "ml_soil_moisture" in training_summary
        assert "ml_water_stress" in training_summary
        assert "ml_stomatal" in training_summary
        assert "ml_ozone_stress" in training_summary
        assert "ml_diurnal_activity" in training_summary
        assert "ml_isoprene_residual" in training_summary
        assert "observation_guidance" in training_summary

        state_manager = simulator.state_manager
        state_manager.reset()
        state_manager.set_state(loader.make_initial_state(), validate=False)
        state_manager.set_state(loader.get_forcing_step(0), validate=False)
        simulator.step(dt_seconds)
        state = state_manager.get_state(clone=False)

        assert "soil_moisture_ML" not in state
        assert "water_stress_factor_ML" not in state
        assert "stomatal_conductance_ML" not in state
        assert "ozone_stress_factor_ML" not in state
        assert "isoprene_correction_factor_ML" not in state

        ml_outputs = compute_ml_path_diagnostics(
            simulator=simulator,
            state=state,
            time_seconds=float(simulator.current_time),
            dt_seconds=dt_seconds,
        )

        required_ml_keys = {
            "soil_moisture_ML",
            "stomatal_conductance_ML",
            "water_stress_factor_ML",
            "ozone_stress_factor_ML",
            "isoprene_activity_correction_ML",
            "isoprene_correction_factor_ML",
            "net_photosynthesis_ML",
            "isoprene_emission_ML",
        }
        assert required_ml_keys.issubset(ml_outputs.keys())

        grad_keys = {
            key
            for key, value in ml_outputs.items()
            if isinstance(value, torch.Tensor) and (value.requires_grad or value.grad_fn is not None)
        }
        assert "soil_moisture_ML" in grad_keys
        assert "stomatal_conductance_ML" in grad_keys
        assert "net_photosynthesis_ML" in grad_keys
        assert "ozone_stress_factor_ML" in grad_keys
        assert "isoprene_activity_correction_ML" in grad_keys
        assert "isoprene_correction_factor_ML" in grad_keys
        assert "isoprene_emission_ML" in grad_keys

        if "leaf_water_potential" in state and "water_stress_ML" in ml_outputs:
            farquhar_state = dict(state)
            farquhar_state.pop("leaf_water_potential", None)
            farquhar_state["water_stress"] = ml_outputs["water_stress_ML"]
            expected_farquhar = simulator.functors["farquhar_photosynthesis"].forward(
                farquhar_state,
                float(simulator.current_time),
                dt_seconds,
            )
            expected_net = expected_farquhar["net_photosynthesis"]
            assert isinstance(expected_net, UpdateItem)
            # The physics path follows the runtime state dtype (float64 forcing) while the ML
            # diagnostics are float32 surrogate outputs; compare values in one dtype.
            assert torch.allclose(
                ml_outputs["net_photosynthesis_ML"],
                expected_net.value.to(ml_outputs["net_photosynthesis_ML"].dtype),
                atol=1e-6,
                rtol=1e-5,
            )

        soil_surrogate = _get_registered_functor(simulator, "ml_soil_moisture")
        water_surrogate = _get_registered_functor(simulator, "ml_water_stress")
        stomatal_surrogate = _get_registered_functor(simulator, "ml_stomatal")
        ozone_surrogate = _get_registered_functor(simulator, "ml_ozone_stress")
        diurnal_surrogate = _get_registered_functor(simulator, "ml_diurnal_activity")
        residual_surrogate = _get_registered_functor(simulator, "ml_isoprene_residual")
        soil_surrogate.zero_grad(set_to_none=True)
        water_surrogate.zero_grad(set_to_none=True)
        stomatal_surrogate.zero_grad(set_to_none=True)
        ozone_surrogate.zero_grad(set_to_none=True)
        if isinstance(diurnal_surrogate, CaseMEGANDiurnalActivitySurrogate):
            diurnal_surrogate.zero_grad(set_to_none=True)
        residual_surrogate.zero_grad(set_to_none=True)

        loss = (
            ml_outputs["soil_moisture_ML"].mean()
            + ml_outputs["water_stress_factor_ML"].mean()
            + ml_outputs["stomatal_conductance_ML"].mean()
            + ml_outputs["ozone_stress_factor_ML"].mean()
            + ml_outputs["isoprene_activity_correction_ML"].mean()
            + ml_outputs["isoprene_correction_factor_ML"].mean()
            + ml_outputs["isoprene_emission_ML"].mean()
        )
        loss.backward()


def test_diagnostics_writer_splits_dual_path_outputs_and_monitor_files() -> None:
    with _workspace_tempdir(prefix="relic_ml_writer_") as temp_dir:
        diagnostics = DiagnosticsWriter(temp_dir)
        timestamp = pd.Timestamp("2014-06-01T00:00:00")
        diagnostics.log_step(
            epoch=0,
            step=0,
            time=timestamp,
            phase_name="simulate",
            state={
                "soil_moisture": torch.tensor([0.25], dtype=torch.float32),
                "soil_moisture_ML": torch.tensor([0.27], dtype=torch.float32),
                "co2_concentration": torch.tensor([401.0], dtype=torch.float32),
                "co2_intercellular": torch.tensor([401.0], dtype=torch.float32),
                "atmospheric_co2_ppm": torch.tensor([401.0], dtype=torch.float32),
                "gross_primary_productivity": torch.tensor([0.88], dtype=torch.float32),
                "isoprene_emission": torch.tensor([1.2], dtype=torch.float32),
                "isoprene_emission_ML": torch.tensor([1.4], dtype=torch.float32),
                "diagnostics.example_metric": torch.tensor([2.0], dtype=torch.float32),
            },
            extra_keys=["soil_moisture_ML", "isoprene_emission_ML", "diagnostics.example_metric"],
        )

        monitor_outputs = diagnostics.write_process_monitor()
        diagnostics.write_timeseries()

        physics_ds = xr.load_dataset(temp_dir / "timeseries" / "state_hourly_physics.nc")
        ml_ds = xr.load_dataset(temp_dir / "timeseries" / "state_hourly_ML.nc")
        assert "soil_moisture" in physics_ds.data_vars
        assert "soil_moisture_ML" not in physics_ds.data_vars
        assert "gross_primary_productivity" in physics_ds.data_vars
        assert "co2_concentration" in physics_ds.data_vars
        assert "isoprene_emission_ML" in ml_ds.data_vars
        assert "soil_moisture" not in ml_ds.data_vars
        assert physics_ds["isoprene_emission"].attrs["units"] == "mol m-2 s-1"
        assert ml_ds["isoprene_emission_ML"].attrs["units"] == "mol m-2 s-1"

        physics_monitor = pd.read_csv(monitor_outputs["physics"])
        ml_monitor = pd.read_csv(monitor_outputs["ml"])
        assert "soil_moisture" in physics_monitor.columns
        assert "soil_moisture_ML" not in physics_monitor.columns
        assert "gross_primary_productivity" in physics_monitor.columns
        assert "soil_moisture_ML" in ml_monitor.columns
        assert "soil_moisture" not in [col for col in ml_monitor.columns if col not in {"epoch", "step", "phase_id", "time"}]


def test_run_forward_pass_writes_monitor_and_simulation_checkpoints() -> None:
    with _workspace_tempdir(prefix="relic_ml_surrogate_") as temp_dir:
        simulation_cfg, ml_cfg = _load_case_configs(temp_dir, ml_enabled=True)
        loader = _build_loader()
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)
        simulator.initialize_state(loader.make_initial_state())
        dt_seconds = float(simulation_cfg["simulation"]["dt_seconds"])
        # The declared training.validation_split = 0.2 holds out round(0.2 n)
        # samples and needs at least one training and one validation sample,
        # so the surrogates train on n = 4 physics samples.
        prepare_ml_surrogates(
            simulator=simulator,
            sim_cfg=simulation_cfg,
            ml_cfg=ml_cfg,
            loader=loader,
            steps=4,
            dt_seconds=dt_seconds,
            log_fn=lambda _: None,
        )
        diagnostics = DiagnosticsWriter(temp_dir / "run")

        def reset_state() -> None:
            simulator.current_time = 0.0
            simulator.step_count = 0
            simulator.state_manager.reset()
            simulator.state_manager.set_state(loader.make_initial_state(), validate=False)

        run_forward_pass(
            simulator=simulator,
            loader=loader,
            diagnostics=diagnostics,
            steps=2,
            dt_seconds=dt_seconds,
            reset_state=reset_state,
            epoch_label=0,
            enable_grad=False,
            phase_name="simulate",
            log_fn=lambda _: None,
            monitoring_cfg={
                "enabled": True,
                "log_interval_steps": 1,
                "checkpoint_interval_steps": 1,
            },
        )
        monitor_outputs = diagnostics.write_process_monitor()
        diagnostics.write_timeseries()

        checkpoint_dir = temp_dir / "run" / "checkpoints"
        checkpoint_files = sorted(checkpoint_dir.glob("simulate_simulate_step*.pt"))
        assert checkpoint_files
        latest_path = checkpoint_dir / "simulate_simulate_latest.pt"
        assert latest_path.exists()
        payload = torch.load(latest_path, map_location="cpu", weights_only=True)
        assert "state_physics" in payload
        assert "state_ml" in payload
        assert "soil_moisture" in payload["state_physics"]
        assert "gross_primary_productivity" in payload["state_physics"]
        assert "soil_moisture_ML" in payload["state_ml"]

        assert monitor_outputs["combined"].exists()
        assert (temp_dir / "run" / "timeseries" / "state_hourly_physics.nc").exists()
        assert (temp_dir / "run" / "timeseries" / "state_hourly_ML.nc").exists()
        physics_ds = xr.load_dataset(
            temp_dir / "run" / "timeseries" / "state_hourly_physics.nc"
        )
        runoff_coefficient = physics_ds["runoff_coefficient"].values
        assert bool(pd.notna(runoff_coefficient).all())
        assert float(runoff_coefficient.min()) >= 0.0
        assert float(runoff_coefficient.max()) <= 1.0
        physics_monitor = pd.read_csv(monitor_outputs["physics"])
        assert "gross_primary_productivity" in physics_monitor.columns
        assert float(physics_monitor["gross_primary_productivity"].iloc[0]) >= 0.0


def test_restore_ml_surrogate_checkpoints_from_run_preserves_normalizers() -> None:
    with _workspace_tempdir(prefix="relic_ml_resume_source_") as source_dir:
        simulation_cfg, ml_cfg = _load_case_configs(source_dir, ml_enabled=True)
        loader = _build_loader()
        simulator = build_simulator(simulation_cfg, ml_cfg=ml_cfg)
        simulator.initialize_state(loader.make_initial_state())
        dt_seconds = float(simulation_cfg["simulation"]["dt_seconds"])
        prepare_ml_surrogates(
            simulator=simulator,
            sim_cfg=simulation_cfg,
            ml_cfg=ml_cfg,
            loader=loader,
            steps=4,
            dt_seconds=dt_seconds,
            log_fn=lambda _: None,
        )

        source_run_dir = source_dir / "resume_source_run"
        source_diagnostics = DiagnosticsWriter(source_run_dir)
        manifest = _persist_ml_surrogate_checkpoints(
            simulator=simulator,
            diagnostics=source_diagnostics,
            log_fn=lambda _: None,
        )
        assert manifest

        soil_source = _get_registered_functor(simulator, "ml_soil_moisture")
        assert isinstance(soil_source._input_mean, torch.Tensor)
        assert isinstance(soil_source._output_scale, torch.Tensor)
        source_input_mean = soil_source._input_mean.detach().clone()
        source_output_scale = soil_source._output_scale.detach().clone()

        with _workspace_tempdir(prefix="relic_ml_resume_target_") as target_dir:
            target_simulation_cfg, target_ml_cfg = _load_case_configs(target_dir, ml_enabled=True)
            target_simulator = build_simulator(target_simulation_cfg, ml_cfg=target_ml_cfg)

            restored_manifest = _restore_ml_surrogate_checkpoints_from_run(
                simulator=target_simulator,
                run_dir=source_run_dir,
                log_fn=lambda _: None,
            )

            assert restored_manifest.keys() == manifest.keys()
            soil_target = _get_registered_functor(target_simulator, "ml_soil_moisture")
            assert isinstance(soil_target._input_mean, torch.Tensor)
            assert isinstance(soil_target._output_scale, torch.Tensor)
            assert bool(getattr(soil_target, "is_trained", False)) is True
            assert torch.allclose(soil_target._input_mean, source_input_mean)
            assert torch.allclose(soil_target._output_scale, source_output_scale)
