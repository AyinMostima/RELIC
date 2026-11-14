"""Tests for stochastic IPHS structural noise contracts."""

import pytest

from relic.computation.operators.stochastic_sdae import (
    ALEGeometryCorrectionSpec,
    ConservativeNoiseFluxSpec,
    HybridTopologyTransitionSpec,
    MemoryKernelSpec,
    NoiseCovarianceSpec,
    StochasticTermSpec,
)


def test_noise_covariance_spec_validates_fdt_relation():
    spec = NoiseCovarianceSpec.from_mapping(
        {
            "name": "fdt_noise",
            "relation": "fdt",
            "sigma": [[2.0]],
            "dissipative_operator": [[1.0]],
            "covariance_scale": 4.0,
            "scale_parameter_key": "thermal.noise_scale",
            "physical_constant_keys": ["boltzmann_constant"],
        }
    )

    audit = spec.validate()

    assert audit["passed"] is True
    assert audit["residual"] == 0.0


def test_noise_covariance_spec_rejects_unmanaged_scale():
    spec = NoiseCovarianceSpec.from_mapping(
        {
            "name": "bad_noise",
            "relation": "coarse_grained",
            "sigma": [[1.0]],
            "dissipative_operator": [[1.0]],
            "covariance_scale": 1.0,
        }
    )

    audit = spec.validate()

    assert audit["passed"] is False
    assert any("scale_parameter_key" in issue for issue in audit["issues"])


def test_noise_covariance_spec_rejects_negative_scale_and_mobility():
    spec = NoiseCovarianceSpec.from_mapping(
        {
            "name": "sign_cancelled_fdt",
            "relation": "fdt",
            "sigma": [[1.0]],
            "dissipative_operator": [[-1.0]],
            "covariance_scale": -1.0,
            "scale_parameter_key": "thermal.noise_scale",
            "physical_constant_keys": ["boltzmann_constant"],
        }
    )

    audit = spec.validate()

    assert audit["passed"] is False
    assert any("nonnegative" in issue for issue in audit["issues"])
    assert any("positive semidefinite" in issue for issue in audit["issues"])


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("sigma", [[float("nan")]], "sigma must be finite"),
        ("dissipative_operator", [], "non-empty rank-2"),
        (
            "dissipative_operator",
            [[1.0, 2.0], [0.0, 1.0]],
            "symmetric",
        ),
        ("covariance_scale", float("inf"), "finite nonnegative"),
    ],
)
def test_noise_covariance_spec_rejects_invalid_numeric_domains(
    field,
    value,
    expected,
):
    payload = {
        "name": "invalid_fdt",
        "relation": "fdt",
        "sigma": [[1.0]],
        "dissipative_operator": [[1.0]],
        "covariance_scale": 1.0,
        "scale_parameter_key": "thermal.noise_scale",
        "physical_constant_keys": ["boltzmann_constant"],
    }
    payload[field] = value
    spec = NoiseCovarianceSpec.from_mapping(payload)

    audit = spec.validate()

    assert audit["passed"] is False
    assert any(expected in issue for issue in audit["issues"])


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_noise_covariance_spec_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance must be finite and nonnegative"):
        NoiseCovarianceSpec(
            name="invalid_tolerance",
            relation="fdt",
            tolerance=tolerance,
        )


def test_conservative_noise_flux_spec_preserves_topology_metadata():
    spec = ConservativeNoiseFluxSpec.from_mapping(
        {
            "name": "edge_noise",
            "flux_space": {"complex_id": "K", "degree": 1},
            "storage_space": {"complex_id": "K", "degree": 0},
            "incidence_operator": "B",
            # The routing flags that the audit evaluates are declared, not
            # defaulted to the passing values.
            "routes_to_storage": True,
            "direct_storage_noise": False,
        }
    )

    assert spec.topology_operator_name == "B"
    assert spec.routes_to_storage is True
    assert spec.direct_storage_noise is False


def test_stochastic_term_spec_requires_existing_owner():
    spec = StochasticTermSpec.from_mapping(
        {
            "name": "rainfall_noise",
            "classification": "external_forcing",
            "runtime_owner": "relic.stochastic.weather_generator",
        }
    )

    assert spec.runtime_owner == "relic.stochastic.weather_generator"


def test_memory_kernel_spec_preserves_passivity_metadata():
    spec = MemoryKernelSpec.from_mapping(
        {
            "name": "soil_delay_kernel",
            "kernel_operator": [[0.5]],
            "hidden_storage_space": {"complex_id": "K", "semantic_name": "memory_state"},
            "effort_variable": "e_h",
            "passivity_evidence": "positive_real_realization",
            "parameter_keys": ["memory.relaxation_time"],
            "residual_route": "diagnostics.memory_residual",
        }
    )

    assert spec.name == "soil_delay_kernel"
    assert spec.kernel_operator is not None
    assert spec.passivity_evidence == "positive_real_realization"


def test_ale_geometry_correction_spec_preserves_port_accounting():
    spec = ALEGeometryCorrectionSpec.from_mapping(
        {
            "name": "moving_mesh_energy_correction",
            "hodge_time_derivative": [[0.1]],
            "state_space": {"complex_id": "K", "semantic_name": "storage"},
            "energy_effort": "e_E",
            "correction_route": "geometry_port",
            "port_accounting": "diagnostics.geometry_power",
            "parameter_keys": ["geometry.mesh_velocity"],
        }
    )

    assert spec.hodge_time_derivative is not None
    assert spec.correction_route == "geometry_port"
    assert spec.port_accounting == "diagnostics.geometry_power"


def test_hybrid_topology_transition_spec_preserves_remap_metadata():
    spec = HybridTopologyTransitionSpec.from_mapping(
        {
            "name": "freeze_thaw_remap",
            "event_name": "freeze_thaw_phase_change",
            "source_complex": {"complex_id": "K_old", "semantic_name": "old_cells"},
            "target_complex": {"complex_id": "K_new", "semantic_name": "new_cells"},
            "state_remap": [[1.0, 0.0], [0.0, 1.0]],
            "ledger_map": [[1.0, 1.0]],
            "ledger_change": [[0.0]],
            "dimension_change": True,
            "reset_route": "hybrid_event",
            "parameter_keys": ["hybrid.phase_threshold"],
            "audit_route": "diagnostics.hybrid_topology_audit",
        }
    )

    assert spec.event_name == "freeze_thaw_phase_change"
    assert spec.state_remap is not None
    assert spec.reset_route == "hybrid_event"
    assert spec.parameter_keys == ("hybrid.phase_threshold",)


# MA-21: NoiseCovarianceSpec and JumpResetSpec tolerances default to ParameterManager
# keys; explicit values override; a declared null tolerance is not replaced.
# Test point: M_noise = diag(1, -5e-11), Sigma = diag(1, 0), scale 1: the PSD defect
# and the FDT residual ||Sigma Sigma^T - scale M_noise||_F are both 5e-11.
_NOISE_TOLERANCE_KEY = "validation.iphs.noise_covariance_tolerance"
_JUMP_TOLERANCE_KEY = "validation.iphs.jump_ledger_tolerance"


def _validation_iphs_entry(key):
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[3]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]


def _near_psd_noise(**overrides):
    payload = {
        "name": "near_psd_noise",
        "relation": "fdt",
        "sigma": [[1.0, 0.0], [0.0, 0.0]],
        "dissipative_operator": [[1.0, 0.0], [0.0, -5.0e-11]],
        "covariance_scale": 1.0,
        "scale_parameter_key": "thermal.noise_scale",
        "physical_constant_keys": ["boltzmann_constant"],
    }
    payload.update(overrides)
    return NoiseCovarianceSpec.from_mapping(payload)


def test_sdae_tolerance_defaults_resolve_from_the_parameter_database():
    from relic.computation.operators.stochastic_sdae import JumpResetSpec
    from relic.parameters import get_global_manager

    for key in (_NOISE_TOLERANCE_KEY, _JUMP_TOLERANCE_KEY):
        entry = _validation_iphs_entry(key)
        for field_name in ("value", "units", "dtype", "bounds", "category", "description",
                           "references", "tags", "requires_grad", "provenance"):
            assert field_name in entry, (key, field_name)
        assert get_global_manager().get(key, required=True) == pytest.approx(float(entry["value"]))
    noise_value = float(_validation_iphs_entry(_NOISE_TOLERANCE_KEY)["value"])
    jump_value = float(_validation_iphs_entry(_JUMP_TOLERANCE_KEY)["value"])
    spec = _near_psd_noise()
    assert spec.tolerance == pytest.approx(noise_value)
    assert spec.validate()["passed"] is True
    assert _near_psd_noise(tolerance=1.0e-12).validate()["passed"] is False
    assert NoiseCovarianceSpec(name="direct", relation="fdt").tolerance == pytest.approx(noise_value)
    assert JumpResetSpec(name="jump").tolerance == pytest.approx(jump_value)
    assert JumpResetSpec.from_mapping({"name": "jump"}).tolerance == pytest.approx(jump_value)
    assert JumpResetSpec.from_mapping({"name": "jump", "tolerance": 1.0e-6}).tolerance == pytest.approx(1.0e-6)
    with pytest.raises(TypeError):
        JumpResetSpec.from_mapping({"name": "jump", "tolerance": None})
    with pytest.raises(TypeError):
        _near_psd_noise(tolerance=None)


@pytest.mark.parametrize("spec_class", [MemoryKernelSpec, HybridTopologyTransitionSpec])
def test_memory_and_topology_specs_expose_no_unused_tolerance(spec_class):
    # No audit consumes a tolerance for these declarations, so none is accepted.
    import dataclasses

    assert "tolerance" not in {item.name for item in dataclasses.fields(spec_class)}
    with pytest.raises(TypeError):
        spec_class(name="declared_tolerance", tolerance=1.0e-10)
    with pytest.raises(ValueError, match="has no tolerance field"):
        spec_class.from_mapping({"name": "declared_tolerance", "tolerance": 1.0e-10})
    assert spec_class.from_mapping({"name": "without_tolerance"}).name == "without_tolerance"


def test_changing_the_noise_covariance_parameter_changes_the_check():
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    original = manager.get(_NOISE_TOLERANCE_KEY, required=True)
    try:
        manager.set_value(_NOISE_TOLERANCE_KEY, 1.0e-11)
        audit = _near_psd_noise().validate()
        assert audit["passed"] is False
        assert "dissipative_operator must be positive semidefinite" in audit["issues"]
        assert "covariance residual exceeds tolerance" in audit["issues"]
    finally:
        manager.set_value(_NOISE_TOLERANCE_KEY, original)
    assert manager.get(_NOISE_TOLERANCE_KEY, required=True) == original
