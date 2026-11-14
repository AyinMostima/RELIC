"""L1 tests for the IPHS/SDAE declaration specs (relic/computation/operators/stochastic_sdae.py).

References:
    Grmela and Ottinger (1997), GENERIC formalism (fluctuation-dissipation
    relation Sigma Sigma^T = scale * M_noise); Hairer and Wanner (1996), ch. VII
    (descriptor constraints A_h x = 0 and A_h dx = 0).

A declaration that the stochastic_sdae_check audit evaluates must be stated by
its producer. A missing relation, routing flag or constraint flag is an error;
it is never filled with the value that makes the audit pass, and a string such
as "false" is not read as the boolean True.
"""

from __future__ import annotations

import pytest

from relic.computation.operators.stochastic_sdae import (
    ConservativeNoiseFluxSpec,
    DescriptorConstraintSpec,
    HybridTopologyTransitionSpec,
    NoiseCovarianceSpec,
    StabilizedAdvectionSpec,
    StochasticTermSpec,
)


def _flux_payload(**overrides):
    payload = {
        "name": "edge_noise",
        "flux_space": {"complex_id": "K", "degree": 1, "semantic_name": "edge_flux"},
        "storage_space": {"complex_id": "K", "degree": 0, "semantic_name": "node_storage"},
        "topology_operator_name": "D_flux",
        "routes_to_storage": True,
        "direct_storage_noise": False,
    }
    payload.update(overrides)
    return payload


def _constraint_payload(**overrides):
    payload = {
        "name": "budget_lock",
        "A_h": [[1.0, -1.0]],
        "constrained_space": {"complex_id": "K", "semantic_name": "storage"},
        "units": "kg",
        "projection_method": "orthogonal_projection",
        "residual_route": "diagnostics.constraint_residual",
        "enforces_state_constraint": True,
        "enforces_tangent_constraint": True,
        "constraint_force_neutral": True,
    }
    payload.update(overrides)
    return payload


def test_noise_covariance_relation_must_be_declared() -> None:
    """A missing relation would read as 'not_applicable', which skips every check."""
    with pytest.raises(ValueError, match="relation"):
        NoiseCovarianceSpec.from_mapping({"name": "thermal_fdt", "sigma": [[1.0]]})


def test_declared_not_applicable_relation_is_accepted() -> None:
    spec = NoiseCovarianceSpec.from_mapping({"name": "external", "relation": "not_applicable"})
    assert spec.validate()["passed"] is True


@pytest.mark.parametrize("field_name", ["routes_to_storage", "direct_storage_noise"])
def test_noise_flux_routing_flags_must_be_declared(field_name) -> None:
    payload = _flux_payload()
    payload.pop(field_name)
    with pytest.raises(ValueError, match=field_name):
        ConservativeNoiseFluxSpec.from_mapping(payload)


@pytest.mark.parametrize(
    "field_name",
    ["enforces_state_constraint", "enforces_tangent_constraint", "constraint_force_neutral"],
)
def test_descriptor_constraint_flags_must_be_declared(field_name) -> None:
    payload = _constraint_payload()
    payload.pop(field_name)
    with pytest.raises(ValueError, match=field_name):
        DescriptorConstraintSpec.from_mapping(payload)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ConservativeNoiseFluxSpec.from_mapping(_flux_payload(direct_storage_noise="false")),
        lambda: DescriptorConstraintSpec.from_mapping(
            _constraint_payload(constraint_force_neutral="true")
        ),
        lambda: StabilizedAdvectionSpec.from_mapping(
            {"name": "upwind", "operator_name": "A_h", "scheme": "upwind", "conservative": "false"}
        ),
    ],
    ids=["noise_flux", "descriptor", "stabilized_advection"],
)
def test_string_flags_are_not_read_as_booleans(build) -> None:
    with pytest.raises(TypeError, match="boolean"):
        build()


def test_optional_flag_absent_keeps_its_fail_safe_default() -> None:
    spec = StabilizedAdvectionSpec.from_mapping(
        {"name": "upwind", "operator_name": "A_h", "scheme": "upwind"}
    )
    assert spec.conservative is False


def test_stochastic_term_runtime_owner_must_be_declared() -> None:
    """A missing owner would read as the package 'relic.stochastic', which exists."""
    with pytest.raises(ValueError, match="runtime_owner"):
        StochasticTermSpec.from_mapping(
            {"name": "thermal_flux_noise", "classification": "fdt_bound"}
        )


def test_hybrid_transition_dimension_change_is_strictly_boolean() -> None:
    with pytest.raises(TypeError, match="boolean"):
        HybridTopologyTransitionSpec.from_mapping(
            {"name": "ponding_switch", "dimension_change": "no"}
        )
