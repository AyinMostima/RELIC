"""Tests for stochastic IPHS/SDAE structural validation."""

import copy

import pytest

from relic.validation.framework.models import ValidationContext
from relic.validation.framework.rules.stochastic_sdae import (
    validate_stochastic_sdae_payload,
    validate_stochastic_sdae_structure,
)


def _valid_topology_transition():
    return {
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


def _valid_payload():
    return {
        "runtime_owner": "relic.stochastic",
        "stochastic_terms": [
            {
                "name": "thermal_flux_noise",
                "classification": "fdt_bound",
                "runtime_owner": "relic.stochastic.sde_solver",
                "state_vars": ["heat_storage"],
                "parameter_keys": ["thermal.alpha_noise"],
                "noise_type": "white",
                "solver_name": "euler_maruyama",
            }
        ],
        "noise_covariances": [
            {
                "name": "thermal_fdt",
                "relation": "fdt",
                "sigma": [[1.4142135623730951, 0.0], [0.0, 1.4142135623730951]],
                "dissipative_operator": [[1.0, 0.0], [0.0, 1.0]],
                "covariance_scale": 2.0,
                "scale_parameter_key": "thermal.alpha_noise",
                "physical_constant_keys": ["boltzmann_constant"],
                "tolerance": 1.0e-9,
            }
        ],
        "conservative_noise_fluxes": [
            {
                "name": "edge_noise",
                "flux_space": {"complex_id": "K", "degree": 1, "semantic_name": "edge_flux"},
                "storage_space": {
                    "complex_id": "K",
                    "degree": 0,
                    "semantic_name": "node_storage",
                },
                "topology_operator_name": "D_flux",
                "routes_to_storage": True,
                "direct_storage_noise": False,
            }
        ],
        "descriptor_constraints": [
            {
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
        ],
        "jump_resets": [
            {
                "name": "closed_jump",
                "reset_map": [[-1.0], [1.0]],
                "conservation_matrix": [[1.0, 1.0]],
                "ledger_behavior": "closed",
                "propensity_keys": ["reaction.jump_rate"],
                "mark_space": "reaction_channel",
                "event_clock": "state_dependent_poisson",
            }
        ],
        "stabilized_advections": [
            {
                "name": "upwind_advection",
                "operator_name": "A_h(v)",
                "scheme": "upwind",
                "conservative": True,
                "cfl_metadata": {"parameter_key": "transport.cfl"},
                "boundary_flux_route": "boundary_outflow",
            }
        ],
        "memory_kernels": [
            {
                "name": "passive_delay_storage",
                "kernel_operator": [[0.25]],
                "hidden_storage_space": {"complex_id": "K", "semantic_name": "memory_state"},
                "effort_variable": "e_h",
                "passivity_evidence": "positive_real_realization",
                "parameter_keys": ["memory.relaxation_time"],
                "residual_route": "diagnostics.memory_residual",
            }
        ],
        "ale_geometry_corrections": [
            {
                "name": "moving_mesh_hodge_correction",
                "hodge_time_derivative": [[0.1]],
                "state_space": {"complex_id": "K", "semantic_name": "moving_storage"},
                "energy_effort": "e_E",
                "correction_route": "geometry_port",
                "port_accounting": "diagnostics.geometry_power",
                "parameter_keys": ["geometry.mesh_velocity"],
            }
        ],
        "hybrid_topology_transitions": [],
        "stochastic_reproducibility": {
            "random_seed": 123,
            "noise_type": "white",
            "solver_method": "euler_maruyama",
            "uses_ensemble": True,
            "ensemble_size": 8,
            "jump_propensities": ["reaction.jump_rate"],
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "rng_state_owner": "relic.core.simulator.checkpoint",
            "rng_substream": "thermal_flux_noise",
            "restart_policy": "restore_and_replay",
            "time_step_mode": "fixed",
        },
        "state_units": {"heat_storage": "J m-2"},
        "memory_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "history_state_keys": ["memory.hidden_state"],
            "history_cursor": "memory.history_cursor",
        },
        "ale_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "geometry_state_keys": ["mesh.coordinates"],
            "geometry_cursor": "mesh.geometry_epoch",
        },
        "descriptor_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "algebraic_state_keys": ["constraint.multiplier"],
            "projection_on_restore": True,
        },
    }


def _isolated_payload(*keys):
    source = _valid_payload()
    payload = {"runtime_owner": source["runtime_owner"]}
    for key in keys:
        payload[key] = copy.deepcopy(source[key])
    return payload


def test_stochastic_sdae_payload_accepts_complete_structure():
    result = validate_stochastic_sdae_payload(_valid_payload())

    assert result.passed is True
    assert result.metrics["noise_covariance_count"] == 1
    assert result.metrics["descriptor_constraint_count"] == 1
    assert result.metrics["memory_kernel_count"] == 1
    assert result.metrics["ale_geometry_correction_count"] == 1
    assert result.metrics["hybrid_topology_transition_count"] == 0


def test_jump_ledger_tolerance_parameter_changes_the_closed_ledger_check():
    # C_h R_h = (1, 1) . (-1, 1 + 5e-11) = 5e-11 for the declared closed jump ledger.
    from relic.parameters import get_global_manager

    key = "validation.iphs.jump_ledger_tolerance"
    payload = _valid_payload()
    payload["jump_resets"][0]["reset_map"] = [[-1.0], [1.0 + 5.0e-11]]
    assert validate_stochastic_sdae_payload(copy.deepcopy(payload)).passed is True
    manager = get_global_manager()
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 1.0e-11)
        result = validate_stochastic_sdae_payload(copy.deepcopy(payload))
    finally:
        manager.set_value(key, original)
    assert result.passed is False
    assert any(detail.expected == "C_h R_h = 0 or C_h N_h = 0" for detail in result.details)
    assert manager.get(key, required=True) == original


def test_stochastic_sdae_payload_rejects_direct_storage_noise():
    payload = _valid_payload()
    payload["conservative_noise_fluxes"][0]["direct_storage_noise"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("direct storage" in str(detail.expected) for detail in result.details)


def test_stochastic_sdae_payload_rejects_missing_fdt_evidence():
    payload = _valid_payload()
    payload["noise_covariances"][0].pop("physical_constant_keys")

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("Sigma_h Sigma_h" in str(detail.expected) for detail in result.details)


def test_stochastic_sdae_payload_rejects_unproven_memory_kernel():
    payload = _valid_payload()
    payload["memory_kernels"][0]["passivity_evidence"] = ""

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "recognized structural passivity declaration" in str(detail.expected)
        for detail in result.details
    )


def test_memory_kernel_or_topology_transition_declaring_a_tolerance_is_rejected():
    payload = _valid_payload()
    payload["memory_kernels"][0]["tolerance"] = 1.0e-10
    result = validate_stochastic_sdae_payload(payload)
    assert result.passed is False
    assert any(detail.expected == "valid MemoryKernelSpec" for detail in result.details)

    payload = _valid_payload()
    payload["hybrid_topology_transitions"] = [dict(_valid_topology_transition(), tolerance=1.0e-10)]
    result = validate_stochastic_sdae_payload(payload)
    assert result.passed is False
    assert any(detail.expected == "valid HybridTopologyTransitionSpec" for detail in result.details)


def _ale_symmetry_result(hodge_time_derivative):
    payload = _valid_payload()
    payload["ale_geometry_corrections"][0]["hodge_time_derivative"] = hodge_time_derivative
    return validate_stochastic_sdae_payload(payload)


def _ale_rejected_as_non_symmetric(result):
    return any(detail.expected == "symmetric dot(star_h)" for detail in result.details)


def test_ale_hodge_symmetry_uses_the_relative_hodge_symmetry_parameter():
    # rho_sym = ||dH - dH^T||_F / ||dH||_F, the HodgeStarSpec definition, compared with
    # max(validation.iphs.hodge_symmetry_rtol, n eps). A scaled matrix with an
    # absolute defect of 1e-5 has rho_sym = 1e-5 / 1e6 = 1e-11 and is symmetric.
    from relic.parameters import get_global_manager

    scaled = _ale_symmetry_result([[1.0e6, 1.0e-5], [0.0, 1.0e6]])
    assert scaled.passed is True, [(d.variable_name, d.expected, d.value) for d in scaled.details]
    near = [[1.0, 5.0e-11], [0.0, 1.0]]
    assert _ale_symmetry_result(near).passed is True
    key = "validation.iphs.hodge_symmetry_rtol"
    manager = get_global_manager()
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 1.0e-12)
        result = _ale_symmetry_result(near)
    finally:
        manager.set_value(key, original)
    assert result.passed is False
    assert _ale_rejected_as_non_symmetric(result)
    assert manager.get(key, required=True) == original


def test_stochastic_sdae_payload_rejects_unrouted_ale_correction():
    payload = _valid_payload()
    payload["ale_geometry_corrections"][0]["correction_route"] = ""
    payload["ale_geometry_corrections"][0]["port_accounting"] = ""

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("Hodge-energy correction route" in str(detail.expected) for detail in result.details)


def test_stochastic_sdae_payload_rejects_unmapped_hybrid_topology_change():
    payload = _valid_payload()
    payload["hybrid_topology_transitions"] = [_valid_topology_transition()]
    payload["hybrid_topology_transitions"][0]["state_remap"] = None
    payload["hybrid_topology_transitions"][0]["ledger_map"] = None
    payload["hybrid_topology_transitions"][0]["ledger_change"] = None

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "ledger-preserving remap" in str(detail.expected)
        or "state remap" in str(detail.expected)
        for detail in result.details
    )


def test_stochastic_sdae_rule_reads_context_metadata():
    results = validate_stochastic_sdae_structure(
        ValidationContext(metadata={"stochastic_sdae": _valid_payload()})
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_stochastic_sdae_empty_payload_is_not_applicable_by_default():
    result = validate_stochastic_sdae_payload({})

    assert result.passed is True
    assert result.metrics["declared_structure_count"] == 0


def test_stochastic_sdae_strict_claim_rejects_empty_payload():
    result = validate_stochastic_sdae_payload({"require_declared_structure": True})

    assert result.passed is False
    assert any(
        detail.expected
        == "at least one canonically represented primary stochastic/hybrid semantic"
        for detail in result.details
    )


def test_stochastic_sdae_fails_closed_for_four_unbound_semantics_independently():
    semantics = (
        "marked_jump",
        "memory",
        "ale",
        "descriptor_dae",
    )
    for semantic in semantics:
        result = validate_stochastic_sdae_payload(
            _valid_payload(),
            required_semantics=(semantic,),
        )
        assert result.passed is False, (semantic, result.details)
        assert any(
            semantic in detail.variable_name and "[N]" in detail.expected
            for detail in result.details
        )


def test_continuous_sde_strict_claim_fails_closed_without_graph_binding():
    result = validate_stochastic_sdae_payload(
        _valid_payload(),
        required_semantics=("continuous_sde",),
    )

    assert result.passed is False
    assert any(
        "implemented GraphIR/Functor drift-diffusion binding [N]" in detail.expected
        for detail in result.details
    )


def test_stochastic_declaration_rejects_unknown_state_unit_label():
    payload = _valid_payload()
    payload["state_units"]["heat_storage"] = "banana"

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        detail.variable_name == "state_units.heat_storage"
        and "canonical dimensionality" in detail.expected
        for detail in result.details
    )


def test_marked_jump_open_ledger_rejects_unknown_unit_label():
    payload = _valid_payload()
    payload["jump_resets"][0].update(
        {
            "ledger_behavior": "open",
            "open_system_route": "port",
            "open_ledger_change": [[0.0]],
            "open_ledger_units": "banana",
        }
    )

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "open jump ledger route" in detail.expected for detail in result.details
    )


def test_stochastic_sdae_rejects_cross_semantic_substitution():
    payload = _valid_payload()
    payload["stochastic_terms"] = []
    payload["jump_resets"] = []
    payload["descriptor_constraints"] = []
    payload["ale_geometry_corrections"] = []
    payload["hybrid_topology_transitions"] = []

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("marked_jump",),
    )

    assert result.passed is False
    assert any(
        detail.expected == "marked_jump declaration count > 0"
        for detail in result.details
    )


def test_stochastic_sdae_caller_requirement_cannot_be_weakened_by_payload():
    payload = _valid_payload()
    payload["required_semantics"] = ["memory"]
    payload["jump_resets"] = []

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("marked_jump",),
    )

    assert result.passed is False
    assert any(detail.variable_name == "marked_jump" for detail in result.details)


def test_continuous_sde_requires_restartable_rng_contract():
    payload = _valid_payload()
    payload["stochastic_reproducibility"].pop("rng_state_owner")

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("continuous_sde",),
    )

    assert result.passed is False
    assert any("rng_state_owner" in detail.expected for detail in result.details)


def test_adaptive_sde_fails_closed_without_brownian_bridge():
    payload = _valid_payload()
    payload["stochastic_reproducibility"]["time_step_mode"] = "adaptive"

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("continuous_sde",),
    )

    assert result.passed is False
    assert any("Brownian bridge" in detail.expected for detail in result.details)


def test_adaptive_sde_rejects_importable_non_bridge_owner():
    payload = _valid_payload()
    payload["stochastic_reproducibility"]["time_step_mode"] = "adaptive"
    payload["stochastic_reproducibility"][
        "brownian_bridge_owner"
    ] = "relic.core.state.StateManager"

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("continuous_sde",),
    )

    assert result.passed is False
    assert any(
        "implemented Brownian bridge owner [N]" in detail.expected
        for detail in result.details
    )


def test_marked_jump_requires_mark_and_event_clock():
    payload = _valid_payload()
    payload["jump_resets"][0].pop("mark_space")

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("marked_jump",),
    )

    assert result.passed is False
    assert any("mark_space" in detail.expected for detail in result.details)


def test_memory_requires_checkpointed_history_cursor():
    payload = _valid_payload()
    payload["memory_restart"].pop("history_cursor")

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("memory",),
    )

    assert result.passed is False
    assert any("history_cursor" in detail.expected for detail in result.details)


def test_topology_change_remains_fail_closed_without_graph_epoch_owner():
    payload = _valid_payload()
    payload["hybrid_topology_transitions"] = [_valid_topology_transition()]
    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("topology_change",),
    )

    assert result.passed is False
    assert any("graph_epoch_owner" in detail.expected for detail in result.details)


def test_declared_topology_change_infers_fail_closed_contract():
    payload = _valid_payload()
    payload["hybrid_topology_transitions"] = [_valid_topology_transition()]
    payload["topology_restart"] = {
        "checkpoint_owner": "relic.core.simulator.checkpoint",
        "graph_epoch_owner": "relic.core.state.StateManager",
        "event_queue_state_key": "events.queue",
    }
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "topology-transition binding on the canonical provider epoch owner [N]" in detail.expected
        for detail in result.details
    )


def _topology_binding_payload():
    functor_class = "relic.scientific.hydrosphere.runoff.surface_ponding.HortonianPondingEventFunctor"
    ponded = "surface_ponded_depth"
    excess = "surface_ponding_excess"
    active = "surface_ponding_active"
    return {
        "required_semantics": ["topology_change"],
        "runtime_owner": "relic.core.simulator.registration",
        "hybrid_topology_transitions": [
            {
                "name": "ponding_onset",
                "event_name": "ponding_onset",
                "source_complex": {"complex_id": "K_disconnected", "semantic_name": "rainfall_and_detector"},
                "target_complex": {"complex_id": "K_connected", "semantic_name": "rainfall_detector_and_store"},
                # P on (h, E, a): h+ = h + E, E+ = 0, a+ = 1 (affine reset).
                "state_remap": [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "ledger_map": [[1.0, 1.0, 0.0]],
                "ledger_change": [[0.0]],
                "dimension_change": True,
                "reset_route": "relic.core.simulator.registration.apply_topology_transition",
                "parameter_keys": [
                    "hydrosphere.runoff.ponding_infiltration_capacity",
                    "hydrosphere.runoff.ponding_onset_depth",
                ],
                "audit_route": "diagnostics.topology_transition.ledger_residual",
                "state_vars": [ponded, excess, active],
            }
        ],
        "topology_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "graph_epoch_owner": "relic.core.simulator.registration",
            "event_queue_state_key": "relic.core.simulator.execution_loop.retry_external_deliveries",
        },
        "state_units": {ponded: "mm", excess: "mm", active: "1"},
        "execution_bindings": {
            "topology_change": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "state_vars": [ponded, excess, active],
                "topology_state_scope": "state_manager",
                "validation_test": "test/core/simulator_topology_transition.test.py",
            }
        },
    }


def test_topology_change_binding_lifts_the_fail_closed_expectation():
    payload = _topology_binding_payload()
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is True, [(d.variable_name, d.expected, d.actual) for d in result.details]
    assert not any("[N]" in d.expected for d in result.details)


def test_topology_change_binding_requires_the_canonical_epoch_and_queue_owners():
    payload = _topology_binding_payload()
    payload["topology_restart"]["graph_epoch_owner"] = "relic.core.state.StateManager"
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is False
    assert any("canonical provider epoch owner" in d.expected for d in result.details)
    assert any("[N]" in d.expected for d in result.details)

    payload = _topology_binding_payload()
    payload["topology_restart"]["event_queue_state_key"] = "events.queue"
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is False
    assert any("delivery queue owner" in d.expected for d in result.details)


def test_topology_change_binding_fails_closed_without_the_functor_binding():
    payload = _topology_binding_payload()
    payload.pop("execution_bindings")
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is False
    assert any("[N]" in d.expected and d.variable_name == "topology_restart" for d in result.details)

    payload = _topology_binding_payload()
    payload["execution_bindings"]["topology_change"]["state_vars"] = ["surface_ponded_depth"]
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is False

    payload = _topology_binding_payload()
    payload["execution_bindings"]["topology_change"]["functor_class"] = (
        "relic.scientific.hydrosphere.runoff.surface_ponding.SurfacePondingFunctor"
    )
    result = validate_stochastic_sdae_payload(payload, required_semantics=("topology_change",))
    assert result.passed is False
    assert any("topology_category" in d.expected for d in result.details)


def test_stochastic_sdae_rule_propagates_requested_semantics():
    payload = _valid_payload()
    payload["jump_resets"] = []
    results = validate_stochastic_sdae_structure(
        ValidationContext(
            metadata={
                "strict": True,
                "required_semantics": ["marked_jump"],
                "stochastic_sdae": payload,
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is False


def test_stochastic_sdae_rule_propagates_strict_nonempty_requirement():
    results = validate_stochastic_sdae_structure(
        ValidationContext(metadata={"strict": True})
    )

    assert len(results) == 1
    assert results[0].passed is False
    assert any(
        "at least one canonically represented primary" in detail.expected
        for detail in results[0].details
    )


def test_stochastic_sdae_rejects_importable_non_owner_and_invalid_rng_domain():
    payload = _valid_payload()
    payload["runtime_owner"] = "math.pi"
    payload["stochastic_terms"][0]["runtime_owner"] = "math.pi"
    payload["stochastic_reproducibility"].update(
        {
            "random_seed": "not-an-int",
            "noise_type": "anything",
            "solver_method": "anything",
        }
    )

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    expected = {detail.expected for detail in result.details}
    assert "callable or module stochastic owner" in expected
    assert "random_seed is a non-negative integer" in expected
    assert "recognized stochastic noise_type" in expected
    assert "recognized stochastic solver_method" in expected


def test_stochastic_sdae_rejects_callable_outside_canonical_runtime_owner():
    payload = _valid_payload()
    payload["runtime_owner"] = "math.sin"
    payload["stochastic_terms"][0]["runtime_owner"] = "math.sin"

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        detail.expected == "callable or module stochastic owner"
        for detail in result.details
    )


def test_marked_jump_rejects_empty_reset_and_vacuous_closed_ledger():
    payload = _valid_payload()
    jump = payload["jump_resets"][0]
    jump["reset_map"] = []
    jump.pop("conservation_matrix")

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    expected = {detail.expected for detail in result.details}
    assert "non-empty finite rank-2 reset_map or stoichiometric_N" in expected
    assert "closed jump ledger requires non-empty conservation_matrix C_h" in expected


def test_marked_jump_rejects_zero_conservation_rows():
    payload = _valid_payload()
    payload["jump_resets"][0]["conservation_matrix"] = [[0.0, 0.0]]

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("nonzero conservation rows" in detail.expected for detail in result.details)


def test_marked_jump_open_ledger_requires_numerical_route():
    payload = _valid_payload()
    payload["jump_resets"][0]["ledger_behavior"] = "open"
    payload["jump_resets"][0].pop("conservation_matrix")

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("open jump ledger route" in detail.expected for detail in result.details)


def test_marked_jump_open_ledger_rejects_malformed_numerical_change():
    payload = _valid_payload()
    payload["jump_resets"][0].update(
        {
            "ledger_behavior": "open",
            "open_system_route": "port",
            "open_ledger_change": "not-a-number",
            "open_ledger_units": "mol",
        }
    )

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("open jump ledger route" in detail.expected for detail in result.details)


def test_memory_kernel_rejects_nonsquare_matrix_and_prose_self_proof():
    payload = _valid_payload()
    payload["memory_kernels"][0]["kernel_operator"] = [[1.0, -2.0]]
    payload["memory_kernels"][0]["passivity_evidence"] = "not passive at all"

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    expected = {detail.expected for detail in result.details}
    assert "non-empty finite square K_h memory kernel" in expected
    assert "recognized structural passivity declaration" in expected


def test_memory_kernel_rejects_scalar_hidden_storage_space():
    payload = _valid_payload()
    payload["memory_kernels"][0]["hidden_storage_space"] = "memory_state"

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("memory",),
    )

    assert result.passed is False
    assert any(
        detail.expected == "declared hidden memory cochain space"
        for detail in result.details
    )


def test_ale_geometry_rejects_nonfinite_hodge_derivative():
    payload = _valid_payload()
    payload["ale_geometry_corrections"][0]["hodge_time_derivative"] = [
        [float("nan")]
    ]

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("finite dot(star_h)" in detail.expected for detail in result.details)


def test_descriptor_dae_rejects_empty_operator_and_unknown_units():
    payload = _valid_payload()
    payload["descriptor_constraints"][0]["A_h"] = []
    payload["descriptor_constraints"][0]["units"] = "banana"

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    expected = {detail.expected for detail in result.details}
    assert "non-empty finite rank-2 A_h constraint operator" in expected
    assert "parseable descriptor constraint units" in expected


def test_strict_continuous_term_fails_without_canonical_semantic_representation():
    payload = _isolated_payload(
        "stochastic_terms",
        "stochastic_reproducibility",
        "state_units",
    )
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "continuous SDE semantic is not canonically representable" in detail.expected
        for detail in result.details
    )
    assert result.metrics["primary_structure_count"] == 0


def test_strict_stochastic_rejects_validator_only_raw_semantic_field():
    payload = _isolated_payload(
        "stochastic_terms",
        "stochastic_reproducibility",
        "state_units",
    )
    payload["stochastic_terms"][0]["semantic"] = "continuous_sde"
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "canonical StochasticTermSpec fields without validator-only semantic" in detail.expected
        for detail in result.details
    )


@pytest.mark.parametrize(
    ("semantic", "keys"),
    [
        (
            "marked_jump",
            ("jump_resets", "stochastic_reproducibility"),
        ),
        ("memory", ("memory_kernels", "memory_restart")),
        ("ale", ("ale_geometry_corrections", "ale_restart")),
        (
            "descriptor_dae",
            ("descriptor_constraints", "descriptor_restart"),
        ),
    ],
)
def test_strict_primary_declaration_infers_independent_unbound_semantic(
    semantic,
    keys,
):
    payload = _isolated_payload(*keys)
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        detail.variable_name == semantic and "[N]" in detail.expected
        for detail in result.details
    )
    assert result.metrics["inferred_semantic_count"] == 1


@pytest.mark.parametrize(
    "key",
    [
        "noise_covariances",
        "conservative_noise_fluxes",
        "stabilized_advections",
    ],
)
def test_strict_support_only_declaration_cannot_satisfy_primary_gate(key):
    payload = _isolated_payload(key)
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert result.metrics["primary_structure_count"] == 0
    assert result.metrics["auxiliary_structure_count"] == 1
    assert result.metrics["declared_structure_count"] == 1
    assert any("primary stochastic/hybrid semantic" in detail.expected for detail in result.details)


def test_strict_jump_rejects_continuous_rng_solver_pairing():
    payload = _isolated_payload("jump_resets", "stochastic_reproducibility")
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any(
        "marked jump noise_type" in detail.expected
        or "marked jump solver_method" in detail.expected
        for detail in result.details
    )


def test_jump_classification_poisson_gillespie_does_not_become_continuous_sde():
    payload = _isolated_payload(
        "stochastic_terms",
        "stochastic_reproducibility",
        "state_units",
    )
    payload["stochastic_terms"][0].update(
        {
            "classification": "jump_reset",
            "noise_type": "poisson",
            "solver_name": "gillespie",
        }
    )
    payload["stochastic_reproducibility"].update(
        {
            "noise_type": "poisson",
            "solver_method": "gillespie",
        }
    )
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert result.metrics["primary_structure_count"] == 0
    assert any("continuous SDE category" in detail.expected for detail in result.details)


@pytest.mark.parametrize("strict_flag", ["false", 0])
def test_require_declared_structure_requires_exact_bool(strict_flag):
    payload = _isolated_payload("noise_covariances")
    payload["require_declared_structure"] = strict_flag

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert any("require_declared_structure is bool" in detail.expected for detail in result.details)


def test_strict_complete_descriptive_payload_infers_each_primary_semantic():
    payload = _valid_payload()
    payload["require_declared_structure"] = True

    result = validate_stochastic_sdae_payload(payload)

    assert result.passed is False
    assert result.metrics["primary_structure_count"] == 4
    assert result.metrics["auxiliary_structure_count"] == 3
    assert result.metrics["inferred_semantic_count"] == 4
    inferred = {
        detail.variable_name
        for detail in result.details
        if "[N]" in detail.expected
    }
    assert {"marked_jump", "memory", "ale", "descriptor_dae"}.issubset(inferred)


def test_wrapper_string_false_does_not_enable_strict_mode():
    results = validate_stochastic_sdae_structure(
        ValidationContext(
            metadata={
                "strict": "false",
                "stochastic_sdae": {},
            }
        )
    )

    assert len(results) == 1
    assert results[0].passed is True


def _continuous_sde_binding_payload():
    """Payload declaring the Ornstein-Uhlenbeck anomaly Functor as the executable binding."""
    functor_class = (
        "relic.scientific.atmosphere.stochastic_temperature."
        "OrnsteinUhlenbeckTemperatureFunctor"
    )
    return {
        "runtime_owner": "relic.stochastic.sde_solver",
        "stochastic_terms": [
            {
                "name": "air_temperature_anomaly_noise",
                "classification": "external_forcing",
                "runtime_owner": "relic.stochastic.sde_solver.EulerMaruyamaSolver",
                "state_vars": ["stochastic_air_temperature_anomaly"],
                "parameter_keys": [
                    "stochastic.weather.temperature.reversion_rate",
                    "stochastic.weather.temperature.volatility",
                ],
                "noise_type": "white",
                "solver_name": "euler_maruyama",
            }
        ],
        "stochastic_reproducibility": {
            "random_seed": 7,
            "noise_type": "white",
            "solver_method": "euler_maruyama",
            "uses_ensemble": False,
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "rng_state_owner": "relic.core.simulator.checkpoint",
            "rng_substream": "stochastic_air_temperature",
            "restart_policy": "restore_and_replay",
            "time_step_mode": "fixed",
        },
        "state_units": {"stochastic_air_temperature_anomaly": "K"},
        "execution_bindings": {
            "continuous_sde": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "solver_owner": "relic.stochastic.sde_solver.EulerMaruyamaSolver",
                "state_vars": ["stochastic_air_temperature_anomaly"],
                "rng_state_scope": "functor_extra_state",
                "validation_test": "test/unit/atmosphere/test_stochastic_temperature.py",
            }
        },
    }


def _marked_jump_binding_payload():
    functor_class = "relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor"
    count = "stochastic_rainfall_event_count"
    depth = "stochastic_rainfall_depth"
    return {
        "required_semantics": ["marked_jump"],
        "runtime_owner": "relic.stochastic.gillespie",
        "jump_resets": [
            {
                "name": "rainfall_arrival",
                "reset_map": [[1.0]],
                "conservation_matrix": [[1.0]],
                "ledger_behavior": "open",
                "open_system_route": "port",
                "open_ledger_change": [[1.0]],
                "open_ledger_units": "1",
                "propensity_keys": ["stochastic.weather.rainfall.event_rate"],
                "state_vars": [count, depth],
                "mark_space": "exponential depth mark z >= 0 [mm] with mean stochastic.weather.rainfall.mean_depth",
                "event_clock": "direct_ssa_fixed_step",
            }
        ],
        "stochastic_reproducibility": {
            "random_seed": 7,
            "noise_type": "marked_poisson",
            "solver_method": "direct_ssa",
            "uses_ensemble": False,
            "jump_propensities": ["stochastic.weather.rainfall.event_rate"],
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "rng_state_owner": "relic.core.simulator.checkpoint",
            "rng_substream": "stochastic_rainfall",
            "restart_policy": "restore_and_replay",
            "time_step_mode": "fixed",
        },
        "state_units": {count: "1", depth: "mm"},
        "execution_bindings": {
            "marked_jump": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "solver_owner": "relic.stochastic.gillespie.GillespieSSA",
                "state_vars": [count, depth],
                "rng_state_scope": "functor_extra_state",
                "validation_test": "test/unit/atmosphere/test_stochastic_rainfall.py",
            }
        },
    }


def _memory_binding_payload():
    functor_class = "relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor"
    mobile = "mobile_solute_concentration"
    immobile = "immobile_solute_concentration"
    return {
        "required_semantics": ["memory"],
        "runtime_owner": "relic.scientific.pedosphere.transport.two_region_mass_transfer",
        "memory_kernels": [
            {
                "name": "two_region_exponential_memory",
                "kernel_operator": [[4.0e-9]],
                "hidden_storage_space": {"complex_id": "K", "degree": 0, "semantic_name": immobile},
                "effort_variable": mobile,
                "passivity_evidence": "storage_function_certificate",
                "parameter_keys": [
                    "pedosphere.transport.two_region_mass_transfer_coefficient",
                    "pedosphere.transport.two_region_immobile_water_content",
                ],
                "residual_route": "diagnostics.two_region_exchange.memory_residual",
                "state_vars": [mobile, immobile],
            }
        ],
        "memory_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "history_state_keys": [immobile],
            "history_cursor": "state_manager_current_step",
        },
        "state_units": {mobile: "kg m-3", immobile: "kg m-3"},
        "execution_bindings": {
            "memory": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "state_vars": [mobile, immobile],
                "history_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_two_region_mass_transfer.py",
            }
        },
    }


def _descriptor_dae_binding_payload():
    functor_class = "relic.scientific.pedosphere.transport.equilibrium_sorption.EquilibriumSorptionFunctor"
    total = "equilibrium_total_solute_mass"
    dissolved = "equilibrium_dissolved_concentration"
    sorbed = "equilibrium_sorbed_concentration"
    return {
        "required_semantics": ["descriptor_dae"],
        "runtime_owner": "relic.scientific.pedosphere.transport.equilibrium_sorption",
        "descriptor_constraints": [
            {
                "name": "mass_partition",
                "A_h": [[-1.0, 0.3, 1300.0]],
                "constrained_space": {"complex_id": "K", "degree": 0, "semantic_name": total},
                "units": "kg m-3",
                "projection_method": "algebraic_elimination",
                "residual_route": "diagnostics.equilibrium_sorption.constraint_residual",
                "enforces_state_constraint": True,
                "enforces_tangent_constraint": True,
                "constraint_force_neutral": True,
                "state_vars": [total, dissolved, sorbed],
            }
        ],
        "descriptor_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "algebraic_state_keys": [dissolved, sorbed],
            "projection_on_restore": True,
        },
        "state_units": {total: "kg m-3", dissolved: "kg m-3", sorbed: "kg kg-1"},
        "execution_bindings": {
            "descriptor_dae": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "state_vars": [total, dissolved, sorbed],
                "algebraic_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_equilibrium_sorption.py",
            }
        },
    }


def test_descriptor_dae_execution_binding_lifts_the_graph_binding_gate():
    result = validate_stochastic_sdae_payload(
        _descriptor_dae_binding_payload(),
        required_semantics=("descriptor_dae",),
    )

    assert result.passed is True, [(d.variable_name, d.expected, d.value) for d in result.details]
    assert not any(
        "DAE projection binding [N]" in detail.expected for detail in result.details
    )


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.pop("descriptor_dae"), "DAE projection binding [N]"),
    (lambda b: b["descriptor_dae"].update({"functor_class": "relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor"}), "constraint_category"),
    (lambda b: b["descriptor_dae"].update({"forward_owner": "relic.core.state.StateManager.get_state"}), "forward_owner"),
    (lambda b: b["descriptor_dae"].update({"state_vars": ["equilibrium_total_solute_mass"]}), "state_vars"),
    (lambda b: b["descriptor_dae"].update({"state_vars": ["heat_storage"]}), "state_vars"),
    (lambda b: b["descriptor_dae"].update({"algebraic_state_scope": "python_attribute"}), "algebraic_state_scope"),
    (lambda b: b["descriptor_dae"].update({"validation_test": "test/missing_file.py"}), "validation_test"),
])
def test_descriptor_dae_execution_binding_rejects_inconsistent_witness(mutate, fragment):
    payload = _descriptor_dae_binding_payload()
    mutate(payload["execution_bindings"])

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("descriptor_dae",),
    )

    assert result.passed is False
    assert any(fragment in detail.expected or fragment in detail.variable_name for detail in result.details), [
        (d.variable_name, d.expected, d.value) for d in result.details
    ]


def test_descriptor_dae_restart_without_projection_on_restore_is_rejected():
    payload = _descriptor_dae_binding_payload()
    payload["descriptor_restart"]["projection_on_restore"] = False

    result = validate_stochastic_sdae_payload(payload, required_semantics=("descriptor_dae",))

    assert result.passed is False
    assert any("projection_on_restore=true" in detail.expected for detail in result.details)


def _ale_binding_payload():
    functor_class = "relic.scientific.pedosphere.transport.moving_domain_diffusion.MovingDomainDiffusionFunctor"
    conc = "moving_domain_solute_concentration"
    length = "moving_domain_length"
    return {
        "required_semantics": ["ale"],
        "runtime_owner": "relic.scientific.pedosphere.transport.moving_domain_diffusion",
        "ale_geometry_corrections": [
            {
                "name": "uniform_stretching_metric",
                "hodge_time_derivative": [[0.25, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [0.0, 0.0, 0.25, 0.0], [0.0, 0.0, 0.0, 0.25]],
                "state_space": {"complex_id": "K", "degree": 0, "semantic_name": conc},
                "energy_effort": conc,
                "correction_route": "diagnostics.moving_domain.geometry_power",
                "port_accounting": "diagnostics.moving_domain.boundary_mass_flux",
                "parameter_keys": [
                    "pedosphere.transport.moving_domain_diffusivity",
                    "pedosphere.transport.moving_domain_mesh_courant_max",
                ],
                "state_vars": [conc, length],
            }
        ],
        "ale_restart": {
            "checkpoint_owner": "relic.core.simulator.checkpoint",
            "geometry_state_keys": [length],
            "geometry_cursor": "state_manager_current_step",
        },
        "state_units": {conc: "kg m-3", length: "m"},
        "execution_bindings": {
            "ale": {
                "functor_class": functor_class,
                "forward_owner": functor_class + ".forward",
                "state_vars": [conc, length],
                "geometry_state_scope": "state_manager",
                "validation_test": "test/unit/pedosphere/test_moving_domain_diffusion.py",
            }
        },
    }


def test_ale_execution_binding_lifts_the_graph_binding_gate():
    result = validate_stochastic_sdae_payload(
        _ale_binding_payload(),
        required_semantics=("ale",),
    )

    assert result.passed is True, [(d.variable_name, d.expected, d.value) for d in result.details]
    assert not any(
        "ALE geometry-power binding [N]" in detail.expected for detail in result.details
    )


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.pop("ale"), "ALE geometry-power binding [N]"),
    (lambda b: b["ale"].update({"functor_class": "relic.scientific.pedosphere.transport.two_region_mass_transfer.TwoRegionMassTransferFunctor"}), "geometry_category"),
    (lambda b: b["ale"].update({"forward_owner": "relic.core.state.StateManager.get_state"}), "forward_owner"),
    (lambda b: b["ale"].update({"state_vars": ["moving_domain_length"]}), "state_vars"),
    (lambda b: b["ale"].update({"state_vars": ["heat_storage"]}), "state_vars"),
    (lambda b: b["ale"].update({"geometry_state_scope": "python_attribute"}), "geometry_state_scope"),
    (lambda b: b["ale"].update({"validation_test": "test/missing_file.py"}), "validation_test"),
])
def test_ale_execution_binding_rejects_inconsistent_witness(mutate, fragment):
    payload = _ale_binding_payload()
    mutate(payload["execution_bindings"])

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("ale",),
    )

    assert result.passed is False
    assert any(fragment in detail.expected or fragment in detail.variable_name for detail in result.details), [
        (d.variable_name, d.expected, d.value) for d in result.details
    ]


def test_memory_execution_binding_lifts_the_graph_binding_gate():
    result = validate_stochastic_sdae_payload(
        _memory_binding_payload(),
        required_semantics=("memory",),
    )

    assert result.passed is True, [(d.variable_name, d.expected, d.value) for d in result.details]
    assert not any(
        "passive-memory execution binding [N]" in detail.expected for detail in result.details
    )


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.pop("memory"), "passive-memory execution binding [N]"),
    (lambda b: b["memory"].update({"functor_class": "relic.scientific.atmosphere.stochastic_temperature.OrnsteinUhlenbeckTemperatureFunctor"}), "memory_category"),
    (lambda b: b["memory"].update({"forward_owner": "relic.core.state.StateManager.get_state"}), "forward_owner"),
    (lambda b: b["memory"].update({"state_vars": ["mobile_solute_concentration"]}), "state_vars"),
    (lambda b: b["memory"].update({"state_vars": ["heat_storage"]}), "state_vars"),
    (lambda b: b["memory"].update({"history_state_scope": "python_attribute"}), "history_state_scope"),
    (lambda b: b["memory"].update({"validation_test": "test/missing_file.py"}), "validation_test"),
])
def test_memory_execution_binding_rejects_inconsistent_witness(mutate, fragment):
    payload = _memory_binding_payload()
    mutate(payload["execution_bindings"])

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("memory",),
    )

    assert result.passed is False
    assert any(fragment in detail.expected or fragment in detail.variable_name for detail in result.details), [
        (d.variable_name, d.expected, d.value) for d in result.details
    ]


def test_marked_jump_execution_binding_lifts_the_graph_binding_gate():
    result = validate_stochastic_sdae_payload(
        _marked_jump_binding_payload(),
        required_semantics=("marked_jump",),
    )

    assert result.passed is True, [(d.variable_name, d.expected, d.value) for d in result.details]
    assert not any(
        "marked-jump execution binding [N]" in detail.expected for detail in result.details
    )


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.pop("marked_jump"), "marked-jump execution binding [N]"),
    (lambda b: b["marked_jump"].update({"functor_class": "relic.scientific.atmosphere.stochastic_temperature.OrnsteinUhlenbeckTemperatureFunctor"}), "stochastic_category"),
    (lambda b: b["marked_jump"].update({"forward_owner": "relic.core.state.StateManager.get_state"}), "forward_owner"),
    (lambda b: b["marked_jump"].update({"solver_owner": "relic.core.state.StateManager"}), "solver_owner"),
    (lambda b: b["marked_jump"].update({"state_vars": ["stochastic_rainfall_event_count"]}), "state_vars"),
    (lambda b: b["marked_jump"].update({"state_vars": ["heat_storage"]}), "state_vars"),
    (lambda b: b["marked_jump"].update({"validation_test": "test/missing_file.py"}), "validation_test"),
    (lambda b: b["marked_jump"].update({"rng_state_scope": "global_torch_rng"}), "rng_state_scope"),
])
def test_marked_jump_execution_binding_rejects_inconsistent_witness(mutate, fragment):
    payload = _marked_jump_binding_payload()
    mutate(payload["execution_bindings"])

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("marked_jump",),
    )

    assert result.passed is False
    assert any(fragment in detail.expected or fragment in detail.variable_name for detail in result.details), [
        (d.variable_name, d.expected, d.value) for d in result.details
    ]


def test_continuous_sde_execution_binding_lifts_the_graph_binding_gate():
    result = validate_stochastic_sdae_payload(
        _continuous_sde_binding_payload(),
        required_semantics=("continuous_sde",),
    )

    assert result.passed is True, [(d.variable_name, d.expected, d.value) for d in result.details]
    assert not any(
        "drift-diffusion binding [N]" in detail.expected for detail in result.details
    )


@pytest.mark.parametrize("mutate,fragment", [
    (lambda b: b.pop("continuous_sde"), "drift-diffusion binding [N]"),
    (lambda b: b["continuous_sde"].update({"functor_class": "relic.core.state.StateManager"}), "stochastic_category"),
    (lambda b: b["continuous_sde"].update({"forward_owner": "relic.core.state.StateManager.get_state"}), "forward_owner"),
    (lambda b: b["continuous_sde"].update({"solver_owner": "relic.core.state.StateManager"}), "solver_owner"),
    (lambda b: b["continuous_sde"].update({"state_vars": ["heat_storage"]}), "state_vars"),
    (lambda b: b["continuous_sde"].update({"validation_test": "test/missing_file.py"}), "validation_test"),
    (lambda b: b["continuous_sde"].update({"rng_state_scope": "global_torch_rng"}), "rng_state_scope"),
])
def test_continuous_sde_execution_binding_rejects_inconsistent_witness(mutate, fragment):
    payload = _continuous_sde_binding_payload()
    mutate(payload["execution_bindings"])

    result = validate_stochastic_sdae_payload(
        payload,
        required_semantics=("continuous_sde",),
    )

    assert result.passed is False
    assert any(fragment in detail.expected or fragment in detail.variable_name for detail in result.details), [
        (d.variable_name, d.expected, d.value) for d in result.details
    ]
