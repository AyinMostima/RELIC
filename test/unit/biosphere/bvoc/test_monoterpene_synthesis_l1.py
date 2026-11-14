"""
L1 Unit Tests for Monoterpene Synthesis Module

Comprehensive testing of the mechanistic monoterpene synthesis and storage model
including pool dynamics, temperature-dependent emission, and photosynthetic coupling.

Test Coverage:
- Basic instantiation and parameter initialization
- Pool model dynamics and storage mechanisms
- Temperature-dependent emission kinetics
- Synthesis-emission decoupling
- Storage capacity and saturation effects
- Degradation and turnover processes
- Photosynthetic coupling validation
- Parameter sensitivity analysis
- Mass conservation and units
- Boundary conditions and robustness

Scientific Validation:
- Storage pool dynamics: d[Pool]/dt = J_synthesis - J_emission - J_degradation  
- Temperature-dependent emission: J_emission = k_emit(T) * [Pool]
- Emission rate constant: k_emit(T) = k_25 * exp(beta * (T - T_25))
- Monoterpene synthase: J_MTS = V_max_MTS * [GPP] / (K_GPP + [GPP])
- Pool capacity: [Pool_max] = alpha_storage * V_storage

References:
- Tingey et al. (1980): Temperature dependence of monoterpene emissions
- Keeling & Bohlmann (2006): Terpene synthase mechanisms  
- Dudareva et al. (2006): Plant volatile biosynthesis
"""

import torch
import pytest
import warnings
import math
import inspect
from pathlib import Path

import yaml

from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from relic.core.simulator import RELICSimulator
from relic.data.standards.unit_converter import unit_converter
from test.utils.forward_call import call_forward

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.scientific.biosphere.bvoc.synthesis.monoterpene import (
    MonoterpenePoolModel,
    MonoterpeneSynthesis
)
from relic.scientific.biosphere.bvoc.synthesis.sesquiterpene import (
    SesquiterpeneSynthesis
)
from relic.core.device_policy import is_cuda_available
from relic.parameters import get_global_manager
from relic.parameters.ops.nn_params import relocate_binding


def test_pool_model_requires_eight_explicit_resolved_parameters():
    """Keep the nn.Module component independent from parameter lookup."""
    signature = inspect.signature(MonoterpenePoolModel)
    required_names = (
        "v_max_mts",
        "k_gpp",
        "pool_capacity",
        "pool_initial",
        "k_emit_25",
        "beta_emission",
        "k_degradation",
        "q10_synthase",
    )

    assert "parameter_getter" not in signature.parameters
    for name in required_names:
        assert signature.parameters[name].default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        MonoterpenePoolModel()


def test_pool_model_has_no_parameter_system_or_refresh_path():
    """Require the parent Functor to own every parameter resolution call."""
    pool_source = inspect.getsource(MonoterpenePoolModel)
    parent_init_source = inspect.getsource(MonoterpeneSynthesis.__init__)

    for forbidden in (
        "parameter_getter",
        "_parameter_getter",
        "get_global_manager",
        "_fetch_parameter",
        "_register_managed_parameter",
        "refresh_from_manager",
    ):
        assert forbidden not in pool_source

    assert "parameter_getter" not in parent_init_source
    for argument in (
        "v_max_mts=",
        "k_gpp=",
        "pool_capacity=",
        "pool_initial=",
        "k_emit_25=",
        "beta_emission=",
        "k_degradation=",
        "q10_synthase=",
    ):
        assert argument in parent_init_source


def test_parent_functor_injects_exact_parameter_manager_values():
    """Bind every pool coefficient to its canonical manager value."""
    model = MonoterpeneSynthesis()
    manager = get_global_manager()
    bindings = {
        "v_max_mts": (
            "biosphere.bvoc.monoterpene.v_max_mts",
            "umol m-2 s-1",
        ),
        "k_gpp": (
            "biosphere.bvoc.monoterpene.k_gpp",
            "umol m-2 s-1",
        ),
        "pool_capacity": (
            "biosphere.bvoc.monoterpene.pool.capacity",
            "umol m-2",
        ),
        "pool_initial": (
            "biosphere.bvoc.monoterpene.pool_initial",
            "umol m-2",
        ),
        "k_emit_25": (
            "biosphere.bvoc.monoterpene.pool.k_emit_25",
            "s-1",
        ),
        "beta_emission": (
            "biosphere.bvoc.monoterpene.pool.beta_emission",
            "K-1",
        ),
        "k_degradation": (
            "biosphere.bvoc.monoterpene.pool.k_degradation",
            "s-1",
        ),
        "q10_synthase": (
            "biosphere.bvoc.monoterpene.q10_synthase",
            "dimensionless",
        ),
    }

    for attribute, (key, units) in bindings.items():
        actual = getattr(model.pool_model, attribute).detach()
        expected = manager.get(
            key,
            units=units,
            required=True,
        )
        expected_tensor = torch.as_tensor(
            expected,
            dtype=actual.dtype,
            device=actual.device,
        )
        assert torch.allclose(actual, expected_tensor)


_F_CARBON_KEY = "biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes"
_SEASONAL_AMPLITUDE_KEY = "biosphere.bvoc.monoterpene.seasonal.amplitude"
_SEASONAL_PEAK_DAY_KEY = "biosphere.bvoc.monoterpene.seasonal.peak_day"
_V_MAX_MTS_KEY = "biosphere.bvoc.monoterpene.v_max_mts"
_Q10_SYNTHASE_KEY = "biosphere.bvoc.monoterpene.q10_synthase"
_K_EMIT_25_KEY = "biosphere.bvoc.monoterpene.pool.k_emit_25"
_BETA_EMISSION_KEY = "biosphere.bvoc.monoterpene.pool.beta_emission"
_K_DEGRADATION_KEY = "biosphere.bvoc.monoterpene.pool.k_degradation"
_ENABLE_SEASONAL_KEY = "biosphere.bvoc.monoterpene.enable_seasonal_variation"
FUNCTOR_BINDING_ATTRIBUTES = {
    "f_carbon_monoterpenes": _F_CARBON_KEY,
    "seasonal_amplitude": _SEASONAL_AMPLITUDE_KEY,
    "seasonal_peak_day": _SEASONAL_PEAK_DAY_KEY,
}
POOL_BINDING_ATTRIBUTES = {
    "v_max_mts": _V_MAX_MTS_KEY,
    "q10_synthase": _Q10_SYNTHASE_KEY,
    "k_emit_25": _K_EMIT_25_KEY,
    "beta_emission": _BETA_EMISSION_KEY,
    "k_degradation": _K_DEGRADATION_KEY,
}
HELD_MONOTERPENE_BINDING_KEYS = sorted(
    [*FUNCTOR_BINDING_ATTRIBUTES.values(), *POOL_BINDING_ATTRIBUTES.values()]
)
MONOTERPENE_STATE_DICT_KEYS = {
    "monoterpene_mw",
    "carbon_fraction_monoterpenes",
    "seasonal_min_factor",
    "seasonal_period_seconds",
    "seconds_per_day",
    "pool_model.k_gpp",
    "pool_model.pool_capacity",
    "pool_model.pool_initial",
    "pool_model.temp_ref",
}
_EMISSION_MOLAR = "diagnostics.bvoc.monoterpene.emission_rate_molar"
_DEGRADATION = "diagnostics.bvoc.monoterpene.degradation_rate"
_ALLOCATION = "bvoc_carbon_allocation_flux"


def _monoterpene_tensors(result) -> dict:
    """Every tensor of a monoterpene update, keyed by output name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


def _seasonal_warm_state(model) -> dict:
    """A warm, assimilating canopy with a partly filled storage pool."""
    capacity = model.pool_model.pool_capacity.detach().cpu()
    return {
        "air_temperature": torch.tensor(303.15, dtype=torch.float32),
        "gross_photosynthesis": torch.tensor(20.0, dtype=torch.float32),
        MonoterpeneSynthesis.POOL_STATE_KEY: (0.3 * capacity).to(torch.float32),
    }


# Day 100 of the seasonal cycle: the sine term is neither zero nor at its clamp.
_SEASONAL_TIME = 100.0 * 24.0 * 3600.0


def _enable_seasonal_variation(manager) -> None:
    spec = manager.get_parameter(_ENABLE_SEASONAL_KEY)
    manager.set_value(
        _ENABLE_SEASONAL_KEY, True, units=spec.units, validate=True, notify=False
    )


def _stored_graph_tensors(module) -> list:
    return sorted(
        name
        for name, attribute in vars(module).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )


def test_every_learnable_monoterpene_coefficient_is_held_by_reference():
    """Every declared coefficient that the parameter database marks learnable is
    the manager's own binding, held by reference (architecture SSOT section
    4.2). The Functor and its pool helper keep no module parameter of their
    own, so the manager calibration path (get_trainable_parameters) and the
    durable checkpoint see all of them."""
    model = MonoterpeneSynthesis()
    manager = model.get_parameter_manager()
    learnable = sorted(
        key
        for key in model.metadata.parameters
        if getattr(manager.get_parameter(key), "requires_grad", False)
    )
    assert learnable == HELD_MONOTERPENE_BINDING_KEYS

    trainable = manager.get_trainable_parameters()
    held = model.held_manager_parameters()
    assert sorted(held) == learnable
    assert all(held[key] is trainable[key] for key in learnable)
    assert list(model.parameters()) == []
    assert set(model.state_dict()) == MONOTERPENE_STATE_DICT_KEYS
    assert _stored_graph_tensors(model) == []
    assert _stored_graph_tensors(model.pool_model) == []

    for attribute, key in FUNCTOR_BINDING_ATTRIBUTES.items():
        assert getattr(model, attribute) is trainable[key], attribute
    for attribute, key in POOL_BINDING_ATTRIBUTES.items():
        value = getattr(model.pool_model, attribute)
        assert value.dtype == torch.float32, attribute
        assert value.device == model.pool_model.device, attribute
        assert torch.equal(
            value.detach(), trainable[key].detach().to(value.device)
        ), attribute


def test_monoterpene_gradients_reach_the_manager_bindings(
    restore_global_parameter_manager,
):
    """Backward through one governed call leaves the gradient on the
    ParameterManager bindings, the tensors the calibration path optimizes; a
    second backward through the same instance gives the same gradients."""
    # References: Tingey et al. (1980), Plant Physiology 65, 797-801
    #   (temperature-dependent release); Niinemets and Reichstein (2002),
    #   Global Biogeochem. Cycles 16(4), 1110 (storage-pool emission kinetics).
    # Mathematical expression: J_emit = k_25 exp(beta (T - T_25)) P, so
    #   dJ_emit/dk_25 = J_emit / k_25; J_deg = k_deg P, so
    #   dJ_deg/dk_deg = J_deg / k_deg;
    #   J_syn = V_max q^((T - T_25)/10) F / (K + F) (1 - P / P_max) and the
    #   allocation flux F_C is proportional to J_syn, so dF_C/dV_max = F_C / V_max.
    # Parameter meanings: J_emit, J_deg, J_syn emission, degradation and
    #   synthesis rates, P and P_max pool and capacity, k_25 and k_deg rate
    #   constants, beta the emission temperature coefficient, V_max and q the
    #   synthase maximum rate and Q10, F the allocated carbon flux, K its
    #   half-saturation, F_C the plant-carbon allocation flux.
    # Units: J_emit, J_deg, J_syn, V_max, F, K [umol m-2 s-1]; P, P_max
    #   [umol m-2]; k_25, k_deg [s-1]; beta [K-1]; T, T_25 [K]; q [1];
    #   F_C [kg C m-2 s-1].
    # Domain/constraints: T differs from T_25, so the beta and Q10 gradients
    #   are nonzero; seasonal variation is enabled at day 100, so the amplitude
    #   and peak-day gradients are nonzero; 0 < P < P_max and the pool bounds
    #   stay inactive over one hour.
    # Sign convention: emission, degradation and allocation are nonnegative
    #   magnitudes; each rate constant increases its own rate.
    manager = restore_global_parameter_manager
    _enable_seasonal_variation(manager)
    model = MonoterpeneSynthesis()
    trainable = manager.get_trainable_parameters()
    held = model.held_manager_parameters()
    assert sorted(held) == HELD_MONOTERPENE_BINDING_KEYS

    gradients = []
    try:
        for _ in range(2):
            for binding in held.values():
                binding.grad = None
            update = model(_seasonal_warm_state(model), time=_SEASONAL_TIME, dt=3600.0)
            total = (
                update[_EMISSION_MOLAR].value.sum()
                + update[_DEGRADATION].value.sum()
                + update[_ALLOCATION].value.sum()
            )
            total.backward()
            for key in HELD_MONOTERPENE_BINDING_KEYS:
                assert trainable[key].grad is not None, key
            gradients.append(
                {
                    key: trainable[key].grad.detach().cpu().clone()
                    for key in HELD_MONOTERPENE_BINDING_KEYS
                }
            )
    finally:
        for binding in held.values():
            binding.grad = None

    for key in HELD_MONOTERPENE_BINDING_KEYS:
        assert torch.isfinite(gradients[0][key]).all(), key
        assert bool((gradients[0][key] != 0).all()), key
        torch.testing.assert_close(
            gradients[1][key], gradients[0][key], rtol=0.0, atol=0.0, msg=key
        )
    for key, output in (
        (_K_EMIT_25_KEY, _EMISSION_MOLAR),
        (_K_DEGRADATION_KEY, _DEGRADATION),
        (_V_MAX_MTS_KEY, _ALLOCATION),
    ):
        expected = (
            update[output].value.detach().cpu().to(torch.float64).sum()
            / float(trainable[key].detach())
        )
        torch.testing.assert_close(
            gradients[0][key].to(torch.float64).reshape(()),
            expected.reshape(()),
            rtol=1.0e-5,
            atol=0.0,
            msg=key,
        )


@pytest.mark.parametrize("key", HELD_MONOTERPENE_BINDING_KEYS)
def test_monoterpene_follows_a_changed_binding(restore_global_parameter_manager, key):
    """The coefficients are the manager bindings read at use, so an instance
    built before a binding changed computes like one built afterwards."""
    manager = restore_global_parameter_manager
    _enable_seasonal_variation(manager)
    model = MonoterpeneSynthesis()
    binding = manager.get_trainable_parameters()[key]
    value = 1.1 * float(binding.detach())
    manager.set_value(key, value, units=manager.get_parameter(key).units, validate=True)
    assert float(binding.detach()) == pytest.approx(value)
    fresh = MonoterpeneSynthesis()

    with torch.no_grad():
        expected = _monoterpene_tensors(
            fresh(_seasonal_warm_state(fresh), time=_SEASONAL_TIME, dt=3600.0)
        )
        actual = _monoterpene_tensors(
            model(_seasonal_warm_state(model), time=_SEASONAL_TIME, dt=3600.0)
        )
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )


@pytest.mark.skipif(
    not is_cuda_available(), reason="Held-binding relocation requires a CUDA device"
)
def test_monoterpene_holds_its_bindings_wherever_they_sit(
    restore_global_parameter_manager,
):
    """A held ParameterManager binding has one device, that of the latest
    registration of any Functor holding it. An instance built while its
    bindings sit on CUDA still holds them by reference, stores no converted
    copy and gives the unchanged update on its own device."""
    manager = restore_global_parameter_manager
    _enable_seasonal_variation(manager)
    bindings = manager.get_trainable_parameters()
    devices = {key: bindings[key].device for key in HELD_MONOTERPENE_BINDING_KEYS}
    try:
        for key in HELD_MONOTERPENE_BINDING_KEYS:
            relocate_binding(manager, key, "cpu")
        reference = MonoterpeneSynthesis()
        with torch.no_grad():
            expected = _monoterpene_tensors(
                reference(
                    _seasonal_warm_state(reference), time=_SEASONAL_TIME, dt=3600.0
                )
            )

        for key in HELD_MONOTERPENE_BINDING_KEYS:
            relocate_binding(manager, key, "cuda")
        model = MonoterpeneSynthesis()
        held = model.held_manager_parameters()
        stored_graphs = _stored_graph_tensors(model) + _stored_graph_tensors(
            model.pool_model
        )
        with torch.no_grad():
            actual = _monoterpene_tensors(
                model(_seasonal_warm_state(model), time=_SEASONAL_TIME, dt=3600.0)
            )
    finally:
        for key, device in devices.items():
            relocate_binding(manager, key, device)

    assert sorted(held) == HELD_MONOTERPENE_BINDING_KEYS
    assert all(held[key] is bindings[key] for key in HELD_MONOTERPENE_BINDING_KEYS)
    assert stored_graphs == []
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        assert tensor.device == model.device, name
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )


class TestMonoterpenePoolModelL1:
    """L1 unit tests for monoterpene storage pool dynamics."""
    
    @pytest.fixture
    def pool_model(self):
        """Create monoterpene pool model for testing."""
        param_values = {
            "biosphere.bvoc.monoterpene.v_max_mts": 0.8,
            "biosphere.bvoc.monoterpene.k_gpp": 100.0,
            "biosphere.bvoc.monoterpene.q10_synthase": 2.0,
            "biosphere.bvoc.monoterpene.pool.capacity": 500.0,
            "biosphere.bvoc.monoterpene.pool_initial": 50.0,
            "biosphere.bvoc.monoterpene.pool.k_degradation": 1e-6,
            "biosphere.bvoc.monoterpene.pool.k_emit_25": 2e-5,
            "biosphere.bvoc.monoterpene.pool.beta_emission": 0.09,
        }

        return MonoterpenePoolModel(
            v_max_mts=param_values["biosphere.bvoc.monoterpene.v_max_mts"],
            k_gpp=param_values["biosphere.bvoc.monoterpene.k_gpp"],
            pool_capacity=param_values[
                "biosphere.bvoc.monoterpene.pool.capacity"
            ],
            pool_initial=param_values[
                "biosphere.bvoc.monoterpene.pool_initial"
            ],
            k_emit_25=param_values[
                "biosphere.bvoc.monoterpene.pool.k_emit_25"
            ],
            beta_emission=param_values[
                "biosphere.bvoc.monoterpene.pool.beta_emission"
            ],
            k_degradation=param_values[
                "biosphere.bvoc.monoterpene.pool.k_degradation"
            ],
            q10_synthase=param_values[
                "biosphere.bvoc.monoterpene.q10_synthase"
            ],
            device=torch.device("cpu"),
        )
    
    @staticmethod
    def run_pool_step(model, pool_state, gpp, temperature, dt):
        """Helper to execute a pool update while managing external state."""
        results = model(
            gpp,
            temperature,
            dt,
            pool_state=pool_state,
        )
        next_state = results['pool_concentration'].detach()
        return results, next_state
    
    def test_basic_instantiation(self, pool_model):
        """Test L1.1: Basic instantiation and parameter initialization."""
        assert isinstance(pool_model, MonoterpenePoolModel)
        
        # Check parameter initialization
        assert hasattr(pool_model, 'v_max_mts')
        assert hasattr(pool_model, 'k_gpp')
        assert hasattr(pool_model, 'pool_capacity')
        assert hasattr(pool_model, 'pool_initial')
        
        # Verify parameter values
        assert pool_model.v_max_mts.item() == pytest.approx(0.8)
        assert pool_model.k_gpp.item() == pytest.approx(100.0)
        assert pool_model.pool_capacity.item() == pytest.approx(500.0)
        assert pool_model.pool_initial.item() == pytest.approx(50.0)
        
        # Check emission parameters  
        assert pool_model.k_emit_25.item() == pytest.approx(2e-5)
        assert pool_model.beta_emission.item() == pytest.approx(0.09)
        
    
    def test_synthesis_kinetics(self, pool_model):
        """Test L1.2: Monoterpene synthesis kinetics."""
        # Standard conditions
        gpp_availability = torch.tensor(150.0)  # umol m-2 s-1
        temperature = torch.tensor(298.15)      # K
        dt = 3600.0  # 1 hour
        
        # Reset to known state
        pool_state = pool_model.reset_pool(torch.tensor(100.0))
        results, _ = self.run_pool_step(pool_model, pool_state, gpp_availability, temperature, dt)
        
        # Check output structure
        assert 'synthesis_rate' in results
        assert 'emission_rate' in results
        assert 'degradation_rate' in results
        assert 'pool_concentration' in results
        assert 'pool_capacity_utilization' in results
        
        # Verify synthesis rate calculation
        synthesis_rate = results['synthesis_rate']
        assert synthesis_rate > 0, "Synthesis rate should be positive"
        
        # Check Michaelis-Menten relationship
        # At high [GPP], rate should approach V_max
        v_max_temp = pool_model.v_max_mts * (pool_model.q10_synthase ** ((temperature - 298.15)/10))
        assert synthesis_rate < v_max_temp, "Rate should be below temperature-adjusted V_max"
        
        # Test substrate limitation
        gpp_low = torch.tensor(20.0)  # Low GPP
        pool_state = pool_model.reset_pool(torch.tensor(100.0))
        results_low, _ = self.run_pool_step(pool_model, pool_state, gpp_low, temperature, dt)
        assert results_low['synthesis_rate'] < results['synthesis_rate']
    
    def test_temperature_dependent_emission(self, pool_model):
        """Test L1.3: Temperature-dependent emission kinetics."""
        # Set known pool state
        pool_state = pool_model.reset_pool(torch.tensor(200.0))
        
        gpp = torch.tensor(100.0)
        dt = 1.0  # 1 second for rate calculation
        
        # Temperature series
        temp_cold = torch.tensor(283.15)  # 10 degC  
        temp_ref = torch.tensor(298.15)   # 25 degC
        temp_warm = torch.tensor(313.15)  # 40 degC
        
        results_cold, pool_state = self.run_pool_step(pool_model, pool_state, gpp, temp_cold, dt)
        results_ref, pool_state = self.run_pool_step(pool_model, pool_state, gpp, temp_ref, dt)
        results_warm, _ = self.run_pool_step(pool_model, pool_state, gpp, temp_warm, dt)
        
        emission_cold = results_cold['emission_rate']
        emission_ref = results_ref['emission_rate'] 
        emission_warm = results_warm['emission_rate']
        
        # Should show exponential temperature response
        assert emission_cold < emission_ref < emission_warm
        
        # Check exponential relationship k(T) = k_25 * exp(尾 * (T - T_25))
        beta = pool_model.beta_emission.item()
        k_25 = pool_model.k_emit_25.item()
        
        expected_k_warm = k_25 * math.exp(beta * (313.15 - 298.15))
        expected_emission_warm = expected_k_warm * 200.0  # Pool concentration
        ref_temp = pool_model.temp_ref.item()
        expected_k_warm_ref = k_25 * math.exp(beta * (313.15 - ref_temp))
        expected_emission_warm_ref = expected_k_warm_ref * 200.0  # Pool concentration

        assert emission_warm.detach().cpu().item() == pytest.approx(expected_emission_warm_ref, rel=0.05)

        # Check temperature factor
        temp_factor_warm = results_warm['temperature_emission_factor']
        expected_factor = math.exp(beta * (313.15 - ref_temp))
        assert temp_factor_warm.detach().cpu().item() == pytest.approx(
            expected_factor, rel=0.01
        )
    
    def test_storage_capacity_effects(self, pool_model):
        """Test L1.4: Storage capacity limitation and saturation."""
        gpp = torch.tensor(200.0)  # High GPP to test capacity
        temperature = torch.tensor(298.15)
        dt = 1.0
        
        # Test near capacity
        pool_state = pool_model.reset_pool(torch.tensor(480.0))  # 96% of 500 capacity
        results_near_capacity, _ = self.run_pool_step(pool_model, pool_state, gpp, temperature, dt)
        
        # Test low capacity usage
        pool_state = pool_model.reset_pool(torch.tensor(50.0))   # 10% of capacity
        results_low_capacity, _ = self.run_pool_step(pool_model, pool_state, gpp, temperature, dt)
        
        # Synthesis should be reduced near capacity
        synthesis_near = results_near_capacity['synthesis_rate']
        synthesis_low = results_low_capacity['synthesis_rate']
        
        assert synthesis_near < synthesis_low, "Synthesis should be reduced near capacity"
        
        # Check capacity utilization
        util_near = results_near_capacity['pool_capacity_utilization']
        util_low = results_low_capacity['pool_capacity_utilization']
        
        assert util_near > 0.9, "Should show high capacity utilization"
        assert util_low < 0.2, "Should show low capacity utilization"
        
        # Pool should not exceed capacity
        pool_conc_near = results_near_capacity['pool_concentration']
        assert pool_conc_near <= pool_model.pool_capacity.item()

    def test_pool_boundary_residual_closes_bounded_update(self, pool_model):
        """A bounded pool update must route, not discard, its residual."""
        dt = 100000.0
        results = pool_model(
            torch.tensor(0.0),
            torch.tensor(360.0),
            dt,
            pool_state=torch.tensor(1.0),
        )
        raw_tendency = (
            results['synthesis_rate']
            - results['emission_rate']
            - results['degradation_rate']
        )
        actual_tendency = results['pool_delta'] / dt
        assert torch.allclose(
            actual_tendency,
            raw_tendency + results['pool_boundary_residual_rate'],
            atol=1e-7,
            rtol=1e-6,
        )

    def test_pool_step_requires_explicit_state(self, pool_model):
        """Production dynamics cannot reuse an internal initial pool."""
        with pytest.raises(TypeError):
            pool_model(torch.tensor(1.0), torch.tensor(298.15), 1.0)
    
    def test_pool_dynamics_integration(self, pool_model):
        """Test L1.5: Complete pool dynamics over time."""
        # Start with empty pool
        pool_state = pool_model.reset_pool(torch.tensor(10.0))
        
        gpp = torch.tensor(120.0)
        temperature = torch.tensor(303.15)  # 30 degC
        dt = 3600.0  # 1 hour time steps
        
        pools = [pool_state.item()]
        
        # Simulate 24 hours
        for hour in range(24):
            results, pool_state = self.run_pool_step(pool_model, pool_state, gpp, temperature, dt)
            pools.append(pool_state.item())
        
        # Pool should increase over time (synthesis > emission + degradation)
        assert pools[-1] > pools[0], "Pool should accumulate over time"
        
        # Should show exponential approach toward equilibrium
        # Growth rate should decrease over time
        growth_rates = [pools[i+1] - pools[i] for i in range(len(pools)-1)]
        assert growth_rates[0] > growth_rates[-1], "Growth rate should decrease over time"
    
    def test_degradation_mechanisms(self, pool_model):
        """Test L1.6: Pool degradation and turnover."""
        # Set high pool concentration
        pool_state = pool_model.reset_pool(torch.tensor(400.0))
        
        # Zero synthesis and emission to isolate degradation
        gpp_zero = torch.tensor(0.0)
        temperature_cold = torch.tensor(273.15)  # 0 degC for minimal emission
        dt = 86400.0  # 24 hours
        
        initial_pool = pool_state.item()
        
        # Multiple time steps to observe degradation
        for _ in range(10):
            _, pool_state = self.run_pool_step(pool_model, pool_state, gpp_zero, temperature_cold, dt)
        
        final_pool = pool_state.item()
        
        # Pool should decrease due to degradation
        assert final_pool < initial_pool, "Pool should decrease due to degradation"
        
        # Check degradation rate calculation
        k_deg = pool_model.k_degradation.item()
        expected_degradation = k_deg * initial_pool * dt
        
        # Verify degradation is proportional to pool size
        assert expected_degradation > 0, "Should have positive degradation"
    
    def test_pool_reset_functionality(self, pool_model):
        """Test L1.7: Pool reset and state management."""
        # Change pool state
        original_initial = pool_model.pool_initial.item()
        
        # Modify pool through dynamics
        gpp = torch.tensor(150.0)
        temperature = torch.tensor(308.15)
        dt = 7200.0  # 2 hours
        
        pool_state = pool_model.reset_pool()
        results, pool_state = self.run_pool_step(pool_model, pool_state, gpp, temperature, dt)
        modified_pool = pool_state.item()
        
        # Reset to original
        reset_pool = pool_model.reset_pool().item()
        
        assert reset_pool == pytest.approx(original_initial)
        assert reset_pool != modified_pool
        
        # Test reset to specific value
        target_concentration = torch.tensor(300.0)
        specific_reset = pool_model.reset_pool(target_concentration).item()
        
        assert specific_reset == pytest.approx(300.0)
        
        # Invalid initial conditions fail instead of hiding a mass residual.
        over_capacity = torch.tensor(600.0)  # Above 500 capacity
        with pytest.raises(ValueError, match="pool_capacity"):
            pool_model.reset_pool(over_capacity)


class TestMonoterpeneSynthesisL1:
    """L1 unit tests for complete monoterpene synthesis model."""
    
    @pytest.fixture
    def monoterpene_model(self):
        """Create monoterpene synthesis model for testing."""
        return MonoterpeneSynthesis()
    
    def test_basic_instantiation(self, monoterpene_model):
        """Test L1.1: Basic instantiation and configuration."""
        assert isinstance(monoterpene_model, MonoterpeneSynthesis)
        
        # Check components
        assert hasattr(monoterpene_model, 'pool_model')
        assert hasattr(monoterpene_model, 'f_carbon_monoterpenes')
        assert hasattr(monoterpene_model, 'monoterpene_mw')
        
        # Check dependencies and provides
        deps = monoterpene_model.get_dependencies()
        provides = monoterpene_model.get_provides()
        
        assert deps == ['farquhar_photosynthesis']
        assert monoterpene_model.metadata.input_variables == [
            "air_temperature",
            "gross_photosynthesis",
            "monoterpene_pool_concentration",
        ]
        assert "seasonal_period_seconds" not in (
            monoterpene_model.metadata.parameters
        )
        
        expected_outputs = {
            "monoterpenes_emission",
            "monoterpene_pool_concentration",
            "bvoc_carbon_allocation_flux",
            "diagnostics.bvoc.monoterpene.emission_rate_molar",
            "diagnostics.bvoc.monoterpene.emission_rate_mass",
            "diagnostics.bvoc.monoterpene.pool_tendency",
            "diagnostics.bvoc.monoterpene.pool_delta",
            "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate",
            "diagnostics.bvoc.monoterpene.storage_change",
            "diagnostics.bvoc.monoterpene.synthesis_rate",
            "diagnostics.bvoc.monoterpene.pool_capacity_utilization",
            "diagnostics.bvoc.monoterpene.temperature_emission_factor",
            "diagnostics.bvoc.monoterpene.carbon_flux_allocated",
            "diagnostics.bvoc.monoterpene.degradation_rate",
            "diagnostics.bvoc.monoterpene.seasonal_factor",
        }
        assert monoterpene_model.name == "mechanistic_monoterpene_synthesis"
        assert set(provides) == expected_outputs
        assert set(monoterpene_model.metadata.output_variables) == expected_outputs
        assert "monoterpene_emission_mechanistic" not in provides
        assert "bvoc_monoterpene_carbon_loss" not in provides
        
        # Check molecular weight
        mw = monoterpene_model.monoterpene_mw.item()
        assert mw == pytest.approx(136.24, rel=0.01)  # Average C10H16

    def test_runtime_writer_name_is_canonical(self):
        """Reject names that cannot match the unique SCM writer identity."""
        with pytest.raises(ValueError, match="mechanistic_monoterpene_synthesis"):
            MonoterpeneSynthesis(name="test_monoterpene_synthesis")

    def test_references_are_registered_bibliographic_records(self):
        """Metadata, certificate and module references name registered works: the
        unregistered Ghirardo et al. (2010) Plant Molecular Biology and Niinemets et
        al. (2014) Biogeosciences entries are retired, Dudareva et al. carries its
        registered 2006 record, and the Niinemets et al. (2014) short citations
        resolve to the registered Plant, Cell & Environment review."""
        import relic.scientific.biosphere.bvoc.synthesis.monoterpene as module
        from relic.computation.operators.iphs_certificate import (
            load_iphs_equivalence_certificate,
        )

        path = (
            Path(__file__).resolve().parents[4]
            / "docs"
            / "source"
            / "theory"
            / "iphs_certificates"
            / "biosphere_bvoc_synthesis_monoterpene_monoterpene_synthesis.yaml"
        )
        certificate = load_iphs_equivalence_certificate(path)
        declared = list(MonoterpeneSynthesis().metadata.references)
        texts = (
            " | ".join(declared),
            " | ".join(certificate.references),
            module.__doc__,
        )
        for text in texts:
            for retired in (
                "Plant Molecular Biology 74",
                "Critical Reviews in Plant Sciences 22",
                "6655-6680",
            ):
                assert retired not in text, retired
        dudareva = (
            "Dudareva, N., et al. (2006). Plant volatiles: recent advances and future "
            "perspectives. Critical Reviews in Plant Sciences 25, 417-440."
        )
        assert dudareva in declared
        assert dudareva in certificate.references
        for text in texts[1:]:
            assert "Plant, Cell & Environment 37, 1790-1809" in text

    def test_emission_reference_is_exactly_25_c(self, monoterpene_model):
        """Normalize k_emit_25 at 298.15 K, not at the MEGAN 30 C reference."""
        pool = torch.tensor(
            200.0, dtype=monoterpene_model.pool_model.k_emit_25.dtype
        )
        result = monoterpene_model.pool_model(
            torch.tensor(100.0),
            torch.tensor(298.15),
            1.0,
            pool_state=pool,
        )
        torch.testing.assert_close(
            result["temperature_emission_factor"],
            torch.ones_like(result["temperature_emission_factor"]),
        )
        torch.testing.assert_close(
            result["emission_rate"],
            monoterpene_model.pool_model.k_emit_25 * pool,
        )

    def test_pool_state_contract_has_exact_runtime_owner(self, monoterpene_model):
        """Authorize only the canonical full-state pool writer."""
        project_root = Path(__file__).resolve().parents[4]
        contract_path = (
            project_root
            / "relic"
            / "contracts"
            / "variables"
            / "biosphere"
            / "monoterpene_pool_concentration.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        assert contract["owner"] == monoterpene_model.name
        assert contract["incrementable"] is False
        assert contract["writers"] == {"increments": [], "tendencies": []}
        assert contract["provenance"]["runtime_writer"] == monoterpene_model.name
        assert contract["provenance"]["runtime_role"] == (
            "monoterpene_pool_full_state"
        )

    def test_canonical_emission_contract_has_exact_runtime_writer(
        self, monoterpene_model
    ):
        """Authorize the actual mechanistic runtime name on the molar port."""
        project_root = Path(__file__).resolve().parents[4]
        contract_path = (
            project_root
            / "relic"
            / "contracts"
            / "variables"
            / "biosphere"
            / "monoterpenes_emission.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        writer = next(
            item
            for item in contract["writers"]["tendencies"]
            if item["module"].endswith("synthesis.monoterpene")
        )

        assert writer == {
            "name": monoterpene_model.name,
            "module": "relic.scientific.biosphere.bvoc.synthesis.monoterpene",
            "provides": ["monoterpenes_emission"],
            "unit": "mol m-2 s-1",
        }
        assert monoterpene_model.name in contract["flux_contract"]["allowed_in"]
        assert monoterpene_model.name in contract["provenance"]["runtime_writers"]

    def test_mass_diagnostic_and_carbon_port_contracts(self, monoterpene_model):
        """Keep mass observation and open carbon export distinct from molar flux."""
        project_root = Path(__file__).resolve().parents[4]
        contract_dir = (
            project_root / "relic" / "contracts" / "variables" / "biosphere"
        )
        mass_diagnostic = yaml.safe_load(
            (contract_dir / "monoterpene_emission_mechanistic.yaml").read_text(
                encoding="utf-8"
            )
        )
        carbon_port = yaml.safe_load(
            (contract_dir / "bvoc_isoprene_carbon_loss.yaml").read_text(
                encoding="utf-8"
            )
        )

        assert mass_diagnostic["variable"] == (
            "diagnostics.bvoc.monoterpene.emission_rate_mass"
        )
        assert mass_diagnostic["kind"] == "diagnostic"
        assert mass_diagnostic["dimension"]["udunits"] == "ug m-2 s-1"
        assert mass_diagnostic["owner"] == monoterpene_model.name
        assert carbon_port["kind"] == "flux"
        assert carbon_port["variable"] == "bvoc_carbon_allocation_flux"
        assert carbon_port["dimension"]["udunits"] == "kg C m-2 s-1"
        assert carbon_port["incrementable"] is False
        assert monoterpene_model.name in (
            carbon_port["provenance"]["runtime_writers"]
        )

    def test_required_state_ports_fail_fast(self, monoterpene_model):
        """All three canonical Tensor inputs must be explicit and valid."""
        pool_state = monoterpene_model.reset_pools()
        canonical = {
            "air_temperature": torch.tensor(298.15),
            "gross_photosynthesis": torch.tensor(10.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: pool_state,
        }
        with pytest.raises(KeyError, match="gross_photosynthesis"):
            monoterpene_model.forward(
                {
                    "air_temperature": torch.tensor(298.15),
                    MonoterpeneSynthesis.POOL_STATE_KEY: pool_state,
                },
                time=0.0,
                dt=1.0,
            )
        with pytest.raises(KeyError, match="monoterpene_pool_concentration"):
            monoterpene_model.forward(
                {
                    "air_temperature": torch.tensor(298.15),
                    "gross_photosynthesis": torch.tensor(10.0),
                },
                time=0.0,
                dt=1.0,
            )
        with pytest.raises(KeyError, match="air_temperature"):
            monoterpene_model.forward(
                {
                    "temperature": torch.tensor(298.15),
                    "gross_photosynthesis": torch.tensor(10.0),
                    MonoterpeneSynthesis.POOL_STATE_KEY: pool_state,
                },
                time=0.0,
                dt=1.0,
            )
        for key in canonical:
            invalid = dict(canonical)
            invalid[key] = 298.15
            with pytest.raises(TypeError, match=key):
                monoterpene_model.forward(invalid, time=0.0, dt=1.0)
        for key in canonical:
            invalid = dict(canonical)
            invalid[key] = torch.tensor(float("nan"))
            with pytest.raises(ValueError, match=key):
                monoterpene_model.forward(invalid, time=0.0, dt=1.0)
        for key in ("gross_photosynthesis", MonoterpeneSynthesis.POOL_STATE_KEY):
            invalid = dict(canonical)
            invalid[key] = torch.tensor(-1.0)
            with pytest.raises(ValueError, match=key):
                monoterpene_model.forward(invalid, time=0.0, dt=1.0)
    
    def test_photosynthetic_coupling_validation(self, monoterpene_model):
        """Test L1.2: Photosynthetic coupling mechanisms."""
        # Standard photosynthetic state
        state = {
            'air_temperature': torch.tensor(301.15),  # K
            'gross_photosynthesis': torch.tensor(25.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        
        results = monoterpene_model.forward(state, time=0.0, dt=3600.0)

        assert set(results) == set(monoterpene_model.metadata.output_variables)
        assert "monoterpene_emission_mechanistic" not in results
        expected_units = {
            "monoterpenes_emission": "mol m-2 s-1",
            "monoterpene_pool_concentration": "umol m-2",
            "bvoc_carbon_allocation_flux": "kg C m-2 s-1",
            "diagnostics.bvoc.monoterpene.emission_rate_molar": "umol m-2 s-1",
            "diagnostics.bvoc.monoterpene.emission_rate_mass": "ug m-2 s-1",
            "diagnostics.bvoc.monoterpene.pool_tendency": "umol m-2 s-1",
            "diagnostics.bvoc.monoterpene.pool_delta": "umol m-2",
            "diagnostics.bvoc.monoterpene.pool_boundary_residual_rate": "umol m-2 s-1",
            "diagnostics.bvoc.monoterpene.storage_change": "kg C m-2 s-1",
            "diagnostics.bvoc.monoterpene.synthesis_rate": "umol m-2 s-1",
            "diagnostics.bvoc.monoterpene.pool_capacity_utilization": "1",
            "diagnostics.bvoc.monoterpene.temperature_emission_factor": "1",
            "diagnostics.bvoc.monoterpene.carbon_flux_allocated": "umol CO2 m-2 s-1",
            "diagnostics.bvoc.monoterpene.degradation_rate": "umol m-2 s-1",
            "diagnostics.bvoc.monoterpene.seasonal_factor": "1",
        }
        for key, expected_unit in expected_units.items():
            item = results[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == expected_unit
            assert item.source_functor == monoterpene_model.name
        
        # Verify carbon allocation
        gpp = state['gross_photosynthesis']
        carbon_allocated = results[
            'diagnostics.bvoc.monoterpene.carbon_flux_allocated'
        ].value
        f_carbon = monoterpene_model.f_carbon_monoterpenes.item()
        
        # Should account for seasonal variation
        seasonal_factor = results[
            'diagnostics.bvoc.monoterpene.seasonal_factor'
        ].value
        expected_allocation = gpp * f_carbon * seasonal_factor
        
        # Convert to CPU before numpy comparison
        carbon_allocated_cpu = carbon_allocated.detach().cpu().item() if hasattr(carbon_allocated, 'cpu') else float(carbon_allocated)
        expected_allocation_val = expected_allocation if isinstance(expected_allocation, (int, float)) else expected_allocation.item()
        assert carbon_allocated_cpu == pytest.approx(expected_allocation_val, rel=0.01)

        synthesis_mol = unit_converter.convert(
            results[
                "diagnostics.bvoc.monoterpene.synthesis_rate"
            ].value,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        expected_carbon_allocation = unit_converter.convert(
            synthesis_mol
            * monoterpene_model.monoterpene_mw
            * monoterpene_model.carbon_fraction_monoterpenes,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        torch.testing.assert_close(
            results["bvoc_carbon_allocation_flux"].value,
            expected_carbon_allocation,
        )
        assert torch.all(
            results["bvoc_carbon_allocation_flux"].value >= 0
        )
        
        # Monoterpene emission should be positive (all three outputs)
        emission_standard = results['monoterpenes_emission'].value
        emission_mass = results[
            'diagnostics.bvoc.monoterpene.emission_rate_mass'
        ].value
        emission_molar = results[
            'diagnostics.bvoc.monoterpene.emission_rate_molar'
        ].value
        
        assert emission_standard > 0, "Standard emission (mol) should be positive"
        assert emission_mass > 0, "Mass emission (ug) should be positive"
        assert emission_molar > 0, "Molar emission (umol) should be positive"
        
        # Check unit conversion: umol m-2 s-1 -> mol m-2 s-1
        expected_standard = unit_converter.convert(
            emission_molar,
            variable_name="molar_flux",
            source_units="umol m-2 s-1",
            target_units="mol m-2 s-1",
        )
        assert torch.allclose(emission_standard, expected_standard, rtol=0.01), \
            "Standard output should match molar/1e6 conversion"
        
        # Check mass unit conversion: umol m-2 s-1 * g/mol -> ug m-2 s-1
        mw = monoterpene_model.monoterpene_mw.item()
        expected_mass = emission_molar * mw
        assert torch.allclose(emission_mass, expected_mass, rtol=0.01), \
            "Mass output should match umol * MW conversion"
    
    def test_seasonal_variation_effects(
        self, monoterpene_model, restore_global_parameter_manager
    ):
        """Test L1.3: Seasonal variation in synthesis."""
        base_state = {
            'air_temperature': torch.tensor(298.15),
            'gross_photosynthesis': torch.tensor(20.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: torch.tensor(50.0, dtype=torch.float32),
        }
        monoterpene_model.enable_seasonal_variation = True
        # The amplitude is the ParameterManager binding the model holds by
        # reference: it is changed through its owner and the fixture restores it.
        manager = restore_global_parameter_manager
        manager.set_value(
            _SEASONAL_AMPLITUDE_KEY,
            0.3,
            units=manager.get_parameter(_SEASONAL_AMPLITUDE_KEY).units,
            validate=True,
        )
        assert monoterpene_model.seasonal_amplitude.item() == pytest.approx(0.3)
        
        # Test different times of year
        winter_time = 0.0                      # Start of year
        spring_time = 120 * 24 * 3600          # Day 120 (late April)
        summer_time = 180 * 24 * 3600          # Day 180 (late June)
        fall_time = 270 * 24 * 3600            # Day 270 (late September)
        
        results_winter = monoterpene_model.forward(base_state, time=winter_time, dt=3600.0)
        results_spring = monoterpene_model.forward(base_state, time=spring_time, dt=3600.0)
        results_summer = monoterpene_model.forward(base_state, time=summer_time, dt=3600.0)
        results_fall = monoterpene_model.forward(base_state, time=fall_time, dt=3600.0)
        
        # Extract seasonal factors
        factor_winter = results_winter[
            'diagnostics.bvoc.monoterpene.seasonal_factor'
        ].value
        factor_spring = results_spring[
            'diagnostics.bvoc.monoterpene.seasonal_factor'
        ].value
        factor_summer = results_summer[
            'diagnostics.bvoc.monoterpene.seasonal_factor'
        ].value
        factor_fall = results_fall[
            'diagnostics.bvoc.monoterpene.seasonal_factor'
        ].value
        
        # Seasonal variation is controlled by enable_seasonal_variation config flag
        # If disabled (default in this test), all factors will be 1.0
        # Just verify factors are positive and reasonable
        for factor in [factor_winter, factor_spring, factor_summer, factor_fall]:
            assert factor > 0.0, "Seasonal factors should be positive"
        
        # All factors should be between 0.1 and 1.5
        for factor in [factor_winter, factor_spring, factor_summer, factor_fall]:
            assert 0.1 <= factor <= 1.5
        
        # Emissions are influenced by both seasonal factors and temperature
        # At the same temperature, seasonal factors should show variation
        assert factor_winter != factor_summer
        flux_winter = results_winter[
            'diagnostics.bvoc.monoterpene.carbon_flux_allocated'
        ].value
        flux_summer = results_summer[
            'diagnostics.bvoc.monoterpene.carbon_flux_allocated'
        ].value
        assert flux_winter.detach().cpu().item() != flux_summer.detach().cpu().item(), "Seasonal variation should affect carbon allocation"
        
        emission_winter = results_winter['monoterpenes_emission'].value
        emission_summer = results_summer['monoterpenes_emission'].value
        assert emission_winter.detach().cpu().item() > 0
        assert emission_summer.detach().cpu().item() > 0
    
    def test_temperature_response_complete_system(self, monoterpene_model):
        """Test L1.4: Complete system temperature response."""
        base_state = {
            'gross_photosynthesis': torch.tensor(22.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        
        # Temperature series
        temperatures = [283.15, 298.15, 313.15, 328.15]  # 10, 25, 40, 55 degC
        emissions = []
        pool_concentrations = []
        
        for temp in temperatures:
            state = base_state.copy()
            state['air_temperature'] = torch.tensor(temp)
            
            # Reset pools for consistent comparison
            monoterpene_model.reset_pools()
            
            results = monoterpene_model.forward(state, time=0.0, dt=3600.0)
            emission = results['monoterpenes_emission'].value.item()
            pool_conc = results['monoterpene_pool_concentration'].value.item()
            
            emissions.append(emission)
            pool_concentrations.append(pool_conc)
        
        # Emissions should increase strongly with temperature
        assert all(emissions[i] < emissions[i+1] for i in range(len(emissions)-1))
        
        # Temperature response should be exponential-like
        temp_coefficient = emissions[-1] / emissions[0]  # 55 degC vs 10 degC
        assert temp_coefficient > 10, "Should show strong temperature sensitivity"
        
        # Pool dynamics respond to temperature
        # Higher temp -> more emission, but also more synthesis (Q10 effect)
        # Net effect depends on pool model parameters
        # Just verify all pool concentrations are reasonable
        for pool_conc in pool_concentrations:
            assert pool_conc >= 0.0, "Pool concentration should be non-negative"
    
    def test_synthesis_emission_decoupling(self, monoterpene_model):
        """Test L1.5: Decoupling between synthesis and emission."""
        # Reset pools to empty state to avoid capacity limitations
        low_pool = monoterpene_model.reset_pools(torch.tensor(10.0))
        
        # High synthesis conditions (cool temp, high GPP)
        state_high_synthesis = {
            'air_temperature': torch.tensor(293.15),
            'gross_photosynthesis': torch.tensor(35.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: low_pool,
        }
        
        results_synthesis = monoterpene_model.forward(state_high_synthesis, time=0.0, dt=3600.0)
        
        # Reset pools to same starting point
        low_pool = monoterpene_model.reset_pools(torch.tensor(10.0))
        
        # Low synthesis conditions (cool temp, low GPP)  
        state_low_synthesis = {
            'air_temperature': torch.tensor(293.15),
            'gross_photosynthesis': torch.tensor(5.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: low_pool,
        }
        
        results_low_synthesis = monoterpene_model.forward(state_low_synthesis, time=0.0, dt=3600.0)
        
        # High GPP should lead to more synthesis when not capacity limited
        synthesis_high_gpp = results_synthesis[
            'diagnostics.bvoc.monoterpene.synthesis_rate'
        ].value
        synthesis_low_gpp = results_low_synthesis[
            'diagnostics.bvoc.monoterpene.synthesis_rate'
        ].value
        
        # Convert to CPU for comparison if needed
        if hasattr(synthesis_high_gpp, 'cpu'):
            synthesis_high_gpp_val = synthesis_high_gpp.detach().cpu().item()
            synthesis_low_gpp_val = synthesis_low_gpp.detach().cpu().item()
        else:
            synthesis_high_gpp_val = float(synthesis_high_gpp)
            synthesis_low_gpp_val = float(synthesis_low_gpp)
        
        assert synthesis_high_gpp_val > synthesis_low_gpp_val, f"High GPP synthesis {synthesis_high_gpp_val} should exceed low GPP synthesis {synthesis_low_gpp_val}"
        
        # Test temperature effect on emission with filled pools
        high_pool = monoterpene_model.reset_pools(torch.tensor(200.0))
        
        # High emission conditions  
        state_high_emission = {
            'air_temperature': torch.tensor(318.15),
            'gross_photosynthesis': torch.tensor(5.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: high_pool,
        }
        
        results_emission = monoterpene_model.forward(state_high_emission, time=0.0, dt=3600.0)
        
        high_pool = monoterpene_model.reset_pools(torch.tensor(200.0))
        
        # Cool emission conditions
        state_cool_emission = {
            'air_temperature': torch.tensor(293.15),
            'gross_photosynthesis': torch.tensor(5.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: high_pool,
        }
        
        results_cool = monoterpene_model.forward(state_cool_emission, time=0.0, dt=3600.0)
        
        # High temperature should lead to more emission
        emission_cool = results_cool['monoterpenes_emission'].value
        emission_warm = results_emission['monoterpenes_emission'].value
        
        assert emission_warm.detach().cpu().item() > emission_cool.detach().cpu().item(), "Warm temperature should increase emission"
        
        # Pool concentrations should reflect synthesis-emission balance
        pool_synthesis = results_synthesis['monoterpene_pool_concentration'].value
        pool_emission = results_emission['monoterpene_pool_concentration'].value
        
        # High synthesis with low initial pool -> pool increase
        # High emission with high initial pool -> pool decrease  
        assert pool_synthesis.detach().cpu().item() > 10.0, "High synthesis should increase pool from low start"
        assert pool_emission.detach().cpu().item() < 200.0, "High emission should decrease pool from high start"
    
    def test_pool_diagnostics_functionality(self, monoterpene_model):
        """Test L1.6: Pool diagnostic capabilities."""
        # Set known pool state
        pool_state = monoterpene_model.reset_pools(torch.tensor(150.0))
    
        diagnostics = monoterpene_model.get_pool_diagnostics(
            {MonoterpeneSynthesis.POOL_STATE_KEY: pool_state}
        )
        
        # Check diagnostic structure
        assert 'current_pool_concentration' in diagnostics
        assert 'pool_capacity' in diagnostics
        assert 'capacity_utilization' in diagnostics
        assert 'pool_status' in diagnostics
        
        # Verify values
        assert diagnostics['current_pool_concentration'] == pytest.approx(150.0)
        
        capacity = monoterpene_model.pool_model.pool_capacity.item()
        expected_utilization = 150.0 / capacity
        assert diagnostics['capacity_utilization'] == pytest.approx(expected_utilization, rel=0.01)
        
        # Test status classification
        # Normal status (10% < utilization < 90%)
        assert diagnostics['pool_status'] == 'normal'
        
        # Test full status
        pool_state = monoterpene_model.reset_pools(torch.tensor(0.95 * capacity))
        diagnostics_full = monoterpene_model.get_pool_diagnostics(
            {MonoterpeneSynthesis.POOL_STATE_KEY: pool_state}
        )
        assert diagnostics_full['pool_status'] == 'full'
        
        # Test empty status
        pool_state = monoterpene_model.reset_pools(torch.tensor(0.05 * capacity))
        diagnostics_empty = monoterpene_model.get_pool_diagnostics(
            {MonoterpeneSynthesis.POOL_STATE_KEY: pool_state}
        )
        assert diagnostics_empty['pool_status'] == 'empty'
    
    def test_parameter_sensitivity_analysis(self, monoterpene_model):
        """Test L1.7: Parameter sensitivity analysis capabilities."""
        state = {
            'air_temperature': torch.tensor(298.15),
            'gross_photosynthesis': torch.tensor(20.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        assert "parameter_sensitivity" not in MonoterpeneSynthesis.__dict__
        hour = get_physical_constant("seconds_per_hour")

        # f_carbon is the ParameterManager binding held by reference: it is no
        # tensor member functional_call can replace, so its sensitivity is taken
        # on the manager calibration path, the gradient on the binding.
        with pytest.raises(TypeError, match="held by reference"):
            monoterpene_model.parameter_sensitivity(
                'f_carbon_monoterpenes', state, time=0.0, dt=hour,
                target_output='monoterpenes_emission',
            )
        f_carbon = get_global_manager().get_trainable_parameters()[_F_CARBON_KEY]
        assert monoterpene_model.f_carbon_monoterpenes is f_carbon
        held = monoterpene_model.held_manager_parameters()
        try:
            # f_carbon sets the synthesis into the storage pool and the carbon allocation;
            # the emission of the step is released from the pool state it starts from.
            for binding in held.values():
                binding.grad = None
            outputs = monoterpene_model(dict(state), time=0.0, dt=hour)
            outputs['monoterpenes_emission'].value.sum().backward()
            assert f_carbon.grad is None or bool((f_carbon.grad == 0).all())

            for binding in held.values():
                binding.grad = None
            outputs = monoterpene_model(dict(state), time=0.0, dt=hour)
            outputs['bvoc_carbon_allocation_flux'].value.sum().backward()
            allocation_sensitivity = f_carbon.grad.detach().clone()
        finally:
            for binding in held.values():
                binding.grad = None
        assert torch.isfinite(allocation_sensitivity).all()
        assert allocation_sensitivity.item() > 0.0

        # The canonical molar atmosphere port is independent of molecular weight.
        sensitivity_mw = monoterpene_model.parameter_sensitivity(
            'monoterpene_mw', state, time=0.0, dt=hour,
            target_output='monoterpenes_emission',
        )
        assert torch.isfinite(sensitivity_mw).all()
        torch.testing.assert_close(
            sensitivity_mw,
            torch.zeros_like(sensitivity_mw),
            atol=1.0e-12,
            rtol=0.0,
        )
    
    def test_boundary_conditions_robustness(self, monoterpene_model):
        """Test L1.8: Boundary conditions and robustness."""
        # Test extreme low conditions
        state_extreme_low = {
            'air_temperature': torch.tensor(268.15),
            'gross_photosynthesis': torch.tensor(0.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        
        results_low = monoterpene_model.forward(state_extreme_low, time=0.0, dt=3600.0)
        
        # Should handle gracefully
        assert torch.isfinite(results_low['monoterpenes_emission'].value).all()
        assert results_low['monoterpenes_emission'].value >= 0
        
        # Synthesis should be near zero with no GPP
        assert results_low[
            'diagnostics.bvoc.monoterpene.synthesis_rate'
        ].value.detach().cpu().item() == pytest.approx(0.0, abs=1e-6)
        
        # Test extreme high conditions
        state_extreme_high = {
            'air_temperature': torch.tensor(333.15),
            'gross_photosynthesis': torch.tensor(200.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        
        results_high = monoterpene_model.forward(state_extreme_high, time=0.0, dt=3600.0)
        
        # Should handle gracefully
        assert torch.isfinite(results_high['monoterpenes_emission'].value).all()
        assert results_high['monoterpenes_emission'].value >= 0
        
        # High temperature should give high emission if pools are available
        if results_high['monoterpene_pool_concentration'].value > 0:
            assert results_high[
                'diagnostics.bvoc.monoterpene.temperature_emission_factor'
            ].value > 5.0
    
    def test_compute_rate_interface_consistency(self, monoterpene_model):
        """Test L1.9: BiologicalProcess interface consistency."""
        state = {
            'air_temperature': torch.tensor(298.15),
            'gross_photosynthesis': torch.tensor(20.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: monoterpene_model.reset_pools(),
        }
        environment = {}
        
        # Test compute_rate method (BiologicalProcess interface)
        rate_result = call_forward(monoterpene_model, {**state, **environment})
        
        assert 'rate' in rate_result
        rate = rate_result['rate']
        assert torch.isfinite(rate).all()
        assert rate >= 0
        
        # Should match forward method result
        forward_result = monoterpene_model.forward({**state, **environment}, time=0.0, dt=1.0)
        forward_emission = forward_result['monoterpenes_emission'].value
        
        assert torch.allclose(rate, forward_emission, rtol=0.01)

    def test_simulator_preserves_runtime_name_units_and_provenance(self):
        """Commit the canonical molar port and pool through the real simulator."""
        model = MonoterpeneSynthesis()
        state = {
            "air_temperature": torch.tensor(303.15),
            "gross_photosynthesis": torch.tensor(20.0),
            MonoterpeneSynthesis.POOL_STATE_KEY: model.reset_pools(),
        }
        direct = model.forward(state, time=0.0, dt=60.0)
        simulator = RELICSimulator(
            {
                "time_step": 60.0,
                "max_time": 120.0,
                "device": "cpu",
            }
        )
        simulator.initialize_state(
            {
                **state,
                "monoterpenes_emission": torch.tensor(0.0),
                "bvoc_carbon_allocation_flux": torch.tensor(0.0),
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulator.register_functor(model)
            simulator.step(dt=60.0)

        assert caught == []
        actual_emission = simulator.state_manager.get_state(
            "monoterpenes_emission"
        )
        expected_emission = direct["monoterpenes_emission"].value.to(
            dtype=actual_emission.dtype
        )
        torch.testing.assert_close(actual_emission, expected_emission)
        actual_pool = simulator.state_manager.get_state(
            MonoterpeneSynthesis.POOL_STATE_KEY
        )
        expected_pool = direct[MonoterpeneSynthesis.POOL_STATE_KEY].value.to(
            dtype=actual_pool.dtype
        )
        torch.testing.assert_close(actual_pool, expected_pool)
        for variable in (
            "monoterpenes_emission",
            MonoterpeneSynthesis.POOL_STATE_KEY,
            "bvoc_carbon_allocation_flux",
        ):
            provenance = simulator.get_variable_provenance(variable)
            assert {record["functor"] for record in provenance} == {model.name}
            final_writer = [
                record for record in provenance if record["role"] == "final_writer"
            ]
            assert final_writer == [
                {
                    "functor": model.name,
                    "mode": "UpdateMode.REPLACE",
                    "step": None,
                    "role": "final_writer",
                }
            ]
        assert simulator.get_effective_device(model.name).type == "cpu"


class TestSesquiterpeneSynthesisL1:
    """L1 unit tests for sesquiterpene synthesis placeholder."""
    
    @pytest.fixture
    def sesquiterpene_model(self):
        """Create sesquiterpene synthesis model for testing."""
        return SesquiterpeneSynthesis()
    
    def test_canonical_functionality(self, sesquiterpene_model):
        """Test L1.1: Canonical molar-flux interface consistency."""
        assert isinstance(sesquiterpene_model, SesquiterpeneSynthesis)

        state = {
            'temperature': torch.tensor(298.15),
            'leaf_area_index': torch.tensor(1.0),
        }
        environment = {'temperature': torch.tensor(298.15)}

        rate_result = sesquiterpene_model._compute_rate_core({**state, **environment})
        assert 'rate' in rate_result
        assert rate_result['rate'] >= 0.0

        forward_result = sesquiterpene_model.forward(state, time=0.0, dt=3600.0)
        assert set(forward_result) == set(
            sesquiterpene_model.metadata.output_variables
        )
        assert 'sesquiterpene_emission_mechanistic' not in forward_result
        emission = forward_result['sesquiterpenes_emission']
        assert isinstance(emission, UpdateItem)
        assert emission.mode is UpdateMode.REPLACE
        assert emission.unit == 'mol m-2 s-1'
        assert emission.value >= 0.0
        assert callable(sesquiterpene_model.parameter_sensitivity)


def test_monoterpene_has_no_uncalled_rate_core():
    """Round 11: the _compute_rate_core wrapper, which only re-ran forward and had no relic
    caller, is retired; forward is the one kernel."""
    assert not hasattr(MonoterpeneSynthesis, "_compute_rate_core")


def test_monoterpene_synthesis_passes_and_assigns_no_bridge_label():
    """12.16: no algorithm reads the BiologicalProcess temperature_response or
    ph_dependence labels (the synthase kinetics call q10_temperature_response directly), so the module neither passes nor assigns them."""
    import inspect

    from relic.scientific.biosphere.bvoc.synthesis import monoterpene

    source = inspect.getsource(monoterpene)
    for label in ("temperature_response", "ph_dependence"):
        assert f"self.{label}" not in source
        assert f"{label}=" not in source


if __name__ == "__main__":
    pytest.main([__file__])


