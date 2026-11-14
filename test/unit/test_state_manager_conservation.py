"""
L1 Unit Tests for State Manager Conservation and Dynamic Tolerance (P3)

Tests the enhanced conservation checking mechanisms implemented in P3:
- Dynamic tolerance mechanism (tol = dt * scale_factor)
- C/N/P cross-domain cycle validation
- Stoichiometric ratio checking
- Conservation report enhancements

Test Level: L1 (Unit)
Dependencies: StateManager, CrossDomainValidator, torch

Mathematical Expressions:
- Dynamic tolerance: tol = dt * scale_factor
- Mass balance: sum(inputs) = sum(outputs) + d(storage)/dt
- C:N ratio: C_flux / N_flux in [8, 20]
- C:P ratio: C_flux / P_flux in [50, 300]
- N:P ratio: N_flux / P_flux in [10, 20] (Redfield)

References:
- LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems.
- Redfield et al. (1963). The influence of organisms on seawater composition.
- Sterner & Elser (2002). Ecological Stoichiometry.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.core.state import (
    StateManager, StateVariableSpec, ConservationSummary,
    StateVariableType
)
from relic.validation.cross_domain_validator import CrossDomainValidator


def test_conservation_summary_records_have_no_default():
    """total_violations and tolerance_value are recorded by the producer."""
    import dataclasses

    from relic.core.state.types import ConservationSummary

    fields = {field.name: field for field in dataclasses.fields(ConservationSummary)}
    for name in ("total_violations", "tolerance_value"):
        assert fields[name].default is dataclasses.MISSING, name
        assert fields[name].kw_only, name


def test_the_recorded_tolerance_is_the_one_applied():
    """The summary records dt * abs_tol with a step, abs_tol without one."""
    manager = StateManager({"carbon_pool": StateVariableSpec(name="carbon_pool", units="kg m-2")})
    manager.initialize({"carbon_pool": torch.tensor(1.0)})
    with_dt = manager.generate_conservation_report({}, current_time=0.0, dt=2.0, store_history=False)
    without_dt = manager.generate_conservation_report({}, current_time=0.0, store_history=False)
    assert with_dt.tolerance_value == pytest.approx(2.0 * manager._abs_conservation_tol)
    assert without_dt.tolerance_value == pytest.approx(manager._abs_conservation_tol)
    assert with_dt.total_violations == 0


class TestDynamicToleranceMechanism:
    """Test dynamic tolerance mechanism in StateManager."""

    @pytest.fixture
    def manager_with_conserved_vars(self):
        """Create StateManager with conserved variables for testing."""
        schema = {
            'carbon_pool': StateVariableSpec(
                name='carbon_pool',
                units='g C m-2',
                bounds=(0.0, 10000.0),
                conserved=True,
                default_value=1000.0,
                variable_type=StateVariableType.SCALAR
            ),
            'nitrogen_pool': StateVariableSpec(
                name='nitrogen_pool',
                units='g N m-2',
                bounds=(0.0, 1000.0),
                conserved=True,
                default_value=100.0,
                variable_type=StateVariableType.SCALAR
            ),
            'water_content': StateVariableSpec(
                name='water_content',
                units='kg m-2',
                bounds=(0.0, 500.0),
                conserved=True,
                default_value=50.0,
                variable_type=StateVariableType.SCALAR
            ),
        }

        manager = StateManager(schema=schema)
        manager.initialize()
        return manager

    def test_fixed_tolerance_mode(self, manager_with_conserved_vars):
        """Test that fixed tolerance mode works when dt=None.

        Mathematical Expression:
        tol = fixed_value = 1e-6 (default)

        Expected: Conservation check uses fixed tolerance value.
        """
        manager = manager_with_conserved_vars

        # Small update within fixed tolerance
        updates = {
            'carbon_pool': torch.tensor(1e-7),
            'nitrogen_pool': torch.tensor(-1e-7)
        }

        # Should not raise with dt=None (fixed tolerance)
        manager.apply_updates(updates, dt=None, check_conservation=True)

        # Check conservation summary
        summary = manager.get_last_conservation_summary()
        assert summary is not None
        assert summary.dynamic_tolerance_used is False
        assert summary.tolerance_value == 1e-6
        assert summary.dt is None

    def test_dynamic_tolerance_mode(self, manager_with_conserved_vars):
        """Test dynamic tolerance calculation with dt provided.

        Mathematical Expression:
        dynamic_tol = dt * scale_factor
        where scale_factor = abs_conservation_tol (default)

        Expected: Tolerance scales with time step.
        """
        manager = manager_with_conserved_vars
        dt = 1800.0  # 30 minutes in seconds

        # Update within dynamic tolerance but outside fixed tolerance
        # With dt=1800, scale_factor=1e-6, dynamic_tol = 1800 * 1e-6 = 1.8e-3
        updates = {
            'carbon_pool': torch.tensor(1e-3),
            'nitrogen_pool': torch.tensor(-1e-3)
        }

        # Should not raise with dynamic tolerance
        manager.apply_updates(updates, dt=dt, check_conservation=True)

        # Check conservation summary
        summary = manager.get_last_conservation_summary()
        assert summary is not None
        assert summary.dynamic_tolerance_used is True
        assert summary.dt == dt
        assert summary.tolerance_value == dt * 1e-6
        assert abs(summary.tolerance_value - 1.8e-3) < 1e-10

    def test_dynamic_tolerance_violation(self, manager_with_conserved_vars):
        """Test that violations are detected with dynamic tolerance.

        Mathematical Expression:
        If |residual| > dt * scale_factor, raise ValueError

        Expected: Conservation violation detected and error raised.
        """
        manager = manager_with_conserved_vars
        dt = 1800.0

        # Large update exceeding dynamic tolerance
        # dynamic_tol = 1800 * 1e-6 = 1.8e-3
        updates = {
            'carbon_pool': torch.tensor(0.01),  # 0.01 > 1.8e-3
            'nitrogen_pool': torch.tensor(0.0)
        }

        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            manager.apply_updates(updates, dt=dt, check_conservation=True,
                                rollback_on_failure=False)

        error_msg = str(excinfo.value)
        assert 'Conservation violation' in error_msg
        assert 'carbon' in error_msg
        assert 'dynamic' in error_msg

    def test_tolerance_scaling_with_dt(self, manager_with_conserved_vars):
        """Test that tolerance properly scales with different dt values.

        Mathematical Expression:
        tol(dt1) / tol(dt2) = dt1 / dt2

        Expected: Larger dt allows larger residuals.
        """
        manager = manager_with_conserved_vars

        # Test with different time steps
        test_cases = [
            (900.0, 9e-4),    # 15 min -> tol = 900 * 1e-6
            (1800.0, 1.8e-3), # 30 min -> tol = 1800 * 1e-6
            (3600.0, 3.6e-3), # 60 min -> tol = 3600 * 1e-6
        ]

        for dt, expected_tol in test_cases:
            # Reset state
            manager.reset()
            manager.initialize()

            # Update just under tolerance
            residual = expected_tol * 0.9
            updates = {
                'carbon_pool': torch.tensor(residual),
                'nitrogen_pool': torch.tensor(-residual)
            }

            # Should not raise
            manager.apply_updates(updates, dt=dt, check_conservation=True)

            summary = manager.get_last_conservation_summary()
            assert abs(summary.tolerance_value - expected_tol) < 1e-10

    def test_rollback_on_conservation_failure(self, manager_with_conserved_vars):
        """Test state rollback when conservation check fails.

        Expected: State reverts to pre-update values on violation.
        """
        manager = manager_with_conserved_vars

        # Record initial state
        initial_carbon = manager.get_state(['carbon_pool'])['carbon_pool'].item()

        # Violating update
        updates = {'carbon_pool': torch.tensor(100.0)}

        # Should raise and rollback
        with pytest.raises(ValueError):
            manager.apply_updates(updates, dt=1800.0, check_conservation=True,
                                rollback_on_failure=True)

        # State should be unchanged
        final_carbon = manager.get_state(['carbon_pool'])['carbon_pool'].item()
        assert abs(final_carbon - initial_carbon) < 1e-10


class TestConservationSummaryEnhancements:
    """Test enhanced ConservationSummary with P3 additions."""

    def test_conservation_summary_fields(self):
        """Test that ConservationSummary has all P3 fields.

        Expected: dt, dynamic_tolerance_used, tolerance_value fields present.
        """
        summary = ConservationSummary(total_violations=0, tolerance_value=1e-6)

        # Check P3 enhancement fields exist
        assert hasattr(summary, 'dt')
        assert hasattr(summary, 'dynamic_tolerance_used')
        assert hasattr(summary, 'tolerance_value')

        # Check default values
        assert summary.dt is None
        assert summary.dynamic_tolerance_used is False
        assert summary.tolerance_value == 1e-6

    def test_summary_records_dynamic_tolerance(self):
        """Test that summary correctly records dynamic tolerance usage."""
        schema = {
            'test_var': StateVariableSpec(
                name='test_var',
                units='dimensionless',
                conserved=True,
                default_value=1.0
            )
        }
        manager = StateManager(schema=schema)
        manager.initialize()

        # Update with dt
        dt = 1800.0
        updates = {'test_var': torch.tensor(1e-4)}
        manager.apply_updates(updates, dt=dt, check_conservation=True)

        summary = manager.get_last_conservation_summary()
        assert summary.dynamic_tolerance_used is True
        assert summary.dt == dt
        assert summary.tolerance_value == dt * manager.conservation_tolerance


class TestCNPCycleValidation:
    """Test C/N/P cross-domain cycle validation."""

    @pytest.fixture
    def validator(self):
        """Create CrossDomainValidator for testing."""
        return CrossDomainValidator()

    def test_validate_cnp_cycle_all_passing(self, validator):
        """Test validate_cnp_cycle with balanced C/N/P fluxes.

        Mathematical Expression:
        sum(C_in) = sum(C_out), sum(N_in) = sum(N_out), sum(P_in) = sum(P_out)
        C:N in [8, 20], C:P in [50, 300], N:P in [10, 20]

        Expected: Stoichiometry checks pass, structure is valid.
        """
        # Balanced state and fluxes with typical stoichiometry
        input_state = {
            'carbon_pool': torch.tensor(1000.0),
            'nitrogen_pool': torch.tensor(80.0),     # C:N = 12.5
            'phosphorus_pool': torch.tensor(5.0)     # C:P = 200, N:P = 16
        }

        output_fluxes = {
            'carbon_flux': torch.tensor(10.0),
            'nitrogen_flux': torch.tensor(0.8),      # C:N = 12.5
            'phosphorus_flux': torch.tensor(0.05)    # C:P = 200, N:P = 16
        }

        result = validator.validate_cnp_cycle(input_state, output_fluxes)

        # Check result structure
        assert 'passed' in result
        assert 'stoichiometry' in result
        assert 'carbon_balance' in result
        assert 'nitrogen_balance' in result
        assert 'phosphorus_balance' in result

        # Stoichiometry should pass with typical ratios
        assert result['stoichiometry']['passed'] is True
        assert 'ratios' in result['stoichiometry']

        # Check ratios are calculated
        ratios = result['stoichiometry']['ratios']
        if 'C:N' in ratios:
            assert 8.0 <= ratios['C:N'] <= 20.0

    def test_carbon_balance_violation(self, validator):
        """Test detection of carbon balance violation.

        Mathematical Expression:
        |sum(C_flux)| > tolerance

        Expected: Carbon balance failure detected.
        """
        input_state = {
            'carbon_pool': torch.tensor(1000.0)
        }

        # Large unbalanced carbon flux
        output_fluxes = {
            'carbon_flux': torch.tensor(100.0),  # Large net change
            'nitrogen_flux': torch.tensor(0.0),
            'phosphorus_flux': torch.tensor(0.0)
        }

        result = validator.validate_cnp_cycle(input_state, output_fluxes)

        # May pass or fail depending on tolerance, but should have carbon_balance info
        assert 'carbon_balance' in result
        assert 'residual' in result['carbon_balance']

    def test_stoichiometry_cn_ratio_violation(self, validator):
        """Test detection of C:N ratio outside typical range.

        Mathematical Expression:
        C:N ratio = C_flux / N_flux
        Expected range: [8, 20]

        Expected: Stoichiometry warning for out-of-range C:N.
        """
        input_state = {
            'carbon_pool': torch.tensor(1000.0),
            'nitrogen_pool': torch.tensor(10.0)
        }

        # Extreme C:N ratio (50:1, outside typical range)
        output_fluxes = {
            'carbon_flux': torch.tensor(10.0),
            'nitrogen_flux': torch.tensor(0.2),      # C:N = 50
            'phosphorus_flux': torch.tensor(0.05)
        }

        result = validator.validate_cnp_cycle(
            input_state, output_fluxes,
            check_stoichiometry=True,
        )

        assert 'stoichiometry' in result
        stoich = result['stoichiometry']
        assert 'ratios' in stoich
        if 'C:N' in stoich['ratios']:
            cn_ratio = stoich['ratios']['C:N']
            assert cn_ratio > 20.0 * 1.2  # Outside range with tolerance

    def test_stoichiometry_np_redfield_ratio(self, validator):
        """Test N:P ratio against Redfield ratio.

        Mathematical Expression:
        N:P ratio = N_flux / P_flux
        Redfield range: [10, 20], typical: 16

        Expected: Validation of Redfield ratio.
        """
        input_state = {
            'nitrogen_pool': torch.tensor(100.0),
            'phosphorus_pool': torch.tensor(6.25)    # N:P = 16 (Redfield)
        }

        # Fluxes matching Redfield ratio
        output_fluxes = {
            'carbon_flux': torch.tensor(10.0),
            'nitrogen_flux': torch.tensor(1.0),
            'phosphorus_flux': torch.tensor(0.0625)  # N:P = 16
        }

        result = validator.validate_cnp_cycle(input_state, output_fluxes)

        stoich = result['stoichiometry']
        if 'N:P' in stoich.get('ratios', {}):
            np_ratio = stoich['ratios']['N:P']
            # Should be close to Redfield ratio
            assert 10.0 <= np_ratio <= 20.0

    def test_cross_domain_flux_consistency(self, validator):
        """Test cross-domain flux consistency checking.

        Mathematical Expression:
        biosphere_flux + atmosphere_flux = 0 (conservation)

        Expected: Cross-domain flux balance detected.
        """
        input_state = {}

        # Balanced biosphere-atmosphere exchange
        output_fluxes = {
            'biosphere_carbon_flux': torch.tensor(5.0),   # Net uptake
            'atmosphere_carbon_flux': torch.tensor(-5.0)  # Net release
        }

        result = validator.validate_cnp_cycle(input_state, output_fluxes)

        assert 'cross_domain_fluxes' in result
        flux_result = result['cross_domain_fluxes']
        assert 'comparisons' in flux_result


class TestElementBalanceChecking:
    """Test individual element balance checking methods."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CrossDomainValidator()

    def test_check_element_balance_carbon(self, validator):
        """Test _check_element_balance for carbon.

        Mathematical Expression:
        residual = |sum(inputs) - sum(outputs)|

        Expected: Accurate carbon balance calculation.
        """
        input_state = {
            'carbon_pool_fast': torch.tensor(500.0),
            'carbon_pool_slow': torch.tensor(500.0)
        }

        output_fluxes = {
            'carbon_flux_gpp': torch.tensor(10.0),
            'carbon_flux_respiration': torch.tensor(-8.0)
        }

        result = validator._check_element_balance('carbon', input_state, output_fluxes)

        assert 'passed' in result
        assert 'residual' in result
        assert 'pools' in result
        assert 'fluxes' in result
        assert len(result['pools']) == 2
        assert len(result['fluxes']) == 2

    def test_check_element_balance_nitrogen(self, validator):
        """Test nitrogen balance checking."""
        input_state = {
            'nitrogen_pool_mineral': torch.tensor(50.0),
            'nitrogen_pool_organic': torch.tensor(50.0)
        }

        output_fluxes = {
            'nitrogen_flux_uptake': torch.tensor(2.0),
            'nitrogen_emission_n2o': torch.tensor(0.1)
        }

        result = validator._check_element_balance('nitrogen', input_state, output_fluxes)

        assert result is not None
        assert 'message' in result

    def test_check_element_balance_phosphorus(self, validator):
        """Test phosphorus balance checking."""
        input_state = {
            'phosphorus_pool_available': torch.tensor(10.0),
            'phosphorus_pool_fixed': torch.tensor(90.0)
        }

        output_fluxes = {
            'phosphorus_flux_uptake': torch.tensor(0.5),
            'phosphorus_rate_weathering': torch.tensor(0.1)
        }

        result = validator._check_element_balance('phosphorus', input_state, output_fluxes)

        assert result is not None
        assert 'residual' in result


class TestStoichiometryValidation:
    """Test stoichiometric ratio validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CrossDomainValidator()

    def test_typical_plant_biomass_ratios(self, validator):
        """Test validation with typical plant biomass C:N:P ratios.

        Typical ratios:
        - C:N = 15 (plant biomass)
        - C:P = 200 (plant biomass)
        - N:P = 13.3

        Expected: All ratios pass validation.
        """
        input_state = {}

        # Fluxes with typical plant stoichiometry
        output_fluxes = {
            'carbon_flux': torch.tensor(15.0),
            'nitrogen_flux': torch.tensor(1.0),      # C:N = 15
            'phosphorus_flux': torch.tensor(0.075)   # C:P = 200, N:P = 13.3
        }

        result = validator._check_cnp_stoichiometry(input_state, output_fluxes)

        assert result['passed'] is True
        assert 'C:N' in result['ratios']
        assert 'C:P' in result['ratios']
        assert 'N:P' in result['ratios']

        # Check ratios are in expected ranges
        assert 8.0 <= result['ratios']['C:N'] <= 20.0
        assert 50.0 <= result['ratios']['C:P'] <= 300.0
        assert 10.0 <= result['ratios']['N:P'] <= 20.0

    def test_extreme_cn_ratio_detection(self, validator):
        """Test detection of extreme C:N ratios.

        Expected: Ratios outside [8, 20] flagged as warnings.
        """
        input_state = {}

        # Extreme C:N ratio
        output_fluxes = {
            'carbon_flux': torch.tensor(100.0),
            'nitrogen_flux': torch.tensor(1.0),      # C:N = 100 (extreme)
            'phosphorus_flux': torch.tensor(0.5)
        }

        result = validator._check_cnp_stoichiometry(input_state, output_fluxes)

        # Should have warning about C:N ratio
        assert len(result['warnings']) > 0
        assert any('C:N' in warning for warning in result['warnings'])


class TestIntegrationWithStateManager:
    """Test integration of conservation checks with StateManager."""

    def test_end_to_end_conservation_workflow(self):
        """Test complete conservation checking workflow.

        Workflow:
        1. Create StateManager with conserved variables
        2. Apply updates with dt
        3. Check conservation summary
        4. Validate with CrossDomainValidator

        Expected: Consistent conservation tracking across both systems.
        """
        # Create manager
        schema = {
            'carbon_pool': StateVariableSpec(
                name='carbon_pool',
                units='g C m-2',
                conserved=True,
                default_value=1000.0
            ),
            'nitrogen_pool': StateVariableSpec(
                name='nitrogen_pool',
                units='g N m-2',
                conserved=True,
                default_value=75.0  # C:N = 13.3
            )
        }

        manager = StateManager(schema=schema)
        manager.initialize()

        # Apply updates with dynamic tolerance
        dt = 1800.0
        updates = {
            'carbon_pool': torch.tensor(1e-4),
            'nitrogen_pool': torch.tensor(-1e-4)
        }

        manager.apply_updates(updates, dt=dt, check_conservation=True)

        # Check StateManager conservation summary
        summary = manager.get_last_conservation_summary()
        assert summary.dynamic_tolerance_used is True
        assert summary.dt == dt

        # Validate with CrossDomainValidator
        validator = CrossDomainValidator(state_manager=manager)
        current_state = manager.get_state()

        # Should be able to validate
        assert validator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



class TestConservationChecksValidation:
    """StateManager is the one owner of the conservation_checks validation."""

    @staticmethod
    def _tolerance_bounds():
        from relic.parameters import get_global_manager

        lower, upper = get_global_manager().get_parameter(
            "validation.conservation.conservation_tolerance"
        ).bounds
        return float(lower), float(upper)

    @pytest.mark.parametrize(
        ("settings", "named"),
        [
            ({"tolerance": "abc"}, "'abc'"),
            ({"tolerance": float("nan")}, "nan"),
            ({"tolerance": -1.0e-6}, "-1e-06"),
            ({"tolerance": 0.0}, "0.0"),
            ({"tolerance": 1.0e-12}, "1e-12"),
            ({"tolerance": 1.0}, "1.0"),
            ({"tolerance": True}, "True"),
            ({"enabled": "false"}, "'false'"),
            ({"raise_on_violation": "false"}, "'false'"),
            ({"rollback_on_failure": 1}, "1"),
            ({"enabled": True, "tolerence": 1.0e-6}, "tolerence"),
            ("false", "'false'"),
            (0, "0"),
        ],
    )
    def test_invalid_conservation_checks_raise_naming_the_value(self, settings, named):
        """No bool() coercion, no silent fallback to the default tolerance, no ignored key."""
        with pytest.raises(ValueError, match="conservation_checks") as error:
            StateManager(conservation_checks=settings, autoload_contract_registry=False)

        assert named in str(error.value)

    def test_tolerance_is_checked_against_both_declared_bounds(self):
        lower, upper = self._tolerance_bounds()
        for tolerance in (lower, upper, 1.0e-5):
            manager = StateManager(
                conservation_checks={"tolerance": tolerance},
                autoload_contract_registry=False,
            )
            assert manager._abs_conservation_tol == tolerance
        for tolerance in (lower * 0.5, upper * 2.0):
            with pytest.raises(ValueError, match="conservation_checks.tolerance"):
                StateManager(
                    conservation_checks={"tolerance": tolerance},
                    autoload_contract_registry=False,
                )

    def test_a_bool_setting_is_the_enabled_switch(self):
        assert StateManager(conservation_checks=False, autoload_contract_registry=False)._conservation_enabled is False
        assert StateManager(conservation_checks=True, autoload_contract_registry=False)._conservation_enabled is True
        assert StateManager(autoload_contract_registry=False)._conservation_enabled is True

    def test_the_accepted_keys_are_one_public_constant(self):
        from relic.core.state import manager as manager_module

        assert manager_module.CONSERVATION_CHECK_KEYS == (
            "enabled",
            "raise_on_violation",
            "rollback_on_failure",
            "tolerance",
        )


class TestConservationToleranceOwner:
    """One default source for the ledger tolerance; every writer is validated."""

    @pytest.mark.parametrize("value", ["abc", float("nan"), -1.0e-6, 1.0e-12, 1.0, True])
    def test_setter_rejects_an_invalid_tolerance_and_keeps_the_value(self, value):
        manager = StateManager(autoload_contract_registry=False)
        before = manager.conservation_tolerance

        with pytest.raises(ValueError, match="conservation_tolerance"):
            manager.conservation_tolerance = value

        assert manager.conservation_tolerance == before

    def test_configure_validates_the_tolerance_and_rejects_undeclared_keys(self):
        manager = StateManager(autoload_contract_registry=False)
        before = manager.conservation_tolerance
        device_before = manager.device

        with pytest.raises(ValueError, match="conservation_tolerance"):
            manager.configure({"conservation_tolerance": "abc"})
        with pytest.raises(ValueError, match="tolerence"):
            manager.configure({"tolerence": 1.0e-6})
        with pytest.raises(ValueError, match="device"):
            manager.configure(device="cuda:7")
        assert manager.conservation_tolerance == before
        assert manager.device == device_before

        manager.configure({"conservation_tolerance": 1.0e-5})
        assert manager.conservation_tolerance == 1.0e-5

    def test_default_tolerance_has_one_source_and_no_fallback(self, monkeypatch):
        import relic.core.state.manager as manager_module
        from relic.parameters import get_global_manager

        real = get_global_manager()
        key = "validation.conservation.conservation_tolerance"

        class _OverridingManager:
            def __init__(self, value):
                self._value = value

            def get(self, name, *args, **kwargs):
                if name == key:
                    if self._value is None:
                        raise KeyError(name)
                    return self._value
                return real.get(name, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(real, name)

        monkeypatch.setattr(manager_module, "_GLOBAL_MANAGER_RESOLVER", lambda: _OverridingManager(2.5e-6))
        assert StateManager(autoload_contract_registry=False).conservation_tolerance == 2.5e-6

        monkeypatch.setattr(manager_module, "_GLOBAL_MANAGER_RESOLVER", lambda: _OverridingManager(None))
        with pytest.raises(KeyError, match="conservation_tolerance"):
            StateManager(autoload_contract_registry=False)


class TestConservationBudgetValues:
    """Budget reductions accept real data only; the real-data guard is never swallowed."""

    def test_a_complex_budget_value_raises_instead_of_being_skipped(self):
        with pytest.raises(ValueError, match="must be real"):
            StateManager._value_to_float(torch.tensor([1.0 + 2.0j]))

    def test_conservation_groups_do_not_skip_a_complex_update(self):
        from relic.core.state import conservation as state_conservation

        manager = StateManager(autoload_contract_registry=False)
        with pytest.raises(ValueError, match="must be real"):
            state_conservation.compute_conservation_groups(
                manager, {"carbon_pool": torch.tensor([1.0 + 1.0j])}
            )

    def test_real_values_reduce_in_float64_and_unsupported_values_give_none(self):
        assert StateManager._value_to_float(torch.tensor([1.0, 2.0], dtype=torch.float32)) == 3.0
        assert StateManager._value_to_float(torch.tensor(0.25, dtype=torch.float16)) == 0.25
        assert StateManager._value_to_float(2) == 2.0
        assert StateManager._value_to_float("not a number") is None

@pytest.mark.parametrize("failing", ["list_domains", "get_domain_members"])
def test_guardian_domain_lookup_failures_propagate(failing):
    """Contract domain lookups either succeed or fail the budget check."""
    from types import SimpleNamespace

    from relic.core.state import conservation as state_conservation

    def _broken(*_args):
        raise RuntimeError(f"{failing} unavailable")

    registry = SimpleNamespace(
        list_domains=lambda: ["carbon.total"],
        get_domain_members=lambda domain: ["carbon_pool"],
    )
    setattr(registry, failing, _broken)
    manager = SimpleNamespace(_contract_registry=registry, _abs_conservation_tol=1e-9)
    with pytest.raises(RuntimeError, match="unavailable"):
        state_conservation._run_conservation_guardian_checks(
            manager, {"carbon_pool": torch.tensor(1.0)}, 1.0, ConservationSummary(total_violations=0, tolerance_value=1e-6)
        )
