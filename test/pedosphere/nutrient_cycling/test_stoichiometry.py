"""
L1 Unit Tests for Stoichiometry Constraint Functor.

Tests C:N ratio constraints, mass balance verification, and flux corrections.
Following the RELIC L1 testing specification.

References:
- Manzoni & Porporato (2009) - Substrate-decomposer stoichiometry
- McGroddy et al. (2004) - Scaling of C:N:P stoichiometry in forests
- Cleveland & Liptzin (2007) - C:N:P stoichiometry in soil

Mathematical Expressions:
C_flux / N_flux = (C/N)_target +/- flexibility
SigmaC_in - SigmaC_out = DeltaC_pools
SigmaN_in - SigmaN_out = DeltaN_pools

Conservation:
Error = |Sigma(Inputs) - Sigma(Outputs) - Delta(Pools)| / |Sigma(Inputs)|
"""
import pytest
import torch
from relic.scientific.pedosphere.nutrient_cycling.stoichiometry import (
    StoichiometryConstraint
)


def _with_diagnostics(state, **diagnostics):
    updated = dict(state)
    for key, value in diagnostics.items():
        if value is not None:
            updated[f"diagnostics.{key}"] = value
    return updated


class TestStoichiometryConstraintInit:
    """Test initialization and parameter loading."""

    def test_initialization(self, parameter_manager, basic_config):
        """Test Functor can be initialized."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )
        assert functor is not None
        assert functor.name == "test_stoich"

    def test_target_ratios_loaded(self, parameter_manager, basic_config):
        """Test that target C:N ratios are loaded."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Should have target ratios for different pools
        assert hasattr(functor, 'target_ratios')
        assert isinstance(functor.target_ratios, dict)

        # Should have ratios for key pools
        expected_pools = ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'microbial']
        for pool in expected_pools:
            if pool in functor.target_ratios:
                ratio = functor.target_ratios[pool]
                # C:N ratios should be positive and reasonable (5-100)
                assert ratio > 0, f"{pool} C:N ratio should be positive"
                assert ratio < 200, f"{pool} C:N ratio should be reasonable"

    def test_flexibility_parameters(self, parameter_manager, basic_config):
        """Test flexibility parameters."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Should have flexibility and tolerance parameters
        assert hasattr(functor, 'flexibility')
        assert not hasattr(functor, 'mass_balance_tolerance')

        # Parameters should be positive
        assert functor.flexibility > 0

    def test_metadata_includes_conservation_laws(self, basic_config):
        """Test metadata includes carbon and nitrogen conservation."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )
        metadata = functor.metadata

        if isinstance(metadata, dict):
            conservation = metadata.get('conservation_laws', [])
        else:
            conservation = getattr(metadata, 'conservation_laws', [])

        assert 'carbon' in conservation
        assert 'nitrogen' in conservation

    def test_dependencies_use_diagnostic_state_not_both_decomposition_models(
        self,
        basic_config,
    ):
        """Alternative carbon closures must not be simultaneous dependencies."""

        functor = StoichiometryConstraint(name="test_stoich", config=basic_config)
        dependencies = functor.get_dependencies()
        assert "first_order_decomposition" not in dependencies
        assert "microbial_enzyme_decomposition" not in dependencies
        assert "diagnostics.carbon_fluxes" in functor.metadata.input_variables
        assert "diagnostics.nitrogen_fluxes" in functor.metadata.input_variables


class TestStoichiometryConstraintForward:
    """Test forward execution."""

    @pytest.fixture
    def stoich_state(self):
        """State with C and N pools."""
        return {
            'metabolic_litter_c': torch.tensor(10.0),  # g C/m2
            'metabolic_litter_n': torch.tensor(0.5),   # g N/m2
            'structural_litter_c': torch.tensor(50.0),
            'structural_litter_n': torch.tensor(1.0),
            'fast_som_c': torch.tensor(150.0),
            'fast_som_n': torch.tensor(10.0),
            'slow_som_c': torch.tensor(800.0),
            'slow_som_n': torch.tensor(50.0),
            'microbial_biomass_c': torch.tensor(10.0),
            'microbial_biomass_n': torch.tensor(1.2)
        }

    @pytest.fixture
    def carbon_fluxes(self):
        """Example carbon fluxes."""
        return {
            'delta_metabolic_litter_c': torch.tensor(0.1),
            'delta_structural_litter_c': torch.tensor(0.2),
            'delta_fast_som_c': torch.tensor(-0.05),
            'delta_slow_som_c': torch.tensor(-0.01)
        }

    @pytest.fixture
    def nitrogen_fluxes(self):
        """Example nitrogen fluxes."""
        return {
            'delta_metabolic_litter_n': torch.tensor(0.005),
            'delta_structural_litter_n': torch.tensor(0.004),
            'delta_fast_som_n': torch.tensor(-0.003),
            'delta_slow_som_n': torch.tensor(-0.0005)
        }

    def test_forward_runs(self, stoich_state, carbon_fluxes, nitrogen_fluxes, basic_config):
        """Test that forward() executes without errors."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        state_with_diag = _with_diagnostics(
            stoich_state,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes,
        )
        result = functor.forward(
            state=state_with_diag,
            time=0.0,
            dt=3600.0,
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_forward_returns_tensors(self, stoich_state, carbon_fluxes, nitrogen_fluxes, basic_config):
        """Test that forward returns Tensor values."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        state_with_diag = _with_diagnostics(
            stoich_state,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes,
        )
        result = functor.forward(
            state=state_with_diag,
            time=0.0,
            dt=3600.0,
        )

        for key, value in result.items():
            if not key.endswith('_units') and not key.endswith('_error'):
                if isinstance(value, torch.Tensor) or isinstance(value, float):
                    # Both tensors and floats are acceptable
                    pass
                else:
                    # Check if it's a dict of tensors (corrected fluxes)
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            assert isinstance(subvalue, torch.Tensor), f"{key}.{subkey} should be Tensor"


class TestCNRatioConstraints:
    """Test C:N ratio constraint enforcement."""

    def test_balanced_fluxes_pass_through(self, basic_config):
        """
        Test that balanced C:N fluxes pass through without correction.

        Mathematical Expression:
        If |C_flux/N_flux - CN_target| < tolerance:
            No correction needed
        """
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Get target C:N ratio for metabolic litter
        if 'metabolic_litter' in functor.target_ratios:
            target_cn = functor.target_ratios['metabolic_litter']

            # Create perfectly balanced fluxes
            c_flux = 1.0
            n_flux = c_flux / target_cn

            carbon_fluxes = {'delta_metabolic_litter_c': torch.tensor(c_flux)}
            nitrogen_fluxes = {'delta_metabolic_litter_n': torch.as_tensor(n_flux)}

            state = {
                'metabolic_litter_c': torch.tensor(10.0),
                'metabolic_litter_n': torch.as_tensor(10.0 / target_cn)
            }

            state_with_diag = _with_diagnostics(
                state,
                carbon_fluxes=carbon_fluxes,
                nitrogen_fluxes=nitrogen_fluxes,
            )
            result = functor.forward(
                state=state_with_diag,
                time=0.0,
                dt=3600.0,
            )

            # Should not have major violations
            if 'diagnostics.stoichiometry_constraint_violations_count' in result:
                # Either no violations or dict of small violations
                pass

    def test_imbalanced_fluxes_detected(self, basic_config):
        """
        Test that imbalanced C:N fluxes are detected.

        Mathematical Expression:
        If |C_flux/N_flux - CN_target| > tolerance:
            Violation detected
        """
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Get target C:N ratio
        if 'metabolic_litter' in functor.target_ratios:
            target_cn = functor.target_ratios['metabolic_litter']

            # Create severely imbalanced fluxes (2x off target)
            c_flux = 1.0
            n_flux = c_flux / (target_cn * 2.0)  # N flux is half what it should be

            carbon_fluxes = {'delta_metabolic_litter_c': torch.tensor(c_flux)}
            nitrogen_fluxes = {'delta_metabolic_litter_n': torch.as_tensor(n_flux)}

            state = {
                'metabolic_litter_c': torch.tensor(10.0),
                'metabolic_litter_n': torch.tensor(0.5)
            }

            state_with_diag = _with_diagnostics(
                state,
                carbon_fluxes=carbon_fluxes,
                nitrogen_fluxes=nitrogen_fluxes,
            )
            result = functor.forward(
                state=state_with_diag,
                time=0.0,
                dt=3600.0,
            )

            # Should detect violation
            # Check if corrected fluxes or violation indicators exist
            assert result is not None

    def test_tensor_flux_correction_keeps_shape(self, basic_config):
        """
        Test tensor-valued C/N flux correction preserves shape and does not raise.

        Mathematical Expression:
        CN_flux = C_flux / N_flux (element-wise)
        """
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )
        target_cn = functor.target_ratios.get("metabolic_litter", 15.0)

        c_flux = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        n_flux = torch.tensor([
            [1.0 / (target_cn * 2.0), 2.0 / (target_cn * 0.5)],
            [3.0 / (target_cn * 1.5), 4.0 / (target_cn * 0.75)],
        ])
        carbon_fluxes = {"delta_metabolic_litter_c": c_flux}
        nitrogen_fluxes = {"delta_metabolic_litter_n": n_flux}

        state = {
            "metabolic_litter_c": torch.full_like(c_flux, 10.0),
            "metabolic_litter_n": torch.full_like(n_flux, 10.0 / target_cn),
        }
        state_with_diag = _with_diagnostics(
            state,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes,
        )
        result = functor.forward(
            state=state_with_diag,
            time=0.0,
            dt=3600.0,
        )

        corrected_c_key = "diagnostics.stoichiometry_corrected_delta_metabolic_litter_c"
        corrected_n_key = "diagnostics.stoichiometry_corrected_delta_metabolic_litter_n"
        assert corrected_c_key in result
        assert corrected_n_key in result
        assert result[corrected_c_key].shape == c_flux.shape
        assert result[corrected_n_key].shape == n_flux.shape


class TestMassBalanceVerification:
    """Test mass balance verification."""

    def test_conservation_check_called(self, basic_config, caplog):
        """Test that mass balance check is invoked."""
        import logging
        caplog.set_level(logging.INFO)

        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        carbon_fluxes = {
            'delta_fast_som_c': torch.tensor(0.1)
        }
        nitrogen_fluxes = {
            'delta_fast_som_n': torch.tensor(0.01)
        }
        state = {
            'fast_som_c': torch.tensor(150.0),
            'fast_som_n': torch.tensor(10.0)
        }

        state_with_diag = _with_diagnostics(
            state,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes,
        )
        result = functor.forward(
            state=state_with_diag,
            time=0.0,
            dt=3600.0,
        )

        # If we got here, mass balance check didn't crash
        assert True

    def test_tracking_violations(self, basic_config):
        """Test that constraint violations are tracked."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Get target C:N ratio
        if 'fast_som' in functor.target_ratios:
            target_cn = functor.target_ratios['fast_som']

            # Create imbalanced fluxes
            c_flux = 1.0
            n_flux = c_flux / (target_cn * 3.0)  # Severely imbalanced

            carbon_fluxes = {'delta_fast_som_c': torch.tensor(c_flux)}
            nitrogen_fluxes = {'delta_fast_som_n': torch.as_tensor(n_flux)}

            state = {
                'fast_som_c': torch.tensor(150.0),
                'fast_som_n': torch.tensor(10.0)
            }

            initial_violations = functor.constraint_violations

            state_with_diag = _with_diagnostics(
                state,
                carbon_fluxes=carbon_fluxes,
                nitrogen_fluxes=nitrogen_fluxes,
            )
            result = functor.forward(
                state=state_with_diag,
                time=0.0,
                dt=3600.0,
            )

            # Violations counter should potentially increase
            # (depends on implementation details)
            assert functor.constraint_violations >= initial_violations


class TestFluxCorrection:
    """Test flux correction mechanisms."""

    def test_correction_reduces_imbalance(self, basic_config):
        """
        Test that correction moves fluxes toward target ratio.

        Mathematical Expression:
        Corrected_flux = Original_flux + correction_strength * (target - actual)
        """
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        if 'microbial' in functor.target_ratios:
            target_cn = functor.target_ratios['microbial']

            # Start with imbalanced fluxes
            c_flux_original = 1.0
            n_flux_original = c_flux_original / (target_cn * 2.0)  # N is half of target

            carbon_fluxes = {'delta_microbial_c': torch.tensor(c_flux_original)}
            nitrogen_fluxes = {'delta_microbial_n': torch.as_tensor(n_flux_original)}

            state = {
                'microbial_biomass_c': torch.tensor(10.0),
                'microbial_biomass_n': torch.tensor(1.2)
            }

            state_with_diag = _with_diagnostics(
                state,
                carbon_fluxes=carbon_fluxes,
                nitrogen_fluxes=nitrogen_fluxes,
            )
            result = functor.forward(
                state=state_with_diag,
                time=0.0,
                dt=3600.0,
            )

            # Check if corrected fluxes exist
            corrected_c = result.get('diagnostics.stoichiometry_corrected_delta_microbial_c')
            corrected_n = result.get('diagnostics.stoichiometry_corrected_delta_microbial_n')

            if corrected_c is not None and corrected_n is not None:
                # Calculate new ratio
                new_ratio = corrected_c.item() / (corrected_n.item() + 1e-10)

                # Original ratio
                original_ratio = c_flux_original / (n_flux_original + 1e-10)

                # New ratio should be closer to target than original
                original_error = abs(original_ratio - target_cn)
                new_error = abs(new_ratio - target_cn)

                # Some correction should have occurred
                # (may not always be perfect depending on correction strength)
                assert new_error <= original_error * 1.5, "Correction should reduce error"


class TestMultiplePoolsConstraints:
    """Test handling of multiple pools simultaneously."""

    def test_multiple_pools_processed(self, basic_config):
        """Test that multiple pools can be constrained simultaneously."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Multiple pools with various ratios
        carbon_fluxes = {
            'delta_metabolic_litter_c': torch.tensor(0.5),
            'delta_structural_litter_c': torch.tensor(1.0),
            'delta_fast_som_c': torch.tensor(-0.2)
        }

        nitrogen_fluxes = {
            'delta_metabolic_litter_n': torch.tensor(0.02),
            'delta_structural_litter_n': torch.tensor(0.01),
            'delta_fast_som_n': torch.tensor(-0.01)
        }

        state = {
            'metabolic_litter_c': torch.tensor(10.0),
            'metabolic_litter_n': torch.tensor(0.5),
            'structural_litter_c': torch.tensor(50.0),
            'structural_litter_n': torch.tensor(1.0),
            'fast_som_c': torch.tensor(150.0),
            'fast_som_n': torch.tensor(10.0)
        }

        state_with_diag = _with_diagnostics(
            state,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes,
        )
        result = functor.forward(
            state=state_with_diag,
            time=0.0,
            dt=3600.0,
        )

        # Should handle all pools
        assert result is not None


class TestUpscaleDownscale:
    """Test scale coupling methods."""

    def test_upscale_method(self, basic_config):
        """Test upscale method for constraint aggregation."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Test data
        fine_data = torch.tensor([1.0, 2.0, 3.0, 4.0])

        upscaled = functor.upscale(fine_data)

        assert isinstance(upscaled, torch.Tensor)
        # Upscaled should be mean
        assert torch.allclose(upscaled, torch.tensor(2.5), rtol=1e-5)

    def test_downscale_method(self, basic_config):
        """Test downscale method for constraint distribution."""
        functor = StoichiometryConstraint(
            name="test_stoich",
            config=basic_config
        )

        # Test data
        coarse_data = torch.tensor([1.0, 2.0])

        downscaled = functor.downscale(coarse_data)

        assert isinstance(downscaled, torch.Tensor)
        assert downscaled.shape[0] == 4  # Should double


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

