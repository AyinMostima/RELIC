"""
L2 Integration Test: Conservation Laws

Tests conservation of carbon, nitrogen, and mass across
multiple coupled pedosphere processes.

Mathematical Expression:
    Carbon Conservation: sum(C_in) - sum(C_out) - delta_C_pools = 0 +/- tolerance
    Nitrogen Conservation: sum(N_in) - sum(N_out) - delta_N_pools = 0 +/- tolerance
    Mass Conservation: delta_mass_total = 0 +/- tolerance

References:
    - Parton et al. (1987) - CENTURY model conservation
    - Li et al. (1992) - DNDC mass balance
    - Oleson et al. (2013) - CLM conservation checks
"""

import pytest
import torch

from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.stoichiometry import StoichiometryConstraint
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestCarbonConservation:
    """Test carbon conservation across decomposition and carbon pool updates."""

    @pytest.fixture
    def param_manager(self):
        """Get global parameter manager."""
        return get_global_manager()

    @pytest.fixture
    def decomposition_functor(self, param_manager):
        """Create decomposition functor."""
        config = {"device": "cpu"}
        return FirstOrderDecomposition("test_decomp", config)

    @pytest.fixture
    def initial_state(self):
        """Create initial state with carbon pools."""
        return {
            'metabolic_litter_c': torch.tensor(100.0),  # g C/m2
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
            'soil_temperature': torch.tensor(293.15),  # K
            'soil_moisture': torch.tensor(0.3),  # m3/m3
            'water_filled_pore_space': torch.tensor(0.5),  # dimensionless
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }

    def test_carbon_conservation_single_step(self, decomposition_functor, initial_state):
        """
        Test carbon conservation in single decomposition step.

        Verifies: sum(C_in) - sum(C_out) - delta_C_pools ~= 0
        """
        # Get initial total carbon
        initial_carbon = sum(
            initial_state[key].item()
            for key in initial_state
            if key.endswith('_c')
        )

        # Execute decomposition
        dt = 3600.0  # 1 hour
        
        # Forward signature: forward(state, time, dt, **kwargs)
        fluxes = decomposition_functor.forward(
            initial_state, time=0.0, dt=dt
        )

        # Calculate carbon outputs (respiration)
        # Note: first_order.py returns 'heterotrophic_respiration' not 'co2_respiration'
        carbon_respired = _unwrap_update(
            fluxes['heterotrophic_respiration']
        ) * dt

        # Calculate carbon pool changes
        delta_carbon_pools = sum(
            _unwrap_update(fluxes[key]).item()
            for key in initial_state
            if key.endswith('_c') and key in fluxes
        )

        # Calculate final total carbon
        final_carbon = initial_carbon + delta_carbon_pools

        # Conservation check: initial - respired = final
        carbon_balance_error = abs(initial_carbon - carbon_respired.item() - final_carbon)
        tolerance = initial_carbon * 1e-6  # 0.0001% relative error

        assert carbon_balance_error < tolerance, (
            f"Carbon conservation violated: error={carbon_balance_error:.2e} g C/m2, "
            f"tolerance={tolerance:.2e} g C/m2\n"
            f"Initial C={initial_carbon:.2f}, Respired={carbon_respired.item():.2f}, "
            f"Final C={final_carbon:.2f}"
        )

    def test_carbon_conservation_multi_step(self, decomposition_functor, initial_state):
        """
        Test carbon conservation over multiple time steps.

        Verifies cumulative carbon balance over 24 hours.
        """
        state = {k: v.clone() for k, v in initial_state.items()}
        dt = 3600.0  # 1 hour
        num_steps = 24  # 24 hours

        initial_carbon = sum(
            state[key].item() for key in state if key.endswith('_c')
        )

        total_respired = 0.0

        for step in range(num_steps):
            # Forward signature: forward(state, time, dt, **kwargs)
            fluxes = decomposition_functor.forward(
                state, time=step*dt, dt=dt
            )

            # Accumulate respiration
            total_respired += _unwrap_update(
                fluxes['heterotrophic_respiration']
            ).item() * dt

            # Update state
            for key in state:
                if key.endswith('_c') and key in fluxes:
                    state[key] = state[key] + _unwrap_update(fluxes[key])

        final_carbon = sum(
            state[key].item() for key in state if key.endswith('_c')
        )

        carbon_balance_error = abs(initial_carbon - total_respired - final_carbon)
        tolerance = initial_carbon * 1e-5  # 0.001% relative error for multi-step

        assert carbon_balance_error < tolerance, (
            f"Multi-step carbon conservation violated: error={carbon_balance_error:.2e} g C/m2\n"
            f"Initial={initial_carbon:.2f}, Respired={total_respired:.2f}, "
            f"Final={final_carbon:.2f}"
        )


class TestNitrogenConservation:
    """Test nitrogen conservation across nitrification and nitrogen pools."""

    @pytest.fixture
    def mineralization_functor(self):
        config = {"device": "cpu"}
        return MineralizationProcess("test_min", config)

    @pytest.fixture
    def nitrification_functor(self):
        config = {"device": "cpu"}
        return NitrificationProcess("test_nit", config)

    @pytest.fixture
    def initial_state(self):
        """
        Create initial state with nitrogen pools.

        ARCHITECTURE FIX: Uses new nitrogen pool structure:
        - Removed: mineral_nitrogen (merged into nh4_nitrogen)
        - Current: organic_nitrogen -> nh4_nitrogen -> no3_nitrogen
        Reference: relic/scientific/pedosphere/nutrient_cycling/mineralization.py L753-765
        """
        return {
            'organic_nitrogen': torch.tensor(50.0),  # g N/m2
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(2.0),  # g N/m2 (microbial biomass N)
            'nh4_nitrogen': torch.tensor(5.0),       # g N/m2 (ammonium pool)
            'no3_nitrogen': torch.tensor(10.0),      # g N/m2 (nitrate pool)
            'soil_temperature': torch.tensor(293.15),
            'soil_moisture': torch.tensor(0.3),
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.5),
        }

    def test_nitrogen_conservation_coupled_processes(
        self, mineralization_functor, nitrification_functor, initial_state
    ):
        """
        Test nitrogen conservation through coupled mineralization -> nitrification.

        Verifies: N_organic -> NH4 -> NO3 conserves total N

        ARCHITECTURE FIX: Updated to use new nitrogen pool structure:
        - Total N = organic_nitrogen + microbial_nitrogen + nh4_nitrogen + no3_nitrogen
        - No longer uses retired mineral_nitrogen pool
        """
        state = {k: v.clone() for k, v in initial_state.items()}
        state['carbon_decomposition_rate'] = torch.tensor(1.0e-6)
        dt = 3600.0

        initial_total_n = (
            state['organic_nitrogen'].item() +
            state['microbial_nitrogen'].item() +
            state['nh4_nitrogen'].item() +
            state['no3_nitrogen'].item()
        )

        # Step 1: Mineralization (organic N -> NH4)
        # Forward signature: forward(state, time, dt, **kwargs)
        min_fluxes = mineralization_functor.forward(state, 0.0, dt)

        # Update pools from mineralization (per-step deltas)
        if 'organic_nitrogen' in min_fluxes:
            state['organic_nitrogen'] = (
                state['organic_nitrogen']
                + _unwrap_update(min_fluxes['organic_nitrogen'])
            )
        if 'nh4_nitrogen' in min_fluxes:
            state['nh4_nitrogen'] = (
                state['nh4_nitrogen']
                + _unwrap_update(min_fluxes['nh4_nitrogen'])
            )
        if 'microbial_nitrogen' in min_fluxes:
            state['microbial_nitrogen'] = (
                state['microbial_nitrogen']
                + _unwrap_update(min_fluxes['microbial_nitrogen'])
            )

        # Step 2: Nitrification (NH4 -> NO3)
        # Forward signature: forward(state, time, dt, **kwargs)
        nit_fluxes = nitrification_functor.forward(state, dt, dt)

        # Update pools (nitrification deltas are per-step, do not multiply by dt)
        if 'nh4_nitrogen' in nit_fluxes:
            state['nh4_nitrogen'] = (
                state['nh4_nitrogen']
                + _unwrap_update(nit_fluxes['nh4_nitrogen'])
            )
        if 'no3_nitrogen' in nit_fluxes:
            state['no3_nitrogen'] = (
                state['no3_nitrogen']
                + _unwrap_update(nit_fluxes['no3_nitrogen'])
            )

        # Account for N2O emissions
        n2o_loss = _unwrap_update(nit_fluxes.get('n2o_flux', torch.tensor(0.0))).item() * dt

        # Calculate final total nitrogen (ARCHITECTURE FIX: includes all 4 pools)
        final_total_n = (
            state['organic_nitrogen'].item() +
            state['microbial_nitrogen'].item() +
            state['nh4_nitrogen'].item() +
            state['no3_nitrogen'].item()
        )

        # Conservation: initial = final + gaseous losses
        n_balance_error = abs(initial_total_n - final_total_n - n2o_loss)
        tolerance = initial_total_n * 1e-6

        assert n_balance_error < tolerance, (
            f"Nitrogen conservation violated: error={n_balance_error:.2e} g N/m2\n"
            f"Initial={initial_total_n:.2f}, Final={final_total_n:.2f}, "
            f"N2O loss={n2o_loss:.2f}"
        )


class TestStoichiometricConservation:
    """Test stoichiometric constraints maintain C:N ratios."""

    @pytest.fixture
    def stoichiometry_functor(self):
        config = {"device": "cpu"}
        return StoichiometryConstraint("test_stoich", config)

    def test_stoichiometric_constraint_enforcement(self, stoichiometry_functor):
        """
        Test stoichiometric correction maintains target C:N ratios.

        Verifies: Corrected fluxes satisfy C:N = target +/- tolerance
        """
        state = {
            'metabolic_litter_c': torch.tensor(100.0),
            'metabolic_litter_n': torch.tensor(10.0),  # C:N = 10
        }

        # Create imbalanced fluxes (C:N = 5, should be corrected to ~10)
        carbon_fluxes = {'delta_metabolic_litter_c': torch.tensor(10.0)}
        nitrogen_fluxes = {'delta_metabolic_litter_n': torch.tensor(2.0)}  # Too much N
        state['diagnostics.carbon_fluxes'] = carbon_fluxes
        state['diagnostics.nitrogen_fluxes'] = nitrogen_fluxes

        dt = 3600.0

        result = stoichiometry_functor.forward(state, 0.0, dt)

        # Check corrected C:N ratio
        corrected_c = result.get(
            'diagnostics.stoichiometry_corrected_delta_metabolic_litter_c',
            carbon_fluxes['delta_metabolic_litter_c'],
        )
        corrected_n = result.get(
            'diagnostics.stoichiometry_corrected_delta_metabolic_litter_n',
            nitrogen_fluxes['delta_metabolic_litter_n'],
        )

        corrected_cn_ratio = corrected_c / (corrected_n + 1e-10)
        target_cn = 10.0
        flexibility = 0.1  # 10% tolerance

        assert abs(corrected_cn_ratio.item() - target_cn) / target_cn < flexibility, (
            f"Stoichiometric constraint not enforced: "
            f"C:N={corrected_cn_ratio.item():.2f}, target={target_cn}"
        )


@pytest.mark.integration
class TestCombinedConservation:
    """Test combined carbon and nitrogen conservation in fully coupled system."""

    def test_full_cycle_conservation(self):
        """
        Test conservation laws in complete C-N cycle simulation.

        Process chain: Decomposition -> Mineralization -> Nitrification -> Stoichiometry
        """
        decomposition = FirstOrderDecomposition("cycle_decomp", {"device": "cpu"})
        mineralization = MineralizationProcess("cycle_min", {"device": "cpu"})
        nitrification = NitrificationProcess("cycle_nit", {"device": "cpu"})
        dt = 3600.0

        state = {
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
            'organic_nitrogen': torch.tensor(50.0),
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(2.0),
            'nh4_nitrogen': torch.tensor(5.0),
            'no3_nitrogen': torch.tensor(10.0),
            'soil_temperature': torch.tensor(293.15),
            'soil_moisture': torch.tensor(0.3),
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.5),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
        }

        carbon_outputs = decomposition.forward(state, 0.0, dt)
        carbon_pool_keys = (
            'metabolic_litter_c',
            'structural_litter_c',
            'fast_som_c',
            'slow_som_c',
            'passive_som_c',
        )
        carbon_residual = torch.stack(
            [_unwrap_update(carbon_outputs[key]) for key in carbon_pool_keys]
        ).sum()
        carbon_residual = carbon_residual + (
            _unwrap_update(carbon_outputs['heterotrophic_respiration']) * dt
        )
        assert torch.allclose(
            carbon_residual,
            torch.zeros_like(carbon_residual),
            atol=1e-9,
            rtol=0.0,
        )

        nitrogen_state = dict(state)
        nitrogen_state['carbon_decomposition_rate'] = _unwrap_update(
            carbon_outputs['carbon_decomposition_rate']
        )
        initial_n = sum(
            nitrogen_state[key]
            for key in (
                'organic_nitrogen',
                'microbial_nitrogen',
                'nh4_nitrogen',
                'no3_nitrogen',
            )
        )
        mineral_outputs = mineralization.forward(nitrogen_state, 0.0, dt)
        for key in ('organic_nitrogen', 'microbial_nitrogen', 'nh4_nitrogen'):
            nitrogen_state[key] = nitrogen_state[key] + _unwrap_update(
                mineral_outputs[key]
            )

        nitrification_outputs = nitrification.forward(nitrogen_state, dt, dt)
        for key in ('nh4_nitrogen', 'no3_nitrogen'):
            nitrogen_state[key] = nitrogen_state[key] + _unwrap_update(
                nitrification_outputs[key]
            )
        final_n = sum(
            nitrogen_state[key]
            for key in (
                'organic_nitrogen',
                'microbial_nitrogen',
                'nh4_nitrogen',
                'no3_nitrogen',
            )
        )
        n2o_export = _unwrap_update(nitrification_outputs['n2o_flux']) * dt
        nitrogen_residual = initial_n - final_n - n2o_export
        assert torch.allclose(
            nitrogen_residual,
            torch.zeros_like(nitrogen_residual),
            atol=1e-5,
            rtol=0.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


