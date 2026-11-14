"""
L3 System Test: Long-Term Pedosphere Simulation

Tests end-to-end simulation of coupled pedosphere processes over
extended time periods (days to weeks) to verify:
- Long-term conservation laws
- Numerical stability
- Physical realism of outputs
- Consistent process coupling

Mathematical Expression:
    Multi-day simulation: dt in [0, T], state(t+dt) = integrate(fluxes, state(t), dt)
    Conservation: |total_element(T) - total_element(0) - losses| < epsilon
    Stability: for all state variables, min_bound < var(t) < max_bound

References:
    - Parton et al. (1987) - CENTURY long-term C dynamics
    - Li et al. (1992) - DNDC multi-year simulations
    - Oleson et al. (2013) - CLM long-term benchmarks
"""

import pytest
import torch
import numpy as np
from relic.core.state import UpdateItem, UpdateMode

from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value, value.mode
    return value, UpdateMode.ACCUMULATE


CARBON_POOL_KEYS = (
    'metabolic_litter_c',
    'structural_litter_c',
    'fast_som_c',
    'slow_som_c',
    'passive_som_c',
)

NITROGEN_POOL_KEYS = (
    'organic_nitrogen',
    'microbial_nitrogen',
    'nh4_nitrogen',
    'no3_nitrogen',
)


def _apply_state_updates(state, outputs, keys):
    for key in keys:
        if key not in outputs:
            continue
        item = outputs[key]
        assert isinstance(item, UpdateItem), f"{key} must use explicit UpdateItem"
        value, mode = _unwrap_update(item)
        assert torch.isfinite(value).all(), f"Non-finite update for {key}"
        if mode == UpdateMode.REPLACE:
            state[key] = value
        else:
            state[key] = state[key] + value


class TestLongTermSimulation:
    """Test long-term (multi-day) pedosphere simulations."""

    @pytest.fixture
    def functors(self):
        """Create coupled pedosphere functors."""
        config = {"device": "cpu"}
        return {
            'decomp': FirstOrderDecomposition("decomp", config),
            'mineral': MineralizationProcess("mineral", config),
            'nitrif': NitrificationProcess("nitrif", config),
            'denitrif': DenitrificationProcess("denitrif", config),
        }

    @pytest.fixture
    def initial_state(self):
        """Create realistic initial state for long-term simulation."""
        return {
            # Carbon pools (g C/m2) - typical temperate forest values
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(2000.0),
            'passive_som_c': torch.tensor(5000.0),
            # Nitrogen pools (g N/m2)
            # ARCHITECTURE FIX: mineral_nitrogen merged into nh4_nitrogen (5+2=7)
            'organic_nitrogen': torch.tensor(150.0),
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(10.0),
            'nh4_nitrogen': torch.tensor(7.0),  # Merged from mineral_nitrogen
            'no3_nitrogen': torch.tensor(3.0),
            # Environmental drivers (typical spring conditions)
            'soil_temperature': torch.tensor(288.15),  # 15 C
            'soil_moisture': torch.tensor(0.35),  # 35% VWC
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.0),
            'oxygen_concentration': torch.tensor(0.21),
            # Labile carbon forcing of denitrification (g C/m2); a required input.
            'labile_carbon': torch.tensor(0.1),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }

    def test_seven_day_simulation(self, functors, initial_state):
        """
        Test 7-day simulation with hourly time steps.

        Verifies:
        - Numerical stability (no NaN, no explosion)
        - Physical realism (positive pools, reasonable ranges)
        - Long-term carbon conservation
        - Long-term nitrogen conservation
        """
        state = {k: v.clone() for k, v in initial_state.items()}
        dt = 3600.0  # 1 hour time step
        num_steps = 7 * 24  # 7 days * 24 hours

        # Track initial totals
        initial_carbon = sum(state[k].item() for k in state if k.endswith('_c'))
        # ARCHITECTURE FIX: Remove mineral_nitrogen (merged into nh4_nitrogen)
        initial_nitrogen = sum(
            state[k].item() for k in state
            if k in ['organic_nitrogen', 'microbial_nitrogen',
                     'nh4_nitrogen', 'no3_nitrogen']
        )

        # Track outputs (all nitrogen loss pathways)
        total_co2_respired = 0.0
        total_n2o_emitted = 0.0  # From both nitrification and denitrification
        total_n2_emitted = 0.0   # From denitrification only
        # Storage for diagnostics
        carbon_history = []
        nitrogen_history = []

        # Run simulation
        for step in range(num_steps):
            time = step * dt

            # Execute the canonical process chain and propagate every state update.
            decomp_fluxes = functors['decomp'].forward(state, time, dt)
            _apply_state_updates(state, decomp_fluxes, CARBON_POOL_KEYS)
            decomp_rate, _ = _unwrap_update(
                decomp_fluxes['carbon_decomposition_rate']
            )
            state['carbon_decomposition_rate'] = decomp_rate

            mineral_fluxes = functors['mineral'].forward(state, time, dt)
            _apply_state_updates(state, mineral_fluxes, NITROGEN_POOL_KEYS)

            nitrif_fluxes = functors['nitrif'].forward(state, time, dt)
            _apply_state_updates(state, nitrif_fluxes, NITROGEN_POOL_KEYS)

            denitrif_fluxes = functors['denitrif'].forward(state, time, dt)
            _apply_state_updates(state, denitrif_fluxes, NITROGEN_POOL_KEYS)

            # Accumulate gaseous losses
            co2_flux, _ = _unwrap_update(decomp_fluxes['heterotrophic_respiration'])
            total_co2_respired += co2_flux.item() * dt

            # Accumulate all nitrogen gas emissions
            # N2O from both nitrification and denitrification
            n2o_flux_value, _ = _unwrap_update(nitrif_fluxes.get('n2o_flux', torch.tensor(0.0)))
            total_n2o_emitted += n2o_flux_value.item() * dt
            denitrif_n2o, _ = _unwrap_update(
                denitrif_fluxes.get('n2o_flux', torch.tensor(0.0))
            )
            total_n2o_emitted += denitrif_n2o.item() * dt
            # N2 from denitrification only
            n2_flux, _ = _unwrap_update(
                denitrif_fluxes.get('n2_production', torch.tensor(0.0))
            )
            total_n2_emitted += n2_flux.item() * dt

            # Record state every 24 hours
            if step % 24 == 0:
                current_carbon = sum(state[k].item() for k in state if k.endswith('_c'))
                current_nitrogen = sum(
                    state[k].item() for k in state
                    if k in ['organic_nitrogen', 'microbial_nitrogen',
                             'nh4_nitrogen', 'no3_nitrogen']
                )
                carbon_history.append(current_carbon)
                nitrogen_history.append(current_nitrogen)

        # Final state
        final_carbon = sum(state[k].item() for k in state if k.endswith('_c'))
        final_nitrogen = sum(
            state[k].item() for k in state
            if k in ['organic_nitrogen', 'microbial_nitrogen',
                     'nh4_nitrogen', 'no3_nitrogen']
        )

        # Verify numerical stability
        assert not torch.isnan(state['metabolic_litter_c']), "Simulation produced NaN values"
        assert not torch.isinf(state['metabolic_litter_c']), "Simulation produced Inf values"

        # Verify physical realism - all pools remain positive
        for key in state:
            if key.endswith('_c') or key.endswith('_nitrogen'):
                assert state[key] >= 0, f"{key} became negative: {state[key]}"
                assert state[key] < 1e6, f"{key} exploded to unrealistic value: {state[key]}"

        # Verify carbon conservation
        carbon_balance_error = abs(
            initial_carbon - total_co2_respired - final_carbon
        )
        carbon_tolerance = initial_carbon * 1e-5

        assert carbon_balance_error < carbon_tolerance, (
            f"Carbon conservation violated over 7 days:\n"
            f"  Initial: {initial_carbon:.2f} g C/m2\n"
            f"  Respired: {total_co2_respired:.2f} g C/m2\n"
            f"  Final: {final_carbon:.2f} g C/m2\n"
            f"  Error: {carbon_balance_error:.2f} g C/m2 (tolerance: {carbon_tolerance:.2f})"
        )

        # Verify nitrogen conservation
        # FIXED: Microbial nitrogen turnover now correctly returns N to organic pool
        # Reference: Manzoni & Porporato (2009), mineralization.py lines 730-745
        total_nitrogen_losses = total_n2o_emitted + total_n2_emitted
        nitrogen_balance_error = abs(
            initial_nitrogen - total_nitrogen_losses - final_nitrogen
        )
        nitrogen_tolerance = initial_nitrogen * 1e-5

        assert nitrogen_balance_error < nitrogen_tolerance, (
            f"Nitrogen conservation violated over 7 days:\n"
            f"  Initial: {initial_nitrogen:.2f} g N/m2\n"
            f"  Gaseous losses: {total_nitrogen_losses:.2f} g N/m2\n"
            f"    - N2O emitted (nitrif + denitrif): {total_n2o_emitted:.2f} g N/m2\n"
            f"    - N2 emitted (denitrif): {total_n2_emitted:.2f} g N/m2\n"
            f"  Final: {final_nitrogen:.2f} g N/m2\n"
            f"  Error: {nitrogen_balance_error:.2f} g N/m2 (tolerance: {nitrogen_tolerance:.2f})"
        )

        # Verify monotonic trends (carbon should decrease over time)
        if len(carbon_history) >= 2:
            # Simple check: last carbon < first carbon
            assert carbon_history[-1] <= carbon_history[0], (
                f"Carbon should decrease over time: "
                f"initial={carbon_history[0]:.2f}, final={carbon_history[-1]:.2f}"
            )

    def test_thirty_day_coarse_timestep(self, functors, initial_state):
        """
        Test 30-day simulation with daily time steps.

        Tests coarser time stepping for computational efficiency while
        maintaining conservation and stability.
        """
        state = {k: v.clone() for k, v in initial_state.items()}
        dt = 86400.0  # 1 day time step
        num_steps = 30  # 30 days

        # Track totals
        initial_carbon = sum(state[k].item() for k in state if k.endswith('_c'))
        total_co2_respired = 0.0

        # Run simulation
        for step in range(num_steps):
            time = step * dt

            # Execute decomposition (primary carbon process)
            decomp_fluxes = functors['decomp'].forward(state, time, dt)
            respiration, _ = _unwrap_update(
                decomp_fluxes['heterotrophic_respiration']
            )
            total_co2_respired += respiration.item() * dt

            # Update carbon pools
            _apply_state_updates(state, decomp_fluxes, CARBON_POOL_KEYS)

        # Final carbon
        final_carbon = sum(state[k].item() for k in state if k.endswith('_c'))

        # Verify stability with coarse time steps
        assert not torch.isnan(state['metabolic_litter_c']), "Coarse time step produced NaN"
        assert all(state[k] >= 0 for k in state if k.endswith('_c')), "Negative carbon pools"

        # Verify conservation
        carbon_balance_error = abs(initial_carbon - total_co2_respired - final_carbon)
        carbon_tolerance = initial_carbon * 1e-5

        assert carbon_balance_error < carbon_tolerance, (
            f"Carbon conservation violated with daily timestep:\n"
            f"  Error: {carbon_balance_error:.2f} g C/m2 (tolerance: {carbon_tolerance:.2f})"
        )

    def test_variable_environmental_conditions(self, functors, initial_state):
        """
        Test simulation under varying environmental conditions.

        Simulates a realistic diurnal/seasonal cycle to test:
        - Response to temperature variation
        - Response to moisture variation
        - Stability under dynamic forcing
        """
        state = {k: v.clone() for k, v in initial_state.items()}
        dt = 3600.0  # 1 hour
        num_steps = 3 * 24  # 3 days

        # Variable environment generator
        def get_environment(hour):
            """Generate realistic diurnal cycle."""
            # Temperature: 10 C at night, 20 C at day
            temp_base = 283.15  # 10 C
            temp_amplitude = 10.0
            temperature = temp_base + temp_amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * hour / 24))

            # Moisture: slight daily variation
            moisture = 0.35 + 0.05 * np.sin(2 * np.pi * hour / 24)

            return {
                'soil_temperature': torch.tensor(temperature),
                'soil_moisture': torch.tensor(moisture),
            }

        # Run with variable environment
        for step in range(num_steps):
            hour = (step % 24)
            env = get_environment(hour)

            # Update environmental state
            state.update(env)

            # Execute decomposition
            decomp_fluxes = functors['decomp'].forward(state, step * dt, dt)

            # Update pools
            _apply_state_updates(state, decomp_fluxes, CARBON_POOL_KEYS)

        # Verify stability under variable conditions
        assert not torch.isnan(state['metabolic_litter_c']), "Variable conditions caused NaN"
        assert all(state[k] >= 0 for k in state if k.endswith('_c')), "Negative pools under variable conditions"
        assert all(state[k] < 1e5 for k in state if k.endswith('_c')), "Unrealistic pool values"


@pytest.mark.slow
class TestExtendedSimulation:
    """Extended simulations (>1 week) marked as slow tests."""

    def test_ninety_day_seasonal_cycle(self):
        """
        Test full seasonal simulation (90 days).

        This is a slow test that verifies long-term behavior over a full season.
        """
        decomposition = FirstOrderDecomposition("seasonal_decomp", {"device": "cpu"})
        state = {
            'metabolic_litter_c': torch.tensor(100.0, dtype=torch.float64),
            'structural_litter_c': torch.tensor(200.0, dtype=torch.float64),
            'fast_som_c': torch.tensor(500.0, dtype=torch.float64),
            'slow_som_c': torch.tensor(2000.0, dtype=torch.float64),
            'passive_som_c': torch.tensor(5000.0, dtype=torch.float64),
            'soil_temperature': torch.tensor(288.15, dtype=torch.float64),
            'soil_moisture': torch.tensor(0.35, dtype=torch.float64),
            'water_filled_pore_space': torch.tensor(0.5, dtype=torch.float64),
            'lignin_n_ratio': torch.tensor(0.3, dtype=torch.float64),
            'clay_content': torch.tensor(0.25, dtype=torch.float64),
            'metabolic_litter_input': torch.tensor(0.0, dtype=torch.float64),
            'structural_litter_input': torch.tensor(0.0, dtype=torch.float64),
        }
        initial_carbon = sum(state[key].item() for key in CARBON_POOL_KEYS)
        total_respired = 0.0
        cumulative_step_residual = 0.0
        dt = 6.0 * 3600.0
        num_steps = 90 * 4

        for step in range(num_steps):
            day = step / 4.0
            state['soil_temperature'] = torch.tensor(
                288.15 + 7.0 * np.sin(2.0 * np.pi * day / 90.0),
                dtype=torch.float64,
            )
            state['soil_moisture'] = torch.tensor(
                0.35 + 0.05 * np.sin(2.0 * np.pi * day / 30.0),
                dtype=torch.float64,
            )
            outputs = decomposition.forward(state, step * dt, dt)
            respiration, _ = _unwrap_update(outputs['heterotrophic_respiration'])
            step_residual = torch.stack(
                [_unwrap_update(outputs[key])[0] for key in CARBON_POOL_KEYS]
            ).sum() + respiration * dt
            cumulative_step_residual += abs(step_residual.item())
            total_respired += respiration.item() * dt
            _apply_state_updates(state, outputs, CARBON_POOL_KEYS)

        final_carbon = sum(state[key].item() for key in CARBON_POOL_KEYS)
        residual = initial_carbon - total_respired - final_carbon
        assert cumulative_step_residual <= initial_carbon * 1e-8
        assert abs(residual) <= cumulative_step_residual + initial_carbon * 1e-12
        assert all(torch.isfinite(state[key]) for key in CARBON_POOL_KEYS)
        assert all(state[key] >= 0 for key in CARBON_POOL_KEYS)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

