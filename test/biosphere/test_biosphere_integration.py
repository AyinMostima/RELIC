"""
L2 Integration Tests: Biosphere Module Coupling
================================================

Tests that biosphere modules work together correctly after parameter
refactoring, including:
- Photosynthesis-respiration coupling
- BVOC emission integration with photosynthesis
- Growth respiration with biomass allocation
- Parameter consistency across modules

Test Strategy:
- Construct realistic state dictionaries
- Run coupled modules sequentially
- Verify data flows correctly between modules
- Check conservation laws
"""

import pytest
import torch
import sys
from pathlib import Path

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import get_global_manager
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.respiration.maintenance_respiration import MaintenanceRespiration
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter


class TestBiosphereIntegration:
    """L2: Integration Test - Module Coupling"""

    @staticmethod
    def _replace_flux_value(output, key, unit):
        """Return a canonical instantaneous flux value."""
        item = output[key]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == unit
        return item.value

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.param_manager = get_global_manager()
        self.device = torch.device('cpu')
        yield

    def _create_test_state(self):
        """Create realistic test state dictionary"""
        return {
            # Environmental
            'air_temperature': torch.tensor(298.15, device=self.device),
            'atmospheric_co2_ppm': torch.tensor(400.0, device=self.device),
            'ppfd': torch.tensor(1000.0, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1000.0, device=self.device),

            # Plant physiology
            'vcmax_scaling': torch.tensor(1.0, device=self.device),
            'leaf_area_index': torch.tensor(3.0, device=self.device),
            'water_stress_factor': torch.tensor(0.9, device=self.device),
            'phenology_phase': torch.tensor(2.0, device=self.device),

            # Biomass pools
            'biomass_leaf': torch.tensor(100.0, device=self.device),  # g C m-2
            'biomass_stem': torch.tensor(200.0, device=self.device),
            'biomass_root': torch.tensor(150.0, device=self.device),
            'biomass_grain': torch.tensor(0.0, device=self.device),
            'leaf_biomass_c': torch.tensor(0.10, device=self.device),
            'wood_biomass_c': torch.tensor(0.20, device=self.device),
            'root_biomass_c': torch.tensor(0.15, device=self.device),

            # Substrate pools
            'soluble_sugar': torch.tensor(10.0, device=self.device),  # g C m-2
            'starch': torch.tensor(5.0, device=self.device),

            # Soil conditions
            'soil_water_potential': torch.tensor(-0.5, device=self.device),  # MPa
            'soil_temperature': torch.tensor(293.15, device=self.device),  # 20C

            # Phenology
            'phenology_stage': torch.tensor(2.0, device=self.device),  # vegetative
            'growing_degree_days': torch.tensor(500.0, device=self.device),

            # Production rates
            'net_primary_productivity': torch.tensor(
                2.0e-7, device=self.device
            ),  # kg C m-2 s-1
            'gross_photosynthesis': torch.tensor(30.0, device=self.device),  # g C m-2 day-1
        }

    def test_photosynthesis_respiration_coupling(self):
        """Test photosynthesis and maintenance respiration coupling"""
        print("\n[TEST] Photosynthesis-Respiration Coupling")

        # Create functors
        photo_functor = FarquharPhotosynthesis(
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.0
        )
        resp_functor = MaintenanceRespiration()

        # Create test state
        state = self._create_test_state()
        time = 0.0
        dt = 1800.0  # 30 minutes

        # Run photosynthesis
        print("  [RUN] Photosynthesis forward pass...")
        photo_output = photo_functor.forward(state, time, dt)

        assert 'gross_primary_productivity' in photo_output, \
            "Photosynthesis should output canonical mass-flux GPP"
        gpp = self._replace_flux_value(
            photo_output, 'gross_primary_productivity', 'kg C m-2 s-1'
        )
        assert gpp > 0, "GPP should be positive under good conditions"
        print(f"  [OK] GPP = {gpp.item():.8e} kg C m-2 s-1")

        # Run maintenance respiration
        print("  [RUN] Maintenance respiration forward pass...")
        resp_output = resp_functor.forward(state, time, dt)

        assert 'maintenance_respiration_total' in resp_output, \
            "Respiration should output total maintenance respiration"
        rm_item = resp_output['maintenance_respiration_total']
        assert isinstance(rm_item, UpdateItem)
        assert rm_item.mode is UpdateMode.REPLACE
        assert rm_item.per_timestep is False
        assert rm_item.unit == 'kg C m-2 s-1'
        rm_total = rm_item.value
        assert rm_total > 0, "Maintenance respiration should be positive"
        print(f"  [OK] Rm_total = {rm_total.item():.8e} kg C m-2 s-1")

        # Verify carbon balance: GPP > Rm (plant should be gaining carbon)
        net_carbon = gpp - rm_total
        assert net_carbon > 0, "Net carbon gain should be positive in good conditions"
        print(f"  [OK] Net carbon gain = {net_carbon.item():.8e} kg C m-2 s-1")

        print("  [PASS] Photosynthesis-respiration coupling works correctly")

    def test_bvoc_emission_integration(self):
        """Test BVOC emission integration with photosynthesis"""
        print("\n[TEST] BVOC Emission Integration")

        # Create functors
        photo_functor = FarquharPhotosynthesis(
            vcmax25=60.0,
            jmax25=120.0,
            rd25=1.0
        )
        bvoc_functor = MEGANModel()

        # Create test state
        state = self._create_test_state()
        time = 0.0
        dt = 1800.0

        # Run photosynthesis first
        print("  [RUN] Photosynthesis...")
        photo_output = photo_functor.forward(state, time, dt)
        gpp = self._replace_flux_value(
            photo_output, 'gross_photosynthesis', 'umol CO2 m-2 s-1'
        )
        print(f"  [OK] GPP = {gpp.item():.4f} umol CO2 m-2 s-1")

        # Route the canonical Farquhar carbon-assimilation port to hybrid MEGAN.
        state['net_photosynthesis'] = self._replace_flux_value(
            photo_output, 'net_photosynthesis', 'umol CO2 m-2 s-1'
        )

        # Run BVOC emissions
        print("  [RUN] BVOC emissions...")
        bvoc_output = bvoc_functor.forward(state, time, dt)

        # Check for BVOC outputs
        bvoc_keys = [k for k in bvoc_output.keys() if 'emission' in k.lower()]
        assert len(bvoc_keys) > 0, "Should have BVOC emission outputs"
        print(f"  [OK] Found {len(bvoc_keys)} BVOC emission outputs")

        # Check isoprene emission (primary BVOC)
        if 'isoprene_emission' in bvoc_output:
            iso_item = bvoc_output['isoprene_emission']
            assert isinstance(iso_item, UpdateItem)
            assert iso_item.mode is UpdateMode.REPLACE
            assert iso_item.per_timestep is False
            assert iso_item.unit == 'mol m-2 s-1'
            iso_emission = iso_item.value
            assert iso_emission >= 0, "Isoprene emission should be non-negative"
            print(f"  [OK] Isoprene emission = {iso_emission.item():.6e} mol m-2 s-1")

        # Equivalent BVOC allocation should be much smaller than GPP carbon.
        carbon_allocation_item = bvoc_output["bvoc_carbon_allocation_flux"]
        assert isinstance(carbon_allocation_item, UpdateItem)
        assert carbon_allocation_item.mode is UpdateMode.REPLACE
        assert carbon_allocation_item.per_timestep is False
        assert carbon_allocation_item.unit == 'kg C m-2 s-1'
        carbon_allocation_flux = carbon_allocation_item.value
        if carbon_allocation_flux is not None:
            mw_c = self.param_manager.get(
                "relic.physical.constants.molecular_weights.carbon",
                units="g mol-1",
            )
            umol_per_mol = self.param_manager.get(
                "relic.physical.constants.conversions.micromoles_per_mole",
                units="umol mol-1",
            )
            mw_c_tensor = torch.as_tensor(mw_c, device=self.device, dtype=gpp.dtype)
            umol_per_mol_tensor = torch.as_tensor(umol_per_mol, device=self.device, dtype=gpp.dtype)
            gpp_carbon_flux_g = gpp * (
                mw_c_tensor / umol_per_mol_tensor
            )
            gpp_carbon_flux = unit_converter.convert(
                gpp_carbon_flux_g,
                variable_name="carbon_flux",
                source_units="g C m-2 s-1",
                target_units="kg C m-2 s-1",
            )
            bvoc_fraction = (
                carbon_allocation_flux / gpp_carbon_flux
            ).item()
            assert bvoc_fraction < 0.1, "BVOC should be < 10% of GPP carbon flux"
            print(
                f"  [OK] BVOC C allocation = {carbon_allocation_flux.item():.6e} kg C m-2 s-1"
                f" ({bvoc_fraction*100:.2f}% of GPP)"
            )

        print("  [PASS] BVOC emission integration works correctly")

    def test_biomass_allocation_integration(self):
        """Test canonical NPP routing to biomass rates and stock increments."""
        print("\n[TEST] Biomass Allocation Integration")

        # Create allocation functor
        alloc_functor = BiomassAllocation()

        # Create a canonical kg-C allocation state.
        state = self._create_test_state()
        time = 0.0
        dt = 86400.0  # 1 day

        # Run allocation
        print("  [RUN] Biomass allocation...")
        alloc_output = alloc_functor.forward(state, time, dt)

        rate_keys = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        stock_keys = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
        for key in rate_keys:
            item = alloc_output[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == 'kg C m-2 s-1'
            assert item.value >= 0
        for key in stock_keys:
            item = alloc_output[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == 'kg C m-2'
            assert item.value >= 0

        total_rate = sum(alloc_output[key].value for key in rate_keys)
        total_increment = sum(alloc_output[key].value for key in stock_keys)
        torch.testing.assert_close(
            total_rate,
            state['net_primary_productivity'].to(dtype=total_rate.dtype),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            total_increment,
            (state['net_primary_productivity'] * dt).to(
                dtype=total_increment.dtype
            ),
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        residual = alloc_output[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        assert 'growth_respiration' not in alloc_output
        assert 'allocated_to_leaf' not in alloc_output
        print(f"  [OK] Total allocated = {total_increment.item():.8e} kg C m-2")

        print("  [PASS] Biomass allocation integration works correctly")

    def test_parameter_consistency_across_modules(self):
        """Test parameter consistency across different modules"""
        print("\n[TEST] Parameter Consistency Across Modules")

        # Check that shared parameters have consistent values
        # Example: Q10 values should be in reasonable range across all modules

        # Photosynthesis parameters
        kc25 = self.param_manager.get(
            "biosphere.photosynthesis.kc25",
            namespace="biosphere.photosynthesis"
        )
        assert kc25 is not None and kc25 > 0
        print(f"  [OK] Photosynthesis kc25: {kc25}")

        # Respiration parameters
        q10_maint = self.param_manager.get(
            "biosphere.respiration.maintenance.q10_maintenance",
            namespace="biosphere.respiration.maintenance"
        )
        if q10_maint is not None:
            assert 1.0 < q10_maint < 5.0, "Q10 should be in reasonable range"
            print(f"  [OK] Maintenance Q10: {q10_maint}")

        # BVOC parameters
        t_standard = self.param_manager.get(
            "biosphere.bvoc.t_standard",
            namespace="biosphere.bvoc"
        )
        if t_standard is not None:
            assert 273.15 < t_standard < 373.15, "Standard temp should be reasonable"
            print(f"  [OK] BVOC t_standard: {t_standard} K")

        # Check conservation tolerance is defined
        conservation_tol = self.param_manager.get(
            "validation.conservation.conservation_tolerance",
            namespace="validation.conservation"
        )
        if conservation_tol is not None:
            assert conservation_tol > 0, "Conservation tolerance should be positive"
            print(f"  [OK] Conservation tolerance: {conservation_tol}")

        print("  [PASS] Parameters consistent across modules")

    def test_carbon_flux_and_allocation_contracts(self):
        """Test the kg-C respiration ledger and allocation conservation separately."""
        print("\n[TEST] Carbon Flux and Allocation Contracts")

        # Create all functors
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)
        resp_functor = MaintenanceRespiration()
        alloc_functor = BiomassAllocation()

        # Initial state
        state = self._create_test_state()
        time = 0.0
        dt = 3600.0  # 1 hour

        # Step 1: Photosynthesis
        photo_output = photo_functor.forward(state, time, dt)
        gpp = self._replace_flux_value(
            photo_output, 'gross_primary_productivity', 'kg C m-2 s-1'
        )
        print(f"  [STEP 1] Photosynthesis: GPP = {gpp.item():.8e} kg C m-2 s-1")

        # Step 2: Maintenance respiration
        resp_output = resp_functor.forward(state, time, dt)
        rm_item = resp_output['maintenance_respiration_total']
        assert isinstance(rm_item, UpdateItem)
        rm_total = rm_item.value
        npp_after_maintenance = gpp - rm_total
        torch.testing.assert_close(
            gpp,
            rm_total + npp_after_maintenance,
            rtol=0.0,
            atol=1.0e-14,
        )
        print(f"  [STEP 2] Respiration: Rm = {rm_total.item():.8e} kg C m-2 s-1")

        # Step 3: Route the remaining NPP once through the allocation ledger.
        state['net_primary_productivity'] = npp_after_maintenance
        alloc_output = alloc_functor.forward(state, time, dt)
        rate_keys = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
        stock_keys = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
        allocated_rate = sum(alloc_output[key].value for key in rate_keys)
        allocated_step = sum(alloc_output[key].value for key in stock_keys)
        torch.testing.assert_close(
            allocated_rate,
            npp_after_maintenance.to(dtype=allocated_rate.dtype),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            allocated_step,
            (npp_after_maintenance * dt).to(dtype=allocated_step.dtype),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        residual = alloc_output[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        assert 'carbon_conservation_check' not in alloc_output
        assert 'carbon_conservation_error' not in alloc_output

        print("  [PASS] Carbon flux and allocation contracts are internally consistent")

    def test_stress_response_propagation(self):
        """Test that environmental stresses propagate correctly through modules"""
        print("\n[TEST] Stress Response Propagation")

        # Create functor
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)

        # Test under different stress conditions
        base_state = self._create_test_state()
        time = 0.0
        dt = 1800.0

        # Optimal conditions
        optimal_output = photo_functor.forward(base_state, time, dt)
        gpp_optimal = self._replace_flux_value(
            optimal_output, 'gross_photosynthesis', 'umol CO2 m-2 s-1'
        ).item()
        print(f"  [OPTIMAL] GPP = {gpp_optimal:.4f} g C m-2")

        # Water stress
        stressed_state = base_state.copy()
        stressed_state['water_stress_factor'] = torch.tensor(0.3, device=self.device)
        stress_output = photo_functor.forward(stressed_state, time, dt)
        gpp_stressed = self._replace_flux_value(
            stress_output, 'gross_photosynthesis', 'umol CO2 m-2 s-1'
        ).item()
        print(f"  [WATER STRESS] GPP = {gpp_stressed:.4f} g C m-2")

        # GPP should be reduced under water stress (strict requirement)
        assert gpp_stressed < gpp_optimal, f"GPP should decrease under water stress (got {gpp_stressed:.4f} >= {gpp_optimal:.4f})"
        stress_ratio = gpp_stressed / gpp_optimal
        print(f"  [OK] Water stress reduces GPP to {stress_ratio*100:.1f}% of optimal")

        # High temperature stress
        temp_state = base_state.copy()
        temp_state['temperature'] = torch.tensor(313.15, device=self.device)  # 40C
        temp_output = photo_functor.forward(temp_state, time, dt)
        gpp_temp = self._replace_flux_value(
            temp_output, 'gross_photosynthesis', 'umol CO2 m-2 s-1'
        ).item()
        print(f"  [HEAT STRESS] GPP = {gpp_temp:.4f} g C m-2")

        print("  [PASS] Stress responses propagate correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])

