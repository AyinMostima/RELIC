"""
L2 Integration tests for PFT module integration with other biosphere components.

This test file verifies:
1. PFT integration with photosynthesis module
2. PFT integration with respiration module
3. PFT integration with BVOC emission module
4. PFT integration with phenology module
5. PFT integration with growth/allocation module
6. Cross-module parameter consistency
"""

import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.biosphere.pft.pft_registry import get_pft_registry
from relic.scientific.biosphere.pft.pft_types import PFTDefinition
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.respiration.maintenance_respiration import MaintenanceRespiration
from relic.scientific.biosphere.bvoc.megan_model import (
    CouplingStrategy,
    EmissionPathway,
    MEGANModel,
)
from relic.scientific.biosphere.phenology.models import GSIPhenology
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation


class TestPFTIntegration:
    """L2 Integration tests for PFT module."""

    def setup_method(self):
        """Set up test environment."""
        self.param_manager = get_global_manager()
        self.registry = get_pft_registry()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Test PFT types
        self.pft_types = [
            'broadleaf_deciduous_temperate',
            'needleleaf_evergreen_temperate',
            'c3_grass_temperate'
        ]

        # Standard test environment conditions
        self.test_environment = {
            'air_temperature': torch.tensor(298.15, device=self.device),
            'atmospheric_co2_ppm': torch.tensor(400.0, device=self.device),
            'ppfd': torch.tensor(1000.0, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1500.0, device=self.device),
            'relative_humidity': torch.tensor(0.7, device=self.device),  # 70%
            'wind_speed': torch.tensor(2.0, device=self.device),  # 2 m/s
            'vcmax_scaling': torch.tensor(1.0, device=self.device)  # No scaling
        }

        self.test_state = {
            'leaf_carbon': torch.tensor(0.1, device=self.device),  # kg C m-2
            'froot_carbon': torch.tensor(0.2, device=self.device),  # kg C m-2
            'leaf_nitrogen': torch.tensor(0.004, device=self.device),  # kg N m-2
            'froot_nitrogen': torch.tensor(0.002, device=self.device),  # kg N m-2
            'stem_carbon': torch.tensor(0.5, device=self.device),  # kg C m-2
            'root_carbon': torch.tensor(0.2, device=self.device),  # kg C m-2
            'leaf_area_index': torch.tensor(4.0, device=self.device),  # m2 m-2
            'soil_moisture': torch.tensor(0.3, device=self.device),  # m3 m-3
            'water_stress_factor': torch.tensor(0.8, device=self.device),
            'phenology_phase': torch.tensor(2.0, device=self.device),
            'leaf_biomass_c': torch.tensor(0.1, device=self.device),
            'wood_biomass_c': torch.tensor(0.5, device=self.device),
            'root_biomass_c': torch.tensor(0.2, device=self.device),
            'biomass_leaf': torch.tensor(0.1, device=self.device),
            'biomass_stem': torch.tensor(0.5, device=self.device),
            'biomass_root': torch.tensor(0.2, device=self.device),
            'soluble_sugar': torch.tensor(0.01, device=self.device),
            'soil_water_potential': torch.tensor(-0.5, device=self.device),
            'growing_temperature': torch.tensor(293.15, device=self.device),
        }

    def test_pft_photosynthesis_integration(self):
        """Test PFT integration with Farquhar photosynthesis model."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Create photosynthesis model with PFT
            photosynthesis = FarquharPhotosynthesis(pft_name=pft_name)

            # Check that photosynthesis uses PFT parameters
            assert photosynthesis.vcmax25 is not None
            assert photosynthesis.jmax25 is not None
            assert photosynthesis.rd25 is not None

            # Compute photosynthesis with environment
            photo_output = photosynthesis.forward(
                {**self.test_state, **self.test_environment},
                time=0.0,
                dt=1.0
            )

            # Verify outputs
            assert 'gross_photosynthesis' in photo_output
            assert 'net_photosynthesis' in photo_output
            gross_photosynthesis = photo_output['gross_photosynthesis']
            assert isinstance(gross_photosynthesis, UpdateItem)
            assert gross_photosynthesis.mode is UpdateMode.REPLACE
            assert gross_photosynthesis.unit == 'umol CO2 m-2 s-1'
            assert gross_photosynthesis.value > 0

            # Check parameter consistency
            # Vcmax25 from PFT should match photosynthesis module
            vcmax_pft = pft.photosynthesis.vcmax25

            if isinstance(photosynthesis.vcmax25, torch.Tensor):
                vcmax_photo = photosynthesis.vcmax25.item()
            else:
                vcmax_photo = photosynthesis.vcmax25

            # They should be close (allowing for unit conversions)
            assert abs(vcmax_pft - vcmax_photo) / vcmax_photo < 0.1, \
                f"Vcmax mismatch for {pft_name}: PFT={vcmax_pft}, Photo={vcmax_photo}"

    def test_pft_respiration_integration(self):
        """Test PFT integration with respiration module."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Create respiration model
            respiration = MaintenanceRespiration(
                name=f"maintenance_resp_{pft_name}",
                config={
                    'maint_resp_leaf_25': pft.respiration.maint_resp_leaf,
                    'maint_resp_stem_25': pft.respiration.maint_resp_stem,
                    'maint_resp_root_25': pft.respiration.maint_resp_root,
                    'q10': pft.respiration.q10_maint_resp,
                }
            )

            # Compute respiration
            resp_output = respiration.forward(
                {**self.test_state, **self.test_environment},
                time=0.0,
                dt=1.0
            )

            # Verify outputs
            assert 'maintenance_respiration_total' in resp_output
            total_respiration = resp_output['maintenance_respiration_total']
            assert isinstance(total_respiration, UpdateItem)
            assert total_respiration.mode is UpdateMode.REPLACE
            assert total_respiration.per_timestep is False
            assert total_respiration.unit == 'kg C m-2 s-1'
            assert total_respiration.value >= 0

    def test_pft_bvoc_integration(self):
        """Test PFT integration with BVOC emission module."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Extract emission factors
            ef_isoprene = pft.bvoc.ef_isoprene
            ef_monoterpene = pft.bvoc.ef_monoterpene

            # Create BVOC emission model
            bvoc_model = MEGANModel(
                name=f"megan_{pft_name}",
                coupling_strategy=CouplingStrategy.INDEPENDENT,
                emission_pathway=EmissionPathway.STANDARD,
                config={'default_pft': pft_name},
            )

            torch.testing.assert_close(
                bvoc_model.pft_isoprene_emission_factor,
                torch.tensor(
                    ef_isoprene,
                    dtype=bvoc_model.pft_isoprene_emission_factor.dtype,
                ),
            )
            torch.testing.assert_close(
                bvoc_model.pft_monoterpenes_emission_factor,
                torch.tensor(
                    ef_monoterpene,
                    dtype=bvoc_model.pft_monoterpenes_emission_factor.dtype,
                ),
            )

            # Compute emissions
            bvoc_output = bvoc_model.forward(
                {
                    **self.test_state,
                    **self.test_environment,
                },
                time=0.0,
                dt=1.0
            )

            # Verify outputs
            assert 'isoprene_emission' in bvoc_output
            assert 'monoterpenes_emission' in bvoc_output

            # Check emissions are consistent with emission factors
            isoprene_item = bvoc_output['isoprene_emission']
            monoterpene_item = bvoc_output['monoterpenes_emission']
            assert isinstance(isoprene_item, UpdateItem)
            assert isinstance(monoterpene_item, UpdateItem)
            assert isoprene_item.mode is UpdateMode.REPLACE
            assert monoterpene_item.mode is UpdateMode.REPLACE
            assert isoprene_item.unit == 'mol m-2 s-1'
            assert monoterpene_item.unit == 'mol m-2 s-1'
            if ef_isoprene > 0:
                assert isoprene_item.value > 0
            if ef_monoterpene > 0:
                assert monoterpene_item.value > 0

    def test_pft_phenology_integration(self):
        """Test PFT integration with phenology module."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Get phenology parameters
            gdd_min = pft.phenology.gdd_min
            gdd_leaf_on = pft.phenology.gdd_leaf_on
            gdd_leaf_off = pft.phenology.gdd_leaf_off

            # Create phenology model using GSI phenology
            phenology = GSIPhenology(
                name=f"phenology_{pft_name}",
                config={
                    'phenology_parameters': {
                        'phenology_type': 'gsi',
                        'gsi_onset_threshold': 0.5,
                        'gsi_offset_threshold': 0.5,
                        'gdd_min': gdd_min,
                        'gdd_leaf_on': gdd_leaf_on,
                        'gdd_leaf_off': gdd_leaf_off
                    }
                }
            )

            # Compute phenology state
            pheno_output = phenology.forward(
                {
                    **self.test_state,
                    **self.test_environment,
                    "day_of_year": torch.tensor(180.0, device=self.device),
                    "air_temperature_min": torch.tensor(
                        293.15,
                        device=self.device,
                    ),
                    "daylength": torch.tensor(14.0 * 3600.0, device=self.device),
                    'phenology_gsi_history': torch.zeros(
                        phenology.memory_days,
                        device=self.device,
                    ),
                    'phenology_gsi_history_index': torch.tensor(
                        0.0,
                        device=self.device,
                    ),
                    'phenology_gsi_history_count': torch.tensor(
                        0.0,
                        device=self.device,
                    ),
                    'phenology_gsi_in_growing_season': torch.tensor(
                        0.0,
                        device=self.device,
                    ),
                },
                time=0.0,
                dt=get_physical_constant("seconds_per_day"),
            )

            # Verify outputs
            assert 'phenological_factor' in pheno_output
            phenological_factor = pheno_output['phenological_factor']
            assert isinstance(phenological_factor, UpdateItem)
            assert phenological_factor.mode is UpdateMode.REPLACE
            assert phenological_factor.unit == 'dimensionless'
            assert 0 <= phenological_factor.value.item() <= 1

    def test_pft_allocation_integration(self):
        """Test PFT integration with biomass allocation module."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Get allocation fractions
            f_leaf = pft.allocation.f_leaf
            f_stem = pft.allocation.f_stem
            f_root = pft.allocation.f_root

            # Verify allocation fractions sum to ~1
            total_allocation = f_leaf + f_stem + f_root
            assert abs(total_allocation - 1.0) < 0.01, \
                f"Allocation fractions don't sum to 1 for {pft_name}: {total_allocation}"

            # Create allocation model
            allocation = BiomassAllocation(
                name=f"allocation_{pft_name}",
                pft_name=pft_name,
                enable_environmental_control=False,
                enable_phenology_control=False,
                device=self.device,
            )

            # Test allocation of new carbon
            npp = torch.tensor(1.0e-6, device=self.device, dtype=torch.float32)
            # BiomassAllocation uses forward method, not allocate_carbon
            alloc_state = {
                **self.test_state,
                **self.test_environment,
                'net_primary_productivity': npp
            }
            alloc_output = allocation.forward(
                alloc_state,
                time=0.0,
                dt=86400.0  # Daily timestep
            )

            # Verify allocation - use correct output keys
            assert 'growth_rate_leaf' in alloc_output
            assert 'growth_rate_stem' in alloc_output
            assert 'growth_rate_root' in alloc_output

            rate_keys = ('growth_rate_leaf', 'growth_rate_stem', 'growth_rate_root')
            state_keys = ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c')
            for key in rate_keys:
                assert isinstance(alloc_output[key], UpdateItem)
                assert alloc_output[key].mode is UpdateMode.REPLACE
                assert alloc_output[key].unit == 'kg C m-2 s-1'
            for key in state_keys:
                assert isinstance(alloc_output[key], UpdateItem)
                assert alloc_output[key].mode is UpdateMode.ACCUMULATE
                assert alloc_output[key].per_timestep is True
                assert alloc_output[key].unit == 'kg C m-2'

            total_rate = sum(alloc_output[key].value for key in rate_keys)
            total_step = sum(alloc_output[key].value for key in state_keys)
            # The runtime state is float64 under the test session default dtype and the
            # Functor follows the runtime dtype; compare values, not the float32 NPP tag.
            torch.testing.assert_close(total_rate, npp.to(total_rate.dtype))
            torch.testing.assert_close(total_step, (npp * 86400.0).to(total_step.dtype))
            assert 'total_construction_respiration' not in alloc_output

            runtime_fractions = (
                alloc_output[
                    'diagnostics.biomass_allocation.allocation_fraction_leaf'
                ].value,
                alloc_output[
                    'diagnostics.biomass_allocation.allocation_fraction_stem'
                ].value,
                alloc_output[
                    'diagnostics.biomass_allocation.allocation_fraction_root'
                ].value,
            )
            torch.testing.assert_close(
                runtime_fractions[0],
                torch.tensor(
                    f_leaf,
                    device=self.device,
                    dtype=runtime_fractions[0].dtype,
                ),
            )
            torch.testing.assert_close(
                runtime_fractions[1],
                torch.tensor(
                    f_stem,
                    device=self.device,
                    dtype=runtime_fractions[1].dtype,
                ),
            )
            torch.testing.assert_close(
                runtime_fractions[2],
                torch.tensor(
                    f_root,
                    device=self.device,
                    dtype=runtime_fractions[2].dtype,
                ),
            )

    def test_cross_module_consistency(self):
        """Test parameter consistency across modules."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)

            # Create modules
            photosynthesis = FarquharPhotosynthesis(pft_name=pft_name)
            assert photosynthesis.vcmax25 == pft.photosynthesis.vcmax25
            assert photosynthesis.jmax25 == pft.photosynthesis.jmax25

            # Test 1: Temperature response consistency
            # Both photosynthesis and respiration should use consistent Q10 values
            q10_resp = pft.respiration.q10_maint_resp
            assert 1.5 <= q10_resp <= 3.0, f"Q10 out of range for {pft_name}: {q10_resp}"

            # Test 2: Leaf traits consistency
            sla = pft.phenology.specific_leaf_area
            leaf_longevity = pft.phenology.leaf_longevity

            # SLA and leaf longevity should be inversely related for non-evergreen types.
            if "evergreen" in pft_name:
                assert leaf_longevity >= 365, f"Evergreen leaf longevity too short for {pft_name}"
            else:
                if sla > 20:  # High SLA
                    assert leaf_longevity < 365, f"High SLA but long leaf longevity for {pft_name}"
                elif sla < 10:  # Low SLA
                    assert leaf_longevity > 365, f"Low SLA but short leaf longevity for {pft_name}"

    def test_pft_parameter_device_materialization(self):
        """Test explicit device materialization of typed PFT parameters."""
        pft = self.registry.get_pft('broadleaf_deciduous_temperate')
        parameter_values = torch.tensor(
            [
                pft.photosynthesis.vcmax25,
                pft.photosynthesis.jmax25,
                pft.allocation.f_leaf,
                pft.allocation.f_stem,
                pft.allocation.f_root,
            ],
            device=torch.device('cpu'),
        )
        assert parameter_values.device.type == 'cpu'
        assert torch.isfinite(parameter_values).all()

        if torch.cuda.is_available():
            gpu_values = parameter_values.to(torch.device('cuda'))
            assert gpu_values.device.type == 'cuda'
            torch.testing.assert_close(gpu_values.cpu(), parameter_values)

    def test_pft_parameter_ranges(self):
        """Test that all PFT parameters are within scientific ranges."""
        for pft_name in self.pft_types:
            pft_type = self.registry.get_pft(pft_name)
            assert isinstance(pft_type, PFTDefinition)

            # Photosynthesis parameters
            assert 10 <= pft_type.photosynthesis.vcmax25 <= 200, \
                f"Vcmax25 out of range for {pft_name}"
            assert 20 <= pft_type.photosynthesis.jmax25 <= 400, \
                f"Jmax25 out of range for {pft_name}"
            assert 0.1 <= pft_type.photosynthesis.rd25 <= 10, \
                f"Rd25 out of range for {pft_name}"

            # Allocation fractions
            assert abs(pft_type.allocation.f_leaf +
                      pft_type.allocation.f_stem +
                      pft_type.allocation.f_root - 1.0) < 0.01, \
                f"Allocation fractions don't sum to 1 for {pft_name}"

            # Phenology parameters
            assert 260.0 <= pft_type.phenology.gdd_min <= 290.0, \
                f"GDD min out of range (expected K) for {pft_name}"
            assert pft_type.phenology.gdd_leaf_on < pft_type.phenology.gdd_leaf_off, \
                f"GDD leaf on/off inconsistent for {pft_name}"

            # BVOC emission factors
            assert pft_type.bvoc.ef_isoprene >= 0, \
                f"Negative isoprene emission for {pft_name}"
            assert pft_type.bvoc.ef_monoterpene >= 0, \
                f"Negative monoterpene emission for {pft_name}"

