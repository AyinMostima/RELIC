"""L2 integration tests for BVOC and stress functor coupling."""

import torch
from unittest.mock import patch

from test.bvoc_parameter_manager_helper import build_realistic_parameter_manager
from relic.core.state.types import UpdateItem
from relic.scientific.biosphere.bvoc.megan_model import (
    CouplingStrategy,
    EmissionPathway,
    MEGANModel,
)
from relic.scientific.biosphere.stress import WaterStress, ThermalStress


class TestBVOCStressCouplingL2:
    """Verify that stress functor outputs feed into MEGAN emissions."""

    def test_stress_functor_outputs_are_consumed(self):
        param_manager = build_realistic_parameter_manager()

        with patch('relic.parameters.manager.get_global_manager', return_value=param_manager):
            water_stress = WaterStress()
            thermal_stress = ThermalStress()
            megan_model = MEGANModel(
                config={"use_stress_functors": True},
                coupling_strategy=CouplingStrategy.SEQUENTIAL,
                emission_pathway=EmissionPathway.STANDARD,
            )
            baseline_model = MEGANModel(
                coupling_strategy=CouplingStrategy.SEQUENTIAL,
                emission_pathway=EmissionPathway.STANDARD,
            )

            water_state = {
                'soil_water_potential': torch.tensor(-0.8),  # MPa
                'air_temperature': torch.tensor(298.15),
            }
            thermal_state = {'air_temperature': torch.tensor(298.15)}

            water_outputs = water_stress.forward(water_state, time=0.0, dt=3600.0)
            thermal_outputs = thermal_stress.forward(thermal_state, time=0.0, dt=3600.0)

            megan_state = {
                "air_temperature": torch.tensor(298.15),
                "ppfd": torch.tensor(1000.0),
                "leaf_area_index": torch.tensor(2.2),
                "atmospheric_co2_ppm": torch.tensor(400.0),
            }
            megan_state['water_stress_factor'] = water_outputs[
                'water_stress_factor'
            ].value
            thermal_item = thermal_outputs['temperature_stress_factor']
            megan_state['temperature_stress_factor'] = (
                thermal_item.value
                if isinstance(thermal_item, UpdateItem)
                else thermal_item
            )
            megan_state["ozone_stress_factor"] = torch.tensor(1.0)

            outputs = megan_model.forward(megan_state, time=0.0, dt=3600.0)
            baseline_outputs = baseline_model.forward(
                {
                    key: value
                    for key, value in megan_state.items()
                    if not key.endswith("_stress_factor")
                },
                time=0.0,
                dt=3600.0,
            )

            expected_stress = (
                water_outputs['water_stress_factor'].value
                * megan_state['temperature_stress_factor']
            )
            stress_diagnostic = outputs[
                "diagnostics.bvoc.megan.environmental_stress_factor"
            ].value
            expected_stress = expected_stress.to(stress_diagnostic.dtype)
            assert torch.allclose(stress_diagnostic, expected_stress)
            assert torch.allclose(
                outputs["isoprene_emission"].value,
                baseline_outputs["isoprene_emission"].value * expected_stress,
                rtol=1.0e-6,
            )

