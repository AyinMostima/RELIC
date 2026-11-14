"""
Shared test fixtures and utilities for L3 system tests.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from typing import Dict, List
from relic.core.functor import (
    FunctorBase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
)
from relic.core.state import UpdateItem, UpdateMode


def _mock_metadata(
    name,
    description,
    input_variables,
    output_variables,
    category=ProcessCategory.BIOLOGICAL_PROCESS,
):
    return FunctorMetadata(
        name=name,
        description=description,
        category=category,
        references=[
            "Test-only fixture for RELIC parameter-system integration; not a scientific process"
        ],
        input_variables=list(input_variables),
        output_variables=list(output_variables),
    )

class MockLitterProduction(FunctorBase):
    """Mock litter production functor for testing."""
    
    def __init__(self, name: str = "litter_production", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only litter input fixture",
                (),
                ("litter_carbon", "litter_nitrogen"),
            ),
        )
        
    def get_dependencies(self) -> List[str]:
        """Mock litter production has no dependencies."""
        return []
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Simple mock forward pass that produces minimal litter."""
        return {
            "litter_carbon": torch.tensor(0.1),  # kg C m-2 s-1
            "litter_nitrogen": torch.tensor(0.01)  # kg N m-2 s-1
        }

class MockEnvironmentalForcing(PhysicalFieldFunctor):
    """Mock environmental forcing for testing."""

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )
    
    def __init__(self, name: str = "environmental_forcing", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only external forcing dependency marker",
                (),
                ("diagnostics.test.environmental_forcing_ready",),
                category=ProcessCategory.PHYSICAL_FIELD,
            ),
        )
        
    def get_dependencies(self) -> List[str]:
        return []
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Provide basic environmental state."""
        # These should not be treated as state variables for accumulation
        # Instead, they should be directly set in state by the test
        return {
            "diagnostics.test.environmental_forcing_ready": self.wrap_update(
                torch.tensor(1.0),
                mode=UpdateMode.REPLACE,
            )
        }

class MockLeafTemperatureCalculator(FunctorBase):
    """Mock leaf temperature calculator for testing."""
    
    def __init__(self, name: str = "leaf_temperature_calculator", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only leaf driver adapter",
                ("temperature.air", "radiation.par", "co2.concentration"),
                ("temperature", "vcmax_scaling", "par", "atmospheric_co2_ppm"),
            ),
        )
        
    def get_dependencies(self) -> List[str]:
        return ["environmental_forcing"]
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Calculate leaf temperature from air temperature."""
        # Get air temperature from state, ensure it's within valid range
        air_temp = state.get(
            "temperature.air",
            state.get("air_temperature", torch.tensor(298.15)),
        )
        
        # Always clamp to valid range [273.15-323.15] K as required by Farquhar
        if isinstance(air_temp, torch.Tensor):
            air_temp = torch.clamp(air_temp, min=275.0, max=320.0)  # Conservative safe range
        else:
            air_temp = torch.tensor(max(275.0, min(320.0, float(air_temp))))
        
        # Get other required inputs with safe defaults
        par = state.get("radiation.par", state.get("par", torch.tensor(400.0)))
        co2_conc = state.get(
            "co2.concentration",
            state.get("co2_concentration", torch.tensor(400.0)),
        )
        
        return {
            "temperature": self.wrap_update(air_temp, mode=UpdateMode.REPLACE),
            "vcmax_scaling": self.wrap_update(torch.tensor(1.0), mode=UpdateMode.REPLACE),
            "par": self.wrap_update(par, mode=UpdateMode.REPLACE),
            "atmospheric_co2_ppm": self.wrap_update(co2_conc, mode=UpdateMode.REPLACE),
        }

class MockSoilTemperatureField(FunctorBase):
    """Mock soil temperature field for testing."""
    
    def __init__(self, name: str = "soil_temperature_field", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only soil temperature adapter",
                ("temperature.air",),
                ("temperature.soil",),
            ),
        )
        
    def get_dependencies(self) -> List[str]:
        return ["environmental_forcing"]
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Provide soil temperature."""
        air_temp = state.get("temperature.air", torch.tensor(298.15))
        soil_temp = air_temp - 2.0  # Soil slightly cooler
        
        # Ensure soil temperature is in valid range [233.15-333.15] K for decomposition
        if isinstance(soil_temp, torch.Tensor):
            soil_temp = torch.clamp(soil_temp, min=235.0, max=330.0)
        else:
            soil_temp = torch.tensor(max(235.0, min(330.0, float(soil_temp))))
            
        # Return as environmental forcing, not as state variable
        # Environmental variables should not accumulate
        return {
            "temperature.soil": soil_temp  # Use dotted notation to indicate environmental variable
        }

class MockSoilMoistureModel(FunctorBase):
    """Mock soil moisture model for testing."""
    
    def __init__(self, name: str = "soil_water_transport", config: Dict = None):
        super().__init__(
            name,
            config or {},
            metadata=_mock_metadata(
                name,
                "Test-only soil moisture adapter",
                (),
                ("soil_moisture",),
            ),
        )
        
    def get_dependencies(self) -> List[str]:
        return []
    
    def forward(self, state: Dict, time: float, dt: float) -> Dict:
        """Provide soil moisture."""
        return {
            "soil_moisture": UpdateItem(
                value=torch.tensor(0.25),
                mode=UpdateMode.REPLACE,
                unit="m3 m-3",
                basis="instantaneous",
                per_timestep=False,
                source_functor=self.name,
            )
        }

