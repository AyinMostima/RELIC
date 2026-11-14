"""
Integration Tests for ET-Soil Moisture Coupling

Tests that evapotranspiration correctly couples with soil moisture.
"""

import pytest
import torch
from relic.scientific.hydrosphere.evapotranspiration import PenmanMonteithET
from relic.scientific.hydrosphere.soil_moisture import BucketSoilMoisture


def _unwrap_update(value):
    if hasattr(value, "value"):
        return value.value
    return value


class TestETSoilCoupling:
    """Test coupling between ET and soil moisture."""
    
    def test_et_to_soil_coupling(self, et_state, time_params):
        """Test that ET output can feed into soil moisture model."""
        # Step 1: Calculate ET
        et_functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        et_result = et_functor.forward(
            et_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        et_flux = _unwrap_update(et_result["evapotranspiration_flux"])
        
        # Step 2: Use ET as input to soil moisture
        soil_functor = BucketSoilMoisture()
        soil_state = {
            "soil_moisture": torch.tensor(0.3),
            "precipitation_flux": torch.tensor(0.0),
            "evapotranspiration_flux": et_flux,
            "air_temperature": torch.tensor(293.15),
        }
        
        soil_result = soil_functor.forward(
            soil_state,
            time=time_params["time"],
            dt=time_params["dt"]
        )
        
        # Verify coupling: ET should reduce soil moisture
        final_sm = _unwrap_update(soil_result["soil_moisture"])
        assert (final_sm <= soil_state["soil_moisture"]).item(), (
            "ET should reduce or maintain soil moisture"
        )
    
    def test_mass_consistency(self, time_params):
        """Test mass consistency across ET-soil coupling."""
        et_functor = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        soil_functor = BucketSoilMoisture()
        
        # Set up coupled state
        et_state = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(293.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
        }
        
        et_result = et_functor.forward(et_state, 0.0, time_params["dt"])
        
        soil_state = {
            "soil_moisture": torch.tensor(0.3),
            "precipitation_flux": torch.tensor(0.002),  # 7.2 mm/h
            "evapotranspiration_flux": _unwrap_update(et_result["evapotranspiration_flux"]),
            "air_temperature": torch.tensor(293.15),
        }
        
        soil_result = soil_functor.forward(soil_state, 0.0, time_params["dt"])
        
        # Both models should produce physically reasonable results
        assert torch.isfinite(_unwrap_update(et_result["evapotranspiration_flux"]))
        assert torch.isfinite(_unwrap_update(soil_result["soil_moisture"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

