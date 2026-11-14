"""
Comprehensive Test Suite for RELIC Hydrosphere Transport Mechanisms

This test suite validates the refactored hydrosphere modules to ensure they
properly inherit from TransportMechanism and implement the required interfaces.

Test Coverage:
1. Richards soil moisture model (transport mechanism)
2. Penman-Monteith evapotranspiration (transport mechanism) 
3. Priestley-Taylor evapotranspiration (transport mechanism)
4. Bucket soil moisture model (transport mechanism)
5. Surface runoff model (transport mechanism)
6. Subsurface runoff model (transport mechanism)

Each test validates:
- Proper inheritance from TransportMechanism
- Implementation of compute_flux method
- consistency with forward method
- Parameter validation
- Physical conservation laws
- Error handling

Authors: Yuchong Long
Date: 2025-01-27
"""

import pytest
import torch

# Import core abstractions
from relic.core.abstractions import PhysicalField
from relic.core.functor import FunctorBase
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from test.utils.forward_call import call_transport_forward

# Import hydrosphere models
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET
from relic.scientific.hydrosphere.runoff.surface import SurfaceRunoff
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff
from relic.scientific.hydrosphere.transport.vertical_percolation import VerticalPercolationTransport


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _get_richards_diagnostics(result):
    diagnostics = result.get("diagnostics.soil_moisture_richards", {})
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


def test_richards_vertical_percolation_mutual_exclusion():
    simulator = RELICSimulator(config={})
    simulator.register_functor(RichardsSoilMoisture(name="richards_test"))
    with pytest.raises(ValueError, match="VerticalPercolationTransport"):
        simulator.register_functor(VerticalPercolationTransport(name="vp_test"))


class MockPhysicalField(PhysicalField):
    """Mock PhysicalField for testing compute_flux methods."""

    SPATIAL_DIMS = 1

    def __init__(self, value: float = 0.0, gradient: float = 0.0):
        super().__init__()
        self.value = value
        self.gradient_val = gradient
    
    def get_value_at(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float32)
    
    def compute_gradient(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.gradient_val], dtype=torch.float32)
    
    def interpolate(self, source_coordinates: torch.Tensor, target_coordinates: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float32)
    
    def update_field(self, new_values: torch.Tensor, coordinates: torch.Tensor) -> None:
        self.value = float(new_values.mean())


class TestRichardsSoilMoisture:
    """Test suite for Richards soil moisture transport mechanism."""

    @pytest.fixture
    def parameter_manager_snapshot(self):
        """Snapshot and restore ParameterManager state."""
        manager = get_global_manager()
        snapshot = manager.snapshot()
        yield manager
        manager.restore(snapshot)

    @pytest.fixture
    def richards_model(self, parameter_manager_snapshot):
        """Create Richards soil moisture model."""
        manager = parameter_manager_snapshot
        manager.set_value(
            "hydrosphere.soil_moisture.layer_count",
            5,
            units="dimensionless",
            validate=True,
            notify=False,
        )
        manager.set_value(
            "hydrosphere.soil_moisture.soil_depth",
            1.0,
            units="m",
            validate=True,
            notify=False,
        )
        return RichardsSoilMoisture(name="test_richards")
    
    def test_inheritance(self, richards_model):
        """Test proper inheritance from FunctorBase with TransportMechanism functionality."""
        assert isinstance(richards_model, FunctorBase)
        assert hasattr(richards_model, 'forward')
        assert hasattr(richards_model, 'transport_type')
        assert hasattr(richards_model, 'diffusivity')
    
    def test_compute_flux_method(self, richards_model):
        """Test the canonical state-driven Richards flux outputs."""
        result = richards_model.forward(
            {
                "soil_moisture": torch.full(
                    (richards_model.soil_layers,),
                    0.3,
                    dtype=torch.float32,
                ),
                "infiltration_flux": torch.tensor(
                    0.0, dtype=torch.float32
                ),
                "evapotranspiration_flux": torch.tensor(
                    0.0, dtype=torch.float32
                ),
            },
            time=0.0,
            dt=3600.0,
        )

        drainage_flux = _unwrap_update(result["drainage_flux"])
        water_flux_profile = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.water_flux_profile"
            ]
        )
        assert isinstance(drainage_flux, torch.Tensor)
        assert isinstance(water_flux_profile, torch.Tensor)
        assert drainage_flux.dtype == torch.float32
        assert water_flux_profile.shape == (richards_model.soil_layers,)
        assert torch.isfinite(drainage_flux).all()
        assert torch.isfinite(water_flux_profile).all()
    
    def test_forward_method(self, richards_model):
        """Test forward method with valid inputs."""
        state = {
            "soil_moisture": torch.full(
                (richards_model.soil_layers,), 0.3
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = richards_model.forward(state, time=0.0, dt=3600.0)

        assert isinstance(result, dict)
        assert (
            "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            in result
        )
        assert "diagnostics.soil_moisture_richards" in result
        diagnostics = _get_richards_diagnostics(result)
        assert "pressure_head" in diagnostics
        soil_change = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        assert isinstance(soil_change, torch.Tensor)
    
    def test_parameter_validation(self, richards_model):
        """Test parameter validation."""
        # Test invalid soil moisture
        state = {
            "soil_moisture": torch.full(
                (richards_model.soil_layers,), -0.1
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        with pytest.raises(ValueError, match="must be in range"):
            richards_model.forward(state, time=0.0, dt=3600.0)
    
    def test_conservation_laws(self, richards_model):
        """Test mass conservation."""
        state = {
            "soil_moisture": torch.full(
                (richards_model.soil_layers,), 0.4
            ),
            "infiltration_flux": torch.tensor(1e-5),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        
        result = richards_model.forward(state, time=0.0, dt=3600.0)

        # Check that moisture changes are physically reasonable
        moisture_change = _unwrap_update(
            result[
                "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
            ]
        )
        assert torch.all(torch.abs(moisture_change) < 1.0)  # Not too large
        assert not torch.isnan(moisture_change).any()


class TestBucketSoilMoisture:
    """Test suite for Bucket soil moisture transport mechanism."""
    
    @pytest.fixture
    def bucket_model(self):
        """Create Bucket soil moisture model."""
        return BucketSoilMoisture(name="test_bucket")
    
    def test_inheritance(self, bucket_model):
        """Test proper inheritance from TransportMechanism."""
        assert isinstance(bucket_model, FunctorBase)
        assert hasattr(bucket_model, 'forward')
        assert hasattr(bucket_model, 'forward')
    
    def test_compute_flux_method(self, bucket_model):
        """Test the canonical state-driven bucket flux outputs."""
        result = bucket_model.forward(
            {
                "soil_moisture": torch.tensor(0.5),
                "precipitation_flux": torch.tensor(2e-5),
                "evapotranspiration_flux": torch.tensor(1e-5),
            },
            time=0.0,
            dt=3600.0,
        )

        drainage_flux = _unwrap_update(result["drainage_flux"])
        runoff_flux = _unwrap_update(result["surface_runoff_flux"])
        assert torch.isfinite(drainage_flux).all()
        assert torch.isfinite(runoff_flux).all()
        assert torch.all(drainage_flux >= 0.0)
        assert torch.all(runoff_flux >= 0.0)
    
    def test_forward_method(self, bucket_model):
        """Test forward method."""
        state = {
            "soil_moisture": torch.tensor(0.4),
            "precipitation_flux": torch.tensor(2e-5),
            "evapotranspiration_flux": torch.tensor(1e-5)
        }
        
        result = bucket_model.forward(state, time=0.0, dt=3600.0)

        assert isinstance(result, dict)
        assert "soil_moisture" in result
        soil_update = result["soil_moisture"]
        if isinstance(soil_update, UpdateItem):
            assert isinstance(soil_update.value, torch.Tensor)
            assert soil_update.mode == UpdateMode.REPLACE
        else:
            assert isinstance(soil_update, torch.Tensor)


class TestEvapotranspirationModels:
    """Test suite for evapotranspiration transport mechanisms."""
    
    @pytest.fixture
    def penman_model(self):
        """Create Penman-Monteith ET model."""
        return PenmanMonteithET(name="test_penman", config={"canopy_resistance_source": "state"})
    
    @pytest.fixture  
    def priestley_model(self):
        """Create Priestley-Taylor ET model."""
        return PriestleyTaylorET(name="test_priestley")
    
    def test_penman_inheritance(self, penman_model):
        """Test Penman-Monteith inheritance."""
        assert isinstance(penman_model, FunctorBase)
        assert hasattr(penman_model, 'forward')
    
    def test_priestley_inheritance(self, priestley_model):
        """Test Priestley-Taylor inheritance."""
        assert isinstance(priestley_model, FunctorBase)
        assert hasattr(priestley_model, 'forward')
    
    def test_penman_compute_flux(self, penman_model):
        """Test canonical state-driven Penman-Monteith transport output."""
        result = penman_model.forward(
            {
                "air_temperature": torch.tensor(298.15),
                "relative_humidity": torch.tensor(0.6),
                "net_radiation": torch.tensor(200.0),
                "wind_speed": torch.tensor(2.0),
                "atmospheric_pressure": torch.tensor(101325.0),
                "surface_resistance": torch.tensor(100.0),
            },
            time=0.0,
            dt=3600.0,
        )

        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert torch.isfinite(et_flux).all()
        assert torch.all(et_flux >= 0)
    
    def test_penman_forward(self, penman_model):
        """Test Penman-Monteith forward method."""
        state = {
            "air_temperature": torch.tensor(298.15),  # 25 C in K
            "relative_humidity": torch.tensor(0.6),
            "net_radiation": torch.tensor(200.0),
            "wind_speed": torch.tensor(2.0),
            # Penman-Monteith reads the canonical atmospheric_pressure; a missing
            # value raises instead of taking a diagnostic default.
            "atmospheric_pressure": torch.tensor(101325.0),
            # A canopy resistance is required (the removed default was 100 s m-1).
            "surface_resistance": torch.tensor(100.0),
            "soil_moisture": torch.tensor(0.4)
        }
        
        result = penman_model.forward(state, time=0.0, dt=3600.0)
        
        assert isinstance(result, dict)
        assert "evapotranspiration_flux" in result
        et_flux = _unwrap_update(result["evapotranspiration_flux"])
        assert et_flux >= 0


class TestRunoffModels:
    """Test suite for runoff transport mechanisms."""
    
    @pytest.fixture
    def surface_model(self):
        """Create surface runoff model."""
        return SurfaceRunoff(name="test_surface")
    
    @pytest.fixture
    def subsurface_model(self):
        """Create subsurface runoff model.""" 
        return SubsurfaceRunoff(name="test_subsurface")
    
    def test_surface_inheritance(self, surface_model):
        """Test surface runoff inheritance."""
        assert isinstance(surface_model, FunctorBase)
        assert hasattr(surface_model, 'forward')
    
    def test_subsurface_inheritance(self, subsurface_model):
        """Test subsurface runoff inheritance."""
        assert isinstance(subsurface_model, FunctorBase)
        assert hasattr(subsurface_model, 'forward')
    
    def test_surface_compute_flux(self, surface_model):
        """Test the canonical state-driven surface runoff flux."""
        result = surface_model.forward(
            {
                "precipitation_flux": torch.tensor(5e-5),
                "soil_moisture": torch.tensor(0.7),
            },
            time=0.0,
            dt=3600.0,
        )

        assert isinstance(result, dict)
        assert "surface_runoff_flux" in result
        flux = _unwrap_update(result["surface_runoff_flux"])
        assert isinstance(flux, torch.Tensor)
        assert torch.isfinite(flux).all()
        assert flux >= 0
    
    def test_surface_forward(self, surface_model):
        """Test surface runoff forward method."""
        state = {
            "precipitation_flux": torch.tensor(5e-5),  # kg m-2 s-1
            "soil_moisture": torch.tensor(0.7)
        }
        
        result = surface_model.forward(state, time=0.0, dt=3600.0)
        
        assert isinstance(result, dict)
        assert "surface_runoff_flux" in result
        surface_flux = _unwrap_update(result["surface_runoff_flux"])
        assert surface_flux >= 0
    
    def test_subsurface_forward(self, subsurface_model):
        """Test subsurface runoff forward method."""
        state = {
            "soil_moisture": torch.tensor(0.5),
            "drainage_flux": torch.tensor(1e-6),
            "groundwater_storage": torch.tensor(100.0),
        }
        
        result = subsurface_model.forward(state, time=0.0, dt=3600.0)
        
        assert isinstance(result, dict)
        assert "subsurface_runoff_flux" in result
        subsurface_flux = _unwrap_update(result["subsurface_runoff_flux"])
        assert subsurface_flux >= 0


class TestIntegrationAndCoupling:
    """Integration tests for hydrosphere module coupling."""
    
    def test_water_balance_conservation(self):
        """Test the SurfaceRunoff open precipitation partition."""
        # SurfaceRunoff and BucketSoilMoisture are alternative runoff closures.
        surface = SurfaceRunoff(name="surface_runoff")
        penman = PenmanMonteithET(name="penman", config={"canopy_resistance_source": "state"})
        
        # Test state
        state = {
            "soil_moisture": torch.tensor(0.4),
            "precipitation_flux": torch.tensor(5e-5),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "net_radiation": torch.tensor(200.0),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
            "evapotranspiration_flux": torch.tensor(1e-5)
        }
        
        # Compute fluxes
        et_result = penman.forward(state, time=0.0, dt=3600.0)
        state["evapotranspiration_flux"] = _unwrap_update(et_result["evapotranspiration_flux"])
        
        runoff_result = surface.forward(state, time=0.0, dt=3600.0)
        state["surface_runoff_flux"] = _unwrap_update(runoff_result["surface_runoff_flux"])
        state["infiltration_flux"] = _unwrap_update(runoff_result["infiltration_flux"])
        
        # Check water balance (simplified)
        precipitation = state["precipitation_flux"]
        et_flux = state["evapotranspiration_flux"] 
        runoff_flux = state["surface_runoff_flux"] 
        infiltration_flux = state["infiltration_flux"]
        
        assert et_flux >= 0, f"ET flux should be non-negative: {et_flux}"
        assert runoff_flux >= 0, f"Runoff flux should be non-negative: {runoff_flux}"
        assert infiltration_flux >= 0, f"Infiltration flux should be non-negative: {infiltration_flux}"
        torch.testing.assert_close(
            precipitation,
            runoff_flux + infiltration_flux,
            rtol=1e-6,
            atol=1e-12,
        )
        
        # Verify fluxes are within physically reasonable bounds
        max_reasonable_et = precipitation * 10  # Allow ET up to 10x precipitation (can happen with stored soil water)
        assert et_flux <= max_reasonable_et, f"ET flux too high: {et_flux} > {max_reasonable_et}"
    
    def test_compute_rate_interface_consistency(self):
        """Test that compute_rate interface works for all models."""
        models = [
            RichardsSoilMoisture(name="richards"),
            BucketSoilMoisture(name="bucket"), 
            PenmanMonteithET(name="penman", config={"canopy_resistance_source": "state"}),
            PriestleyTaylorET(name="priestley"),
            SurfaceRunoff(name="surface"),
            SubsurfaceRunoff(name="subsurface")
        ]
        
        base_state = {
            "soil_moisture": torch.tensor(0.4),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "precipitation_flux": torch.tensor(2e-5),
            "evapotranspiration_flux": torch.tensor(1e-5),
            "net_radiation": torch.tensor(200.0),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
            "groundwater_storage": torch.tensor(100.0),
            "drainage_flux": torch.tensor(1e-6),
            "time": 0.0,
            "dt": 3600.0
        }
        
        for model in models:
            model_state = dict(base_state)
            if isinstance(model, RichardsSoilMoisture):
                model_state["soil_moisture"] = torch.full(
                    (model.soil_layers,), 0.3, dtype=torch.float32
                )
                model_state["infiltration_flux"] = torch.tensor(
                    0.0, dtype=torch.float32
                )
                model_state["evapotranspiration_flux"] = torch.tensor(
                    0.0, dtype=torch.float32
                )
            result = call_transport_forward(
                model,
                model_state,
                {},
                dt=float(base_state["dt"]),
            )
            assert isinstance(result, dict)
    
    def test_transport_mechanism_interface_compliance(self):
        """Test that all models properly implement TransportMechanism-like interface."""
        models = [
            RichardsSoilMoisture(name="richards"),
            BucketSoilMoisture(name="bucket"),
            PenmanMonteithET(name="penman", config={"canopy_resistance_source": "state"}),
            PriestleyTaylorET(name="priestley"),
            SurfaceRunoff(name="surface"),
            SubsurfaceRunoff(name="subsurface")
        ]
        
        base_state = {
            "soil_moisture": torch.tensor(0.4),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.6),
            "precipitation_flux": torch.tensor(2e-5),
            "evapotranspiration_flux": torch.tensor(1e-5),
            "net_radiation": torch.tensor(200.0),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0),
            "groundwater_storage": torch.tensor(100.0),
            "drainage_flux": torch.tensor(1e-6),
        }
        
        for model in models:
            # Test the single state-driven transport entry point.
            assert hasattr(model, 'forward')
            model_state = dict(base_state)
            if isinstance(model, RichardsSoilMoisture):
                model_state["soil_moisture"] = torch.full(
                    (model.soil_layers,), 0.3, dtype=torch.float32
                )
                model_state["infiltration_flux"] = torch.tensor(
                    0.0, dtype=torch.float32
                )
                model_state["evapotranspiration_flux"] = torch.tensor(
                    0.0, dtype=torch.float32
                )
            flux = call_transport_forward(model, model_state, {}, dt=3600.0)
            # Accept both Tensor and Dict (for FluxDict returns)
            if isinstance(flux, dict):
                # If it's a FluxDict, check for main flux value
                main_keys = [
                    'flux',
                    'transport_flux',
                    'vertical_flux',
                    'evapotranspiration_flux',
                    'drainage_flux',
                    'percolation_drainage_flux',
                    'subsurface_runoff_flux',
                ]
                found_flux = None
                for key in main_keys:
                    if key in flux:
                        found_flux = _unwrap_update(flux[key])
                        break
                if found_flux is None:
                    # Use first tensor value
                    for value in flux.values():
                        if isinstance(value, torch.Tensor):
                            found_flux = value
                            break
                assert found_flux is not None, f"No tensor flux found in {list(flux.keys())}"
                assert not torch.isnan(found_flux).any()
            else:
                assert isinstance(flux, torch.Tensor)
                assert not torch.isnan(flux).any()
            
            # Test transport_type attribute
            assert hasattr(model, 'transport_type')
            assert hasattr(model, 'diffusivity')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

