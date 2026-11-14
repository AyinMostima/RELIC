"""Unit tests for ML surrogate dual-path integration in the tutorial case."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Phase A: ML Functor Scaffold
# ---------------------------------------------------------------------------

class TestMLFunctorScaffold:
    """Verify CaseSoilMoistureSurrogate and CaseStomatalSurrogate classes."""

    def test_soil_moisture_surrogate_instantiation(self, tmp_path: Path) -> None:
        """SM surrogate can be created without a pre-trained checkpoint."""
        from tutorial.case.functors.ml_surrogates import CaseSoilMoistureSurrogate

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        surrogate = CaseSoilMoistureSurrogate(
            name="test_sm_ml",
            model_dir=model_dir,
        )
        assert surrogate.name == "test_sm_ml"
        assert "soil_moisture_ML" in surrogate.output_variables
        assert len(surrogate.input_variables) == 7

    def test_stomatal_surrogate_instantiation(self, tmp_path: Path) -> None:
        """Stomatal surrogate can be created without a pre-trained checkpoint."""
        from tutorial.case.functors.ml_surrogates import CaseStomatalSurrogate

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        surrogate = CaseStomatalSurrogate(
            name="test_gs_ml",
            model_dir=model_dir,
        )
        assert surrogate.name == "test_gs_ml"
        assert "stomatal_conductance_ML" in surrogate.output_variables
        assert len(surrogate.input_variables) == 6

    def test_soil_surrogate_output_suffix(self, tmp_path: Path) -> None:
        """Output variable names must have _ML suffix."""
        from tutorial.case.functors.ml_surrogates import CaseSoilMoistureSurrogate

        s = CaseSoilMoistureSurrogate(model_dir=tmp_path)
        for var in s.output_variables:
            assert var.endswith("_ML"), f"{var} does not end with _ML"

    def test_stomatal_surrogate_output_suffix(self, tmp_path: Path) -> None:
        """Stomatal output variable names must have _ML suffix."""
        from tutorial.case.functors.ml_surrogates import CaseStomatalSurrogate

        s = CaseStomatalSurrogate(model_dir=tmp_path)
        for var in s.output_variables:
            assert var.endswith("_ML"), f"{var} does not end with _ML"


# ---------------------------------------------------------------------------
# Phase B: Simulator Integration Helpers
# ---------------------------------------------------------------------------

class TestSimulatorHelpers:
    """Test utility functions from run_case.py for ML path support."""

    @pytest.fixture(autouse=True)
    def _add_case_to_path(self) -> None:
        case_dir = str(PROJECT_ROOT / "tutorial" / "case")
        if case_dir not in sys.path:
            sys.path.insert(0, case_dir)

    def test_suffix_outputs(self) -> None:
        """_suffix_outputs should add _ML to tensor keys."""
        from run_case import _suffix_outputs

        outputs = {
            "soil_moisture": torch.tensor(0.35),
            "temperature": torch.tensor(300.0),
        }
        result = _suffix_outputs(outputs, ["soil_moisture"])
        assert "soil_moisture_ML" in result
        assert "temperature_ML" not in result
        assert torch.isclose(result["soil_moisture_ML"], torch.tensor(0.35))

    def test_resolve_ml_runtime_settings(self) -> None:
        """_resolve_ml_runtime_settings parses simulation.yaml ml_surrogate block."""
        from run_case import _resolve_ml_runtime_settings

        cfg = {
            "simulation": {
                "ml_surrogate": {
                    "enabled": True,
                    "soil_moisture": True,
                    "stomatal": False,
                }
            }
        }
        settings = _resolve_ml_runtime_settings(cfg)
        assert settings["enabled"] is True
        assert settings["soil_moisture"] is True
        assert settings["stomatal"] is False

    def test_resolve_ml_runtime_settings_missing(self) -> None:
        """Missing ml_surrogate section returns empty dict."""
        from run_case import _resolve_ml_runtime_settings

        result = _resolve_ml_runtime_settings({"simulation": {}})
        assert result == {}


# ---------------------------------------------------------------------------
# Phase D: Plotting Auto-detection
# ---------------------------------------------------------------------------

class TestPlottingAutoDetection:
    """Verify _detect_ml_variables from plot_results.py."""

    def test_detect_ml_variables_present(self) -> None:
        """When _ML variables exist alongside base, they are found."""
        from tutorial.case.plot_results import _detect_ml_variables

        ds = xr.Dataset(
            {
                "soil_moisture": ("time", np.random.rand(10)),
                "soil_moisture_ML": ("time", np.random.rand(10)),
                "net_photosynthesis": ("time", np.random.rand(10)),
                "net_photosynthesis_ML": ("time", np.random.rand(10)),
                "temperature": ("time", np.random.rand(10)),
            }
        )
        ml_map = _detect_ml_variables(ds)
        assert "soil_moisture" in ml_map
        assert ml_map["soil_moisture"] == "soil_moisture_ML"
        assert "net_photosynthesis" in ml_map
        assert ml_map["net_photosynthesis"] == "net_photosynthesis_ML"
        assert "temperature" not in ml_map

    def test_detect_ml_variables_empty(self) -> None:
        """When no _ML variables exist, returns empty dict."""
        from tutorial.case.plot_results import _detect_ml_variables

        ds = xr.Dataset(
            {
                "soil_moisture": ("time", np.random.rand(10)),
                "temperature": ("time", np.random.rand(10)),
            }
        )
        ml_map = _detect_ml_variables(ds)
        assert ml_map == {}

    def test_detect_ml_variables_orphan(self) -> None:
        """_ML variable without matching base is NOT detected."""
        from tutorial.case.plot_results import _detect_ml_variables

        ds = xr.Dataset(
            {
                "orphan_variable_ML": ("time", np.random.rand(10)),
                "temperature": ("time", np.random.rand(10)),
            }
        )
        ml_map = _detect_ml_variables(ds)
        assert ml_map == {}
