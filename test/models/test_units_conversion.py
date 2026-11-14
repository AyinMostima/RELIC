"""
Test suite for unit conversion functionality in relic.models

This module tests the unit conversion fixes applied to the models directory,
ensuring that all surrogate models handle units correctly and consistently
with the RELIC unit conversion system.

Test Categories:
    1. SurrogateFunctor unit conversion integration
    2. PyTorchSurrogate training data unit validation
    3. Explicit and SCM-backed unit resolution
    4. Error handling for invalid units

References:
    - CF-1.8 conventions for units
    - RELIC unit conversion system documentation
"""

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from pathlib import Path

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.models.pytorch_surrogate import PyTorchSurrogate, SurrogateDataset
from relic.models.surrogates import ONNXSurrogate, SurrogateError


_SURROGATE_REFERENCES = [
    "CF Conventions Committee. CF Metadata Conventions 1.13, section 3.1.",
    "Razavi, S., Tolson, B. A., and Burn, D. H. (2012). "
    "Review of surrogate modeling in water resources.",
]


def _metadata(
    name: str,
    input_variables: list[str],
    output_variables: list[str],
) -> FunctorMetadata:
    """Build biological-surrogate metadata for unit-contract tests."""
    return FunctorMetadata(
        name=name,
        description="Biological surrogate backend used for unit-contract tests.",
        category=ProcessCategory.BIOLOGICAL_PROCESS,
        references=list(_SURROGATE_REFERENCES),
        input_variables=list(input_variables),
        output_variables=list(output_variables),
        spatial_scale=SpatialScale.POINT,
        variable_kind=VariableKind.DIAGNOSTIC,
    )


def _create_temp_onnx_model(tmp_path: Path, input_size: int, output_size: int) -> Path:
    """Create a minimal linear ONNX model for tests."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_size])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, output_size])
    weight_data = np.zeros((input_size, output_size), dtype=np.float32)
    for idx in range(min(input_size, output_size)):
        weight_data[idx, idx] = 1.0
    bias_data = np.zeros(output_size, dtype=np.float32)
    weight_tensor = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [input_size, output_size],
        weight_data.flatten(),
    )
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [output_size], bias_data)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["matmul_out"]),
            helper.make_node("Add", ["matmul_out", "bias"], ["output"]),
        ],
        "unit_conversion_test_model",
        [input_tensor],
        [output_tensor],
        [weight_tensor, bias_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="relic_unit_conversion_test",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    model.ir_version = 10
    model_path = tmp_path / f"mock_model_{input_size}_{output_size}.onnx"
    onnx.save(model, model_path)
    return model_path


class TestSurrogateUnitConversion:
    """Test unit conversion in SurrogateFunctor and its subclasses."""
    
    def test_surrogate_units_conversion_integration(self, tmp_path):
        """Test that SurrogateFunctor integrates with authoritative unit converter."""
        model_path = _create_temp_onnx_model(tmp_path, input_size=3, output_size=2)

        config = {
            "model_path": str(model_path),
            "input_variables": ["custom_temperature", "custom_molar_flux", "custom_co2"],
            "output_variables": [
                "diagnostics.surrogate.primary",
                "diagnostics.surrogate.secondary",
            ],
            "units_conversion": {
                "custom_temperature": ("celsius", "K"),
                "custom_molar_flux": ("mol m-2 s-1", "umol m-2 s-1"),
                "custom_co2": ("ppm", "umol mol-1"),
            },
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
        }
        metadata = _metadata(
            "test_surrogate",
            config["input_variables"],
            config["output_variables"],
        )

        surrogate = ONNXSurrogate("test_surrogate", config, metadata)

        temp_c = 25.0
        temp_k = surrogate._convert_units(temp_c, "celsius", "K")
        assert abs(temp_k - 298.15) < 1e-5, f"Temperature conversion failed: {temp_c} C -> {temp_k}K"

        molar_flux = 0.0001
        flux_umol = surrogate._convert_units(
            molar_flux,
            "mol m-2 s-1",
            "umol m-2 s-1",
        )
        assert abs(flux_umol - 100.0) < 1e-5, (
            f"Molar flux conversion failed: {molar_flux}mol/m2/s -> {flux_umol}umol/m2/s"
        )

        co2_ppm = 400.0
        co2_umol = surrogate._convert_units(co2_ppm, "ppm", "umol mol-1")
        assert abs(co2_umol - co2_ppm) < 1e-6, f"CO2 conversion failed: {co2_ppm}ppm -> {co2_umol}umol/mol"
    
    def test_invalid_unit_conversion_raises(self, tmp_path):
        """Test unit conversion errors for invalid units."""
        model_path = _create_temp_onnx_model(tmp_path, input_size=1, output_size=1)
        config = {
            "model_path": str(model_path),
            "input_variables": ["custom_temperature"],
            "output_variables": ["diagnostics.surrogate.output"],
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
        }

        metadata = _metadata(
            "test_invalid_units",
            config["input_variables"],
            config["output_variables"],
        )
        surrogate = ONNXSurrogate("test_invalid_units", config, metadata)

        with pytest.raises(SurrogateError):
            surrogate._convert_units(100.0, "unknown_unit", "another_unknown")


class TestPyTorchSurrogateUnits:
    """Test unit conversion in PyTorchSurrogate training and inference."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data with explicitly declared units."""
        states = [
            {
                "custom_temperature": 20.0,
                "custom_radiation": 200.0,
                "custom_co2": 400.0,
                "custom_moisture": 0.3,
            },
            {
                "custom_temperature": 25.0,
                "custom_radiation": 400.0,
                "custom_co2": 450.0,
                "custom_moisture": 0.4,
            },
            {
                "custom_temperature": 30.0,
                "custom_radiation": 600.0,
                "custom_co2": 500.0,
                "custom_moisture": 0.2,
            },
        ]
        fluxes = [
            {
                "diagnostics.surrogate.primary": 10.0,
                "diagnostics.surrogate.secondary": 2.0,
            },
            {
                "diagnostics.surrogate.primary": 15.0,
                "diagnostics.surrogate.secondary": 3.0,
            },
            {
                "diagnostics.surrogate.primary": 12.0,
                "diagnostics.surrogate.secondary": 4.0,
            },
        ]
        return states, fluxes
    
    def test_dataset_unit_validation(self, sample_data):
        """Test SurrogateDataset unit validation functionality."""
        states, fluxes = sample_data
        
        input_variables = [
            "custom_temperature",
            "custom_radiation",
            "custom_co2",
            "custom_moisture",
        ]
        output_variables = [
            "diagnostics.surrogate.primary",
            "diagnostics.surrogate.secondary",
        ]
        input_units = {
            "custom_temperature": "celsius",
            "custom_radiation": "W m-2",
            "custom_co2": "ppm",
            "custom_moisture": "1",
        }
        output_units = {
            "diagnostics.surrogate.primary": "umol m-2 s-1",
            "diagnostics.surrogate.secondary": "umol m-2 s-1",
        }

        dataset = SurrogateDataset(
            states=states,
            fluxes=fluxes,
            input_variables=input_variables,
            output_variables=output_variables,
            input_units=input_units,
            output_units=output_units,
            validate_units=True,
        )

        assert len(dataset) == 3
        input_tensor, output_tensor = dataset[0]
        assert input_tensor.shape == torch.Size([4])
        assert output_tensor.shape == torch.Size([2])
        assert input_tensor.dtype == torch.float32
        assert output_tensor.dtype == torch.float32

        assert torch.equal(
            input_tensor,
            torch.tensor([20.0, 200.0, 400.0, 0.3], dtype=torch.float32),
        )
        assert torch.equal(
            output_tensor,
            torch.tensor([10.0, 2.0], dtype=torch.float32),
        )

    def test_dataset_invalid_declared_unit_fails_fast(self, sample_data):
        """Reject invalid unit declarations instead of warning and continuing."""
        states, fluxes = sample_data

        with pytest.raises(ValueError, match="Invalid input unit"):
            SurrogateDataset(
                states=states,
                fluxes=fluxes,
                input_variables=["custom_temperature"],
                output_variables=["diagnostics.surrogate.primary"],
                input_units={"custom_temperature": "not_a_unit"},
                output_units={
                    "diagnostics.surrogate.primary": "umol m-2 s-1",
                },
                validate_units=True,
            )
    
    def test_explicit_and_scm_unit_resolution(self, tmp_path):
        """Resolve canonical SCM units and require explicit synthetic units."""
        config = {
            "model_path": str(tmp_path / "unit_resolution.pth"),
            "input_variables": [
                "air_temperature",
                "ppfd",
                "atmospheric_co2_ppm",
                "custom_feature",
            ],
            "output_variables": [
                "gross_primary_productivity",
                "diagnostics.surrogate.custom_target",
            ],
            "input_units": {"custom_feature": "1"},
            "output_units": {
                "diagnostics.surrogate.custom_target": "1",
            },
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
            "validate_model_path": False,
        }
        metadata = _metadata(
            "test_unit_resolution",
            config["input_variables"],
            config["output_variables"],
        )
        surrogate = PyTorchSurrogate("test_unit_resolution", config, metadata)

        input_units = surrogate._get_expected_input_units()
        output_units = surrogate._get_expected_output_units()

        assert input_units == {
            "air_temperature": "K",
            "ppfd": "umol photons m-2 s-1",
            "atmospheric_co2_ppm": "umol mol-1",
            "custom_feature": "1",
        }
        assert output_units == {
            "gross_primary_productivity": "kg C m-2 s-1",
            "diagnostics.surrogate.custom_target": "1",
        }

    def test_missing_synthetic_unit_fails_fast(self, tmp_path):
        """Reject a variable that has neither an SCM nor an explicit unit."""
        config = {
            "model_path": str(tmp_path / "missing_unit.pth"),
            "input_variables": ["unknown_surrogate_feature"],
            "output_variables": ["diagnostics.surrogate.output"],
            "output_units": {"diagnostics.surrogate.output": "1"},
            "validate_model_path": False,
        }
        metadata = _metadata(
            "missing_unit",
            config["input_variables"],
            config["output_variables"],
        )
        surrogate = PyTorchSurrogate("missing_unit", config, metadata)

        with pytest.raises(
            ValueError,
            match="No unit declaration or SCM contract for input variable",
        ):
            surrogate._get_expected_input_units()

    @pytest.mark.parametrize(
        ("config_patch", "message"),
        [
            (
                {"input_units": {"unused_feature": "1"}},
                "input_units contains undeclared variables",
            ),
            (
                {"input_units": {"air_temperature": "celsius"}},
                "conflicts with SCM unit",
            ),
        ],
    )
    def test_invalid_unit_configuration_fails_fast(
        self,
        tmp_path,
        config_patch,
        message,
    ):
        """Reject extra keys and canonical SCM unit conflicts at construction."""
        config = {
            "model_path": str(tmp_path / "invalid_unit_config.pth"),
            "input_variables": ["air_temperature"],
            "output_variables": ["diagnostics.surrogate.output"],
            "output_units": {"diagnostics.surrogate.output": "1"},
            "validate_model_path": False,
            **config_patch,
        }

        with pytest.raises(ValueError, match=message):
            PyTorchSurrogate(
                "invalid_unit_config",
                config,
                _metadata(
                    "invalid_unit_config",
                    config["input_variables"],
                    config["output_variables"],
                ),
            )

    def test_training_with_unit_validation(self, sample_data, tmp_path):
        """Test PyTorchSurrogate training with unit validation enabled."""
        states, fluxes = sample_data

        config = {
            "model_path": str(tmp_path / "training_test_model.pth"),
            "input_variables": [
                "custom_temperature",
                "custom_radiation",
                "custom_co2",
            ],
            "output_variables": ["diagnostics.surrogate.primary"],
            "input_units": {
                "custom_temperature": "celsius",
                "custom_radiation": "W m-2",
                "custom_co2": "ppm",
            },
            "output_units": {
                "diagnostics.surrogate.primary": "umol m-2 s-1",
            },
            "architecture": {
                "hidden_dims": [16, 8],
                "activation": "relu",
            },
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 0.01,
            },
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
            "validate_model_path": False,
        }
        metadata = _metadata(
            "training_test",
            config["input_variables"],
            config["output_variables"],
        )
        surrogate = PyTorchSurrogate("training_test", config, metadata)
        results = surrogate.train(
            training_data=(states, fluxes),
            validate_units=True,
            verbose=False,
        )

        assert results["epochs_trained"] >= 1

        outputs = surrogate(
            {
                "custom_temperature": 22.0,
                "custom_radiation": 300.0,
                "custom_co2": 420.0,
            },
            time=0.0,
            dt=1800.0,
        )
        update = outputs["diagnostics.surrogate.primary"]
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.source_functor == "training_test"
        assert torch.isfinite(update.value).all()


class TestUnitConversionIntegration:
    """Test integration between different unit conversion components."""
    
    def test_cross_model_unit_consistency(self, tmp_path):
        """Test that different models handle units consistently."""
        model_path = _create_temp_onnx_model(tmp_path, input_size=1, output_size=1)
        test_temps = [0.0, 25.0, 100.0]

        onnx_config = {
            "model_path": str(model_path),
            "input_variables": ["custom_temperature"],
            "output_variables": ["diagnostics.surrogate.output"],
            "units_conversion": {
                "custom_temperature": ("celsius", "K"),
            },
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
        }
        pytorch_config = {
            "model_path": str(tmp_path / "pytorch_consistency.pth"),
            "input_variables": ["custom_temperature"],
            "output_variables": ["diagnostics.surrogate.output"],
            "input_units": {"custom_temperature": "celsius"},
            "output_units": {"diagnostics.surrogate.output": "1"},
            "units_conversion": {
                "custom_temperature": ("celsius", "K"),
            },
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
            "validate_model_path": False,
        }
        onnx_surrogate = ONNXSurrogate(
            "onnx_consistency",
            onnx_config,
            _metadata(
                "onnx_consistency",
                onnx_config["input_variables"],
                onnx_config["output_variables"],
            ),
        )
        pytorch_surrogate = PyTorchSurrogate(
            "pytorch_consistency",
            pytorch_config,
            _metadata(
                "pytorch_consistency",
                pytorch_config["input_variables"],
                pytorch_config["output_variables"],
            ),
        )

        for temp_c in test_temps:
            temp_k_onnx = onnx_surrogate._convert_units(
                temp_c,
                "celsius",
                "K",
            )
            temp_k_pytorch = pytorch_surrogate._convert_units(
                temp_c,
                "celsius",
                "K",
            )
            expected_k = temp_c + 273.15
            assert abs(temp_k_onnx - expected_k) < 1e-5
            assert abs(temp_k_pytorch - expected_k) < 1e-5
            assert abs(temp_k_pytorch - temp_k_onnx) < 1e-7

        assert pytorch_surrogate._get_expected_input_units() == {
            "custom_temperature": "celsius",
        }
    
    def test_error_handling_for_invalid_units(self, tmp_path):
        """Test error handling in unit conversion systems."""
        model_path = _create_temp_onnx_model(tmp_path, input_size=1, output_size=1)
        
        config = {
            "model_path": str(model_path),
            "input_variables": ["unknown_var"],
            "output_variables": ["diagnostics.surrogate.output"],
            "output_update_mode": "replace",
            "scale_rate_outputs_by_dt": False,
        }
        metadata = _metadata(
            "error_test",
            config["input_variables"],
            config["output_variables"],
        )
        surrogate = ONNXSurrogate("error_test", config, metadata)

        with pytest.raises(SurrogateError):
            surrogate._convert_units(100.0, "unknown_from", "unknown_to")

