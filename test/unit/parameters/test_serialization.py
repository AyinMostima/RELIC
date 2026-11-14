"""
L1 Unit Tests for Parameter Serialization Component

Tests for parameter serialization and deserialization functionality.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
import torch

from relic.parameters.serialization import (
    export,
    load,
    convert_format,
    load_yaml,
    load_json,
    load_toml,
    save_yaml,
    save_json,
    save_toml,
    serialize_tensor,
    deserialize_tensor,
    serialize_parameter_spec,
    deserialize_parameter_spec,
)
from relic.parameters import ParameterSpec, ParameterCategory


def _schema(parameters: dict) -> dict:
    """Build canonical serialization schema payload."""
    return {
        "version": "1.0.0",
        "timestamp": "2026-01-01T00:00:00",
        "parameters": parameters,
    }


class TestSerializationFormats:
    """Test different serialization formats."""

    def test_json_encoder_preserves_enum_value(self, tmp_path):
        path = tmp_path / "enum.json"
        save_json(_schema({"test.category": {"value": ParameterCategory.PHYSICAL}}), path)
        assert load_json(path)["parameters"]["test.category"]["value"] == ParameterCategory.PHYSICAL.value
    
    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for testing."""
        return {
            "biosphere.photosynthesis.vcmax25": {
                "value": 60.0,
                "units": "umol CO2 m-2 s-1",
                "bounds": [10.0, 200.0],
                "dtype": "float32",
                "category": "biophysical",
                "description": "Maximum carboxylation rate at 25掳C",
                "references": ["Farquhar et al. 1980"]
            },
            "atmosphere.radiation.albedo": {
                "value": 0.3,
                "units": "dimensionless",
                "bounds": [0.0, 1.0],
                "dtype": "float32",
                "category": "environmental",
                "description": "Surface albedo"
            }
        }
    
    @pytest.fixture
    def sample_specs(self):
        """Create sample ParameterSpec objects."""
        return {
            "vcmax25": ParameterSpec(
                key="biosphere.photosynthesis.vcmax25",
                value=60.0,
                units="umol CO2 m-2 s-1",
                bounds=(10.0, 200.0),
                dtype="float32",
                category=ParameterCategory.BIOPHYSICAL,
                description="Maximum carboxylation rate at 25掳C",
                references=["Farquhar et al. 1980"]
            ),
            "albedo": ParameterSpec(
                key="atmosphere.radiation.albedo",
                value=0.3,
                units="dimensionless",
                bounds=(0.0, 1.0),
                dtype="float32",
                category=ParameterCategory.ENVIRONMENTAL,
                description="Surface albedo"
            )
        }
    
    def test_yaml_serialization(self, sample_parameters):
        """Test YAML format serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "params.yaml"
            
            # Save to YAML
            save_yaml(_schema(sample_parameters), filepath)
            assert filepath.exists()
            
            # Load from YAML
            loaded = load_yaml(filepath)
            assert loaded["parameters"] == sample_parameters
            
            # Check YAML content structure
            with open(filepath) as f:
                content = f.read()
                assert "vcmax25" in content
                assert "albedo" in content
                assert "Farquhar" in content
    
    def test_json_serialization(self, sample_parameters):
        """Test JSON format serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "params.json"
            
            # Save to JSON
            save_json(_schema(sample_parameters), filepath)
            assert filepath.exists()
            
            # Load from JSON
            loaded = load_json(filepath)
            assert loaded["parameters"] == sample_parameters
            
            # Check JSON is valid
            with open(filepath) as f:
                json_data = json.load(f)
                assert "parameters" in json_data
                assert "biosphere.photosynthesis.vcmax25" in json_data["parameters"]
    
    def test_toml_serialization(self, sample_parameters):
        """Test TOML format serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "params.toml"
            
            # TOML has limitations with nested dicts, so we need to flatten
            flattened = {}
            for key, value in sample_parameters.items():
                # Convert dots to underscores for TOML sections
                section_key = key.replace(".", "_")
                flattened[section_key] = value
            
            # Save to TOML
            save_toml(_schema(flattened), filepath)
            assert filepath.exists()
            
            # Load from TOML
            loaded = load_toml(filepath)
            loaded_params = loaded["parameters"]
            for key in flattened:
                assert key in loaded_params
                assert loaded_params[key]["value"] == flattened[key]["value"]
    
    def test_export_function(self, sample_specs):
        """Test the unified export function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to YAML
            yaml_path = Path(tmpdir) / "export.yaml"
            content = export(sample_specs, format="yaml", output_path=yaml_path)
            assert yaml_path.exists()
            assert content is None  # Returns None when writing to file
            
            # Export to string (no output_path)
            yaml_string = export(sample_specs, format="yaml")
            assert isinstance(yaml_string, str)
            assert "vcmax25" in yaml_string
            
            # Export to JSON
            json_path = Path(tmpdir) / "export.json"
            export(sample_specs, format="json", output_path=json_path)
            assert json_path.exists()
            
            # Test invalid format
            with pytest.raises(ValueError, match="SerializationFormat"):
                export(sample_specs, format="invalid")
    
    def test_load_function(self, sample_parameters):
        """Test the unified load function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save parameters in different formats
            yaml_path = Path(tmpdir) / "params.yaml"
            json_path = Path(tmpdir) / "params.json"
            
            save_yaml(_schema(sample_parameters), yaml_path)
            save_json(_schema(sample_parameters), json_path)
            
            # Load with auto-detection
            yaml_loaded = load(yaml_path)
            assert yaml_loaded["parameters"] == sample_parameters
            
            json_loaded = load(json_path)
            assert json_loaded["parameters"] == sample_parameters
            
            # Load with explicit format
            yaml_explicit = load(yaml_path, format="yaml")
            assert yaml_explicit["parameters"] == sample_parameters
            
            # Test loading non-existent file
            with pytest.raises(FileNotFoundError):
                load(Path(tmpdir) / "nonexistent.yaml")
    
    def test_convert_format(self, sample_parameters):
        """Test format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "source.yaml"
            json_path = Path(tmpdir) / "target.json"
            
            # Save as YAML
            save_yaml(_schema(sample_parameters), yaml_path)
            
            # Convert to JSON
            convert_format(yaml_path, json_path, input_format="yaml", output_format="json")
            assert json_path.exists()
            
            # Verify conversion
            json_loaded = load_json(json_path)
            loaded_params = json_loaded["parameters"]
            original_params = sample_parameters
            assert loaded_params == original_params
            
            # Test auto-format detection
            yaml_path2 = Path(tmpdir) / "params.yaml"
            json_path2 = Path(tmpdir) / "converted.json"
            save_yaml(_schema(sample_parameters), yaml_path2)
            
            # Convert without specifying formats (auto-detect)
            convert_format(yaml_path2, json_path2)
            assert json_path2.exists()
            
            json_loaded2 = load_json(json_path2)
            assert "biosphere.photosynthesis.vcmax25" in str(json_loaded2)


class TestTensorSerialization:
    """Test PyTorch tensor serialization."""
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int16, torch.uint8])
    @pytest.mark.parametrize("shape", [(1,), (0, 3), (2, 0, 3)])
    def test_tensor_wire_preserves_dtype_and_empty_dimensions(self, dtype, shape):
        original = torch.full(shape, 2, dtype=dtype)
        restored = deserialize_tensor(serialize_tensor(original))
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape
        assert torch.equal(restored, original)

    @pytest.mark.parametrize("field,value", [
        ("dtype", "invalid_float"), ("shape", [2]),
        ("shape", [True]), ("shape", [-1]), ("requires_grad", "false"),
    ])
    def test_tensor_wire_rejects_invalid_contract(self, field, value):
        encoded = serialize_tensor(torch.tensor([1.0]))
        encoded[field] = value
        with pytest.raises((TypeError, ValueError)):
            deserialize_tensor(encoded)

    @pytest.mark.parametrize("source_shape,declared_shape", [((0, 3), [2, 0]), ((2, 0, 3), [0, 2])])
    def test_empty_tensor_wire_rejects_changed_observable_dimensions(self, source_shape, declared_shape):
        encoded = serialize_tensor(torch.empty(source_shape))
        encoded["shape"] = declared_shape
        with pytest.raises(ValueError, match="shape"):
            deserialize_tensor(encoded)

    def test_serialize_tensor(self):
        """Test tensor serialization to dict."""
        # Regular tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        serialized = serialize_tensor(tensor)
        
        assert "data" in serialized
        assert "dtype" in serialized
        assert "shape" in serialized
        assert "requires_grad" in serialized
        
        assert serialized["data"] == [1.0, 2.0, 3.0]
        # With the global default dtype set to float64 in conftest,
        # the serialized dtype should reflect that default.
        assert serialized["dtype"] == "float64"
        assert serialized["shape"] == [3]
        assert serialized["requires_grad"] == False
        
        # Gradient-enabled tensor
        grad_tensor = torch.tensor([1.0], requires_grad=True)
        grad_serialized = serialize_tensor(grad_tensor)
        assert grad_serialized["requires_grad"] == True
        
        # Multi-dimensional tensor
        multi_tensor = torch.randn(2, 3, 4)
        multi_serialized = serialize_tensor(multi_tensor)
        assert multi_serialized["shape"] == [2, 3, 4]
        assert len(multi_serialized["data"]) == 2  # First dimension
    
    def test_deserialize_tensor(self):
        """Test tensor deserialization from dict."""
        # Simple tensor
        serialized = {
            "data": [1.0, 2.0, 3.0],
            "dtype": "float32",
            "shape": [3],
            "requires_grad": False
        }
        tensor = deserialize_tensor(serialized)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == torch.Size([3])
        assert tensor.dtype == torch.float32
        assert not tensor.requires_grad
        # Explicitly construct the comparison tensor with the same dtype
        # to avoid default-dtype mismatches.
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        
        # With gradient
        grad_serialized = {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "dtype": "float64",
            "shape": [2, 2],
            "requires_grad": True
        }
        grad_tensor = deserialize_tensor(grad_serialized)
        assert grad_tensor.requires_grad
        assert grad_tensor.dtype == torch.float64
        assert grad_tensor.shape == torch.Size([2, 2])
    
    def test_tensor_roundtrip(self):
        """Test tensor serialization roundtrip."""
        # Create various tensors
        tensors = [
            torch.randn(10),
            torch.randn(2, 3, 4),
            torch.tensor([1.0], requires_grad=True),
            torch.zeros(5, dtype=torch.int32),
            torch.ones(3, 3, dtype=torch.float64)
        ]
        
        for original in tensors:
            # Serialize and deserialize
            serialized = serialize_tensor(original)
            restored = deserialize_tensor(serialized)
            
            # Check properties match
            assert restored.shape == original.shape
            assert restored.dtype == original.dtype
            assert restored.requires_grad == original.requires_grad
            
            # Check values match
            assert torch.allclose(
                restored.float(),
                original.float(),
                rtol=1e-5,
                atol=1e-7
            )


class TestParameterSpecSerialization:
    """Test ParameterSpec serialization."""

    def test_parameter_helpers_use_the_complete_spec_contract(self):
        spec = ParameterSpec(
            key="test.complete_spec", value="wet", default="dry",
            dtype="str", allowed_values=["dry", "wet"], units="dimensionless", bounds=(-float("inf"), float("inf")), category="model_parameter",
        )
        restored = deserialize_parameter_spec(serialize_parameter_spec(spec))
        assert restored.default == "dry"
        assert restored.allowed_values == ["dry", "wet"]
        assert restored.validate_value("unknown")

    def test_parameter_spec_serializes_tags_in_canonical_order(self):
        """Unordered tags must have one deterministic serialized form."""
        tags = {
            "alpha",
            "beta",
            "bounds",
            "calibration",
            "delta",
            "epsilon",
            "gamma",
            "scientific",
            "temperature",
            "zeta",
        }
        spec = ParameterSpec(
            key="test.canonical_tags",
            value=1.0,
            tags=tags, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
        )

        assert spec.to_dict()["tags"] == sorted(tags)

    def test_parameter_spec_rejects_non_text_tags(self):
        """Tag metadata must not retain YAML boolean coercions."""
        with pytest.raises(TypeError, match="strings"):
            ParameterSpec(
                key="test.invalid_tags",
                value=1.0,
                tags={"valid", False}, units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter",
            )

    def test_parameter_spec_export(self):
        """Test exporting ParameterSpec objects."""
        specs = {
            "param1": ParameterSpec(
                key="test.param1",
                value=1.0,
                units="m/s",
                bounds=(0.0, 10.0),
                category=ParameterCategory.PHYSICAL,
                description="Test parameter 1", dtype="float32"
            ),
            "param2": ParameterSpec(
                key="test.param2",
                value=torch.tensor([1.0, 2.0]),
                units="dimensionless",
                category=ParameterCategory.COMPUTATIONAL,
                description="Test parameter 2", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
        }
        
        # Export to dict
        exported = {}
        for key, spec in specs.items():
            spec_dict = spec.model_dump()
            # Handle tensor serialization
            if isinstance(spec_dict["value"], torch.Tensor):
                spec_dict["value"] = serialize_tensor(spec_dict["value"])
            exported[spec.key] = spec_dict
        
        # Verify structure
        assert "test.param1" in exported
        assert "test.param2" in exported
        assert exported["test.param1"]["units"] == "m/s"
        assert isinstance(exported["test.param2"]["value"], dict)  # Serialized tensor
    
    def test_parameter_spec_import(self):
        """Test importing ParameterSpec objects."""
        data = {
            "test.param": {
                "key": "test.param",
                "value": 42.0,
                "units": "K",
                "bounds": [200.0, 400.0],
                "dtype": "float32",
                "category": "environmental",
                "description": "Temperature parameter",
                "default": 300.0,
                "requires_grad": False
            }
        }
        
        # Import to ParameterSpec
        spec_data = data["test.param"]
        spec = ParameterSpec(**spec_data)
        
        assert spec.key == "test.param"
        assert spec.value == 42.0
        assert spec.units == "K"
        assert spec.bounds == (200.0, 400.0)
        assert spec.description == "Temperature parameter"


class TestSerializationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_parameters(self):
        """Test serializing empty parameters."""
        empty_params = {}
        
        # Should handle empty dict gracefully  
        yaml_str = export(empty_params, format="yaml")
        # Empty parameters get schema wrapper with timestamp and version
        assert "parameters: {}" in yaml_str
        assert "version:" in yaml_str
        
        json_str = export(empty_params, format="json")
        loaded = json.loads(json_str)
        assert "parameters" in loaded
        assert loaded["parameters"] == {}
    
    def test_special_values(self):
        """Test serializing special values."""
        special_params = {
            "nan_param": {
                "value": float('nan'),
                "units": "dimensionless"
            },
            "inf_param": {
                "value": float('inf'),
                "units": "dimensionless"
            },
            "none_param": {
                "value": None,
                "units": "dimensionless"
            }
        }
        
        # JSON should handle these specially
        json_str = export(special_params, format="json")
        loaded = json.loads(json_str)
        
        # Data is wrapped in "parameters" key after schema migration
        params = loaded.get("parameters", loaded)
        
        # NaN and Inf should be handled specially in JSON
        # Note: NaN in JSON is not standard, so it may be preserved or become null
        nan_value = params["nan_param"]["value"]
        assert nan_value is None or (isinstance(nan_value, float) and (nan_value != nan_value))  # NaN != NaN
        assert params["none_param"]["value"] is None
    
    def test_large_parameters(self):
        """Test serializing large parameter sets."""
        # Create large parameter set
        large_params = {}
        for i in range(1000):
            large_params[f"param_{i}"] = {
                "value": float(i),
                "units": "dimensionless",
                "description": f"Parameter {i}"
            }
        
        # Should handle large sets
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large.json"
            save_json(_schema(large_params), filepath)
            
            loaded = load_json(filepath)
            params = loaded["parameters"]
            params_only = {k: v for k, v in params.items() if k.startswith("param_")}
            assert len(params_only) == 1000
            assert params["param_500"]["value"] == 500.0
    
    def test_malformed_files(self):
        """Test loading malformed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Malformed JSON
            bad_json_path = Path(tmpdir) / "bad.json"
            bad_json_path.write_text("{invalid json}")
            
            with pytest.raises(json.JSONDecodeError):
                load_json(bad_json_path)
            
            # Malformed YAML - unterminated array  
            bad_yaml_path = Path(tmpdir) / "bad.yaml"
            bad_yaml_path.write_text("invalid: [unclosed array")
            
            with pytest.raises(yaml.YAMLError):
                load_yaml(bad_yaml_path)
    
    def test_unicode_handling(self):
        """Test Unicode character handling."""
        unicode_params = {
            "unicode_param": {
                "value": 1.0,
                "units": "渭mol/m虏/s",  # Unicode units
                "description": "Parameter with 渭 (mu) symbol"
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test all formats
            for fmt, save_func, load_func in [
                ("yaml", save_yaml, load_yaml),
                ("json", save_json, load_json)
            ]:
                filepath = Path(tmpdir) / f"unicode.{fmt}"
                save_func(_schema(unicode_params), filepath)
                loaded = load_func(filepath)
                loaded_params = loaded["parameters"]
                assert loaded_params["unicode_param"]["units"] == "渭mol/m虏/s"
                assert "渭" in loaded_params["unicode_param"]["description"]
    
    def test_path_handling(self):
        """Test various path inputs."""
        params = _schema({"test": {"value": 1.0}})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # String path
            str_path = f"{tmpdir}/test.yaml"
            save_yaml(params, str_path)
            assert Path(str_path).exists()
            
            # Path object
            path_obj = Path(tmpdir) / "test2.yaml"
            save_yaml(params, path_obj)
            assert path_obj.exists()
            
            # Nested directories (should create)
            nested_path = Path(tmpdir) / "nested" / "deep" / "test.yaml"
            save_yaml(params, nested_path)
            assert nested_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

