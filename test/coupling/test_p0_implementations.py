"""
L1 Unit Test: Canonical coupling implementations availability and adapter metadata

Ensures non-retired scale coupling functors and the WRF external adapter are
importable without emitting historical deprecation warnings, and that required
metadata (references, category, signatures) is present.
"""

import inspect
from pathlib import Path

import pytest

from relic.core.functor import ProcessCategory
from relic.coupling.scales.temporal.temporal_scaling import TemporalScaling
from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
from relic.coupling.external.atmospheric_models.wrf_adapter import WRFAdapter, WRFConfig


class TestTemporalScalingCanonical:
    """Verify canonical TemporalScaling functor is available without historical adapters."""

    def test_temporal_scaling_exists(self):
        """TemporalScaling class should be importable."""
        assert TemporalScaling is not None, "TemporalScaling should be importable"
        assert inspect.isclass(TemporalScaling), "TemporalScaling should be a class"

    def test_temporal_scaling_instantiation(self):
        """TemporalScaling can be instantiated with default parameters."""
        functor = TemporalScaling(name="test_temporal", method="mean", source_scale="source", target_scale="target")

        assert functor is not None, "TemporalScaling should instantiate successfully"
        assert functor.name == "test_temporal"
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING
        assert functor.metadata.references, "TemporalScaling should declare scientific references"

    def test_temporal_scaling_forward_signature(self):
        """TemporalScaling forward() exposes state/time/dt without historical args."""
        functor = TemporalScaling(name="test_temporal", method="mean", source_scale="source", target_scale="target")

        assert hasattr(functor, "forward"), "TemporalScaling should have forward()"
        sig = inspect.signature(functor.forward)
        assert "state" in sig.parameters, "forward() should have 'state' parameter"
        assert "time" in sig.parameters, "forward() should have 'time' parameter"
        assert "dt" in sig.parameters, "forward() should have 'dt' parameter"


class TestSpatialScalingNohistoricalWeights:
    """Verify SpatialScaling forward signature avoids historical weights parameter."""

    def test_spatial_scaling_exists(self):
        """SpatialScaling class should be importable."""
        assert SpatialScaling is not None, "SpatialScaling should be importable"
        assert inspect.isclass(SpatialScaling), "SpatialScaling should be a class"

    def test_spatial_scaling_instantiation(self):
        """SpatialScaling can be instantiated with minimal configuration."""
        functor = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
        )

        assert functor is not None, "SpatialScaling should instantiate successfully"
        assert functor.metadata.references, "SpatialScaling should declare scientific references"

    def test_spatial_scaling_forward_no_weights_param(self):
        """
        Verify forward() does not expose a historical 'weights' parameter.
        """
        functor = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
        )

        assert hasattr(functor, "forward"), "SpatialScaling should have forward() method"

        sig = inspect.signature(functor.forward)
        param_names = list(sig.parameters.keys())

        assert "weights" not in param_names, "forward() should not have explicit 'weights' parameter"
        assert "state" in param_names, "forward() should have 'state' parameter"
        assert "time" in param_names, "forward() should have 'time' parameter"
        assert "dt" in param_names, "forward() should have 'dt' parameter"


class TestWRFAdapterImplementation:
    """Verify WRF adapter implementation exists and metadata is populated."""

    def test_wrf_adapter_exists(self):
        """Verify WRFAdapter class exists and is importable."""
        assert WRFAdapter is not None, "WRFAdapter should be importable"
        assert inspect.isclass(WRFAdapter), "WRFAdapter should be a class"

    def test_wrf_config_exists(self):
        """Verify WRFConfig dataclass exists."""
        assert WRFConfig is not None, "WRFConfig should be importable"

        import dataclasses

        assert dataclasses.is_dataclass(WRFConfig), "WRFConfig should be a dataclass"

    def test_wrf_adapter_instantiation(self):
        """Verify WRFAdapter can be instantiated with minimal config and references."""
        config = WRFConfig(root_path="./test_wrf_path")

        try:
            adapter = WRFAdapter(config=config)

            assert adapter is not None, "WRFAdapter should instantiate successfully"
            assert adapter.config == config, "WRFAdapter should store config"
            assert adapter.metadata.references, "WRFAdapter should declare scientific references"
            assert adapter.metadata.category == ProcessCategory.SCALE_COUPLING
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"WRFAdapter instantiation failed: {exc}")

    def test_wrf_adapter_key_methods_exist(self):
        """Verify WRFAdapter has all required methods."""
        config = WRFConfig(root_path="./test_wrf_path")
        adapter = WRFAdapter(config=config)

        required_methods = [
            "discover",
            "open",
            "read",
            "close",
            "validate_data",
            "transform_to_internal",
            "get_dependencies",
            "forward",
        ]

        for method_name in required_methods:
            assert hasattr(adapter, method_name), f"WRFAdapter should have {method_name}() method"
            assert callable(getattr(adapter, method_name)), f"{method_name}() should be callable"

    def test_wrf_adapter_forward_signature(self):
        """Verify forward() method signature for external coupling."""
        config = WRFConfig(root_path="./test_wrf_path")
        adapter = WRFAdapter(config=config)

        assert hasattr(adapter, "forward"), "WRFAdapter should have forward() method"

        sig = inspect.signature(adapter.forward)
        assert "state" in sig.parameters, "forward() should have 'state' parameter"
        assert "time" in sig.parameters, "forward() should have 'time' parameter"
        assert "dt" in sig.parameters, "forward() should have 'dt' parameter"

    def test_wrf_adapter_file_location(self):
        """Verify WRF adapter is in correct location."""
        import relic.coupling.external.atmospheric_models.wrf_adapter as wrf_module

        module_path = Path(wrf_module.__file__)

        assert module_path.exists(), "WRF adapter file should exist"
        assert "atmospheric_models" in str(module_path), "WRF adapter should be in atmospheric_models directory"
        assert module_path.name == "wrf_adapter.py", "File should be named wrf_adapter.py"


class TestCanonicalImplementationsSmoke:
    """Comprehensive verification that canonical coupling implementations are callable."""

    def test_all_canonical_implementations_exist(self):
        """Verify canonical TemporalScaling, SpatialScaling, and WRFAdapter can be instantiated."""
        temporal_functor = TemporalScaling(name="test_temp", method="mean", source_scale="source", target_scale="target")
        assert temporal_functor is not None, "TemporalScaling should instantiate"
        assert temporal_functor.metadata.references

        spatial_functor = SpatialScaling(
            source_scale="patch",
            target_scale="hillslope",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
        )
        assert spatial_functor is not None, "SpatialScaling should instantiate"
        assert spatial_functor.metadata.references

        wrf_config = WRFConfig(root_path="./test_path")
        wrf_adapter = WRFAdapter(config=wrf_config)
        assert wrf_adapter is not None, "WRFAdapter should instantiate"
        assert wrf_adapter.metadata.references

    def test_all_canonical_forward_callable(self):
        """Verify forward() is callable for all canonical implementations."""
        implementations = [
            ("TemporalScaling", TemporalScaling(name="test1", method="mean", source_scale="source", target_scale="target")),
            (
                "SpatialScaling",
                SpatialScaling(
                    source_scale="patch",
                    target_scale="hillslope",
                    source_resolution=(1.0, 1.0),
                    target_resolution=(2.0, 2.0),
                    method="conservative",
                ),
            ),
            ("WRFAdapter", WRFAdapter(config=WRFConfig(root_path="./test_path"))),
        ]

        for impl_name, impl_instance in implementations:
            assert hasattr(impl_instance, "forward"), f"{impl_name} should have forward() method"
            assert callable(impl_instance.forward), f"{impl_name}.forward() should be callable"


