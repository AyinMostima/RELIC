import importlib
from pathlib import Path

import pytest

import relic.scientific.isotopes.parameter_namespace_mapping as parameter_mapping_module
import relic.scientific.isotopes.registry as registry_module
import relic.scientific.isotopes.tracker_service as tracker_service_module


def test_private_multi_isotope_runtimes_are_absent() -> None:
    modules = (
        "multi_isotope_tracker",
        "multi_isotope_pools",
    )
    for module_name in modules:
        module_path = Path(
            f"relic/scientific/isotopes/tracking/{module_name}.py"
        )
        assert not module_path.exists()
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(
                f"relic.scientific.isotopes.tracking.{module_name}"
            )

    config_source = Path(
        "relic/scientific/isotopes/config/multi_isotope_models.py"
    ).read_text(encoding="utf-8")
    assert "multi_isotope_tracker" not in config_source
    assert "MultiIsotopeConfig" not in config_source


def test_parameter_access_has_no_legacy_short_name_aliases() -> None:
    source = Path(parameter_mapping_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "backward compatibility",
        '"N15_standard"',
        '"C13_standard"',
        '"O18_standard"',
        "_build_namespace_mapping",
    )
    assert not [token for token in forbidden if token in source]


def test_registry_has_no_dead_parallel_configuration_constants() -> None:
    source = Path(registry_module.__file__).read_text(encoding="utf-8")
    assert "INTEGRATION_TIMEOUT" not in source
    assert "DEFAULT_COMPRESSION_LEVEL" not in source
    assert "default configuration for 15N tracking" not in source


def test_isotope_runtime_uses_only_the_functor_lifecycle() -> None:
    source = (
        Path(registry_module.__file__).read_text(encoding="utf-8")
        + Path(tracker_service_module.__file__).read_text(encoding="utf-8")
    )
    forbidden = (
        "add_conservation_check",
        "register_with_state_manager",
        "on_apply_updates",
        "process_conservation_callback",
        "create_isotope_tracker_service",
        "callback_registered",
    )
    assert not [token for token in forbidden if token in source]
