from pathlib import Path

import pytest

import relic.scientific.isotopes as isotope_module
import relic.scientific.isotopes.tracking as tracking_module
from relic.scientific.isotopes.config import IsotopeConfig
from relic.scientific.isotopes.config.models import Nitrogen15Config
from relic.scientific.isotopes import fractionation as fractionation_module
from relic.scientific.isotopes.fractionation import (
    Carbon13FractionationCalculator,
    FractionationError,
    FractionationParameterError,
    IsotopeFractionationCalculator,
    Nitrogen15FractionationCalculator,
    Oxygen18FractionationCalculator,
    TemperatureError,
)
from relic.scientific.isotopes.fractionation.base import (
    IsotopeFractionationCalculator as BaseIsotopeFractionationCalculator,
    FractionationError as BaseFractionationError,
    FractionationParameterError as BaseFractionationParameterError,
    TemperatureError as BaseTemperatureError,
)


def test_fractionation_base_has_one_canonical_owner() -> None:
    assert IsotopeFractionationCalculator is BaseIsotopeFractionationCalculator
    with pytest.raises(TypeError):
        IsotopeFractionationCalculator("unknown")


def test_fractionation_exception_origin_is_unique() -> None:
    assert FractionationError is BaseFractionationError
    assert fractionation_module.FractionationError is BaseFractionationError
    assert TemperatureError is BaseTemperatureError
    assert FractionationParameterError is BaseFractionationParameterError


def test_species_calculators_are_explicit_subclasses() -> None:
    for calculator_type in (
        Carbon13FractionationCalculator,
        Nitrogen15FractionationCalculator,
        Oxygen18FractionationCalculator,
    ):
        assert issubclass(calculator_type, IsotopeFractionationCalculator)


def test_fractionation_package_has_no_compatibility_api() -> None:
    forbidden = (
        "BaseIsotopeFractionationCalculator",
        "TemperatureOutOfBoundsError",
        "FractionationValidationError",
        "validate_temperature",
        "compute_q10_fractionation",
        "compute_arrhenius_fractionation",
        "build_default_n15_calculator",
    )
    assert not [name for name in forbidden if hasattr(fractionation_module, name)]


def test_old_single_file_facade_contains_no_duplicate_formula() -> None:
    package_path = Path(fractionation_module.__file__).resolve().parent
    old_module_path = package_path.parent / "fractionation.py"
    source = old_module_path.read_text(encoding="utf-8")
    forbidden = (
        "def compute_q10_fractionation",
        "def compute_arrhenius_fractionation",
        "TemperatureOutOfBoundsError =",
        "FractionationValidationError =",
        "IsotopeFractionationCalculator = Nitrogen15FractionationCalculator",
    )
    assert not [token for token in forbidden if token in source]


def test_root_module_reports_runtime_unbound_boundary() -> None:
    info = isotope_module.get_module_info()
    assert info["status"] == "[N] runtime_unbound"
    assert info["runtime_available"] is False
    assert info["enabled_isotopes"] == []
    assert info["output_formats"] == []
    assert set(info["blockers"]) == {"13C", "15N", "18O"}


def test_public_surface_has_one_fail_closed_entry_and_no_runtime_services() -> None:
    config = isotope_module.get_default_config()
    assert type(config) is IsotopeConfig
    assert config.enabled is False
    forbidden = {
        Nitrogen15Config.__name__,
        "MultiIsotopeTracker",
        "IsotopeTrackerService",
        "IsotopeHeartbeatFunctor",
        "IsotopeNetCDFWriter",
        "validate_integration",
        "IsotopeParameterAuditor",
        "IsotopeParameterQualityAssessor",
    }
    for name in forbidden:
        assert name not in isotope_module.__all__
        assert not hasattr(isotope_module, name)


def test_tracking_package_does_not_export_private_state_ledgers() -> None:
    for name in ("IsotopePool", "IsotopeEvent"):
        assert name not in tracking_module.__all__
        assert not hasattr(tracking_module, name)
