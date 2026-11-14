import warnings

from relic.scientific.isotopes.config import get_default_isotope_config


def test_isotope_config_construction_emits_no_unit_conversion_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_default_isotope_config()
    unit_warnings = [
        str(item.message)
        for item in caught
        if "unit" in str(item.message).lower() and "conversion" in str(item.message).lower()
    ]
    assert not unit_warnings
