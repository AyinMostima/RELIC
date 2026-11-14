"""Hydrosphere parameter-wrapper consistency checks."""

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_no_legacy_local_require_wrapper_name_in_snow_modules():
    snowpack_text = _read_text(
        "relic/scientific/hydrosphere/snow/snowpack.py"
    )
    albedo_text = _read_text(
        "relic/scientific/hydrosphere/snow/albedo.py"
    )

    assert "def _require(" not in snowpack_text
    assert "def _require(" not in albedo_text
    assert "def _require_parameter(" not in snowpack_text
    assert "def _require_parameter(" not in albedo_text


def test_target_modules_do_not_define_legacy_require_parameter_wrapper():
    bucket_text = _read_text(
        "relic/scientific/hydrosphere/soil_moisture/bucket.py"
    )
    capillary_text = _read_text(
        "relic/scientific/hydrosphere/transport/capillary_rise.py"
    )
    penman_text = _read_text(
        "relic/scientific/hydrosphere/evapotranspiration/penman_monteith.py"
    )

    assert "def _require_parameter(" not in bucket_text
    assert "def _require_parameter(" not in capillary_text
    assert "def _require_parameter(" not in penman_text
