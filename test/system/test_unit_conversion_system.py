"""
L3 System Smoke Tests for Explicit DataWriter Unit Metadata

This suite validates that DataWriter preserves caller-owned values and unit
metadata. Unit conversion is an explicit upstream operation, not a writer side
effect.

References:
- CF-1.8 conventions for standard_name and units
"""

from pathlib import Path
import tempfile

import numpy as np
import pytest
import xarray as xr

from relic.data.io.writer import DataWriter


@pytest.fixture
def project_tmp_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
        dir=project_root,
        prefix=".unit-conversion-system-",
    ) as tmp_dir:
        yield Path(tmp_dir)


def test_datawriter_preserves_explicit_units_and_values(
    project_tmp_path: Path,
) -> None:
    out_path = project_tmp_path / "forcing.nc"

    ds = xr.Dataset(
        data_vars={
            "temperature": (["time"], np.array([0.0, 10.0], dtype=np.float32)),
            "precipitation": (["time"], np.array([0.001, 0.0], dtype=np.float32)),
            "pressure": (["time"], np.array([1013.25, 1010.00], dtype=np.float32)),
        },
        coords={"time": np.array([0, 1], dtype=np.int32)},
    )

    ds["time"].attrs.update(
        {"units": "days since 2000-01-01", "calendar": "noleap"}
    )
    ds["temperature"].attrs["units"] = "degree_Celsius"
    ds["temperature"].attrs["standard_name"] = "air_temperature"

    ds["precipitation"].attrs["units"] = "kg m-2 s-1"
    ds["precipitation"].attrs["standard_name"] = "precipitation_flux"

    ds["pressure"].attrs["units"] = "hPa"
    ds["pressure"].attrs["standard_name"] = "air_pressure"

    writer = DataWriter()
    result = writer.write_dataset(
        ds,
        out_path,
        format="netcdf4",
        compression="none",
        parallel=False,
        validate_output=False,
    )
    assert result.output_path.exists()

    ds_out = xr.open_dataset(out_path, decode_times=False)
    try:
        assert ds_out["temperature"].attrs.get("units") == "degree_Celsius"
        assert ds_out["precipitation"].attrs.get("units") == "kg m-2 s-1"
        assert ds_out["pressure"].attrs.get("units") == "hPa"

        temp_vals = ds_out["temperature"].values.astype(np.float64)
        precip_vals = ds_out["precipitation"].values.astype(np.float64)
        press_vals = ds_out["pressure"].values.astype(np.float64)

        assert np.allclose(temp_vals, np.array([0.0, 10.0]), rtol=0.0, atol=1e-6)
        assert np.allclose(press_vals, np.array([1013.25, 1010.0]), rtol=0.0, atol=1e-6)
        assert np.allclose(
            precip_vals,
            np.array([0.001, 0.0]),
            rtol=0.0,
            atol=1e-9,
        )
    finally:
        ds_out.close()
