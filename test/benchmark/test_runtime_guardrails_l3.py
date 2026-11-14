from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from relic.core.functor import PhysicalFieldFunctor, FunctorMetadata, ProcessCategory
from relic.core.simulator import RELICSimulator
from relic.data.io.writer import DataWriter
from relic.interfaces.web.app import create_web_app


class _MinimalL3Functor(PhysicalFieldFunctor):

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )
    def __init__(self, name: str = "l3_guardrail_functor") -> None:
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.PHYSICAL_FIELD,
            description="L3 guardrail functor.",
            mathematical_expression="guardrail.delta = 0",
            references=["RELIC architecture_overview.md"],
            input_variables=[],
            output_variables=["guardrail.delta"],
        )
        super().__init__(name=name, metadata=metadata, config={})

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"guardrail.delta": torch.tensor(0.0)}


def test_runtime_guardrails_l3(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_web_app(
        config={"security": {"api_auth": {"enabled": False}}},
        enable_cors=False,
        debug=False,
    )
    client = TestClient(app)
    status = client.get("/status")
    assert status.status_code == 200
    assert isinstance(status.json(), dict)

    writer = DataWriter()
    dataset = xr.Dataset(
        {
            "air_temperature": ("time", np.array([20.0], dtype=np.float64)),
            "label": ("time", np.array(["ok"], dtype=object)),
        },
        coords={"time": np.array([0])},
    )
    dataset["air_temperature"].attrs["units"] = "C"
    dataset["time"].attrs.update(
        {"units": "days since 2000-01-01", "calendar": "noleap"}
    )
    output_path = tmp_path / "guardrail.nc"
    write_result = writer.write_dataset(
        dataset,
        output_path,
        format="netcdf4",
        compression="none",
        parallel=False,
        cf_sync=False,
        add_bounds=False,
        auto_chunk=False,
        validate_output=False,
    )
    assert write_result.output_path.exists()
    assert not hasattr(DataWriter, "_apply_unit_conversions")
    assert not hasattr(DataWriter, "_get_cf_standard_units")
    with xr.open_dataset(output_path, decode_times=False) as written:
        assert written["air_temperature"].attrs["units"] == "C"
        assert written["air_temperature"].values.tolist() == [20.0]
        assert written["label"].values.tolist() == ["ok"]

    simulator = RELICSimulator()
    simulator.register_functor(_MinimalL3Functor())
    assert "l3_guardrail_functor" in simulator.functors
