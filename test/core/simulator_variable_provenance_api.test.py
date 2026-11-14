
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateMode


def test_get_variable_provenance_with_var_name_returns_list() -> None:
    sim = RELICSimulator({"conservation_checks": False})
    sim._step_update_sources = {
        "soil_moisture": ("bucket_soil_moisture", UpdateMode.REPLACE),
    }

    records = sim.get_variable_provenance("soil_moisture")

    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["functor"] == "bucket_soil_moisture"
    assert "REPLACE" in records[0]["mode"]
    assert records[0]["step"] is None


def test_get_variable_provenance_without_var_name_returns_mapping() -> None:
    sim = RELICSimulator({"conservation_checks": False})
    sim._step_update_sources = {
        "soil_moisture": ("bucket_soil_moisture", UpdateMode.REPLACE),
    }
    sim._step_provenance_detail = {
        "evapotranspiration_flux": [
            ("penman_monteith_et", UpdateMode.ACCUMULATE, 3),
            ("surface_runoff", UpdateMode.ACCUMULATE, 4),
        ]
    }

    records = sim.get_variable_provenance()

    assert isinstance(records, dict)
    assert "soil_moisture" in records
    assert "evapotranspiration_flux" in records
    assert records["evapotranspiration_flux"][0]["functor"] == "penman_monteith_et"
    assert records["evapotranspiration_flux"][0]["step"] == 3


def test_provenance_query_exposes_eviction_without_changing_default_shape() -> None:
    sim = RELICSimulator({"conservation_checks": False})
    sim._step_provenance_detail = {
        "soil_moisture": [("bucket", UpdateMode.REPLACE, 9)],
        "temperature": [("forcing", UpdateMode.REPLACE, 9)],
    }
    sim._provenance_evicted_counts = {"soil_moisture": 8}
    record = sim.get_variable_provenance("soil_moisture", include_retention=True)
    assert record["records"] == sim.get_variable_provenance("soil_moisture")
    assert record["evicted_records"] == 8
    all_records = sim.get_variable_provenance(include_retention=True)
    assert all_records["records"] == sim.get_variable_provenance()
    assert all_records["evicted_records"] == {"soil_moisture": 8, "temperature": 0}
    assert sim.get_variable_provenance("missing", include_retention=True) == {
        "records": [], "evicted_records": 0,
    }
    record["records"].clear()
    all_records["evicted_records"]["soil_moisture"] = 0
    assert sim._provenance_evicted_counts["soil_moisture"] == 8
    assert len(sim._step_provenance_detail["soil_moisture"]) == 1