from pathlib import Path

import yaml


def test_evapotranspiration_diagnostic_parameter_dimensions_are_physical():
    yaml_path = (
        Path("relic")
        / "parameters"
        / "database"
        / "relic"
        / "hydrosphere"
        / "evapotranspiration.yaml"
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    params = data["parameters"]

    expected_dimensions = {
        "hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure": "pressure",
        "hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance": "velocity",
    }
    for key, expected_dimension in expected_dimensions.items():
        assert key in params, f"Missing parameter entry: {key}"
        assert params[key]["dimension"] == expected_dimension
