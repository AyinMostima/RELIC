from pathlib import Path

import yaml

from relic.scientific.biosphere.bvoc.megan_model import MEGANModel


def test_bvoc_yaml_functor_metadata_matches_megan_model_interface() -> None:
    yaml_path = Path("relic\\parameters\\database\\relic\\biosphere\\bvoc_parameters.yaml")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    yaml_inputs = set(payload["functor_metadata"]["input_variables"])
    yaml_outputs = set(payload["functor_metadata"]["output_variables"])

    metadata = MEGANModel().metadata

    assert yaml_inputs == set(metadata.input_variables)
    assert yaml_outputs == set(metadata.output_variables)
    yaml_conservation_laws = set(payload["functor_metadata"]["conservation_laws"])
    assert yaml_conservation_laws == set(metadata.conservation_laws)
    assert yaml_conservation_laws == {"carbon_boundary"}
