import pytest
import torch

from relic.scientific.pedosphere import interface_mapping
from relic.scientific.pedosphere.interface_mapping import get_pedosphere_interface_mapper


LITTER_INPUTS = {
    "leaf_litterfall_carbon": torch.tensor(1.0e-6, dtype=torch.float64),
    "froot_litterfall_carbon": torch.tensor(2.0e-6, dtype=torch.float64),
    "leaf_litterfall_nitrogen": torch.tensor(1.0e-7, dtype=torch.float64),
    "froot_litterfall_nitrogen": torch.tensor(2.0e-7, dtype=torch.float64),
    "lignin_n_ratio": torch.tensor(0.45, dtype=torch.float64),
}

GRAZING_INPUTS = {
    "dung_c_flux": torch.tensor(4.0e-6, dtype=torch.float64),
    "dung_n_flux": torch.tensor(5.0e-7, dtype=torch.float64),
}


def test_ensure_mass_balance_tensor_safe() -> None:
    mapper = get_pedosphere_interface_mapper()
    pools_before = {
        "fast_c": torch.tensor([1.0, 2.0]),
        "slow_c": torch.tensor([0.5, 0.5]),
    }
    pools_after = {
        "fast_c": torch.tensor([1.1, 2.1]),
        "slow_c": torch.tensor([0.5, 0.5]),
    }
    fluxes = {
        "respiration_c": torch.tensor([-0.2, 0.0]),
    }

    errors = mapper.ensure_mass_balance(pools_before, pools_after, fluxes, dt=1.0)

    assert errors == {}


def test_ensure_mass_balance_detects_error() -> None:
    mapper = get_pedosphere_interface_mapper()
    pools_before = {
        "fast_c": torch.tensor([1.0]),
    }
    pools_after = {
        "fast_c": torch.tensor([1.2]),
    }
    fluxes = {
        "respiration_c": torch.tensor([0.0]),
    }

    errors = mapper.ensure_mass_balance(
        pools_before,
        pools_after,
        fluxes,
        dt=1.0,
        tolerance=0.01,
    )

    assert "carbon_balance_error" in errors


def test_scale_mapping_requires_explicit_bridges() -> None:
    mapper = get_pedosphere_interface_mapper()

    fine = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    with pytest.raises(RuntimeError, match="typed chain-map"):
        mapper.upscale(fine)
    with pytest.raises(RuntimeError, match="typed chain-map"):
        mapper.downscale(torch.tensor([2.0, 3.0]))


def test_explicit_scale_bridges_are_used() -> None:
    mapper = get_pedosphere_interface_mapper(
        upscale_bridge=lambda value: value.mean(dim=-1),
        downscale_bridge=lambda value: value.repeat_interleave(2),
    )

    fine = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    coarse = mapper.upscale(fine)
    restored = mapper.downscale(coarse)
    assert torch.allclose(coarse, torch.tensor([2.0, 3.0]))
    assert torch.allclose(restored, torch.tensor([2.0, 2.0, 3.0, 3.0]))


@pytest.mark.parametrize(
    ("calculation", "constant_key", "constant_value"),
    [
        (
            "doc",
            "relic.physical.constants.time.seconds_per_day",
            172800.0,
        ),
        (
            "nitrate",
            "relic.physical.constants.time.seconds_per_hour",
            7200.0,
        ),
    ],
)
def test_leaching_limits_use_physical_constants_ssot(
    monkeypatch: pytest.MonkeyPatch,
    calculation: str,
    constant_key: str,
    constant_value: float,
) -> None:
    original_get_constant = interface_mapping.get_physical_constant
    original_require_parameter = interface_mapping._require_mapper_param
    physical_constant_calls: list[str] = []
    parameter_calls: list[str] = []

    def tracked_get_constant(name: str):
        physical_constant_calls.append(name)
        if name == constant_key:
            return constant_value
        return original_get_constant(name)

    def tracked_require_parameter(mapper, name: str, *, namespace: str):
        parameter_calls.append(f"{namespace}.{name}")
        return original_require_parameter(mapper, name, namespace=namespace)

    monkeypatch.setattr(interface_mapping, "get_physical_constant", tracked_get_constant)
    monkeypatch.setattr(interface_mapping, "_require_mapper_param", tracked_require_parameter)

    state = {"drainage_flux": torch.tensor(100.0), "bulk_density": torch.tensor(1300.0), "soil_depth": torch.tensor(1.0)}
    soil_water_content = torch.tensor(0.3)
    if calculation == "doc":
        pool = torch.tensor(100.0)
        result = interface_mapping.calculate_doc_leaching(
            state,
            pool,
            soil_water_content,
            doc_solubility=0.01,
        )
        expected = pool * 0.01 / constant_value
    else:
        pool = torch.tensor(10.0)
        result = interface_mapping.calculate_nitrate_leaching(
            state,
            pool,
            soil_water_content,
        )
        expected = pool / constant_value

    torch.testing.assert_close(result, expected)
    assert constant_key in physical_constant_calls
    assert constant_key not in parameter_calls


def test_litterfall_mapper_declares_one_explicit_cn_chain_map() -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper(
        dependencies=["gdd_phenology"]
    )

    assert set(mapper.metadata.input_variables) == {
        *LITTER_INPUTS,
        *GRAZING_INPUTS,
    }
    assert mapper.optional_input_variables == set(GRAZING_INPUTS)
    assert mapper.metadata.output_variables == [
        "metabolic_litter_input",
        "structural_litter_input",
        "organic_nitrogen_input",
    ]
    assert mapper.manure_metabolic_fraction == pytest.approx(0.5)
    assert mapper.get_dependencies() == ["gdd_phenology"]


def test_litterfall_mapper_converts_partitions_and_preserves_cn() -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper()
    state = {key: value.clone() for key, value in LITTER_INPUTS.items()}

    output = mapper.forward(state, time=0.0, dt=3600.0)

    assert set(output) == set(mapper.metadata.output_variables)
    metabolic = output["metabolic_litter_input"]
    structural = output["structural_litter_input"]
    organic_n = output["organic_nitrogen_input"]
    for item in output.values():
        assert item.mode.value == "replace"
        assert item.per_timestep is False
        assert item.source_functor == mapper.name
    assert metabolic.unit == "g C m-2 s-1"
    assert structural.unit == "g C m-2 s-1"
    assert organic_n.unit == "g N m-2 s-1"
    assert metabolic.element == "C"
    assert structural.element == "C"
    assert organic_n.element == "N"
    torch.testing.assert_close(
        metabolic.value,
        torch.tensor(2.7e-3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        structural.value,
        torch.tensor(3.0e-4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        organic_n.value,
        torch.tensor(3.0e-4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        metabolic.value + structural.value,
        (state["leaf_litterfall_carbon"] + state["froot_litterfall_carbon"])
        * 1000.0,
    )
    torch.testing.assert_close(
        organic_n.value,
        (
            state["leaf_litterfall_nitrogen"]
            + state["froot_litterfall_nitrogen"]
        )
        * 1000.0,
    )
    assert all(
        torch.equal(state[key], value)
        for key, value in LITTER_INPUTS.items()
    )


def test_litterfall_mapper_routes_grazing_dung_into_existing_cn_ports() -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper(
        dependencies=["gdd_phenology", "grazing_management"]
    )
    state = {
        **{key: value.clone() for key, value in LITTER_INPUTS.items()},
        **{key: value.clone() for key, value in GRAZING_INPUTS.items()},
    }

    output = mapper.forward(state, time=0.0, dt=3600.0)

    assert not set(GRAZING_INPUTS).intersection(
        mapper.optional_input_variables
    )
    torch.testing.assert_close(
        output["metabolic_litter_input"].value,
        torch.tensor(4.7e-3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output["structural_litter_input"].value,
        torch.tensor(2.3e-3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        output["organic_nitrogen_input"].value,
        torch.tensor(8.0e-4, dtype=torch.float64),
    )
    residuals = mapper.check_conservation_laws(state, output)
    assert residuals == {"carbon": 0.0, "nitrogen": 0.0}
    assert all(
        torch.equal(state[key], value)
        for key, value in {**LITTER_INPUTS, **GRAZING_INPUTS}.items()
    )


@pytest.mark.parametrize("missing_key", sorted(GRAZING_INPUTS))
def test_litterfall_mapper_requires_selected_grazing_dung_ports(
    missing_key: str,
) -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper(
        dependencies=["gdd_phenology", "grazing_management"]
    )
    state = {key: value.clone() for key, value in LITTER_INPUTS.items()}
    provided_key = next(key for key in GRAZING_INPUTS if key != missing_key)
    state[provided_key] = GRAZING_INPUTS[provided_key].clone()

    with pytest.raises(KeyError, match=missing_key):
        mapper.forward(state, time=0.0, dt=3600.0)


def test_litterfall_mapper_rejects_undeclared_grazing_dung_boundary() -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper(
        dependencies=["gdd_phenology"]
    )
    state = {
        **{key: value.clone() for key, value in LITTER_INPUTS.items()},
        **{key: value.clone() for key, value in GRAZING_INPUTS.items()},
    }

    with pytest.raises(
        ValueError,
        match="requires the grazing_management dependency",
    ):
        mapper.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("missing_key", sorted(LITTER_INPUTS))
def test_litterfall_mapper_requires_every_canonical_input(missing_key: str) -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper()
    state = {key: value.clone() for key, value in LITTER_INPUTS.items()}
    del state[missing_key]

    with pytest.raises(KeyError, match=missing_key):
        mapper.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("leaf_litterfall_carbon", torch.tensor(-1.0e-6)),
        ("froot_litterfall_nitrogen", torch.tensor(float("nan"))),
        ("lignin_n_ratio", torch.tensor(float("inf"))),
    ],
)
def test_litterfall_mapper_rejects_invalid_inputs(
    key: str,
    value: torch.Tensor,
) -> None:
    mapper = interface_mapping.get_litterfall_interface_mapper()
    state = {name: item.clone() for name, item in LITTER_INPUTS.items()}
    state[key] = value

    with pytest.raises(ValueError, match=key):
        mapper.forward(state, time=0.0, dt=3600.0)

# Round 35: the leaching helpers read the soil geometry forcing from the state
# (bulk_density, soil_depth); no bulk_density_default and no 1 m soil depth.
@pytest.mark.parametrize("missing", ["bulk_density", "soil_depth"])
def test_leaching_helpers_require_the_soil_geometry_inputs(missing):
    state = {
        "drainage_flux": torch.tensor(100.0),
        "bulk_density": torch.tensor(1300.0),
        "soil_depth": torch.tensor(1.0),
    }
    del state[missing]
    with pytest.raises(ValueError, match=missing):
        interface_mapping.calculate_nitrate_leaching(state, torch.tensor(10.0), torch.tensor(0.3))
    with pytest.raises(ValueError, match=missing):
        interface_mapping.calculate_doc_leaching(
            state, torch.tensor(100.0), torch.tensor(0.3), doc_solubility=0.01
        )


def test_interface_mapper_declares_no_bulk_density_default():
    mapper = interface_mapping.PedosphereInterfaceMapper()
    assert "pedosphere.physical.bulk_density_default" not in mapper.metadata.parameters


def test_leaching_readers_use_the_canonical_drainage_flux():
    """MA-21 item 207: drainage_input was a writerless reader-side name of drainage_flux, the
    downward water mass flux [kg m-2 s-1] leaving the bottom of the soil column (grid cell,
    ground area, non-negative) written by the selected soil water model. Nitrate leaching
    J = (q / rho_w) C f and the hydrosphere-to-pedosphere port read that flux, so both read
    drainage_flux; drainage_input is neither a contract nor a catalog name nor an alias."""
    from relic.contracts import load_contract_registry
    from relic.scientific.hydrosphere.interface_mapping import hydrosphere_mapper
    from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess

    registry = load_contract_registry()
    assert registry.get_contract("drainage_input") is None
    assert not registry.is_alias("drainage_input")
    drainage = registry.get_contract("drainage_flux")
    assert drainage.udunits == "kg m-2 s-1" and drainage.raw["numerics"]["lower_bound"] == 0.0
    assert "nitrogen_leaching" in drainage.raw["description"]
    assert "drainage_input" not in hydrosphere_mapper.get_all_names()["standard"]
    mapper = get_pedosphere_interface_mapper()
    for inputs in (mapper.metadata.input_variables, NitrogenLeachingProcess().metadata.input_variables):
        assert "drainage_flux" in inputs and "drainage_input" not in inputs
    flux = torch.tensor([2.0e-5])
    state = {"drainage_flux": flux, "subsurface_runoff_flux": flux, "surface_runoff_flux": flux}
    assert torch.equal(interface_mapping.get_leaching_flux(state), flux)
    with pytest.raises(ValueError, match="drainage_flux"):
        interface_mapping.get_leaching_flux({"drainage_input": flux})
