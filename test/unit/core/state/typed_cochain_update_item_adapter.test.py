"""Tests for lowering TypedCochain deltas to StateManager UpdateItem payloads."""

from pathlib import Path

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry


def _write_contract(path: Path) -> Path:
    path.write_text(
        "variable: soil_water_storage\n"
        "title: \"Soil Water Storage\"\n"
        "kind: state\n"
        "dimension:\n"
        "  udunits: \"kg m-2\"\n"
        "  cf_standard_name: soil_water_storage\n"
        "owner: Hydrology\n"
        "aliases: [S_w]\n"
        "incrementable: true\n"
        "accumulates: water.total\n"
        "writers:\n"
        "  tendencies:\n"
        "    - name: SoilWaterFlux\n",
        encoding="utf-8",
    )
    return path


def _typed_delta():
    from relic.computation.operators.dec import CochainSpaceSpec, TypedCochain

    space = CochainSpaceSpec(
        complex_id="soil_column",
        degree=2,
        primality="primal",
        n_dofs=2,
        units="kg m-2 s-1",
        extensivity="extensive",
        physical_domain="hydrosphere",
        semantic_name="soil_water_storage",
        orientation_convention="outward_positive",
    )
    return TypedCochain(
        name="soil_water_storage",
        values=torch.tensor([1.0, 2.0]),
        space=space,
    )


def test_typed_cochain_delta_lowers_to_update_item_and_state_manager_commit(tmp_path: Path):
    from relic.computation.operators.dec import typed_cochain_delta_to_update_item
    from relic.core.state import StateManager, UpdateItem, UpdateMode

    registry = ContractRegistry(strict=False)
    registry.load_contract_file(_write_contract(tmp_path / "soil_water_storage.yaml"))
    cochain = _typed_delta()

    canonical_name, item = typed_cochain_delta_to_update_item(
        cochain,
        source_functor="SoilWaterFlux",
        registry=registry,
        per_timestep=False,
        element="water",
    )

    assert canonical_name == "soil_water_storage"
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.FLUX
    assert item.per_timestep is False
    assert item.unit == "kg m-2 s-1"
    assert item.element == "water"
    assert item.source_functor == "SoilWaterFlux"

    manager = StateManager(schema={}, autoload_contract_registry=False, device="cpu")
    storage_spec = registry.to_state_variable_spec(canonical_name)
    assert storage_spec is not None
    assert storage_spec.units == "kg m-2"
    assert storage_spec.accumulates == "water.total"
    storage_spec.default = torch.zeros(2)
    manager.register_variable(storage_spec)
    manager.apply_updates({canonical_name: item}, dt=2.0, check_conservation=False)

    assert torch.allclose(manager.state_vars["soil_water_storage"], torch.tensor([2.0, 4.0]))


def test_typed_cochain_writer_contract_rejects_unauthorized_writer(tmp_path: Path):
    from relic.computation.operators.dec import typed_cochain_delta_to_update_item

    registry = ContractRegistry(strict=False)
    registry.load_contract_file(_write_contract(tmp_path / "soil_water_storage.yaml"))

    with pytest.raises(PermissionError, match="not authorized"):
        typed_cochain_delta_to_update_item(
            _typed_delta(),
            source_functor="OtherFunctor",
            registry=registry,
            per_timestep=False,
        )


def test_typed_cochain_adapter_resolves_alias_to_canonical_variable(tmp_path: Path):
    from relic.computation.operators.dec import typed_cochain_delta_to_update_item

    registry = ContractRegistry(strict=False)
    registry.load_contract_file(_write_contract(tmp_path / "soil_water_storage.yaml"))
    cochain = _typed_delta()

    canonical_name, item = typed_cochain_delta_to_update_item(
        cochain,
        variable_name="S_w",
        source_functor="SoilWaterFlux",
        registry=registry,
        per_timestep=True,
    )

    assert canonical_name == "soil_water_storage"
    assert item.mode.value == "accumulate"


def test_typed_cochain_adapter_requires_contract_registry():
    from relic.computation.operators.dec import typed_cochain_delta_to_update_item

    with pytest.raises(ValueError, match="registry"):
        typed_cochain_delta_to_update_item(
            _typed_delta(),
            source_functor="UnknownWriter",
            registry=None,
        )


def test_typed_cochain_adapter_rejects_registry_without_writer_contract_api():
    from relic.computation.operators.dec import typed_cochain_delta_to_update_item

    with pytest.raises(TypeError, match="can_increment"):
        typed_cochain_delta_to_update_item(
            _typed_delta(),
            source_functor="UnknownWriter",
            registry=object(),
        )
