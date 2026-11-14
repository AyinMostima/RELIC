"""G12-MAP: executable chain-map contracts on declared cochain spaces.

References: Hatcher (2002) ch. 2; Desbrun et al. (2005); Parton et al.
(1987) for the litterfall partition consumer.
Mathematical expression: y = M x; (M2 o M1) = M2 M1; C_t M = C_s;
closeness d(M1, M2) <= eps is a distance bound, not an equivalence.
Parameter meanings: M linear chain map, C_s/C_t source and target ledgers.
Units: unit_factor carries the coefficient-unit conversion (kg -> g = 1000).
Domain/constraints: finite declared spaces; shape and space mismatches are refused.
Sign convention: as the orientation conventions of the spaces.
"""

from __future__ import annotations

import pytest
import torch

from relic.computation.operators.dec import CochainSpaceSpec, LinearChainMap, TypedCochain, close_within
from relic.scientific.pedosphere.interface_mapping import LitterfallInterfaceMapper

F64 = torch.float64


def _space(name: str, n: int, units: str) -> CochainSpaceSpec:
    return CochainSpaceSpec(
        complex_id="litter_ports",
        degree=0,
        primality="primal",
        n_dofs=n,
        units=units,
        extensivity="extensive",
        physical_domain="pedosphere",
        semantic_name=name,
        orientation_convention="positive_into_soil",
        boundary_role="port",
    )


class TestContract:
    def test_identity_composition_and_shape_refusal(self):
        a = _space("a", 3, "kg m-2 s-1")
        b = _space("b", 2, "kg m-2 s-1")
        identity = LinearChainMap.identity(a)
        m = LinearChainMap(name="m", source=a, target=b, matrix=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=F64), unit_factor=1.0)
        assert m.compose(identity).equals(m)
        x = TypedCochain(name="x", values=torch.tensor([1.0, 2.0, 3.0], dtype=F64), space=a)
        assert m.apply(x).values.tolist() == [3.0, 3.0]
        assert m.apply(identity.apply(x)).values.tolist() == [3.0, 3.0]
        with pytest.raises(ValueError, match="does not match"):
            LinearChainMap(name="bad", source=a, target=b, matrix=torch.eye(3, dtype=F64), unit_factor=1.0)
        with pytest.raises(ValueError, match="declared source space"):
            m.apply(TypedCochain(name="y", values=torch.tensor([1.0, 2.0], dtype=F64), space=b))
        with pytest.raises(ValueError, match="composition requires"):
            m.compose(m)

    def test_ledger_preservation_with_unit_factor(self):
        a = _space("a", 2, "kg m-2 s-1")
        b = _space("b", 2, "g m-2 s-1")
        m = LinearChainMap(name="kg_to_g", source=a, target=b, matrix=torch.eye(2, dtype=F64), unit_factor=1000.0)
        source_ledger = torch.ones(1, 2, dtype=F64)
        target_ledger = torch.ones(1, 2, dtype=F64) / 1000.0
        ok, residual = m.preserves_ledger(source_ledger, target_ledger)
        assert ok and residual == pytest.approx(0.0)
        leaky = LinearChainMap(name="leaky", source=a, target=b, matrix=torch.tensor([[1.0, 0.0], [0.0, 0.9]], dtype=F64), unit_factor=1000.0)
        ok, residual = leaky.preserves_ledger(source_ledger, target_ledger)
        assert not ok and residual == pytest.approx(0.1)

    def test_closeness_is_not_transitive_and_error_bound_holds(self):
        a = _space("a", 1, "1")
        m1 = LinearChainMap(name="m1", source=a, target=a, matrix=torch.tensor([[1.0]], dtype=F64), unit_factor=1.0)
        m2 = LinearChainMap(name="m2", source=a, target=a, matrix=torch.tensor([[1.6]], dtype=F64), unit_factor=1.0)
        m3 = LinearChainMap(name="m3", source=a, target=a, matrix=torch.tensor([[2.2]], dtype=F64), unit_factor=1.0)
        eps = 0.7
        assert close_within(m1, m2, eps) and close_within(m2, m3, eps)
        assert not close_within(m1, m3, eps)
        assert not m1.equals(m2) and m1.equals(LinearChainMap.identity(a, name="m1"))
        perturbation = torch.tensor([0.5], dtype=F64)
        assert m3.error_bound(perturbation) == pytest.approx(1.1)


class TestLedgerToleranceParameter:
    """preserves_ledger defaults to ParameterManager key validation.iphs.chain_map_ledger_tolerance."""

    KEY = "validation.iphs.chain_map_ledger_tolerance"

    def test_default_resolves_from_the_parameter_database_and_changes_the_check(self):
        # C_t (u M) - C_s = (0, 5e-13) for M = I, u = 1, C_s = (1, 1), C_t = (1, 1 + 5e-13).
        from pathlib import Path

        import yaml

        from relic.parameters import get_global_manager

        path = (
            Path(__file__).resolve().parents[5]
            / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
        )
        entry = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][self.KEY]
        for metadata_field in ("value", "units", "dtype", "bounds", "category", "description",
                               "references", "tags", "requires_grad", "provenance"):
            assert metadata_field in entry, metadata_field
        manager = get_global_manager()
        original = manager.get(self.KEY, required=True)
        assert original == pytest.approx(float(entry["value"]))
        chain = LinearChainMap.identity(_space("x", 2, "kg"))
        source = torch.tensor([[1.0, 1.0]], dtype=F64)
        target = torch.tensor([[1.0, 1.0 + 5.0e-13]], dtype=F64)
        preserved, residual = chain.preserves_ledger(source, target)
        assert preserved is True
        assert residual == pytest.approx(5.0e-13, rel=1.0e-3)
        assert chain.preserves_ledger(source, target, tolerance=1.0e-14)[0] is False
        try:
            manager.set_value(self.KEY, 1.0e-14)
            assert chain.preserves_ledger(source, target)[0] is False
        finally:
            manager.set_value(self.KEY, original)
        assert manager.get(self.KEY, required=True) == original


class TestLitterfallConsumer:
    """The litterfall C/N partition is a linear chain map for a fixed lignin ratio."""

    def _outputs(self, mapper, carbon_leaf, carbon_root, nitrogen_leaf, nitrogen_root):
        state = {
            "leaf_litterfall_carbon": torch.tensor([carbon_leaf], dtype=F64),
            "froot_litterfall_carbon": torch.tensor([carbon_root], dtype=F64),
            "leaf_litterfall_nitrogen": torch.tensor([nitrogen_leaf], dtype=F64),
            "froot_litterfall_nitrogen": torch.tensor([nitrogen_root], dtype=F64),
            "lignin_n_ratio": torch.tensor(0.3, dtype=F64),
        }
        outputs = mapper.forward(state, time=0.0, dt=1.0)
        return torch.tensor([float(outputs[k].value.reshape(-1)[0]) for k in ("metabolic_litter_input", "structural_litter_input", "organic_nitrogen_input")], dtype=F64)

    def test_mapper_is_the_declared_chain_map_and_preserves_the_element_ledgers(self):
        mapper = LitterfallInterfaceMapper(name="litterfall_interface_mapper")
        # Probe the columns with unit basis inputs (kg element m-2 s-1).
        columns = [self._outputs(mapper, *basis) for basis in ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))]
        matrix = torch.stack(columns, dim=1) / 1000.0
        source = _space("vegetation_litter", 4, "kg m-2 s-1")
        target = _space("soil_litter_inputs", 3, "g m-2 s-1")
        chain_map = LinearChainMap(name="litterfall_partition", source=source, target=target, matrix=matrix, unit_factor=1000.0)
        # Linearity on a random state: the mapper output equals M x.
        x = torch.tensor([0.4, 0.25, 0.02, 0.01], dtype=F64)
        expected = chain_map.apply(TypedCochain(name="x", values=x, space=source)).values
        actual = self._outputs(mapper, *x.tolist())
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        # Element ledgers: carbon (columns 0, 1 -> rows 0, 1) and nitrogen (2, 3 -> 2).
        source_ledger = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=F64)
        target_ledger = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=F64) / 1000.0
        ok, residual = chain_map.preserves_ledger(source_ledger, target_ledger, tolerance=1e-9)
        assert ok, residual
        # The metabolic fraction is a genuine partition (non-identity map).
        f_met = float(matrix[0, 0])
        assert 0.0 < f_met < 1.0 and float(matrix[1, 0]) == pytest.approx(1.0 - f_met)
        # Composition with the source identity leaves the declared map unchanged.
        assert chain_map.compose(LinearChainMap.identity(source)).equals(chain_map)


class TestRound5DeclaredUnitFactor:
    """Round 5: the unit factor of a chain map is declared (no 1.0 default)."""

    def test_unit_factor_is_required(self):
        a = _space("a", 2, "kg m-2 s-1")
        with pytest.raises(TypeError, match="unit_factor"):
            LinearChainMap(name="m", source=a, target=a, matrix=torch.eye(2, dtype=F64))

    def test_identity_declares_the_exact_unit_factor_one(self):
        a = _space("a", 2, "kg m-2 s-1")
        assert LinearChainMap.identity(a).unit_factor == 1.0
