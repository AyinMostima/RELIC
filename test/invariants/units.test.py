"""I2 unit consistency invariant tests with PBT.

References:
- UDUNITS / CF unit-conversion principle for reversible compatible units.
- RELIC UnitConverter implementation: relic.data.standards.unit_converter.

Mathematical expressions:
- Round-trip: convert(convert(x, u_src, u_dst), u_dst, u_src) = x
- Monotonicity for linear transforms: x1 < x2 => f(x1) < f(x2)

Parameter meanings:
- x, x1, x2: scalar physical values in source units.
- u_src, u_dst: unit strings supported by UnitConverter.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from relic.data.standards.unit_converter import ConversionContext, UnitConverter
from relic.utils.physical_constants import get_physical_constant


CONVERTIBLE_CASES = [
    ("generic", "K", "C", ConversionContext()),
    ("generic", "kg", "g", ConversionContext()),
    ("generic", "Pa", "kPa", ConversionContext()),
    ("generic", "day", "s", ConversionContext()),
    (
        "precipitation_flux",
        "kg m-2 s-1",
        "mm day-1",
        ConversionContext(
            water_density=get_physical_constant("water_density")
        ),
    ),
]


def _convert(
    x: float,
    variable_name: str,
    source_unit: str,
    target_unit: str,
    context: ConversionContext,
) -> float:
    """Convert a scalar using its declared variable semantics."""
    converter = UnitConverter()
    value = converter.convert(
        x,
        variable_name=variable_name,
        source_units=source_unit,
        target_units=target_unit,
        context=context,
    )
    return float(value)


@pytest.mark.property
class TestUnitConsistencyPBT:
    """I2 properties for unit conversion behavior."""

    @given(
        x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        pair_index=st.integers(min_value=0, max_value=len(CONVERTIBLE_CASES) - 1),
    )
    @settings(max_examples=30, deadline=None)
    def test_round_trip(self, x: float, pair_index: int) -> None:
        """Round-trip conversion should recover the original value."""
        variable_name, source_unit, target_unit, context = CONVERTIBLE_CASES[pair_index]
        y = _convert(x, variable_name, source_unit, target_unit, context)
        x_back = _convert(y, variable_name, target_unit, source_unit, context)
        tol = max(2e-5, 1e-5 * abs(x))
        assert abs(x_back - x) <= tol

    @given(
        x1=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
        x2=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
        pair_index=st.integers(min_value=0, max_value=len(CONVERTIBLE_CASES) - 1),
    )
    @settings(max_examples=30, deadline=None)
    def test_monotonicity(self, x1: float, x2: float, pair_index: int) -> None:
        """Linear-compatible conversions should preserve ordering."""
        assume(x1 < x2)
        assume(abs(x2 - x1) > 1e-9)
        variable_name, source_unit, target_unit, context = CONVERTIBLE_CASES[pair_index]
        y1 = _convert(x1, variable_name, source_unit, target_unit, context)
        y2 = _convert(x2, variable_name, source_unit, target_unit, context)
        assert y1 <= y2 + 1e-9


def _unit_conversion_literal_keys() -> list:
    """Return the UNIT_CONVERSIONS dict-literal keys in source order via the AST.

    The runtime dict cannot reveal duplicated literal keys because the later entry
    silently overwrites the earlier one; the source literal is the only place where
    a duplicate can be detected.
    """
    import ast
    import inspect

    source = inspect.getsource(UnitConverter)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "UNIT_CONVERSIONS" for target in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return [ast.literal_eval(key) for key in node.value.keys if key is not None]
    raise AssertionError("UNIT_CONVERSIONS literal not found")


def test_unit_conversion_table_has_no_duplicate_literal_keys() -> None:
    """Every (variable, source, target) key appears exactly once in the source literal."""
    keys = _unit_conversion_literal_keys()
    seen: dict = {}
    duplicates = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        seen[key] = True
    assert not duplicates, f"duplicated UNIT_CONVERSIONS keys: {duplicates}"
    assert len(keys) == len(UnitConverter.UNIT_CONVERSIONS)


@pytest.mark.parametrize(
    ("variable_name", "source_unit", "target_unit", "method_name"),
    [
        ("carbon_flux", "g C m-2 day-1", "kg C m-2 s-1", "carbon_flux_g_day_to_kg_s"),
        ("carbon_flux", "kg C m-2 s-1", "g C m-2 day-1", "carbon_flux_kg_s_to_g_day"),
        ("generic", "g C m-2 day-1", "kg C m-2 s-1", "carbon_flux_g_day_to_kg_s"),
        ("generic", "kg C m-2 s-1", "g C m-2 day-1", "carbon_flux_kg_s_to_g_day"),
    ],
)
def test_daily_carbon_flux_entries_resolve_to_one_method(
    variable_name: str, source_unit: str, target_unit: str, method_name: str
) -> None:
    """The formerly duplicated daily carbon-flux entries map to a single existing method."""
    assert UnitConverter.UNIT_CONVERSIONS[(variable_name, source_unit, target_unit)] == method_name
    assert callable(getattr(UnitConverter, method_name))
