import pytest

import torch
from relic.core.functor import FunctorBase, FunctorMetadata, ProcessCategory
from relic.parameters import ParameterCategory, ParameterSpec
from relic.parameters import get_global_manager

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class _DummyFunctor(FunctorBase):
    def forward(self, state, time=None, dt=None, **kwargs):
        return {}

    def get_dependencies(self):
        return []


def test_get_parameter_value_raises_on_unconvertible_units():
    manager = get_global_manager()
    param_key = "test.unit_mismatch.value"
    manager.parameters.pop(param_key, None)

    spec = ParameterSpec(
        key=param_key,
        value=1.0,
        units="dimensionless",
        bounds=(0.0, 2.0),
        category=ParameterCategory.MODEL_PARAMETER,
        description="Dummy parameter for unit conversion failure test", dtype="float32",
    )
    manager.register_parameter(spec)

    functor = _DummyFunctor(
        name="dummy_unit_conversion",
        config={},
        metadata=FunctorMetadata(
            name="dummy_unit_conversion",
            description="Dummy functor for unit conversion validation",
            category=ProcessCategory.SCALE_COUPLING,
            references=["Unit conversion validation stub"],
            input_variables=["diagnostics.unit_test.input"],
            output_variables=["diagnostics.unit_test.output"],
        ),
        parameter_manager=manager,
    )

    try:
        with pytest.raises(ValueError):
            functor.get_parameter_value(
                "unit_mismatch.value",
                namespace="test",
                units="kg",
                validate=True,
                required=True,
            )
    finally:
        manager.parameters.pop(param_key, None)


def _state_range_functor():
    return _DummyFunctor(
        name="dummy_state_range",
        config={},
        metadata=FunctorMetadata(
            name="dummy_state_range",
            description="Dummy functor for state range validation",
            category=ProcessCategory.SCALE_COUPLING,
            references=["State range validation stub"],
            input_variables=["diagnostics.state_range.input"],
            output_variables=["diagnostics.state_range.output"],
        ),
    )


def test_get_state_variable_raises_for_out_of_range_state_instead_of_clamping():
    """An out-of-range state value raises for every element (NaN included); an
    in-range value is returned unchanged, never clipped."""
    functor = _state_range_functor()
    inside = torch.tensor([0.0, 1.0, 2.0])
    returned = functor._get_state_variable({"x": inside}, "x", validate_range=(0.0, 2.0))
    assert torch.equal(returned, inside)
    for values in (
        torch.tensor(3.0),
        torch.tensor([1.0, 2.5]),
        torch.tensor([-0.1, 1.0]),
        torch.tensor([float("nan")]),
    ):
        with pytest.raises(ValueError, match=r"'x'.*outside"):
            functor._get_state_variable({"x": values}, "x", validate_range=(0.0, 2.0))


def test_get_state_variable_without_a_default_does_not_advise_a_yaml_default():
    """A missing state key raises KeyError reporting NOT FOUND; the message asks for
    the input, not for a YAML runtime default."""
    functor = _state_range_functor()
    with pytest.raises(KeyError) as raised:
        functor._get_state_variable({}, "x")
    message = str(raised.value)
    assert "NOT FOUND" in message
    assert "Supply 'x'" in message
    assert "Define parameter" not in message


def test_get_state_variable_has_no_default_route():
    """No YAML-default tier: the reader declares no default parameter and a
    missing key is reported as NOT FOUND."""
    import inspect

    parameters = inspect.signature(FunctorBase._get_state_variable).parameters
    for removed in ("default_param_name", "default_param_namespace", "shape"):
        assert removed not in parameters, removed
    functor = _state_range_functor()
    with pytest.raises(TypeError):
        functor._get_state_variable({}, "x", default_param_name="x_default")


def test_get_parameter_value_respects_spec_dtype():
    manager = get_global_manager()
    param_key = "test.dtype.float32_value"
    manager.parameters.pop(param_key, None)

    spec = ParameterSpec(
        key=param_key,
        value=1.5,
        units="dimensionless",
        dtype="float32",
        bounds=(0.0, 2.0),
        category=ParameterCategory.MODEL_PARAMETER,
        description="Dummy parameter for dtype enforcement test",
    )
    manager.register_parameter(spec)

    functor = _DummyFunctor(
        name="dummy_dtype_enforcement",
        config={},
        metadata=FunctorMetadata(
            name="dummy_dtype_enforcement",
            description="Dummy functor for dtype validation",
            category=ProcessCategory.SCALE_COUPLING,
            references=["Dtype enforcement test"],
            input_variables=["diagnostics.dtype_test.input"],
            output_variables=["diagnostics.dtype_test.output"],
        ),
        parameter_manager=manager,
    )

    original_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        value = functor.get_parameter_value(
            "dtype.float32_value",
            namespace="test",
            validate=True,
            required=True,
        )
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float32
    finally:
        torch.set_default_dtype(original_default)
        manager.parameters.pop(param_key, None)

