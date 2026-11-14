import torch
import pytest

from relic.core.functor import FunctorMetadata, ProcessCategory
from relic.core.functor import (
    BiologicalProcessFunctor,
    ScaleCouplingFunctor,
    TransportMechanismFunctor,
)
from relic.contracts.contract_loader import ContractRegistry, VariableContract
import relic.core.functor.biological_bridge as biological_bridge
import relic.core.functor.functor_bridge_utils as functor_bridge_utils
import relic.core.functor.physical_field_bridge as physical_field_bridge
import relic.core.functor.scale_coupling_bridge as scale_coupling_bridge
import relic.core.functor.transport_bridge as transport_bridge

from relic.parameters.namespaces import ParameterNamespace
from relic.utils.unified_response_functions import beer_lambert_effective_leaf_area


def _fixture_metadata(
    name: str,
    category: ProcessCategory,
    output_variable: str,
) -> FunctorMetadata:
    """Build complete metadata for a bridge response fixture."""
    return FunctorMetadata(
        name=name,
        category=category,
        description=f"Bridge response fixture for {name}",
        references=["RELIC bridge response test fixture"],
        output_variables=[output_variable],
    )


class DummyParameterManager:
    """Minimal parameter manager stub for bridge tests."""

    def __init__(self, values):
        self.values = values
        self.parameters = {}
        self.requests = []

    def list(self, namespace=None):
        return []

    def get(
        self,
        key,
        default=None,
        namespace=None,
        units=None,
        expected_type=None,
        required=False,
    ):
        self.requests.append((key, namespace))
        if key in self.values:
            return self.values[key]
        if required:
            raise KeyError(key)
        return default

    def get_parameter(self, key):
        class Spec:
            def __init__(self, units_value):
                self.units = units_value

        return Spec(units_value="dimensionless")


class DummyBioFunctor(BiologicalProcessFunctor):
    """Lightweight biological functor for response function regression tests."""

    def __init__(self, *args, **kwargs):
        # Keep unit test isolated from global parameter defaults.
        self.q10_value = 2.0
        self.temperature_optimum = 298.15
        self.ph_optimum = 7.0
        super().__init__(*args, **kwargs)

    def forward(self, state, time, dt, **kwargs):
        return {"biological_rate": torch.tensor([1.0], device=self.device)}


class DummyScaleCoupling(ScaleCouplingFunctor):
    """Minimal scale coupling functor for namespace resolution checks."""

    def forward(self, state, time, dt, **kwargs):
        return {"coarse_result": torch.tensor(3.5, device=self.device)}


class MissingPayloadScaleCoupling(ScaleCouplingFunctor):
    """Scale coupler whose forward result omits the required scale payload."""

    def forward(self, state, time, dt, **kwargs):
        return {}


class ExplicitPayloadScaleCoupling(ScaleCouplingFunctor):
    """Scale coupler that exposes explicit payload keys for up/downscale."""

    def forward(self, state, time, dt, **kwargs):
        operation = state.get("operation")
        if operation == "upscale":
            return {"coarse_result": state["fine_data"] * 2.0}
        if operation == "downscale":
            return {"fine_result": state["coarse_data"] * 0.5}
        return {}


class DummyTransportFunctor(TransportMechanismFunctor):
    """Minimal transport functor for alias resolution tests."""

    TRANSPORT_TYPE = "diffusion"

    def forward(self, state, time=None, dt=None, **kwargs):
        return {"transport_flux": torch.tensor(0.0, device=self.device)}


def test_beer_lambert_effective_leaf_area_has_ground_area_basis():
    """The shared canopy operator integrates attenuated leaf area exactly."""
    lai = torch.tensor([0.0, 1.0, 3.0])
    extinction = torch.tensor(0.5)

    result = beer_lambert_effective_leaf_area(lai, extinction)
    expected = (1.0 - torch.exp(-extinction * lai)) / extinction

    assert torch.allclose(result, expected)
    with pytest.raises(ValueError, match="leaf_area_index"):
        beer_lambert_effective_leaf_area(torch.tensor(-1.0), extinction)
    with pytest.raises(ValueError, match="extinction"):
        beer_lambert_effective_leaf_area(lai, torch.tensor(0.0))


def test_temperature_laws_follow_the_temperature_device_and_dtype():
    """The input temperature decides the computation device and dtype. A
    coefficient elsewhere (a ParameterManager-held theta stays on the device the
    manager keeps it on) is moved with the differentiable Tensor.to, so the
    result stays on the temperature's device and the gradient reaches theta.
    Without CUDA the strict meta probe checks placement and the gradient path."""
    from relic.utils.unified_response_functions import (
        arrhenius_temperature_response,
        peaked_temperature_response,
        q10_temperature_response,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
    temperature = torch.tensor([288.15, 298.15, 308.15], dtype=torch.float32, device=device)
    theta = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    laws = {
        "q10": lambda: q10_temperature_response(theta, temperature, 2.0),
        "arrhenius": lambda: arrhenius_temperature_response(
            theta, temperature, 50000.0, gas_constant=8.314462618, reference_temp=298.15,
        ),
        "peaked": lambda: peaked_temperature_response(
            theta, temperature, 50000.0, 200000.0, 650.0,
            gas_constant=8.314462618, reference_temp=298.15,
        ),
    }
    for name, law in laws.items():
        theta.grad = None
        response = law()
        assert response.device.type == device.type, name
        assert response.dtype == torch.float32, name
        response.sum().backward()
        assert theta.grad is not None, name
        assert theta.grad.device.type == "cpu", name
        if device.type != "meta":
            assert float(theta.grad) != 0.0, name
    with pytest.raises(TypeError, match="floating"):
        q10_temperature_response(theta, torch.tensor([298], device=device), 2.0)


def test_temperature_response_is_owned_by_the_concrete_process():
    """The bridge holds no generic temperature law and no law label: a process such
    as the MEGAN2.1 temperature activity (Guenther et al. 2012, Eq. 7, the LDF blend of
    Eqs. 8 and 11) evaluates its own law, and a label keyword is rejected."""
    manager = DummyParameterManager(
        {
            "validation.numerical.response.q10": 2.0,
            "validation.numerical.response.ph_optimum": 7.0,
        }
    )
    functor = DummyBioFunctor(
        name="megan21_dummy",
        parameter_manager=manager,
        metadata=_fixture_metadata(
            "megan21_dummy",
            ProcessCategory.BIOLOGICAL_PROCESS,
            "biological_rate",
        ),
    )
    # The bridge supplies no generic law: the abstract contract refuses until the
    # concrete process implements temperature_function.
    with pytest.raises(RuntimeError, match="concrete BiologicalProcessFunctor"):
        functor.temperature_function(torch.tensor([300.0]))
    with pytest.raises(TypeError, match="temperature_response"):
        DummyBioFunctor(
            name="megan21_dummy",
            parameter_manager=manager,
            metadata=_fixture_metadata(
                "megan21_dummy",
                ProcessCategory.BIOLOGICAL_PROCESS,
                "biological_rate",
            ),
            temperature_response="megan21",
        )


def test_unused_parabolic_temperature_stress_helper_is_retired():
    """StressResponseFunctions.temperature_stress_optimal had no caller and carried
    hard-coded default optima; it is removed."""
    from relic.utils.unified_response_functions import (
        StressResponseFunctions,
        stress_response_functions,
    )

    assert not hasattr(StressResponseFunctions, "temperature_stress_optimal")
    assert not hasattr(stress_response_functions, "temperature_stress_optimal")
    assert hasattr(StressResponseFunctions, "water_stress_linear")


def test_unused_vpd_stress_helper_is_retired():
    """StressResponseFunctions.vapor_pressure_deficit_stress had no caller and carried
    hard-coded threshold and sensitivity defaults; it is removed."""
    from relic.utils.unified_response_functions import (
        StressResponseFunctions,
        stress_response_functions,
    )

    assert not hasattr(StressResponseFunctions, "vapor_pressure_deficit_stress")
    assert not hasattr(stress_response_functions, "vapor_pressure_deficit_stress")


def test_scale_coupling_functor_namespaced_parameter_resolution():
    """get_namespaced_parameter should honor ParameterNamespace resolution and device casting."""
    param_value = 7.5
    manager = DummyParameterManager(
        {f"{ParameterNamespace.COUPLING_DEFAULTS.value}.test_param": param_value}
    )
    functor = DummyScaleCoupling(
        name="coupling.scales.test_functor",
        parameter_manager=manager,
        metadata=_fixture_metadata(
            "coupling_scales_test_functor",
            ProcessCategory.SCALE_COUPLING,
            "coarse_result",
        ),
        source_scale="fine",
        target_scale="coarse",
        coupling_type="spatial",
    )

    value_scalar = functor.get_namespaced_parameter(
        "test_param",
        namespace=ParameterNamespace.COUPLING_DEFAULTS,
        units="dimensionless",
        as_scalar=True,
    )
    value_tensor = functor.get_namespaced_parameter(
        "test_param",
        namespace=ParameterNamespace.COUPLING_DEFAULTS,
        units="dimensionless",
        as_tensor=True,
    )

    assert value_scalar == param_value
    assert torch.is_tensor(value_tensor)
    assert torch.allclose(
        value_tensor,
        torch.tensor(param_value, device=value_tensor.device, dtype=value_tensor.dtype),
    )


def test_transport_aliases_from_contract_registry(monkeypatch) -> None:
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="soil_moisture",
            kind="state",
            udunits="dimensionless",
            aliases=["theta"],
        )
    )

    monkeypatch.setattr(transport_bridge, "load_contract_registry", lambda: registry)

    functor = DummyTransportFunctor(
        name="dummy_transport",
        metadata=FunctorMetadata(
            name="dummy_transport",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            description="Alias resolution test",
            references=["Unit test stub"],
            output_variables=["transport_flux"],
        ),
    )

    # A transport target named by its alias resolves to the canonical contract,
    # whose udunits give the target rate of the flux conversion.
    canonical, state_units, rate_units = functor._declared_target_rate_units(
        "theta", None
    )
    assert canonical == "soil_moisture"
    assert state_units == "dimensionless"
    assert rate_units == "dimensionless s-1"


def test_bridge_classes_keep_no_contract_registry_cache() -> None:
    """The contract loader is the single owner of the packaged registry; no
    bridge module keeps its own registry or loader."""
    for module in (
        functor_bridge_utils,
        biological_bridge,
        physical_field_bridge,
        scale_coupling_bridge,
        transport_bridge,
    ):
        assert not hasattr(module, "_CONTRACT_REGISTRY")
        assert not hasattr(module, "_get_contract_registry")


def test_scale_coupling_base_rejects_missing_upscale_downscale_payloads() -> None:
    """Scale transforms must not silently degrade to identity behavior."""
    functor = MissingPayloadScaleCoupling(
        name="missing_payload_scale",
        metadata=_fixture_metadata(
            "missing_payload_scale",
            ProcessCategory.SCALE_COUPLING,
            "missing_payload",
        ),
        source_scale="fine",
        target_scale="coarse",
        coupling_type="spatial",
    )
    fine = torch.tensor([1.0, 2.0], dtype=torch.float32)
    coarse = torch.tensor([3.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="explicit upscale payload"):
        functor.upscale(fine)
    with pytest.raises(RuntimeError, match="explicit downscale payload"):
        functor.downscale(coarse)


def test_scale_coupling_base_recovers_explicit_upscale_downscale_payloads() -> None:
    """Base ScaleCouplingFunctor should recover explicit payload keys from forward outputs."""
    functor = ExplicitPayloadScaleCoupling(
        name="explicit_scale",
        metadata=_fixture_metadata(
            "explicit_scale",
            ProcessCategory.SCALE_COUPLING,
            "coarse_result",
        ),
        source_scale="fine",
        target_scale="coarse",
        coupling_type="spatial",
    )
    fine = torch.tensor([2.0], dtype=torch.float32)
    coarse = torch.tensor([8.0], dtype=torch.float32)

    upscaled = functor.upscale(fine)
    downscaled = functor.downscale(coarse)

    assert torch.equal(upscaled, torch.tensor([4.0], dtype=torch.float32))
    assert torch.equal(downscaled, torch.tensor([4.0], dtype=torch.float32))
