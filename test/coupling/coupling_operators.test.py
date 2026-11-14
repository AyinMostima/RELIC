import torch

from relic.coupling.core.conservation_checker import ConservationFixer
from relic.coupling.core.coupling_operators import (
    ConstraintProjector,
    FluxKernel,
    TemporalLagOperator,
)


def test_conservation_fixer_multiplicative() -> None:
    source = torch.tensor([2.0, 2.0], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0], dtype=torch.float32)
    fixer = ConservationFixer(method="multiplicative", tolerance=0.0)

    corrected, diagnostics = fixer.apply(source, target, variable_name="water")

    assert torch.isclose(torch.sum(corrected), torch.sum(source))
    assert diagnostics.is_conserved
    assert diagnostics.enforcement_applied


def test_conservation_fixer_additive() -> None:
    source = torch.tensor([3.0, 3.0], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0], dtype=torch.float32)
    fixer = ConservationFixer(method="additive", tolerance=0.0)

    corrected, diagnostics = fixer.apply(source, target, variable_name="carbon")

    assert torch.isclose(torch.sum(corrected), torch.sum(source))
    assert diagnostics.is_conserved
    assert diagnostics.enforcement_applied


def test_temporal_lag_linear() -> None:
    operator = TemporalLagOperator(method="linear")
    previous = torch.tensor([0.0], dtype=torch.float32)
    current = torch.tensor([10.0], dtype=torch.float32)

    result = operator.apply(
        previous,
        current,
        t_prev=0.0,
        t_current=10.0,
        t_target=5.0,
    )

    assert torch.isclose(result["result"], torch.tensor([5.0], dtype=torch.float32)).all()


def test_constraint_projector_non_negative() -> None:
    projector = ConstraintProjector(constraint_type="non_negative")
    values = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)

    result = projector.apply(values)

    assert torch.all(result["result"] >= 0.0)
    assert result["diagnostics"]["clipped_count"] == 1


def test_flux_kernel_basic() -> None:
    kernel = FluxKernel()
    source = torch.tensor([3.0, 4.0], dtype=torch.float32)
    target = torch.tensor([1.0, 1.0], dtype=torch.float32)
    transfer = torch.tensor(2.0, dtype=torch.float32)
    area = torch.tensor(3.0, dtype=torch.float32)

    result = kernel.apply(
        source_concentration=source,
        target_concentration=target,
        transfer_coefficient=transfer,
        interface_area=area,
    )

    expected = torch.tensor([12.0, 18.0], dtype=torch.float32)
    assert torch.allclose(result["result"], expected)
