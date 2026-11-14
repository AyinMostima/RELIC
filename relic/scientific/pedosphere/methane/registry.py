"""Registration utilities for the pedosphere methane process chain."""

from typing import Optional

from ....core.simulator import RELICSimulator

from .production import MethaneProductionFunctor
from .oxidation import MethaneOxidationFunctor
from .transport import MethaneTransportFunctor
from .budget_coupler import MethaneBudgetCoupler
from .scales import MethaneScaleCouplingFunctor


def register_methane_functors(simulator: RELICSimulator, device: Optional[str] = None) -> None:
    """Register the methane functors with a simulator in dependency order."""
    device_kw = {"device": None}
    if device is not None:
        import torch
        device_kw["device"] = torch.device(device)

    scale_functor = MethaneScaleCouplingFunctor(**device_kw)
    production_functor = MethaneProductionFunctor(**device_kw)
    oxidation_functor = MethaneOxidationFunctor(**device_kw)
    transport_functor = MethaneTransportFunctor(**device_kw)
    budget_functor = MethaneBudgetCoupler(**device_kw)

    simulator.register_functor(scale_functor)
    simulator.register_functor(production_functor)
    simulator.register_functor(oxidation_functor)
    simulator.register_functor(transport_functor)
    simulator.register_functor(budget_functor)


__all__ = ["register_methane_functors"]
