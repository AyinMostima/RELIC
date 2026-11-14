"""Plant tissue nutrient limitation diagnostic for RELIC.

The module maps canonical plant-tissue nitrogen and phosphorus mass fractions
to a bounded control factor. It is a read-only phenomenological diagnostic; it
does not own or advance nitrogen or phosphorus state.

References
----------
- Greenwood, D.J. et al. (1990). Decline in percentage N of C3 and C4 crops
  with increasing plant mass. Annals of Botany 66, 425-436
- Agren, G.I. et al. (2012). Nutrient limitation on terrestrial plant growth.
  New Phytologist 194, 953-960
- Aerts, R. and Chapin III, F.S. (2000). The mineral nutrition of wild plants
  revisited. Advances in Ecological Research 30, 1-67

Mathematical Framework
----------------------
For i in {N, P}:
    z_i = (c_i - c_i,min) / (c_i,crit - c_i,min)
    f_i = clip(max(z_i, 0)^p, f_min, f_max)
    f_NP = min(f_N, f_P)

Greenwood et al. (1990) support the concept of biomass-dependent critical
nitrogen concentration, not this fixed-threshold power response. The minimum
operator is a Liebig-inspired first-order closure. The implemented parameters
therefore require calibration and are not asserted as a universal plant law.

Units
-----
- Tissue nutrient mass fractions: kg element kg^-1 dry matter
- Nutrient factors: 1
"""

from typing import Dict, Optional, Any
import torch
from torch import Tensor

from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
)


class NutrientStress(BiologicalProcessFunctor):
    """Compute a calibratable N/P tissue-concentration control factor."""
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(
        self,
        name: str = "nutrient_stress",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        # Initialize metadata for proper Functor integration
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description='Nutrient stress response model for nitrogen and phosphorus limitations',
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            parameters={
                'biosphere.stress.nutrient.n_critical_concentration': (
                    'Critical nitrogen concentration in plant tissue [kg N kg-1 DM]'
                ),
                'biosphere.stress.nutrient.n_minimum_concentration': (
                    'Minimum nitrogen concentration in plant tissue [kg N kg-1 DM]'
                ),
                'biosphere.stress.nutrient.p_critical_concentration': (
                    'Critical phosphorus concentration in plant tissue [kg P kg-1 DM]'
                ),
                'biosphere.stress.nutrient.p_minimum_concentration': (
                    'Minimum phosphorus concentration in plant tissue [kg P kg-1 DM]'
                ),
                'biosphere.stress.nutrient.nutrient_stress_exponent': (
                    'Exponent of the nutrient stress response [dimensionless]'
                ),
                'biosphere.stress.global.factor_min': (
                    'Lower bound of the stress factor [dimensionless]'
                ),
                'biosphere.stress.global.factor_max': (
                    'Upper bound of the stress factor [dimensionless]'
                ),
            },
            mathematical_expression=(
                'z_i = (c_i - c_i,min) / (c_i,crit - c_i,min); '
                'f_i = clip(max(z_i, 0)^p, f_min, f_max); '
                'f_NP = min(f_N, f_P)'
            ),
            references=[
                'Greenwood, D.J. et al. (1990). Annals of Botany 66, 425-436',
                'Agren, G.I. et al. (2012). New Phytologist 194, 953-960',
                'Aerts, R. and Chapin III, F.S. (2000). Advances in Ecological Research 30, 1-67',
            ],
            input_variables=[
                'nitrogen_concentration',
                'phosphorus_concentration',
            ],
            output_variables=[
                'nutrient_stress_factor',
                'diagnostics.nutrient_stress.nitrogen_factor',
                'diagnostics.nutrient_stress.phosphorus_factor',
            ],
            conservation_laws=[],
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        super().__init__(
            name,
            config or {},
            metadata=metadata,
            device=device,
        )
        
        # Unified namespace for all nutrient stress parameters
        self.namespace = "biosphere.stress.nutrient"
        
        # Get nutrient parameters from ParameterManager
        _ = self.get_parameter_value(
            "n_critical_concentration",
            namespace=self.namespace, required=True)
        
        _ = self.get_parameter_value(
            "n_minimum_concentration",
            namespace=self.namespace, required=True)
        
        _ = self.get_parameter_value(
            "p_critical_concentration",
            namespace=self.namespace, required=True)
        
        _ = self.get_parameter_value(
            "p_minimum_concentration",
            namespace=self.namespace, required=True)
        
        _ = self.get_parameter_value(
            "nutrient_stress_exponent",
            namespace=self.namespace, required=True)
        
        # Parameters will be accessed dynamically from ParameterManager (no caching)
            
    def get_provides(self) -> list:
        """Define output variables provided by this Functor."""
        return list(self.metadata.output_variables)

    @staticmethod
    def _require_tissue_fraction(state: StateDict, name: str) -> Tensor:
        """Return a finite, non-negative floating tissue mass fraction."""
        if name not in state:
            raise ValueError(
                f"Required state variable '{name}' not found in state. "
                f"Available keys: {list(state.keys())}."
            )
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating torch.Tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")
        if bool((value < 0).any().item()):
            raise ValueError(f"{name} must be non-negative")
        return value
        
    def forward(self, state: StateDict, time: float, dt: float, **kwargs) -> FluxDict:
        """
        Main forward pass for nutrient stress calculation.

        Compute individual response factors and their limiting minimum.

        Parameters
        ----------
        state : StateDict
            Current system state including nutrient concentrations
        time : float
            Current simulation time
        dt : float
            Time step
        **kwargs
            Additional arguments

        Returns
        -------
        FluxDict
            Dictionary containing nutrient stress factors
        """
        # References: Greenwood et al. (1990); Agren et al. (2012);
        # Aerts and Chapin (2000). These references support critical tissue
        # nutrient concepts and limitation context, not a universal fixed curve.
        # Mathematical expression:
        # z_i = (c_i - c_i,min) / (c_i,crit - c_i,min)
        # f_i = clip(max(z_i, 0)^p, f_min, f_max)
        # f_NP = min(f_N, f_P)
        # Parameter meanings:
        # c_i, c_i,min, c_i,crit [kg element kg-1 dry matter];
        # p, f_min, f_max, f_i, f_NP [1].
        # Open/closed system route:
        # This read-only observation has no material source, sink, or state advance.
        n_conc = self._require_tissue_fraction(state, 'nitrogen_concentration')
        p_conc = self._require_tissue_fraction(state, 'phosphorus_concentration')

        n_critical = self.get_parameter_value(
            "n_critical_concentration", namespace=self.namespace, required=True)
        n_minimum = self.get_parameter_value(
            "n_minimum_concentration", namespace=self.namespace, required=True)
        p_critical = self.get_parameter_value(
            "p_critical_concentration", namespace=self.namespace, required=True)
        p_minimum = self.get_parameter_value(
            "p_minimum_concentration", namespace=self.namespace, required=True)
        nutrient_exponent = self.get_parameter_value(
            "nutrient_stress_exponent", namespace=self.namespace, required=True)

        stress_min = self.get_parameter_value(
            "global.factor_min",
            namespace="biosphere.stress", required=True)
        stress_max = self.get_parameter_value(
            "global.factor_max",
            namespace="biosphere.stress", required=True)

        target = {"device": n_conc.device, "dtype": n_conc.dtype}
        n_critical_tensor = torch.as_tensor(n_critical, **target)
        n_minimum_tensor = torch.as_tensor(n_minimum, **target)
        p_critical_tensor = torch.as_tensor(p_critical, **target)
        p_minimum_tensor = torch.as_tensor(p_minimum, **target)
        exponent_tensor = torch.as_tensor(nutrient_exponent, **target)
        stress_min_tensor = torch.as_tensor(stress_min, **target)
        stress_max_tensor = torch.as_tensor(stress_max, **target)

        parameter_tensors = (
            n_critical_tensor,
            n_minimum_tensor,
            p_critical_tensor,
            p_minimum_tensor,
            exponent_tensor,
            stress_min_tensor,
            stress_max_tensor,
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in parameter_tensors):
            raise ValueError("nutrient stress parameters must contain only finite values")
        if not bool((n_critical_tensor > n_minimum_tensor).all().item()):
            raise ValueError("n_critical_concentration must exceed n_minimum_concentration")
        if not bool((p_critical_tensor > p_minimum_tensor).all().item()):
            raise ValueError("p_critical_concentration must exceed p_minimum_concentration")
        if not bool((exponent_tensor > 0).all().item()):
            raise ValueError("nutrient_stress_exponent must be positive")
        valid_bounds = (
            (stress_min_tensor >= 0)
            & (stress_min_tensor < stress_max_tensor)
            & (stress_max_tensor <= 1)
        )
        if not bool(valid_bounds.all().item()):
            raise ValueError("stress factor bounds must satisfy 0 <= min < max <= 1")

        n_normalized = (n_conc - n_minimum_tensor) / (
            n_critical_tensor - n_minimum_tensor
        )
        p_target = {
            "device": p_conc.device,
            "dtype": p_conc.dtype,
        }
        p_minimum_on_input = p_minimum_tensor.to(**p_target)
        p_critical_on_input = p_critical_tensor.to(**p_target)
        exponent_on_p = exponent_tensor.to(**p_target)
        stress_min_on_p = stress_min_tensor.to(**p_target)
        stress_max_on_p = stress_max_tensor.to(**p_target)
        p_normalized = (p_conc - p_minimum_on_input) / (
            p_critical_on_input - p_minimum_on_input
        )
        n_factor = torch.clamp(
            torch.clamp_min(n_normalized, 0).pow(exponent_tensor),
            min=stress_min_tensor,
            max=stress_max_tensor,
        )
        p_factor = torch.clamp(
            torch.clamp_min(p_normalized, 0).pow(exponent_on_p),
            min=stress_min_on_p,
            max=stress_max_on_p,
        )
        if n_factor.device != p_factor.device:
            raise ValueError("nutrient concentration tensors must use the same device")
        overall_stress = torch.minimum(n_factor, p_factor)

        return {
            'nutrient_stress_factor': UpdateItem(
                value=overall_stress,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit='1',
                source_functor=self.name,
            ),
            'diagnostics.nutrient_stress.nitrogen_factor': UpdateItem(
                value=n_factor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit='1',
                source_functor=self.name,
            ),
            'diagnostics.nutrient_stress.phosphorus_factor': UpdateItem(
                value=p_factor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit='1',
                source_functor=self.name,
            ),
        }


__all__ = ['NutrientStress']
