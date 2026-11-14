"""
BVOC Response Functions for RELIC

Shared activity-factor helper composed inside SesquiterpeneSynthesis. The
MEGAN2.1 activity laws are owned by the module-level functions of
relic/scientific/biosphere/bvoc/megan_model.py, which MEGANModel also
evaluates: megan_light_activity (Guenther et al. 2012, Eq. 3),
megan_light_dependent_activity (Eq. 4), megan_temperature_activity (Eq. 7),
megan_ldf_temperature_activity (Eq. 8) and megan_lif_temperature_activity
(Eq. 11). This module composes them with ParameterManager coefficients and
adds the water and combined stress factors:

1. Light activity, Eq. (3): gamma_L = (1 - LDF) + LDF gamma_P_LDF, with the
   light-dependent fraction LDF of Guenther et al. (2012) Table 4. Isoprene is
   fully light dependent (LDF = 1), so Eq. (3) reduces to Eq. (4); the lumped
   monoterpene class uses the representative 'Other Monoterpenes' row and the lumped
   OVOC class the Table 6 emission-weighted mixture of its methanol, acetone and
   acetaldehyde rows. A compound class without a registered Table 4 row is rejected.
2. Temperature activity, Eq. (7): gamma_T = (1 - LDF) gamma_T,LIF + LDF gamma_T,LDF
   with the Eq. (11) and Eq. (8) laws and the coefficients of one Table 4 row per
   class (Eq. 8 alone for isoprene, LDF = 1); the lumped OVOC class is the Table 6
   emission-weighted mixture of its component rows.
3. Stress: the relative soil moisture or the linear water-potential ramp, and
   the minimum of the explicit stress factors in [0, 1].

CO2 inhibition is owned by MEGANModel._compute_co2_inhibition_factor
(Guenther et al. 2012, Eq. 14); no leaf-age factor is applied.

References
----------
- Guenther, A., et al. (2012). The Model of Emissions of Gases and Aerosols from
  Nature version 2.1 (MEGAN2.1). Geoscientific Model Development 5, 1471-1492
- Niinemets, U., et al. (2010). The leaf-level emission factor of volatile
  isoprenoids. Biogeosciences 7, 1809-1832

Units
-----
- Activity factors: dimensionless
- PPFD: umol photons m-2 s-1
- Temperature: K
- Soil water potential: MPa; relative soil moisture: dimensionless [0-1]
"""

from typing import Dict, Optional, Union, Any, Literal, overload
import torch
from torch import Tensor

# Import from unified response function library
from relic.utils.unified_response_functions import stress_response_functions
from relic.utils.physical_constants import get_physical_constant
from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import (
    megan_class_light_activity,
    megan_class_temperature_activity,
)


def _validated_driver(
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
) -> Tensor:
    """Return ``value`` as a tensor after rejecting non-finite or out-of-domain entries."""
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        below = tensor <= minimum if strict_minimum else tensor < minimum
        if bool(below.any()):
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and bool((tensor > maximum).any()):
        raise ValueError(f"{name} must be <= {maximum}")
    return tensor


def _resolve_manager(manager):
    """Return explicitly injected manager when available."""
    if manager is not None:
        return manager
    return get_global_manager()


@overload
def _require_scalar(
    manager: Any, key: str, namespace: str, units: Optional[str], *, required: Literal[True] = True,
) -> float: ...


@overload
def _require_scalar(
    manager: Any, key: str, namespace: str, units: Optional[str], *, required: Literal[False],
) -> Optional[float]: ...


def _require_scalar(
    manager,
    key: str,
    namespace: str,
    units: Optional[str],
    *,
    required: bool = True,
) -> Optional[float]:
    """Fetch a scalar parameter using unified parameter access."""
    full_key = f"{namespace}.{key}" if namespace else key
    active_manager = _resolve_manager(manager)
    value = active_manager.get(
        full_key,
        units=units,
        required=required,
        expected_type=None,
    )
    if value is None:
        if required:
            raise ValueError(f"Required scalar parameter '{full_key}' is missing.")
        return None

    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected one value for scalar parameter '{full_key}'.")
        value = value.item()
    return float(value)


class BVOCLightResponse:
    """
    MEGAN2.1 light activity factors composed from the shared megan_model law.
    """

    def __init__(self, parameter_manager=None):
        self._parameter_manager = _resolve_manager(parameter_manager)

    def calculate_factor(
        self,
        ppfd: Tensor,
        compound: str = "isoprene",
    ) -> Tensor:
        return self.get_compound_light_response(
            compound,
            ppfd.to(dtype=torch.float32),
            param_manager=self._parameter_manager,
        )

    @staticmethod
    def get_compound_light_response(
        compound: str,
        ppfd: Tensor,
        param_manager=None,
    ) -> Tensor:
        """
        Light activity factor of a registered BVOC compound class.

        Mathematical Expression
        -----------------------
        gamma_L = (1 - LDF) + LDF gamma_P_LDF (Guenther et al. 2012, Eq. 3) with
        gamma_P_LDF = alpha CL PPFD / sqrt(1 + alpha^2 PPFD^2) (Eq. 4), evaluated
        by megan_model.megan_light_activity. Isoprene (LDF = 1) returns
        gamma_P_LDF of megan_model.megan_light_dependent_activity.

        References
        ----------
        - Guenther et al. (2012): Eqs. (3)-(4) and Table 4 (LDF = 1 for isoprene,
          0.4 for 'Other Monoterpenes', the representative value of the lumped
          monoterpene class, 0.5 for alpha-farnesene, beta-caryophyllene and
          other sesquiterpenes, and 0.8, 0.2 and 0.8 for methanol, acetone and the
          bidirectional VOC class, mixed by their Table 6 emission shares for the
          lumped OVOC class)
        - Guenther et al. (2006): MEGAN light-dependent isoprene activity

        Parameters
        ----------
        compound : str
            Compound class with registered Table 4 rows: "isoprene", "monoterpenes",
            "sesquiterpenes" or "ovocs".
        ppfd : Tensor
            Photosynthetic photon flux density [umol photons m-2 s-1], finite and >= 0.
        param_manager : optional
            ParameterManager instance; the global manager when omitted.

        Returns
        -------
        Tensor
            Light activity factor gamma_L [dimensionless] in the dtype of ``ppfd``.
        """
        manager = _resolve_manager(param_manager)
        compound_class = compound.lower()
        _validated_driver(ppfd, "ppfd", minimum=0.0)
        alpha = _require_scalar(manager, "alpha_light", "biosphere.bvoc.megan", "m2 s umol-1")
        cl = _require_scalar(manager, "cl_light", "biosphere.bvoc.megan", "dimensionless")
        alpha_tensor = torch.as_tensor(alpha, dtype=ppfd.dtype, device=ppfd.device)
        cl_tensor = torch.as_tensor(cl, dtype=ppfd.dtype, device=ppfd.device)

        def lookup(key: str, units: str) -> Tensor:
            return torch.as_tensor(
                _require_scalar(manager, key, "biosphere.bvoc.megan", units),
                dtype=ppfd.dtype,
                device=ppfd.device,
            )

        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (3)-(4)
        #   and Tables 4 and 6.
        # Mathematical expression: gamma_P_LDF = alpha CL PPFD / sqrt(1 + alpha^2 PPFD^2);
        #   gamma_L,j = (1 - LDF_j) + LDF_j gamma_P_LDF (gamma_P_LDF for isoprene);
        #   gamma_L = sum_j w_j gamma_L,j for the lumped OVOC class.
        # Parameter meanings: alpha is the PPFD response slope, CL the light scaling
        #   coefficient, LDF_j the light-dependent fraction of Table 4 row j and w_j the
        #   Table 6 emission share.
        # Units: PPFD [umol photons m-2 s-1]; alpha [m2 s umol-1]; CL, LDF_j, w_j,
        #   gamma_L [1].
        # Domain/constraints: finite PPFD >= 0 (checked above); an unregistered class
        #   raises ValueError; the law is megan_class_light_activity, shared with
        #   MEGANModel; no clamp is applied.
        # Sign convention: gamma_L is a non-negative multiplier of the emission magnitude.
        return megan_class_light_activity(
            compound_class, ppfd, alpha_tensor, cl_tensor, lookup
        )


class BVOCTemperatureResponse:
    """
    MEGAN2.1 temperature activity factors composed from the shared megan_model laws.
    """

    def __init__(self, parameter_manager=None):
        self._parameter_manager = _resolve_manager(parameter_manager)

    def calculate_factor(
        self,
        temperature: Tensor,
        compound: str = "isoprene",
    ) -> Tensor:
        return self.get_compound_temperature_response(
            compound,
            temperature.to(dtype=torch.float32),
            param_manager=self._parameter_manager,
        )

    @staticmethod
    def get_compound_temperature_response(
        compound: str,
        temperature: Tensor,
        param_manager=None,
    ) -> Tensor:
        """
        Temperature activity factor of a registered BVOC compound class.

        Mathematical Expression
        -----------------------
        gamma_T = (1 - LDF) exp(beta (T - T_s)) + LDF gamma_T,LDF (Guenther et al.
        2012, Eq. 7) with gamma_T,LDF = E_opt C_T2 exp(C_T1 x) /
        (C_T2 - C_T1 (1 - exp(C_T2 x))), x = (1/T_opt - 1/T) / R (Eq. 8, at the fixed
        standard history), and the coefficients of the class's Table 4 row (isoprene:
        LDF = 1, Eq. 8 alone). The lumped OVOC class: gamma_T = sum_j w_j gamma_T,j over
        its methanol, acetone and acetaldehyde rows with Table 6 emission shares w_j.

        References
        ----------
        - Guenther et al. (2012): Eqs. (7), (8) and (11) and Tables 4 and 6
        - Guenther et al. (1993): Exponential monoterpene temperature response
        - Tingey et al. (1980): Experimental basis of the exponential response

        Parameters
        ----------
        compound : str
            Compound class: "isoprene", "monoterpenes", "sesquiterpenes" or "ovocs".
        temperature : Tensor
            Leaf temperature [K], finite and > 0.
        param_manager : optional
            ParameterManager instance; the global manager when omitted.

        Returns
        -------
        Tensor
            Temperature activity factor gamma_T [dimensionless] in the dtype of
            ``temperature``.
        """
        manager = _resolve_manager(param_manager)
        compound_class = compound.lower()
        _validated_driver(temperature, "temperature", minimum=0.0, strict_minimum=True)

        def as_temperature_tensor(value: float) -> Tensor:
            return torch.as_tensor(value, dtype=temperature.dtype, device=temperature.device)

        topt = _require_scalar(manager, "topt", "biosphere.bvoc.megan", "K")
        ct2 = _require_scalar(manager, "ct2", "biosphere.bvoc.megan", "J mol-1")
        standard_temperature = _require_scalar(manager, "t_standard", "biosphere.bvoc", "K")
        gas_constant = float(get_physical_constant("gas_constant"))

        def lookup(key: str, units: str) -> Tensor:
            return as_temperature_tensor(
                _require_scalar(manager, key, "biosphere.bvoc.megan", units)
            )

        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (7),
        #   (8) and (11) and Tables 4 and 6.
        # Mathematical expression: gamma_T,j = (1 - LDF_j) exp(beta_j (T - T_s)) + LDF_j
        #   E_opt,j C_T2 exp(C_T1,j x) / (C_T2 - C_T1,j (1 - exp(C_T2 x))),
        #   x = (1/T_opt - 1/T)/R (the Eq. (8) term alone when LDF = 1);
        #   gamma_T = sum_j w_j gamma_T,j for the lumped OVOC class.
        # Parameter meanings: LDF_j, beta_j, C_T1,j and E_opt,j (Table 4 C_eo) come from
        #   Table 4 row j; w_j is the Table 6 emission share; T_opt and C_T2 are class
        #   independent; T_s is the standard temperature shared with MEGANModel; R is the
        #   physical-constant SSOT gas constant.
        # Units: T, T_s, T_opt [K]; beta_j [K-1]; C_T1,j, C_T2 [J mol-1];
        #   R [J mol-1 K-1]; LDF_j, E_opt,j, w_j, gamma_T [1].
        # Domain/constraints: finite T > 0 (checked above); an unregistered class raises
        #   ValueError; the shared law raises on a non-positive Eq. (8) denominator; no
        #   clamp is applied.
        # Sign convention: gamma_T is a non-negative multiplier of the emission magnitude.
        return megan_class_temperature_activity(
            compound_class,
            temperature,
            as_temperature_tensor(standard_temperature),
            as_temperature_tensor(topt),
            as_temperature_tensor(ct2),
            as_temperature_tensor(gas_constant),
            lookup,
        )


class BVOCEnvironmentalResponse:
    """
    Water and combined stress factors of the BVOC activity product.

    CO2 inhibition is owned by MEGANModel._compute_co2_inhibition_factor
    (Guenther et al. 2012, Eq. 14); no leaf-age factor is applied.
    """

    def __init__(self, parameter_manager=None):
        self._parameter_manager = _resolve_manager(parameter_manager)

    @staticmethod
    def drought_stress_response(
        soil_moisture: Optional[Tensor] = None,
        water_potential: Optional[Tensor] = None,
        wilting_point: Optional[float] = None,  # MPa
        field_capacity: Optional[float] = None,  # MPa
        param_manager=None
    ) -> Tensor:
        """
        Drought stress effect on BVOC emissions.

        Mathematical Expression
        -----------------------
        Water potential: gamma_W = clip((psi - psi_wp) / (psi_fc - psi_wp), 0, 1).
        Relative soil moisture: gamma_W = theta_rel, theta_rel in [0, 1].

        References
        ----------
        - Guenther et al. (2006): MEGAN soil-moisture activity factor, a linear
          ramp above the wilting point
        - Brilli et al. (2007): Drought response of isoprene emission
        Both support the linear-ramp shape; ramping over the whole wilting-point
        to field-capacity range is a RELIC reduced choice.

        Parameters
        ----------
        soil_moisture : Tensor, optional
            Relative soil moisture [0-1] between wilting point and field capacity
        water_potential : Tensor, optional
            Soil water potential [MPa]
        wilting_point : float, optional
            Permanent wilting point [MPa]; the ParameterManager value when omitted
        field_capacity : float, optional
            Field capacity [MPa]; the ParameterManager value when omitted

        Returns
        -------
        Tensor
            Drought stress factor gamma_W [0-1]
        """
        if water_potential is not None:
            manager = _resolve_manager(param_manager)
            wilting_point_value = float(wilting_point) if wilting_point is not None else float(
                _require_scalar(
                    manager,
                    "wilting_point_potential",
                    "biosphere.stress.water",
                    "MPa",
                    required=True,
                )
            )
            field_capacity_value = float(field_capacity) if field_capacity is not None else float(
                _require_scalar(
                    manager,
                    "field_capacity_potential",
                    "biosphere.stress.water",
                    "MPa",
                    required=True,
                )
            )
            # References: Guenther et al. (2006); Guenther et al. (2012) Eq. (13) linear
            #   soil-moisture ramp.
            # Mathematical expression: gamma_W = clip((psi - psi_wp)/(psi_fc - psi_wp), 0, 1).
            # Parameter meanings: psi is soil water potential, psi_wp the wilting point and
            #   psi_fc the field capacity.
            # Units: psi, psi_wp, psi_fc [MPa]; gamma_W [1].
            # Domain/constraints: psi_fc > psi_wp; the ramp saturates at 0 below the wilting
            #   point and at 1 above field capacity by definition of the ramp.
            # Sign convention: gamma_W is a non-negative multiplier of the emission magnitude.
            return stress_response_functions.water_stress_linear(
                water_potential, wilting_point_value, field_capacity_value)
        if soil_moisture is not None:
            # References: Guenther et al. (2006) linear soil-moisture ramp; RELIC reduced
            #   choice of the relative wilting-point to field-capacity range.
            # Mathematical expression: gamma_W = theta_rel.
            # Parameter meanings: theta_rel is the relative soil moisture between wilting
            #   point (0) and field capacity (1).
            # Units: theta_rel and gamma_W [1].
            # Domain/constraints: finite theta_rel in [0, 1]; values outside raise
            #   ValueError instead of being clamped; no lower floor is applied.
            # Sign convention: gamma_W is a non-negative multiplier of the emission magnitude.
            return _validated_driver(soil_moisture, "soil_moisture", minimum=0.0, maximum=1.0)
        raise ValueError(
            "drought_stress_response requires soil_moisture or water_potential."
        )

    @staticmethod
    def comprehensive_stress_response(state: Dict[str, Tensor]) -> Tensor:
        """
        Combined stress factor of the explicit stress drivers.

        Mathematical Expression
        -----------------------
        gamma_stress = min(gamma_W, gamma_temperature_stress, gamma_nutrient_stress)
        over the supplied factors; gamma_W is state["water_stress_factor"] or the drought
        factor of state["soil_moisture"]. Temperature enters only through gamma_T,
        so no temperature-stress default is added. Without any stress driver the
        neutral factor 1 is returned.

        Parameters
        ----------
        state : Dict[str, Tensor]
            The declared stress inputs: the contracted factor outputs
            water_stress_factor, temperature_stress_factor and nutrient_stress_factor
            of the stress Functors (1 = unstressed) and the soil_moisture driver, used
            when water_stress_factor is absent; no other mapping is read

        Returns
        -------
        Tensor
            Combined stress factor [0-1]
        """
        stress_factors = []
        if 'water_stress_factor' in state:
            stress_factors.append(
                _validated_driver(
                    state['water_stress_factor'], "water_stress_factor", minimum=0.0, maximum=1.0
                )
            )
        elif 'soil_moisture' in state:
            stress_factors.append(
                BVOCEnvironmentalResponse.drought_stress_response(
                    soil_moisture=state['soil_moisture'])
            )
        for key in ('temperature_stress_factor', 'nutrient_stress_factor'):
            if key in state:
                stress_factors.append(
                    _validated_driver(state[key], key, minimum=0.0, maximum=1.0)
                )

        if not stress_factors:
            return torch.ones(1, dtype=torch.float32)
        # References: RELIC reduced choice of the most limiting explicit stress factor;
        #   Guenther et al. (2012) Eq. (2) instead multiplies separate MEGAN activity factors.
        # Mathematical expression: gamma_stress = min_k gamma_k.
        # Parameter meanings: gamma_k are the supplied dimensionless stress factors.
        # Units: gamma_k and gamma_stress [1].
        # Domain/constraints: every gamma_k is finite and in [0, 1] (validated above); no
        #   floor or clamp is applied to the minimum.
        # Sign convention: gamma_stress is a non-negative multiplier of the emission magnitude.
        return torch.stack(stress_factors).min(dim=0)[0]


class BVOCResponseIntegrator:
    """
    Integrates all BVOC response functions.
    
    Provides unified interface for calculating all activity factors
    for BVOC emission models.

    This is a shared helper composed inside the synthesis Functors (for
    example SesquiterpeneSynthesis); it is not a Functor, declares no
    FunctorMetadata and writes no state. The executed activity factors are
    published by the owning Functors as their own diagnostics
    (diagnostics.bvoc.megan.<compound>.gamma_* and
    diagnostics.bvoc.sesquiterpene.gamma_*).
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
    ):
        """
        Initialize response integrator.
        
        Parameters
        ----------
        device : torch.device, optional
            Computation device
        parameter_manager : optional
            Explicit ParameterManager instance; defaults to global manager
        """
        self.device = device or (
            torch.device("cpu")
        )
        self.parameter_manager = _resolve_manager(parameter_manager)
        self.light_response = BVOCLightResponse(self.parameter_manager)
        self.temp_response = BVOCTemperatureResponse(self.parameter_manager)
        self.env_response = BVOCEnvironmentalResponse(self.parameter_manager)
    
    def calculate_all_activity_factors(
        self,
        compound: str,
        temperature: Tensor,
        ppfd: Tensor,
        state: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Calculate all activity factors for a BVOC compound.

        Parameters
        ----------
        compound : str
            BVOC compound class
        temperature : Tensor
            Temperature [K]
        ppfd : Tensor
            PPFD [umol m-2 s-1]
        state : Dict[str, Tensor]
            Declared stress inputs (explicit stress factors and soil_moisture)

        Returns
        -------
        Dict[str, Tensor]
            gamma_light, gamma_temperature, gamma_stress and gamma_total
        """
        gamma_light = self.light_response.get_compound_light_response(
            compound, ppfd, param_manager=self.parameter_manager)
        gamma_temp = self.temp_response.get_compound_temperature_response(
            compound, temperature, param_manager=self.parameter_manager)
        gamma_stress = self.env_response.comprehensive_stress_response(state)
        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (2).
        # Mathematical expression: gamma_total = gamma_L gamma_T gamma_stress.
        # Parameter meanings: gamma_L, gamma_T and gamma_stress are the light,
        #   temperature and combined stress activity factors.
        # Units: all factors [1].
        # Domain/constraints: each factor is evaluated by its validated owner above.
        # Sign convention: gamma_total is a non-negative multiplier of the emission magnitude.
        return {
            'gamma_light': gamma_light,
            'gamma_temperature': gamma_temp,
            'gamma_stress': gamma_stress,
            'gamma_total': gamma_light * gamma_temp * gamma_stress,
        }

    def _ensure_tensor(self, value: Union[Tensor, float, int]) -> Tensor:
        """Convert scalars to torch.Tensor on the configured device."""
        if isinstance(value, Tensor):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def calculate_integrated_response(
        self,
        state: Dict[str, Union[Tensor, float, int]],
        *,
        compound: str = "isoprene",
    ) -> Dict[str, Tensor]:
        """
        Compute the multiplicative MEGAN response (gamma_total) for a compound.

        Mathematical Expression:
            gamma_total = gamma_T x gamma_L x gamma_stress

        References:
            - Guenther et al. (2012) MEGAN2.1 formulation (temperature and light factors)
            - Brilli et al. (2007) drought response of isoprene emission
            - Niinemets et al. (2010) compound-specific activity factors

        Parameters
        ----------
        state : Dict[str, Union[Tensor, float, int]]
            One input mapping: temperature [K] (key temperature, t_leaf or
            leaf_temperature), PAR/PPFD [umol m-2 s-1] (key par, ppfd or
            photosynthetically_active_radiation), and the stress inputs
            water_stress_factor, temperature_stress_factor, nutrient_stress_factor and
            soil_moisture
        compound : str, optional
            BVOC compound class, default "isoprene"

        Returns
        -------
        Dict[str, Tensor]
            temperature_response, light_response, stress_response, integrated_response
        """
        if state is None:
            raise ValueError("state dictionary is required for response integration")

        state_tensors = {k: self._ensure_tensor(v) for k, v in state.items()}

        temperature_tensor = None
        for key in ("temperature", "t_leaf", "leaf_temperature"):
            if key in state_tensors:
                temperature_tensor = state_tensors[key]
                break
        if temperature_tensor is None:
            raise KeyError("State must include 'temperature' (K) for BVOC responses.")

        par_tensor = None
        for key in ("par", "ppfd", "photosynthetically_active_radiation"):
            if key in state_tensors:
                par_tensor = state_tensors[key]
                break
        if par_tensor is None:
            raise KeyError("State must include 'par' or 'ppfd' for BVOC responses.")

        results = self.calculate_all_activity_factors(
            compound=compound,
            temperature=temperature_tensor,
            ppfd=par_tensor,
            state=state_tensors,
        )

        integrated = {
            "temperature_response": results["gamma_temperature"],
            "light_response": results["gamma_light"],
            "stress_response": results["gamma_stress"],
            "integrated_response": results["gamma_total"],
        }
        return integrated


# Export all classes and functions
__all__ = [
    'BVOCLightResponse',
    'BVOCTemperatureResponse', 
    'BVOCEnvironmentalResponse',
    'BVOCResponseIntegrator'
]



