# Atmospheric Environmental Forcing (VPD)

## 1. Process Definition and Scope

`EnvironmentalForcingFunctor` (`relic\scientific\atmosphere\environmental_forcing.py`) is the single preprocessing owner of `vapor_pressure_deficit` ($\mathrm{Pa}$), computed pointwise from external `air_temperature` ($\mathrm{K}$) and `relative_humidity` ($1$). Psychrometric relations for atmospheric forcing appear in *Methods of Ecological Analysis*, Chapters 17 and 19.

External meteorological adapters and `StateManager` own raw forcing ingestion. This Functor does not echo its inputs, supply default meteorology, or compute radiation, precipitation, $\mathrm{CO_2}$, or surface heat fluxes. Being a point model (`SPATIAL_DIMS = None`), `get_value_at` raises `NotImplementedError`.

## 2. Governing Equations

Given Kelvin air temperature $T_K$ and dimensionless relative humidity $\mathrm{RH} \in [0, 1]$, Celsius temperature $T_C$ and Magnus-Tetens saturation vapor pressure $e_s$ ($\mathrm{kPa}$) are

$$T_C = T_K - T_0, \qquad e_s(T_C) = e_{s0}\exp\!\left(\frac{a\,T_C}{T_C + b}\right).$$

Intermediate vapor pressure deficit in $\mathrm{kPa}$ and canonical output in $\mathrm{Pa}$ are

$$\mathrm{VPD}_{\mathrm{kPa}} = \max\!\left(e_s(T_C)(1 - \mathrm{RH}),\; 0\right), \qquad \mathrm{VPD}_{\mathrm{Pa}} = \operatorname{kPa\_to\_Pa}\!\left(\mathrm{VPD}_{\mathrm{kPa}}\right),$$

where unit scaling uses `relic\data\standards\unit_converter.py` (`UnitConverter.kpa_to_pa`) without a duplicated literal conversion factor.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T_K$ | Near-surface air temperature | $\mathrm{K}$ | `air_temperature` | `forcing` (input) |
| $\mathrm{RH}$ | Relative humidity in $[0, 1]$ | $1$ | `relative_humidity` | `forcing` (input) |
| $\mathrm{VPD}_{\mathrm{Pa}}$ | Derived atmospheric vapor pressure deficit | $\mathrm{Pa}$ | `vapor_pressure_deficit` | `forcing` (output, `REPLACE`, `per_timestep=True`) |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $e_{s0}$ | `atmosphere.environmental_forcing.vpd.es_coefficient` | $\mathrm{kPa}$ | `atmosphere/environmental_forcing.yaml` (`0.6108`) | Allen et al. (1998), FAO-56 Eq. 11 |
| $a$ | `atmosphere.environmental_forcing.vpd.a_coefficient` | $1$ | `atmosphere/environmental_forcing.yaml` (`17.27`) | Allen et al. (1998), FAO-56 Eq. 11 |
| $b$ | `atmosphere.environmental_forcing.vpd.b_coefficient` | $\mathrm{^\circ C}$ | `atmosphere/environmental_forcing.yaml` (`237.3`) | Allen et al. (1998), FAO-56 Eq. 11 |
| $T_0$ | `get_physical_constant("celsius_to_kelvin")` | $\mathrm{K}$ | `physical/constants.yaml` (`273.15`) | Physical-constants SSOT |

## 5. Numerical Methods and Discretization

`forward()` is purely algebraic and instantaneous (independent of `dt`). Both inputs must be non-empty, finite floating-point `torch.Tensor` objects. Any $\mathrm{RH} \notin [0, 1]$ or temperature hitting the Magnus denominator singularity $T_C + b = 0$ (producing non-finite output) raises `ValueError` immediately without clamping or silent repair. Continuous tensor operations preserve `torch.autograd` differentiability across the valid interior domain.

## 6. Conservation, Boundary Conditions, and Ledgers

`conservation_laws` is empty (`[]`). External `air_temperature` and `relative_humidity` act as open forcing ports (`y_h = G_h u_forcing`), and no internal mass or energy reservoir is integrated.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `EnvironmentalForcingFunctor` | `relic\scientific\atmosphere\environmental_forcing.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.PREPROCESSING` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented preprocessing forcing-derivation Functor emitting `UpdateItem(REPLACE, per_timestep=True, unit="Pa")` with writer provenance `environmental_forcing`.
- **IPHS certificate**: **Level 2** (`docs\source\theory\iphs_certificates\atmosphere_environmental_forcing_environmental_forcing_functor.yaml`, IPHS-wrapped port process). No internal Hamiltonian state, storage functional, or conjugate port-power balance is claimed.

## 9. References

1. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO.
2. Campbell, G. S., & Norman, J. M. (1998). *An Introduction to Environmental Biophysics* (2nd ed.). Springer.
3. Monteith, J. L., & Unsworth, M. H. (2008). *Principles of Environmental Physics* (3rd ed.). Academic Press.
