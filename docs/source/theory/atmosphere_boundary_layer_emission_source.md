# Boundary-Layer Emission Source

## 1. Process Definition and Scope

`BoundaryLayerEmissionSource` (`relic\scientific\atmosphere\transport\boundary_layer_emission_source.py`) is the declared hand-over of the surface isoprene molar flux into the well-mixed planetary boundary layer. It converts `isoprene_emission`, written by `megan_bvoc_model` or `mechanistic_isoprene_synthesis` on a ground-area basis, into one timestep-integrated increment of `isoprene_mixing_ratio`.

It is the source counterpart of `DryDepositionAggregator`: both treat the boundary layer as one well-mixed box and both exchange matter across its lower boundary. It owns no chemistry, no turbulent exchange across the boundary-layer top, and no vertical profile. `SurfaceFluxToVolumeSourceAdapter` serves a different target: it maps a flux to a volumetric mass source [kg m-3 s-1] in the lowest layer of a finite-volume column, which is not a mixing-ratio increment.

## 2. Governing Equations

For a surface molar flux $F$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$) entering a well-mixed layer of height $h$ ($\mathrm{m}$) with air molar density $n$ ($\mathrm{mol\,m^{-3}}$), the mixed-layer budget gives the source tendency

$$\frac{d\chi}{dt} = \frac{F}{h\,n}, \qquad n = \frac{P}{R\,T},$$

and the increment over a step $\Delta t$ ($\mathrm{s}$), with $F$, $h$, $P$ and $T$ held at their values in the step,

$$\Delta\chi = \frac{F\,\Delta t\,R\,T}{P\,h}.$$

The law is evaluated by the existing `SpeciesMapper.flux_to_mixing_ratio` (see `anthroposphere_emissions_mapping.md`); this process adds no second formula. It uses the total (moist) air molar density of the ideal-gas law, the same ideal-gas closure as the air number density $P/(k_B T)$ of `ChemistryFunctor`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $F$ | Surface isoprene molar flux, ground-area basis | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `isoprene_emission` | `input` |
| $h$ | Planetary boundary-layer height | $\mathrm{m}$ | `pbl_height` | `forcing` |
| $T$ | Air temperature | $\mathrm{K}$ | `air_temperature` | `input` |
| $P$ | Atmospheric pressure | $\mathrm{Pa}$ | `atmospheric_pressure` | `input` |
| $\chi, \Delta\chi$ | Isoprene mixing ratio and per-step source increment | $\mathrm{mol\,mol^{-1}}$ | `isoprene_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $r$ | Relative column molar balance residual | $1$ | `diagnostics.boundary_layer_emission.relative_column_balance_residual` | `diagnostic` |

## 4. Parameter Table

The process has no ParameterManager parameter. The universal gas constant $R$ is read from `relic.physical.constants.universal.gas_constant` by `SpeciesMapper.flux_to_mixing_ratio`.

## 5. Numerical Methods and Discretization

`forward()` requires finite floating-point `torch.Tensor` inputs with $F \ge 0$, $h > 0$, $T > 0$, $P > 0$ and $\Delta t > 0$; a missing or invalid value raises and no default is substituted. Inputs of different precision are promoted to the widest dtype before the law is evaluated. The increment is returned on the canonical state key through `UpdateItem(ACCUMULATE, per_timestep=True)`, so StateManager advances the state exactly once. The result is exact for constant $F$, $h$, $T$ and $P$ in the step; with a time-varying flux it is first order in $\Delta t$.

## 6. Conservation, Boundary Conditions, and Ledgers

This is an open lower-boundary source. The column molar burden gained by the box equals the molar amount crossing the boundary. The process conservation hook evaluates

$$r = \frac{\Delta\chi\,h\,n - F\,\Delta t}{F\,\Delta t}, \qquad r = 0 \text{ where } F = 0,$$

on the increment actually returned, without cancellation across cells. The hook reports $\max|r|$ under the law name `mass`. The burden gained by the box leaves the biosphere through the boundary port `isoprene_emission`; that variable has no state ledger, so the hand-over adds no hidden source or sink. There is no limiter: a non-negative source cannot overdraw a stock.

The process must not run together with any other route that places the same surface flux into `isoprene_mixing_ratio`. `AtmosphereCoupling` is not constructible at present (`runtime_dependency_unbound`), so no such route exists; `bvoc_emission_field_3d` has no reader.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `BoundaryLayerEmissionSource` | `relic\scientific\atmosphere\transport\boundary_layer_emission_source.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

The process declares `megan_bvoc_model` and `mechanistic_isoprene_synthesis` as optional same-step providers. When one of them is registered it runs first; otherwise `isoprene_emission` must be supplied as forcing.

## 8. Implementation Status

- **Runtime status**: `[I]` implemented algebraic source of one species. Other BVOC species are not covered.
- **IPHS certificate**: **Level 1** (`docs\source\theory\iphs_certificates\atmosphere_transport_boundary_layer_emission_source_boundary_layer_emission_source.yaml`, IPHS-compatible descriptive transport). No DEC form, Hodge term, port power or entropy statement is claimed.

## 9. References

1. Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
2. Vila-Guerau de Arellano, J., van Heerwaarden, C. C., van Stratum, B. J. H., & van den Dries, K. (2015). *Atmospheric Boundary Layer: Integrating Air Chemistry and Land Interactions*. Cambridge University Press.
3. Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.). Wiley.
4. U.S. EPA (2019). *CMAQ User's Guide*, emissions conversion.
