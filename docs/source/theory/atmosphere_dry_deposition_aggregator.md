# Atmospheric Dry Deposition Aggregator

## 1. Process Definition and Scope

`DryDepositionAggregator` (`relic\scientific\atmosphere\transport\dry_deposition_aggregator.py`) computes surface dry-deposition removal from a well-mixed planetary boundary layer for the four canonical gas-phase species `no`, `no2`, `o3`, and `so2`. Theoretical background on resistance-in-series surface uptake appears in *Methods of Ecological Analysis*, Chapters 17 and 18.

By default its sole upstream scheduler dependency is `atmospheric_turbulence`, which supplies `aerodynamic_resistance`. The GraphIR selection field `surface_layer_source = "external_forcing"` removes that dependency: $R_a$ then arrives through `State:ExternalForcing`, for example from an observed tower series, and the law below is unchanged. A missing value is refused before the first step and no default is substituted. A graph with this selection must not also contain an `atmospheric_turbulence` node, because that node would replace the forcing. Constructor or `config["parameters"]` numerical overrides are rejected; all numerical settings are read exclusively from `ParameterManager`. When `atmosphere.dry_deposition.enabled` is false, the Functor emits `diagnostics.dry_deposition.enabled = 0` and zero mass-balance residual without registering state updates.

## 2. Governing Equations

For each canonical species $s \in \{\mathrm{no}, \mathrm{no2}, \mathrm{o3}, \mathrm{so2}\}$ with effective non-aerodynamic resistance $R_{\mathrm{eff},s} = R_{b,s} + R_{c,s}$ ($\mathrm{s\,m^{-1}}$) and aerodynamic resistance $R_a$ ($\mathrm{s\,m^{-1}}$), the series deposition velocity is

$$v_{d,s} = \frac{1}{R_a + R_{\mathrm{eff},s}}.$$

Over a well-mixed planetary boundary layer of height $h_{\mathrm{pbl}}$ ($\mathrm{m}$) and positive step $\Delta t$ ($\mathrm{s}$), the unconstrained tendency, potential normalized export $P_s$, availability-limited export $E_s$, and submitted mixing-ratio increment $\Delta C_s$ are

$$\frac{dC_s}{dt} = -\frac{v_{d,s}}{h_{\mathrm{pbl}}} C_s, \qquad P_s = \frac{v_{d,s} C_s \Delta t}{h_{\mathrm{pbl}}},$$

$$E_s = \min(P_s, C_s), \qquad L_s = P_s - E_s, \qquad \Delta C_s = -E_s = \max\!\left(-C_s,\; -\frac{v_{d,s} C_s \Delta t}{h_{\mathrm{pbl}}}\right).$$

In discrete structural cochain notation, the Level 3 transport audit form is

$$\dot{x}_h = -D_h q_h + G_h u_h, \qquad q_h = \star_{\mathrm{dep}} C_h, \qquad y_h = G_h^{\mathsf{T}} e_h,$$

where $C_h$ is the species mixing-ratio 0-cochain ($\mathrm{mol\,mol^{-1}}$), $q_h$ is the surface sink cochain ($\mathrm{mol\,mol^{-1}\,s^{-1}}$), $\star_{\mathrm{dep}}$ is the non-negative deposition mobility, and $G_h u_h$ is the open surface export port.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $C_{\mathrm{NO}}, \Delta C_{\mathrm{NO}}$ | Nitric oxide mixing ratio and per-step sink increment | $\mathrm{mol\,mol^{-1}}$ | `no_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{NO_2}}, \Delta C_{\mathrm{NO_2}}$ | Nitrogen dioxide mixing ratio and per-step sink increment | $\mathrm{mol\,mol^{-1}}$ | `no2_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{O_3}}, \Delta C_{\mathrm{O_3}}$ | Ozone mixing ratio and per-step sink increment | $\mathrm{mol\,mol^{-1}}$ | `o3_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $C_{\mathrm{SO_2}}, \Delta C_{\mathrm{SO_2}}$ | Sulfur dioxide mixing ratio and per-step sink increment | $\mathrm{mol\,mol^{-1}}$ | `so2_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $h_{\mathrm{pbl}}$ | Planetary boundary-layer height | $\mathrm{m}$ | `pbl_height` | `forcing` |
| $R_a$ | Aerodynamic scalar resistance from `atmospheric_turbulence` or external forcing | $\mathrm{s\,m^{-1}}$ | `aerodynamic_resistance` | `diagnostic` / upstream input or forcing |
| $\sum_s \Delta C_s$ | Sum of species mixing-ratio increments | $\mathrm{mol\,mol^{-1}}$ | `diagnostics.dry_deposition.total_mixing_ratio_increment` | `diagnostic` |
| $\sum_s L_s$ | Total availability-limiter difference $\sum_s (P_s - E_s)$ | $\mathrm{mol\,mol^{-1}}$ | `diagnostics.dry_deposition.total_limiter_residual` | `diagnostic` |
| $\max_s \lvert R_s \rvert$ | Per-species open-boundary balance residual | $\mathrm{mol\,mol^{-1}}$ | `diagnostics.dry_deposition.normalized_mass_balance_residual` | `diagnostic` |
| $I_{\mathrm{on}}$ | Runtime activation indicator (`0` or `1`) | $1$ | `diagnostics.dry_deposition.enabled` | `diagnostic` |

## 4. Parameter Table

| Symbol | `ParameterManager` Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $I_{\mathrm{on}}$ | `atmosphere.dry_deposition.enabled` | $1$ | `atmosphere/dry_deposition.yaml` | Runtime activation switch |
| $\mathcal{S}$ | `atmosphere.dry_deposition.species_list` | $1$ | `atmosphere/dry_deposition.yaml` (`[no, no2, o3, so2]`) | Wesely (1989) |
| $z_{\mathrm{ref}}$ | `atmosphere.dry_deposition.reference_height` | $\mathrm{m}$ | `atmosphere/dry_deposition.yaml` | Wesely (1989) |
| $v_{d,\mathrm{ref}}$ | `atmosphere.dry_deposition.structural_reference_deposition_velocity` | $\mathrm{m\,s^{-1}}$ | `atmosphere/dry_deposition.yaml` | Static Hodge mobility scale |
| $R_{\mathrm{eff,NO}}$ | `atmosphere.dry_deposition.species.no.effective_non_aerodynamic_resistance` | $\mathrm{s\,m^{-1}}$ | `atmosphere/dry_deposition.yaml` | Wesely (1989); Zhang et al. (2003) |
| $R_{\mathrm{eff,NO_2}}$ | `atmosphere.dry_deposition.species.no2.effective_non_aerodynamic_resistance` | $\mathrm{s\,m^{-1}}$ | `atmosphere/dry_deposition.yaml` | Wesely (1989); Zhang et al. (2003) |
| $R_{\mathrm{eff,O_3}}$ | `atmosphere.dry_deposition.species.o3.effective_non_aerodynamic_resistance` | $\mathrm{s\,m^{-1}}$ | `atmosphere/dry_deposition.yaml` | Wesely (1989); Zhang et al. (2003) |
| $R_{\mathrm{eff,SO_2}}$ | `atmosphere.dry_deposition.species.so2.effective_non_aerodynamic_resistance` | $\mathrm{s\,m^{-1}}$ | `atmosphere/dry_deposition.yaml` | Wesely (1989); Zhang et al. (2003) |

## 5. Numerical Methods and Discretization

`forward()` requires finite `torch.Tensor` inputs with $C_s \ge 0$, $h_{\mathrm{pbl}} > 0$, $R_a > 0$, and $\Delta t > 0$; missing or invalid values fail fast without fallback defaults. The explicit single-step update limits the negative tendency to $-C_s / \Delta t$ prior to integration so one step never extracts more mixing ratio than is present in the column, while keeping $\Delta C_s \le 0$ for all positive inputs. Increments are returned directly on the canonical state keys via `UpdateItem(ACCUMULATE, per_timestep=True)` without `delta_*` aliases.

## 6. Conservation, Boundary Conditions, and Ledgers

Dry deposition is an open lower-boundary atmospheric sink. For every species $s$, the process conservation hook verifies the exact normalized mixing-ratio ledger

$$R_s = \Delta C_s + P_s - L_s = 0.$$

A positive availability-limiter diagnostic $L_s > 0$ records mass demand denied by finite column inventory and is accounted for explicitly in $R_s$. Conversion to physical molar surface flux ($\mathrm{mol\,m^{-2}\,s^{-1}}$) requires air molar density and is outside this mixing-ratio ledger.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `DryDepositionAggregator` | `relic\scientific\atmosphere\transport\dry_deposition_aggregator.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented multi-species dry-deposition sink Functor coupled downstream of `atmospheric_turbulence`, or of external forcing under `surface_layer_source = "external_forcing"`.
- **IPHS certificate**: **Level 3** (`docs\source\theory\iphs_certificates\atmosphere_transport_dry_deposition_aggregator_dry_deposition_aggregator.yaml`, IPHS-structured transport). No thermodynamic storage functional or conjugate surface-receptor port-power closure is claimed.

## 9. References

1. Wesely, M. L. (1989). Parameterization of surface resistances to gaseous dry deposition in regional-scale numerical models. *Atmospheric Environment*, 23(6), 1293-1304.
2. Zhang, L., Brook, J. R., & Vet, R. (2003). A revised parameterization for gaseous dry deposition in air-quality models. *Atmospheric Chemistry and Physics*, 3, 2067-2082.
3. Baldocchi, D. D., Hicks, B. B., & Camara, P. (1987). A canopy stomatal resistance model for gaseous deposition to vegetated surfaces. *Atmospheric Environment*, 21(1), 91-101.
