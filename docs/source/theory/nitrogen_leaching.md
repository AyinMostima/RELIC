# Soil Nitrogen Leaching

## 1. Process Definition and Scope

`NitrogenLeachingProcess` (`relic/scientific/pedosphere/nutrient_cycling/nitrogen_leaching.py`) computes vertical advective export of dissolved soil nitrate ($\mathrm{NO_3^-}$) carried by hydrological drainage below the active soil column (`[I]`). The Functor couples soil nitrate inventory (`no3_nitrogen`) to hydrological mass drainage (`drainage_flux`), volumetric liquid water content (`soil_moisture`), active layer thickness (`soil_depth`), and soil bulk density (`bulk_density`) using linear equilibrium adsorption retardation and explicit donor-pool inventory safeguards.

All five geometric, hydraulic, and chemical inputs are mandatory runtime tensors in `state`; no `ParameterManager` fallback defaults exist for `soil_depth` or `bulk_density`, and non-canonical drainage aliases are rejected. Multi-layer dispersive-advective nitrate profiling is owned separately by general 1-D solute transport (`SoluteAdvectionDiffusion1D`), whereas `NitrogenLeachingProcess` represents grid-cell root-zone drainage export (`SpatialScale.GRID_CELL`).

## 2. Governing Equations

Hydrological drainage mass flux $F_{\mathrm{water}} \ge 0$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) is converted to volumetric Darcy drainage flux $q_{\mathrm{drain}}$ ($\mathrm{m\,s^{-1}}$) using liquid water density $\rho_w$ ($\mathrm{kg\,m^{-3}}$):

$$
q_{\mathrm{drain}} = \frac{F_{\mathrm{water}}}{\rho_w}.
$$

Over active soil depth $z_{\mathrm{soil}} > 0$ ($\mathrm{m}$) with volumetric water content $\theta \in (0, 1]$ ($\mathrm{m^3\,m^{-3}}$), the liquid water volume per unit ground area is $V_w = \theta\,z_{\mathrm{soil}}$ ($\mathrm{m}$), yielding bulk-solution nitrate concentration

$$
C_{\mathrm{NO_3}} = \frac{N_{\mathrm{NO_3}}}{V_w} = \frac{N_{\mathrm{NO_3}}}{\theta\,z_{\mathrm{soil}}}.
$$

Linear equilibrium partitioning between soil solution and solid matrix with distribution coefficient $K_d \ge 0$ ($\mathrm{L\,kg^{-1}}$), liter-to-cubic-meter factor $c_{\mathrm{L}} = 10^{-3}\,\mathrm{m^3\,L^{-1}}$, and dry bulk density $\rho_b > 0$ ($\mathrm{kg\,m^{-3}}$) defines the dimensionless retardation factor $R$ and mobile dissolved fraction $f_{\mathrm{mobile}}$:

$$
R = 1 + \frac{\rho_b\,\left(K_d\,c_{\mathrm{L}}\right)}{\theta}, \qquad f_{\mathrm{mobile}} = \frac{1}{R}.
$$

The raw advective leaching rate $F_{N,\mathrm{raw}}$ ($\mathrm{g\,N\,m^{-2}\,s^{-1}}$) and time-scale-bounded potential rate $F_{N,\tau}$ are

$$
F_{N,\mathrm{raw}} = q_{\mathrm{drain}}\,C_{\mathrm{NO_3}}\,f_{\mathrm{mobile}}, \qquad F_{N,\tau} = \min\left(F_{N,\mathrm{raw}},\,\frac{N_{\mathrm{NO_3}}}{\tau_{\mathrm{leach}}}\right),
$$

where $\tau_{\mathrm{leach}} > 0$ ($\mathrm{s}$) is the minimum nitrate export time constant. Over time step $\Delta t > 0$ ($\mathrm{s}$), `NumericalStabilityController.limit_fluxes` enforces the donor-pool fraction bound $f_{\max}$ above minimum pool threshold $N_{\mathrm{NO_3},\min}$:

$$
F_{\mathrm{exec}} = \mathcal{L}_{\mathrm{donor}}\!\left(F_{N,\tau};\,N_{\mathrm{NO_3}},\,\Delta t,\,f_{\max},\,N_{\mathrm{NO_3},\min}\right) \le \frac{f_{\max}\,N_{\mathrm{NO_3}}}{\Delta t},
$$

$$
\frac{d N_{\mathrm{NO_3}}}{dt} = -F_{\mathrm{exec}}, \qquad \Delta N_{\mathrm{NO_3}} = -F_{\mathrm{exec}}\,\Delta t.
$$

In IPHS Level 3 structured transport notation, open drainage export is expressed on primal 0-cochains ($x_h = N_{\mathrm{NO_3}}$, $C_{\mathrm{NO_3}}$) and dual boundary 2-cochain ($q_h = F_{\mathrm{exec}}$) as

$$
\frac{d x_h}{dt} = -D_h q_h + G_h u_h, \qquad q_h = \star_Q C_{\mathrm{NO_3}}, \qquad y_h = G_h^{\mathsf{T}} e_h,
$$

with runtime dissipative mobility $M_h = \mathrm{diag}(q_{\mathrm{drain}}\,f_{\mathrm{mobile}})$ ($\mathrm{m\,s^{-1}}$) and outward matter port `nitrate_drainage_export`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| :--- | :--- | :--- | :--- | :--- |
| $N_{\mathrm{NO_3}}$ | Soil nitrate pool inventory | $\mathrm{g\,N\,m^{-2}}$ | `no3_nitrogen` | `input` / `output` (`state increment`) |
| $\theta$ | Soil volumetric liquid water content | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `input` |
| $F_{\mathrm{water}}$ | Hydrological water drainage mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `input` |
| $z_{\mathrm{soil}}$ | Active soil layer thickness | $\mathrm{m}$ | `soil_depth` | `input` |
| $\rho_b$ | Dry soil bulk density | $\mathrm{kg\,m^{-3}}$ | `bulk_density` | `input` |
| $F_{\mathrm{exec}}$ | Realized outward nitrate leaching flux | $\mathrm{g\,N\,m^{-2}\,s^{-1}}$ | `nitrogen_leaching_flux` | `output` (`flux`) |
| $C_{\mathrm{NO_3}}$ | Nitrate concentration in soil water | $\mathrm{g\,N\,m^{-3}}$ | `diagnostics.nitrogen_leaching.nitrate_concentration` | `diagnostic` |
| $f_{\mathrm{mobile}}$ | Mobile solution fraction $1 / R$ | $1$ | `diagnostics.nitrogen_leaching.mobile_fraction` | `diagnostic` |
| $V_w$ | Soil water column per unit area $\theta\,z_{\mathrm{soil}}$ | $\mathrm{m}$ | `diagnostics.nitrogen_leaching.soil_water_volume` | `diagnostic` |
| $\Delta t$ | Executed step duration for public ledger closure | $\mathrm{s}$ | `diagnostics.nitrogen_leaching.step_seconds` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| :--- | :--- | :--- | :--- | :--- |
| $\tau_{\mathrm{leach}}$ | `pedosphere.nutrient_cycling.leaching_time_constant` | $\mathrm{s}$ | `pedosphere/nutrient_cycling.yaml` ($3600.0$) | Li et al. (2006) |
| $N_{\mathrm{NO_3},\min}$ | `pedosphere.nutrient_cycling.min_no3_pool` | $\mathrm{g\,N\,m^{-2}}$ | `pedosphere/nutrient_cycling.yaml` ($10^{-8}$) | Numerical stability floor |
| $K_d$ | `pedosphere.transport.nitrate_distribution_coefficient` | $\mathrm{L\,kg^{-1}}$ | `pedosphere/transport.yaml` ($0.1$) | van Genuchten and Wierenga (1986) |
| $z_{\mathrm{ref}}$ | `pedosphere.nutrient_cycling.hodge_reference_soil_depth` | $\mathrm{m}$ | `pedosphere/nutrient_cycling.yaml` ($1.0$) | Static Hodge metric reference depth |
| $f_{\max}$ | `pedosphere.numerical.max_flux_fraction` | $1$ | `pedosphere/numerical.yaml` ($0.5$) | Explicit donor-pool step fraction cap |
| $\varepsilon_{\mathrm{cons}}$ | `validation.conservation.conservation_tolerance` | $1$ | `validation/conservation.yaml` ($10^{-6}$) | Local ledger relative tolerance |
| $s_{\mathrm{strict}}$ | `validation.conservation.strict_conservation` | $1$ | `validation/conservation.yaml` (`false`) | Blocking conservation enforcement switch |
| $\rho_w$ | `relic.physical.constants.water.density_liquid` (alias `water_density`) | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` ($1000.0$) | IAPWS liquid water reference density |
| $c_{\mathrm{L}}$ | `relic.physical.constants.conversions.liter_to_cubic_meter` | $\mathrm{m^3\,L^{-1}}$ | `physical/constants.yaml` ($10^{-3}$) | BIPM SI Brochure |

`nitrate_distribution_coefficient` is retained as a live `ParameterManager` tensor binding without float detachment so autograd sensitivities with respect to $K_d$ propagate directly through $R$, $f_{\mathrm{mobile}}$, and $F_{\mathrm{exec}}$.

## 5. Numerical Methods and Discretization

1. **Strict Input Domain Validation**: `_compute_leaching_rates` verifies that all five required inputs (`no3_nitrogen`, `soil_moisture`, `drainage_flux`, `soil_depth`, `bulk_density`) exist in `state` and contain finite values. `no3_nitrogen` and `drainage_flux` must be non-negative; `soil_moisture` must satisfy $0 < \theta \le 1$; `soil_depth` and `bulk_density` must be strictly positive. `dt` must be a finite positive scalar.
2. **Two-Stage Explicit Donor Limiting**: `_compute_leaching_rates` first caps raw drainage export by $N_{\mathrm{NO_3}} / \tau_{\mathrm{leach}}$. `forward` then invokes `NumericalStabilityController.limit_fluxes` with `source_pool_by_flux={'nitrogen_leaching_flux': 'no3_nitrogen'}`, preventing single-step depletion beyond $f_{\max}\,N_{\mathrm{NO_3}}$ or below $N_{\mathrm{NO_3},\min}$. Unexported nitrate remains in `no3_nitrogen`.
3. **Static vs Runtime Structural Hodge Operators**: When runtime state is absent, `build_hodge_contract` constructs a static nominal diagonal Hodge operator $\star_{\mathrm{nom}} = \mathrm{diag}(\max(f_{\max}/\tau_{\mathrm{leach}}, 0))$ ($\mathrm{s^{-1}}$) tagged with `metric_keys=("pedosphere.nutrient_cycling.hodge_reference_soil_depth",)`. At runtime, `forward` always uses the forced `soil_depth` tensor and $q_{\mathrm{drain}}\,f_{\mathrm{mobile}}$ (`m s-1`). Similarly, `energy_terms` ($\frac{1}{2}\sum \max(N_{\mathrm{NO_3}},0)^2$) and `entropy_terms` ($\sum f_{\mathrm{mobile}}\,C_{\mathrm{NO_3}}^2$) provide non-negative structural audit views without establishing a thermodynamic chemical-potential balance, which is why the process is certified at IPHS Level 3 (`[I]`) and not Level 4.
4. **UpdateItem Modes**: `no3_nitrogen` is emitted with `UpdateMode.ACCUMULATE` (`per_timestep=True`, `basis="per_step"`, `unit="g N m-2"`, `element="N"`). `nitrogen_leaching_flux` (`basis="per_second"`, `unit="g N m-2 s-1"`) and all four `diagnostics.nitrogen_leaching.*` entries (`basis="instantaneous"`) use `UpdateMode.REPLACE` with `per_timestep=False`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Open Boundary Ledger**: Nitrate export leaves the pedosphere domain through the bottom drainage boundary (`nitrate_drainage_export` matter port) for downstream hydrosphere routing. Inside `forward`, `_enforce_forward_conservation` checks $\Delta N_{\mathrm{NO_3}} + F_{\mathrm{exec}}\,\Delta t = 0$.
- **Public Step Ledger**: `check_conservation_laws` calls `step_ledger_residual` using increment `("no3_nitrogen",)`, rate `("nitrogen_leaching_flux",)`, and `step_key="diagnostics.nitrogen_leaching.step_seconds"`:

$$
r_N = \Delta N_{\mathrm{NO_3}} + F_{\mathrm{exec}}\,\Delta t = 0,
$$

reporting $\max \lvert r_N \rvert$ across both declared laws (`"nitrogen"` and `"mass"`).

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| :--- | :--- | :--- | :--- | :--- |
| `NitrogenLeachingProcess` | `relic/scientific/pedosphere/nutrient_cycling/nitrogen_leaching.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

`NitrogenLeachingProcess` declares `get_dependencies() -> ["nitrification"]` merged with any instance-configured dependencies.

## 8. Implementation Status

- `NitrogenLeachingProcess` single-layer retarded advective nitrate drainage export with donor-pool stability limiting and closed open-boundary step ledger: `[I]` (certified IPHS Level 3 structured transport; certificate `pedosphere_nutrient_cycling_nitrogen_leaching_nitrogen_leaching_process.yaml`).
- Strict Level 4 thermodynamic chemical-potential storage functional and dimensional unification between the static `[s-1]` nominal Hodge matrix and runtime `[m s-1]` drainage mobility: `[N]`.

Certificate expressions (`pedosphere_nutrient_cycling_nitrogen_leaching_nitrogen_leaching_process.yaml`):

```text
legacy_expression: F_N = Q_drain * C_NO3 * mobile_fraction
iphs_expression: x_dot = -D_h q_h + G_h u_h, q_h = star_Q C_NO3, y_h = G_h^T e_h
```

## 9. References

1. van Genuchten, M. T., and Wierenga, P. J. (1986). Solute dispersion coefficients and retardation factors. In A. Klute (Ed.), *Methods of Soil Analysis, Part 1: Physical and Mineralogical Methods* (2nd ed., pp. 1025-1054). ASA and SSSA.
2. Cameron, K. C., and Haynes, R. J. (1986). Retention and movement of nitrogen in soils. In R. J. Haynes (Ed.), *Mineral Nitrogen in the Plant-Soil System* (pp. 166-241). Academic Press.
3. Jury, W. A., and Horton, R. (2004). *Soil Physics* (6th ed.). Wiley.
4. Li, C., Farahbakhshazad, N., Jaynes, D. B., Dinnes, D. L., Salas, W., and McLaughlin, D. (2006). Modeling nitrate leaching with a biogeochemical model modified based on observations in a row-crop field in Iowa. *Ecological Modelling*, 196(1-2), 116-130.
5. Manzoni, S., and Porporato, A. (2009). Soil carbon and nitrogen mineralization: Theory and models across scales. *Soil Biology and Biochemistry*, 41(7), 1355-1379.
