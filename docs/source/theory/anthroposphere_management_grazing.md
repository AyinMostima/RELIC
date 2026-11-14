# Livestock Grazing Biomass Removal and Excreta Partitioning

## 1. Process Definition and Scope

`GrazingManagement` (`relic/scientific/anthroposphere/management/grazing.py`) computes livestock dry-matter intake from `aboveground_biomass`, debits the grazed vegetation stock directly, and partitions ingested forage carbon and nitrogen into fecal organic return (`dung_c_flux`, `dung_n_flux`), urinary mineral nitrogen return (`urine_n_flux`), and open animal-system exports (`livestock_carbon_export_flux`, `livestock_nitrogen_export_flux`) alongside mechanical `trampling_damage`.

Five `LivestockType` classes (`cattle`, `sheep`, `goat`, `horse`, `buffalo`) are parameterized in `ParameterManager`. Stocking density $D_{\mathrm{stock}}$ ($\mathrm{animals\,ha^{-1}}$) and animal body mass $W$ ($\mathrm{kg\,animal^{-1}}$) come from the active `GrazingEvent`; event `supplement_rate` and livestock `metabolism_factor` are stored without entering the runtime intake formula (which scales linearly in $W$, corresponding to allometric exponent $b = 1$ in the historical expression). Ambient air temperature is excluded so downstream soil and atmospheric Functors own temperature-dependent decomposition and volatilization. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.5.4).

## 2. Governing Equations

### 2.1 Calendar Window, Saturating Forage Intake, and Donor-Stock Cap

An event starting on day $d_{\mathrm{start}}$ with duration $\tau_{\mathrm{dur}} \in [0, Y(C)]$ is active when $\delta_C = (d - d_{\mathrm{start}}) \bmod Y(C) < \tau_{\mathrm{dur}}$. From `aboveground_biomass` $B_{\mathrm{ag}} \ge 0$ ($\mathrm{kg\,DM\,m^{-2}}$) and `vegetation_quality` $Q \in [0, 1]$, diagnostic grazing pressure is $P_g = D_{\mathrm{stock}}\,Q$ ($\mathrm{animals\,ha^{-1}}$), and uncapped daily areal intake $I_{\mathrm{raw}}$ ($\mathrm{kg\,DM\,m^{-2}\,d^{-1}}$) follows Hodgson (1990):

$$
I_{\mathrm{raw}} = W\,\frac{a}{100}\,Q\,s_Q\bigl(1 - \exp(-10000\,k\,B_{\mathrm{ag}})\bigr)\frac{D_{\mathrm{stock}}}{10000}
$$

Applying daily utilization fraction $U_{\max}$ and non-negativity of donor stock over timestep $\Delta t$ yields bounded removal flux $F_{\mathrm{graz}}$ ($\mathrm{kg\,DM\,m^{-2}\,s^{-1}}$) and integrated donor decrement $\Delta B_{\mathrm{ag}}$:

$$
I_{\mathrm{day}} = \min\!\left(I_{\mathrm{raw}},\; B_{\mathrm{ag}}\,U_{\max},\; B_{\mathrm{ag}}\frac{s_{\mathrm{day}}}{\Delta t}\right), \qquad
F_{\mathrm{graz}} = \frac{I_{\mathrm{day}}}{s_{\mathrm{day}}}, \qquad
\Delta B_{\mathrm{ag}} = -F_{\mathrm{graz}}\,\Delta t
$$

### 2.2 Carbon-Nitrogen Excreta Partition, Trampling, and Structural Mobility

With forage carbon fraction $f_{\mathrm{C}}$, livestock forage nitrogen content $f_{\mathrm{N}}$, digestibility $d_{\mathrm{dig}} \in [0, 1]$, animal nitrogen retention $f_{\mathrm{N,ret}} \in [0, 1]$, and positive excreta weights $(w_{\mathrm{dung}}, w_{\mathrm{urine}})$, boundary rates ($\mathrm{kg\,element\,m^{-2}\,s^{-1}}$) close as:

$$
F_{\mathrm{C,in}} = f_{\mathrm{C}}\,F_{\mathrm{graz}}, \qquad
F_{\mathrm{C,dung}} = (1 - d_{\mathrm{dig}})\,f_{\mathrm{C}}\,F_{\mathrm{graz}}, \qquad
F_{\mathrm{C,exp}} = F_{\mathrm{C,in}} - F_{\mathrm{C,dung}}
$$

$$
F_{\mathrm{N,in}} = f_{\mathrm{N}}\,F_{\mathrm{graz}}, \qquad
F_{\mathrm{N,dung}} = \frac{w_{\mathrm{dung}}\,(1 - f_{\mathrm{N,ret}})}{w_{\mathrm{dung}} + w_{\mathrm{urine}}}\,F_{\mathrm{N,in}}, \qquad
F_{\mathrm{N,urine}} = \frac{w_{\mathrm{urine}}\,(1 - f_{\mathrm{N,ret}})}{w_{\mathrm{dung}} + w_{\mathrm{urine}}}\,F_{\mathrm{N,in}}
$$

$$
F_{\mathrm{N,exp}} = F_{\mathrm{N,in}} - F_{\mathrm{N,dung}} - F_{\mathrm{N,urine}}
$$

Moisture-amplified trampling severity $D_{\mathrm{tramp}} \in [c_{\min}, c_{\max}]$ and the four-channel positive diagonal Hodge mobility $H_{\mathrm{graz}}$ (`max_utilization`, `digestibility_coefficient`, `n_retention`, `forage_carbon_fraction`) are

$$
D_{\mathrm{tramp}} = \operatorname{clip}\!\left(f_{\mathrm{tramp}}\,i_{\mathrm{tramp}}\,\frac{D_{\mathrm{stock}}}{D_{\mathrm{ref}}}\bigl(1 + s_\theta \max(\theta - \theta_0,\; 0)\bigr),\; c_{\min},\; c_{\max}\right), \qquad
H_{\mathrm{graz}} = \operatorname{diag}\!\left(\frac{w_i}{\sum_{j=1}^{4} w_j}\right) \succ 0
$$

Nine rate channels pass through `advance_reaction_network` ($N_h = I_9$, `open_system_route="port"`).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $B_{\mathrm{ag}}$ | Aboveground dry-matter biomass stock (input and `ACCUMULATE` debit) | $\mathrm{kg\,m^{-2}}$ | `aboveground_biomass` | `state` |
| $Q, \theta$ | Canonical forage quality factor and volumetric soil moisture | $\mathrm{1}$, $\mathrm{m^3\,m^{-3}}$ | `vegetation_quality`, `soil_moisture` | `state` |
| $d$ | Optional explicit simulator day-of-year coordinate | $\mathrm{d}$ | `day_of_year` | `forcing` |
| $F_{\mathrm{graz}}$ | Dry-matter grazing removal mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `grazing_flux` | `flux` |
| $F_{\mathrm{C,dung}}, F_{\mathrm{N,dung}}, F_{\mathrm{N,urine}}$ | Fecal organic C/N and separate urinary mineral-N soil return fluxes | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `dung_c_flux`, `dung_n_flux`, `urine_n_flux` | `flux` |
| $F_{\mathrm{C,in}}, F_{\mathrm{C,exp}}, F_{\mathrm{N,in}}, F_{\mathrm{N,exp}}$ | Livestock C/N intake and open animal-system export fluxes | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$, $\mathrm{kg\,N\,m^{-2}\,s^{-1}}$ | `livestock_carbon_intake_flux`, `livestock_carbon_export_flux`, `livestock_nitrogen_intake_flux`, `livestock_nitrogen_export_flux` | `flux` |
| $I_{\mathrm{cum}}, t_{\mathrm{graz}}$ | Integrated dry-matter intake and elapsed active grazing days | $\mathrm{kg\,m^{-2}}$, $\mathrm{d}$ | `cumulative_intake`, `days_grazed` | `state` |
| $D_{\mathrm{tramp}}, g_{\mathrm{act}}, D_{\mathrm{stock}}, u_{\mathrm{util}}, P_g$ | Trampling severity, active flag, stocking density, utilization, and grazing pressure | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{animals\,ha^{-1}}$, $\mathrm{1}$, $\mathrm{animals\,ha^{-1}}$ | `trampling_damage`, `grazing_active`, `current_stocking_rate`, `utilization_rate`, `diagnostics.grazing_pressure` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $s_Q, k, U_{\max}, f_{\mathrm{C}}$ | `anthroposphere.management.grazing.selectivity_factor`, `anthroposphere.management.grazing.functional_response_scalar`, `anthroposphere.management.grazing.max_utilization`, `anthroposphere.management.grazing.forage_carbon_fraction` | $\mathrm{1}$, $\mathrm{ha\,kg^{-1}}$, $\mathrm{1}$, $\mathrm{1}$ | `anthroposphere/management.yaml` | Hodgson (1990); Milchunas and Lauenroth (1993) |
| $d_0, f_{\mathrm{N,ret}}, w_{\mathrm{dung}}, w_{\mathrm{urine}}$ | `anthroposphere.management.grazing.digestibility_coefficient`, `anthroposphere.management.grazing.n_retention`, `anthroposphere.management.grazing.dung_partition`, `anthroposphere.management.grazing.urine_partition` | $\mathrm{1}$ | `anthroposphere/management.yaml` | Saggar et al. (2007); Conant et al. (2001) |
| $f_{\mathrm{tramp}}, \theta_0, s_\theta, D_{\mathrm{ref}}$ | `anthroposphere.management.grazing.trampling_factor`, `anthroposphere.management.grazing.trampling_moisture_threshold`, `anthroposphere.management.grazing.trampling_moisture_scale`, `anthroposphere.management.grazing.trampling_stocking_normalization` | $\mathrm{1}$, $\mathrm{1}$, $\mathrm{1}$, $\mathrm{animals\,ha^{-1}}$ | `anthroposphere/management.yaml` | Greenwood and McKenzie (2001) |
| $a, d_{\mathrm{dig}}, f_{\mathrm{N}}, i_{\mathrm{tramp}}$ | `anthroposphere.management.grazing.livestock.*`, `anthroposphere.management.grazing.default_schedule.*` | $\%~\mathrm{d^{-1}}$, $\mathrm{1}$, $\mathrm{d}$, $\mathrm{animals\,ha^{-1}}$, $\mathrm{kg}$ | `anthroposphere/management.yaml` | Species intake and schedule defaults |
| $c_{\min}, c_{\max}, \tau_{\mathrm{cons}}$ | `anthroposphere.management.clamp_min`, `anthroposphere.management.clamp_max`, `validation.conservation.conservation_tolerance` | $\mathrm{1}$ | `anthroposphere/management.yaml`, `validation/conservation.yaml` | Clamping and conservation tolerances |
| $s_{\mathrm{day}}$ | `relic.physical.constants.time.seconds_per_day` | $\mathrm{s\,d^{-1}}$ | `physical/constants.yaml` | SI day length ($86400$) |

## 5. Numerical Methods and Discretization

`GrazingManagement` holds no internal prognostic tensor buffers. Step-integrated quantities (`aboveground_biomass` with $-\Delta t F_{\mathrm{graz}}$, `cumulative_intake` with $+\Delta t F_{\mathrm{graz}}$, and `days_grazed` with $g_{\mathrm{act}}\Delta t / s_{\mathrm{day}}$) use `UpdateItem(UpdateMode.ACCUMULATE, per_timestep=True)`. Twelve instantaneous flux and status channels use `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`, and `diagnostics.grazing_pressure` plus `diagnostics.grazing_management.canonical_iphs_runtime` are emitted as diagnostic tensors.

## 6. Conservation, Boundary Conditions, and Ledgers

`GrazingManagement.check_conservation_laws()` validates `UpdateItem` modes, units, shapes, dtypes, devices, and finiteness before evaluating the three normalized `grazing_matter_boundary` ledgers:

$$
R_{\mathrm{DM}} = \Delta B_{\mathrm{ag}} + I_{\mathrm{step}} = 0, \qquad
R_{\mathrm{C}} = F_{\mathrm{C,in}} - F_{\mathrm{C,dung}} - F_{\mathrm{C,exp}} = 0, \qquad
R_{\mathrm{N}} = F_{\mathrm{N,in}} - F_{\mathrm{N,dung}} - F_{\mathrm{N,urine}} - F_{\mathrm{N,exp}} = 0
$$

Downstream `LitterfallInterfaceMapper` pairs `dung_c_flux` and `dung_n_flux` into organic pedosphere inputs while keeping `urine_n_flux` as a distinct mineral-nitrogen boundary.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `GrazingManagement` | `relic/scientific/anthroposphere/management/grazing.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`relic/core/functor/biological_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |

## 8. Implementation Status

- `GrazingManagement`: `[I]`, IPHS **Level 3** (`reaction`, certificate `anthroposphere_management_grazing_grazing_management.yaml`). Saturating livestock dry-matter removal with direct `aboveground_biomass` debit, explicit dung/urine/export C and N boundary closure, and positive diagonal Hodge mobility $H_{\mathrm{graz}}$.

Frozen certificate expressions:

```text
I = a W^b D Q (1 - exp(-k B)); N_return = I N_content (1 - N_retention).
The intake and excreta equations define entries of the standard reaction vector r_h inside the single IPHS biomass-return port map.
```

## 9. References

- Conant, R. T., Paustian, K., and Elliott, E. T. (2001). Grassland management and conversion into grassland: effects on soil carbon. *Ecological Applications*, 11(2), 343-355.
- Greenwood, K. L., and McKenzie, B. M. (2001). Grazing effects on soil physical properties and the consequences for pastures: a review. *Australian Journal of Experimental Agriculture*, 41(8), 1231-1250.
- Hodgson, J. (1990). *Grazing Management: Science into Practice*. Longman Scientific & Technical.
- Milchunas, D. G., and Lauenroth, W. K. (1993). Quantitative effects of grazing on vegetation and soils over a global range of environments. *Ecological Monographs*, 63(4), 327-366.
- Saggar, S., Hedley, C. B., Giltrap, D. L., and Lambie, S. M. (2007). Measured and modelled estimates of nitrous oxide emission and methane consumption from a sheep-grazed pasture. *Agriculture, Ecosystems & Environment*, 122(3), 357-365.
