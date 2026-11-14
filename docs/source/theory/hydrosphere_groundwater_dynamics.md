# Hydrosphere Groundwater Dynamics

## 1. Process Definition and Scope

`GroundwaterDynamics` (`relic/scientific/hydrosphere/groundwater/dynamics.py`) evolves catchment-scale unconfined aquifer storage (`groundwater_storage`, $\mathrm{mm}$) and water-table depth (`water_table_depth`, $\mathrm{m}$) consistently through an analytical exponential-porosity profile integral and a TOPMODEL-inspired baseflow recession law.

The Functor declares optional dependencies on vadose-zone recharge producers (`bucket_soil_moisture`, `richards_soil_moisture`, `vertical_percolation_transport`) and a lagged dependency on `capillary_rise_transport`. Lateral two-dimensional Darcy groundwater routing across catchment cells and confined artesian pressure heads remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.9.1).

## 2. Governing Equations

### 2.1 Exponential Porosity Profile and Storage Bias Preservation

With surface porosity $n_0$, decay length $p$ ($\mathrm{m}$), and bedrock depth $z_b$ ($\mathrm{m}$), porosity decays with depth $z$ as $n(z) = n_0 \exp(-z / p)$ (Beven and Kirkby, 1979; Wigmosta et al., 1994). Analytical integration from water-table depth $z_{\mathrm{wt}}$ to bedrock yields theoretical profile storage ($\mathrm{mm}$, $1000\,\mathrm{mm\,m^{-1}}$) and preserved initial storage offset $S_{\mathrm{bias}}$:

$$
S_{\mathrm{prof}}(z_{\mathrm{wt}}) = 1000\,n_0 p\left[\exp\!\left(-\frac{z_{\mathrm{wt}}}{p}\right) - \exp\!\left(-\frac{z_b}{p}\right)\right], \qquad
S_{\mathrm{bias}} = S^n - S_{\mathrm{prof}}(z_{\mathrm{wt}}^n)
$$

### 2.2 Explicit Baseflow Recession

From start-of-step storage $S^n$ ($\mathrm{mm}$), water-table depth $z_{\mathrm{wt}}^n$ ($\mathrm{m}$), maximum depth $z_{\max}$, minimum depth factor $f_{\min}$, topographic index $\mathrm{TI}$, and clamped recession constant $\tau = \max(\tau_{\mathrm{gw}}, 1.0\,\mathrm{s})$ (Beven and Kirkby, 1979; Brutsaert and Nieber, 1977; Nathan and McMahon, 1990):

$$
q_{\mathrm{base}}^n = \frac{S^n}{\tau}\max\!\left(1 - \frac{z_{\mathrm{wt}}^n}{z_{\max}},\; f_{\min}\right)\exp\!\left(-\frac{\mathrm{TI}}{10}\right), \qquad
F_{\mathrm{base}} = \frac{\rho_w}{1000}\,q_{\mathrm{base}}^n
$$

where $q_{\mathrm{base}}^n$ is in $\mathrm{mm\,s^{-1}}$ and $F_{\mathrm{base}}$ (`baseflow_flux`) is in $\mathrm{kg\,m^{-2}\,s^{-1}}$.

### 2.3 Five-Port Open Storage Balance and Analytical Water-Table Inversion

With lateral inflow gated by $e_{\mathrm{lat}} \in \{0, 1\}$ (`enable_lateral_flow`, so $F_{\mathrm{lat,eff}} = e_{\mathrm{lat}} F_{\mathrm{lat}}$), net liquid depth increment $\Delta h_{\mathrm{net}}$ ($\mathrm{m}$) and storage increment $\Delta S_{\mathrm{gw}}$ ($\mathrm{mm}$) over step $\Delta t$ are:

$$
\Delta h_{\mathrm{net}} = \frac{\Delta t}{\rho_w}\bigl(F_{\mathrm{drain}} + F_{\mathrm{lat,eff}} - F_{\mathrm{pump}} - F_{\mathrm{cap}} - F_{\mathrm{base}}\bigr), \qquad
\Delta S_{\mathrm{gw}} = 1000\,\Delta h_{\mathrm{net}}
$$

When $\lvert\Delta h_{\mathrm{net}}\rvert < \varepsilon_{\mathrm{small}}$, $z_{\mathrm{wt}}^{n+1} = z_{\mathrm{wt}}^n$; otherwise analytical inversion with $C_p = \max(n_0 p, \varepsilon_{\mathrm{small}})$ gives:

$$
z_{\mathrm{wt}}^{n+1} = \operatorname{clip}\!\left(-p\ln\!\left[\operatorname{clip}\!\left(\exp\!\left(-\frac{z_{\mathrm{wt}}^n}{p}\right) + \frac{\Delta h_{\mathrm{net}}}{C_p},\; \varepsilon_{\mathrm{small}},\; 1\right)\right],\; z_{\min},\; z_{\max}\right)
$$

$$
S^{n+1} = \max\bigl(0,\; S_{\mathrm{prof}}(z_{\mathrm{wt}}^{n+1}) + S_{\mathrm{bias}}\bigr)
$$

### 2.4 Structural Hodge Energy and Dissipation Audit

With specific-yield operator $\star_{S_y} = \operatorname{diag}(\rho_w S_y)$, effective dissipation time $\tau_d = \max(\tau_{\mathrm{gw}}, \tau_{\mathrm{tol}})$, and water-table state $z_h$, the Level-3 field functional and dissipation rate are:

$$
H_h = \frac{1}{2} z_h^{\mathsf{T}} \star_{S_y} z_h, \qquad
e_z = \star_{S_y} z_h, \qquad
\sigma = z_h^{\mathsf{T}}\left(\frac{\star_{S_y}}{\tau_d}\right) z_h \ge 0
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $S$ | Unconfined groundwater storage (input stock and updated replacement) | $\mathrm{mm}$ | `groundwater_storage` | `state` |
| $z_{\mathrm{wt}}$ | Water-table depth below surface (input state and updated replacement) | $\mathrm{m}$ | `water_table_depth` | `state` |
| $F_{\mathrm{drain}}$ | Vadose-zone recharge mass flux (required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `flux` |
| $F_{\mathrm{cap}}$ | Upward capillary export mass flux (optional; 0 when absent) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `capillary_rise_flux` | `flux` |
| $F_{\mathrm{lat}}$ | Lateral groundwater inflow mass flux (optional; gated by `enable_lateral_flow`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `lateral_inflow` | `flux` |
| $F_{\mathrm{pump}}$ | Managed groundwater abstraction mass flux (optional; 0 when absent) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `groundwater_pumping` | `flux` |
| $F_{\mathrm{base}}$ | Groundwater baseflow export mass flux | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `baseflow_flux` | `flux` |
| $\mathcal{D}_{\mathrm{gw}}$ | Nested diagnostics (`storage_deficit`, `saturation_fraction`, `storage_balance_residual`, `storage_balance_roundoff_bound`, `dissipation_rate`, etc.) | mixed SI | `diagnostics.groundwater` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $n_0, p$ | `hydrosphere.groundwater.surface_porosity`, `hydrosphere.groundwater.porosity_decay_length` | $\mathrm{1}$, $\mathrm{m}$ | `hydrosphere/groundwater.yaml` | Beven and Kirkby (1979); Wigmosta et al. (1994) |
| $z_{\max}, z_b, z_{\min}$ | `hydrosphere.groundwater.max_water_table_depth`, `hydrosphere.groundwater.bedrock_depth`, `hydrosphere.groundwater.minimum_water_table_depth` | $\mathrm{m}$ | `hydrosphere/groundwater.yaml` | Wigmosta et al. (1994) |
| $S_y$ | `hydrosphere.groundwater.specific_yield` | $\mathrm{1}$ | `hydrosphere/groundwater.yaml` | Bear (1972) |
| $f_{\min}, \tau_{\mathrm{gw}}$ | `hydrosphere.groundwater.minimum_depth_factor`, `hydrosphere.groundwater.drainage_time_constant` | $\mathrm{1}$, $\mathrm{s}$ | `hydrosphere/groundwater.yaml` | Brutsaert and Nieber (1977); Nathan and McMahon (1990) |
| $e_{\mathrm{lat}}, \mathrm{TI}$ | `hydrosphere.groundwater.enable_lateral_flow`, `hydrosphere.groundwater.topographic_index` | `bool`, $\mathrm{1}$ | `hydrosphere/groundwater.yaml` | Beven and Kirkby (1979) |
| $\tau_{\mathrm{tol}}, \varepsilon_{\mathrm{small}}$ | `hydrosphere.numerical.minimum_timestep_tolerance`, `hydrosphere.numerical.epsilon_small` | $\mathrm{s}$, $\mathrm{1}$ | `hydrosphere/numerical.yaml` | RELIC numerical bounds |
| $\rho_w$ | `relic.physical.constants.water_density` | $\mathrm{kg\,m^{-3}}$ | `physical/constants.yaml` | IAPWS reference density |

## 5. Numerical Methods and Discretization

`GroundwaterDynamics` rejects non-finite, vector-valued, or non-positive $\Delta t$ and rejects legacy `soil_drainage` aliases. Both `groundwater_storage` and `water_table_depth` are emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Direct calls to `get_value_at()` or `interpolate()` raise explicitly so `StateManager` remains the sole prognostic storage owner.

## 6. Conservation, Boundary Conditions, and Ledgers

The five-port open control-volume residual $r_{\mathrm{gw}} = (S^{n+1} - S^n) - \Delta S_{\mathrm{ports}}$ ($\mathrm{mm}$) is evaluated against Higham (2002) floating-point roundoff bound $b_{\mathrm{round}} = \gamma_8\bigl(\lvert S^{n+1}\rvert + \lvert S^n\rvert + \lvert\Delta S_{\mathrm{ports}}\rvert\bigr)$ with $\gamma_n = n\epsilon / (1 - n\epsilon)$:

$$
e_{\mathrm{ledger}} = \max\bigl(\lvert r_{\mathrm{gw}}\rvert - b_{\mathrm{round}},\; 0\bigr)
$$

Depth clipping at $z_{\min}$ or $z_{\max}$ routes any unaccommodated storage change into explicit boundary diagnostics without silent mass loss.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `GroundwaterDynamics` | `relic/scientific/hydrosphere/groundwater/dynamics.py` | `PhysicalField` (`SPATIAL_DIMS=None`) | `PhysicalFieldFunctor` (`relic/core/functor/physical_field_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.CATCHMENT`) |

## 8. Implementation Status

- `GroundwaterDynamics`: `[I]`, IPHS **Level 3** (`field`, certificate `hydrosphere_groundwater_dynamics_groundwater_dynamics.yaml`). Analytical porosity-integral water-table update with $\star_{S_y}$ dissipative audit and Higham roundoff-bounded five-port mass balance.
- Distributed 2D Boussinesq groundwater PDE solver: `[N]`.

Frozen certificate expressions:

```text
S_new = storage_profile(z_new) + bias, z_new from exponential porosity integral and net groundwater input.
S_dot = F_drain + F_lat - F_pump - F_cap - F_base; H_h = 0.5 * z_h.T star_Sy z_h; e_z = star_Sy z_h; G_h = -inverse(star_Sy); y_h = G_h.T e_z = -z_h; sigma = z_h.T (star_Sy / tau_gw) z_h >= 0; dH/dt = -sigma + y_h.T u_h + P_residual.
```

## 9. References

- Bear, J. (1972). *Dynamics of Fluids in Porous Media*. Elsevier.
- Beven, K. J., and Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43-69.
- Brutsaert, W., and Nieber, J. L. (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. *Water Resources Research*, 13(3), 637-643.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Nathan, R. J., and McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26(7), 1465-1473.
- Wigmosta, M. S., Vail, L. W., and Lettenmaier, D. P. (1994). A distributed hydrology-vegetation model for complex terrain. *Water Resources Research*, 30(6), 1665-1679.
