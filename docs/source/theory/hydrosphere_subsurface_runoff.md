# Hydrosphere Subsurface Runoff and Threshold Interflow

## 1. Process Definition and Scope

`SubsurfaceRunoff` (`relic/scientific/hydrosphere/runoff/subsurface.py`) advances patch-scale subsurface storage (`groundwater_storage`, $\mathrm{mm}$) under vadose-zone `drainage_flux` recharge and exports lateral `subsurface_runoff_flux` ($\mathrm{kg\,m^{-2}\,s^{-1}}$) combining configurable baseflow (`baseflow_method` in `{"exponential", "linear", "topmodel"}`), soil-moisture-gated threshold interflow, and storage-capacity overflow.

`SubsurfaceRunoff` declares optional dependencies on alternative drainage providers (`bucket_soil_moisture`, `richards_soil_moisture`, `vertical_percolation_transport`) and rejects removed legacy `drainage_input` and `subsurface_drainage` aliases. Multi-cell hillslope kinematic-wave routing across patches remains `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 19 (Section 19.9.2).

## 2. Governing Equations

### 2.1 Post-Recharge Candidate Storage and Capacity Overflow

From start-of-step storage $S_0 \ge 0$ ($\mathrm{mm}$), recharge depth rate $R = 1000\,F_{\mathrm{drain}} / \rho_w$ ($\mathrm{mm\,s^{-1}}$), step $\Delta t > 0$, and storage capacity $S_{\max}$ ($\mathrm{mm}$), post-recharge capped storage $S^*$ and routed capacity overflow $O_{\mathrm{cap}}$ ($\mathrm{mm}$) are evaluated first:

$$
S^* = \operatorname{clip}(S_0 + R\Delta t,\; 0,\; S_{\max}), \qquad
O_{\mathrm{cap}} = \max(S_0 + R\Delta t - S_{\max},\; 0)
$$

Both candidate baseflow and candidate interflow are evaluated at $S^*$.

### 2.2 Configurable Baseflow and Threshold Interflow at $S^*$

1. **Baseflow candidate $Q_b(S^*)$** ($\mathrm{mm\,s^{-1}}$):
   - `"exponential"` or `"linear"` (with recession constant $k$, $\mathrm{s^{-1}}$):

$$
Q_b(S^*) = k\,S^*
$$

   - `"topmodel"` (point-scale application of catchment-mean transmissivity $T_0$, mean topographic index $\lambda$, deficit $S_{\mathrm{def}} = \max(S_{\mathrm{def,max}} - S^*, 0)$, scale $m = c_m S_{\mathrm{def,max}}$, and empirical moisture factor $f_\theta = \operatorname{clip}(\theta / \theta_{\mathrm{ref}}, f_{\min}, f_{\max})$; Beven and Kirkby, 1979):

$$
Q_b(S^*) = 1000\,T_0 e^{-\lambda}\exp\!\left(-\frac{S_{\mathrm{def}}}{m}\right)f_\theta
$$

2. **Threshold interflow candidate $Q_i(\theta, S^*)$** ($\mathrm{mm\,s^{-1}}$), with derived coefficient $k_i = (f_r c_s) / s_{\mathrm{day}}$ (`return_flow_fraction * interflow_scaling_factor / seconds_per_day`), rate scale $c_i$, threshold moisture $\theta_t$, and exponent $n$:

$$
Q_i(\theta, S^*) = k_i c_i\,\bigl[\max(0, \theta - \theta_t)\bigr]^n\,S^*
$$

### 2.3 Finite-Volume Outflow Limiter, Mass Export, and Diagnostics

Total candidate withdrawal $Q_{\mathrm{cand}}\Delta t = \bigl(Q_b(S^*) + Q_i(\theta, S^*)\bigr)\Delta t$ is bounded by available storage $S^*$:

$$
Q_{\mathrm{act}}\Delta t = \min(Q_{\mathrm{cand}}\Delta t,\; S^*), \qquad
Q_{\mathrm{tot}} = Q_{\mathrm{act}} + \frac{O_{\mathrm{cap}}}{\Delta t}, \qquad
S_1 = S_0 + R\Delta t - Q_{\mathrm{tot}}\Delta t
$$

$$
F_{\mathrm{sub}} = \frac{\rho_w}{1000}\,Q_{\mathrm{tot}}, \qquad
\phi_{\mathrm{lim}} = \frac{Q_{\mathrm{act}}\Delta t}{\max(Q_{\mathrm{cand}}\Delta t,\; q_{\mathrm{out,min}})}
$$

Diagnostic baseflow and interflow mass fluxes scale by $\phi_{\mathrm{lim}}$, while diagnostic water-table depth and recession time constant ($\mathrm{day}$) evaluate as:

$$
z_{\mathrm{wt,diag}} = \operatorname{clip}\!\left(z_{\mathrm{scale}}\!\left(1 - \frac{S_1}{S_{\max}}\right),\; 0,\; z_{\max}\right), \qquad
\tau_{\mathrm{days}} = \frac{1}{\max(k, k_{\min})\,s_{\mathrm{day}}}
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $\theta$ | Volumetric soil moisture ($\in [0, 1]$, required) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $F_{\mathrm{drain}}$ | Vadose-zone drainage mass flux (required) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `drainage_flux` | `flux` |
| $S_0, S_1$ | Subsurface groundwater storage (start-of-step input and updated replacement) | $\mathrm{mm}$ | `groundwater_storage` | `state` |
| $F_{\mathrm{sub}}$ | Total lateral subsurface runoff mass flux (including capacity overflow) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `subsurface_runoff_flux` | `flux` |
| $F_b, F_i$ | Limiter-scaled baseflow and interflow diagnostic mass fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.subsurface_runoff.baseflow_flux`, `diagnostics.subsurface_runoff.interflow_flux` | `diagnostic` |
| $f_S, z_{\mathrm{wt,diag}}, \tau_{\mathrm{days}}$ | Storage fraction, diagnostic water-table depth, and recession time constant | $\mathrm{1}$, $\mathrm{m}$, $\mathrm{day}$ | `diagnostics.subsurface_runoff.storage_fraction`, `water_table_depth`, `recession_time_constant_days` | `diagnostic` |
| $R, Q_{\mathrm{tot}}$ | Recharge and total runoff depth rates | $\mathrm{mm\,s^{-1}}$ | `diagnostics.subsurface_runoff.drainage_rate_mm_s`, `total_runoff_rate_mm_s` | `diagnostic` |
| $O_{\mathrm{cap}}, r_{\mathrm{out}}, r_z, \epsilon_S$ | Step-integrated overflow, outflow limiter, depth clipping, and water-balance residuals | $\mathrm{mm}$, $\mathrm{m}$ | `diagnostics.subsurface_runoff.storage_capacity_overflow`, `outflow_limiter_residual`, `water_table_depth_clipping_residual`, `water_balance_residual` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $m_b, k$ | `hydrosphere.runoff.baseflow_method`, `baseflow_recession_constant` | `str`, $\mathrm{s^{-1}}$ | `hydrosphere/runoff.yaml` | Nathan and McMahon (1990); Tallaksen (1995) |
| $\lambda, T_0, S_{\mathrm{def,max}}, c_m$ | `hydrosphere.runoff.topographic_index_mean`, `transmissivity`, `topmodel_max_deficit`, `topmodel_m_scaling_factor` | $\mathrm{1}$, $\mathrm{m^2\,s^{-1}}$, $\mathrm{mm}$, $\mathrm{1}$ | `hydrosphere/runoff.yaml` | Beven and Kirkby (1979) |
| $\theta_{\mathrm{ref}}, f_{\min}, f_{\max}$ | `hydrosphere.runoff.topmodel_moisture_reference`, `topmodel_moisture_factor_min`, `topmodel_moisture_factor_max` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$ | `hydrosphere/runoff.yaml` | Empirical TOPMODEL moisture modifier |
| $\theta_t, f_r, c_s, c_i, n$ | `hydrosphere.runoff.threshold_moisture`, `return_flow_fraction`, `interflow_scaling_factor`, `interflow_rate_scaling_factor`, `interflow_exponent` | $\mathrm{m^3\,m^{-3}}$, $\mathrm{1}$, $\mathrm{day^{-1}}$, $\mathrm{1}$, $\mathrm{1}$ | `hydrosphere/runoff.yaml` | Brutsaert and Nieber (1977) |
| $S_{\max}, z_{\mathrm{scale}}, z_{\max}$ | `hydrosphere.runoff.storage_capacity`, `water_table_depth_scale`, `water_table_depth_max` | $\mathrm{mm}$, $\mathrm{m}$, $\mathrm{m}$ | `hydrosphere/runoff.yaml` | Reservoir geometry |
| $q_{\mathrm{out,min}}, k_{\min}$ | `hydrosphere.numerical.minimum_storage_outflow`, `recession_coefficient_minimum` | $\mathrm{mm}$, $\mathrm{s^{-1}}$ | `hydrosphere/numerical.yaml` | RELIC numerical guards |
| $\rho_w, s_{\mathrm{day}}$ | `relic.physical.constants.water_density`, `relic.physical.constants.seconds_per_day` | $\mathrm{kg\,m^{-3}}$, $\mathrm{s\,day^{-1}}$ | `physical/constants.yaml` | IAPWS / SI constants |

## 5. Numerical Methods and Discretization

`_ensure_tensor` normalizes inputs to `torch.float32` on the Functor device. `subsurface_runoff_flux` is emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`, whereas `groundwater_storage` is emitted as `UpdateItem(UpdateMode.REPLACE, per_timestep=True)` (an already step-integrated stock value). All 11 flat `diagnostics.subsurface_runoff.*` ports use `UpdateMode.REPLACE`.

## 6. Conservation, Boundary Conditions, and Ledgers

Capacity overflow $O_{\mathrm{cap}} / \Delta t$ is included directly inside exported `subsurface_runoff_flux`, closing the open inlet-storage-outlet step budget $\epsilon_S = S_1 - S_0 - R\Delta t + Q_{\mathrm{tot}}\Delta t = 0$. To prevent `float32` representational scale false positives at large $S_1$, `SubsurfaceRunoff.check_conservation_laws()` evaluates the normalized relative residual:

$$
r_S =
\begin{cases}
\dfrac{\max\lvert\epsilon_S\rvert}{\max\lvert S_1\rvert}, & \max\lvert S_1\rvert > 0 \\
\max\lvert\epsilon_S\rvert, & \max\lvert S_1\rvert = 0
\end{cases}
$$

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SubsurfaceRunoff` | `relic/scientific/hydrosphere/runoff/subsurface.py` | `TransportMechanism` (`TRANSPORT_TYPE="advection"`) | `TransportMechanismFunctor` (`relic/core/functor/transport_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.PATCH`) |

## 8. Implementation Status

- `SubsurfaceRunoff`: `[I]`, IPHS **Level 3** (`transport`, certificate `hydrosphere_runoff_subsurface_subsurface_runoff.yaml`). Finite-volume bounded subsurface reservoir transport with explicit capacity-overflow routing and relative mass-balance verification.

Frozen certificate expressions:

```text
Q_candidate = Q_base(S, theta; T0 exp(-lambda)) + Q_interflow(theta, S), retained only as certificate evidence for the single production closure.
S1 = S0 + R*dt - Q_total*dt; Q_total = Q_actual + O_capacity/dt; Q_actual*dt = min(Q_candidate*dt, S_available).
```

## 9. References

- Beven, K. J., and Kirkby, M. J. (1979). A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43-69.
- Brutsaert, W., and Nieber, J. L. (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. *Water Resources Research*, 13(3), 637-643.
- Nathan, R. J., and McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26(7), 1465-1473.
- Tallaksen, L. M. (1995). A review of baseflow recession analysis. *Journal of Hydrology*, 165(1-4), 349-370.
