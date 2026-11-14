# Biogenic Volatile Organic Compound Emissions

## 1. Process Definition and Scope

The Biogenic Volatile Organic Compound (BVOC) emission subsystem represents foliar synthesis, optional storage-pool buffering, plant-carbon allocation debits, and molar emissions to the atmosphere across five concrete `BiologicalProcessFunctor` implementations (`MEGANModel`, `IsopreneSynthesis`, `MonoterpeneSynthesis`, `SesquiterpeneSynthesis`, and `SesquiterpeneStressModel`) plus two non-Functor support components (`BVOCResponseIntegrator` and `EmissionFactorDatabase`).

`MEGANModel` and the mechanistic synthesis Functors share mutually exclusive atmosphere-facing molar ports (`isoprene_emission`, `monoterpenes_emission`, `sesquiterpenes_emission`) and the positive plant-carbon allocation debit port `bvoc_carbon_allocation_flux`. Simulator registration enforces single-writer ownership per species port. By default all executable equations treat `air_temperature` (or `temperature` in `SesquiterpeneSynthesis`) and `atmospheric_co2_ppm` as grid-cell proxies (big-leaf $T_{\mathrm{leaf}} = T_{\mathrm{air}}$; no simultaneous intercellular carbon-dioxide solver). `MEGANModel` reads the big-leaf leaf temperature of `LeafEnergyBalance` only with `temperature_source: leaf_temperature` (Section 2.1.3), carries the 24 h and 240 h temperature history only with `temperature_history: true` (Section 2.1.2), accepts threshold heat and cold controls through `stress_factors` (Section 2.1.1), and evaluates its law at daylight Gauss samples with `temporal_support: daylight_gauss` (Section 2.1.4); absent keys keep the previous behaviour bitwise. The 24 h and 240 h light histories, sunlit/shaded multilayer radiative transfer, and canopy-history carbon-dioxide inhibition remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 22.

## 2. Governing Equations

### 2.1 Reduced Instantaneous MEGAN Closure (`MEGANModel`)

For compound class $i \in \{\text{isoprene}, \text{monoterpenes}, \text{sesquiterpenes}, \text{ovocs}\}$, the standard dry-leaf-mass pathway computes ground-area mass emission $E_i$ ($\mathrm{\mu g\,m^{-2}\,s^{-1}}$) as (Guenther et al., 2012, Eqs. 3--4, 7--8, 14):

$$
\rho_f = \mathrm{LAI}\,k_f, \qquad
E_i = \frac{\epsilon_i\,\gamma_{T,i}\,\gamma_{P,i}\,\gamma_{\mathrm{CO_2},i}\,f_{\mathrm{stress}}\,\rho_f}{t_{\mathrm{hour}}}
$$

where $\gamma_{\mathrm{CO_2},i} = \gamma_{\mathrm{CO_2}}$ for isoprene and $1$ for the other three compound classes, and $f_{\mathrm{stress}}$ is the product of explicitly enabled stress controls (or $1$ when stress coupling is disabled). The light and temperature activity factors are:

$$
\gamma_P = \frac{\alpha C_L I}{\sqrt{1 + \alpha^2 I^2}}, \qquad
\gamma_{P,i} = (1 - \mathrm{LDF}_i) + \mathrm{LDF}_i\,\gamma_P
$$

$$
x = \frac{T - T_{\mathrm{opt}}}{T\,T_{\mathrm{opt}}\,R}, \qquad
\gamma_{T,\mathrm{LDF},i} = \frac{E_{\mathrm{opt},i}\,C_{T2}\exp(C_{T1,i}x)}{C_{T2} - C_{T1,i}\bigl[1 - \exp(C_{T2}x)\bigr]}
$$

$$
\gamma_{T,i} = (1 - \mathrm{LDF}_i)\exp\bigl[\beta_i(T - T_s)\bigr] + \mathrm{LDF}_i\,\gamma_{T,\mathrm{LDF},i}
$$

Here isoprene uses $\mathrm{LDF}_{\mathrm{iso}} = 1$, by default (without `temperature_history`) the fixed standard history $T_{24}=T_{240}=297\ \mathrm{K}$ yielding $T_{\mathrm{opt}}=313\ \mathrm{K}$ and $E_{\mathrm{opt},i}=C_{\mathrm{eo},i}$, and dry-mass emission-factor standard temperature $T_s = 303.15\ \mathrm{K}$. The representative OVOC class mixes the activity factors of methanol, acetone, and acetaldehyde using global-emission weights from Table 6 of Guenther et al. (2012). Isoprene carbon-dioxide inhibition uses:

$$
C_i = r_{Ci} C_a, \qquad
\gamma_{\mathrm{CO_2}} = \mathrm{IS}_{\max} - \mathrm{IS}_{\max}\frac{C_i^h}{C_*^h + C_i^h}
$$

When the photosynthesis or hybrid pathway is configured, available net assimilation and its nonnegative projection residual are:

$$
A_{\mathrm{avail}} = \max(A_n, A_{\min}), \qquad
R_{\mathrm{floor}} = A_{\mathrm{avail}} - A_n \ge 0, \qquad
E_{i,\mathrm{photo}} = A_{\mathrm{avail}}\,f_{\mathrm{photo},i}\frac{M_i}{n_{C,i}}\gamma_{T,i}\,\gamma_{\mathrm{CO_2},i}\,f_{\mathrm{stress}}
$$

and the hybrid pathway evaluates $E_{i,\mathrm{hyb}} = w_h E_{i,\mathrm{std}} + (1 - w_h) E_{i,\mathrm{photo}}$. Public molar species fluxes $F_i$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$) and the vegetation-carbon allocation debit $F_{C,\mathrm{alloc}}$ ($\mathrm{kg\,C\,m^{-2}\,s^{-1}}$) are:

$$
F_i = E_i \frac{k_{\mathrm{\mu g \to g}}}{M_i}, \qquad
F_{C,\mathrm{alloc}} = 10^{-3}\sum_{i \in \mathcal{E}} F_i\,M_i\,f_{C,i}
$$

#### 2.1.1 Declared Stress Controls, Including Threshold Heat and Cold (`stress_factors`)

With `use_stress_functors: true` the stress product is taken over the declared set $S$ of controls, in the fixed order of `STRESS_FACTORS`:

$$
f_{\mathrm{stress}} = \prod_{k \in S} f_k, \qquad f_{\mathrm{stress}} = 1 \ \text{for} \ S = \varnothing, \qquad
S \subseteq \{\text{water}, \text{temperature}, \text{ozone}, \text{heat}, \text{cold}\}
$$

`stress_factors` (a non-empty, duplicate-free list; requires `use_stress_functors`) selects $S$; without it the enabled path declares $S = \{\text{water}, \text{temperature}, \text{ozone}\}$ and the disabled path $S = \varnothing$ (both the previous behaviour). Every declared control is a required input with its provider as a required dependency; an undeclared control is no term of $f_{\mathrm{stress}}$ even if present in the state. The labels map to `water_stress_factor` (`water_stress`), `temperature_stress_factor` (`thermal_stress`), `ozone_stress_factor` (`ozone_stress`), `heat_stress_factor` (`thermal_stress`) and `cold_stress_factor` (`thermal_stress_cold`). The heat and cold controls are the threshold responses of `ThermalStress` (`stress.md`, Section 2.2; Salvucci and Crafts-Brandner, 2004, heat; Hatfield and Prueger, 2015, cold); no law is added in `MEGANModel`:

$$
f_{\mathrm{hot}} = \begin{cases} 1, & T \le T_{\mathrm{hot}} \\ \max\bigl(0, 1 + k_{\mathrm{hot}}(T - T_{\mathrm{hot}})\bigr), & T_{\mathrm{hot}} < T \le T_{\mathrm{crit}} \\ 0, & T > T_{\mathrm{crit}} \end{cases}, \qquad
f_{\mathrm{cold}} = \begin{cases} 1, & T \ge T_{\mathrm{cold}} \\ \max\bigl(0, 1 + k_{\mathrm{cold}}(T_{\mathrm{cold}} - T)\bigr), & T_{\mathrm{freeze}} \le T < T_{\mathrm{cold}} \\ 0, & T < T_{\mathrm{freeze}} \end{cases}
$$

with the owner parameters `biosphere.stress.thermal.*` ($T_{\mathrm{hot}} = 308.15\ \mathrm{K}$, $k_{\mathrm{hot}} = -0.1\ \mathrm{K^{-1}}$, $T_{\mathrm{crit}} = 318.15\ \mathrm{K}$, $T_{\mathrm{cold}} = 278.15\ \mathrm{K}$, $k_{\mathrm{cold}} = -0.15\ \mathrm{K^{-1}}$, $T_{\mathrm{freeze}} = 271.15\ \mathrm{K}$) and $T$ the temperature the provider instance reads. When heat reads the daily maximum (or `leaf_temperature_max`) and cold the daily minimum air temperature, they come from two `ThermalStress` instances, the cold provider being the second instance `thermal_stress_cold` (`air_temperature_source: air_temperature_min`, `stress_outputs: [cold_stress_factor]`), since graph node names are unique.

#### 2.1.2 Temperature History (`temperature_history: true`)

The optimum of Eq. (8) follows the carried 24 h and 240 h temperature means (Guenther et al., 2012, p. 1476, Eqs. 9-10, with $T_{24}$ and $T_{240}$ the average leaf temperatures of the past 24 h and 240 h):

$$
T_{\mathrm{opt}} = T_{\mathrm{opt},s} + a\,(T_{240} - T_{s,h}) \quad \text{(Eq. 9)}, \qquad
E_{\mathrm{opt},i} = C_{\mathrm{eo},i}\,h, \quad h = \exp\bigl[c_{24}(T_{24} - T_{s,h})\bigr]\exp\bigl[c_{240}(T_{240} - T_{s,h})\bigr] \quad \text{(Eq. 10)}
$$

with $T_{\mathrm{opt},s} = 313\ \mathrm{K}$ (`biosphere.bvoc.megan.topt`), $T_{s,h} = 297\ \mathrm{K}$, $a = 0.6$ and $c_{24} = c_{240} = 0.05\ \mathrm{K^{-1}}$. At $T_{24} = T_{240} = T_{s,h}$ the result is exactly $(T_{\mathrm{opt},s}, 1)$, the fixed standard history of the default; no clamp is applied. The running means relax exponentially towards the temperature of the step, advanced by the explicit-Euler step of $dX/dt = (T - X)/\tau$:

$$
X^{n+1} = (1 - w)\,X^n + w\,T, \qquad w = \min\!\left(\frac{\Delta t}{\tau},\, 1\right), \qquad (X, \tau) \in \{(T_{24}, \tau_{24}), (T_{240}, \tau_{240})\}
$$

$\tau_{24} = 86400\ \mathrm{s}$ and $\tau_{240} = 864000\ \mathrm{s}$. For $N = \tau/\Delta t \ge 1$ this is the running mean ("runmean" accumulation) $((N - 1)X + T)/N$ of the Community Land Model that carries `t_veg24` and `t_veg240` for MEGAN2.1 (Oleson et al., 2010, CLM4 `accumulMod`); the running-mean form is that CLM accumulation, not a MEGAN2.1 equation. $w \le 1$, so a step at least as long as $\tau$ gives $X^{n+1} = T$ exactly (at $\Delta t = 86400\ \mathrm{s}$, $T_{24}$ becomes the mean of the step just completed), and $X^{n+1}$ lies between $X^n$ and $T$. $T_{24}$ and $T_{240}$ are declared states owned by this Functor (`air_temperature_24h_mean`, `air_temperature_240h_mean`, or `leaf_temperature_24h_mean`, `leaf_temperature_240h_mean` with the leaf source), read at the start of the call (the committed history through the previous step, never the temperature of this step) and replaced after it, a one-step lag by construction; $\Delta t \ge 0$ finite is required. Diagnostics: `diagnostics.bvoc.megan.optimum_temperature` ($T_{\mathrm{opt}}$, K) and `diagnostics.bvoc.megan.optimum_activity_history_factor` ($h$, 1). Without the option no history port exists and $E_{\mathrm{opt}}$, $T_{\mathrm{opt}}$ stay at the standard history.

#### 2.1.3 Temperature Source (`temperature_source`)

`air_temperature` (default): the near-surface air temperature forcing, the big-leaf $T_{\mathrm{leaf}} = T_{\mathrm{air}}$ of the previous interface. `leaf_temperature`: the declared input becomes the leaf temperature of `LeafEnergyBalance` (`biosphere_leaf_energy_balance.md`), the leaf temperature on which MEGAN2.1 defines its response and history (Guenther et al., 2012, Sect. 2.2); it replaces $T$ in $\gamma_T$ (Eqs. 7-8 and 11) and, with `temperature_history`, in $T_{24}$ and $T_{240}$, whose states are then the leaf-temperature means. One source per instance; a graph binds only same-named ports, so the selected name is the declared input.

#### 2.1.4 Daylight Gauss Temporal Support (`temporal_support: daylight_gauss`)

Daily step only ($\Delta t$ = `seconds_per_day`). The whole emission law is evaluated at the daylight Gauss samples $k$ of `relic/scientific/atmosphere/daylight_quadrature.py` (construction and weights $W_k$: `photosynthesis.md`, Section 2.5; Goudriaan, 1986; Goudriaan and van Laar, 1994) and averaged:

$$
E_i = \sum_k W_k\,E_i(T_k, I_k, \dots), \qquad I_n = 0 \ \Rightarrow\ E_{i,n} \propto (1 - \mathrm{LDF}_i)\,\gamma_{T,i}(T_n)
$$

so the night sample keeps the light-independent fraction of each class. $I_k$ is the 24-hour mean `ppfd` redistributed by the solar elevation; $T_k$ is the SWAT+ Eq. 1:1.3.1 course of the daily mean, maximum and minimum air temperature (air source; `air_temperature_max` and `air_temperature_min` are added) or the per-sample leaf temperature `leaf_temperature_daylight_samples` (leaf source). Per-sample ports replace their daily ports where those are declared inputs: `net_photosynthesis_daylight_samples` (hybrid and photosynthesis pathways), `ozone_stress_factor_daylight_samples` and `leaf_temperature_daylight_samples`; `latitude` and `day_of_year` are added. LAI, $C_a$ and the other (daily) stress factors are held constant over the day. With `temperature_history`, $T_{24}$ and $T_{240}$ are read once (start of the day) and advance once per day with the daily-mean temperature $T_d$, the `air_temperature` input or $\sum_k W_k T_{\mathrm{leaf},k}$. Outputs are 24-hour means; no sample port is added. Configuration: `{"temporal_support": "daylight_gauss", "daylight_gauss_nodes": 3 | 5}` and the optional `daylight_shortwave_shape`; daylight keys on the default support raise. In the design probe (one-minute chained reference) the per-sample leaf temperature gave isoprene errors of $-0.3$ to $+0.2\,\%$ with $n = 5$, whereas the air-temperature course shifted by the daily-mean leaf-air difference gave $-24$ to $-32\,\%$; this is why the leaf temperature travels as samples.

### 2.2 Mechanistic MEP Isoprene Synthesis (`IsopreneSynthesis`)

Ground-area gross assimilation $A_{\mathrm{ground}}$ and electron transport $J_{\mathrm{ground}}$ from `FarquharPhotosynthesis` are downscaled to illuminated leaf-area support through Beer-Lambert effective leaf area $L_{\mathrm{eff}}$ (Monsi and Saeki, 1953; de Pury and Farquhar, 1997):

$$
L_{\mathrm{eff}} = \frac{1 - \exp(-k\,\mathrm{LAI})}{k}, \qquad
A_{\mathrm{leaf}} = \begin{cases} A_{\mathrm{ground}} / L_{\mathrm{eff}}, & L_{\mathrm{eff}} > 0 \\ 0, & L_{\mathrm{eff}} = 0 \end{cases}, \qquad
J_{\mathrm{leaf}} = \begin{cases} J_{\mathrm{ground}} / L_{\mathrm{eff}}, & L_{\mathrm{eff}} > 0 \\ 0, & L_{\mathrm{eff}} = 0 \end{cases}
$$

Substrate concentrations and energetic supplies in the methylerythritol 4-phosphate (MEP) pathway (Lichtenthaler, 1999; Rasulov et al., 2009; Morfopoulos et al., 2014) are:

$$
[\mathrm{G3P}] = f_{\mathrm{G3P}}\,A_{\mathrm{leaf}}\,\tau_{\mathrm{G3P}}, \qquad
[\mathrm{Pyr}] = f_{\mathrm{Pyr}}\,A_{\mathrm{leaf}}\,\tau_{\mathrm{Pyr}}
$$

$$
S_{\mathrm{ATP}} = \nu_{\mathrm{ATP},J}\,J_{\mathrm{leaf}}, \qquad
S_{\mathrm{NADPH}} = \nu_{\mathrm{NADPH},J}\,J_{\mathrm{leaf}}
$$

The 1-deoxy-D-xylulose 5-phosphate synthase (DXS) rate, energy limitation $f_E$, dimethylallyl diphosphate (DMAPP) flux, and realized energetic demands are:

$$
J_{\mathrm{DXS}} = \frac{V_{\max,\mathrm{DXS}}(T)\,[\mathrm{G3P}]\,[\mathrm{Pyr}]}{(K_{\mathrm{G3P}} + [\mathrm{G3P}])(K_{\mathrm{Pyr}} + [\mathrm{Pyr}]) + \epsilon}, \qquad
V_{\max,\mathrm{DXS}}(T) = V_{\max,\mathrm{DXS},25}\,Q_{10,\mathrm{DXS}}^{(T - T_{\mathrm{ref}})/10}
$$

$$
f_E = \min\!\left(\frac{S_{\mathrm{ATP}}}{J_{\mathrm{DXS}}\nu_{\mathrm{ATP}} + \epsilon_E},\; \frac{S_{\mathrm{NADPH}}}{J_{\mathrm{DXS}}\nu_{\mathrm{NADPH}} + \epsilon_E},\; c_{\max}\right), \qquad
J_{\mathrm{DMAPP}} = J_{\mathrm{DXS}}\,f_E
$$

$$
A_{\mathrm{ATP}} = J_{\mathrm{DMAPP}}\,\nu_{\mathrm{ATP}}, \qquad
A_{\mathrm{NADPH}} = J_{\mathrm{DMAPP}}\,\nu_{\mathrm{NADPH}}
$$

Steady-state DMAPP concentration proxy $[\mathrm{DMAPP}] = J_{\mathrm{DMAPP}}\,\tau_{\mathrm{DMAPP}}$ drives isoprene synthase (Wilkinson et al., 2009):

$$
V_{\max,\mathrm{IS}}(T) = V_{\max,\mathrm{IS},25}\,Q_{10,\mathrm{IS}}^{(T - T_{\mathrm{ref}})/10}, \qquad
\gamma_{\mathrm{CO_2},\mathrm{IS}} = \begin{cases} \dfrac{K_{\mathrm{CO_2}}}{K_{\mathrm{CO_2}} + C_a}, & \text{inhibition enabled} \\ 1, & \text{otherwise} \end{cases}
$$

$$
F_{\mathrm{leaf},\mu\mathrm{mol}} = \frac{V_{\max,\mathrm{IS}}(T)\,[\mathrm{DMAPP}]}{K_{\mathrm{DMAPP}} + [\mathrm{DMAPP}]}\,\gamma_{\mathrm{CO_2},\mathrm{IS}}, \qquad
F_{\mathrm{ground},\mu\mathrm{mol}} = F_{\mathrm{leaf},\mu\mathrm{mol}}\,L_{\mathrm{eff}}
$$

Public outputs convert $F_{\mathrm{ground},\mu\mathrm{mol}}$ to $F_{\mathrm{iso}} = 10^{-6}F_{\mathrm{ground},\mu\mathrm{mol}}$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$) and $F_{C,\mathrm{alloc}} = 10^{-3}F_{\mathrm{iso}}M_{\mathrm{iso}}f_{C,\mathrm{iso}}$ ($\mathrm{kg\,C\,m^{-2}\,s^{-1}}$).

### 2.3 Stored Monoterpene Pool Dynamics (`MonoterpeneSynthesis`)

For stored monoterpene pool $P$ ($\mathrm{\mu mol\,m^{-2}}$) driven by gross photosynthesis $A_{\mathrm{gross}}$ and air temperature $T$ (Tingey et al., 1980; Niinemets and Reichstein, 2002; Niinemets et al., 2014):

$$
F_{\mathrm{alloc}} = f_{C,\mathrm{photo,mono}}\,f_{\mathrm{seas}}\,A_{\mathrm{gross}}, \qquad
V_{\max,\mathrm{MTS}}(T) = V_{\max,\mathrm{MTS},25}\,Q_{10,\mathrm{MTS}}^{(T - T_{\mathrm{ref}})/10}
$$

$$
J_{\mathrm{syn}} = V_{\max,\mathrm{MTS}}(T)\frac{F_{\mathrm{alloc}}}{K_{\mathrm{GPP}} + F_{\mathrm{alloc}}}\left(1 - \frac{P^n}{P_{\max}}\right), \qquad
J_{\mathrm{emit}} = k_{\mathrm{emit},25}\exp\bigl[\beta_{\mathrm{mono}}(T - T_{\mathrm{ref}})\bigr]P^n, \qquad
J_{\mathrm{deg}} = k_{\mathrm{deg}}P^n
$$

Over step $\Delta t$, the bounded pool update and boundary-clipping residual rate are:

$$
r_{\mathrm{raw}} = J_{\mathrm{syn}} - J_{\mathrm{emit}} - J_{\mathrm{deg}}, \qquad
P^* = P^n + \Delta t\,r_{\mathrm{raw}}, \qquad
P^{n+1} = \min\!\bigl(P_{\max}, \max(0, P^*)\bigr), \qquad
r_{\mathrm{bnd}} = \frac{P^{n+1} - P^*}{\Delta t}
$$

The public atmosphere flux is $F_{\mathrm{mono}} = 10^{-6}J_{\mathrm{emit}}$ ($\mathrm{mol\,m^{-2}\,s^{-1}}$), while the plant-carbon allocation debit tracks synthesis independently of pool release: $F_{C,\mathrm{alloc}} = 10^{-9}J_{\mathrm{syn}}M_{\mathrm{mono}}f_{C,\mathrm{mono}}$ ($\mathrm{kg\,C\,m^{-2}\,s^{-1}}$).

### 2.4 Reduced Sesquiterpene Synthesis and Stress Observation (`SesquiterpeneSynthesis`, `SesquiterpeneStressModel`)

`SesquiterpeneSynthesis` evaluates MEGAN2.1 Eqs. (3), (4), (7), and (8) for the Other Sesquiterpenes parameter row via `BVOCResponseIntegrator` (Duhl et al., 2008; Guenther et al., 2012):

$$
\gamma_{T,\mathrm{SQT}} = (1 - \mathrm{LDF}_{\mathrm{SQT}})\exp\bigl[\beta_{\mathrm{SQT}}(T - T_s)\bigr] + \mathrm{LDF}_{\mathrm{SQT}}\,\gamma_{T,\mathrm{LDF,SQT}}, \qquad
\gamma_{L,\mathrm{SQT}} = (1 - \mathrm{LDF}_{\mathrm{SQT}}) + \mathrm{LDF}_{\mathrm{SQT}}\,\gamma_P
$$

with $\gamma_{L,\mathrm{SQT}} = 1$ when `ppfd` is absent and $\gamma_{\mathrm{stress}}$ equal to the minimum of supplied optional stress factors (or $1$ when none is supplied):

$$
E_{\mathrm{area},\mathrm{SQT}} = \frac{\mathrm{EF}_{\mathrm{SQT}}\,\gamma_{T,\mathrm{SQT}}\,\gamma_{L,\mathrm{SQT}}\,\gamma_{\mathrm{stress}}\,\mathrm{LAI}\,k_f}{t_{\mathrm{hour}}}, \qquad
F_{\mathrm{SQT}} = 10^{-6}\frac{E_{\mathrm{area},\mathrm{SQT}}}{M_{\mathrm{SQT}}}, \qquad
F_{C,\mathrm{alloc}} = 10^{-3}F_{\mathrm{SQT}}M_{\mathrm{SQT}}f_{C,\mathrm{SQT}}
$$

`SesquiterpeneStressModel` is an independent algebraic observation mapping aggregate dimensionless `stress_level` $S$ to `stress_response_factor`:

$$
\gamma_{\mathrm{SQT,obs}} = \operatorname{clip}\!\bigl(1 + A\max(0, S - S_{\mathrm{thr}}),\; 1,\; 1 + A\bigr)
$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T$ | Grid-cell air temperature (`temperature` in `SesquiterpeneSynthesis`), or the leaf temperature with `MEGANModel` `temperature_source: leaf_temperature` | $\mathrm{K}$ | `air_temperature` / `temperature` / `leaf_temperature` | `forcing` / `diagnostic` |
| $T_{24}, T_{240}$ | Carried 24 h and 240 h temperature means (`MEGANModel` `temperature_history`) | $\mathrm{K}$ | `air_temperature_24h_mean`, `air_temperature_240h_mean` (or `leaf_temperature_24h_mean`, `leaf_temperature_240h_mean`) | `state` |
| $T_{\mathrm{opt}}, h$ | History optimum temperature and $E_{\mathrm{opt}}$ factor (`temperature_history`) | $\mathrm{K}$, $1$ | `diagnostics.bvoc.megan.optimum_temperature`, `diagnostics.bvoc.megan.optimum_activity_history_factor` | `diagnostic` |
| $f_k$ | Declared stress controls of `MEGANModel` (`stress_factors`) | $1$ | `water_stress_factor`, `temperature_stress_factor`, `ozone_stress_factor`, `heat_stress_factor`, `cold_stress_factor` | `diagnostic` |
| $\varphi$, $J$, $T_{\max}$, $T_{\min}$ | Daylight geometry and diurnal-course inputs (`MEGANModel` `daylight_gauss`; extremes with the air source) | $\mathrm{degrees\_north}$, $1$, $\mathrm{K}$ | `latitude`, `day_of_year`, `air_temperature_max`, `air_temperature_min` | `forcing` |
| $A_{n,k}, f_{\mathrm{O_3},k}, T_{\mathrm{leaf},k}$ | Per-sample inputs of `MEGANModel` `daylight_gauss` | $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$, $1$, $\mathrm{K}$ | `net_photosynthesis_daylight_samples`, `ozone_stress_factor_daylight_samples`, `leaf_temperature_daylight_samples` | `flux` / `diagnostic` |
| $I$ | Photosynthetic photon flux density | $\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$ | `ppfd` | `forcing` |
| $\mathrm{LAI}$ | One-sided leaf area index | $\mathrm{m^2\,m^{-2}}$ | `leaf_area_index` | `state` |
| $C_a$ | Ambient atmospheric $\mathrm{CO_2}$ mole fraction proxy | $\mathrm{\mu mol\,mol^{-1}}$ | `atmospheric_co2_ppm` | `forcing` |
| $A_{\mathrm{ground}}, A_n$ | Canopy gross and net molar $\mathrm{CO_2}$ assimilation | $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$ | `gross_photosynthesis`, `net_photosynthesis` | `flux` |
| $J_{\mathrm{ground}}$ | Canopy photosynthetic electron transport rate | $\mathrm{\mu mol\,e^-\,m^{-2}\,s^{-1}}$ | `electron_transport_rate` | `flux` |
| $P$ | Stored foliar monoterpene pool | $\mathrm{\mu mol\,m^{-2}}$ | `monoterpene_pool_concentration` | `state` |
| $S$ | Aggregate abiotic stress driver | $1$ | `stress_level` | `forcing` |
| $F_i$ | Species molar emission to the atmosphere | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `isoprene_emission`, `monoterpenes_emission`, `sesquiterpenes_emission`, `ovocs_emission`, `total_bvoc_emission` | `flux` |
| $F_{C,\mathrm{alloc}}$ | Plant-carbon allocation debit to BVOC synthesis | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `bvoc_carbon_allocation_flux` | `flux` |
| $\gamma_{\mathrm{SQT,obs}}$ | Induced sesquiterpene stress amplification factor | $1$ | `stress_response_factor` | `diagnostic` |
| $R_{\mathrm{floor}}, r_{\mathrm{bnd}}$ | MEGAN carbon-floor residual and monoterpene pool-clipping rate | $\mathrm{\mu mol\,C\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$ | `diagnostics.bvoc.megan.photosynthesis_carbon_floor_residual`, `diagnostics.bvoc.monoterpene.pool_boundary_residual_rate` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\epsilon_i$ | `biosphere.bvoc.emission_factors.*` or `biosphere.pft.<pft>.bvoc.ef_*` | $\mathrm{\mu g\,g^{-1}\,DW\,h^{-1}}$ | `bvoc_parameters.yaml` / `pft/*.yaml` | Guenther et al. (2012) |
| $k_f, T_s$ | `biosphere.bvoc.foliar_density_factor`, `biosphere.bvoc.t_standard` | $\mathrm{g\,DW\,m^{-2}}$, $\mathrm{K}$ | `bvoc_parameters.yaml` | Guenther et al. (2012) |
| $\alpha, C_L, T_{\mathrm{opt}}, C_{T2}$ | `biosphere.bvoc.megan.{alpha_light,cl_light,topt,ct2}` | $\mathrm{m^2\,s\,\mu mol^{-1}}$, $1$, $\mathrm{K}$, $\mathrm{J\,mol^{-1}}$ | `bvoc_parameters.yaml` | Guenther et al. (2012) |
| $\mathrm{LDF}_i, \beta_i, C_{T1,i}, E_{\mathrm{opt},i}$ | `biosphere.bvoc.megan.{ldf_*,beta_*,ct1_*,eopt_*}` and `megan.ovoc.*` | $1$, $\mathrm{K^{-1}}$, $\mathrm{J\,mol^{-1}}$, $1$ | `bvoc_parameters.yaml` | Guenther et al. (2012) Tables 4, 6 |
| $T_{s,h}, a, c_{24}, c_{240}, \tau_{24}, \tau_{240}$ | `biosphere.bvoc.megan.{history_standard_temperature,topt_history_slope,eopt_history_coefficient_24h,eopt_history_coefficient_240h,history_time_scale_24h,history_time_scale_240h}` (`temperature_history` only) | $\mathrm{K}$, $1$, $\mathrm{K^{-1}}$, $\mathrm{K^{-1}}$, $\mathrm{s}$, $\mathrm{s}$ | `bvoc_parameters.yaml` ($297.0, 0.6, 0.05, 0.05, 86400, 864000$) | Guenther et al. (2012) Eqs. 9-10; Oleson et al. (2010) |
| $T_{\mathrm{hot}}, k_{\mathrm{hot}}, T_{\mathrm{crit}}, T_{\mathrm{cold}}, k_{\mathrm{cold}}, T_{\mathrm{freeze}}$ | `biosphere.stress.thermal.*`, read by the `ThermalStress` providers of `heat` and `cold`, not by `MEGANModel` | $\mathrm{K}$, $\mathrm{K^{-1}}$ | `biosphere/stress.yaml` | Salvucci and Crafts-Brandner (2004); Hatfield and Prueger (2015) |
| $\mathrm{IS}_{\max}, C_*, h, r_{Ci}$ | `biosphere.bvoc.co2_inhibition.{ismax,c_star,hill_exponent,intercellular_to_ambient_ratio}` | $1$, $\mathrm{ppm}$, $1$, $1$ | `bvoc_parameters.yaml` (`float32`) | Guenther et al. (2012); Heald et al. (2009) |
| $V_{\max,\mathrm{DXS}}, K_{\mathrm{G3P}}, K_{\mathrm{Pyr}}, Q_{10,\mathrm{DXS}}$ | `biosphere.bvoc.mep.{v_max_dxs,k_g3p,k_pyruvate,q10_dxs}` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,m^{-3}}$, $\mathrm{\mu mol\,m^{-3}}$, $1$ | `bvoc_parameters.yaml` | Lichtenthaler (1999); Vickers et al. (2009) |
| $\nu_{\mathrm{ATP}}, \nu_{\mathrm{NADPH}}$ | `biosphere.bvoc.mep.{atp_per_dmapp,nadph_per_dmapp}` | $1$ | `bvoc_parameters.yaml` | Vickers et al. (2009); Rasulov et al. (2009) |
| $V_{\max,\mathrm{IS}}, K_{\mathrm{DMAPP}}, K_{\mathrm{CO_2}}, Q_{10,\mathrm{IS}}$ | `biosphere.bvoc.isoprene_synthase.{v_max,k_dmapp,k_co2_inhibition,q10}` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,m^{-3}}$, $\mathrm{\mu mol\,mol^{-1}}$, $1$ | `bvoc_parameters.yaml` | Rasulov et al. (2009); Wilkinson et al. (2009) |
| $\tau_{\mathrm{G3P}}, \tau_{\mathrm{Pyr}}, \tau_{\mathrm{DMAPP}}$ | `biosphere.bvoc.isoprene_synthase.{g3p,pyruvate,dmapp}_flux_to_concentration` | $\mathrm{s\,m^{-1}}$ | `bvoc_parameters.yaml` | Rasulov et al. (2009); Morfopoulos et al. (2014) |
| $V_{\max,\mathrm{MTS}}, K_{\mathrm{GPP}}, P_{\max}, k_{\mathrm{emit},25}, \beta_{\mathrm{mono}}, k_{\mathrm{deg}}$ | `biosphere.bvoc.monoterpene.{v_max_mts,k_gpp,pool.capacity,pool.k_emit_25,pool.beta_emission,pool.k_degradation}` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,m^{-2}}$, $\mathrm{s^{-1}}$, $\mathrm{K^{-1}}$, $\mathrm{s^{-1}}$ | `bvoc_parameters.yaml` | Tingey et al. (1980); Niinemets and Reichstein (2002) |
| $\mathrm{EF}_{\mathrm{SQT}}, S_{\mathrm{thr}}, A$ | `biosphere.bvoc.default.sesquiterpenes`, `biosphere.bvoc.sesquiterpene.{stress_threshold,stress_amplification}` | $\mathrm{\mu g\,g^{-1}\,DW\,h^{-1}}$, $1$, $1$ | `bvoc_parameters.yaml` | Duhl et al. (2008); Guenther et al. (2012) |
| $M_{\mathrm{iso}}, R, T_{\mathrm{ref}}, t_{\mathrm{hour}}, k_{\mathrm{\mu g \to g}}$ | `relic.physical.constants.{molecular_weights.isoprene,universal.gas_constant,temperature.reference_temperature,time.seconds_per_hour,conversions.micrograms_to_grams}` | $\mathrm{g\,mol^{-1}}$, $\mathrm{J\,mol^{-1}\,K^{-1}}$, $\mathrm{K}$, $\mathrm{s}$, $\mathrm{g\,\mu g^{-1}}$ | `physical/constants.yaml` | Physical-constant SSOT |

## 5. Numerical Methods and Discretization

- **Reciprocal-cancellation avoidance in MEGAN Eq. (8)**: `MEGANModel` evaluates $x = (T - T_{\mathrm{opt}})/(T\,T_{\mathrm{opt}}\,R)$ directly, avoiding subtraction of two separately rounded reciprocals $1/T_{\mathrm{opt}} - 1/T$ (Goldberg, 1991).
- **Declared `float32` precision and continuous 400 ppm evaluation**: The four `biosphere.bvoc.co2_inhibition.*` entries are declared `float32` (`1.343999981880188`, `1.461400032043457`, `0.699999988079071`, `585.0`) so checkpoint restoration preserves binding dtype. Unlike the official Fortran `GAMMA_CO2` `WHERE (CO2 .eq. 400.0)` branch that jumps to $1.0$, RELIC evaluates Eq. (14) continuously across all $C_a$, giving $1.0024713532535012$ (`float64`) and $1.0024713277816772$ (`float32`) at $400\ \mathrm{ppm}$.
- **Explicit Euler pool step with boundary residual**: `MonoterpeneSynthesis` advances `monoterpene_pool_concentration` with explicit Euler over $\Delta t$, clamps to $[0, P_{\max}]$, and records the exact signed boundary rate $r_{\mathrm{bnd}} = (P^{n+1} - P^*)/\Delta t$ so $(P^{n+1} - P^n)/\Delta t = r_{\mathrm{raw}} + r_{\mathrm{bnd}}$.
- **Held learnable coefficients**: `MEGANModel` holds a learnable ParameterManager binding of a scalar coefficient by reference, so gradients of every consumer reach the one manager-owned value and the binding stays visible to checkpoints; every other value keeps a `float32` buffer copy.
- **Input domain**: `biosphere.bvoc.validation.temperature.min_operational` is $203.15\ \mathrm{K}$ (lowered from $233.15\ \mathrm{K}$ on 2026-11-05 for global daily runs; $\gamma_T \approx 10^{-9}$ there with the default coefficients).
- **Reaction-network lowering and update modes**: All four reaction Functors lower their open ports through `advance_reaction_network(..., open_system_route="port")` and emit `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` outputs (including the full-state replacement of `monoterpene_pool_concentration`).

## 6. Conservation, Boundary Conditions, and Ledgers

- **Open species-to-carbon boundary consistency (`carbon_boundary`)**: For `MEGANModel`, `IsopreneSynthesis`, and `SesquiterpeneSynthesis`, `check_conservation_laws(...)` verifies $\mathcal{R}_C = \lVert F_{C,\mathrm{alloc}} - 10^{-3}\sum_i F_i M_i f_{C,i}\rVert_\infty = 0$ within machine precision.
- **Stored-monoterpene open carbon ledger**: `MonoterpeneSynthesis.check_conservation_laws(...)` verifies $\mathcal{R}_{C,\mathrm{mono}} = \lVert F_{C,\mathrm{alloc}} - F_{C,\mathrm{emit}} - F_{C,\mathrm{deg}} + F_{C,\mathrm{bnd}} - F_{C,\mathrm{storage}}\rVert_\infty$, ensuring pool release is not debited a second time from `BiomassAllocation`.
- **Compile-time `ppfd_source` and cross-sphere handoff**: `MEGANModel` accepts `ppfd_source="atmospheric_radiation"` (default dependency) or `"state"` (externally supplied `ppfd`). Downstream `BiomassAllocation` debits `bvoc_carbon_allocation_flux` from NPP, while `AtmosphereCoupling` injects the molar species ports vertically into `bvoc_emission_field_3d`.

## 7. Code Mapping

| Component | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `MEGANModel` | `relic/scientific/biosphere/bvoc/megan_model.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `IsopreneSynthesis` | `relic/scientific/biosphere/bvoc/synthesis/isoprene.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `MonoterpeneSynthesis` | `relic/scientific/biosphere/bvoc/synthesis/monoterpene.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `SesquiterpeneSynthesis` | `relic/scientific/biosphere/bvoc/synthesis/sesquiterpene.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `SesquiterpeneStressModel` | `relic/scientific/biosphere/bvoc/synthesis/sesquiterpene.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`GRID_CELL`) |
| `BVOCResponseIntegrator`, `EmissionFactorDatabase` | `relic/scientific/biosphere/bvoc/{responses,emission_factors}.py` | Non-Functor support helpers | None | Support utilities |

## 8. Implementation Status

- `MEGANModel`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_bvoc_megan_model_megan_model.yaml`.
- `IsopreneSynthesis`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_bvoc_synthesis_isoprene_isoprene_synthesis.yaml`.
- `MonoterpeneSynthesis`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_bvoc_synthesis_monoterpene_monoterpene_synthesis.yaml`.
- `SesquiterpeneSynthesis`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_synthesis.yaml`.
- `SesquiterpeneStressModel`: `[I]` Level 1 (`observation`, `diagnostic`) in `biosphere_bvoc_synthesis_sesquiterpene_sesquiterpene_stress_model.yaml`.
- `MEGANModel` options `stress_factors` (heat, cold), `temperature_history`, `temperature_source` and `temporal_support: daylight_gauss`: `[I]`; the frozen expressions below are unchanged (the options change $f_{\mathrm{stress}}$, $T$, $E_{\mathrm{opt}}$ and $T_{\mathrm{opt}}$, or evaluate the same law at samples).
- Full MEGAN2.1 24/240-hour light histories, multilayer sunlit/shaded canopy integration, and prognostic intracellular DMAPP pool dynamics: `[N]` (the 24/240-hour temperature history is available through `temperature_history`).

Frozen certificate equivalence expressions:

```text
E_i = epsilon_i gamma_T,i gamma_P,i rho_f / seconds_per_hour
x_dot = N_h r_h; A_available = max(A_n, A_min); R_floor = A_available - A_n; F_i = E_i k_ug_to_g / MW_i; F_C = convert(sum_i(F_i MW_i f_C,i), g C m-2 s-1, kg C m-2 s-1)
J_DXS = V_max_DXS [G3P][Pyr] / ((K_G3P + [G3P])(K_Pyr + [Pyr])); J_IS = V_max_IS(T_air) [DMAPP] / (K_DMAPP + [DMAPP]) K_CO2/(K_CO2 + C_a); L_eff = (1 - exp(-k LAI)) / k; F_ground = F_leaf L_eff
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
The bounded stored-pool tendency is evaluated once from the phenomenological synthesis, emission, degradation, and boundary-residual rates.
The reduced empirical closure evaluates one nonnegative leaf-mass emission factor, maps canonical RELIC leaf area to effective dry-leaf mass with LAI times k_f, converts the result to the canonical molar atmosphere port, and exposes the corresponding emitted-compound carbon as a positive allocation proxy. It owns no storage pool and claims neither canonical MEGAN2.1 equivalence nor synthesis-release equivalence for stored emitters.
stress_response_factor = min(max(1 + max(S - S_th, 0) * A, 1), 1 + A); S = stress_level [1], one aggregated stress driver; S_th = biosphere.bvoc.sesquiterpene.stress_threshold [1]; A = biosphere.bvoc.sesquiterpene.stress_amplification [1]; algebraic in S, independent of dt, no recovery state
y_h = O_h x_h through the declared observation operator
```

## 9. References

1. Guenther, A. B., Jiang, X., Heald, C. L., et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling biogenic emissions. *Geoscientific Model Development*, 5, 1471-1492.
2. Heald, C. L., Wilkinson, M. J., Monson, R. K., et al. (2009). Response of isoprene emission to ambient $\mathrm{CO_2}$ changes and implications for global budgets. *Global Change Biology*, 15, 1127-1140.
3. Lichtenthaler, H. K. (1999). The 1-deoxy-D-xylulose-5-phosphate pathway of isoprenoid biosynthesis in plants. *Annual Review of Plant Physiology and Plant Molecular Biology*, 50, 47-65.
4. Rasulov, B., Copolovici, L., Laisk, A., and Niinemets, U. (2009). Postillumination isoprene emission: in vivo estimation of dimethylallyl diphosphate pool size and isoprene synthase kinetics in aspen leaves. *Plant Physiology*, 149, 1609-1618.
5. Morfopoulos, C., Sperlich, D., Peñuelas, J., et al. (2014). A model of plant isoprene emission based on available reducing power captures responses to atmospheric $\mathrm{CO_2}$. *New Phytologist*, 203, 125-139.
6. Wilkinson, M. J., Monson, R. K., Trahan, N., et al. (2009). Leaf isoprene emission rate as a function of atmospheric $\mathrm{CO_2}$ concentration. *Global Change Biology*, 15, 1189-1200.
7. Niinemets, U., and Reichstein, M. (2002). A model analysis of the effects of nonspecific monoterpenoid storage in leaf tissues on emission kinetics. *Global Biogeochemical Cycles*, 16, 1110.
8. Niinemets, U., Fares, S., Harley, P., and Jardine, K. J. (2014). Bidirectional exchange of biogenic volatiles with vegetation: emission sources, reactions, breakdown and deposition. *Plant, Cell & Environment*, 37, 1790-1809.
9. Duhl, T. R., Helmig, D., and Guenther, A. (2008). Sesquiterpene emissions from vegetation: a review. *Biogeosciences*, 5, 761-777.
10. Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys*, 23, 5-48.
11. Oleson, K. W., et al. (2010). *Technical Description of version 4.0 of the Community Land Model (CLM)* (NCAR/TN-478+STR). NCAR (CLM4 `accumulMod` running mean).
12. Salvucci, M. E., and Crafts-Brandner, S. J. (2004). Inhibition of photosynthesis by heat stress. *Plant, Cell & Environment*, 27, 1063-1071.
13. Hatfield, J. L., and Prueger, J. H. (2015). Temperature extremes: Effect on plant growth and development. *Weather and Climate Extremes*, 10, 4-10.
14. Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily totals of crop photosynthesis. *Agricultural and Forest Meteorology*, 38, 249-254 (cited for the scheme; text not read).
15. Goudriaan, J., and van Laar, H. H. (1994). *Modelling Potential Crop Growth Processes*. Kluwer Academic Publishers (cited for the scheme; text not read).
