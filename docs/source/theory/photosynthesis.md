# Photosynthesis and Conditional Stomatal Conductance

## 1. Process Definition and Scope

The photosynthesis module evaluates leaf-to-canopy biochemical carbon assimilation (`FarquharPhotosynthesis`) for $\mathrm{C_3}$ (Farquhar, von Caemmerer, and Berry, 1980) and $\mathrm{C_4}$ (Collatz et al., 1992) pathways and provides a conditional leaf-support stomatal conductance diagnostic (`MedlynStomatalConductance`; Medlyn et al., 2011) alongside a tutorial linear proxy (`StomatalProxyFunctor`).

`FarquharPhotosynthesis` treats `atmospheric_co2_ppm` directly as effective intercellular $\mathrm{CO_2}$ ($C_i$) and upscales leaf rates to patch-canopy ground support through Beer-Lambert effective leaf area $L_{\mathrm{eff}}$. `MedlynStomatalConductance` requires explicit construction flag `config["surface_conditions"] = "leaf_surface_equals_canopy_air"` and diagnoses `stomatal_conductance` from supplied `net_photosynthesis` without solving a coupled $C_i$--$g_s$ fixed-point system. Simultaneous coupled leaf boundary-layer/intercellular gas-exchange iteration and sunlit/shaded multilayer radiative transfer remain `[N]`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 20.

Configuration. Absent keys keep the previous behaviour of both Functors bitwise (metadata and every output). `temporal_support` selects `instantaneous` (default: the law once on the supplied inputs) or `daylight_gauss` (Section 2.5, daily step only) in both Functors. `MedlynStomatalConductance` also accepts `canopy_conductance_upscaling: effective_leaf_area_index` (Section 2.4.1, a bulk canopy conductance output) and `nonpositive_leaf_area: raise | no_canopy` (Section 2.4.2). The big-leaf leaf temperature driven by the Medlyn canopy conductance is documented in `biosphere_leaf_energy_balance.md`.

## 2. Governing Equations

### 2.1 $\mathrm{C_3}$ Biochemical Limitations (`pathway = "C3"`)

At leaf level, RuBisCO-limited ($A_c$) and electron-transport-limited ($A_j$) carboxylation rates ($\mathrm{\mu mol\,CO_2\,m^{-2}\,leaf\,s^{-1}}$) are (Farquhar et al., 1980; Bernacchi et al., 2001):

$$
A_c = \max\!\left(0,\; \frac{V_{c\max}(T)\,(C_i - \Gamma^*(T))}{C_i + K_c(T)\bigl(1 + O / K_o(T)\bigr) + \epsilon_{\mathrm{div}}}\right), \qquad
A_j = \max\!\left(0,\; \frac{J\,(C_i - \Gamma^*(T))}{a_1\bigl(C_i + a_2\Gamma^*(T)\bigr) + \epsilon_{\mathrm{div}}}\right)
$$

where $(a_1, a_2) = (4, 2)$ from `rubisco_kinetic_factor1` and `rubisco_kinetic_factor2`, and electron transport $J$ ($\mathrm{\mu mol\,e^-\,m^{-2}\,leaf\,s^{-1}}$) is the lower root of the nonrectangular hyperbola (Farquhar and Wong, 1984):

$$
J = \frac{\alpha I + J_{\max}(T) - \sqrt{\max\!\bigl(0,\; (\alpha I + J_{\max}(T))^2 - 4\theta\alpha I J_{\max}(T)\bigr)}}{2\theta}
$$

With vapor-pressure-deficit factor $f_{\mathrm{VPD}} = (1 + D/D_0)^{-1}$ (Leuning, 1995; $D, D_0$ in $\mathrm{Pa}$) and dimensionless controls $f_w, f_T, f_{\mathrm{O_3}} \in [0, 1]$, the $\mathrm{C_3}$ leaf gross, dark-respiration, and net rates are:

$$
A_{\mathrm{gross,leaf}}^{\mathrm{C3}} = \min(A_c, A_j)\,f_w\,f_T\,f_{\mathrm{O_3}}\,f_{\mathrm{VPD}}, \qquad
R_{d,\mathrm{leaf}}^{\mathrm{C3}} = R_d(T)\,f_w, \qquad
A_{\mathrm{net,leaf}}^{\mathrm{C3}} = A_{\mathrm{gross,leaf}}^{\mathrm{C3}} - R_{d,\mathrm{leaf}}^{\mathrm{C3}}
$$

### 2.2 $\mathrm{C_4}$ Three-Limitation Closure (`pathway = "C4"`)

Following Collatz et al. (1992) and von Caemmerer (2000), PEP-carboxylase ($A_p$), electron-transport ($A_j^{\mathrm{C4}}$), and bundle-sheath RuBisCO ($A_e$) limitations are:

$$
A_p = \max\!\left(0,\; \frac{V_{p\max}(T)\,C_i}{C_i + K_p(T) + \epsilon_{\mathrm{div}}}(1 - \phi_{\mathrm{leak}})\right), \qquad
A_j^{\mathrm{C4}} = \max\!\left(0,\; \frac{J}{f_{e,\mathrm{C4}}}\right), \qquad
A_e = \max\bigl(0,\; V_{e,\max}(T)\bigr)
$$

On the $\mathrm{C_4}$ branch, `water_stress_factor`, `temperature_stress_factor`, and $f_{\mathrm{VPD}}$ scale gross assimilation while neither `ozone_stress_factor` nor water-stress scaling of $R_d(T)$ is applied:

$$
A_{\mathrm{gross,leaf}}^{\mathrm{C4}} = \min(A_p, A_j^{\mathrm{C4}}, A_e)\,f_w\,f_T\,f_{\mathrm{VPD}}, \qquad
R_{d,\mathrm{leaf}}^{\mathrm{C4}} = R_d(T), \qquad
A_{\mathrm{net,leaf}}^{\mathrm{C4}} = A_{\mathrm{gross,leaf}}^{\mathrm{C4}} - R_{d,\mathrm{leaf}}^{\mathrm{C4}}
$$

### 2.3 Temperature Responses, Canopy Upscaling, and Carbon Mass Conversion

Monotonic kinetic constants ($K_c, K_o, \Gamma^*, K_p$) use Arrhenius scaling, dark respiration uses $Q_{10}$ scaling, and enzymatic capacities ($V_{c\max}, J_{\max}, V_{p\max}, V_{e,\max}$) use the normalized peaked Arrhenius law (Medlyn et al., 2002):

$$
f_{\mathrm{Arr}}(T) = f_{25}\exp\!\left[\frac{E_a}{R}\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T}\right)\right], \qquad
R_d(T) = R_{d,25}\,Q_{10,R_d}^{(T - T_{\mathrm{ref}})/10}
$$

$$
f_{\mathrm{peak}}(T) = f_{25}\exp\!\left[\frac{E_a}{R}\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T}\right)\right]\frac{1 + \exp\!\bigl[(S\,T_{\mathrm{ref}} - H_d)/(R\,T_{\mathrm{ref}})\bigr]}{1 + \exp\!\bigl[(S\,T - H_d)/(R\,T)\bigr]}
$$

All leaf fluxes $X \in \{A_{\mathrm{gross}}, A_{\mathrm{net}}, R_d, J, A_j\}$ upscale to ground-area canopy support via $L_{\mathrm{eff}}$ (Monsi and Saeki, 1953; de Pury and Farquhar, 1997), and molar gross assimilation converts to mass GPP:

$$
L_{\mathrm{eff}} = \frac{1 - \exp(-k\,\mathrm{LAI})}{k}, \qquad
X_{\mathrm{canopy}} = X_{\mathrm{leaf}}\,L_{\mathrm{eff}}, \qquad
\mathrm{GPP}_{\mathrm{kgC}} = A_{\mathrm{gross,canopy}}\,M_C\,10^{-9}
$$

### 2.4 Conditional Medlyn Stomatal Conductance (`MedlynStomatalConductance`)

For positive finite $\mathrm{LAI}$ ($L_{\mathrm{eff}} > 0$), representative leaf net assimilation is $A_{\mathrm{net,leaf}} = A_{\mathrm{net,canopy}} / L_{\mathrm{eff}}$ and effective VPD is $D_{\mathrm{eff,kPa}} = 10^{-3}\max(D_{\mathrm{Pa}}, D_{\min,\mathrm{Pa}})$ ($D_{\min,\mathrm{Pa}} = 50\ \mathrm{Pa}$). Following Medlyn et al. (2011) and CTSM Tech Note Eq. 2.9.1:

$$
g_s =
\begin{cases}
g_0 + r_{wc}\!\left(1 + \dfrac{g_1}{\sqrt{D_{\mathrm{eff,kPa}}}}\right)\dfrac{A_{\mathrm{net,leaf}}}{C_{s,\mathrm{ppm}}}, & I > 0 \text{ and } A_{\mathrm{net,leaf}} > 0 \\
g_0, & I \le 0 \text{ or } A_{\mathrm{net,leaf}} \le 0
\end{cases}
$$

#### 2.4.1 Canopy Conductance Upscaling (`canopy_conductance_upscaling: effective_leaf_area_index`)

Purpose: an explicit bulk canopy conductance `canopy_conductance` for canopy-scale readers (`PenmanMonteithET` with `canopy_resistance_source: state`, `LeafEnergyBalance`). The representative-leaf conductance is summed over the same Beer-Lambert effective leaf area that inverts $A_{\mathrm{net,leaf}} = A_{\mathrm{net,canopy}} / L_{\mathrm{eff}}$, so canopy assimilation and canopy conductance share one leaf support (de Pury and Farquhar, 1997; Monteith and Unsworth, 2013, big-leaf canopy of parallel leaf conductances), and the molar conductance is converted to a velocity with the canopy-air temperature and pressure (Campbell and Norman, 1998; Nobel, 2009; `UnitConverter` `conductance_mol_to_ms`, Gerlein-Safdi et al., 2018, Eq. 13):

$$
G_c = g_s\,L_{\mathrm{eff}}, \qquad
\frac{G_c}{A_{\mathrm{net,canopy}}} = \frac{g_s}{A_{\mathrm{net,leaf}}}, \qquad
g_c = G_c\,\frac{R\,T}{P}
$$

with $G_c$ in $\mathrm{mol\,H_2O\,m^{-2}\,ground\,s^{-1}}$, $g_c$ in $\mathrm{m\,s^{-1}}$, $R$ the universal gas constant of the physical-constant SSOT, and $T$ (`air_temperature`) and $P$ (`atmospheric_pressure`) the canopy-air values of `leaf_surface_equals_canopy_air`; both inputs are declared and required only with this key, and $T > 0$, $P > 0$ are checked by `UnitConverter`. $0 < L_{\mathrm{eff}} < \min(\mathrm{LAI}, 1/k)$ and $G_c \ge 0$. Declared approximation: no separate shaded-leaf, leaf boundary-layer or soil-surface conductance enters $g_c$. Absent key: no `canopy_conductance` output and no extra inputs (the previous leaf-only interface).

#### 2.4.2 Non-Positive Leaf Area (`nonpositive_leaf_area`)

`raise` (default, previous behaviour): $\mathrm{LAI} \le 0$ leaves the representative-leaf inversion undefined and raises `ValueError`. `no_canopy` (requires `canopy_conductance_upscaling`): where $\mathrm{LAI} \le 0$ no leaves exist, so

$$
\mathcal{N} = (\mathrm{LAI} \le 0), \qquad g_s = 0 \ \text{and} \ g_c = 0 \ \text{where} \ \mathcal{N},
$$

flagged in `diagnostics.medlyn.no_canopy`. The inversion uses a placeholder $\mathrm{LAI} = 1$ in those cells only to keep the arithmetic finite; every result there is replaced by 0. Downstream, $g_s = 0$ gives $F_{\mathrm{O_3}} = 0$ and $f_{\mathrm{O_3}} = 1$ in `OzoneStress` (`stress.md`), and $g_c = 0$ the dry-leaf or no-canopy branch of `LeafEnergyBalance`. This is an explicit configuration choice, not a literature law.

### 2.5 Daylight Gauss Temporal Support (`temporal_support: daylight_gauss`)

Purpose: on a daily step ($\Delta t = 86400\,\mathrm{s}$) an instantaneous, nonlinear light and temperature law evaluated on 24-hour mean drivers carries the Jensen bias $f(\bar{x}) \ne \overline{f(x)}$. The option keeps the daily step and changes only how the law is evaluated: at $n$ Gauss-Legendre instants of the daylight period and once for the night, averaged with the sample weights (Goudriaan, 1986; Goudriaan and van Laar, 1994; the Gauss scheme of the WOFOST/SUCROS family). The texts of Goudriaan (1986) and Goudriaan and van Laar (1994) were not read for this implementation (Crossref identity check only for Goudriaan, 1986); the verified implementation is the WOFOST source of PCSE (functions `astro` and `totass8`). The shared owner is `relic/scientific/atmosphere/daylight_quadrature.py`: pure tensor functions with no state and no ParameterManager parameter (the coefficients are those of the cited equations and are not tunable). The same support, with the same samples, serves `FarquharPhotosynthesis`, `MedlynStomatalConductance`, `OzoneStress` (`stress.md`), `LeafEnergyBalance` (`biosphere_leaf_energy_balance.md`), `MEGANModel` (`bvoc_emissions.md`) and `PenmanMonteithET` (`hydrosphere_penman_monteith.md`).

Daylight geometry from `latitude` $\varphi$ and `day_of_year` $J$, with the declination $\delta$ of Eq. 24 (Allen et al., 1998, FAO-56 Eqs. 24, 25, 28 and 34; `relic/scientific/atmosphere/solar_geometry.py`, see `radiation.md`):

$$
c = -\tan\varphi\tan\delta, \qquad
\omega_s = \begin{cases} \arccos c, & |c| < 1 \\ 0, & c \ge 1 \ \text{(polar night)} \\ \pi, & c \le -1 \ \text{(polar day)} \end{cases}, \qquad
N = \frac{24\,\omega_s}{\pi}
$$

Samples $k = 0, \dots, n-1$ are the Gauss instants in time order and $k = n$ is the night, on a trailing axis of length $K = n + 1$ ($n = 3$ or $5$). With the Gauss-Legendre nodes $x_i \in [-1, 1]$ and weights $w_i$ normalised to $\sum_i w_i = 1$ (Abramowitz and Stegun, 1964, Section 25.4.29; for $n = 3$ the PCSE `totass8` abscissae and weights):

$$
t_i = 12 + \frac{N}{2}x_i, \qquad
\sin\beta_i = \max\!\left(\sin\varphi\sin\delta + \cos\varphi\cos\delta\cos(\omega_s x_i),\; 0\right)
$$

$$
W_i = \frac{N}{24}\,w_i \ (i < n), \qquad
W_n = 1 - \frac{N}{24}, \qquad
\sum_{k=0}^{n} W_k = 1, \qquad
X = \sum_{k=0}^{n} W_k\,X_k
$$

Shortwave-proportional fluxes (PPFD, net shortwave) are redistributed by the solar elevation and normalised by the Gauss sum, so that the daily total is conserved exactly by the quadrature (PCSE `totass8`, $\mathrm{PAR} = 0.5\,\mathrm{AVRAD}\,s\,/\,\mathrm{DSINBE}$, normalised by the Gauss sum instead of the analytic $\mathrm{DSINBE}$):

$$
s_i = \sin\beta_i\,(1 + 0.4\sin\beta_i) \ \text{(default)} \ \text{or} \ s_i = \sin\beta_i, \qquad
F_i = \frac{\bar{F}\,s_i}{\frac{N}{24}\sum_j w_j s_j}, \qquad
F_n = 0, \qquad
\sum_k W_k F_k = \bar{F}
$$

The factor $1 + 0.4\sin\beta$ is the increase of the atmospheric transmissivity with solar elevation used in PCSE `astro`/`totass8` (Goudriaan and van Laar, 1994, as implemented there). Where $N = 0$ (polar night) the daylight weights vanish and $F_n = \bar{F}$ (the uniform 24-hour placement of the daily-mean support, flagged `light_fallback`).

Air temperature follows the SWAT+ Theoretical Documentation Eq. 1:1.3.1 (Campbell, 1985, cited by SWAT+ and not read here; $0.2618 = 2\pi/24$ rounded there, $\pi/12$ exactly here) with $\bar{T}$ the daily-mean input `air_temperature`; the night sample is the analytic mean of the course over the night $[12 + N/2,\, 36 - N/2]$ of length $Z = 24 - N$:

$$
T(h) = \bar{T} + \frac{T_{\max} - T_{\min}}{2}\cos\!\left(\frac{\pi}{12}(h - 15)\right), \qquad
T_i = T(t_i), \qquad
T_n = \bar{T} + \frac{T_{\max} - T_{\min}}{2}\,\frac{12}{\pi Z}\left[\sin\!\frac{\pi}{12}\!\left(21 - \frac{N}{2}\right) - \sin\!\frac{\pi}{12}\!\left(\frac{N}{2} - 3\right)\right]
$$

The 24-hour mean of $T(h)$ is $\bar{T}$; $T_n = \bar{T}$ at polar day ($Z = 0$, weight 0) and at polar night; $T_{\min} \le \bar{T} \le T_{\max}$ is required. The course reaches $\bar{T} + (T_{\max} - T_{\min})/2$ at 15 h solar time, which exceeds $T_{\max}$ where $\bar{T} > (T_{\max} + T_{\min})/2$. The vapour pressure deficit of each sample uses the daily actual vapour pressure $e_a$ (`actual_vapor_pressure`, held constant over the day) and the Tetens $e^0$ of the shared psychrometrics helper (FAO-56 Eqs. 11 and 13):

$$
D_k = \max\!\left(e^0(T_k) - e_a,\; 0\right)
$$

the floor being flagged (supersaturation of the course at cool instants), not silent. Parton and Logan (1981) was not accessible and is not used.

Configuration: `{"temporal_support": "daylight_gauss", "daylight_gauss_nodes": 3 | 5}` (required node count) and the optional `daylight_shortwave_shape`, `sine_elevation_transmissivity` (default) or `sine_elevation`. $\Delta t$ must equal `seconds_per_day`; `daylight_gauss_nodes` or `daylight_shortwave_shape` on another support raise `ValueError` (no silent ignore). Absent `temporal_support` keeps `instantaneous`, bitwise the previous behaviour.

`FarquharPhotosynthesis` (`_forward_daylight_gauss`): every output $X$ of the law ($A_{\mathrm{net}}$, $A_{\mathrm{gross}}$, $R_d$, GPP, $J$, $A_j$, $J_{\max}$, the limitation indicator) is $X = \sum_k W_k X(T_k, I_k, D_k)$ with $I_k$ the redistributed 24-hour mean `ppfd`; $C_a$, LAI and the scaling and stress factors are held constant over the day. $I_n = 0$, so the night term of $A_{\mathrm{net}}$ is the negative canopy dark respiration $-R_{d,\mathrm{canopy}}(T_n)$. Inputs: `latitude`, `day_of_year`, `air_temperature_max`, `air_temperature_min` and `actual_vapor_pressure` are added, and `actual_vapor_pressure` replaces `vapor_pressure_deficit`. Outputs on the existing ports are 24-hour means; the added port `net_photosynthesis_daylight_samples` ($[\mathrm{cell}, \mathrm{daylight\_sample}]$, $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$) carries $A_{\mathrm{net},k}$ for a `daylight_gauss` Medlyn. The carbon identities are enforced at every sample and hold for the weighted means (they are linear). High-temperature bound: every $T_k$ passes the input check $T_{\min,\mathrm{bio}} \le T_k \le T_{\max,\mathrm{bio}}$ (`validation.numerical.clamp_min_temperature`, `clamp_max_temperature`; defaults 233.15 K and 323.15 K), which raises, never clips. Since $T_k$ can exceed $T_{\max}$ (above), hot days can be refused at the default upper bound; designs that need a wider domain set `validation.numerical.clamp_max_temperature` (and `clamp_min_temperature`) in their run configuration (for example 333.15 K and 203.15 K in the 2026-11-08 daylight-support site validation) and record it; the database default stays 323.15 K. Farquhar with forcing-only inputs needs `environment_source: external_forcing` in the node configuration.

`MedlynStomatalConductance` (`_forward_daylight_gauss`): the law of Section 2.4 (with 2.4.1 and 2.4.2 when configured) at every sample,

$$
g_{s,k} = g_s(A_k, I_k, D_k), \qquad
g_s = \sum_k W_k\,g_{s,k}, \qquad
g_c = \sum_k W_k\,g_{c,k} = \frac{N}{24}\sum_i w_i\,g_{c,i} + \left(1 - \frac{N}{24}\right) g_{c,n}
$$

with $A_k$ the Farquhar net assimilation of the same sample (`net_photosynthesis_daylight_samples` replaces `net_photosynthesis`), $I_k$ the redistributed `ppfd` (dark selector) and $D_k$ from $e_a$ (`actual_vapor_pressure` replaces `vapor_pressure_deficit`); `air_temperature`, `air_temperature_max`, `air_temperature_min`, `latitude` and `day_of_year` are added (the sample $T_k$ also enters the $g_c$ conversion). At the night sample $I_n = 0$ and $A_n = -R_d$ select the $g_0$ branch ($0$ with `no_canopy`). Outputs: 24-hour means on the existing ports plus `stomatal_conductance_daylight_samples` (read by a `daylight_gauss` `OzoneStress`) and, with an upscaling, `canopy_conductance_daylight_samples` (read by a `daylight_gauss` `LeafEnergyBalance` or `PenmanMonteithET`). Boolean diagnostics are combined over the samples by a logical OR (true if the branch is active at any sample).

Coupling design: every instantaneous quantity that one daylight process hands to another travels as its samples (`*_daylight_samples`), and all processes derive bitwise identical instants from the same forcing through the one helper, so sample $k$ of every port refers to the same instant and no daily-mean value is paired with an instantaneous driver. There is no Farquhar-Medlyn cycle: Farquhar evaluates $A_c$ and $A_j$ at $C_i = C_a$ and reads no conductance, so the chain Farquhar $\rightarrow$ Medlyn $\rightarrow$ {OzoneStress, LeafEnergyBalance, PenmanMonteithET} is one-directional. Daily-step readers (bucket, $T_{24}$/$T_{240}$, ThermalStress) read the 24-hour means. In the design probe (one-minute chained reference, four days), pairing each sample with the 24-hour mean assimilation instead of $A_k$ gave daily-mean $g_c$ errors of $-21$ to $-57\,\%$, against $-1.0$ to $+0.2\,\%$ for the per-sample pairing with $n = 5$.

Declared approximations of the support: the daylight shortwave shape is set by the solar elevation only (no diurnal cloud course, no refraction); the temperature course has its maximum at 15 h and its minimum at 3 h solar time, and one night sample at the night-mean temperature slightly biases convex night laws such as $R_d(T)$; $e_a$, $C_a$, LAI, wind, pressure and the daily stress factors are constant over the day; near the polar-night edge ($N \to 0^+$) a positive daily flux is concentrated on a short low-sun day and $F_i$ can exceed the clear-sky value. Accuracy measured against a one-minute reference of the same laws on five days (design note `daylight_gauss_20261106`): largest errors with $n = 3$ were 3.1 % (GPP), 3.7 % ($A_{\mathrm{net}}$) and 5.8 % ($g_s$), and with $n = 5$ 0.35 %, 0.38 % and 1.0 %; $n = 5$ is recommended for production runs and $n = 3$ only as a sensitivity.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T, C_i, I, D$ | Air temperature, ambient $\mathrm{CO_2}$ proxy, PPFD, VPD | $\mathrm{K}$, $\mathrm{\mu mol\,mol^{-1}}$, $\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$, $\mathrm{Pa}$ | `air_temperature`, `atmospheric_co2_ppm`, `ppfd`, `vapor_pressure_deficit` | `forcing` |
| $\mathrm{LAI}$ | One-sided canopy leaf area index | $\mathrm{m^2\,m^{-2}}$ | `leaf_area_index` | `state` |
| $f_w, f_T, f_{\mathrm{O_3}}$ | Optional water, thermal, and ozone stress multipliers (default $1$) | $1$ | `{water,temperature,ozone}_stress_factor` | `diagnostic` / `forcing` |
| $A_{\mathrm{gross}}, A_{\mathrm{net}}$ | Canopy gross and net molar $\mathrm{CO_2}$ assimilation | $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$ | `gross_photosynthesis`, `net_photosynthesis` | `flux` |
| $\mathrm{GPP}_{\mathrm{kgC}}$ | Canopy gross primary productivity mass flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `gross_primary_productivity` | `flux` |
| $J_{\mathrm{canopy}}$ | Canopy electron transport rate | $\mathrm{\mu mol\,e^-\,m^{-2}\,s^{-1}}$ | `electron_transport_rate` | `flux` |
| $g_s$ | Conditional leaf-support stomatal conductance for water vapor | $\mathrm{mol\,H_2O\,m^{-2}\,leaf\,s^{-1}}$ | `stomatal_conductance` | `diagnostic` |
| $g_c$, $\mathcal{N}$ | Bulk canopy conductance (`canopy_conductance_upscaling`) and no-canopy mask (`nonpositive_leaf_area: no_canopy`) | $\mathrm{m\,s^{-1}}$, $1$ | `canopy_conductance`, `diagnostics.medlyn.no_canopy` | `diagnostic` |
| $P$ | Canopy-air pressure of the $g_c$ conversion (`canopy_conductance_upscaling`; $T$ is `air_temperature`) | $\mathrm{Pa}$ | `atmospheric_pressure` | `forcing` |
| $\varphi$, $J$ | Latitude and day of year of the daylight samples (`daylight_gauss`) | $\mathrm{degrees\_north}$, $1$ | `latitude`, `day_of_year` | `forcing` |
| $T_{\max}, T_{\min}$ | Daily maximum and minimum air temperature of the diurnal course (`daylight_gauss`) | $\mathrm{K}$ | `air_temperature_max`, `air_temperature_min` | `forcing` |
| $e_a$ | Daily actual vapour pressure (`daylight_gauss`, replaces $D$) | $\mathrm{Pa}$ | `actual_vapor_pressure` | `forcing` |
| $A_{\mathrm{net},k}$ | Canopy net assimilation at the daylight samples (Farquhar output; Medlyn input with `daylight_gauss`) | $\mathrm{\mu mol\,CO_2\,m^{-2}\,s^{-1}}$ | `net_photosynthesis_daylight_samples` | `flux` |
| $g_{s,k}, g_{c,k}$ | Leaf and canopy conductance at the daylight samples (Medlyn, `daylight_gauss`) | $\mathrm{mol\,m^{-2}\,s^{-1}}$, $\mathrm{m\,s^{-1}}$ | `stomatal_conductance_daylight_samples`, `canopy_conductance_daylight_samples` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $V_{c\max,25}, J_{\max,25}, R_{d,25}$ | `biosphere.photosynthesis.{vcmax25,jmax25,rd25}` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$ | `photosynthesis.yaml` ($60.0, 120.0, 1.2$) | Wullschleger (1993); Bernacchi et al. (2001) |
| $K_{c,25}, K_{o,25}, \Gamma^*_{25}, O$ | `biosphere.photosynthesis.{kc25,ko25,gamma_star25,o2_concentration}` | $\mathrm{\mu mol\,mol^{-1}}$, $\mathrm{mmol\,mol^{-1}}$ | `photosynthesis.yaml` ($404.0, 248.0, 42.75, 210.0$) | Bernacchi et al. (2001) |
| $E_a, H_d, S$ | `biosphere.photosynthesis.{ha_*,hd_*,s_*}` | $\mathrm{J\,mol^{-1}}$, $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `photosynthesis.yaml` | Bernacchi et al. (2001); Medlyn et al. (2002) |
| $\alpha, \theta, k$ | `biosphere.photosynthesis.{alpha,theta,canopy_light_extinction}` | $\mathrm{mol\,e^-\,mol^{-1}}$, $1$, $1$ | `photosynthesis.yaml` ($0.24, 0.7, 0.5$) | Farquhar and Wong (1984); de Pury and Farquhar (1997) |
| $V_{p\max,25}, K_{p,25}, \phi_{\mathrm{leak}}, V_{e,\max,25}, f_{e,\mathrm{C4}}$ | `biosphere.photosynthesis.{vpmax25,kp25,phi,ve_max25}`, `validation.numerical.c4_electron_factor` | $\mathrm{\mu mol\,m^{-2}\,s^{-1}}$, $\mathrm{\mu mol\,mol^{-1}}$, $1$ | `photosynthesis.yaml` / `numerical.yaml` | Collatz et al. (1992); von Caemmerer (2000) |
| $D_0, \epsilon_{\mathrm{div}}$ | `validation.numerical.{vpd_response_d0,epsilon_divide}` | $\mathrm{Pa}$, $\mathrm{\mu mol\,mol^{-1}}$ | `validation/numerical.yaml` | Leuning (1995) |
| $g_0, g_1, D_{\min,\mathrm{Pa}}$ | `coupling.scales.biosphere_atmosphere.{g0,g1,vpd_min_pa}` | $\mathrm{mol\,m^{-2}\,s^{-1}}$, $\mathrm{kPa^{1/2}}$, $\mathrm{Pa}$ | `coupling/biosphere_atmosphere.yaml` ($0.01, 4.0, 50.0$) | Medlyn et al. (2011); CTSM Tech Note |
| $T_{\min,\mathrm{bio}}, T_{\max,\mathrm{bio}}$ | `validation.numerical.clamp_min_temperature`, `validation.numerical.clamp_max_temperature` | $\mathrm{K}$ | `validation_numerical.yaml` ($233.15, 323.15$; set by run designs where a wider domain is needed) | Physiological temperature limits (input check, raises) |
| $e^0$ Tetens constants | `relic.physical.constants.psychrometrics.{tetens_base_pressure,tetens_coefficient_a,tetens_coefficient_b}` (`daylight_gauss`) | $\mathrm{Pa}$, $1$, $^\circ\mathrm{C}$ | `physical/constants.yaml` | Allen et al. (1998) Eqs. 11, 13 |
| $R, T_{\mathrm{ref}}, M_C, r_{wc}$ | `relic.physical.constants.{universal.gas_constant,temperature.reference_temperature,molecular_weights.carbon,conversions.h2o_co2_diffusivity_ratio}` | $\mathrm{J\,mol^{-1}\,K^{-1}}$, $\mathrm{K}$, $\mathrm{g\,mol^{-1}}$, $1$ | `physical/constants.yaml` | Physical-constant SSOT |

## 5. Numerical Methods and Discretization

- `FarquharPhotosynthesis` promotes parameters to the widest floating dtype of its input tensors (`_runtime_dtype`), preserves `ParameterManager` binding identity for `vcmax25`, `jmax25`, and `rd25` across format-3 checkpoints, lowers `[A_gross, A_net, R_d]` through `advance_reaction_network(..., open_system_route="port")` with $N_h = I_3$, and emits nine `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` outputs (`pathway_type = 0` for $\mathrm{C_3}$, `1` for $\mathrm{C_4}$).
- `MedlynStomatalConductance` enforces `EXCLUSIVE` writer ownership on `stomatal_conductance`, rejects zero or negative $\mathrm{LAI}$ or $C_{s,\mathrm{ppm}}$, and preserves autograd within each smooth piecewise branch while reporting `diagnostics.medlyn.{leaf_net_assimilation,dark_residual,nonpositive_assimilation_residual,vpd_floor_active}`.
- `daylight_gauss` (both Functors): the samples are built once per call by `daylight_sample_state`; the light is evaluated on $|x_i|$ and mirrored, so mirrored instants receive bitwise identical light, whereas the temperature (maximum at 15 h) is evaluated at every node, hence the full-day rule rather than the WOFOST half-day rule. $\arccos$ is evaluated only on $|c| < 1$, so no infinite derivative is formed at the polar limits; every builder is differentiable (gradcheck in float64 through the chain). The input checks of the instantaneous law apply to every sample, and $\Delta t \ne 86400\,\mathrm{s}$ raises.

## 6. Conservation, Boundary Conditions, and Ledgers

`FarquharPhotosynthesis` declares the open atmosphere-biosphere carbon boundary check (`carbon_boundary`), verifying both molar and molar-to-mass identities on every call:

$$
R_{\mathrm{mol}} = A_{\mathrm{net}} + R_d - A_{\mathrm{gross}} = 0, \qquad
R_{\mathrm{mass}} = \mathrm{GPP}_{\mathrm{kgC}} - A_{\mathrm{gross}}\,M_C\,10^{-9} = 0
$$

Gross assimilation, dark respiration, and GPP are enforced nonnegative, whereas nighttime $A_{\mathrm{net}}$ may be negative. `MedlynStomatalConductance` advances no material stock and declares `conservation_laws=[]`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `FarquharPhotosynthesis` | `relic/scientific/biosphere/photosynthesis/farquhar.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`PATCH`) |
| `MedlynStomatalConductance` | `relic/scientific/biosphere/photosynthesis/stomatal.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `MAIN_PROCESSES` (`PATCH`) |

## 8. Implementation Status

- `FarquharPhotosynthesis`: `[I]` Level 2 (`reaction`, `iphs_wrapped_empirical`) in `biosphere_photosynthesis_farquhar_farquhar_photosynthesis.yaml` (requires upstream `environmental_forcing` and `atmospheric_radiation` by default; the GraphIR selection field `environment_source = "external_forcing"` removes both, so that $D$ and PPFD arrive through `State:ExternalForcing`, for example from observed tower series, with the law unchanged, and a missing value is refused before the first step).
- `MedlynStomatalConductance`: `[I]` Level 1 (`observation`, `diagnostic`) in `biosphere_photosynthesis_stomatal_medlyn_stomatal_conductance.yaml`.
- Simultaneous coupled leaf-energy, boundary-layer, and intercellular $C_i$--$g_s$ solver and multilayer sunlit/shaded canopy integration: `[N]` (the one-directional big-leaf `LeafEnergyBalance` reads the Medlyn $g_c$ and is documented in `biosphere_leaf_energy_balance.md`).
- `temporal_support: daylight_gauss`, `canopy_conductance_upscaling` and `nonpositive_leaf_area`: `[I]`; the certificates list the added ports, and the frozen expressions below are unchanged (the daylight support evaluates the same expressions at each sample and averages them).

Frozen certificate equivalence expressions:

```text
A = min(Ac, Aj) - Rd; Ac = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + O/Ko)); Aj = J * (Ci - Gamma_star) / (4 * (Ci + 2*Gamma_star))
x_dot = N_h r_h with ParameterManager-backed process rates r_h and declared stoichiometric/output map N_h
L_eff = (1 - exp(-k LAI)) / k; A_leaf = A_ground / L_eff; D_eff = max(D_air, D_min); g_s = g0 + r_wc (1 + g1 / sqrt(D_eff_kPa)) A_leaf / C_air_ppm if ppfd > 0 and A_leaf > 0, else g_s = g0
y_h = O_h x_h through the declared observation operator
```

## 9. References

1. Farquhar, G. D., von Caemmerer, S., and Berry, J. A. (1980). A biochemical model of photosynthetic $\mathrm{CO_2}$ assimilation in leaves of $\mathrm{C_3}$ species. *Planta*, 149, 78-90.
2. Farquhar, G. D., and Wong, S. C. (1984). An empirical model of stomatal conductance. *Australian Journal of Plant Physiology*, 11, 191-210.
3. Collatz, G. J., Ribas-Carbo, M., and Berry, J. A. (1992). Coupled photosynthesis-stomatal conductance model for leaves of $\mathrm{C_4}$ plants. *Australian Journal of Plant Physiology*, 19, 519-538.
4. Wullschleger, S. D. (1993). Biochemical limitations to carbon assimilation in $\mathrm{C_3}$ plants: a retrospective analysis of the $A/C_i$ curves from 109 species. *Journal of Experimental Botany*, 44, 907-920.
5. Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for $\mathrm{C_3}$ plants. *Plant, Cell & Environment*, 18, 339-355.
6. de Pury, D. G. G., and Farquhar, G. D. (1997). Simple scaling of photosynthesis from leaves to canopies without the errors of big-leaf models. *Plant, Cell & Environment*, 20, 537-557.
7. von Caemmerer, S. (2000). *Biochemical Models of Leaf Photosynthesis*. CSIRO Publishing.
8. Bernacchi, C. J., Singsaas, E. L., Pimentel, C., Portis, A. R., and Long, S. P. (2001). Improved temperature response functions for models of Rubisco-limited photosynthesis. *Plant, Cell & Environment*, 24, 253-259.
9. Medlyn, B. E., Dreyer, E., Ellsworth, D., et al. (2002). Temperature response of parameters of a biochemically based model of photosynthesis. II. A review of experimental data. *Plant, Cell & Environment*, 25, 1167-1179.
10. Medlyn, B. E., Duursma, R. A., Eamus, D., et al. (2011). Reconciling the optimal and empirical approaches to modelling stomatal conductance. *Global Change Biology*, 17, 2134-2144.
11. Monsi, M., and Saeki, T. (1953). Ueber den Lichtfaktor in den Pflanzengesellschaften und seine Bedeutung fuer die Stoffproduktion. *Japanese Journal of Botany*, 14, 22-52.
12. Allen, R. G., Pereira, L. S., Raes, D., and Smith, M. (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO. Eqs. 11, 13, 24, 25, 28 and 34.
13. Goudriaan, J. (1986). A simple and fast numerical method for the computation of daily totals of crop photosynthesis. *Agricultural and Forest Meteorology*, 38, 249-254 (cited for the scheme; text not read).
14. Goudriaan, J., and van Laar, H. H. (1994). *Modelling Potential Crop Growth Processes*. Kluwer Academic Publishers (cited for the scheme; text not read).
15. WOFOST implementation in PCSE (Wageningen Environmental Research), functions `astro` and `totass8`, source read 2026-11-06 at pcse.readthedocs.io.
16. Abramowitz, M., and Stegun, I. A. (1964). *Handbook of Mathematical Functions*, Section 25.4.29 (Gauss-Legendre quadrature).
17. SWAT+ Theoretical Documentation, Section 1:1.3.2 Hourly Air Temperature, Eq. 1:1.3.1, read 2026-11-06 at swatplus.gitbook.io.
18. Campbell, G. S. (1985). *Soil Physics with BASIC*. Elsevier (cited by SWAT+; not read).
19. Monteith, J. L., and Unsworth, M. H. (2013). *Principles of Environmental Physics* (4th ed.). Academic Press.
20. Campbell, G. S., and Norman, J. M. (1998). *An Introduction to Environmental Biophysics* (2nd ed.). Springer.
21. Nobel, P. S. (2009). *Physicochemical and Environmental Plant Physiology* (4th ed.). Academic Press.
22. Gerlein-Safdi et al. (2018), Eq. 13, as cited by `UnitConverter` `conductance_mol_to_ms` (`relic/data/standards/unit_converter.py`).
