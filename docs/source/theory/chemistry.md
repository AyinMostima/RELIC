# Atmospheric Chemistry

## 1. Process Definition and Scope

`ChemistryFunctor` (`relic\scientific\atmosphere\chemistry.py`) implements box-model gas-phase atmospheric chemistry under the `PhysicalField` contract (`SPATIAL_DIMS = None`). Theoretical foundations for gas-phase reaction kinetics, photolysis, and GENERIC reaction thermodynamics appear in *Methods of Ecological Analysis*, Chapters 8, 9, 17, and 18.

Construction requires an explicit `GraphIR` selection keyword `physical_boundary: Literal["open_mechanism", "closed_constant_volume"]`, which selects one of two governed process laws on the same class:

1. **Open tropospheric mechanism (`physical_boundary="open_mechanism"`)**: advances a reduced $\mathrm{NO_x}$-$\mathrm{O_3}$-isoprene gas-phase reaction network (`no_mixing_ratio`, `no2_mixing_ratio`, `o3_mixing_ratio`, and optional `isoprene_mixing_ratio`) driven by broadband shortwave irradiance, solar geometry, Arrhenius kinetics, primary $\mathrm{O(^1D)}+\mathrm{H_2O}$ steady-state $\mathrm{OH}$, and high-$\mathrm{NO_x}$ peroxy-radical ($\mathrm{RO_2}+\mathrm{NO}$) conversion. Dry deposition is excluded here and owned solely by `DryDepositionAggregator`.
2. **Closed constant-volume adiabatic box (`physical_boundary="closed_constant_volume"`)**: advances a closed, rigid, adiabatic scalar ideal-gas mixture of $\mathrm{NO}$, $\mathrm{NO_2}$, $\mathrm{O_3}$, and $\mathrm{O_2}$ plus internal energy $U$ at fixed volume $V$ under the single forward titration reaction $\mathrm{NO} + \mathrm{O_3} \rightarrow \mathrm{NO_2} + \mathrm{O_2}$ via the canonical GENERIC absorbed-reaction drift.

## 2. Governing Equations

### 2.1 Open Mechanism (`physical_boundary="open_mechanism"`)

Ideal-gas air number density $n_{\mathrm{air}}$ ($\mathrm{molecule\,cm^{-3}}$) and species number density $[X]$ from mixing ratio $\chi_X$ ($\mathrm{mol\,mol^{-1}}$) are

$$n_{\mathrm{air}} = \frac{p}{k_B T c_v}, \qquad [X] = \chi_X n_{\mathrm{air}}, \qquad c_v = 10^6\;\mathrm{cm^3\,m^{-3}}.$$

Bimolecular rate coefficients use the reference-anchored Arrhenius form at $T_{\mathrm{ref}} = 298.15\;\mathrm{K}$:

$$k(T) = k_{298}\exp\!\left[\frac{E_a}{R}\left(\frac{1}{T_{\mathrm{ref}}} - \frac{1}{T}\right)\right].$$

Broadband global shortwave irradiance $G =$ `shortwave_down` ($\mathrm{W\,m^{-2}}$) and solar zenith cosine $\mu = \cos\chi =$ `cos_zenith` drive the Trebs et al. (2009) $\mathrm{NO_2}$ photolysis closure (clamped at parabola vertex $G_v = B_1 / (2\lvert B_2 \rvert)$) and the MCM v3.3.1 $\mathrm{O_3} \rightarrow \mathrm{O(^1D)}$ ratio $\rho(\chi)$:

$$G_h = \min(G, G_v), \qquad G_v = \frac{B_1}{2\lvert B_2 \rvert}, \qquad J_{\mathrm{ref}} = (1 + \alpha)\left(B_1 G_h + B_2 G_h^2\right),$$

$$\rho(\chi) = \begin{cases} \dfrac{l_1}{l_4}\mu^{m_1 - m_4}\exp\!\left(-\dfrac{n_1 - n_4}{\mu}\right), & \mu > 0, \\ 0, & \mu \le 0, \end{cases}$$

$$J_{\mathrm{NO_2}} = J_{\mathrm{ref}}\,R_{\mathrm{NO_2}}(T, \mu), \qquad J_{\mathrm{O(^1D)}} = J_{\mathrm{ref}}\,\rho(\chi)\,R_{\mathrm{O(^1D)}}(T, \mu).$$

In default mode `photolysis_temperature_mode="none"`, $R_X(T, \mu) = 1$. In `"spectral_quantum_yield"`, $R_X(T, \mu) = S_X(T, \mu) / S_X(T_{\mathrm{ref}}, \mu)$ (or $1$ when $\mu \le 0$) with direct-beam spectral weight

$$S_X(T, \mu) = \sum_{\lambda} \sigma_X(\lambda, T)\,\phi_X(\lambda, T)\,F_0(\lambda)\exp\!\left[-\frac{\tau_R(\lambda) + \tau_{\mathrm{O_3}}(\lambda)}{\mu}\right].$$

From specific humidity $q \in [0, 1)$ ($\mathrm{kg\,kg^{-1}}$), water-vapor mole fraction $x_w$, $\mathrm{O(^1D)}+\mathrm{H_2O}$ branching fraction $f$, primary $\mathrm{OH}$ production $P_{\mathrm{OH}}$, first-order $\mathrm{OH}$ loss frequency $L_{\mathrm{OH}} > 0$, and diagnostic steady-state $[\mathrm{OH}]$ ($\mathrm{molecule\,cm^{-3}}$) are

$$x_w = \frac{q / M_w}{q / M_w + (1 - q)/M_d}, \qquad f = \frac{k_{\mathrm{H_2O}}[\mathrm{H_2O}]}{k_{\mathrm{H_2O}}[\mathrm{H_2O}] + k_{\mathrm{N_2}}[\mathrm{N_2}] + k_{\mathrm{O_2}}[\mathrm{O_2}]},$$

$$P_{\mathrm{OH}} = y\,f\,J_{\mathrm{O(^1D)}}[\mathrm{O_3}], \qquad L_{\mathrm{OH}} = k_{\mathrm{OH+CO}}[\mathrm{CO}] + k_{\mathrm{OH+CH_4}}[\mathrm{CH_4}] + k_{\mathrm{OH+NO_2}}[\mathrm{NO_2}] + k_{\mathrm{OH+C_5H_8}}[\mathrm{C_5H_8}], \qquad [\mathrm{OH}] = \frac{P_{\mathrm{OH}}}{L_{\mathrm{OH}}}.$$

In mixing-ratio rate units ($\mathrm{mol\,mol^{-1}\,s^{-1}}$), the five reactions evaluated through `advance_reaction_network` on state vector $x_h = \begin{bmatrix}\chi_{\mathrm{NO}} & \chi_{\mathrm{NO_2}} & \chi_{\mathrm{O_3}} & \chi_{\mathrm{C_5H_8}}\end{bmatrix}^{\mathsf{T}}$ are

$$R_{\mathrm{photo,NO_2}} = J_{\mathrm{NO_2}}\chi_{\mathrm{NO_2}}, \qquad R_{\mathrm{NO+O_3}} = k_{\mathrm{NO+O_3}}(T)n_{\mathrm{air}}\chi_{\mathrm{NO}}\chi_{\mathrm{O_3}},$$

$$R_{\mathrm{C_5H_8+OH}} = k_{\mathrm{C_5H_8+OH}}(T)[\mathrm{OH}]\chi_{\mathrm{C_5H_8}}, \qquad R_{\mathrm{C_5H_8+O_3}} = k_{\mathrm{C_5H_8+O_3}}(T)n_{\mathrm{air}}\chi_{\mathrm{C_5H_8}}\chi_{\mathrm{O_3}},$$

$$\beta_{\mathrm{NO}} = \frac{k_{\mathrm{RO_2+NO}}(T)[\mathrm{NO}]}{k_{\mathrm{RO_2+NO}}(T)[\mathrm{NO}] + k_{\mathrm{loss,RO_2}}}, \qquad R_{\mathrm{RO_2+NO}} = \gamma_{\mathrm{NO_2}}\,\beta_{\mathrm{NO}}\,R_{\mathrm{C_5H_8+OH}},$$

yielding the stoichiometric network drift $\dot{x}_h = N_h r_h$:

$$\frac{d\chi_{\mathrm{NO}}}{dt} = R_{\mathrm{photo,NO_2}} - R_{\mathrm{NO+O_3}} - R_{\mathrm{RO_2+NO}}, \qquad \frac{d\chi_{\mathrm{NO_2}}}{dt} = R_{\mathrm{NO+O_3}} - R_{\mathrm{photo,NO_2}} + R_{\mathrm{RO_2+NO}},$$

$$\frac{d\chi_{\mathrm{O_3}}}{dt} = R_{\mathrm{photo,NO_2}} - R_{\mathrm{NO+O_3}} - R_{\mathrm{C_5H_8+O_3}}, \qquad \frac{d\chi_{\mathrm{C_5H_8}}}{dt} = -\left(R_{\mathrm{C_5H_8+OH}} + R_{\mathrm{C_5H_8+O_3}}\right),$$

with odd-oxygen tendency $\frac{d\chi_{\mathrm{O_x}}}{dt} = \frac{d(\chi_{\mathrm{O_3}} + \chi_{\mathrm{NO_2}})}{dt} = R_{\mathrm{RO_2+NO}} - R_{\mathrm{C_5H_8+O_3}}$ and oxidized carbon rate $R_{\mathrm{C,ox}} = N_{\mathrm{C,isop}}\left(R_{\mathrm{C_5H_8+OH}} + R_{\mathrm{C_5H_8+O_3}}\right)$ (the latter three rates equal zero when `isoprene_mixing_ratio` is absent). The pure diagnostic helper `compute_isoprene_chemical_lifetime` evaluates pseudo-first-order isoprene lifetime $\tau_{\mathrm{C_5H_8}} = \left(k_{\mathrm{OH}}(T)[\mathrm{OH}] + k_{\mathrm{O_3}}(T)[\mathrm{O_3}] + k_{\mathrm{NO_3}}(T)[\mathrm{NO_3}]\right)^{-1}$.

### 2.2 Closed Constant-Volume GENERIC Box (`physical_boundary="closed_constant_volume"`)

For ordered species $(\mathrm{NO}, \mathrm{NO_2}, \mathrm{O_3}, \mathrm{O_2})$ over $200\text{--}6000\;\mathrm{K}$ (two NASA7 intervals split at $1000\;\mathrm{K}$, standard pressure $p^\circ = 1\;\mathrm{bar}$), `standard_molar_properties` and `ideal_gas_mixture_properties` evaluate

$$\frac{c_{p,i}^\circ}{R} = \sum_{m=1}^{5} a_{m,i} T^{m-1}, \qquad \frac{h_i^\circ}{R T} = \sum_{m=1}^{5} \frac{a_{m,i}}{m} T^{m-1} + \frac{a_{6,i}}{T}, \qquad \frac{s_i^\circ}{R} = a_{1,i}\ln\!\left(\frac{T}{1\;\mathrm{K}}\right) + \sum_{m=2}^{5} \frac{a_{m,i}}{m-1} T^{m-1} + a_{7,i},$$

$$u_i^\circ = h_i^\circ - R T, \qquad c_{v,i}^\circ = c_{p,i}^\circ - R, \qquad p_i = \frac{n_i R T}{V}, \qquad \mu_i = h_i^\circ - T s_i^\circ + R T \ln\!\left(\frac{p_i}{p^\circ}\right),$$

$$U(T, n) = \sum_{i=1}^{4} n_i u_i^\circ(T), \qquad S(U, n, V) = \sum_{i=1}^{4} n_i \left[s_i^\circ(T) - R\ln\!\left(\frac{p_i}{p^\circ}\right)\right], \qquad F = U - T S.$$

On scalar state $x = (n_{\mathrm{NO}}, n_{\mathrm{NO_2}}, n_{\mathrm{O_3}}, n_{\mathrm{O_2}}, U)^{\mathsf{T}}$ at fixed $V > 0$, temperature $T(U, n)$ is inverted via `bracketed_bisection` on $\sum_i n_i u_i^\circ(T) - U = 0$ within one continuous NASA7 interval. With stoichiometric vector $\nu = (-1, +1, -1, +1, 0)^{\mathsf{T}}$, forward affinity $A > 0$, and mass-action rate $r$ ($\mathrm{mol\,s^{-1}}$):

$$A = -\nu^{\mathsf{T}}\tilde{\mu} > 0, \qquad r = k_{\mathrm{NO+O_3}}(T)\,N_A\,\frac{n_{\mathrm{NO}} n_{\mathrm{O_3}}}{c_v V}, \qquad E = U, \qquad \nabla E = (0, 0, 0, 0, 1)^{\mathsf{T}}, \qquad \nabla S = \left(-\frac{\mu^{\mathsf{T}}}{T},\; \frac{1}{T}\right)^{\mathsf{T}},$$

$$L = 0, \qquad M = \frac{r T}{A}\nu \nu^{\mathsf{T}} \succeq 0, \qquad \dot{x} = L\nabla E + M\nabla S = \nu r, \qquad M\nabla E = 0, \qquad \frac{dE}{dt} = 0, \qquad \sigma = \nabla S^{\mathsf{T}} M \nabla S = \frac{dS}{dt} = \frac{A r}{T} \ge 0.$$

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $T, p, q$ | Air temperature, atmospheric pressure, specific humidity (`open_mechanism`) | $\mathrm{K}$, $\mathrm{Pa}$, $\mathrm{kg\,kg^{-1}}$ | `air_temperature`, `atmospheric_pressure`, `specific_humidity` | `forcing` |
| $G, \mu$ | Downwelling shortwave irradiance and solar zenith cosine (`open_mechanism`) | $\mathrm{W\,m^{-2}}$, $1$ | `shortwave_down`, `cos_zenith` | `forcing` |
| $\chi_{\mathrm{CO}}, \chi_{\mathrm{CH_4}}$ | Required $\mathrm{OH}$-sink mixing ratios (`open_mechanism`) | $\mathrm{mol\,mol^{-1}}$ | `co_mixing_ratio`, `ch4_mixing_ratio` | `forcing` |
| $\chi_{\mathrm{NO}}, \chi_{\mathrm{NO_2}}, \chi_{\mathrm{O_3}}$ | Required species mixing ratios and per-step increments (`open_mechanism`) | $\mathrm{mol\,mol^{-1}}$ | `no_mixing_ratio`, `no2_mixing_ratio`, `o3_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $\chi_{\mathrm{C_5H_8}}$ | Optional isoprene mixing ratio and per-step increment (`open_mechanism`) | $\mathrm{mol\,mol^{-1}}$ | `isoprene_mixing_ratio` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $J_{\mathrm{NO_2}}, k_{\mathrm{NO+O_3}}$ | $\mathrm{NO_2}$ photolysis frequency and $\mathrm{NO}+\mathrm{O_3}$ rate constant | $\mathrm{s^{-1}}$, $\mathrm{cm^3\,molecule^{-1}\,s^{-1}}$ | `diagnostics.chemistry.{photolysis_rate_no2,rate_constant_no_o3}` | `diagnostic` |
| $R_{\mathrm{photo}}, R_{\mathrm{NO+O_3}}, R_{\mathrm{iso}}$ | Mixing-ratio rates of photolysis, titration, total isoprene oxidation | $\mathrm{mol\,mol^{-1}\,s^{-1}}$ | `diagnostics.chemistry.{photolysis_no2_rate,reaction_no_o3_rate,isoprene_oxidation_rate}` | `diagnostic` |
| $R_{\mathrm{RO_2+NO}}, \beta_{\mathrm{NO}}$ | Peroxy-radical $\mathrm{NO}$ conversion rate and branching fraction | $\mathrm{mol\,mol^{-1}\,s^{-1}}$, $1$ | `diagnostics.chemistry.{ro2_no_conversion_rate,ro2_no_branching_fraction}` | `diagnostic` |
| $\dot{\chi}_{\mathrm{O_x}}, R_{\mathrm{C,ox}}$ | Net odd-oxygen tendency and oxidized isoprene carbon rate | $\mathrm{mol\,mol^{-1}\,s^{-1}}$ | `diagnostics.chemistry.{ox_net_chemical_tendency,carbon_oxidized_rate}` | `diagnostic` |
| $[\mathrm{OH}], n_{\mathrm{air}}, G$ | Steady-state $\mathrm{OH}$, air number density, shortwave used | $\mathrm{molecule\,cm^{-3}}$, $\mathrm{molecule\,cm^{-3}}$, $\mathrm{W\,m^{-2}}$ | `diagnostics.chemistry.{oh_concentration,air_number_density,shortwave_radiation_used}` | `diagnostic` |
| $n_{\mathrm{NO}}, n_{\mathrm{NO_2}}, n_{\mathrm{O_3}}, n_{\mathrm{O_2}}$ | Closed-box molar inventories (`closed_constant_volume`) | $\mathrm{mol}$ | `box_{no,no2,o3,o2}_amount` | `state` (`ACCUMULATE`, `per_timestep=True`) |
| $U, V$ | Closed-box internal energy (`state`) and fixed volume (`forcing`) | $\mathrm{J}$, $\mathrm{m^3}$ | `box_internal_energy`, `box_volume` | `state` / `forcing` |
| $T, p, S, \sigma, A, r, P, k$ | Closed-box thermodynamic and GENERIC reaction diagnostics | SI | `diagnostics.chemistry.box_{temperature,pressure,entropy,entropy_production_rate,reaction_affinity,reaction_rate,reaction_power,rate_constant_no_o3}` | `diagnostic` |

## 4. Parameter Table

| Symbol | `ParameterManager` / Physical-Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $B_1, B_2, \alpha$ | `atmosphere.chemistry.{j_no2_irradiance_linear_coefficient,j_no2_irradiance_quadratic_coefficient,uv_a_surface_albedo}` | $\mathrm{W^{-1}\,m^2\,s^{-1}}$, $\mathrm{W^{-2}\,m^4\,s^{-1}}$, $1$ | `atmosphere/chemistry.yaml` | Trebs et al. (2009) |
| $l_1, m_1, n_1, l_4, m_4, n_4$ | `atmosphere.chemistry.mcm_{o1d,no2}_photolysis_{l,m,n}` | $\mathrm{s^{-1}}$, $1$, $1$ | `atmosphere/chemistry.yaml` | Saunders et al. (2003), MCM v3.3.1 |
| $\text{mode}$ | `atmosphere.chemistry.photolysis_temperature_mode` | $1$ | `atmosphere/chemistry.yaml` (`"none"`) | Troe (2000); Matsumi et al. (2002) |
| $\sigma_X, \phi_X, F_0, \tau_R$ | `atmosphere.chemistry.photolysis.*` (35 spectral keys) | SI / CGS | `atmosphere/photolysis_spectra.yaml` | Warneck and Williams (2012); Bodhaine et al. (1999) |
| $k_{298}, E_a$ | `atmosphere.chemistry.k_{no_o3,isoprene_oh,isoprene_o3,isoprene_no3,oh_co,oh_ch4,oh_no2,o1d_h2o,o1d_n2,o1d_o2,ro2_no}_{298,activation_energy}` | $\mathrm{cm^3\,molecule^{-1}\,s^{-1}}$, $\mathrm{J\,mol^{-1}}$ | `atmosphere/chemistry.yaml` | Atkinson et al. (2004, 2006); Sander et al. (2006); Jenkin et al. (2015) |
| $y, k_{\mathrm{loss,RO_2}}, \gamma_{\mathrm{NO_2}}, N_{\mathrm{C,isop}}$ | `atmosphere.chemistry.{o1d_water_oh_yield,k_ro2_termination_loss,isoprene_ro2_no2_stoichiometric_yield,isoprene_carbon_atoms}` | $1$, $\mathrm{s^{-1}}$, $1$, $1$ | `atmosphere/chemistry.yaml` | Seinfeld and Pandis (2016); Peeters et al. (2014) |
| $T_{\min}, T_{\max}$ | `atmosphere.chemistry.{temperature_min,temperature_max}` | $\mathrm{K}$ | `atmosphere/chemistry.yaml` (`200`, `400`) | Operational kinetic bounds |
| $Q_{10,\mathrm{chem}}, \chi_{\min}$ | `atmosphere.chemistry.{chemistry_temperature_q10,minimum_concentration}` | $1$, $\mathrm{mol\,mol^{-1}}$ | `atmosphere/chemistry.yaml` | Level 3 Hodge audit scale only |
| $a_1 \dots a_7$ | `atmosphere.chemistry.nasa7.{model,species_order,temperature_ranges,a1..a7}` | NASA7 | `atmosphere/chemistry.yaml` | McBride et al. (1993), NASA TM-4513 |
| $\text{solver}, \epsilon_{\mathrm{IPHS}}$ | `computation.solvers.bracketed_bisection.{max_iterations,rtol,atol}`, `validation.iphs.runtime_point_tolerance` | $1$ | `computation/`, `validation/` YAML | Solver and Level 5 audit tolerances |
| $R, k_B, N_A, h, c, p^\circ, M_w, M_d$ | `get_physical_constant(...)` | SI | `physical/constants.yaml` | Physical-constants SSOT |

## 5. Numerical Methods and Discretization

In `open_mechanism`, `advance_reaction_network` forms the explicit Euler increment $\Delta x_h = \Delta t\,N_h r_h$. Before emitting any `UpdateItem`, `_validate_euler_positivity` verifies for all retained species with net negative tendency $f_i < 0$ that

$$\Delta t \le \Delta t_{\max} = \min_{i:\,f_i < 0} \frac{\chi_i^n}{-f_i}.$$

Violating this positivity bound raises `ValueError` naming the violating species and $\Delta t_{\max}$; no concentration floor or increment clipping is applied, and smaller chemical steps are handled by `FunctorMetadata.substep_count` in `RELICSimulator`.

In `closed_constant_volume`, `IPHSStructureSpec.canonical_drift` (`poisson_operator="zero_field"`, `reaction_representation="absorbed"`) evaluates $\dot{x}$ using `torch.autograd` gradients of $E$ and $S$ (including implicit VJP gradients through `bracketed_bisection`). Both the pre-step state $x$ and the explicit Euler candidate $x^{\mathrm{cand}} = x + \Delta t\,\dot{x}$ must satisfy $n_i > 0$, $V > 0$, single-interval NASA7 temperature inversion within $[T_{\min}, T_{\max}]$, and $A > 0$. Every emitted step executes `IPHSStructureSpec.validate(required_level=5)` to verify $L\nabla S = 0$, $M\nabla E = 0$, $\sigma \ge 0$, $(\nabla E)^{\mathsf{T}} N r = 0$, $(\nabla S)^{\mathsf{T}} N r \ge 0$, and $M\nabla S = N r$ within `validation.iphs.runtime_point_tolerance`.

## 6. Conservation, Boundary Conditions, and Ledgers

- **Open mechanism (`open_mechanism`)**: the reduced $\mathrm{NO}$/$\mathrm{NO_2}$ family obeys exact local nitrogen conservation $C_{\mathrm{NO_x}} N_h = 0$ with $C_{\mathrm{NO_x}} = \begin{bmatrix}1 & 1 & 0 & 0\end{bmatrix}$, so $\Delta\chi_{\mathrm{NO}} + \Delta\chi_{\mathrm{NO_2}} = 0$ across photolysis, titration, and $\mathrm{RO_2}+\mathrm{NO}$ conversion. Unretained $\mathrm{O_2}$, radicals, and isoprene oxidation products leave through the declared `boundary_export` open-system route.
- **Closed box (`closed_constant_volume`)**: zero boundary ports (`ports=()`), exact internal-energy conservation $\Delta U = 0$, non-negative entropy production $\sigma = A r / T \ge 0$, and exact elemental conservation $C_{\mathrm{N}}\nu = 0$ with $C_{\mathrm{N}} = (1, 1, 0, 0, 0)$ and $C_{\mathrm{O}}\nu = 0$ with $C_{\mathrm{O}} = (1, 2, 3, 2, 0)$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `ChemistryFunctor` | `relic\scientific\atmosphere\chemistry.py` | `PhysicalField` | `PhysicalFieldFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **Runtime status**: `[I]` implemented dual-boundary gas-phase chemistry Functor (`open_mechanism` and `closed_constant_volume`), returning pre-integrated `UpdateItem(ACCUMULATE, per_timestep=True)` species increments and `UpdateItem(REPLACE)` diagnostics.
- **IPHS certificates**:
  - **Level 3** main class certificate (`docs\source\theory\iphs_certificates\atmosphere_chemistry_chemistry_functor.yaml`, IPHS-structured field) for `physical_boundary="open_mechanism"`; its diagonal Hodge audit mobility $\star_{\mathrm{rxn}} = \operatorname{diag}(J_{\max} Q_{10,\mathrm{chem}})$ with $J_{\max} = (1+\alpha)B_1^2 / (4\lvert B_2 \rvert)$ is a static structural audit object and does not claim strict thermodynamic GENERIC closure.
  - **Level 5** constructor-scoped runtime-point certificate (`docs\source\theory\iphs_certificates\atmosphere_chemistry_chemistry_functor_closed_constant_volume.yaml`, `constructor_selection: {physical_boundary: closed_constant_volume}`, strict GENERIC absorbed-reaction field), validated on emitted governed steps via `last_strict_iphs_audit()` and a committed step witness.

## 9. References

1. Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics: From Air Pollution to Climate Change* (3rd ed.). Wiley.
2. Finlayson-Pitts, B. J., & Pitts, J. N. (2000). *Chemistry of the Upper and Lower Atmosphere*. Academic Press.
3. Trebs, I., Bohn, B., Ammann, C., Rummel, U., Blumthaler, M., Konigstedt, R., Meixner, F. X., Fan, S., & Andreae, M. O. (2009). Relationship between the $\mathrm{NO_2}$ photolysis frequency and the solar global irradiance. *Atmospheric Measurement Techniques*, 2, 725-739.
4. Atkinson, R., Baulch, D. L., Cox, R. A., Crowley, J. N., Hampson, R. F., Hynes, R. G., Jenkin, M. E., Rossi, M. J., & Troe, J. (2004, 2006). Evaluated kinetic and photochemical data for atmospheric chemistry: Volumes I and II. *Atmospheric Chemistry and Physics*, 4, 1461-1738; 6, 3625-4055.
5. Saunders, S. M., Jenkin, M. E., Derwent, R. G., & Pilling, M. J. (2003). Protocol for the development of the Master Chemical Mechanism, MCM v3 (Part A). *Atmospheric Chemistry and Physics*, 3, 161-180.
6. Jenkin, M. E., Young, J. C., & Rickard, A. R. (2015). The MCM v3.3.1 degradation scheme for isoprene. *Atmospheric Chemistry and Physics*, 15, 11433-11459.
7. Peeters, J., Muller, J.-F., Stavrakou, T., & Nguyen, V. S. (2014). Hydroxyl radical recycling in isoprene oxidation driven by hydrogen bonding and hydrogen tunneling: The upgraded LIM1 mechanism. *Journal of Physical Chemistry A*, 118(38), 8625-8643.
8. Warneck, P., & Williams, J. (2012). *The Atmospheric Chemist's Companion*. Springer.
9. Matsumi, Y., Comes, F. J., Hancock, G., Hofzumahaus, A., Hynes, A. J., Kawasaki, M., & Ravishankara, A. R. (2002). Quantum yields for production of $\mathrm{O(^1D)}$ in the ultraviolet photolysis of ozone. *Journal of Geophysical Research*, 107(D3), 4024.
10. Troe, J. (2000). Are primary quantum yields of $\mathrm{NO_2}$ photolysis at $\lambda \le 398\;\mathrm{nm}$ smaller than unity? *Zeitschrift fur Physikalische Chemie*, 214, 573-581.
11. Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R. (1999). On Rayleigh optical depth calculations. *Journal of Atmospheric and Oceanic Technology*, 16, 1854-1861.
12. McBride, B. J., Gordon, S., & Reno, M. A. (1993). *Coefficients for Calculating Thermodynamic and Transport Properties of Individual Species* (NASA Technical Memorandum 4513). NASA.
13. Grmela, M., & Ottinger, H. C. (1997). Dynamics and thermodynamics of complex fluids. I. Development of a general formalism. *Physical Review E*, 56(6), 6620-6632.
14. Mielke, A. (2011). Formulation of thermoelastic dissipative material behavior using GENERIC. *Continuum Mechanics and Thermodynamics*, 23, 233-256.
15. Sandu, A., Verwer, J. G., Blom, J. G., Spee, E. J., Carmichael, G. R., & Potra, F. A. (1997). Benchmarking stiff ODE solvers for atmospheric chemistry problems I: Implicit versus explicit. *Atmospheric Environment*, 31(19), 3151-3166.
