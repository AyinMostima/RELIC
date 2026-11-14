# Stable Isotopes and General Transport Theory

## Scope, component inventory, and certificate map

This page synthesizes two cross-cutting RELIC packages (*Methods of Ecological
Analysis*, Chapters 23, 25, and 26):

1. **Stable-isotope fractionation and ledger lifecycle (`relic.scientific.isotopes`)**:
   stateless exact-partition binary fractionation calculators (`[I]`), the
   fail-closed `Level 0` runtime boundary `IsotopeHeartbeatFunctor` (`[N]`), and
   the retired legacy pool-tracker status (`[X]`).
2. **General continuum and discrete cochain transport (`relic.scientific.transport`)**:
   finite-volume upwind advection, Fickian diffusion (at both `Level 3` open
   boundaries and its `Level 4` closed 1D dissipative port-Hamiltonian runtime
   point), and operator-split advection-diffusion.

Across these two packages, four concrete scientific Functors carry five
declarative `IPHSEquivalenceCertificate` files (four primary class certificates
plus one secondary `Level 4` constructor-conditioned certificate on
`FickianDiffusion`):

| Package | Component | Canonical bridge / type | Cert. level | Status | Primary runtime role and state semantics |
| --- | --- | --- | --- | --- | --- |
| `relic.scientific.isotopes.functor` | {ref}`IsotopeHeartbeatFunctor <class-relic-scientific-isotopes-functor-isotopeheartbeatfunctor>` | `ScaleCouplingFunctor` | `Level 0` | `[N]` | Fail-closed isotope ledger boundary (`runtime_unbound`); `wire_isotope_tracker(...)` rejects before mutation |
| `relic.scientific.isotopes.fractionation` | `Carbon13FractionationCalculator`, `Nitrogen15FractionationCalculator`, `Oxygen18FractionationCalculator` | Stateless utility classes (`IsotopeFractionationCalculator`) | Uncertified utility | `[I]` | Exact binary rational heavy/light flux partitioning, Rayleigh distillation, C3 discrimination, and Craig-Gordon ratios |
| Historical (`MultiIsotopePoolTracker`) | `MultiIsotopePoolTracker` | Retired legacy class | Removed | `[X]` | Deleted non-Functor pool tracker; replaced by governed `StateManager` single-writer rules |
| `relic.scientific.transport.advection` | {ref}`LinearAdvection <class-relic-scientific-transport-advection-linearadvection>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Conservative finite-volume upwind tracer advection (`ACCUMULATE`, `per_timestep=True`) |
| `relic.scientific.transport.diffusion` | {ref}`FickianDiffusion <class-relic-scientific-transport-diffusion-fickiandiffusion>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Multi-dimensional open/bounded finite-volume Fickian diffusion (`ACCUMULATE`, `per_timestep=True`) |
| `relic.scientific.transport.diffusion` | {ref}`FickianDiffusion (strict 1D closed point) <class-transport-diffusion-fickian-diffusion-strict-runtime-point>` | `TransportMechanismFunctor` | `Level 4` | `[I]` | Closed 1D source-free dissipative port-Hamiltonian runtime point verified per step (`ACCUMULATE`, `per_timestep=True`) |
| `relic.scientific.transport.advection_diffusion` | {ref}`AdvectionDiffusion <class-relic-scientific-transport-advection-diffusion-advectiondiffusion>` | `TransportMechanismFunctor` | `Level 3` | `[I]` | Operator-split conservative advection-diffusion with Peclet and CFL audits (`ACCUMULATE`, `per_timestep=True`) |

General conservation rules follow [Mathematical Foundations](foundations.md),
cochain and port-Hamiltonian definitions follow [DEC, IPHS, and GENERIC](dec_iphs_generic.md),
and splitting accuracy orders follow [Numerical Methods](numerics.md).

## Stable-isotope ratios, exact binary fractionation, and fail-closed ledger status

### Ratio, fractional abundance, and delta notation

For a binary isotope pair with heavy amount $M_h \ge 0$ and strictly positive
light amount $M_l > 0$ (for example $^{13}\mathrm{C}/^{12}\mathrm{C}$,
$^{15}\mathrm{N}/^{14}\mathrm{N}$, or $^{18}\mathrm{O}/^{16}\mathrm{O}$), the
isotope ratio $R$, heavy and light fractional abundances $(x_h, x_l)$, and
per-mil $\delta$ value relative to international reference ratio $R_{\mathrm{std}}$
(VPDB, AIR-$\mathrm{N}_2$, or VSMOW; Fry, 2006; Sharp, 2017) satisfy

$$
R = \frac{M_h}{M_l},
\qquad
x_h = \frac{M_h}{M_h + M_l} = \frac{R}{1 + R},
\qquad
x_l = \frac{M_l}{M_h + M_l} = \frac{1}{1 + R},
\qquad
\delta = \left(\frac{R}{R_{\mathrm{std}}} - 1\right)\times 1000\quad [\text{per mil}].
$$

Because $\delta$ is a nonlinear ratio transform and lacks extensive additivity,
mixing two reservoirs $A$ and $B$ conserves heavy and light
masses separately ($M_h = M_{h,A} + M_{h,B}$, $M_l = M_{l,A} + M_{l,B}$); a
total-mass-weighted average of $\delta$ is only a small-$R$ approximation. Zero
or negative light-isotope denominators raise a domain error without masking
depletion behind arbitrary regularizers.

### Stateless exact binary rational fractionation calculators (`[I]`)

`relic.scientific.isotopes.fractionation` implements stateless calculation
utilities (`IsotopeFractionationCalculator`, `Carbon13FractionationCalculator`,
`Nitrogen15FractionationCalculator`, and `Oxygen18FractionationCalculator`,
all `[I]`). Given product-to-substrate fractionation factor
$\alpha = R_p / R_s > 0$ (related to per-mil enrichment by
$\varepsilon = (\alpha - 1)\times 1000$ and kinetic isotope effect
$\mathrm{KIE} = k_l / k_h$), a total elemental flux $F$ leaving a donor pool of
ratio $R_s$ carries product ratio $R_F = \alpha R_s$ and is partitioned via the
**exact binary rational formulas**:

$$
F_h = F\,\frac{\alpha R_s}{1 + \alpha R_s},
\qquad
F_l = F\,\frac{1}{1 + \alpha R_s},
\qquad
F_h + F_l = F.
$$

Unlike the trace-abundance linearization $F_h \approx \alpha R_s F$, exact
rational partitioning preserves total elemental mass $F_h + F_l = F$ identically
for all $R_s \ge 0$ and $\alpha > 0$. The calculators also provide closed-system
Rayleigh substrate evolution $R(f)/R_0 = f^{\alpha - 1}$ for remaining fraction
$f \in (0, 1]$, Farquhar et al. (1982) $\mathrm{C_3}$ photosynthetic
$^{13}\mathrm{C}$ discrimination

$$
\Delta_{\mathrm{C_3}} = a + (b - a)\,\frac{c_i}{c_a},
$$

Mariotti et al. (1981) nitrogen kinetic fractionation relations, and Craig and
Gordon (1965) evaporative $^{18}\mathrm{O}$ vapor-ratio closures

$$
R_E = \frac{R_L / \alpha_{\mathrm{eq}}(T) - h\,R_A - \varepsilon_k R_L}{1 - h + \varepsilon_k}.
$$

### Fail-closed `IsotopeHeartbeatFunctor` (`[N]`) and retired `MultiIsotopePoolTracker` (`[X]`)

Prognostic multi-pool isotope state coupling across arbitrary Functor graphs is
intentionally fail-closed until full component `UpdateItem` bindings, parent
process ownership, and checkpoint-replay contracts are wired:

- `relic.scientific.isotopes.functor.IsotopeHeartbeatFunctor` (`Level 0`, `[N]`)
  and `relic.scientific.isotopes.wire_isotope_tracker` reject execution before
  any simulator or `StateManager` mutation with `runtime_unbound`.
- The historical out-of-band `MultiIsotopePoolTracker` class has been deleted
  (`[X]`) because mutating private pool dictionaries outside `StateManager`
  violated single-writer provenance and conservation auditing.

## General conservative transport Functors

All three Functors in `relic.scientific.transport` inherit
`TransportMechanismFunctor`, assemble oriented face fluxes on finite-volume
cochain complexes, emit integrated concentration increments via
`UpdateItem(ACCUMULATE, per_timestep=True)`, and account explicitly for boundary
fluxes and any configured non-negativity clipping residual $R_{\mathrm{clip}}$ in
the discrete volume ledger

$$
\sum_i V_i c_i^{n+1} - \sum_i V_i c_i^n
= -\Delta t \sum_{f \in \partial\Omega} A_f q_f + \Delta t \sum_i V_i s_i + R_{\mathrm{clip}}.
$$

### Finite-volume upwind linear advection (`LinearAdvection`, Level 3)

`relic.scientific.transport.advection.LinearAdvection` (`Level 3`) discretizes
$\partial_t c + \nabla \cdot (\mathbf{v} c) = s$ with oriented upwind face fluxes
(LeVeque, 2002):

$$
\frac{\mathrm{d}c_i}{\mathrm{d}t} = -\frac{q_{i+1/2} - q_{i-1/2}}{\Delta x_i} + s_i,
\qquad
q_{i+1/2} =
\begin{cases}
v_{i+1/2}\,c_i, & v_{i+1/2} \ge 0,\\
v_{i+1/2}\,c_{i+1}, & v_{i+1/2} < 0,
\end{cases}
$$

enforcing the ParameterManager-governed advective Courant limit
$\mathrm{CFL} = \max_i (\lvert v_i \rvert \Delta t / \Delta x_i) \le \mathrm{CFL}_{\max}$
and distinguishing periodic, closed no-flux wall, Dirichlet, Neumann, and open
inflow/outflow boundary contracts.

### Fickian diffusion: Level-3 general operator and Level-4 dissipative port-Hamiltonian point

1. **General multi-dimensional open/bounded mode (`FickianDiffusion`, `Level 3`)**:
   `relic.scientific.transport.diffusion.FickianDiffusion` discretizes
   $\partial_t c = \nabla \cdot (\mathbf{K}\nabla c) + s$ in cochain form
   $\dot{\mathbf{c}}_h = -D_h \mathbf{q}_{\mathrm{diff},h} + G_h \mathbf{u}_h$
   with harmonic face diffusivity $\mathbf{q}_{\mathrm{diff},h} = -\star_K d_h \mathbf{c}_h$
   ($\star_K = \star_K^{\mathsf{T}} \succeq 0$) and explicit diffusion stability
   audit $\mu = K \Delta t / \Delta x^2 \le \mu_{\max}$ across 1D, 2D, and 3D
   supports (Crank, 1975).

2. **Closed 1D source-free dissipative port-Hamiltonian runtime point (`Level 4`)**:
   Under the secondary constructor-conditioned certificate
   `constructor_selection = {strict_runtime_point: True, spatial_dims: 1}`
   ({ref}`FickianDiffusion strict runtime point <class-transport-diffusion-fickian-diffusion-strict-runtime-point>`,
   `Level 4`), `FickianDiffusion` restricts execution to a closed (periodic or
   zero-flux wall) 1D uniform grid of spacing $\Delta x$ with zero boundary
   ports ($G_h = 0$) and fail-closed rejection of any nonzero `source_term`.
   Binding the quadratic $L^2$ storage functional and autograd effort covector

   $$H_h(\mathbf{c}_h) = \frac{1}{2}\,\mathbf{c}_h^{\mathsf{T}}\mathbf{c}_h, \qquad \mathbf{e}_H = \frac{\partial H_h}{\partial \mathbf{c}_h} = \mathbf{c}_h,$$

   the production finite-volume diffusion drift coincides identically with the
   canonical dissipative port-Hamiltonian structure ($J_h = 0$):

   $$\frac{\mathrm{d}\mathbf{c}_h}{\mathrm{d}t} = (J_h - R_h)\,\mathbf{e}_H = -R_h\,\mathbf{c}_h, \qquad R_h = \frac{1}{\Delta x^2}\,d_h^{\mathsf{T}}\operatorname{diag}(K_f)\,d_h = R_h^{\mathsf{T}} \succeq 0,$$

   satisfying monotone storage dissipation

   $$\frac{\mathrm{d}H_h}{\mathrm{d}t} = -\mathbf{e}_H^{\mathsf{T}} R_h\,\mathbf{e}_H = -\sum_f K_f\left(\frac{\Delta_f c_h}{\Delta x}\right)^2 \le 0$$

   and exact discrete mass conservation $\mathbf{1}^{\mathsf{T}} R_h \mathbf{c}_h = 0$.
   On every emitted step in this configuration, `forward()` executes
   `relic.computation.operators.iphs.IPHSStructureSpec.validate` at
   `required_level = 4` and records the live audit on `last_strict_iphs_audit`
   (van der Schaft and Jeltsema, 2014). Because no thermodynamic entropy
   functional $S_h$ or reversible-irreversible GENERIC cross-degeneracy is
   claimed for isothermal tracer diffusion, the certificate terminates at
   `Level 4` (with `Level 5` reserved for GENERIC entropy-coupled formulations).

### Combined advection-diffusion and operator-splitting accuracy (`AdvectionDiffusion`, Level 3)

`relic.scientific.transport.advection_diffusion.AdvectionDiffusion` (`Level 3`)
combines advective and diffusive cochain sub-flows
($\mathbf{q} = \mathbf{v}c - \mathbf{K}\nabla c$, cell Peclet diagnostic
$\mathrm{Pe} = \lvert \mathbf{v} \rvert \Delta x / K$) via configurable
sequential (`"sequential"` Lie-Trotter, $\Phi_D^{\Delta t}\circ \Phi_A^{\Delta t}$)
or symmetric (`"symmetric"` Strang,
$\Phi_D^{\Delta t/2}\circ \Phi_A^{\Delta t}\circ \Phi_D^{\Delta t/2}$) operator
splitting (Hundsdorfer and Verwer, 2003):

- Symmetric Strang composition eliminates the leading first-order Baker-Campbell-Hausdorff
  commutator $[A, D]\Delta t^2 / 2$, achieving $O(\Delta t^2)$ splitting
  commutator error when paired with second-order sub-flow integrators such as
  `"implicit_midpoint"`.
- Under the default first-order explicit `"euler"` sub-integrators, local
  sub-step truncation error is $O(\Delta t^2)$ per step, so the composite
  `AdvectionDiffusion` trajectory remains globally first-order $O(\Delta t)$ even
  when `"symmetric"` splitting is selected.

When transporting stable isotopes across spatial grids, callers transport the
conserved heavy and light concentrations $(c_h, c_l)$ and reconstruct $\delta$ diagnostically after transport, avoiding direct advection or
diffusion of non-conservative $\delta$ values.

## References

- Bird, R. B., Stewart, W. E., and Lightfoot, E. N. (2007). *Transport Phenomena* (2nd ed.). Wiley.
- Craig, H. and Gordon, L. I. (1965). Deuterium and oxygen-18 variations in the ocean and the marine atmosphere. In *Stable Isotopes in Oceanographic Studies and Paleotemperatures*, CNR Laboratorio di Geologia Nucleare, Pisa, 9-130.
- Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
- Farquhar, G. D., O'Leary, M. H., and Berry, J. A. (1982). On the relationship between carbon isotope discrimination and the intercellular carbon dioxide concentration in leaves. *Australian Journal of Plant Physiology*, 9, 121-137.
- Fry, B. (2006). *Stable Isotope Ecology*. Springer.
- Hundsdorfer, W. and Verwer, J. G. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations*. Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- Mariotti, A. et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation: some principles; illustration for the denitrification and nitrification processes. *Plant and Soil*, 62, 413-430.
- Sharp, Z. (2017). *Principles of Stable Isotope Geochemistry* (2nd ed.). Prentice Hall / Open Educational Edition.
- van der Schaft, A. and Jeltsema, D. (2014). Port-Hamiltonian systems theory: An introductory overview. *Foundations and Trends in Systems and Control*, 1(2-3), 173-378.
