# DEC, IPHS, GENERIC, Ports, and Reaction Structure

## Structural governance lifecycle

A constitutive rate formula specifies process speed under local conditions;
structural governance additionally specifies discrete spatial support, conserved
elemental ledgers, conjugate thermodynamic efforts, port power exchange, and
overflow residual routing (*Methods of Ecological Analysis*, Chapters 8, 9, 12,
and 13). In RELIC, Discrete Exterior Calculus (DEC), Irreversible
Port-Hamiltonian System (IPHS), and General Equation for Non-Equilibrium
Reversible-Irreversible Coupling (GENERIC) operators augment the single
runtime lifecycle without introducing a second simulator, state manager, or
registry:

$$
\text{Scientific ABC}
\;\Longrightarrow\;
\text{Canonical Bridge}
\;\Longrightarrow\;
\texttt{FunctorBase}
\;\Longrightarrow\;
\texttt{ParameterManager} + \texttt{StateManager}
\;\Longrightarrow\;
\text{DEC / IPHS / GraphIR Audit}.
$$

Every Graph Intermediate Representation (`GraphIR`, `relic.orchestration.graph_ir`)
compilation pass (`compile_graph` in `relic.orchestration.compiler`) emits a
normalized 13-key `report.audit` summary (`syntax_ok`, `semantic_ok`,
`assertion_ok`, `assertion_audit`, `conservation_audit`, `port_matching_audit`,
`overflow_routing_audit`, `equivalence_certificate_audit`,
`reaction_network_audit`, `integration_plan_audit`, `global_compile_audit`,
`validation_error_count`, `validation_warning_count`). Software wiring nullspace
vectors (`topology_signature.conservation_vectors` in
`relic.orchestration.topology_algebra`) summarize topological incidence
$\ker(\mathbf{B}^{\mathsf{T}})$ and never substitute for physical mass or energy
conservation committed through `StateManager`.

## Discrete exterior calculus (`relic.computation.operators.dec`)

### Ten-field typed cochain spaces (`CochainSpaceSpec`)

On an oriented cellular complex $K$, a discrete $k$-form is a $k$-cochain
$\alpha^k \in C^k(K)$ ($k = 0$ on vertices, $1$ on oriented edges, $2$ on faces,
$3$ on volumes). `CochainSpaceSpec` (`relic.computation.operators.dec.forms`)
declares each finite-dimensional cochain space via ten frozen fields:

| Field | Python type | Mathematical and structural role |
| --- | --- | --- |
| `complex_id` | `str` | Identifier of the cellular complex $K$ |
| `degree` | `FormDegree` (`0`, `1`, `2`, `3`) | Cochain degree $k$ |
| `primality` | `Primality` (`"primal"`, `"dual"`) | Primal complex $K$ or dual complex $K^\ast$ placement |
| `n_dofs` | `int` | Positive finite number of degrees of freedom ($n > 0$) |
| `units` | `str` | Canonical physical unit of cochain pairings |
| `extensivity` | `Extensivity` (`"intensive"`, `"extensive"`, `"density"`, `"integral"`) | DEC cochain pairing semantics (distinct from `SpatialQuantityBasis`) |
| `physical_domain` | `str` | Governed physical or elemental domain label |
| `semantic_name` | `str` | Canonical variable or field semantic identifier |
| `orientation_convention` | `str` | Inner or outer orientation convention on $K$ or $K^\ast$ |
| `boundary_role` | `BoundaryRole` (`"interior"`, `"boundary"`, `"port"`, `"mixed"`) | Topological boundary participation role |

### Coboundary exactness and Stokes pairing

The discrete exterior derivative $D_k: C^k(K) \rightarrow C^{k+1}(K)$ is the
transpose of the chain boundary incidence matrix, $D_k = \partial_{k+1}^{\mathsf{T}}$,
satisfying the exact topological nilpotency identity (*Methods of Ecological
Analysis*, Chapter 8):

$$
D_{k+1} D_k = 0,
\qquad
\langle D_k \alpha,\; c \rangle = \langle \alpha,\; \partial_{k+1} c \rangle.
$$

`cochain_complex_check` (`relic.validation.framework.rules.dec_complex`) verifies
$D_{k+1} D_k = 0$ against `validation.dec_complex.cochain_residual_tolerance`
(strict validation also requires integer incidence entries, $D_0 \mathbf{1} = 0$,
and boundary subcomplex closure). `typed_stokes_audit`
(`relic.computation.operators.dec.complex`) evaluates the discrete Stokes
pairing on supplied cochain/chain vectors as a finite point check.

### Linear chain maps (`LinearChainMap`)

`LinearChainMap` (`relic.computation.operators.dec.chain_map`) represents a
unit-aware linear operator $M \in \mathbb{R}^{n_t \times n_s}$ between declared
`source` and `target` `CochainSpaceSpec` spaces with positive unit-conversion
factor $u > 0$. Construction (`__post_init__`) enforces four fail-closed checks:

1. **Two-dimensional matrix rank**: `matrix.ndim == 2`;
2. **Exact degree-of-freedom shape match**: `tuple(matrix.shape) == (target.n_dofs, source.n_dofs)`;
3. **Finite entries**: `torch.isfinite(matrix).all()`; and
4. **Strictly positive unit scale**: `unit_factor > 0.0`.

Application, composition, unit-adjusted ledger preservation (`preserves_ledger`
against `validation.iphs.chain_map_ledger_tolerance`), perturbation error
bounding (`error_bound`), and operator distance (`distance`, `close_within`)
satisfy:

$$
y = u M x,
\qquad
\max_{i,j}\lvert \left(C_t (u M) - C_s\right)_{ij} \rvert \le \varepsilon_{\mathrm{cm}},
\qquad
\lVert u M \delta x \rVert_2 \le \lVert u M \rVert_2\,\lVert \delta x \rVert_2,
$$

$$
d(M_1, M_2) = \lVert u_1 M_1 - u_2 M_2 \rVert_2.
$$

Exact equality `equals` ($d = 0$ with identical spaces and `unit_factor`) is an
equivalence relation; `close_within` ($d(M_1, M_2) \le \varepsilon$) is
reflexive and symmetric without transitivity. Coboundary commutation
$P_{k+1} D_k^{(a)} = D_k^{(b)} P_k$ is audited separately by `chain_map_check`
(`relic.validation.framework.rules.chain_map`).

### Hodge star and constitutive metrics (`HodgeStarSpec`)

`HodgeStarSpec` (`relic.computation.operators.dec.hodge`) maps primal $k$-cochains
to dual $(n - k)$-cochains, $\star_k: C^k(K) \rightarrow C^{n-k}(K^\ast)$,
inducing the discrete inner product
$\langle \alpha, \beta \rangle_k = \alpha^{\mathsf{T}} H_k \beta$ via matrix
$H_k$ backed by `ParameterManager` `parameter_keys` and `metric_keys`. Calling
`HodgeStarSpec.validate()` and `hodge_psd_check`
(`relic.validation.framework.rules.hodge`) verifies compatible domain/codomain
spaces, finite entries, and role-specific definiteness:

$$
H_k = H_k^{\mathsf{T}},
\qquad
z^{\mathsf{T}} H_k z > 0 \quad \forall z \neq 0 \;\; (\text{metric SPD}),
\qquad
z^{\mathsf{T}} H_k z \ge 0 \;\text{with}\; H_k K = 0,\; \dim K = \operatorname{nullity}(H_k) \;\; (\text{material PSD}).
$$

A singular material operator must supply its complete linearly independent
`kernel_basis` $K$ and never substitutes for an invertible metric Hodge star.
With storage cochain $x$ and flux cochain $f$, semi-discrete conservative
transport $\frac{\mathrm{d}x}{\mathrm{d}t} = -D^{\mathsf{T}} f + s + b$ lowers
into governed `StateManager` updates via `typed_cochain_delta_to_update_item`
(`relic.computation.operators.dec.adapters`).

## Irreversible port-Hamiltonian and GENERIC lowering

`IPHSStructureSpec`, `PortSpecIPHS`, `assemble_iphs_structure`,
`advance_reaction_network`, and `poisson_jacobi_residual`
(`relic.computation.operators.iphs`) separate dissipative port-Hamiltonian
(`dissipative_ph`) realizations from GENERIC / metriplectic (`generic`)
realizations without mixing energy resistance $R_h$ and entropy mobility $M_h$
(*Methods of Ecological Analysis*, Chapters 9, 12, and 13).

### Dissipative port-Hamiltonian (`dissipative_ph`) realization

For storage functional $H(x)$ with conjugate effort $e_H = \nabla_x H(x)$, port
input $u_h$, port matrix $G_h$, and conjugate output $y_h = G_h^{\mathsf{T}} e_H$,
the canonical `dissipative_ph` drift (`M_h = 0`) evaluated by
`IPHSStructureSpec.canonical_drift` is:

$$
\frac{\mathrm{d}x}{\mathrm{d}t} = (J_h - R_h) e_H + G_h u_h + N_h r_h,
\qquad
J_h^{\mathsf{T}} = -J_h,
\qquad
R_h = R_h^{\mathsf{T}} \succeq 0.
$$

Because $e_H^{\mathsf{T}} J_h e_H = 0$, closed reaction-free trajectories satisfy
$\frac{\mathrm{d}H}{\mathrm{d}t} = -e_H^{\mathsf{T}} R_h e_H \le 0$, and open
trajectories satisfy $\frac{\mathrm{d}H}{\mathrm{d}t} = -e_H^{\mathsf{T}} R_h e_H + y_h^{\mathsf{T}} u_h \le y_h^{\mathsf{T}} u_h$.

### GENERIC / metriplectic (`generic`) realization and Jacobi bases

For total energy $E(x)$ ($e_E = \nabla_x E$) and entropy $S(x)$
($e_S = \nabla_x S$), the canonical `generic` metriplectic drift (`R_h = 0`)
evaluated by `IPHSStructureSpec.canonical_drift` and audited via
`IPHSStructureSpec.validate` is:

$$
\frac{\mathrm{d}x}{\mathrm{d}t} = J_h(x)\,e_E + M_h(x)\,e_S + G_h u_h + N_h r_h,
$$

subject to skew-symmetry, positive semidefiniteness, and mutual degeneracy:

$$
J_h^{\mathsf{T}} = -J_h,
\qquad
M_h = M_h^{\mathsf{T}} \succeq 0,
\qquad
J_h e_S = 0,
\qquad
M_h e_E = 0,
\qquad
e_S^{\mathsf{T}} M_h e_S \ge 0.
$$

For an isolated reaction-free system ($u_h = 0$, $r_h = 0$), energy is conserved
($\frac{\mathrm{d}E}{\mathrm{d}t} = e_E^{\mathsf{T}} J_h e_E + e_E^{\mathsf{T}} M_h e_S = 0$)
and entropy increases monotonically
($\frac{\mathrm{d}S}{\mathrm{d}t} = e_S^{\mathsf{T}} M_h e_S \ge 0$). Note that
RELIC lowers GENERIC/metriplectic tendencies through
`IPHSStructureSpec.canonical_drift` and `advance_reaction_network` (`[I]`); no
separate standalone metriplectic time-integrator class is exposed (`[N]`).

Skew-symmetry of $J_h$ (carrying the Poisson operator $L_h$) does not imply the
Jacobi identity. `IPHSStructureSpec` admits three executable `poisson_operator`
bases audited against `validation.iphs.jacobi_tolerance`:

1. `"zero_field"`: $L(x) = 0$ identically (`canonical_zero_field`); all live
   certified production processes use zero reversible Poisson fields.
2. `"constant_field"`: constant skew matrix $L$ with $\partial_l L_{ij} = 0$,
   so the Jacobi residual vanishes identically (`constant_field_exact`).
3. `"state_field"`: importable callable $L(x)$ audited by autograd via
   `poisson_jacobi_residual` over declared `poisson_domain` state samples and
   at bound Level 5 runtime points (`sampled_domain_jacobi_residual`):

$$
R_{ijk}(x)
= \sum_{l} \left(
L_{il}\frac{\partial L_{jk}}{\partial x_l}
+ L_{jl}\frac{\partial L_{ki}}{\partial x_l}
+ L_{kl}\frac{\partial L_{ij}}{\partial x_l}
\right).
$$

Static `matrix_snapshot` tensors carry no derivative information and fail closed
for state-dependent Jacobi claims (`[N]`).

### Reaction networks, open-system routing, and port interconnection

`advance_reaction_network` (`relic.computation.operators.iphs`) and
`reaction_network_check` (`relic.validation.framework.rules.reaction_network`)
govern stoichiometric kinetics $\dot{x}_{\mathrm{rxn}} = N_h r_h$ ($N_h \in \mathbb{R}^{n_s \times n_r}$, $r_h \in \mathbb{R}^{n_r}$):

- **Independent vs absorbed reaction representation**: Within one
  `IPHSStructureSpec`, reactions either appear explicitly as $+N_h r_h$
  (requiring $e_H^{\mathsf{T}} N_h r_h = 0$ for total energy or $\le 0$ for free
  energy) or are absorbed into the dissipative operator ($N_h r_h = -R_h e_H$ in
  `dissipative_ph` or $N_h r_h = M_h e_S$ in `generic`, omitting $+N_h r_h$ so
  rates are never double-counted).
- **Closed stoichiometry and physical affinity power**: Internal conservation
  requires $C_h N_h = 0$ (and optional charge neutrality $z_h^{\mathsf{T}} N_h = 0$).
  With chemical potentials $\mu$, reaction affinity
  $\mathcal{A} = -N_h^{\mathsf{T}} \mu$ satisfies dissipative power non-negativity
  $P_{\mathrm{rxn}} = \mathcal{A}^{\mathsf{T}} r_h \ge 0$; division by positive
  temperature $\sigma_{\mathrm{rxn}} = \mathcal{A}^{\mathsf{T}} r_h / T$ yields
  entropy production only when an executable $T > 0$ and thermodynamic closure
  are bound.
- **Open-system exchange and overflow routing**: Open systems keep internal
  stoichiometry closed ($C_h N_h = 0$) and route external exchange separately as
  $\frac{\mathrm{d}x}{\mathrm{d}t} = N_h r_h + q_{\mathrm{open}}$ with
  $C_h q_{\mathrm{open}} = \dot{\ell}_{\mathrm{open}}$ typed via `OpenSystemRoute`
  (`"port"`, `"boundary_export"`, `"managed_export"`, `"diagnostic_residual"`,
  or `"observation"`). Any bounds-clipping residual
  $r_{\mathrm{clip}} = (x + \Delta x) - \operatorname{clip}(x + \Delta x, x_{\min}, x_{\max})$
  must declare a valid route in `overflow_routing_check` (`"storage"`,
  `"boundary_outflow"`, `"managed_export"`, `"diagnostic_residual"`, or
  `"rejected_step"`).
- **Port matching vs runtime port power**: `port_matching_check`
  (`relic.validation.framework.rules.port_matching`) verifies static port
  declarations (effort/flow units $[P] = [e][f]$, dimensions, orientations, and
  signed incidence balance $\sum_i s_i w_i = 0$). Skew interconnection
  $C = -C^{\mathsf{T}}$ induces $J_{\mathrm{port}} = G_h C G_h^{\mathsf{T}}$,
  and passive feedback $u_{\mathrm{int}} = -D y_h$ ($D \succeq 0$) induces
  $R_{\mathrm{port}} = G_h D G_h^{\mathsf{T}}$ in `dissipative_ph` only (never
  added directly to GENERIC $M_h$). Actual power balance
  $P_a + P_b \le 0$ is a runtime evaluation (`port_power_units_check` /
  `feedback_power_ledger`), distinct from static wiring checks.

## Equivalence certificates and live repository inventory

`IPHSEquivalenceCertificate` (`relic.computation.operators.iphs_certificate`,
audited by `equivalence_certificate_check` in
`relic.validation.framework.rules.equivalence_certificate`) is the sole owner of
each process's structural claim level (`Level 0` through `Level 5`).

Across the live repository, `90` unique concrete scientific `Functor` classes
(`37` `BiologicalProcessFunctor`, `27` `TransportMechanismFunctor`, `14`
`PhysicalFieldFunctor`, and `12` `ScaleCouplingFunctor`; `0` uncertified) carry
`90` primary class certificates plus `2` secondary constructor-scoped
(`constructor_selection`) runtime-point certificates, totaling `92` YAML
certificates under `relic/parameters/database/relic/`:

| Level | Classification | Primary classes (`90`) | Secondary `constructor_selection` (`2`) | Total YAML certificates (`92`) | Permitted scientific and runtime claim |
| --- | --- | --- | --- | --- | --- |
| `Level 0` | Unavailable / empirical fail-closed boundary | `4` | `0` | `4` | Explicit `[N]` fail-closed boundary (`AtmosphereCoupling`, `EmissionInventory`, `AnthroposphereInterfaceMapping`, `IsotopeHeartbeatFunctor`); zero executable IPHS claim |
| `Level 1` | IPHS-compatible interface | `15` | `0` | `15` | Typed read-only diagnostic, validator, or interface mapper participating in governed graph composition |
| `Level 2` | IPHS-wrapped process | `39` | `0` | `39` | Governed process lowered through canonical reaction, algebraic-DAE, event, or stochastic operator wrappers |
| `Level 3` | IPHS-structured process | `32` | `0` | `32` | Declared cochain, incidence, stoichiometric, port, and conservation-ledger structure with audit tolerances |
| `Level 4` | Strict dissipative port-Hamiltonian runtime point | `0` | `1` | `1` | `FickianDiffusion` with `constructor_selection: {strict_runtime_point: true, spatial_dims: 1}` on closed source-free 1D support with quadratic $L^2$ storage $H_h = \frac{1}{2} c_h^{\mathsf{T}} c_h$, autograd effort, $R_h \succeq 0$, and governed execution witness |
| `Level 5` | Strict GENERIC runtime point | `0` | `1` | `1` | `ChemistryFunctor` with `constructor_selection: {physical_boundary: closed_constant_volume}` on closed adiabatic rigid-box $\mathrm{NO} + \mathrm{O}_3 \rightarrow \mathrm{NO}_2 + \mathrm{O}_2$ kinetics with bound $(E, S)$ autograd efforts, $J_h e_S = M_h e_E = 0$, $e_S^{\mathsf{T}} M_h e_S \ge 0$, Jacobi check, and governed execution witness |

Both strict certificates (`Level 4` and `Level 5`) audit the continuous-time
canonical drift at the pre-step runtime state and require a matching
`governed_execution` witness from a committed `FunctorBase.__call__` step
(without a witness, `IPHSEquivalenceCertificate` reports
`process_bound_runtime_unavailable`). The primary certificates of
`FickianDiffusion` and `ChemistryFunctor` remain `Level 3`, and Level 4/5
runtime points with open external port columns remain unimplemented (`[N]`).

## References

- Arnold, D. N., Falk, R. S., and Winther, R. (2006). Finite element exterior calculus, homological techniques, and applications. *Acta Numerica* 15, 1--155.
- Desbrun, M., Kanso, E., and Tong, Y. (2008). Discrete differential forms for computational modeling. In *Discrete Differential Geometry*, 287--324. Birkhauser.
- Feinberg, M. (2019). *Foundations of Chemical Reaction Network Theory*. Springer.
- Grmela, M. and Ottinger, H. C. (1997). Dynamics and thermodynamics of complex fluids. I. Development of a general formalism. *Physical Review E* 56, 6620--6632.
- Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration*, 2nd ed. Springer.
- Hatcher, A. (2002). *Algebraic Topology*. Cambridge University Press.
- Hirani, A. N. (2003). *Discrete Exterior Calculus*. Ph.D. thesis, California Institute of Technology.
- Ottinger, H. C. (2005). *Beyond Equilibrium Thermodynamics*. Wiley.
- Rao, R. and Esposito, M. (2016). Nonequilibrium thermodynamics of chemical reaction networks: Wisdom from stochastic thermodynamics. *Physical Review X* 6, 041064.
- van der Schaft, A. and Jeltsema, D. (2014). Port-Hamiltonian systems theory: An introductory overview. *Foundations and Trends in Systems and Control* 1, 173--378.
