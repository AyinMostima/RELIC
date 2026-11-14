# Structural Ecology Foundation: GraphIR Governance, DEC/IPHS Audits, and Certificates

## 1. Process Definition and Scope

Structural Ecology in RELIC is implemented as a compile-time and runtime-point governance layer spanning `relic/orchestration/` (`GraphIR`, `validator.py`, `compiler.py`, `topology_algebra.py`), `relic/computation/operators/` (`iphs.py`, `iphs_certificate.py`, `dec/`), `relic/computation/solvers/geometric.py` (`IntegrationPlan`), and `relic/validation/framework/rules/`. It introduces no secondary runtime simulator, no duplicate state manager, and no symbolic MSFOL theorem prover: runtime stepping remains owned by `RELICSimulator` (`relic/core/simulator/`) and `GraphExecutor` (`relic/orchestration/executor.py`). Theoretical foundations for structural assertions, discrete exterior calculus (DEC), irreversible port-Hamiltonian / GENERIC realizations, and Level 0–5 equivalence certificates appear in *Methods of Ecological Analysis*, Chapters 1, 2, 3, 27, and 29 (following notation in [00_unified_theoretical_framework.md](00_unified_theoretical_framework.md)).

Every orchestration compile/validation pass emits a single normalized `report.audit` dictionary (`build_audit_summary` in `relic/orchestration/compiler.py`) containing `syntax_ok`, `semantic_ok`, `assertion_ok`, `assertion_audit`, `conservation_audit`, `port_matching_audit`, `overflow_routing_audit`, `equivalence_certificate_audit`, `reaction_network_audit`, `integration_plan_audit`, `global_compile_audit`, `validation_error_count`, and `validation_warning_count`.

## 2. Governing Equations

### 2.1 Software Topology Incidence vs. Physical Conservation Ledgers

An optional top-level `GraphIR.assertion` declares a bounded structural triple $\langle \phi, \mathrm{Op}, \mathrm{Scope} \rangle$ (`phi` predicate string, `op` node/class/SSOT/protocol target, `scope` non-empty string or dictionary, plus `context` and `metadata`). For a declared `GraphIR` with $n$ software nodes and $m$ oriented software links, `topology_algebra.py` forms the dimensionless incidence matrix $\mathbf{B} \in \mathbb{R}^{n \times m}$ and its left nullspace (Strang, 1988; Godsil and Royle, 2001):

$$
\mathbf{c} \in \ker\!\left(\mathbf{B}^{\mathsf{T}}\right), \qquad \dim\ker\!\left(\mathbf{B}^{\mathsf{T}}\right) = n - \operatorname{rank}(\mathbf{B}).
$$

Software incidence vectors (`topology_signature.conservation_vectors` reported in `report.audit.conservation_audit`) summarize wiring topology only; unit casts, control links, and diagnostic edges carry no mass or energy, so $\ker(\mathbf{B}^{\mathsf{T}})$ never proves physical conservation. Physical conservation requires actual state changes on a dimensional linear ledger $C$ committed through `StateManager`:

$$
C\left(x_{\mathrm{after}} - x_{\mathrm{before}}\right) - \Delta L_{\mathrm{external}} = 0
\qquad \text{or} \qquad
C_{\mathrm{after}} x_{\mathrm{after}} - C_{\mathrm{before}} x_{\mathrm{before}} - \Delta L_{\mathrm{external}} = 0.
$$

### 2.2 Discrete Exterior Calculus (DEC), Port Matching, and Residual Routing

1. **Cochain complex, Hodge star, and coboundary chain maps** (Hirani, 2003; Arnold, Falk, and Winther, 2006):
   - `cochain_complex_check` (`relic/validation/framework/rules/dec_complex.py`) verifies exactness $D_{k+1} D_k = 0$ on integer coboundary matrices (strict mode also verifies $D_0 \mathbf{1} = 0$ and boundary subcomplex closure), while `typed_stokes_audit` (`relic/computation/operators/dec/complex.py`) checks discrete Stokes pairing $\langle D_k a, c \rangle = \langle a, d_{k+1} c \rangle$.
   - `hodge_psd_check` (`relic/validation/framework/rules/hodge.py`) verifies $H = H^{\mathsf{T}} \succeq 0$ (strict metric Hodge: $H \succ 0$; material PSD Hodge: declared kernel basis $K$ satisfying $H K \approx 0$ with rank matching numerical nullity).
   - `chain_map_check` (`relic/validation/framework/rules/chain_map.py`) audits coboundary commutation $R_{k+1} D_s = D_t R_k$ (distinct from `LinearChainMap.preserves_ledger`, [bounded_mathematical_contracts.md](bounded_mathematical_contracts.md)).

2. **Port declaration matching vs. runtime port power** (van der Schaft and Maschke, 1995; Ramirez et al., 2022): `port_matching_check` (`relic/validation/framework/rules/port_matching.py`) verifies unit, space, and signed incidence balance across each internal matched port group ($s_i = +1$ for `into_system`, $s_i = -1$ for `out_of_system`):

   $$\sum_i s_i w_i = 0,$$

   which is a declaration-level prerequisite and does not evaluate per-step power closure $\sum_i y_i^{\mathsf{T}} u_i = 0$ (`relic/validation/framework/rules/port_power.py`).

3. **Explicit overflow and open-system routing** (Hundsdorfer and Verwer, 2003; Finlayson, 2012): Every non-zero clipping, overflow, or unconsumed-flux residual $r_k$ audited by `overflow_routing_check` (`relic/validation/framework/rules/overflow_routing.py`) must declare a route in `_ALLOWED_ROUTES`:

   $$\rho_{\mathrm{rule}}(r_k) \in \{\texttt{"storage"},\; \texttt{"boundary\_outflow"},\; \texttt{"managed\_export"},\; \texttt{"diagnostic\_residual"},\; \texttt{"rejected\_step"}\},$$

   whereas operator-level `OpenSystemRoute` (`relic/computation/operators/iphs.py`) types open IPHS ports using `Literal["boundary_export", "managed_export", "diagnostic_residual", "port", "observation"]`.

### 2.3 Canonical dPH / GENERIC Operators, Reaction Networks, and Level 0–5 Certificates

`IPHSStructureSpec` and `assemble_iphs_structure` (`relic/computation/operators/iphs.py`) define two canonical irreversible realizations (van der Schaft and Maschke, 1995; Grmela and Ottinger, 1997):

$$
f_{\mathrm{dPH}}(x, \theta, t) = (J_h - R_h) e_H + G_h u_h + N_h r_h, \qquad
f_{\mathrm{GENERIC}}(x, \theta, t) = J_h e_E + M_h e_S + G_h u_h + N_h r_h,
$$

with skew-symmetric $J_h^{\mathsf{T}} = -J_h$, symmetric positive semidefinite resistance $R_h = R_h^{\mathsf{T}} \succeq 0$ (admitting dissipative port feedback $G_h D G_h^{\mathsf{T}}$ in dPH only), symmetric positive semidefinite mobility $M_h = M_h^{\mathsf{T}} \succeq 0$, and GENERIC double degeneracy $J_h e_S = 0$, $M_h e_E = 0$. In absorbed reaction representations, $N_h r_h = -R_h e_H$ (dPH) or $N_h r_h = M_h e_S$ (GENERIC) replaces the explicit $+N_h r_h$ term without double counting.

For stoichiometric networks (`reaction_network_check` in `relic/validation/framework/rules/reaction_network.py`; Rao and Esposito, 2016; Feinberg, 2019), internal conservation, optional charge neutrality, thermodynamic reaction power, and open-exchange closure satisfy:

$$
C_h N_h = 0, \qquad z_h^{\mathsf{T}} N_h = 0, \qquad
P_{\mathrm{reaction}} = A_{\mathrm{phys}}^{\mathsf{T}} r \ge 0 \quad \left(A_{\mathrm{phys}} = -N_h^{\mathsf{T}}\mu\right), \qquad
C_h q_{\mathrm{open}} = \dot{\ell}_{\mathrm{open}}.
$$

Only under an explicitly bound positive isothermal temperature $T > 0$ does $\sigma_{\mathrm{reaction}} = P_{\mathrm{reaction}} / T$ represent entropy production rate.

`IPHSEquivalenceCertificate` (`relic/computation/operators/iphs_certificate.py`, audited by `equivalence_certificate_check` in `relic/validation/framework/rules/equivalence_certificate.py`) governs behavioral equivalence $f_{\mathrm{legacy}}(x, \theta, t) = f_{\mathrm{canonical}}(x, \theta, t)$ across six strict levels:
- **Level 0–3**: Metadata/port signature (`L0`), dimensional/domain bounds (`L1`), discrete conservation/stoichiometric ledger closure (`L2`), and operator-split or subexpression numerical equivalence (`L3`).
- **Level 4 (dPH runtime point)** and **Level 5 (GENERIC runtime point)**: Require autograd effort evaluation ($e_H = \partial H / \partial x$ or $e_E, e_S$), passivity or double-degeneracy + executable Jacobi identity verification (`canonical_zero_field`, `constant_field_exact`, or `sampled_domain_jacobi_residual` via `poisson_jacobi_residual`), and a governed runtime-execution witness (`FunctorBase.__call__` commit). Two constructor-scoped runtime points (`constructor_selection`) carry Level 4/5 certificates: closed 1D `FickianDiffusion` (`Level 4`) and closed constant-volume $\mathrm{NO}+\mathrm{O_3}$ chemistry (`Level 5`), while their general open-configuration process certificates remain `Level 3`.

### 2.4 Splitting Integration Plans vs. Runtime Schedule Witnesses

`IntegrationPlan` (`relic/computation/solvers/geometric.py`) records declared Strang-compatible stage decompositions $\Phi(\Delta t) = \Phi_A(\Delta t/2)\circ \Phi_B(\Delta t)\circ \Phi_A(\Delta t/2)$ (Strang, 1968; Hairer, Lubich, and Wanner, 2006). Calling `IntegrationPlan.validate()` audits stage-DAG acyclicity (`lagged` edges excluded) and metadata completeness only, returning `execution_semantics="metadata_only"`, `verified=False`, and `runtime_ready=False`. Runtime schedule verification occurs separately when `RELICSimulator.get_composition_plan()` exports committed simulator stages and `GraphExecutor` checks `integration_plan_witness` against `StateManager` commit provenance ([spatiotemporal_scale_topology.md](spatiotemporal_scale_topology.md)).

## 3. Variable Table

| Structure / Audit Key | Meaning | Type / Units | Owner | Role |
| --- | --- | --- | --- | --- |
| `GraphIR.assertion` | Optional $\langle \phi, \mathrm{Op}, \mathrm{Scope} \rangle$ structural claim | JSON / `@dataclass` | `orchestration/graph_ir.py` | Compile-time assertion input |
| `topology_signature.conservation_vectors` | Software incidence left-nullspace basis $\ker(\mathbf{B}^{\mathsf{T}})$ | `1` (topological only) | `orchestration/topology_algebra.py` | Software wiring summary |
| `IPHSStructureSpec`, `PortSpecIPHS` | Canonical dPH / GENERIC matrices $(J_h, R_h, M_h, G_h, N_h)$ and ports | Declared state/effort units | `computation/operators/iphs.py` | Canonical operator bundle |
| `IPHSEquivalenceCertificate` | Level 0–5 equivalence metadata, tolerances, and `constructor_selection` | YAML-backed `@dataclass` | `computation/operators/iphs_certificate.py` | Process equivalence certificate |
| `IntegrationPlan` | Declared stage splitting graph (`execution_semantics="metadata_only"`) | `@dataclass` ($\Delta t$ in $\mathrm{s}$) | `computation/solvers/geometric.py` | Compile-time stage declaration |
| `report.audit.*` | 13-field unified compile/validation audit dictionary | `dict[str, Any]` | `orchestration/compiler.py` | Canonical CLI / API audit payload |

## 4. Parameter Table

| Key | Units | Default (Bounds) | Source YAML | Consumer |
| --- | --- | --- | --- | --- |
| `validation.iphs.runtime_point_tolerance` | `dimensionless` | `1.0e-10` ($[10^{-16}, 10^{-6}]$) | `validation/iphs.yaml` | `equivalence_certificate_check` & dPH/GENERIC point audits |
| `validation.iphs.jacobi_tolerance` | `dimensionless` | `1.0e-8` ($[10^{-16}, 10^{-6}]$) | `validation/iphs.yaml` | `poisson_jacobi_residual` (Level 5 Jacobi check) |
| `validation.iphs.skew_tolerance`, `psd_tolerance`, `degeneracy_tolerance` | `dimensionless` | `1.0e-10` ($[0.0, 10^{-6}]$) | `validation/iphs.yaml` | $J_h^{\mathsf{T}} = -J_h$, $R_h/M_h \succeq 0$, $J_h e_S = M_h e_E = 0$ |
| `validation.iphs.reaction_conservation_tolerance`, `reaction_thermodynamic_rtol`, `reaction_power_scale_floor` | `dimensionless` | `1.0e-10`, `1.0e-10`, `1.0` | `validation/iphs.yaml` | `reaction_network_check` ($C_h N_h = 0$, $A_{\mathrm{phys}}^{\mathsf{T}} r \ge 0$) |
| `validation.iphs.hodge_psd_rtol`, `hodge_symmetry_rtol`, `hodge_kernel_rtol` | `dimensionless` | `1.0e-10` ($[0.0, 10^{-8}]$) | `validation/iphs.yaml` | `hodge_psd_check` |
| `validation.iphs.noise_covariance_tolerance`, `jump_ledger_tolerance` | `dimensionless` | `1.0e-10` ($[0.0, 10^{-6}]$) | `validation/iphs.yaml` | `stochastic_sdae` validation rules |
| `validation.dec_complex.cochain_residual_tolerance` | `dimensionless` | `0.0` ($[0.0, 10^{-3}]$) | `validation/dec_complex.yaml` | `cochain_complex_check` ($D_{k+1} D_k = 0$) |
| `coupling.external.variables.*`, `coupling.scales.defaults.variables.*` | catalog schema | SSOT variable definitions | `coupling/external_variables.yaml`, `scale_variables.yaml` | `SSOTCatalog` (`orchestration/ssot_catalog.py`) |

## 5. Numerical Methods and Discretization

1. **Compile-time assertion and rule dispatch**: `relic/orchestration/validator.py` validates `GraphIR` node/port schemas against `SSOTCatalog`, checks `_validate_assertion_payload`, and dispatches enabled structural checks through `ValidationRegistry` (`relic/validation/framework/registry.py`) without instantiating heavy scientific tensors when metadata-level rules suffice.
2. **Runtime-point autograd and Jacobi evaluation**: At Levels 4 and 5, `relic/validation/framework/rules/iphs.py` evaluates storage/entropy gradients via `torch.autograd.grad`, checks operator skew-symmetry and eigenvalue floors in `float64`, and evaluates the Poisson-bracket Jacobi residual over `sampled_domain_jacobi_residual` points (tensor-only `matrix_snapshot` Jacobi claims lack derivative information and fail closed).

## 6. Conservation, Boundary Conditions, and Ledgers

Structural governance enforces three non-negotiable separation boundaries:
- **Topology vs. physics**: `topology_signature.conservation_vectors` never substitutes for `StateManager` element ledgers or stoichiometric $C_h N_h = 0$ verification.
- **Port incidence vs. power balance**: `port_matching_check` verifies static wiring weights $\sum_i s_i w_i = 0$, whereas energetic/exergetic closure requires runtime effort-flow evaluation and `StateManager` commit intervals.
- **No unrouted residuals**: Non-zero clipping or overflow terms lacking an explicit `_ALLOWED_ROUTES` destination fail `overflow_routing_check`. Mature empirical Functors without thermodynamic potentials validate their stoichiometric/donor-recipient mass ledgers at Levels 1–3 without fabricating artificial chemical potentials $\mu$ or temperatures $T$.

## 7. Code Mapping

| Component / Rule | File Path | Architectural Role |
| --- | --- | --- |
| `GraphIR`, `validator.py`, `compiler.py`, `ssot_catalog.py`, `topology_algebra.py` | `relic/orchestration/` | Graph IR schema, assertion validation, `report.audit`, and incidence nullspace |
| `GraphExecutor` | `relic/orchestration/executor.py` | Runtime graph execution and `integration_plan_witness` verification |
| `IPHSStructureSpec`, `PortSpecIPHS`, `OpenSystemRoute`, `assemble_iphs_structure`, `poisson_jacobi_residual` | `relic/computation/operators/iphs.py` | Canonical dPH / GENERIC assembly and Jacobi residual operator |
| `IPHSEquivalenceCertificate`, `load_iphs_equivalence_certificate`, `resolve_iphs_level_requirement` | `relic/computation/operators/iphs_certificate.py` | Level 0–5 equivalence certificate loader and evidence gate |
| `DECComplexBundle`, `typed_stokes_audit` | `relic/computation/operators/dec/complex.py` | Discrete cochain complex and Stokes pairing audit |
| `IntegrationPlan` | `relic/computation/solvers/geometric.py` | Metadata-only stage-splitting plan (`verified=False`, `runtime_ready=False`) |
| `dec_complex.py`, `hodge.py`, `chain_map.py`, `port_matching.py`, `port_power.py`, `overflow_routing.py`, `reaction_network.py`, `iphs.py`, `equivalence_certificate.py` | `relic/validation/framework/rules/` | Registered structural and runtime-point validation rules |

## 8. Implementation Status

- **Implemented (`[I]`)**: `GraphIR.assertion` validation and 13-key `report.audit` compilation; software incidence nullspace summary; `cochain_complex_check`, `hodge_psd_check`, `chain_map_check`, `typed_stokes_audit`, `port_matching_check`, `overflow_routing_check`, `reaction_network_check`, and `equivalence_certificate_check`; `IPHSStructureSpec` dPH/GENERIC assembly; and 92 cataloged process/constructor `IPHSEquivalenceCertificate` YAML files (`L0 = 4`, `L1 = 15`, `L2 = 39`, `L3 = 32`, `L4 = 1`, `L5 = 1`).
- **Metadata-only / Not implemented (`[M]` / `[N]`)**: `IntegrationPlan.validate()` is `[M]` (`execution_semantics="metadata_only"`, `verified=False`, `runtime_ready=False`); tensor-only `matrix_snapshot` Jacobi verification is `[N]` (fail-closed); general MSFOL theorem proving, undecidability classification across arbitrary ecological queries, and any secondary orchestration runtime remain `[N]`.
- **IPHS Equivalence Certificates**: None (`0` process certificates use `structural_ecology_foundation.md` as their `theory_document`, because this page specifies the cross-repository certificate governance framework itself).

## 9. References

1. Arnold, D. N., Falk, R. S., and Winther, R. (2006). Finite element exterior calculus, homological techniques, and applications. *Acta Numerica*, 15, 1-155.
2. Feinberg, M. (2019). *Foundations of Chemical Reaction Network Theory*. Springer.
3. Finlayson, B. A. (2012). *Introduction to Chemical Engineering Computing* (2nd ed.). Wiley.
4. Godsil, C., and Royle, G. (2001). *Algebraic Graph Theory*. Springer.
5. Grmela, M., and Ottinger, H. C. (1997). Dynamics and thermodynamics of complex fluids. I. Development of a general formalism. *Physical Review E*, 56(6), 6620-6632.
6. Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed.). Springer.
7. Hirani, A. N. (2003). *Discrete Exterior Calculus* (Ph.D. thesis). California Institute of Technology.
8. Hundsdorfer, W., and Verwer, J. G. (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations*. Springer.
9. Ramirez, H., Le Gorrec, Y., Maschke, B., and Couenne, F. (2022). On the passivity based control of irreversible port-Hamiltonian systems. *Automatica*, 142, 110399.
10. Rao, R., and Esposito, M. (2016). Nonequilibrium thermodynamics of chemical reaction networks: Wisdom from stochastic thermodynamics. *Physical Review X*, 6(4), 041064.
11. Strang, G. (1968). On the construction and comparison of difference schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
12. Strang, G. (1988). *Linear Algebra and Its Applications* (3rd ed.). Harcourt Brace Jovanovich.
13. van der Schaft, A. J., and Maschke, B. M. (1995). The Hamiltonian formulation of energy conserving physical systems with external ports. *Archiv fur Elektronik und Ubertragungstechnik*, 49(5-6), 362-371.
