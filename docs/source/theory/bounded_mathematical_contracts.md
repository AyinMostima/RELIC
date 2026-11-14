# Bounded Mathematical Contracts: Chain Maps, Invasion Thresholds, and Relative Storage

## 1. Process Definition and Scope

`relic/computation/operators/` implements three finite-dimensional, decidable mathematical operator contracts over declared state, cochain, and rate matrices:
1. **Linear chain maps on typed cochain spaces** (`LinearChainMap` and `close_within` in `relic/computation/operators/dec/chain_map.py`, operating on `CochainSpaceSpec` and `TypedCochain` in `relic/computation/operators/dec/forms.py`).
2. **Resource-activity Metzler invasion thresholds** (`activity_threshold`, `is_nonsingular_m_matrix`, `resource_background`, and `InvasionThresholdCertificate` in `relic/computation/operators/invasion_threshold.py`).
3. **Relative storage Lyapunov certification and open-system feedback power ledgers** (`relative_storage`, `relative_storage_rate`, `certify_relative_storage`, `feedback_power_ledger`, and `RelativeStorageCertificate` in `relic/computation/operators/relative_storage.py`).

Theoretical derivations and counterexample boundaries for all three contracts appear in *Methods of Ecological Analysis*, Chapters 1, 2, 3, and 10. Each contract is a pure finite-dimensional operator audit (not a `FunctorBase` subclass and not a general symbolic theorem prover): every premise is verified numerically against `ParameterManager` tolerances in `relic/parameters/database/relic/validation/iphs.yaml`, and inputs violating any premise are refused explicitly.

## 2. Governing Equations

### 2.1 Linear Chain Maps on Declared Cochain Spaces

A `CochainSpaceSpec` declares a finite-dimensional $k$-cochain space $\mathcal{C}^k(\mathcal{K})$ via ten frozen fields: `complex_id`, `degree` $k \in \{0, 1, 2, 3\}$, `primality` $\in \{\texttt{"primal"}, \texttt{"dual"}\}$, `n_dofs` $n > 0$, `units`, `extensivity` $\in \{\texttt{"intensive"}, \texttt{"extensive"}, \texttt{"density"}, \texttt{"integral"}\}$, `physical_domain`, `semantic_name`, `orientation_convention`, and `boundary_role` $\in \{\texttt{"interior"}, \texttt{"boundary"}, \texttt{"port"}, \texttt{"mixed"}\}$ (note that DEC `extensivity` is structural cochain metadata, distinct from scale-coupling `SpatialQuantityBasis` `cell_total`/`area_density` in `relic/coupling/scales/types.py`).

For matrix $M \in \mathbb{R}^{n_t \times n_s}$ mapping $x \in \mathcal{C}^k(\mathcal{K}_s)$ to $y \in \mathcal{C}^k(\mathcal{K}_t)$ (Hatcher, 2002; Desbrun et al., 2005):

$$
y = M x, \qquad (M_2 \circ M_1) = M_2 M_1, \qquad \lVert M x - M x' \rVert_2 \le \lVert M \rVert_2\,\lVert x - x' \rVert_2.
$$

With scalar unit-conversion factor $u > 0$ from source to target cochain units and linear conservation-ledger matrices $C_s, C_t$, ledger preservation (`preserves_ledger`) and operator closeness (`distance`, `close_within`) satisfy:

$$
\lVert C_t (u M) - C_s \rVert_\infty \le \varepsilon_{\mathrm{cm}}, \qquad d(M_1, M_2) = \lVert u_1 M_1 - u_2 M_2 \rVert_2 \le \varepsilon,
$$

where $d(M_1, M_2) \le \varepsilon$ is reflexive and symmetric without transitivity; exact `equals` ($d = 0$ on identical spaces) is the sole equivalence relation. `LinearChainMap` does not own a coboundary commutation method; coboundary commutation $R_{k+1} D_s = D_t R_k$ is verified separately by `chain_map_check` (`relic/validation/framework/rules/chain_map.py`). In pedosphere production code, `LitterfallInterfaceMapper` (`relic/scientific/pedosphere/interface_mapping.py`; Parton et al., 1987) forms a linear chain map at fixed lignin ratio that preserves carbon and nitrogen element ledgers.

### 2.2 Resource-Activity Metzler Invasion Threshold

Consider a linearized resource-activity subsystem around an activity-free resource background $R_0 = \Lambda_R^{-1} I_R$ (with nonsingular M-matrix resource loss $\Lambda_R$ and supply $I_R$), nonnegative growth matrix $G = G(R_0) \ge 0$, and square loss operator $\Delta \in \mathbb{R}^{m \times m}$. By Theorem 6.2.3 (Condition N38) of Berman and Plemmons (1994), $\Delta$ is a nonsingular M-matrix iff $\Delta$ is a Z-matrix ($\Delta_{ij} \le \varepsilon_M$ for $i \neq j$) and $\Delta^{-1} \ge -\varepsilon_M$ entrywise. Under $G \ge 0$ and $\Delta$ a nonsingular M-matrix, $G - \Delta$ is Metzler and the next-generation activity threshold $\mathcal{R}_{\mathrm{act}}$ satisfies the exact sign equivalence (Diekmann, Heesterbeek, and Metz, 1990; van den Driessche and Watmough, 2002):

$$
\mathcal{R}_{\mathrm{act}} = \rho\!\left(\Delta^{-1} G\right), \qquad \operatorname{sign}\!\left(\mathcal{R}_{\mathrm{act}} - 1\right) = \operatorname{sign}\!\left(s(G - \Delta)\right),
$$

where $\rho(A) = \max_j \lvert \lambda_j(A) \rvert$ is the spectral radius and $s(A) = \max_j \operatorname{Re}\lambda_j(A)$ is the spectral abscissa. For $\mathcal{R}_{\mathrm{act}} > 1 + \varepsilon_{\mathrm{crit}}$ the activity-free state is linearly unstable (`"invasion"`), and for $\mathcal{R}_{\mathrm{act}} < 1 - \varepsilon_{\mathrm{crit}}$ it is linearly asymptotically stable (`"decay"`). Inside the non-hyperbolic critical band $\lvert \mathcal{R}_{\mathrm{act}} - 1 \rvert \le \varepsilon_{\mathrm{crit}}$, `activity_threshold` returns `verdict="critical"` and `hyperbolic=False` without claiming stability or instability. No global persistence, transcritical bifurcation normal form, or unique positive equilibrium is inferred, and naive left-eigenvector inequalities $G v < \Delta v$ without the M-matrix inverse premise fail on off-diagonal counterexample matrices.

In `MicrobialEnzymeDecomposition` (`relic/scientific/pedosphere/decomposition/microbial_enzyme.py`), linearizing near minimum admitted microbial biomass $\mathrm{MIC}$ yields scalar $1 \times 1$ rates $G = \mathrm{CUE}\sum_i V_i / \mathrm{MIC}$ and $\Delta = \tau + \tau_{\mathrm{maint}} f_T$, whose threshold $\mathcal{R}_{\mathrm{act}} = G / \Delta$ matches the sign of net microbial biomass tendency across low- and high-substrate regimes.

### 2.3 Relative Storage on an Affine Ledger Leaf and Open-System Power Ledger

For a differentiable storage functional $F: \mathbb{R}^n \to \mathbb{R}$ (energy or exergy) on an affine conservation leaf $\mathcal{L}_{x^*} = \{x \in \mathbb{R}^n : C x = C x^*\}$ ($C \in \mathbb{R}^{r \times n}$), the Bregman-type relative storage and its exact chain-rule time derivative along $\dot{x}$ are (Willems, 1972; Khalil, 2002, Theorem 4.1):

$$
V(x) = F(x) - F(x^*) - \nabla F(x^*)^{\mathsf{T}}(x - x^*), \qquad
\dot{V}(x) = \left(\nabla F(x) - \nabla F(x^*)\right)^{\mathsf{T}} \dot{x}.
$$

`certify_relative_storage` verifies two premises at candidate equilibrium $x^*$ in `float64` via exact second-order PyTorch autograd and an SVD orthonormal basis $Z$ of $\ker C$:

$$
\min_{\lambda \in \mathbb{R}^r} \lVert \nabla F(x^*) - C^{\mathsf{T}}\lambda \rVert_2 \le \varepsilon_{\mathrm{rs}} \max\!\left(g_{\mathrm{floor}},\; \lVert \nabla F(x^*) \rVert_2\right), \qquad
\lambda_{\min}\!\left(Z^{\mathsf{T}} \nabla^2 F(x^*) Z\right) > \varepsilon_{\mathrm{rs}}.
$$

When both hold and $C \dot{x} = 0$, $\nabla F(x^*)^{\mathsf{T}}\dot{x} = \lambda^{\mathsf{T}} C \dot{x} = 0$ so $\dot{V} = \dot{F} = -\mathcal{D}(x) \le 0$ (`verdict="lyapunov_candidate"`). The linear term $-\nabla F(x^*)^{\mathsf{T}}\dot{x}$ is never omitted: for the non-stationary scalar counterexample $F(z) = 2 + z + \tfrac{1}{2}z^2$ at $z^* = 0$ ($\nabla F(0) = 1 \neq 0$, unconstrained) under $\dot{z} = -z^2$, `certify_relative_storage` refuses $z^* = 0$ while `relative_storage_rate` reproduces $\dot{V}(z) = -z^3 > 0$ for $z < 0$.

For open systems with port efforts $y_F$, port inputs $u$, dissipation $\mathcal{D} \ge 0$, and boundary storage flow $\Phi_F$, `feedback_power_ledger` audits the Willems passivity balance:

$$
r_P = \dot{F} - \left(y_F^{\mathsf{T}} u - \mathcal{D} + \Phi_F\right), \qquad
\lvert r_P \rvert \le \varepsilon_{\mathrm{fp}} \max\!\left(p_{\mathrm{floor}},\; \lvert \dot{F} \rvert,\; \lvert y_F^{\mathsf{T}} u \rvert,\; \lvert \mathcal{D} \rvert,\; \lvert \Phi_F \rvert\right).
$$

In closed 1D `FickianDiffusion` (`relic/scientific/transport/diffusion.py`; Crank, 1975) with quadratic storage $H(c) = \tfrac{1}{2} c^{\mathsf{T}} c$ and mass ledger $C = \mathbf{1}^{\mathsf{T}}$, the spatially uniform mean concentration $c^*$ satisfies $\nabla H(c^*) = c^* = \bar{c}\,\mathbf{1} = C^{\mathsf{T}}\bar{c}$ with leaf Hessian $Z^{\mathsf{T}} I Z = I \succ 0$, decreasing $V(c)$ monotonically and closing `feedback_power_ledger` at zero external port power.

## 3. Variable Table

| Symbol / Type | Meaning | Units | Owner | Role |
| --- | --- | --- | --- | --- |
| `CochainSpaceSpec`, `TypedCochain` | 10-field discrete $k$-cochain space descriptor and payload wrapper | `space.units` | `operators/dec/forms.py` | Chain-map domain/codomain |
| $M$, $C_s$, $C_t$, $u$ | Linear chain-map matrix, source/target ledger matrices, unit scale | `[target]/[source]` | `LinearChainMap` | Chain-map operator |
| $G$, $\Delta$, $R_0$ | Nonnegative growth matrix, nonsingular M-matrix loss, resource background | $\mathrm{s^{-1}}$, `[resource]` | `invasion_threshold.py` | Threshold inputs |
| `InvasionThresholdCertificate` | `activity_threshold` ($\mathcal{R}_{\mathrm{act}}$), `spectral_bound` ($s$), `verdict`, `hyperbolic` | $1$, $\mathrm{s^{-1}}$ | `invasion_threshold.py` | Threshold certificate |
| $V(x)$, $\dot{V}(x)$ | Bregman relative storage and full chain-rule storage rate | `[storage]`, `[storage] s-1` | `relative_storage.py` | Lyapunov diagnostics |
| `RelativeStorageCertificate` | `stationary`, `multiplier` ($\lambda$), `gradient_residual`, `hessian_min_eigenvalue_on_leaf`, `definite_on_leaf`, `verdict` | `[storage]/[state]` | `relative_storage.py` | Lyapunov certificate |

## 4. Parameter Table

All default tolerances reside in `relic/parameters/database/relic/validation/iphs.yaml` and are loaded via `get_global_manager().get(key, required=True)` without Python fallback constants.

| Symbol | `ParameterManager` Key | Units | Default (Bounds) | Consumer |
| --- | --- | --- | --- | --- |
| $\varepsilon_{\mathrm{cm}}$ | `validation.iphs.chain_map_ledger_tolerance` | `dimensionless` | `1.0e-12` ($[0.0, 10^{-6}]$) | `LinearChainMap.preserves_ledger` |
| $\varepsilon_M$ | `validation.iphs.m_matrix_sign_tolerance` | `dimensionless` | `1.0e-12` ($[0.0, 10^{-6}]$) | `is_nonsingular_m_matrix` ($Z$-sign & $\Delta^{-1} \ge 0$) |
| $\varepsilon_{\mathrm{crit}}$ | `validation.iphs.invasion_threshold_critical_band` | `dimensionless` | `1.0e-9` ($[0.0, 10^{-3}]$) | `activity_threshold` hyperbolicity band |
| $\varepsilon_{\mathrm{rs}}$ | `validation.iphs.relative_storage_tolerance` | `dimensionless` | `1.0e-9` ($[0.0, 10^{-3}]$) | `certify_relative_storage` stationarity & leaf-Hessian check |
| $g_{\mathrm{floor}}$ | `validation.iphs.relative_storage_gradient_scale_floor` | `dimensionless` | `1.0` ($[0.0, 10^6]$) | `certify_relative_storage` mixed gradient scale |
| $\varepsilon_{\mathrm{fp}}$ | `validation.iphs.feedback_power_ledger_rtol` | `dimensionless` | `1.0e-9` ($[0.0, 10^{-3}]$) | `feedback_power_ledger` open-system tolerance |
| $p_{\mathrm{floor}}$ | `validation.iphs.feedback_power_ledger_scale_floor` | `dimensionless` | `1.0` ($[0.0, 10^6]$) | `feedback_power_ledger` mixed power scale |

## 5. Numerical Methods and Discretization

1. **Matrix and SVD linear algebra**: `LinearChainMap` evaluates operator 2-norms via `torch.linalg.matrix_norm(..., ord=2)` and converts units via `relic.data.standards.unit_converter`. In `invasion_threshold.py`, `is_nonsingular_m_matrix` checks finite square shape, off-diagonal $Z$-signs, and entrywise nonnegativity of `torch.linalg.inv(Delta)`; `activity_threshold` solves $\Delta^{-1} G$ via `torch.linalg.solve` and cross-checks $\operatorname{sign}(\mathcal{R}_{\mathrm{act}} - 1) == \operatorname{sign}(s(G - \Delta))$ outside the critical band, raising `RuntimeError` if numerical conditioning breaks the Metzler equivalence.
2. **Autograd Hessian projection onto $\ker C$**: `certify_relative_storage` promotes $x^*$ to `torch.float64`, computes exact $\nabla F(x^*)$ and $\nabla^2 F(x^*)$ via `torch.autograd.grad(..., create_graph=True)`, solves least-squares multipliers $\lambda$ via `torch.linalg.lstsq(C.T, grad)`, extracts $\ker C$ from `torch.linalg.svd(C, full_matrices=True)` above singular-value cutoff $\epsilon_{\mathrm{mach}}\max(r, n)\sigma_{\max}$ (Golub and Van Loan, 2013, Section 5.4.1), and computes `torch.linalg.eigvalsh` of the projected symmetric Hessian $Z^{\mathsf{T}}\nabla^2 F(x^*) Z$.

## 6. Conservation, Boundary Conditions, and Ledgers

Each operator enforces an explicit conservation or passivity boundary:
- `LinearChainMap.preserves_ledger` verifies exact linear conservation across domain transfers ($C_t(u M) = C_s$) without modifying cochain payloads.
- `certify_relative_storage` restricts Lyapunov definiteness strictly to affine conservation leaves $C x = C x^*$; curved nonlinear constraint manifolds cannot be passed as linear ledger matrices and remain outside scope.
- `feedback_power_ledger` refuses negative dissipation ($\mathcal{D} < 0$) with `ValueError` and separates internal dissipation $\mathcal{D}$ from external port power $y_F^{\mathsf{T}} u$ and boundary storage flux $\Phi_F$.

## 7. Code Mapping

| Symbol / Class | File Path | Architectural Role | Production Consumer |
| --- | --- | --- | --- |
| `CochainSpaceSpec`, `ChainSpaceSpec`, `TypedCochain`, `TypedChain` | `relic/computation/operators/dec/forms.py` | Typed discrete chain/cochain space views | `LinearChainMap`, DEC validation rules |
| `LinearChainMap`, `close_within` | `relic/computation/operators/dec/chain_map.py` | Finite-dimensional cochain morphism & ledger check | `LitterfallInterfaceMapper` (`relic/scientific/pedosphere/interface_mapping.py`) |
| `InvasionThresholdCertificate`, `activity_threshold`, `is_nonsingular_m_matrix`, `resource_background` | `relic/computation/operators/invasion_threshold.py` | Metzler / next-generation spectral threshold contract | `MicrobialEnzymeDecomposition` (`relic/scientific/pedosphere/decomposition/microbial_enzyme.py`) |
| `RelativeStorageCertificate`, `relative_storage`, `relative_storage_rate`, `certify_relative_storage`, `feedback_power_ledger` | `relic/computation/operators/relative_storage.py` | Affine-leaf Bregman storage & Willems passivity ledger | Closed 1D `FickianDiffusion` (`relic/scientific/transport/diffusion.py`) |

## 8. Implementation Status

- **Implemented (`[I]`)**: 10-field `CochainSpaceSpec` and `LinearChainMap` composition, unit-aware ledger preservation, operator-norm error bounds, and non-transitive `close_within`; nonsingular M-matrix verification and `activity_threshold` with Metzler spectral-abscissa cross-checking and critical-band refusal; `float64` autograd `certify_relative_storage` on SVD-derived affine ledger leaves and `feedback_power_ledger`.
- **Not implemented (`[N]`)**: Nonlinear curved conservation-manifold Lyapunov certification, global persistence or bifurcation continuation beyond local linear invasion thresholds, infinite-dimensional PDE Sobolev norm bounds inside `LinearChainMap`, and general first-order logic theorem proving.
- **IPHS Equivalence Certificates**: None (`0` certificates reference `bounded_mathematical_contracts.md`; consumer Functors hold their own domain certificates on [phenology.md](phenology.md), [decomposition.md](decomposition.md), and [transport_fickian_diffusion.md](transport_fickian_diffusion.md)).

## 9. References

1. Berman, A., and Plemmons, R. J. (1994). *Nonnegative Matrices in the Mathematical Sciences* (Chapter 6, Theorem 6.2.3). SIAM.
2. Crank, J. (1975). *The Mathematics of Diffusion* (2nd ed.). Oxford University Press.
3. Desbrun, M., Hirani, A. N., Leok, M., and Marsden, J. E. (2005). Discrete exterior calculus. *arXiv:math/0508341*.
4. Diekmann, O., Heesterbeek, J. A. P., and Metz, J. A. J. (1990). On the definition and the computation of the basic reproduction ratio $R_0$ in models for infectious diseases in heterogeneous populations. *Journal of Mathematical Biology*, 28(4), 365-382.
5. Golub, G. H., and Van Loan, C. F. (2013). *Matrix Computations* (4th ed., Section 5.4.1). Johns Hopkins University Press.
6. Hatcher, A. (2002). *Algebraic Topology* (Chapter 2). Cambridge University Press.
7. Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed., Theorem 4.1). Prentice Hall.
8. Parton, W. J., Schimel, D. S., Cole, C. V., and Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. *Soil Science Society of America Journal*, 51(5), 1173-1179.
9. van den Driessche, P., and Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
10. Willems, J. C. (1972). Dissipative dynamical systems, Part I: General theory. *Archive for Rational Mechanics and Analysis*, 45(5), 321-351.
