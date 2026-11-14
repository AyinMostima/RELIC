(api-scientific-structure-extensions)=
# Scientific Structure and Extension APIs

This reference covers four separate API families:

1. DEC typed structural views.
2. IPHS structure, reaction advancement, and equivalence certificates.
3. Python Functor extensions described by `relic_extension.yaml`.
4. C ABI coupling plugins hosted through `PluginHost`.

These families interoperate with existing RELIC owners. They do not introduce
another StateManager, ParameterManager, simulator, compiler, port schema, or
registry.

:::{important}
The Python extension loader and the C ABI coupling-plugin host are different
protocols. A Python scientific extension implements one of the four bridge
classes. A C ABI plugin crosses a language boundary and is wrapped by
`PluginFunctorAdapter`. Their manifests, validation paths, and lifecycle roles
must not be conflated.
:::

## DEC typed structural views

Stable imports:

```python
from relic.computation.operators.dec import (
    CochainSpaceSpec,
    DECComplexBundle,
    DiagnosticSpec,
    HodgeStarSpec,
    LinearChainMap,
    ObservationOperatorSpec,
    TypedCochain,
    close_within,
    cochain_space_from_ssot_entry,
    typed_cochain_delta_to_update_item,
)
```

These classes are typed views over current state, SSOT metadata, and runtime
contracts. They do not replace `StateManager`, `SSOTCatalog`, or
`ContractRegistry`.

### `CochainSpaceSpec`

Implementation:
`relic.computation.operators.dec.forms.CochainSpaceSpec`

```python
CochainSpaceSpec(
    complex_id: str,
    degree: Literal[0, 1, 2, 3],
    primality: Literal["primal", "dual"],
    n_dofs: int,
    units: str,
    extensivity: Literal[
        "intensive",
        "extensive",
        "density",
        "integral",
    ],
    physical_domain: str,
    semantic_name: str,
    orientation_convention: str,
    boundary_role: Literal[
        "interior",
        "boundary",
        "port",
        "mixed",
    ] = "interior",
)
```

| Field | Purpose | Failure semantics |
| --- | --- | --- |
| `complex_id` | Identifier of the discrete complex. | Empty text raises `ValueError`. |
| `degree` | Form degree in the discrete cochain space. | Boolean, non-integer, or values outside 0 through 3 raise `ValueError`. |
| `primality` | Select the primal or dual complex. | Unsupported values raise `ValueError`. |
| `n_dofs` | Number of cochain degrees of freedom. | Non-positive or non-integer values raise `ValueError`. |
| `units` | Physical units inherited from SSOT metadata. | Empty text raises `ValueError`. |
| `extensivity` | Intensive, extensive, density, or integral interpretation. | Unsupported values raise `ValueError`. |
| `physical_domain` | Domain in which the cochain is defined. | Empty text raises `ValueError`. |
| `semantic_name` | Scientific variable identity. | Empty text raises `ValueError`. |
| `orientation_convention` | Orientation convention shared by operators. | Empty text raises `ValueError`. |
| `boundary_role` | Interior, boundary, port, or mixed support. | Unsupported values raise `ValueError`. |

### `TypedCochain`

```python
TypedCochain(
    name: str,
    values: torch.Tensor,
    space: CochainSpaceSpec,
    time: Optional[float] = None,
    provenance: Optional[Mapping[str, str]] = None,
)
```

| Field | Purpose | Failure semantics |
| --- | --- | --- |
| `name` | Variable name bound to the cochain. | Empty text raises `ValueError`. |
| `values` | Rank-one cochain coefficients. | Non-Tensor input raises `TypeError`; wrong rank, size, or non-finite values raise `ValueError`. |
| `space` | Typed support-space contract. | `values.numel()` must equal `space.n_dofs`. |
| `time` | Optional finite observation or state time. | Boolean, non-numeric, or non-finite values raise `ValueError`. |
| `provenance` | Optional string-to-string provenance. | Other mappings raise `ValueError`. |

### `ObservationOperatorSpec`

```python
ObservationOperatorSpec(
    name: str,
    input_state_vars: Tuple[str, ...],
    output_name: str,
    output_units: str,
    support: str,
    averaging_window: str,
    footprint_description: str,
    error_model: str,
    conservative_projection_required: bool = False,
    metadata: Mapping[str, Any] = {},
)
```

This immutable declaration describes a typed map from model state to data
space. `input_state_vars` must not be empty. Required text fields must be
non-empty or construction raises `ValueError`.

### `DiagnosticSpec`

```python
DiagnosticSpec(
    name: str,
    input_state_vars: Tuple[str, ...],
    output_units: str,
    diagnostic_kind: str,
    writes_state: bool = False,
    feedback_port: Optional[str] = None,
    metadata: Mapping[str, Any] = {},
)
```

`DiagnosticSpec` remains read-only: `writes_state` must be the exact boolean
`False`, even with a `feedback_port`. That declaration never authorizes a
state write. Managed feedback has a separate existing non-diagnostic owner:
`relic.coupling.observation.state_feedback.ObservationStateFeedbackFunctor`.
It forms the direct-observation innovation `d=y-x`, checks
`abs(d)/sqrt(B+R)<=nu_max`, and proposes `x_a=x+B/(B+R)*d` as a `REPLACE`
update through StateManager. The target SCM must authorize
`observation_state_feedback`; its gain, increment, and analysis variance are
diagnostics on the observation route. Tower/support observation operators
remain read-only. This scalar identity-observation binding is not a general
assimilation algorithm or authorization to mutate diagnostics.

### `cochain_space_from_ssot_entry()`

```python
cochain_space_from_ssot_entry(
    entry: Any,
    *,
    complex_id: str,
    n_dofs: int,
    degree: Optional[int] = None,
    primality: Optional[str] = None,
    extensivity: Optional[str] = None,
    physical_domain: Optional[str] = None,
    orientation_convention: Optional[str] = None,
    boundary_role: Optional[str] = None,
) -> CochainSpaceSpec
```

Explicit arguments override optional DEC metadata in `entry.raw`. Variable
identity and units remain owned by the SSOT entry.

The adapter raises `ValueError` when the SSOT entry lacks units or a semantic
name, or when required DEC properties cannot be resolved.

### `DECComplexBundle`

Implementation:
`relic.computation.operators.dec.complex.DECComplexBundle`

```python
DECComplexBundle(
    complex_id: str,
    dimension: int,
    n_cells: Mapping[int, int],
    D: Mapping[int, torch.Tensor],
    boundary_masks: Mapping[int, torch.Tensor],
    orientation: Mapping[int, torch.Tensor],
    cell_measures: Mapping[int, torch.Tensor],
    dual_measures: Mapping[int, torch.Tensor],
    orientation_convention: str,
)
```

| Field | Purpose |
| --- | --- |
| `complex_id` | Discrete-complex identity. |
| `dimension` | Topological dimension from 0 through 3. |
| `n_cells` | Number of cells for each topological degree. |
| `D` | Discrete exterior derivative matrices keyed by source degree. |
| `boundary_masks` | Binary boundary masks by degree. |
| `orientation` | Orientation signs by degree. |
| `cell_measures` | Positive primal measures by degree. |
| `dual_measures` | Positive dual measures by degree. |
| `orientation_convention` | Shared orientation convention. |

Constructor validation checks basic identities, dimensions, and derivative
matrix shapes.

```python
bundle.validate_complex(
    tol: float = 0.0,
    *,
    strict: bool = False,
) -> Dict[Tuple[int, int], float]
```

The method evaluates:

$$
D_{k+1,h} D_{k,h} = 0.
$$

It returns residual norms keyed by the composed degree pair. A negative,
non-finite, or non-numeric tolerance raises `ValueError`.

With `strict=True`, the bundle must contain the complete derivative sequence
and all support tensors on one device. Boundary masks must be binary,
orientations must be `-1` or `1`, and primal and dual measures must be
positive. Missing data, invalid shapes, non-finite values, or residuals above
`tol` raise `ValueError`.

Strict validation also requires integer-valued incidence within the exact
integer range of the dtype, `D[0] @ 1 = 0`, `0 <= tol < 0.5`, no boundary mark
on top-degree cells, and a boundary subcomplex closed under the boundary
operator. Non-strict validation checks only the consecutive `D` pairs actually
supplied. `coboundary_matrix(k)`, `boundary_matrix(k)` (the transpose view of
the unique `D` owner), `trace_matrix(k)`, and `typed_stokes_audit(alpha, chain)`
are also public; the Stokes audit evaluates one pairing
$\langle D_k\alpha,c\rangle-\langle\alpha,\partial_{k+1}c\rangle$ and does
not prove mesh provenance or boundary-mask completeness.

### `HodgeStarSpec`

Implementation:
`relic.computation.operators.dec.hodge.HodgeStarSpec`

```python
HodgeStarSpec(
    name: str,
    domain: CochainSpaceSpec,
    codomain: CochainSpaceSpec,
    matrix: torch.Tensor,
    parameter_keys: Tuple[str, ...],
    metric_keys: Tuple[str, ...],
    material_law: str,
    symmetry_tol: float | None = None,
    psd_tol: float | None = None,
    complex_dimension: Optional[int] = None,
    operator_units: Optional[str] = None,
    role: str = "unspecified",
    definiteness: str = "unspecified",
    kernel_basis: Optional[torch.Tensor] = None,
    kernel_rtol: float | None = None,
)
```

| Field | Purpose |
| --- | --- |
| `name` | Operator identity. |
| `domain` | Source cochain space. |
| `codomain` | Target cochain space. |
| `matrix` | Discrete Hodge matrix with shape `codomain.n_dofs` by `domain.n_dofs`. |
| `parameter_keys` | ParameterManager keys used by material terms. |
| `metric_keys` | ParameterManager keys used by metric terms. |
| `material_law` | Declared material closure. |
| `symmetry_tol` | Relative symmetry tolerance for passive square operators; `None` reads `validation.iphs.hodge_symmetry_rtol`. |
| `psd_tol` | Relative spectral tolerance for negative eigenvalues; `None` reads `validation.iphs.hodge_psd_rtol`. |
| `complex_dimension` | Dimension used to check complementary form degrees. |
| `operator_units` | Units multiplying domain units into codomain units. |
| `role` | `unspecified`, `metric_hodge`, or `material_hodge`. |
| `definiteness` | `unspecified`, `positive_definite`, or `positive_semidefinite`. |
| `kernel_basis` | Columns spanning the kernel of a singular material Hodge; `None` for a positive definite operator. |
| `kernel_rtol` | Relative tolerance for the kernel residual and rank checks; `None` reads `validation.iphs.hodge_kernel_rtol`. |

Construction resolves each `None` tolerance through ParameterManager; a
missing key raises. It rejects invalid matrix types, rank, shape, finite
values, empty keys, invalid tolerances, or invalid complex dimensions.

```python
hodge.validate(
    *,
    passive: bool = True,
    strict: bool = False,
    parameter_resolver: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Union[float, bool, str]]
```

Strict validation requires:

- one complex and physical domain;
- opposite primality and complementary form degrees;
- one orientation and boundary convention;
- resolvable, non-wildcard parameter and metric keys;
- a parseable unit relation in which `operator_units * domain.units` is
  convertible to `codomain.units`.

A strict passive operator must be square, declare `role` (`metric_hodge` or
`material_hodge`) and `definiteness`, and use tolerances not above
`sqrt(machine epsilon)` for its dtype. Every passive square matrix must be
symmetric and positive semidefinite within tolerance. A strict `metric_hodge`
must be `positive_definite`; a strict `positive_semidefinite` operator must be
a `material_hodge` with a linearly independent `kernel_basis` whose width
equals the numerical nullity and whose image `matrix @ kernel_basis` vanishes
within `kernel_rtol`. Violations raise `ValueError`. These checks concern the
supplied matrix value, not mesh provenance or refinement convergence.

### `typed_cochain_delta_to_update_item()`

```python
typed_cochain_delta_to_update_item(
    cochain: TypedCochain,
    *,
    source_functor: str,
    registry: Any = None,
    variable_name: Optional[str] = None,
    mode: Optional[Union[UpdateMode, str]] = None,
    write_mode: Optional[Union[StateWriteMode, str]] = None,
    per_timestep: bool = True,
    element: Optional[str] = None,
) -> Tuple[str, UpdateItem]
```

| Parameter | Purpose |
| --- | --- |
| `cochain` | Typed state delta to lower into the current runtime contract. |
| `source_functor` | Writer identity checked by `ContractRegistry`. |
| `registry` | Existing `ContractRegistry`; it is required. |
| `variable_name` | Optional target name; otherwise `cochain.name`. |
| `mode` | Explicit `UpdateMode`; defaults to `ACCUMULATE` for per-step values and `FLUX` for rates. |
| `write_mode` | Optional `StateWriteMode`. |
| `per_timestep` | Distinguishes already integrated deltas from per-second rates. |
| `element` | Optional conservation-ledger label such as water, carbon, or nitrogen. |

Returns the canonical variable name and an existing `UpdateItem`. A wrong
cochain type raises `TypeError`; missing registry or names raise `ValueError`;
an incompatible registry API raises `TypeError`; unauthorized writes raise
`PermissionError`.

### `LinearChainMap` and `close_within()`

Implementation:
`relic.computation.operators.dec.chain_map.LinearChainMap` and
`relic.computation.operators.dec.chain_map.close_within`

```python
LinearChainMap(
    name: str,
    source: CochainSpaceSpec,
    target: CochainSpaceSpec,
    matrix: torch.Tensor,
    unit_factor: float,
)
```

| Field | Purpose | Constructor validation (`__post_init__`) |
| --- | --- | --- |
| `name` | Chain-map identifier used in output cochain labels and diagnostics. | Stored on the frozen dataclass. |
| `source` | Domain `CochainSpaceSpec` ($n_{\mathrm{source}}$ DOFs). | Checked by `apply()`, `compose()`, `equals()`, and `distance()`. |
| `target` | Codomain `CochainSpaceSpec` ($n_{\mathrm{target}}$ DOFs). | Checked against `matrix.shape[0]`. |
| `matrix` | Linear operator $M$ mapping source coefficients to target coefficients. | Must satisfy (1) `matrix.ndim == 2`, (2) `tuple(matrix.shape) == (target.n_dofs, source.n_dofs)`, and (3) `torch.isfinite(matrix).all()`; violations raise `ValueError`. |
| `unit_factor` | Dimensionless scalar factor $u > 0$ converting source units to target units. | Must satisfy (4) `unit_factor > 0.0`; non-positive or non-finite values raise `ValueError`. |

Methods and distance helper:

```python
LinearChainMap.identity(
    space: CochainSpaceSpec,
    *,
    name: str = "identity",
) -> LinearChainMap

chain_map.apply(cochain: TypedCochain) -> TypedCochain

chain_map.compose(
    first: LinearChainMap,
    *,
    name: Optional[str] = None,
) -> LinearChainMap

chain_map.preserves_ledger(
    source_ledger: torch.Tensor,
    target_ledger: torch.Tensor,
    *,
    tolerance: Optional[float] = None,
) -> Tuple[bool, float]

chain_map.equals(other: LinearChainMap) -> bool
chain_map.distance(other: LinearChainMap) -> float
chain_map.error_bound(perturbation: torch.Tensor) -> float

close_within(
    first: LinearChainMap,
    second: LinearChainMap,
    epsilon: float,
) -> bool
```

`apply(cochain)` requires `cochain.space == self.source` (`ValueError`
otherwise) and evaluates $(M x) \cdot u$ into a `TypedCochain` on
`self.target` preserving `time` and `provenance`. `compose(first)` forms
$M_2 \circ M_1$ when `first.target == self.source`, multiplying matrices in
`float64` and multiplying `unit_factor` values. `preserves_ledger(...)` checks
$\max \lvert C_{\mathrm{target}} (u M) - C_{\mathrm{source}} \rvert \le \tau$,
resolving `validation.iphs.chain_map_ledger_tolerance` from
`get_global_manager()` when `tolerance is None`. `equals(other)` tests exact
equivalence of spaces, `unit_factor`, and `float64` matrices, whereas
`distance(other)` returns the operator 2-norm
$\lVert u_1 M_1 - u_2 M_2 \rVert_2$ on identical spaces and
`close_within(first, second, epsilon)` evaluates that reflexive, symmetric,
non-transitive distance bound.

```python
import torch
from relic.computation.operators.dec import (
    CochainSpaceSpec,
    LinearChainMap,
    TypedCochain,
    close_within,
)

fine = CochainSpaceSpec(
    complex_id="col_2",
    degree=2,
    primality="primal",
    n_dofs=2,
    units="kg",
    extensivity="extensive",
    physical_domain="soil",
    semantic_name="soil_carbon_layer",
    orientation_convention="right_handed",
)
coarse = CochainSpaceSpec(
    complex_id="col_1",
    degree=2,
    primality="primal",
    n_dofs=1,
    units="g",
    extensivity="extensive",
    physical_domain="soil",
    semantic_name="soil_carbon_total",
    orientation_convention="right_handed",
)
agg = LinearChainMap(
    name="sum_kg_to_g",
    source=fine,
    target=coarse,
    matrix=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    unit_factor=1000.0,
)
out = agg.apply(TypedCochain(name="c", values=torch.tensor([1.5, 2.5]), space=fine))
ok, residual = agg.preserves_ledger(
    source_ledger=torch.tensor([[1000.0, 1000.0]]),
    target_ledger=torch.tensor([[1.0]]),
    tolerance=1e-12,
)
assert ok and residual == 0.0 and float(out.values.item()) == 4000.0
assert close_within(agg, agg.compose(LinearChainMap.identity(fine)), 1e-12)
```

## IPHS structure and reaction advancement

Stable imports:

```python
from relic.computation.operators.iphs import (
    IPHSStructureSpec,
    PortSpecIPHS,
    ReactionNetworkAdvance,
    advance_reaction_network,
    assemble_iphs_structure,
)
```

### `PortSpecIPHS`

```python
PortSpecIPHS(
    name: str,
    effort_var: str,
    flow_var: str,
    effort_space: CochainSpaceSpec,
    flow_space: CochainSpaceSpec,
    power_units: str,
    positive_power_convention: Literal[
        "into_system",
        "out_of_system",
    ],
    port_kind: Literal[
        "energy",
        "entropy",
        "matter",
        "heat",
        "momentum",
        "radiation",
        "management",
    ],
    role: Literal["unspecified", "internal", "external"] = "unspecified",
)
```

| Field | Purpose |
| --- | --- |
| `name` | Port identity. |
| `effort_var` | Effort variable name. |
| `flow_var` | Flow variable name. |
| `effort_space` | Typed cochain support for effort. |
| `flow_space` | Typed cochain support for flow. |
| `power_units` | Units of the effort-flow product. |
| `positive_power_convention` | Direction treated as positive power. |
| `port_kind` | Physical or management port classification. |

Construction validates non-empty identity/unit strings, typed cochain
spaces, known sign/kind/role values, matching effort/flow degrees of freedom,
and boundary or port support. `role` is `unspecified`, `internal`, or
`external`; it defaults to `unspecified`. Physical unit compatibility and
power pairing still require the surrounding structural checks.

### `IPHSStructureSpec`

```python
IPHSStructureSpec(
    name: str,
    state_vars: Tuple[str, ...],
    energy_functional: str,
    entropy_functional: str,
    J: torch.Tensor,
    M: torch.Tensor,
    G: Optional[torch.Tensor],
    stoichiometric_N: Optional[torch.Tensor],
    ports: Tuple[PortSpecIPHS, ...],
    R: Optional[torch.Tensor] = None,
    formulation: Literal["unspecified", "dissipative_ph", "generic"] = "unspecified",
    reversible_operator_state_dependence: Literal[
        "unspecified", "constant", "state_dependent"
    ] = "unspecified",
    jacobi_identity_evidence: str = "",
    skew_tol: Optional[float] = None,
    psd_tol: Optional[float] = None,
    degeneracy_tol: Optional[float] = None,
    energy_gradient: Optional[torch.Tensor] = None,
    entropy_gradient: Optional[torch.Tensor] = None,
    contribution_count: int = 0,
    contribution_sources: Tuple[str, ...] = (),
    storage_kind: Literal["unspecified", "total_energy", "free_energy"] = "unspecified",
    poisson_field: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    poisson_domain: Optional[torch.Tensor] = None,
    jacobi_tol: Optional[float] = None,
    *,
    poisson_operator: Literal[
        "matrix_snapshot", "zero_field", "constant_field", "state_field"
    ],
    reaction_representation: Literal["independent", "absorbed"],
)
```

`poisson_operator` and `reaction_representation` are keyword-only and
required: no reversible-operator policy or reaction representation is
assumed.

The formulation-specific canonical drifts are:

$$
\begin{aligned}
\text{dissipative pH:}\quad
\dot{x}&=(J-R)e_H+Gu+Nr,\\
\text{GENERIC:}\quad
\dot{x}&=J e_E+M e_S+Gu+Nr.
\end{aligned}
$$

| Field | Purpose |
| --- | --- |
| `state_vars` | Ordered global state basis. |
| `energy_functional` | Identity of the energy or storage functional. |
| `entropy_functional` | Identity of the entropy functional. |
| `J` | Reversible skew-symmetric interconnection matrix. |
| `R` | Symmetric PSD resistance acting on the energy effort in `dissipative_ph`. |
| `M` | Symmetric PSD entropy-gradient mobility in `generic`. |
| `G` | Optional port-incidence matrix. |
| `stoichiometric_N` | Optional species-by-reaction stoichiometric matrix. |
| `ports` | Typed external or internal ports. |
| `formulation` | `unspecified`, `dissipative_ph`, or `generic`; strict validation rejects `unspecified`. |
| `reversible_operator_state_dependence` | Declared constant/state-dependent claim, checked against `poisson_operator`: `zero_field` and `constant_field` reject `state_dependent`; `state_field` requires it. The label alone proves nothing. |
| `jacobi_identity_evidence` | Descriptive evidence reference only; it cannot self-certify executable Jacobi verification. |
| `skew_tol` | Tolerance for skew and symmetry residuals; `None` reads `validation.iphs.skew_tolerance`. |
| `psd_tol` | Tolerance for dissipative eigenvalues; `None` reads `validation.iphs.psd_tolerance`. |
| `degeneracy_tol` | Tolerance for GENERIC degeneracy and entropy production; `None` reads `validation.iphs.degeneracy_tolerance`. |
| `energy_gradient` | Optional default effort `e_E`. |
| `entropy_gradient` | Optional default effort `e_S`. |
| `contribution_count` | Number of runtime-linked assembly contributions. |
| `contribution_sources` | Runtime owner identities for those contributions. |
| `poisson_operator` | `matrix_snapshot` (no field claim), `zero_field`, `constant_field` (`constant_field_exact`), or `state_field` (`sampled_domain_jacobi_residual`: finite-point evidence on the declared samples, plus the bound state at a Level 5 runtime point). |
| `poisson_field` / `poisson_domain` / `jacobi_tol` | Callable `J(x)`, `(m, n)` sample points, and Jacobi residual tolerance for `constant_field`/`state_field`; `jacobi_tol=None` reads `validation.iphs.jacobi_tolerance`. With a bound runtime state, `J(x)` stays on the autograd path, so the drift carries `dJ/dx` and the sensitivity to any parameter the field closes over. |
| `reaction_representation` | `independent` adds `N r` to the drift; `absorbed` requires the whole irreversible term to equal `N r` (`-R e_H` or `M e_S`) and does not add it again. The two are mutually exclusive. |
| `storage_kind` | dPH storage class used only by the Level 4 reaction pairing: `total_energy` requires `e_H^T N r=0`; `free_energy` requires `e_H^T N r<=0`. |

Construction resolves each `None` tolerance through ParameterManager (a
missing key raises) and checks a unique non-empty state basis, matrix ranks
and shapes, finite values, one dtype and device, valid gradients, finite
non-negative `skew_tol`, `psd_tol`, and `degeneracy_tol`, and a finite
positive `jacobi_tol`. Violations raise `TypeError` or `ValueError`.

```python
structure.validate(
    *,
    required_level: Optional[int] = None,
    energy_effort: Optional[torch.Tensor] = None,
    port_input: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    energy_value: Optional[torch.Tensor] = None,
    entropy_value: Optional[torch.Tensor] = None,
    entropy_effort: Optional[torch.Tensor] = None,
    reaction_rates: Optional[torch.Tensor] = None,
) -> Dict[str, Union[float, bool, str]]
```

The base validation checks:

$$
J + J^\mathsf{T} = 0,
\qquad
R = R^\mathsf{T}\succeq0,
\qquad
M = M^\mathsf{T},
\qquad
M \succeq 0.
$$

Strict structure verification and governed process acceptance are separate
checks. `required_level=4` selects `formulation="dissipative_ph"` with `M=0`
and PSD `R`; `required_level=5` selects `formulation="generic"` with `R=0`.
Neither formulation silently adds the other's dissipative term.

For Level 4, explicit `energy_effort` alone supplies the single-point
structural pre-gate. A bound `state=x` and `energy_value=H` derive `dH/dx`
through autograd and can return `runtime_verified_level=4` with
`process_level_status="runtime_point_verified"`. A nonzero reaction also
requires `reaction_rates` and `storage_kind`: `total_energy` requires
`e_H^T N r=0`, while `free_energy` requires `e_H^T N r<=0`. An absorbed
reaction must satisfy `N r=-R e_H` and is not added a second time. These
reaction pairings are implemented at the operator level; no production dPH
reaction process is currently bound. Level 4 runtime points with external
port columns remain unavailable because `port_input` cannot accompany bound
potentials; internal port columns eliminated through `C`/`D` feedback are
accepted.

For Level 5, the reversible operator must use an executable Jacobi basis:
`zero_field` (`J(x)=0`; every local `J` and reversible feedback `C` exactly zero
before conversion and assembly), `constant_field` (constant `J`), or
`state_field` (callable `J(x)` with skew and Jacobi residuals checked on the
declared samples and again at the runtime point: the bound state, or the first
declared sample when explicit efforts are supplied; residuals above
`skew_tol` or `jacobi_tol` are rejected, and the audit reports
`runtime_point_skew_residual`, `runtime_point_jacobi_residual`, and
`runtime_point_jacobi_point`). This is finite-point evidence, not a
continuum proof. `matrix_snapshot`, a constant label, or a text reference does
not prove a global field identity.
Bind the actual state and energy/entropy potentials, and supply reaction rates
for a declared `N`. The runtime audit verifies `J e_S=0`, `M e_E=0`,
nonnegative entropy production, closed-reaction energy/entropy pairings, and,
when absorbed, `M e_S=N r`. It can return `runtime_verified_level=5` and
`process_level_status="runtime_point_verified"`. Level 5 runtime points with
declared port columns remain unavailable; no production process binds a
nonzero field.

A runtime-point result alone is not a process certificate. Pass it as
`runtime_point_audit` to the existing certificate owner together with the
`governed_execution` witness: `entry="FunctorBase.__call__"`, the executed
Functor, committed non-diagnostic variables, and a positive accepted step
count. Missing witnesses retain `runtime_binding="descriptive"` and the
`process_bound_runtime_unavailable [N]` issue; inconsistent witnesses fail.
The witness is a mapping the caller builds from an executed step; the
certificate owner validates its fields and does not query StateManager
provenance, so it is inspectable evidence, not an automatically issued
certificate. A runtime point audits the continuous-time drift at the bound
pre-step state, not the discrete time step.
The bounded production selections are closed one-dimensional
`FickianDiffusion(strict_runtime_point=True)` at Level 4 and the closed
constant-volume NO+O3 selection
`ChemistryFunctor(physical_boundary="closed_constant_volume")` at Level 5.
Their runtime-point certificates record these selections in the certificate
field `constructor_selection`. Other configurations do not inherit these
claims. The Fickian selection requires one unbatched one-dimensional state
with at least two cells, finite positive diffusivities, and a zero-flux or
periodic boundary; its storage is a quadratic norm in kg2 m-6, not joules, and
a nonzero `source_term` is rejected before integration, because the audited
drift is source-free. The NO+O3 selection requires positive amounts, positive
affinity, a temperature inside the declared NASA7 interval, fixed volume, no
heat or work exchange, and forward mass action. The main certificates of both
classes remain Level 3.

Invalid levels and failed structural checks raise `ValueError`.

```python
structure.canonical_drift(
    *,
    state: Optional[torch.Tensor] = None,
    energy_value: Optional[torch.Tensor] = None,
    entropy_value: Optional[torch.Tensor] = None,
    energy_effort: Optional[torch.Tensor] = None,
    entropy_effort: Optional[torch.Tensor] = None,
    port_input: Optional[torch.Tensor] = None,
    reaction_rates: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

The method uses either scalar potentials bound to `state` through autograd
(`energy_value` for `dissipative_ph`; `energy_value` and `entropy_value` for
`generic`; `state.requires_grad=True`) or supplied/stored efforts; the two
routes are mutually exclusive.
The compatibility parameter name `energy_effort` denotes $e_H=\nabla H$ for
`dissipative_ph` and $e_E=\nabla E$ for `generic`; this name does not merge the
two formulations or authorize a mixed drift.
Declared `G` requires a matching `port_input`; declared `stoichiometric_N`
requires matching `reaction_rates`. Providing an input for an undeclared
operator, mismatched shapes, or non-finite values raises `ValueError`.

### `ReactionNetworkAdvance`

```python
ReactionNetworkAdvance(
    name: str,
    species: Tuple[str, ...],
    reactions: Tuple[str, ...],
    stoichiometric_N: torch.Tensor,
    rates: torch.Tensor,
    state_derivative: torch.Tensor,
    state_delta: Optional[torch.Tensor] = None,
)
```

`conservation_audit`, `thermodynamic_audit`, and `open_system_audit` are
non-constructor attributes, `None` by default, that
`advance_reaction_network()` fills.

| Field | Purpose |
| --- | --- |
| `name` | Reaction-network identity. |
| `species` | Ordered rows of `stoichiometric_N`. |
| `reactions` | Ordered columns and reaction-rate channels. |
| `stoichiometric_N` | Signed stoichiometric matrix `N_h`. |
| `rates` | Evaluated reaction rates `r_h`. |
| `state_derivative` | Computed $N_h r_h$. |
| `state_delta` | Optional `state_derivative * dt`. |
| `conservation_audit` | Result or declaration status for $C_h N_h = 0$. |
| `thermodynamic_audit` | Optional single-point aggregate affinity-power audit. |
| `open_system_audit` | Optional single-point audit of $C_h q_{\mathrm{open}}$ against the declared ledger rate. |

```python
advance.species_rate(species_name: str) -> torch.Tensor
```

Returns one species row from `state_derivative`; an unknown name raises
`KeyError`.

```python
advance.validate_conservation(
    conservation_matrix: Union[torch.Tensor, Any],
    *,
    tolerance: Optional[float] = None,
) -> Dict[str, Union[float, bool, str]]
```

`tolerance=None` reads `validation.iphs.reaction_conservation_tolerance`.
Returns an audit with `passed`, `max_abs_residual`, and `tolerance`.
A failed conservation condition is represented by `passed=False`; this method
does not raise merely because the residual exceeds the tolerance. Invalid
matrix rank or species width raises `ValueError`.

### `advance_reaction_network()`

```python
advance_reaction_network(
    *,
    name: str,
    species: Tuple[str, ...],
    reactions: Tuple[str, ...],
    stoichiometric_N: Union[torch.Tensor, Any],
    rates: Union[torch.Tensor, Iterable[Any]],
    dt: Optional[Union[float, torch.Tensor]] = None,
    conservation_matrix: Optional[Union[torch.Tensor, Any]] = None,
    conservation_tolerance: Optional[float] = None,
    strict_conservation: bool = True,
    require_conservation_matrix: bool = False,
    open_system_route: Optional[str] = None,
    affinities: Optional[Union[torch.Tensor, Any]] = None,
    affinity_sign_convention: Optional[str] = None,
    thermodynamic_rtol: Optional[float] = None,
    open_exchange: Optional[Union[torch.Tensor, Any]] = None,
    open_ledger_rate: Optional[Union[torch.Tensor, Any]] = None,
    require_thermodynamic_point_audit: bool = False,
) -> ReactionNetworkAdvance
```

The function evaluates:

$$
\dot{x}_h = N_h r_h,
\qquad
\Delta x_h = \dot{x}_h \Delta t.
$$

| Parameter | Purpose |
| --- | --- |
| `name` | Non-empty reaction-network identity. |
| `species` | Non-empty ordered output basis. |
| `reactions` | Non-empty ordered reaction basis. |
| `stoichiometric_N` | Matrix with one row per species and one column per reaction. |
| `rates` | Tensor or iterable with reaction channel on axis zero. |
| `dt` | Optional time increment used to create `state_delta`. |
| `conservation_matrix` | Optional ledger matrix `C_h`. |
| `conservation_tolerance` | Maximum accepted absolute residual in $C_h N_h$; `None` reads `validation.iphs.reaction_conservation_tolerance`. |
| `strict_conservation` | Reject a declared but failing conservation matrix. |
| `require_conservation_matrix` | Require `C_h` unless an open-system route is declared. |
| `open_system_route` | Explicit route for a non-closed state map. |
| `affinities` | Signed reaction affinities with exactly the shape of `rates`; no broadcasting. |
| `affinity_sign_convention` | Must be `affinity_times_rate_nonnegative` when the point audit runs. |
| `thermodynamic_rtol` | Relative tolerance of the point audits; `None` reads `validation.iphs.reaction_thermodynamic_rtol`. Values below the dtype reduction floor raise, so a float32 point audit needs an explicit larger value. |
| `open_exchange` | Separately routed exchange $q_{\mathrm{open}}$ with the shape of the state derivative. |
| `open_ledger_rate` | Declared ledger rate compared with $C_h q_{\mathrm{open}}$. |
| `require_thermodynamic_point_audit` | Require affinities, a conservation matrix, and complete open fields. |

Accepted open-system routes are `boundary_export`, `managed_export`,
`diagnostic_residual`, `port`, and `observation`.

Invalid names, empty bases, matrix/rate shape mismatches, and invalid route
names raise `ValueError`. If a conservation matrix is supplied and fails while
`strict_conservation=True`, the function raises `ValueError`. Without a
matrix, `require_conservation_matrix=True` raises only when no open-system
route is declared.

Supplying any point-audit input requires a non-empty `conservation_matrix`
whose $C_hN_h=0$ check passes, independent of `strict_conservation`. The
optional point audit checks the aggregate affinity power
$\sum_jA_jr_j\ge-\mathrm{rtol}\sum_j\lvert A_jr_j\rvert$ under
`affinity_sign_convention="affinity_times_rate_nonnegative"`; this is a power
check, and no temperature or entropy rate is inferred. With an open route it
checks $C_hq_{\mathrm{open}}=$ `open_ledger_rate` without adding
$q_{\mathrm{open}}$ to the derivative; the time-integrated no-double-count
proof remains `[N]`. Open fields without an open route raise `ValueError`.
Results are stored in `thermodynamic_audit` and `open_system_audit`.

This function is a low-level compatibility helper: an allowed open route may
replace `conservation_matrix` only when the caller explicitly selects that
policy. Graph strict `reaction_network_check` is stronger. Its closed and open
payloads both provide a non-empty `C_h` and prove `C_h N_h=0`; an open payload
then adds route, exchange vector, ledger-rate vector, units, and
`C_h q_open=ledger_rate`. Payload-local `strict=false` cannot weaken that
outer Graph contract.

### `assemble_iphs_structure()`

```python
assemble_iphs_structure(
    *,
    name: str,
    state_vars: Tuple[str, ...],
    energy_functional: str,
    entropy_functional: str,
    contributions: Iterable[Any],
    formulation: Literal["unspecified", "dissipative_ph", "generic"] = "unspecified",
    poisson_operator: Literal[
        "matrix_snapshot", "zero_field", "constant_field", "state_field"
    ],
    reaction_representation: Literal["independent", "absorbed"],
    reversible_operator_state_dependence: Literal[
        "unspecified", "constant", "state_dependent"
    ] = "unspecified",
    jacobi_identity_evidence: str = "",
    poisson_field: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    poisson_domain: Optional[torch.Tensor] = None,
    jacobi_tol: Optional[float] = None,
    port_interconnection_C: Optional[Any] = None,
    port_dissipation_D: Optional[Any] = None,
    validate: bool = False,
    strict: bool = False,
    required_level: Optional[int] = None,
    skew_tol: Optional[float] = None,
    psd_tol: Optional[float] = None,
    degeneracy_tol: Optional[float] = None,
) -> IPHSStructureSpec
```

Contributions may be mappings or objects implementing
`contribute_iphs_structure()`. Local operators are embedded into the declared
global state basis:

$$
J = \sum_i \iota_i J_i \iota_i^\mathsf{T} + G C G^\mathsf{T},
$$

$$
R = \sum_i \iota_i R_i \iota_i^\mathsf{T} + G D G^\mathsf{T},
\qquad
M = \sum_i \iota_i M_i \iota_i^\mathsf{T}.
$$

`port_interconnection_C` must be skew-symmetric.
`port_dissipation_D` must be symmetric positive semidefinite and contributes
only to the energy resistance `R`, producing `-G D G^T e_H` in the
dissipative-pH drift. It never augments GENERIC `M`; `formulation="generic"`
together with `port_dissipation_D` raises `ValueError`. Strict assembly
requires at least one contribution and a traceable runtime owner or source
node. `required_level` is applied only when `validate=True`. The assembler
resolves `None` for `skew_tol`, `psd_tol`, and `degeneracy_tol` from
`validation.iphs.skew_tolerance`, `validation.iphs.psd_tolerance`, and
`validation.iphs.degeneracy_tolerance` before it checks `C` and `D`, and
passes the Poisson-field arguments unchanged to `IPHSStructureSpec`, where
`jacobi_tol=None` reads `validation.iphs.jacobi_tolerance`.

Contributions are embedded by exact `state_vars` name: contributions naming the
same variable act on one coordinate, and different names are never aliased or
unit-converted. Declared energy/entropy gradient contributions are summed
coordinatewise; the assembler constructs no `H`, `E`, or `S` functional,
effort, affinity, or shared-state equivalence. It has no `storage_kind`
argument, so the Level 4 reaction pairing needs a directly constructed
`IPHSStructureSpec`.

Every argument is keyword-only; `poisson_operator` and
`reaction_representation` have no default and are required. The
graph-metadata compile path (`iphs_assembly` in graph metadata) forwards
`poisson_operator` and `reaction_representation`, both required (a
declaration without one raises `ValueError` naming the missing field), and,
when declared, `poisson_field`, `poisson_domain`,
`jacobi_tol`, `skew_tol`, `psd_tol`, and `degeneracy_tol`. `poisson_field`
must be an importable `package.module:attribute` reference, resolved at
compile time to a callable; `poisson_domain` must be a non-empty list of
numeric state vectors of equal length and becomes a float64 `(m, n)` tensor.
These three keys are accepted only with `poisson_operator` `constant_field` or
`state_field`. A declared tolerance must be a number (a null, string, bool, or
list is refused); its range is checked by the structure. An omitted tolerance
selects the ParameterManager default. A refused declaration makes the IPHS
assembly audit report `passed=False` with the reason. A `state_field`
structure can therefore be compiled from graph metadata.

Invalid contributions raise `TypeError`; duplicate or unknown state variables,
shape mismatches, missing `G`, or invalid port matrices raise `ValueError`.

## IPHS equivalence certificates

Stable imports:

```python
from relic.computation.operators import (
    IPHS_LEVEL_DEFINITIONS,
    IPHSEquivalenceCertificate,
    load_iphs_equivalence_certificate,
)
```

Implementation:
`relic.computation.operators.iphs_certificate.IPHSEquivalenceCertificate`

The certificate is the only owner of the IPHS standardization level. In the
packaged repository, all `90` concrete scientific `Functor` classes (`37`
`BiologicalProcessFunctor`, `27` `TransportMechanismFunctor`, `14` `PhysicalFieldFunctor`, and
`12` `ScaleCouplingFunctor`; `0` uncertified) carry validated YAML certificates,
totaling `92` certificates: `90` primary class certificates distributed across
Levels `0` through `5` as `4 / 15 / 39 / 32 / 0 / 0` (including the two
certified Level `2` `[I]` runtime-bound stochastic Functors
`OrnsteinUhlenbeckTemperatureFunctor` and `MarkedPoissonRainfallFunctor`),
plus `2` constructor-scoped runtime-point certificates (`constructor_selection`
at Levels `4` and `5`) yielding `4 / 15 / 39 / 32 / 1 / 1` across all `92`
certificates.

| Level | Name | Claim boundary |
| --- | --- | --- |
| 0 | `ordinary_empirical` | No IPHS structural claim |
| 1 | `iphs_compatible` | Compatible conservation, dissipation, port, or DEC metadata |
| 2 | `iphs_wrapped` | Canonical runtime wraps an existing tendency with an IPHS operator |
| 3 | `iphs_structured` | Forms, ports, Hodge, or `J/R/M/G/N` assembly evidence is declared |
| 4 | `strict_iphs` | `dissipative_ph`: storage effort, `M=0`, PSD `R`, runtime contribution, port power, and energy inequality |
| 5 | `strict_iphs_generic` | `generic`: `R=0`, energy/entropy gradients, both degeneracies, nonnegative entropy production, and an executable Jacobi basis (`canonical_zero_field`, `constant_field_exact`, or `sampled_domain_jacobi_residual` with finite-point evidence on the declared samples plus the evaluated runtime point) |

Evidence keys become stronger across the taxonomy, but the level-4 and
level-5 governing equations are formulation-specific and mutually exclusive;
level 5 does not add `M` to a running level-4 `R` equation.
The default `matrix_snapshot` cannot prove a global reversible-field
identity. The `zero_field`, `constant_field`, and `state_field` profiles have
executable Jacobi bases (the last as finite-point evidence: the declared
samples plus the evaluated runtime point). The certificate's
`generic_degeneracy` evidence pairs `reversible_operator_state_dependence`
`zero_field` with `jacobi_identity_basis=canonical_zero_field`, `constant` with
`constant_field_exact`, and `state_dependent` with
`sampled_domain_jacobi_residual` plus a positive `jacobi_sample_count` and a
positive finite `jacobi_tolerance`. Level 5 process acceptance also
requires the runtime-point audit and governed-execution witness described
above; for the sampled basis that audit must report a finite
`runtime_point_skew_residual` and a finite `runtime_point_jacobi_residual`
within `jacobi_tolerance`. No production process binds a nonzero field;
strict runtime points with external port columns (Level 4) or any port columns
(Level 5) remain `[N]`.

### `IPHSEquivalenceCertificate`

```python
IPHSEquivalenceCertificate(
    module_name: str,
    legacy_expression: str,
    iphs_expression: str,
    input_variables: Tuple[str, ...],
    output_variables: Tuple[str, ...],
    parameter_keys: Tuple[str, ...],
    parameter_free: bool = False,
    formulation: Literal["unspecified", "dissipative_ph", "generic"] = "unspecified",
    process_id: str = "",
    canonical_role: str = "",
    iphs_level: Optional[int] = None,
    iphs_level_evidence: Mapping[str, Any] = {},
    theory_document: str = "",
    theory_document_status: str = "",
    literature_formula: str = "",
    state_map: Mapping[str, Any] = {},
    parameter_map: Mapping[str, Any] = {},
    operator_map: Mapping[str, Any] = {},
    state_advancement_map: Mapping[str, Any] = {},
    unit_map: Mapping[str, Any] = {},
    input_output_map: Mapping[str, Any] = {},
    composition_ports: Tuple[str, ...] = (),
    references: Tuple[str, ...] = (),
    validation_tests: Tuple[str, ...] = (),
    runtime_replacement_map: Mapping[str, Any] = {},
    parameter_management_map: Mapping[str, Any] = {},
    composition_management_map: Mapping[str, Any] = {},
    dependency_interconnection_map: Mapping[str, Any] = {},
    overflow_routing_map: Mapping[str, Any] = {},
    conservation_map: Mapping[str, Any] = {},
    entropy_audit: Mapping[str, Any] = {},
    stochastic_sdae_map: Mapping[str, Any] = {},
    noise_covariance_map: Mapping[str, Any] = {},
    conservative_noise_flux_map: Mapping[str, Any] = {},
    descriptor_constraint_map: Mapping[str, Any] = {},
    jump_reset_map: Mapping[str, Any] = {},
    stochastic_reproducibility_map: Mapping[str, Any] = {},
    equivalence_type: str = "",
    interface_map: Mapping[str, Any] = {},
    certificate_version: str = "1.0",
    constructor_selection: Mapping[str, Any] = {},
)
```

The displayed mappings and tuples are created by dataclass factories and are
not shared between instances.

### Field groups

| Group | Fields | Purpose |
| --- | --- | --- |
| Identity | `module_name`, `process_id`, `canonical_role` | Identify the scientific process and its role |
| Equations | `legacy_expression`, `literature_formula`, `iphs_expression`, `formulation` | Relate the documented process to the executable IPHS representation; `formulation` selects `dissipative_ph` or `generic` |
| Interface | `input_variables`, `output_variables`, `interface_map`, `input_output_map` | Prove compatible runtime inputs and outputs |
| Parameters | `parameter_keys`, `parameter_free`, `parameter_map`, `parameter_management_map` | Bind every scientific parameter to ParameterManager metadata |
| Structure | `state_map`, `operator_map`, `state_advancement_map`, `unit_map` | Declare state, operators, advancement, and units |
| Composition | `composition_ports`, `composition_management_map`, `dependency_interconnection_map` | Bind process composition to existing ports and dependency owners |
| Closure | `overflow_routing_map`, `conservation_map`, `entropy_audit` | Account for clipping, export, conservation, and entropy behavior |
| Runtime replacement | `runtime_replacement_map` | Identify the executable owner and confirm whether the legacy path was replaced |
| Stochastic structure | stochastic, covariance, flux, constraint, jump, and reproducibility maps | Bind declared SDAE or jump behavior to current stochastic owners |
| Evidence | `iphs_level`, `iphs_level_evidence`, `references`, `validation_tests` | Bound the level claim with inspectable evidence |
| Runtime-point selection | `constructor_selection` | Map constructor keywords of the process class to JSON scalar values; a non-empty mapping marks a constructor-scoped Level 4/5 runtime-point certificate |
| Version | `certificate_version` | Version the certificate schema independently of the package release |

Allowed canonical roles are `transport`, `reaction`, `field`,
`scale_coupling`, `port`, and `observation`.

Allowed equivalence types are `exact`, `thermodynamic_embedding`,
`port_equivalent`, `diagnostic`, and `reaction_network`.

An observation role or diagnostic equivalence cannot claim above level 1.

`constructor_selection` is empty for a main process certificate. A non-empty
mapping requires `iphs_level` 4 or 5, and `process_id` must resolve to an
importable process class. Each key must be an identifier that is not a Python
keyword and a keyword parameter of that class's constructor; each value must
be a JSON scalar (null, bool, finite number, or string). Violations appear as
validation issues. `from_mapping()` raises `ValueError` for a non-mapping
value and treats an absent or null value as empty. The `validate()` report
includes the selection.

### Loading and normalization

```python
IPHSEquivalenceCertificate.from_mapping(
    payload: Mapping[str, Any],
) -> IPHSEquivalenceCertificate

load_iphs_equivalence_certificate(
    source: Union[Mapping[str, Any], str, Path],
) -> IPHSEquivalenceCertificate
```

`from_mapping()` accepts current field names and selected legacy YAML aliases.
The loader accepts a mapping, JSON file, or YAML file. A file payload that
does not resolve to a mapping raises `TypeError`; file-system and parser
errors propagate.

### `validate()`

```python
certificate.validate(
    *,
    strict: bool = True,
    required_level: Optional[int] = None,
    parameter_resolver: Optional[Callable[[str], Any]] = None,
    runtime_point_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]
```

The method returns an audit report. Incomplete certificate evidence normally
produces `passed=False` and an `issues` list without raising.

Base validation covers identity, role, level, evidence keys, theory document,
formulas, interfaces, parameters, references, validation evidence, and
equivalence type.

Strict validation also requires state, operator, advancement, unit,
interface, dependency, overflow, conservation, and entropy maps. It verifies
ParameterManager coverage, composition ownership, runtime replacement
declarations, and any required stochastic maps. A reaction certificate must
declare explicit `N_h` and `r_h` use in both operator and state-advancement
evidence.

For level 4 or 5, structured evidence may cause the validator to obtain
`get_global_manager().get_parameter` when no resolver is supplied. Each
strict parameter key must resolve to complete `ParameterSpec` metadata and
must not contain a wildcard.

## Python Functor extensions

Stable public imports:

```python
from relic.extensions import (
    ExtensionInfo,
    ExtensionManifest,
    LoadedFunctors,
    ValidationResult,
    discover_extensions,
    get_loaded_extensions,
    load_and_register_extensions,
    load_extension_parameters,
    load_extensions,
    load_single_extension,
    register_extension_parameter_source,
    register_with_orchestration,
    save_extension_parameters,
    unload_extension,
    unload_from_runtime,
    validate_extension,
)
```

Schema components and explicit error types:

```python
from relic.extensions.protocol import (
    DependencyDeclaration,
    FunctorDeclaration,
    ManifestError,
    ParameterFileDeclaration,
    parse_manifest,
)

from relic.extensions.loader import (
    ExtensionLoadError,
    ExtensionValidationError,
)

from relic.extensions.validator import ExtensionValidationResult
```

`ValidationResult` exported by `relic.extensions` is an alias for
`ExtensionValidationResult`. New architecture documentation should use the
domain-prefixed class name.

### `FunctorDeclaration`

```python
FunctorDeclaration(
    *,
    class_path: str,
    bridge: str,
    category: str,
    execution_group: str,
    config: Dict[str, Any] = {},
)
```

The manifest key is `class`; Pydantic exposes it as `class_path`.

| Field | Purpose | Validation |
| --- | --- | --- |
| `class` | Dotted Python class path. | Must contain a module and class name. |
| `bridge` | Scientific bridge contract. | Must name one of the four bridge classes. |
| `category` | Scientific category. | Must be `biological_process`, `physical_field`, `transport_mechanism`, or `scale_coupling`. |
| `execution_group` | Simulator cadence group. | Required; a manifest without it raises `ManifestError`. Must be `fast`, `medium`, or `slow`. |
| `config` | Constructor configuration passed to the Functor. | Extra keys are preserved within this mapping. |

Allowed bridge classes are:

- `BiologicalProcessFunctor`
- `PhysicalFieldFunctor`
- `TransportMechanismFunctor`
- `ScaleCouplingFunctor`

### `ParameterFileDeclaration`

```python
ParameterFileDeclaration(
    *,
    path: str,
    namespace_prefix: Optional[str] = None,
)
```

`path` identifies an extension parameter source and must not be empty.
`namespace_prefix` optionally places loaded `ParameterSpec` keys beneath one
namespace.

### `DependencyDeclaration`

```python
DependencyDeclaration(
    *,
    required: List[str] = [],
    optional: List[str] = [],
)
```

These fields are declarative ordering hints. The current loader does not
install dependencies or derive runtime ordering from them. A Functor's
runtime dependencies remain governed by `get_dependencies()`, metadata,
GraphIR, ports, and simulator registration.

### `ExtensionManifest`

```python
ExtensionManifest(
    *,
    name: str,
    version: str,
    author: str = "",
    description: str = "",
    min_relic_version: Optional[str] = None,
    calibrated_output: Optional[str] = None,
    functors: List[FunctorDeclaration],
    parameters: List[ParameterFileDeclaration] = [],
    dependencies: DependencyDeclaration = DependencyDeclaration(),
)
```

| Field | Purpose |
| --- | --- |
| `name` | Non-empty extension identity. |
| `version` | Extension schema version in strict `X.Y.Z` form. |
| `author` | Optional author or organization. |
| `description` | Optional extension summary. |
| `min_relic_version` | Declared minimum RELIC version in strict `X.Y.Z` form. |
| `calibrated_output` | Optional extension-local calibrated output directory. |
| `functors` | One or more Functor declarations. |
| `parameters` | Parameter files loaded into the existing parameter system. |
| `dependencies` | Declarative required and optional dependencies. |

The manifest ignores unknown top-level fields. It raises Pydantic validation
errors for invalid construction. `parse_manifest()` translates YAML and schema
errors into `ManifestError`.

Current implementation limits:

- `min_relic_version` is format-validated but is not compared with the running
  RELIC version by the loader.
- `DependencyDeclaration` is not a package installer or a second dependency
  scheduler.

### `parse_manifest()`

```python
parse_manifest(path: Path) -> ExtensionManifest
```

`path` may be an extension root or `relic_extension.yaml`. Missing files raise
`FileNotFoundError`. Invalid YAML, read failures, non-mapping payloads, and
schema failures raise `ManifestError`.

### `ExtensionInfo`

```python
ExtensionInfo(
    name: str,
    root_path: Path,
    manifest: ExtensionManifest,
    source: str,
)
```

This frozen discovery result binds one validated manifest to its root.
`source` is `directory` or `entry_point`.

### Discovery

```python
discover_extensions(
    extra_dirs: Optional[List[Path]] = None,
    use_entry_points: bool = True,
    use_env_dirs: bool = True,
) -> List[ExtensionInfo]
```

Discovery order is caller-provided directories, directories in
`RELIC_EXTENSION_DIRS`, and Python entry points in the `relic.extensions`
group. The first extension with a given name wins. An invalid manifest raises
`ManifestError` naming the manifest (for an entry point, also the entry
point); an entry point that cannot be loaded or whose extension root cannot be
located raises.

### `ExtensionValidationResult` and validation

```python
ExtensionValidationResult(
    valid: bool,
    errors: List[str] = [],
    warnings: List[str] = [],
)

validate_extension(
    info: ExtensionInfo,
) -> ExtensionValidationResult
```

Validation checks parameter-file presence, class importability, bridge
inheritance, the `forward(self, state, time, dt, **kwargs)` contract,
`FunctorMetadata`, category agreement, declared variables, and
`get_dependencies()`.

Ordinary extension defects are accumulated in `errors`; missing or unknown
spatial-scale and variable-kind metadata may appear as warnings.

### Loading

```python
load_extensions(
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
    discovered_infos: Optional[List[ExtensionInfo]] = None,
) -> LoadedFunctors

load_single_extension(
    extension_path: Path,
    validate: bool = True,
    strict: bool = True,
    populate_core_registry: bool = False,
) -> LoadedFunctors
```

Both return a `LoadedFunctors` mapping (instance name to instance) with
`skipped_extensions`. A constructor error, including `TypeError`, is never
retried without the declared config.

| Parameter | Purpose |
| --- | --- |
| `extra_dirs` | Additional extension search roots. |
| `validate` | Validate declarations before import and construction. |
| `strict` | Default True: a validation failure raises `ExtensionValidationError` before anything is loaded and a duplicate instance name raises `ExtensionLoadError`. `strict=False` loads the valid extensions and lists each skipped extension or duplicate with its reason in `skipped_extensions`. |
| `populate_core_registry` | Optionally register loaded Functor classes with the existing scientific registry. |
| `discovered_infos` | Reuse an already discovered set without repeating discovery. |

Import, construction, and core-registry failures raise `ExtensionLoadError` in
every mode.

The loader's module-level maps are implementation caches. They are not a
public replacement for `NodeRegistry` or simulator registration.

### Runtime integration

```python
load_and_register_extensions(
    simulator: RELICSimulator,
    *,
    parameter_manager: Any,
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
) -> Dict[str, Any]
```

`parameter_manager` is required: RELICSimulator owns no parameter manager, so
the caller passes the manager that receives the extension parameters
(normally `get_global_manager()`), and every loaded Functor is bound to that
manager before `simulator.register_functor()`. Extensions are validated
first; parameters are loaded only for extensions that passed validation. The
summary contains load, registration, and parameter counts and
`skipped_extensions` (always empty in strict mode, which raises instead); a
parameter-source or registration failure propagates.

```python
register_with_orchestration(
    registry: NodeRegistry,
    extra_dirs: Optional[List[Path]] = None,
    validate: bool = True,
    strict: bool = True,
) -> List[str]
```

Registers loaded instances with the existing `NodeRegistry` and returns node
type identifiers.

### Extension parameter integration

```python
register_extension_parameter_source(
    manager: Any,
    extension_root: Path,
    output_dir: Optional[Path] = None,
) -> Path

load_extension_parameters(
    manager: Any,
    extension_info: ExtensionInfo,
) -> Dict[str, Any]

save_extension_parameters(
    manager: Any,
    extension_info: ExtensionInfo,
    keys: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
) -> Path
```

These functions reuse `manager.database.add_source()` and
`manager.register_parameter()`. They do not create an extension-specific
ParameterManager. Missing source files raise `FileNotFoundError`.
`save_extension_parameters()` raises `ValueError` when no matching
`ParameterSpec` can be selected.

### Unloading

```python
get_loaded_extensions() -> Dict[str, List[str]]

unload_extension(
    extension_name: str,
    *,
    simulator: Optional[Any] = None,
    registry: Optional[Any] = None,
    strict: bool = False,
    native_drain_timeout: Optional[float] = None,
) -> Dict[str, Any]

unload_from_runtime(
    simulator: RELICSimulator,
    extension_name: str,
    registry: Optional[NodeRegistry] = None,
    strict: bool = False,
    native_drain_timeout: Optional[float] = None,
) -> Dict[str, Any]
```

The unload report contains removed Functors, removed node types, and missing
Functor names. Unloading an extension that is not loaded raises `KeyError` in
every mode.

## C ABI coupling plugins

Stable import:

```python
from relic.coupling.plugin import PluginFunctorAdapter, PluginHost
```

The ABI is declared by:

```text
relic\coupling\plugin\coupler_plugin.h
```

A plugin must export:

```c
int plugin_probe(char* manifest_json_out, size_t buf_len);

int plugin_negotiate(
    const char* host_capabilities_json,
    char* agreed_manifest_out,
    size_t buf_len
);

int plugin_exchange(
    void* handle,
    const char* port_name,
    void* buffer_ptr,
    size_t buffer_len,
    double timestamp_seconds
);
```

Return code zero means success.

### Coupling manifest contract

Both probe and negotiation return a JSON object. Required top-level fields are
`name`, `interface_version`, `metadata`, and `ports`.

Each port declares:

| Field | Accepted values and purpose |
| --- | --- |
| `name` | Non-empty state or coupling-port name |
| `direction` | `in`, `out`, or `inout` |
| `type_id` | `INT32`, `INT64`, `F32`, `F64`, `STR`, or `BYTES` |
| `update_mode` | `STATE`, `INCREMENT`, or `FLUX` |
| `shape_hint` | Optional dimensions used for buffer allocation and scale inference |
| `units` | Optional physical units checked against SSOT when available |
| `per_timestep` | Whether the buffer is already integrated over the time step |

Manifest validation uses the existing coupling schema, `SSOTCatalog`, and
port-dimension checks. It does not define a second port schema.

### `PluginHost`

Implementation:
`relic.coupling.plugin.host.PluginHost`

```python
PluginHost(
    path: Optional[str] = None,
    manifest_buffer_size: int = 65536,
)
```

| Parameter | Purpose |
| --- | --- |
| `path` | Optional shared-library path loaded during construction. |
| `manifest_buffer_size` | Byte capacity for probe and negotiation JSON. |

```python
host.load(path: str) -> Dict[str, Any]
```

Loads the library, binds the three required symbols, validates its declared
interface version, and returns a lifecycle receipt including ABI metadata.

- A missing file raises `FileNotFoundError`.
- A native loading failure propagates from `ctypes`.
- Missing required symbols raise `RuntimeError`.
- An incompatible declared interface version raises `PluginABIError` and
  releases the library before exchange.
- Loading an already loaded or retired host raises `PluginLifecycleError`.

```python
host.retire() -> Dict[str, Any]
host.drain(timeout: Optional[float] = None) -> bool
host.unload(*, drain_timeout: Optional[float] = None) -> Dict[str, Any]
host.reload() -> Dict[str, Any]
```

Retirement rejects new native calls while calls already in flight continue.
`drain` waits for their completion. `unload` releases the library only after
draining; a timeout returns `released=False` and `drained=False`, retaining a
retired host whose unload can be retried. A repeated unload after successful
release is rejected. `reload` drains and reopens the same path. Read-only
`state`, `loaded`, `inflight_calls`, `completed_calls`, `abi`, and `receipts`
expose the current lifecycle and its outcomes. Returning from a C entry point
is the completion signal available to this ABI.

```python
host.probe() -> Dict[str, Any]
host.negotiate(
    host_capabilities: Dict[str, Any],
) -> Dict[str, Any]
```

Both require a loaded plugin. A nonzero plugin return code, empty JSON, or an
unloaded host raises `RuntimeError`. Invalid JSON propagates its parser error;
a non-object JSON payload raises `ValueError`.

```python
host.exchange(
    port_name: str,
    buffer: Any,
    timestamp: float,
) -> int
```

Supported buffers are:

- `numpy.ndarray`
- `torch.Tensor`
- `memoryview`
- `bytearray`

The method returns zero on success. Unsupported buffer types raise
`TypeError`; an empty buffer raises `ValueError`; a nonzero plugin return code
raises `RuntimeError`.

### `PluginFunctorAdapter`

Implementation:
`relic.coupling.plugin.adapter.PluginFunctorAdapter`

```python
PluginFunctorAdapter(
    name: str,
    plugin_path: str,
    config: Optional[Dict[str, Any]] = None,
    host: Optional[PluginHost] = None,
    metadata: Optional[FunctorMetadata] = None,
    **kwargs: Any,
)
```

| Parameter | Purpose |
| --- | --- |
| `name` | Functor instance name. |
| `plugin_path` | Native shared-library path. |
| `config` | Adapter configuration, including an optional `spatial_scale`. `source_scale`, `target_scale`, and `coupling_type` (`spatial`, `temporal`, `hierarchical`, or `stochastic`) are required: a plugin coupling declares the scales it couples, and a missing key raises `KeyError`. |
| `host` | Optional injected `PluginHost`, useful for controlled hosting. |
| `metadata` | Optional initial `FunctorMetadata`; negotiated metadata replaces it after initialization. |
| `kwargs` | Additional `FunctorBase` constructor arguments. |

```python
adapter.initialize(
    host_capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]
```

Loads the library, validates the probe manifest, negotiates host capabilities,
validates the agreed manifest, stores ports, and rebuilds Functor metadata.
Invalid manifests or unresolved spatial scale raise `ValueError`; native host
failures propagate.

```python
adapter.get_dependencies() -> List[str]
```

The adapter does not override `get_dependencies()`. It inherits the
`ScaleCouplingFunctor` bridge default, which returns the names declared on
the instance through `config["dependencies"]`, `set_dependencies(...)` or
`add_dependency(...)`, in declaration order and without duplicates. The list
is empty only when nothing is declared. When `GraphExecutor` builds a node
from a graph step that has input links, it calls `set_dependencies(...)`
with the sorted source-node names of those links. This replaces the earlier
instance list, so the graph links become the adapter's dependencies.

```python
adapter.forward(
    state: Dict[str, Any],
    time: float,
    dt: float,
    **kwargs: Any,
) -> Dict[str, UpdateItem]
```

The adapter initializes lazily, exchanges each declared port, and lowers
outputs with the existing `FunctorBase.wrap_update()` path.

Update-mode mapping:

| Plugin mode | RELIC mode |
| --- | --- |
| `STATE` | `UpdateMode.REPLACE` |
| `INCREMENT` | `UpdateMode.ACCUMULATE` |
| `FLUX` | `UpdateMode.FLUX` |

Missing required input ports raise `KeyError`. Host, manifest, conversion, and
exchange errors propagate. Ports with no name or an unsupported direction are
skipped by the current compatibility path.

`PluginFunctorAdapter` is coupling infrastructure and subclasses the
`ScaleCouplingFunctor` bridge class, which reaches `FunctorBase`. It is not a
template for Python scientific modules. A Python scientific extension must
still follow:

```text
Scientific abstraction -> bridge class -> FunctorBase -> ParameterManager
```

## Anti-redundancy boundary

The APIs on this page compose with current owners:

- DEC views use existing SSOT and state contracts.
- IPHS assembly uses declared runtime contributions and existing ports.
- IPHS levels are owned only by `IPHSEquivalenceCertificate`.
- Python extensions register through the existing simulator and
  `NodeRegistry`.
- Extension parameters load through the existing
  `UnifiedParameterManager`.
- C ABI plugins return existing `UpdateItem` objects through
  `PluginFunctorAdapter`.

Do not add another simulator, StateManager, ParameterManager,
ValidationRegistry, NodeRegistry, SSOTCatalog, port schema, extension registry,
IPHS level registry, or compiler.
