# General Transport Scientific Classes

This reference covers all 4 certificate-backed scientific classes in `relic.scientific.transport`: 4 executable concrete Functors, 0 fail-closed unavailable boundaries, and 0 abstract contracts. Each entry is generated from the live class, its IPHS certificate, State Contract Model, and ParameterManager metadata. Fail-closed unavailable classes are not instantiated during documentation generation.

The tables distinguish constructor controls, runtime variables, and ParameterManager scientific parameters. Values shown as package defaults belong to the documented RELIC architecture version and must not be copied as hardcoded constants into scientific code.

Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their exact current YAML entries. Every other declared parameter is matched to the parameter database by its exact full key; a declared name is never mapped to a key that merely ends with the same name.

Exact ParameterManager rows report the live YAML value, bounds, gradient policy, description, and provenance. A row whose value reads `not declared in the parameter database` is intentionally not assigned an invented default: the class metadata or certificate declares the quantity, with the meaning and units it states, but the parameter database has no key of that exact name.

IPHS levels are read from the sole certificate owner. Levels 1 to 3 describe compatibility, wrapping, or declared structure; they are not strict IPHS/GENERIC claims.

The Channel column of each runtime output contract applies `relic.core.state.types.partition_outputs`, the only runtime output channel classifier, to the declared output name. A name that the classifier reserves for diagnostics, such as `diagnostics.*`, goes to the diagnostics channel and is never committed as state. For any other name the channel is decided at run time by the write mode of the returned value: `StateWriteMode.DERIVED` routes the output to the diagnostics channel, and any other write mode sends it to the StateManager commit (a value without a write mode takes the run's semantic-mode rules or the StateManager schema). The generator does not run `forward()`, so it does not predict that write mode. The Kind column is the State Contract Model kind of the variable and does not set its channel.

(class-relic-scientific-transport-advection-diffusion-advectiondiffusion)=
## advection_diffusion

Combined advection-diffusion transport mechanism.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.transport.advection_diffusion.AdvectionDiffusion` |
| Source | `relic\scientific\transport\advection_diffusion.py:66` |
| Runtime name | `advection_diffusion` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
advection_diffusion(spatial_dims: int = 1, diffusivity: torch.Tensor | float | None = None, velocity_field: torch.Tensor | Tuple[torch.Tensor, ...] | None = None, grid_spacing: float | Tuple[float, ...] | None = None, splitting_method: str | None = None, advection_scheme: str = 'upwind', boundary_method: str | None = None, non_negative: bool | None = None, track_peclet: bool = False, device: torch.device | None = None, name: str = 'advection_diffusion', config: Dict[str, Any] | None = None, metadata: relic.core.functor.functor_types.FunctorMetadata | None = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `spatial_dims` | `int` | `1` | Constructor control for spatial dims; see the class runtime contract and ParameterManager table below. |
| `diffusivity` | <code>torch.Tensor &#124; float &#124; None</code> | `None` | Constructor control for diffusivity; see the class runtime contract and ParameterManager table below. |
| `velocity_field` | <code>torch.Tensor &#124; typing.Tuple[torch.Tensor, ...] &#124; None</code> | `None` | Constructor control for velocity field; see the class runtime contract and ParameterManager table below. |
| `grid_spacing` | <code>float &#124; typing.Tuple[float, ...] &#124; None</code> | `None` | Constructor control for grid spacing; see the class runtime contract and ParameterManager table below. |
| `splitting_method` | <code>str &#124; None</code> | `None` | Constructor control for splitting method; see the class runtime contract and ParameterManager table below. |
| `advection_scheme` | `str` | `'upwind'` | Constructor control for advection scheme; see the class runtime contract and ParameterManager table below. |
| `boundary_method` | <code>str &#124; None</code> | `None` | Constructor control for boundary method; see the class runtime contract and ParameterManager table below. |
| `non_negative` | <code>bool &#124; None</code> | `None` | Constructor control for non negative; see the class runtime contract and ParameterManager table below. |
| `track_peclet` | `bool` | `False` | Constructor control for track peclet; see the class runtime contract and ParameterManager table below. |
| `device` | <code>torch.device &#124; None</code> | `None` | Optional construction device. The simulator assigns the effective runtime device when the Functor is registered. |
| `name` | `str` | `'advection_diffusion'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | <code>typing.Dict[str, typing.Any] &#124; None</code> | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `metadata` | <code>relic.core.functor.functor_types.FunctorMetadata &#124; None</code> | `None` | Optional explicit FunctorMetadata object for advanced composition. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | state cochain c_h |
| `source_term` | diagnostic | `kg m-3 s-1` | open source-control contribution in G_h u_h |
| `velocity_field` | forcing | `m s-1` | advective vector field v_h |
| `tracer_concentration_boundary_state` | forcing | `kg m-3` | conditional external face effort for Dirichlet and open advection |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | decided at run time by the returned write mode | state cochain c_h |
| `tracer_concentration_advection_diffusion_boundary_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | outward-positive total boundary dual cochain |
| `diagnostics.cfl_number` | diagnostic | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.diffusion_stability` | diagnostic | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.peclet_number` | diagnostic | `1` | diagnostics | Output returned by forward(); the certificate gives no description. |
| `diagnostics.conservation_error` | diagnostic | `dimensionless` | diagnostics | Output returned by forward(); the certificate gives no description. |

### Governing equations and canonical operator

**Literature or process form**

Advection-diffusion in conservative form with $q_{\mathrm{adv}} = v c$, $q_{\mathrm{diff}} = - K \nabla ( c )$, and $\frac{\mathrm{d}c}{\mathrm{d}t} = - \operatorname{div} ( q_{\mathrm{adv}} + q_{\mathrm{diff}} ) + S .$

**Canonical RELIC structural form**

$$
\dot{c} = - D_{h} ( q_{\mathrm{adv},h} + q_{\mathrm{diff},h} ) + G_{h} u_{h}; q_{\mathrm{adv},h} = i_{v} c_{h}; q_{\mathrm{diff},h} = - \star_{K} d_{h} c_{h} .
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing canonical finite-volume advection_diffusion divergence stencil |
| `i_v` | AdvectionDiffusion advective HodgeStarSpec and functional flux operator |
| `star_K` | AdvectionDiffusion diffusive HodgeStarSpec and functional flux operator |
| `splitting` | existing splitting_method in the unified advection_diffusion operator |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | tracer concentration cochain $c_{h}$ |
| `effort_variable` | combined transport flux $q_{h} = q_{\mathrm{adv},h} + q_{\mathrm{diff},h}$ |
| `state_advancer` | $\dot{c} = - D_{h} ( q_{\mathrm{adv},h} + q_{\mathrm{diff},h} ) + G_{h} u_{h}$ with $q_{\mathrm{adv},h} = i_{v} c_{h}$ and $q_{\mathrm{diff},h} = - \star_{K} d_{h} c_{h}$ |
| `runtime_usage` | _compute_iphs_advection_diffusion_update returns the canonical integrated operator state, and forward emits its single delta through an explicit ACCUMULATE UpdateItem. |

### Composition channels and executable PortSpec API

Certificate composition channels: `advection_diffusion_boundary_flux`

Executable `ports()` owner: `AdvectionDiffusion`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `advection_diffusion_boundary_flux` | `tracer_concentration_boundary_state [state_unit]` | `tracer_concentration_advection_diffusion_boundary_flux [state_unit m s-1]` | `state_unit2 m s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `transport.advection.default_velocity_magnitude` | `m s-1` | `1.0; bounds=(1e-06, 100.0)` | `fixed` | Default velocity magnitude for uniform flow | Source: literature; Typical atmospheric/water velocities |
| `transport.diffusion.diffusivity` | `m2 s-1` | `1e-05; bounds=(1e-12, 0.01)` | `fixed` | Default molecular diffusion coefficient used when species-specific data are unavailable | Source: RELIC transport module; Typical molecular diffusion in air at 25 degC |
| `transport.advection_diffusion.grid_spacing` | `m` | `1.0; bounds=(1e-06, 1000.0)` | `fixed` | Default grid spacing for coupled advection-diffusion simulations | Source: RELIC transport module; Spatial discretization requirement for finite-volume schemes |
| `transport.advection_diffusion.splitting_method` | `dimensionless` | `'symmetric'; bounds=None` | `fixed` | Operator splitting method used to couple advection and diffusion | Source: RELIC transport module; Strang (1968) SIAM Journal on Numerical Analysis 5(3): 506-517 |
| `transport.numerical.enforce_non_negative` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Enforce non-negative concentrations for transported scalars | Source: RELIC transport module; Physical non-negativity constraint |
| `transport.numerical.track_conservation` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Track mass and energy conservation statistics during simulation | Source: RELIC transport module; Validation requirement |
| `transport.numerical.default_boundary_type` | `dimensionless` | `'neumann'; bounds=('dirichlet', 'periodic')` | `fixed` | Default boundary condition type supported by all canonical transport functors | Source: RELIC transport module; Boundary condition specification |
| `transport.numerical.cfl_advection_max` | `dimensionless` | `1.0; bounds=(0.5, 1.0)` | `fixed` | Maximum allowable CFL number for advection-dominated schemes | Source: RELIC transport module; Courant et al. (1928) |
| `transport.numerical.cfl_diffusion_max` | `dimensionless` | `0.5; bounds=(0.25, 0.5)` | `fixed` | Maximum allowable CFL number for diffusion (Von Neumann condition) | Source: RELIC transport module; Von Neumann and Richtmyer (1950) |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | integrated tracer increment uses UpdateItem ACCUMULATE with per_timestep true |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Transport theory](../theory/isotopes_transport.md).

Certificate theory owner: `docs\source\theory\transport_advection_diffusion.md`.

### References

- Hundsdorfer, W. and Verwer, J.G. (2003). Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations. (FunctorMetadata, certificate)
- Morton, K.W. and Mayers, D.F. (2005). Numerical Solution of Partial Differential Equations. (FunctorMetadata)
- LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations. (FunctorMetadata, certificate)
- Strang, G. (1968). On the construction and comparison of difference schemes. (FunctorMetadata, certificate)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Hirani, A.N. (2003). Discrete Exterior Calculus. (certificate)
- Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow. (certificate)
- LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems. (certificate)

(class-relic-scientific-transport-advection-linearadvection)=
## linear_advection

Linear advection functor with rich diagnostics and parameter integration.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.transport.advection.LinearAdvection` |
| Source | `relic\scientific\transport\advection.py:93` |
| Runtime name | `linear_advection` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
linear_advection(spatial_dims: int = 1, velocity_field: Optional[Union[Tensor, Sequence[Tensor]]] = None, grid_spacing: Optional[Union[float, Sequence[float]]] = None, tracer_key: str = 'tracer_concentration', tracer_units: str = 'kg m-3', scheme: str = 'upwind', boundary_method: Optional[str] = None, non_negative: Optional[bool] = None, track_conservation: Optional[bool] = None, device: Optional[torch.device] = None, name: str = 'linear_advection', config: Optional[Dict[str, Union[str, float, int, bool]]] = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `spatial_dims` | `int` | `1` | Number of spatial dimensions (1, 2, or 3). Units: dimensionless. |
| `velocity_field` | `Optional[Union[Tensor, Sequence[Tensor]]]` | `None` | Velocity components (``m s-1``). Sequence length must equal ``spatial_dims``. If omitted, each forward state must provide the canonical ``velocity_field`` input. |
| `grid_spacing` | `Optional[Union[float, Sequence[float]]]` | `None` | Grid spacing in metres in tensor storage order: ``(dy, dx)`` for 2D and ``(dz, dy, dx)`` for 3D. |
| `tracer_key` | `str` | `'tracer_concentration'` | Constructor control for tracer key; see the class runtime contract and ParameterManager table below. |
| `tracer_units` | `str` | `'kg m-3'` | Constructor control for tracer units; see the class runtime contract and ParameterManager table below. |
| `scheme` | `str` | `'upwind'` | Numerical scheme identifier: ``"upwind"``, ``"lax_wendroff"``, ``"superbee"`` or ``"van_leer"``. |
| `boundary_method` | `Optional[str]` | `None` | Canonical operator boundary method. Explicit constructor values take precedence over the config ``boundary_method`` and ParameterManager ``transport.numerical.default_boundary_type``. |
| `non_negative` | `Optional[bool]` | `None` | Whether to enforce non-negative concentrations. |
| `track_conservation` | `Optional[bool]` | `None` | Enable conservation diagnostics using ``validation.conservation`` tolerances. |
| `device` | `Optional[torch.device]` | `None` | Target PyTorch device. |
| `name` | `str` | `'linear_advection'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Dict[str, Union[str, float, int, bool]]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | state cochain c_h |
| `velocity_field` | forcing | `m s-1` | advective vector field v_h |
| `source_term` | diagnostic | `kg m-3 s-1` | optional source tendency integrated exactly once with the effective time step |
| `tracer_concentration_boundary_state` | forcing | `kg m-3` | externally supplied ghost-cell concentration for open or Dirichlet boundaries |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | decided at run time by the returned write mode | state cochain c_h |
| `tracer_concentration_advective_boundary_flux` | flux | `kg m-2 s-1` | decided at run time by the returned write mode | signed outward-positive boundary face flux cochain |
| `diagnostics.cfl_number` | diagnostic | `1` | diagnostics | CFL stability diagnostic |
| `diagnostics.conservation_error` | diagnostic | `dimensionless` | diagnostics | finite-volume mass balance residual for the selected boundary ledger |

### Governing equations and canonical operator

**Literature or process form**

Conservative linear advection with flux $q = v c$ and storage tendency $\frac{\mathrm{d}c}{\mathrm{d}t} = - \operatorname{div} ( q ) + S .$

**Canonical RELIC structural form**

$$
\dot{c} = - D_{h} q_{h} + s_{h}; q_{h} = i_{v} c_{h} .
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | existing advection operator divergence stencil |
| `i_v` | velocity contraction/interior product realized by the unified advection operator |
| `star_abs_v` | build_hodge_contract magnitude view for structural passivity checks |
| `boundary_operator` | single canonical operator method resolved by constructor, config, then ParameterManager |
| `normal_boundary_port` | boundary_conditions evaluates $F_{b} = ( v_{b} \cdot n_{b} ) c_{b}$ on canonical x/y/z faces |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | tracer concentration cochain $c_{h}$ |
| `effort_variable` | advective flux cochain $q_{h} = i_{v} c_{h}$ |
| `state_advancer` | $\dot{c} = - D_{h} q_{h} + s_{h}$ with $q_{h} = i_{v} c_{h}$ |
| `runtime_usage` | The unified finite-volume operator returns $c_{\mathrm{new}}$; forward forms $c_{\mathrm{new}}$ - $c_{\mathrm{old}}$, adds source_term times dt once, and wraps that integrated increment without a second dt factor. |

### Composition channels and executable PortSpec API

Certificate composition channels: `advective_boundary_flux`

Executable `ports()` owner: `LinearAdvection`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `advective_boundary_flux` | `tracer_concentration_boundary_state [state_unit]` | `tracer_concentration_advective_boundary_flux [state_unit m s-1]` | `state_unit2 m s-1` | into_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `transport.advection.grid_spacing` | `m` | `1.0; bounds=(1e-06, 1000.0)` | `fixed` | Default grid spacing for advection calculations | Source: numerical_method; Spatial discretization requirement |
| `transport.advection.default_velocity_magnitude` | `m s-1` | `1.0; bounds=(1e-06, 100.0)` | `fixed` | Default velocity magnitude for uniform flow | Source: literature; Typical atmospheric/water velocities |
| `transport.advection.inflow_boundary_type` | `dimensionless` | `'dirichlet'; bounds=None` | `fixed` | Boundary condition type for inflow boundaries | Source: numerical_method; Boundary condition specification for hyperbolic systems |
| `transport.advection.outflow_boundary_type` | `dimensionless` | `'open'; bounds=None` | `fixed` | Boundary condition type for outflow boundaries | Source: numerical_method; Open boundary conditions for advection-dominated flows |
| `transport.numerical.cfl_advection_max` | `dimensionless` | `1.0; bounds=(0.5, 1.0)` | `fixed` | Maximum allowable CFL number for advection-dominated schemes | Source: RELIC transport module; Courant et al. (1928) |
| `transport.numerical.track_conservation` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Track mass and energy conservation statistics during simulation | Source: RELIC transport module; Validation requirement |
| `transport.numerical.enforce_non_negative` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Enforce non-negative concentrations for transported scalars | Source: RELIC transport module; Physical non-negativity constraint |
| `transport.numerical.default_boundary_type` | `dimensionless` | `'neumann'; bounds=('dirichlet', 'periodic')` | `fixed` | Default boundary condition type supported by all canonical transport functors | Source: RELIC transport module; Boundary condition specification |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt) |
| `state_write_mode` | UpdateItem mode ACCUMULATE with per_timestep true; StateManager applies no second dt factor |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Transport theory](../theory/isotopes_transport.md).

Certificate theory owner: `docs\source\theory\transport_linear_advection.md`.

### References

- Courant et al. (1928) Mathematische Annalen 100:32-74 (FunctorMetadata)
- LeVeque (2007) Finite Difference Methods for ODE/PDE (FunctorMetadata)
- Sweby (1984) SIAM J. Numer. Anal. 21(5):995-1011 (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata, certificate)
- LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations. SIAM. (certificate)
- Sweby, P.K. (1984). High resolution schemes using flux limiters for hyperbolic conservation laws. (certificate)
- Hirani, A.N. (2003). Discrete Exterior Calculus. (certificate)

(class-relic-scientific-transport-diffusion-fickiandiffusion)=
## fickian_diffusion

Pure diffusion transport functor with parameter-manager integration.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.transport.diffusion.FickianDiffusion` |
| Source | `relic\scientific\transport\diffusion.py:70` |
| Runtime name | `fickian_diffusion` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_3` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
fickian_diffusion(spatial_dims: int = 1, diffusivity: Optional[Union[float, Tensor]] = None, grid_spacing: Optional[Union[float, Sequence[float]]] = None, boundary_method: Optional[str] = None, non_negative: Optional[bool] = None, track_conservation: Optional[bool] = None, device: Optional[torch.device] = None, name: str = 'fickian_diffusion', config: Optional[Dict[str, Union[str, float, int, bool]]] = None, time_integration: str = 'explicit', diffusivity_parameter_key: str = 'transport.diffusion.diffusivity', grid_spacing_parameter_key: str = 'transport.diffusion.grid_spacing', strict_runtime_point: bool = False, diffusivity_layout: Optional[str] = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `spatial_dims` | `int` | `1` | Number of spatial dimensions (1, 2, or 3). Units: dimensionless. |
| `diffusivity` | `Optional[Union[float, Tensor]]` | `None` | Diffusion coefficient ``K`` (``m^2 s^-1``). If omitted the default value registered under ``transport.diffusion.diffusivity`` is used. A number or a 0-d tensor is a scalar ``D`` and is also stored as the base ``TransportMechanism`` scalar; a tensor with ``ndim >= 1`` is a field and leaves the base scalar ``None``. |
| `grid_spacing` | `Optional[Union[float, Sequence[float]]]` | `None` | Grid spacing per spatial axis (``m``). |
| `boundary_method` | `Optional[str]` | `None` | Constructor control for boundary method; see the class runtime contract and ParameterManager table below. |
| `non_negative` | `Optional[bool]` | `None` | Whether to enforce non-negative concentrations after the update. |
| `track_conservation` | `Optional[bool]` | `None` | Enable conservation diagnostics using ``transport.numerical.conservation_tolerance``. |
| `device` | `Optional[torch.device]` | `None` | Target computation device. |
| `name` | `str` | `'fickian_diffusion'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Dict[str, Union[str, float, int, bool]]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `time_integration` | `str` | `'explicit'` | Constructor control for time integration; see the class runtime contract and ParameterManager table below. |
| `diffusivity_parameter_key` | `str` | `'transport.diffusion.diffusivity'` | Fully qualified ParameterManager key used when ``diffusivity`` is omitted. Runtime metadata and IPHS Hodge evidence use the same key. |
| `grid_spacing_parameter_key` | `str` | `'transport.diffusion.grid_spacing'` | Fully qualified ParameterManager key used when ``grid_spacing`` is omitted. Runtime metadata and IPHS metric evidence use the same key. |
| `strict_runtime_point` | `bool` | `False` | Constructor control for strict runtime point; see the class runtime contract and ParameterManager table below. |
| `diffusivity_layout` | `Optional[str]` | `None` | ``"scalar_field"`` or ``"directional"``; required for a tensor diffusivity with ``ndim >= 1`` (it is never inferred from the shape). A scalar ``D`` has the ``"scalar_field"`` layout. |

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | state cochain c_h |
| `source_term` | diagnostic | `kg m-3 s-1` | optional volumetric source contribution integrated exactly once |
| `tracer_concentration_boundary_state` | forcing | `kg m-3` | explicit external Dirichlet boundary state |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | decided at run time by the returned write mode | state cochain c_h |
| `tracer_concentration_diffusive_boundary_flux` | diagnostic | `kg m-2 s-1` | decided at run time by the returned write mode | conditional outward-positive oriented boundary-face flux cochain |
| `diagnostics.diffusion_stability` | diagnostic | `dimensionless` | diagnostics | diffusion CFL audit |
| `diagnostics.conservation_error` | diagnostic | `dimensionless` | diagnostics | mass conservation residual |

### Governing equations and canonical operator

**Literature or process form**

Fickian diffusion in conservative form with flux $J = - D \nabla ( c )$ and storage tendency $\frac{\mathrm{d}c}{\mathrm{d}t} = - \operatorname{div} ( J ) + S .$

**Canonical RELIC structural form**

$$
\dot{c} = - D_{h} q_{h} + G_{h} u_{h}; q_{h} = - \star_{K} d_{h} c_{h}; \star_{K} = \star_{K}^{\mathsf{T}} \geq 0 .
$$

| Entry | Meaning |
| --- | --- |
| `D_h` | canonical finite-volume divergence in OperatorFactory transport.diffusion |
| `d_h` | canonical cell-to-face finite-volume gradient |
| `star_K` | runtime harmonic finite-volume face coefficients; the HodgeStarSpec is a local material diagonal and is not the global face/volume generator |
| `boundary_operator` | explicit periodic, zero, neumann, open, dirichlet, or extrapolate finite-volume closure |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | tracer concentration cochain $c_{h}$ |
| `effort_variable` | storage effort $e_{h} = c_{h}$ and transport force $X_{h} = d_{h} c_{h}$ |
| `state_advancer` | $\dot{c} = - D_{h} q_{h} + G_{h} u_{h}$ with $q_{h} = - \star_{K} d_{h} c_{h}$ |
| `runtime_usage` | forward calls the single OperatorFactory diffusion path and emits one integrated UpdateItem ACCUMULATE; energy_terms and entropy_terms remain structural level 3 audit hooks only. |

### Composition channels and executable PortSpec API

Certificate composition channels: `diffusive_boundary_flux`

Executable `ports()` owner: `FickianDiffusion`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `diffusive_boundary_flux` | `tracer_concentration_boundary_state [state_unit]` | `tracer_concentration_diffusive_boundary_flux [state_unit m s-1]` | `state_unit2 m s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `transport.diffusion.diffusivity` | `m2 s-1` | `1e-05; bounds=(1e-12, 0.01)` | `fixed` | Default molecular diffusion coefficient used when species-specific data are unavailable | Source: RELIC transport module; Typical molecular diffusion in air at 25 degC |
| `transport.diffusion.grid_spacing` | `m` | `1.0; bounds=(1e-06, 1000.0)` | `fixed` | Default computational grid spacing for diffusion-only discretizations | Source: RELIC transport module; Spatial discretization requirement for finite-difference diffusion solvers |
| `transport.diffusion.temperature_exponent` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `fixed` | Exponent n in D(T) = D0 * (T/T0)^n for temperature scaling of gas diffusivity | Source: RELIC transport module; Fuller et al. (1966) Industrial & Engineering Chemistry 5(7):260-267 |
| `transport.numerical.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-12, 0.02)` | `fixed` | Relative tolerance for mass or energy conservation diagnostics | Source: RELIC transport module; Machine precision and physical accuracy |
| `transport.numerical.enforce_non_negative` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Enforce non-negative concentrations for transported scalars | Source: RELIC transport module; Physical non-negativity constraint |
| `transport.numerical.track_conservation` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Track mass and energy conservation statistics during simulation | Source: RELIC transport module; Validation requirement |
| `transport.numerical.cfl_diffusion_max` | `dimensionless` | `0.5; bounds=(0.25, 0.5)` | `fixed` | Maximum allowable CFL number for diffusion (Von Neumann condition) | Source: RELIC transport module; Von Neumann and Richtmyer (1950) |
| `transport.numerical.default_boundary_type` | `dimensionless` | `'neumann'; bounds=('dirichlet', 'periodic')` | `fixed` | Default boundary condition type supported by all canonical transport functors | Source: RELIC transport module; Boundary condition specification |
| `transport.numerical.implicit_midpoint_max_iter` | `dimensionless` | `50; bounds=(1.0, 100.0)` | `fixed` | Maximum fixed-point (Picard) iterations of the implicit midpoint and Crank-Nicolson transport steps (functional._integrate_rate_system); exhausting it raises RuntimeError and never returns an unconverged state. The map y <- y_n + dt f((y_n + y)/2) (theta = 1/2 for Crank-Nicolson) contracts with q = (dt/2) L, L being the infinity-norm Lipschitz constant of the semi-discrete rate, so about ln(tol)/ln(q) iterations reach the increment tolerance transport.numerical.implicit_midpoint_tolerance (1e-10). L = \\|u\\|/dx for central advection, 2\\|u\\|/dx for upwind and 4K/dx^2 for diffusion give q = C/2, C and 2 sigma (C = \\|u\\| dt/dx, sigma = K dt/dx^2). 50 covers q <= 0.6: central advection over the whole documented range C <= transport.numerical.cfl_advection_max = 1, upwind up to C = 0.6 and diffusion up to sigma = 0.3 (probe of 2026-09-23, 64-cell periodic step profile, both methods: at most 30 iterations for central at C = 1, 41 for upwind at C = 0.6 and for diffusion at sigma = 0.3). At q >= 1 (upwind at C = 1, diffusion at sigma = transport.numerical.cfl_diffusion_max = 0.5) the iteration cannot converge for any limit and the step raises; reduce dt. Changed from 4 on 2026-09-23: 4 converged only for q up to about 0.01 (for example central advection failed at C = 0.1, which needs 8 iterations); converged steps are unchanged because the iteration returns its first converged iterate. Distinct from computation.solvers.implicit_midpoint.max_iterations and computation.solvers.crank_nicolson.max_iterations, the solvers own defaults for callers that pass no configuration; the transport kernel always passes this value. | Source: RELIC transport module; Hairer et al. (2006) Geometric Numerical Integration; Hairer, E. and Wanner, G. (1996) Solving Ordinary Differential Equations II, 2nd ed., Springer, Sec. IV.8 (fixed-point iteration of implicit Runge-Kutta equations) |
| `transport.numerical.implicit_midpoint_tolerance` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Fixed-point convergence tolerance for implicit midpoint and Crank-Nicolson transport stepping, passed by the transport kernel as both rtol and atol of the successive-iterate criterion. Distinct from computation.solvers.implicit_midpoint.rtol/atol and computation.solvers.crank_nicolson.rtol/atol, the solvers own defaults for callers that pass no configuration. | Source: RELIC transport module; Hairer et al. (2006) Geometric Numerical Integration |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Transport theory](../theory/isotopes_transport.md).

Certificate theory owner: `docs\source\theory\transport_fickian_diffusion.md`.

### References

- Crank (1975) The Mathematics of Diffusion (FunctorMetadata)
- Bird et al. (2007) Transport Phenomena (FunctorMetadata)
- LeVeque (2007) Finite Difference Methods for ODE/PDE (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press. (certificate)
- Bird, R.B., Stewart, W.E., and Lightfoot, E.N. (2007). Transport Phenomena. Wiley. (certificate)
- LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations. SIAM. (certificate)
- Hirani, A.N. (2003). Discrete Exterior Calculus. (certificate)

(class-transport-diffusion-fickian-diffusion-strict-runtime-point)=
## fickian_diffusion

Pure diffusion transport functor with parameter-manager integration.

### Identity and architecture

| Field | Current contract |
| --- | --- |
| Import path | `relic.scientific.transport.diffusion.FickianDiffusion` |
| Source | `relic\scientific\transport\diffusion.py:70` |
| Runtime name | `fickian_diffusion` |
| Scientific abstraction | `TransportMechanism` |
| Canonical bridge | `TransportMechanismFunctor` |
| Structural role | `transport` |
| Spatial scale | `grid_cell` |
| Variable kind | `state` |
| IPHS level | `level_4` |
| Implementation status | `implemented` |
| Class status | `concrete scientific Functor` |

### Constructor and argument meanings

```python
fickian_diffusion(spatial_dims: int = 1, diffusivity: Optional[Union[float, Tensor]] = None, grid_spacing: Optional[Union[float, Sequence[float]]] = None, boundary_method: Optional[str] = None, non_negative: Optional[bool] = None, track_conservation: Optional[bool] = None, device: Optional[torch.device] = None, name: str = 'fickian_diffusion', config: Optional[Dict[str, Union[str, float, int, bool]]] = None, time_integration: str = 'explicit', diffusivity_parameter_key: str = 'transport.diffusion.diffusivity', grid_spacing_parameter_key: str = 'transport.diffusion.grid_spacing', strict_runtime_point: bool = False, diffusivity_layout: Optional[str] = None) -> None
```

| Argument | Type | Default | Purpose |
| --- | --- | --- | --- |
| `spatial_dims` | `int` | `1` | Number of spatial dimensions (1, 2, or 3). Units: dimensionless. |
| `diffusivity` | `Optional[Union[float, Tensor]]` | `None` | Diffusion coefficient ``K`` (``m^2 s^-1``). If omitted the default value registered under ``transport.diffusion.diffusivity`` is used. A number or a 0-d tensor is a scalar ``D`` and is also stored as the base ``TransportMechanism`` scalar; a tensor with ``ndim >= 1`` is a field and leaves the base scalar ``None``. |
| `grid_spacing` | `Optional[Union[float, Sequence[float]]]` | `None` | Grid spacing per spatial axis (``m``). |
| `boundary_method` | `Optional[str]` | `None` | Constructor control for boundary method; see the class runtime contract and ParameterManager table below. |
| `non_negative` | `Optional[bool]` | `None` | Whether to enforce non-negative concentrations after the update. |
| `track_conservation` | `Optional[bool]` | `None` | Enable conservation diagnostics using ``transport.numerical.conservation_tolerance``. |
| `device` | `Optional[torch.device]` | `None` | Target computation device. |
| `name` | `str` | `'fickian_diffusion'` | Functor identity used by registration, provenance, and dependency resolution. |
| `config` | `Optional[Dict[str, Union[str, float, int, bool]]]` | `None` | Structural and runtime configuration mapping. Scientific numeric parameters remain owned by ParameterManager unless the signature explicitly exposes an override. |
| `time_integration` | `str` | `'explicit'` | Constructor control for time integration; see the class runtime contract and ParameterManager table below. |
| `diffusivity_parameter_key` | `str` | `'transport.diffusion.diffusivity'` | Fully qualified ParameterManager key used when ``diffusivity`` is omitted. Runtime metadata and IPHS Hodge evidence use the same key. |
| `grid_spacing_parameter_key` | `str` | `'transport.diffusion.grid_spacing'` | Fully qualified ParameterManager key used when ``grid_spacing`` is omitted. Runtime metadata and IPHS metric evidence use the same key. |
| `strict_runtime_point` | `bool` | `False` | Constructor control for strict runtime point; see the class runtime contract and ParameterManager table below. |
| `diffusivity_layout` | `Optional[str]` | `None` | ``"scalar_field"`` or ``"directional"``; required for a tensor diffusivity with ``ndim >= 1`` (it is never inferred from the shape). A scalar ``D`` has the ``"scalar_field"`` layout. |

The runtime metadata below is introspected from the documented configuration `strict_runtime_point=True, spatial_dims=1`.

### Runtime input contract

| Variable | Kind | Units | Use |
| --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | state cochain c_h of the closed one-dimensional selection; the runtime point binds it as the dPH state x |
| `source_term` | diagnostic | `kg m-3 s-1` | optional input that must be absent or identically zero for this selection; forward rejects a nonzero source_term before integration (fail-closed), so the executed step equals the audited source-free drift |
| `tracer_concentration_boundary_state` | forcing | `kg m-3` | rejected for closed boundaries; not consumed by this selection |

### Runtime output contract

| Variable | Kind | Units | Channel | Meaning |
| --- | --- | --- | --- | --- |
| `tracer_concentration` | state | `kg m-3` | decided at run time by the returned write mode | state cochain c_h of the closed one-dimensional selection; the runtime point binds it as the dPH state x |
| `diagnostics.diffusion_stability` | diagnostic | `dimensionless` | diagnostics | diffusion CFL audit |
| `diagnostics.conservation_error` | diagnostic | `dimensionless` | diagnostics | mass conservation residual |

:::{note}
The certificate interface and the FunctorMetadata of the documented constructor configuration (the union over all documented modes when there are several) are not identical for this class. The tables above report the complete certificate-backed interface, including any configuration-conditional ports. The difference is listed here so callers do not infer an unconditional interface.
Output declaration difference: `tracer_concentration_diffusive_boundary_flux`.
:::

### Governing equations and canonical operator

**Literature or process form**

Fickian diffusion in conservative form with flux $J = - D \nabla ( c )$ and storage tendency $\frac{\mathrm{d}c}{\mathrm{d}t} = - \operatorname{div} ( J ) + S$; closed finite-volume form $\frac{\mathrm{d}H}{\mathrm{d}t} = - \sum_{f} K_{f} ( \frac{\mathrm{d}c_{f}}{\mathrm{d}x} ) ^ 2 \leq 0$ for $H = 0.5 \mathtt{c.c} ( \mathrm{Crank} 1975 ; \mathrm{Patankar} 1980 ; \mathrm{van} \mathrm{der} \mathrm{Schaft} \mathrm{and} \mathrm{Jeltsema} 2014 )$

**Canonical RELIC structural form**

$$
\dot{c} = ( J_{h} - R_{h} ) e_{H}; J_{h} = 0; R_{h} = d_{h} ^{\mathsf{T}} \operatorname{diag} ( K_{f} ) d_{h} / \mathrm{d}x ^ 2; e_{H} = c_{h}; \frac{\mathrm{d}H_{h}}{\mathrm{d}t} = - e_{H} ^{\mathsf{T}} R_{h} e_{H} \leq 0
$$

| Entry | Meaning |
| --- | --- |
| `J_h` | zero reversible operator declared through $\mathrm{poisson}_{\mathrm{operator}} = \mathrm{zero}_{\mathrm{field}}$ |
| `R_h` | $d_{h} ^{\mathsf{T}} \operatorname{diag} ( K_{f} ) d_{h} / \mathrm{d}x ^ 2 , \mathrm{symmetric} \mathrm{positive} \mathrm{semidefinite} \mathrm{for} \mathrm{positive} \mathrm{diffusivities} ; d_{h} \mathrm{is} \mathrm{the} \mathrm{cell} - \mathrm{to} - \mathrm{face} \mathrm{difference} \mathrm{over} \mathrm{interior} \mathrm{faces} \mathrm{plus} \mathrm{the} \mathrm{periodic} \mathrm{seam} \mathrm{and} K_{f} \mathrm{the} \mathrm{harmonic} \mathrm{face} \mathrm{coefficient}$ |
| `G_h` | none for the certified closed selection |
| `D_h` | canonical finite-volume divergence in OperatorFactory transport.diffusion, reproduced exactly by -$R_{h}$ $e_{H}$ at the runtime point |

### State advancement, conservation, and open boundaries

| Entry | Meaning |
| --- | --- |
| `potential_basis` | tracer concentration cochain $c_{h}$ bound as the dPH state with $H_{h} = 0.5 c_{h} ^{\mathsf{T}} c_{h}$ on the same autograd graph |
| `effort_variable` | $e_{H} = \frac{\mathrm{d}H_{h}}{\mathrm{d}c_{h}} = c_{h} \mathrm{obtained}$ by torch.autograd.grad inside canonical_drift |
| `state_advancer` | $\dot{c} = ( J_{h} - R_{h} ) e_{H}$ with $J_{h} = 0$ and $R_{h} = d_{h} ^{\mathsf{T}} \operatorname{diag} ( K_{f} ) d_{h} / \mathrm{d}x ^ 2$, integrated by the production finite-volume operator as $\Delta c = \mathrm{d}t \dot{c}$ for explicit Euler |
| `runtime_usage` | forward calls the single OperatorFactory diffusion path, emits one integrated UpdateItem ACCUMULATE, and for this selection verifies the Level 4 runtime point through IPHSStructureSpec.validate($\mathrm{required}_{\mathrm{level}} = 4 )$ with the bound storage on every emitted step; the audit is exposed by last_strict_iphs_audit and cleared by a rejected step. |

### Composition channels and executable PortSpec API

Certificate composition channels: none declared.

Executable `ports()` owner: `FickianDiffusion`.

| Port | Effort and units | Flow and units | Pairing units | Convention and kind |
| --- | --- | --- | --- | --- |
| `diffusive_boundary_flux` | `tracer_concentration_boundary_state [state_unit]` | `tracer_concentration_diffusive_boundary_flux [state_unit m s-1]` | `state_unit2 m s-1` | out_of_system; matter |

Metadata dependencies: no direct dependency list is declared; composition uses ports, owners, or graph wiring.

### Scientific parameter and numeric binding contract

| Parameter key | Units | Default and bounds | Gradient | Purpose | Provenance or reference |
| --- | --- | --- | --- | --- | --- |
| `transport.diffusion.diffusivity` | `m2 s-1` | `1e-05; bounds=(1e-12, 0.01)` | `fixed` | Default molecular diffusion coefficient used when species-specific data are unavailable | Source: RELIC transport module; Typical molecular diffusion in air at 25 degC |
| `transport.diffusion.grid_spacing` | `m` | `1.0; bounds=(1e-06, 1000.0)` | `fixed` | Default computational grid spacing for diffusion-only discretizations | Source: RELIC transport module; Spatial discretization requirement for finite-difference diffusion solvers |
| `transport.diffusion.temperature_exponent` | `dimensionless` | `1.5; bounds=(1.0, 2.0)` | `fixed` | Exponent n in D(T) = D0 * (T/T0)^n for temperature scaling of gas diffusivity | Source: RELIC transport module; Fuller et al. (1966) Industrial & Engineering Chemistry 5(7):260-267 |
| `transport.numerical.conservation_tolerance` | `dimensionless` | `1e-06; bounds=(1e-12, 0.02)` | `fixed` | Relative tolerance for mass or energy conservation diagnostics | Source: RELIC transport module; Machine precision and physical accuracy |
| `transport.numerical.enforce_non_negative` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Enforce non-negative concentrations for transported scalars | Source: RELIC transport module; Physical non-negativity constraint |
| `transport.numerical.track_conservation` | `dimensionless` | `True; bounds=(0.0, 1.0)` | `fixed` | Track mass and energy conservation statistics during simulation | Source: RELIC transport module; Validation requirement |
| `transport.numerical.cfl_diffusion_max` | `dimensionless` | `0.5; bounds=(0.25, 0.5)` | `fixed` | Maximum allowable CFL number for diffusion (Von Neumann condition) | Source: RELIC transport module; Von Neumann and Richtmyer (1950) |
| `transport.numerical.default_boundary_type` | `dimensionless` | `'neumann'; bounds=('dirichlet', 'periodic')` | `fixed` | Default boundary condition type supported by all canonical transport functors | Source: RELIC transport module; Boundary condition specification |
| `transport.numerical.implicit_midpoint_max_iter` | `dimensionless` | `50; bounds=(1.0, 100.0)` | `fixed` | Maximum fixed-point (Picard) iterations of the implicit midpoint and Crank-Nicolson transport steps (functional._integrate_rate_system); exhausting it raises RuntimeError and never returns an unconverged state. The map y <- y_n + dt f((y_n + y)/2) (theta = 1/2 for Crank-Nicolson) contracts with q = (dt/2) L, L being the infinity-norm Lipschitz constant of the semi-discrete rate, so about ln(tol)/ln(q) iterations reach the increment tolerance transport.numerical.implicit_midpoint_tolerance (1e-10). L = \\|u\\|/dx for central advection, 2\\|u\\|/dx for upwind and 4K/dx^2 for diffusion give q = C/2, C and 2 sigma (C = \\|u\\| dt/dx, sigma = K dt/dx^2). 50 covers q <= 0.6: central advection over the whole documented range C <= transport.numerical.cfl_advection_max = 1, upwind up to C = 0.6 and diffusion up to sigma = 0.3 (probe of 2026-09-23, 64-cell periodic step profile, both methods: at most 30 iterations for central at C = 1, 41 for upwind at C = 0.6 and for diffusion at sigma = 0.3). At q >= 1 (upwind at C = 1, diffusion at sigma = transport.numerical.cfl_diffusion_max = 0.5) the iteration cannot converge for any limit and the step raises; reduce dt. Changed from 4 on 2026-09-23: 4 converged only for q up to about 0.01 (for example central advection failed at C = 0.1, which needs 8 iterations); converged steps are unchanged because the iteration returns its first converged iterate. Distinct from computation.solvers.implicit_midpoint.max_iterations and computation.solvers.crank_nicolson.max_iterations, the solvers own defaults for callers that pass no configuration; the transport kernel always passes this value. | Source: RELIC transport module; Hairer et al. (2006) Geometric Numerical Integration; Hairer, E. and Wanner, G. (1996) Solving Ordinary Differential Equations II, 2nd ed., Springer, Sec. IV.8 (fixed-point iteration of implicit Runge-Kutta equations) |
| `transport.numerical.implicit_midpoint_tolerance` | `dimensionless` | `1e-10; bounds=(1e-15, 1e-06)` | `fixed` | Fixed-point convergence tolerance for implicit midpoint and Crank-Nicolson transport stepping, passed by the transport kernel as both rtol and atol of the successive-iterate criterion. Distinct from computation.solvers.implicit_midpoint.rtol/atol and computation.solvers.crank_nicolson.rtol/atol, the solvers own defaults for callers that pass no configuration. | Source: RELIC transport module; Hairer et al. (2006) Geometric Numerical Integration |

### Runtime flow

| Entry | Meaning |
| --- | --- |
| `forward_signature` | forward(self, state, time, dt, **kwargs) |
| `state_write_mode` | UpdateItem ACCUMULATE with per_timestep true |
| `abstraction_contract` | TransportMechanism |
| `bridge_class` | TransportMechanismFunctor |

Detailed derivation: [Transport theory](../theory/isotopes_transport.md).

Certificate theory owner: `docs\source\theory\transport_fickian_diffusion.md`.

### References

- Crank (1975) The Mathematics of Diffusion (FunctorMetadata)
- Bird et al. (2007) Transport Phenomena (FunctorMetadata)
- LeVeque (2007) Finite Difference Methods for ODE/PDE (FunctorMetadata)
- Fick, A. (1855). On liquid diffusion. (FunctorMetadata)
- Courant, R., Friedrichs, K., and Lewy, H. (1928). On the partial difference equations of mathematical physics. (FunctorMetadata)
- Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press. (certificate)
- Patankar, S.V. (1980). Numerical Heat Transfer and Fluid Flow. Hemisphere. (certificate)
- van der Schaft, A. & Jeltsema, D. (2014). Port-Hamiltonian Systems Theory, Foundations and Trends in Systems and Control. (certificate)
- LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations. SIAM. (certificate)
