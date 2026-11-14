# Two-Region Mobile-Immobile Mass Transfer

## 1. Process Definition and Scope

`TwoRegionMassTransferFunctor` (`relic/scientific/pedosphere/transport/two_region_mass_transfer.py`, `[I]`) implements first-order interphase solute mass transfer between mobile (flowing) and immobile (stagnant) pore-water regions in structured soils (van Genuchten and Wierenga, 1976; Haggerty and Gorelick, 1995). It fulfills the `TransportMechanism` abstract contract via `TransportMechanismFunctor` (`TRANSPORT_TYPE = "interphase_mass_transfer"`, `ExecutionPhase.MAIN_PROCESSES`) with declared exact constant-coefficient sub-flow accuracy (`SubFlowAccuracy.EXACT`).

Within the RELIC memory and Volterra realization semantics, eliminating the immobile concentration yields an integro-differential Volterra equation on the mobile region with a single-exponential completely monotone kernel, while retaining `immobile_solute_concentration` in `StateManager` provides the exact finite-dimensional Markovian realization required for bitwise-consistent checkpoint and restart continuation. Multirate mobile-immobile exchange with $M > 1$ immobile compartments is `[N]`.

## 2. Governing Equations

Let $c_m$ and $c_{im}$ ($\mathrm{kg\,m^{-3}}$) denote the mobile and immobile pore-water solute concentrations, $\theta_m > 0$ and $\theta_{im} > 0$ ($\mathrm{m^3\,m^{-3}}$) the mobile and immobile volumetric water contents, and $\alpha > 0$ ($\mathrm{s^{-1}}$) the first-order mass-transfer rate coefficient. The coupled two-region exchange ODE system is (van Genuchten and Wierenga, 1976):

$$
\theta_m \frac{dc_m}{dt} = -\alpha \left(c_m - c_{im}\right), \qquad \theta_{im} \frac{dc_{im}}{dt} = \alpha \left(c_m - c_{im}\right)
$$

Eliminating $c_{im}(t)$ with relaxation frequency $\omega = \alpha / \theta_{im}$ ($\mathrm{s^{-1}}$) produces the equivalent Volterra memory equation on the mobile phase (Haggerty and Gorelick, 1995):

$$
\theta_m \frac{dc_m}{dt} = -\alpha c_m + \alpha e^{-\omega t} c_{im}(0) + \int_0^t \alpha \omega e^{-\omega (t - s)} c_m(s)\,ds
$$

with exponential memory kernel $K(t) = \alpha \omega e^{-\omega t}$ and hidden realization state $\eta = c_{im}$ obeying $\frac{d\eta}{dt} = -\omega \eta + \omega c_m$.

In extensive mass-density coordinates $x = (\theta_m c_m, \theta_{im} c_{im})^{\mathsf{T}}$, the quadratic storage functional $H(x)$, effort covector $e(x)$, and symmetric positive semidefinite resistive operator $R \succeq 0$ form a closed dissipative system:

$$
H(x) = \frac{1}{2}\left(\theta_m c_m^2 + \theta_{im} c_{im}^2\right), \qquad e = \frac{\partial H}{\partial x} = \begin{pmatrix} c_m \\ c_{im} \end{pmatrix}, \qquad \frac{dx}{dt} = -R e, \qquad R = \alpha \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
$$

The nonnegative internal dissipation rate returned by `entropy_terms` is $\sigma = e^{\mathsf{T}} R e = \alpha (c_m - c_{im})^2 \ge 0$, whereas the signed storage-rate diagnostic emitted under `diagnostics.two_region_exchange.storage_dissipation` records the nonpositive time derivative of $H$:

$$
\frac{dH}{dt} = -\sigma = -\alpha \left(c_m - c_{im}\right)^2 \le 0
$$

Even when $c_{im} > c_m$ returns stored solute and delivers positive instantaneous power $-\alpha c_m (c_m - c_{im}) > 0$ to the mobile region, the joint storage $H(x)$ never increases.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
| --- | --- | --- | --- | --- |
| $c_m$ | Mobile-region pore-water solute concentration | $\mathrm{kg\,m^{-3}}$ | `mobile_solute_concentration` | Required state input and `ACCUMULATE` output (`per_timestep=True`) |
| $c_{im}$ | Immobile-region pore-water solute concentration (memory realization $\eta$) | $\mathrm{kg\,m^{-3}}$ | `immobile_solute_concentration` | Required state input and `ACCUMULATE` output (`per_timestep=True`) |
| $q_{\mathrm{ex}}$ | Instantaneous mobile-to-immobile mass-transfer rate $\alpha(c_m - c_{im})$ at step start | $\mathrm{kg\,m^{-3}\,s^{-1}}$ | `diagnostics.two_region_exchange.mass_transfer_rate` | Diagnostic output (`REPLACE`, `per_timestep=False`) |
| $\dot{H}$ | Signed quadratic storage rate $-\alpha(c_m - c_{im})^2 \le 0$ at step start | $\mathrm{kg^2\,m^{-6}\,s^{-1}}$ | `diagnostics.two_region_exchange.storage_dissipation` | Diagnostic output (`REPLACE`, `per_timestep=False`) |
| $r_{\mathrm{mem}}$ | Weighted step mass-ledger residual $\theta_m \Delta c_m + \theta_{im} \Delta c_{im}$ | $\mathrm{kg\,m^{-3}}$ | `diagnostics.two_region_exchange.memory_residual` | Diagnostic output (`REPLACE`, `per_timestep=False`) |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
| --- | --- | --- | --- | --- |
| $\alpha$ | `pedosphere.transport.two_region_mass_transfer_coefficient` | $\mathrm{s^{-1}}$ | `pedosphere/transport.yaml` ($2.0 \times 10^{-5}$) | van Genuchten and Wierenga (1976) |
| $\theta_m$ | `pedosphere.transport.two_region_mobile_water_content` | $\mathrm{m^3\,m^{-3}}$ | `pedosphere/transport.yaml` ($0.25$) | van Genuchten and Wierenga (1976) |
| $\theta_{im}$ | `pedosphere.transport.two_region_immobile_water_content` | $\mathrm{m^3\,m^{-3}}$ | `pedosphere/transport.yaml` ($0.10$) | van Genuchten and Wierenga (1976) |

## 5. Numerical Methods and Discretization

For parameters $(\alpha, \theta_m, \theta_{im})$ held constant over a call duration $\Delta t > 0$, `TwoRegionMassTransferFunctor.forward` integrates the $2 \times 2$ linear generator

$$
A = \begin{pmatrix} -\alpha / \theta_m & \alpha / \theta_m \\ \alpha / \theta_{im} & -\alpha / \theta_{im} \end{pmatrix}
$$

in closed analytical form ($\Phi_{\Delta t} c = \exp(\Delta t A) c$), justifying `SubFlowAccuracy.EXACT` (Hairer et al., 2006). Defining the invariant total volumetric solute mass $m$ and concentration difference $d = c_m - c_{im}$:

$$
m = \theta_m c_m(t) + \theta_{im} c_{im}(t), \qquad \kappa = \frac{\alpha \left(\theta_m + \theta_{im}\right)}{\theta_m \theta_{im}}, \qquad d(t + \Delta t) = d(t) \exp(-\kappa \Delta t)
$$

the exact post-step concentrations and emitted `UpdateMode.ACCUMULATE` (`per_timestep=True`) increments are:

$$
c_m(t + \Delta t) = \frac{m + \theta_{im} d(t + \Delta t)}{\theta_m + \theta_{im}}, \qquad c_{im}(t + \Delta t) = \frac{m - \theta_m d(t + \Delta t)}{\theta_m + \theta_{im}}
$$

$$
\Delta c_m = c_m(t + \Delta t) - c_m(t), \qquad \Delta c_{im} = c_{im}(t + \Delta t) - c_{im}(t)
$$

The analytical propagator satisfies the exact semigroup identity $\Phi_{\Delta t_2} \circ \Phi_{\Delta t_1} = \Phi_{\Delta t_1 + \Delta t_2}$ and preserves nonnegativity for all $c_m(t), c_{im}(t) \ge 0$ and $\Delta t > 0$. Non-finite, negative, or shape-mismatched concentration tensors, nonpositive $(\alpha, \theta_m, \theta_{im})$, or nonpositive $\Delta t$ raise `ValueError` immediately.

## 6. Conservation, Boundary Conditions, and Ledgers

The two-region exchange operator has no external boundary port; solute mass is redistributed strictly between the mobile and immobile water fractions within each control volume. Across every step $\Delta t$, the $\theta$-weighted solute mass density is conserved up to floating-point rounding:

$$
r_{\mathrm{mem}} = \theta_m \Delta c_m + \theta_{im} \Delta c_{im} = 0
$$

`check_conservation_laws` evaluates $\max \lvert \theta_m \Delta c_m + \theta_{im} \Delta c_{im} \rvert$ directly from the emitted `UpdateItem`s. Because the hidden Volterra realization $\eta = c_{im}$ is stored as the governed state variable `immobile_solute_concentration` in `StateManager` (with no Python-instance history buffer), accepted-step snapshots and checkpoint restarts preserve both total solute mass $m$ and stored memory energy $H(x)$.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
| --- | --- | --- | --- | --- |
| `TwoRegionMassTransferFunctor` | `relic/scientific/pedosphere/transport/two_region_mass_transfer.py` | `TransportMechanism` | `TransportMechanismFunctor` | `ExecutionPhase.MAIN_PROCESSES` |

## 8. Implementation Status

- **`TwoRegionMassTransferFunctor`**: `[I]` (IPHS Level 3 `transport`, passive single-exponential Volterra memory realization with `SubFlowAccuracy.EXACT`; quadratic concentration storage $H$ is a structural passivity functional without a thermodynamic chemical-potential claim). Multirate $M$-compartment immobile memory series are `[N]`. Certificate: `pedosphere_transport_two_region_mass_transfer_two_region_mass_transfer_functor.yaml`.

```text
d(t + dt) = d(t) exp(-kappa dt) with d = c_m - c_im and kappa = alpha (theta_m + theta_im)/(theta_m theta_im); m = theta_m c_m + theta_im c_im conserved
theta_m dc_m/dt = -alpha c_m + alpha e^{-omega t} c_im(0) + int_0^t alpha omega e^{-omega (t-s)} c_m(s) ds, omega = alpha/theta_im; hidden realization eta = c_im with d eta/dt = -omega eta + omega c_m
```

## 9. References

1. Haggerty, R., and Gorelick, S. M. (1995). Multiple-rate mass transfer for modeling diffusion and surface reactions in media with pore-scale heterogeneity. *Water Resources Research*, 31(10), 2383-2400. doi:10.1029/95WR10583
2. Hairer, E., Lubich, C., and Wanner, G. (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations* (2nd ed., Sect. II.4-II.5). Springer.
3. van Genuchten, M. Th., and Wierenga, P. J. (1976). Mass transfer studies in sorbing porous media: I. Analytical solutions. *Soil Science Society of America Journal*, 40(4), 473-480. doi:10.2136/sssaj1976.03615995004000040011x
