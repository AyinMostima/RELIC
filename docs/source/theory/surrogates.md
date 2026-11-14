# Surrogate Process Closure and Machine-Learning Emulator Runtime

## 1. Process Definition and Scope

RELIC provides two distinct surrogate surfaces for biological process emulation:

1. `SimpleCanopyPhotosynthesis` (`relic/scientific/biosphere/surrogates/simple_photosynthesis.py`): a certified analytical canopy carbon surrogate activated only through explicit opt-in (`create_standard_biosphere_functors(use_surrogate_photosynthesis=True)`). When selected, it replaces the Farquhar photosynthesis, autotrophic respiration, and growth chain with one self-contained mass-flux closure (`gross_primary_productivity`, `ecosystem_respiration`, `net_ecosystem_exchange`) and rejects `include_growth=True` or default MEGAN molar-port coupling at factory construction time.
2. `SurrogateFunctor`, `ONNXSurrogate` (`relic/models/surrogates.py`), and `PyTorchSurrogate` (`relic/models/pytorch_surrogate.py`): generic machine-learning emulator wrappers inheriting from `BiologicalProcessFunctor` (not a category-neutral `FunctorBase` bypass), supporting ONNX Runtime inference, PyTorch MLP/ResNet/Transformer architectures, optional ledger matrix checking, and dropout Monte Carlo or forked-generator ensemble uncertainty.

Standalone Gaussian-process emulators (`GaussianProcessSurrogate`), thermodynamic physics-loss terms (`thermodynamic_weight`), and surrogate-compatible MEGAN BVOC closures remain `[N]`. Deeper derivations of surrogate approximation error and calibration workflows appear in *Methods of Ecological Analysis*, Chapters 20 and 28.

## 2. Governing Equations

### 2.1 Empirical Canopy Carbon Surrogate (`SimpleCanopyPhotosynthesis`)

From photosynthetic photon flux density $Q$ (`ppfd`, $\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$), Kelvin air temperature $T$, volumetric soil moisture $\theta \in [0, 1]$, and canopy leaf area index $\mathrm{LAI}$ (Baly and Morgan, 1934; Baldocchi, 2003; Chapin et al., 2006; Mahecha et al., 2010):

$$
f_Q = \frac{Q}{Q + K_Q}, \qquad
f_T = \exp\!\left[-\left(\frac{T - T_{\mathrm{opt}}}{w_T}\right)^2\right], \qquad
f_w = \exp\!\left[-\left(\frac{\theta - \theta_{\mathrm{opt}}}{w_\theta}\right)^2\right]
$$

$$
\mathrm{GPP} = G_{\max}\,f_Q\,f_T\,f_w\,\mathrm{LAI}, \qquad
R_{\mathrm{eco}} = R_{\mathrm{base}}\,Q_{10}^{(T - T_{\mathrm{ref}})/10}\,\mathrm{LAI}, \qquad
\mathrm{NEE} = R_{\mathrm{eco}} - \mathrm{GPP}
$$

Positive $\mathrm{NEE}$ follows the AmeriFlux sign convention (net carbon release from ecosystem to atmosphere). The two primary rates $r_h = [\mathrm{GPP}, R_{\mathrm{eco}}]^{\mathsf{T}}$ lower through `advance_reaction_network` with open carbon port incidence:

$$
y_h = N_h r_h =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
-1 & 1
\end{bmatrix}
\begin{bmatrix}
\mathrm{GPP} \\
R_{\mathrm{eco}}
\end{bmatrix}
$$

### 2.2 Neural Emulator Mapping, Physics Losses, and Ledger Residual (`relic.models`)

A configured biological emulator maps standardized input vector $X \in \mathbb{R}^{B \times d}$ to output vector $Y = \mathcal{S}(X; \phi) \in \mathbb{R}^{B \times m}$ using an `mlp`, `resnet`, or `transformer` network (`PyTorchSurrogate`) or a serialized ONNX graph (`ONNXSurrogate`). When `conservation_check: true` is configured with ledger matrix $C$ and explicit `conservation_tolerance` $\varepsilon$, each inference verifies:

$$
R = Y C^{\mathsf{T}}, \qquad \max\bigl(\lvert R \rvert\bigr) \le \varepsilon
$$

During `PyTorchSurrogate.train(...)`, optional physics-informed regularization combines data loss $\mathcal{L}_{\mathrm{data}}$ with configured conservation, monotonicity, output-bound, and smoothness penalties; total predictive variance under `dropout_mc` or `ensemble` decomposes into $\sigma_{\mathrm{total}}^2 = \sigma_{\mathrm{approx}}^2 + \sigma_{\mathrm{param}}^2 + \sigma_{\mathrm{input}}^2$ (Sacks et al., 1989; Kennedy and O'Hagan, 2001; Raissi et al., 2019).

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $Q$ | Photosynthetic photon flux density (from `atmospheric_radiation`) | $\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$ | `ppfd` | `forcing` |
| $T$ | Near-surface air temperature ($> 0$) | $\mathrm{K}$ | `air_temperature` | `forcing` |
| $\mathrm{LAI}$ | Canopy leaf area index ($\ge 0$) | $\mathrm{m^2\,m^{-2}}$ | `leaf_area_index` | `state` |
| $\theta$ | Volumetric soil moisture ($\in [0, 1]$) | $\mathrm{m^3\,m^{-3}}$ | `soil_moisture` | `state` |
| $\mathrm{GPP}$ | Canopy gross primary productivity mass flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `gross_primary_productivity` | `flux` |
| $R_{\mathrm{eco}}$ | Ecosystem respiration mass flux | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `ecosystem_respiration` | `flux` |
| $\mathrm{NEE}$ | Net ecosystem carbon exchange ($R_{\mathrm{eco}} - \mathrm{GPP}$) | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `net_ecosystem_exchange` | `flux` |
| $f_w, f_T$ | Surrogate soil-moisture and temperature response factors | $\mathrm{1}$ | `diagnostics.photosynthesis.water_stress`, `diagnostics.photosynthesis.temperature_factor` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $G_{\max}$ | `biosphere.photosynthesis.surrogate.max_gpp_rate` | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `biosphere/photosynthesis.yaml` ($3.0 \times 10^{-7}$) | Baly and Morgan (1934); Chapin et al. (2006) |
| $K_Q$ | `biosphere.photosynthesis.surrogate.ppfd_half_saturation` | $\mathrm{\mu mol\,photons\,m^{-2}\,s^{-1}}$ | `biosphere/photosynthesis.yaml` | Baly and Morgan (1934) |
| $T_{\mathrm{opt}}, w_T$ | `biosphere.photosynthesis.surrogate.temperature_optimum`, `biosphere.photosynthesis.surrogate.temperature_width` | $\mathrm{K}$ | `biosphere/photosynthesis.yaml` | Baldocchi (2003) |
| $R_{\mathrm{base}}$ | `biosphere.photosynthesis.surrogate.base_respiration_rate` | $\mathrm{kg\,C\,m^{-2}\,s^{-1}}$ | `biosphere/photosynthesis.yaml` ($4.8 \times 10^{-8}$) | Mahecha et al. (2010) |
| $Q_{10}$ | `biosphere.photosynthesis.surrogate.respiration_q10` | $\mathrm{1}$ | `biosphere/photosynthesis.yaml` ($1.4$) | Mahecha et al. (2010) |
| $\theta_{\mathrm{opt}}, w_\theta$ | `biosphere.photosynthesis.surrogate.soil_moisture_optimum`, `biosphere.photosynthesis.surrogate.soil_moisture_width` | $\mathrm{m^3\,m^{-3}}$ | `biosphere/photosynthesis.yaml` | Baldocchi (2003) |
| $T_{\mathrm{ref}}$ | `relic.physical.constants.reference_temperature` | $\mathrm{K}$ | `physical/constants.yaml` ($298.15$) | Standard biochemical reference |
| $\phi_{\mathrm{pytorch}}$ | `models.surrogate.pytorch.*` | architecture/training units | `models/pytorch_surrogate.yaml` | Rasp et al. (2018); Raissi et al. (2019) |

## 5. Numerical Methods and Discretization

`SimpleCanopyPhotosynthesis` broadcasts its four required floating tensors, rejects out-of-domain inputs without silent clipping, and emits `UpdateItem(UpdateMode.REPLACE, per_timestep=False)` outputs. `SurrogateFunctor` subclasses enforce fail-fast runtime contracts: model checkpoint paths must exist unless `validate_model_path: false` is explicitly declared; unreadable checkpoints raise `ModelLoadError`; `ONNXSurrogate` validates exact tensor widths, rejects `uncertainty_quantification: true`, and raises if `device: cuda` lacks `CUDAExecutionProvider`; `PyTorchSurrogate` reads undeclared hyperparameters exclusively from `models/pytorch_surrogate.yaml`, validates architecture-specific keys (`mlp`, `resnet`, `transformer` with `d_model` divisible by `n_heads`), requires positive dropout and explicit sample counts for `dropout_mc` or `ensemble` uncertainty, and restores a deep copy of best-validation weights.

## 6. Conservation, Boundary Conditions, and Ledgers

`SimpleCanopyPhotosynthesis` declares `conservation_laws=["carbon"]` and `open_system_route="port"`, closing the algebraic flux identity $\mathrm{NEE} - (R_{\mathrm{eco}} - \mathrm{GPP}) = 0$ on `REPLACE` rate ports without advancing carbon mass pools or emitting Farquhar molar ports (`gross_photosynthesis`, `net_photosynthesis`). For generic `SurrogateFunctor` instances, conservation checking is opt-in via `conservation_check: true` and `conservation_tolerance`, evaluating $\max(\lvert Y C^{\mathsf{T}} \rvert) \le \varepsilon$ in physical output units.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SimpleCanopyPhotosynthesis` | `relic/scientific/biosphere/surrogates/simple_photosynthesis.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `SurrogateFunctor`, `ONNXSurrogate` | `relic/models/surrogates.py` | `BiologicalProcess` | `BiologicalProcessFunctor` | Configured `ExecutionPhase` |
| `PyTorchSurrogate` | `relic/models/pytorch_surrogate.py` | `BiologicalProcess` | `BiologicalProcessFunctor` (`SurrogateFunctor`) | Configured `ExecutionPhase` |

## 8. Implementation Status

- `SimpleCanopyPhotosynthesis`: `[I]`, IPHS **Level 2** (`reaction`, certificate `biosphere_surrogates_simple_photosynthesis_simple_canopy_photosynthesis.yaml`). Opt-in empirical canopy carbon closure wrapped in `advance_reaction_network`.
- `SurrogateFunctor`, `ONNXSurrogate`, `PyTorchSurrogate`: `[I]` generic `BiologicalProcessFunctor` emulator runtime frameworks (not standalone scientific catalog processes).
- `GaussianProcessSurrogate`, `PhysicsInformedSurrogate` subclass, `thermodynamic_weight` loss term, and surrogate-coupled MEGAN BVOC workflows: `[N]`.

Frozen certificate expressions:

```text
GPP and Reco empirical rates with the canonical positive-to-atmosphere NEE orientation
y_h = N_h r_h with ParameterManager-backed rates and an explicit open carbon port
```

## 9. References

- Baldocchi, D. (2003). Assessing the eddy covariance technique for evaluating carbon dioxide exchange rates of ecosystems: Past, present and future. *Global Change Biology*, 9(4), 479-492.
- Baly, E. C. C., and Morgan, L. B. (1934). Kinetics of photosynthesis and allied processes. *Nature*, 133, 414.
- Chapin, F. S., Woodwell, G. M., Randerson, J. T., et al. (2006). Reconciling carbon-cycle concepts, terminology, and methods. *Ecosystems*, 9(7), 1041-1050.
- Kennedy, M. C., and O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B*, 63(3), 425-464.
- Mahecha, M. D., Reichstein, M., Carvalhais, N., et al. (2010). Global convergence in the temperature sensitivity of respiration at ecosystem level. *Science*, 329(5993), 838-840.
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- Rasp, S., Pritchard, M. S., and Gentine, P. (2018). Deep learning to represent subgrid processes in climate models. *Proceedings of the National Academy of Sciences*, 115(39), 9684-9689.
- Sacks, J., Welch, W. J., Mitchell, T. J., and Wynn, H. P. (1989). Design and analysis of computer experiments. *Statistical Science*, 4(4), 409-423.
