# Theory, Design Tradeoffs, and RELIC Advantages

This chapter explains the scientific logic behind the Tapajos tutorial case.

---

## Design Principle

> [!IMPORTANT]
> The case does **not** treat machine learning as a replacement for the entire ecosystem model. Instead, it follows a **selective replacement** strategy:
>
> 1. Keep the **best-validated physiological cores** explicit
> 2. Replace the **most data-uncertain and structurally simplified** sub-processes
> 3. Preserve **dual-path outputs** so that physics and ML can be audited side by side

---

## Physics Baseline

The physics baseline is built around three explicit cores.

### 1. Canopy Energy and Hydrology Support

The baseline uses physically interpretable micrometeorology and water-balance operators:

- **Turbulence** --- Monin-Obukhov similarity
- **Penman-Monteith** --- energy-balance evapotranspiration
- **Precipitation partitioning** --- throughfall routing
- **Surface runoff** --- saturation-excess
- **Bucket soil moisture** --- simple water balance

These modules provide a transparent baseline state trajectory even when some forcing variables are externally supplied.

### 2. Biochemical Carbon Assimilation

The case keeps the **Farquhar-von Caemmerer-Berry** (FvCB) C3 photosynthesis model as the carbon assimilation core.

The standard formulation:

$$A_n = \min(W_c,\; W_j,\; W_p) - R_d$$

where:

| Symbol | Description                                             |
| :----: | :------------------------------------------------------ |
| $W_c$  | Rubisco-limited assimilation rate                       |
| $W_j$  | Electron-transport-limited assimilation rate            |
| $W_p$  | Triose phosphate utilization (TPU) or export limitation |
| $R_d$  | Dark (mitochondrial) respiration                        |

> [!NOTE]
> This remains the **most defensible mechanistic core** for leaf-level C3 photosynthesis in a site-scale tutorial.

### 3. MEGAN BVOC Emission Core

The case keeps **MEGAN 2.1** as the mechanistic BVOC emission engine. The simplified emission equation:

$$E_{\text{iso}} = \varepsilon_{\text{iso}} \cdot \gamma_L \cdot \gamma_T \cdot \gamma_A \cdot \gamma_{\text{SM}} \cdot \gamma_{\text{CO}_2} \cdot \gamma_{\text{other}}$$

where:

|           Symbol           | Description                                             |
| :------------------------: | :------------------------------------------------------ |
| $\varepsilon_{\text{iso}}$ | Isoprene emission potential (landscape emission factor) |
|         $\gamma_L$         | Light response activity factor                          |
|         $\gamma_T$         | Temperature response activity factor                    |
|         $\gamma_A$         | Canopy and leaf age activity factor                     |
|    $\gamma_{\text{SM}}$    | Soil moisture stress modulation factor                  |
|   $\gamma_{\text{CO}_2}$   | CO2 inhibition factor                                   |
|  $\gamma_{\text{other}}$   | Other environmental modifiers                           |

> [!NOTE]
> Keeping MEGAN explicit matters because it maintains a **process-level link** between canopy physiology and BVOC emission physics.

---

## Why the ML Path Replaces Only Specific Links

The archived data situation makes some sub-processes much less trustworthy than others.

### Soil Moisture

The bucket soil module is structurally simple, and in this case the forcing is largely externally reconstructed rather than tower-native. Soil moisture therefore becomes a **major cascade amplifier**.

### Water Stress Mapping

A simple linear stress function is practical but rigid:

$$f_W = \text{clip}\left(\frac{\theta - \theta_{\text{wp}}}{\theta_{\text{fc}} - \theta_{\text{wp}}},\; f_{\min},\; f_{\max}\right)$$

where:

|        Symbol        | Description                      | Units         |
| :------------------: | :------------------------------- | :------------ |
|       $\theta$       | Current volumetric soil moisture | m3/m3         |
| $\theta_{\text{wp}}$ | Wilting point                    | m3/m3         |
| $\theta_{\text{fc}}$ | Field capacity                   | m3/m3         |
|      $f_{\min}$      | Minimum stress factor            | dimensionless |
|      $f_{\max}$      | Maximum stress factor            | dimensionless |

This is often too inflexible when soil hydraulic traits are weakly constrained.

### Stomatal Conductance Proxy

The classical Ball-Berry style linkage is useful but simplified. In the tutorial baseline, the stomatal layer is intentionally lightweight, which makes it an appropriate surrogate target.

### Ozone Stress

Ozone damage and detoxification capacity are difficult to constrain at tropical sites. A surrogate is acceptable here because it is explicitly limited to the uncertain link, not the entire carbon-BVOC chain.

---

## The Hybrid ML Formulation

> [!IMPORTANT]
> The ML path is **not** an unconstrained deep model from meteorology to isoprene. It is a **structured comparison path** that preserves the physiological cascade.

### Structured Hybrid Pipeline

```mermaid
graph LR
    SM["Soil Moisture<br/>(surrogate)"]
    WS["Water Stress<br/>(surrogate)"]
    FQ["Farquhar<br/>(mechanistic)"]
    GS["Stomatal<br/>(surrogate)"]
    O3["Ozone Stress<br/>(surrogate)"]
    DA["Diurnal Activity<br/>(surrogate)"]
    MG["MEGAN<br/>(mechanistic)"]
    RES["Residual<br/>(bounded)"]

    SM --> WS --> FQ --> GS --> O3 --> DA --> MG --> RES

    classDef mech fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20,stroke-width:2px
    classDef surr fill:#ffe0b2,stroke:#e65100,color:#bf360c,stroke-width:1.5px

    class SM,WS,GS,O3,DA,RES surr
    class FQ,MG mech
```

This means:

- **Photosynthesis** is still mechanistic
- **BVOC emission physics** is still mechanistic
- The **residual correction** only adjusts the final isoprene magnitude _after_ the mechanistic chain has already run

> [!CAUTION]
> A direct black-box isoprene model would be more flexible, but it would also **erase the causal interpretability** of the chain. The residual design explicitly prevents this.

---

## Why This Fits RELIC Better Than a Standalone Surrogate Workflow

RELIC is not only a simulator. It is a **structured scientific execution system**.

The Tapajos case takes advantage of that by keeping the full chain compatible with:

- Functor metadata
- Execution scheduling
- Parameter SSOT
- Checkpointing
- Provenance tracking
- Differentiable optimization

The result is a workflow where **both physics and ML remain inspectable**.

---

## Main Tradeoffs

### Tradeoff 1: Fidelity vs Interpretability

A direct end-to-end neural network might improve one metric quickly, but it would hide where the improvement comes from. RELIC instead keeps the high-value process cores explicit.

### Tradeoff 2: Proxy Targets vs Real Observations

The case contains mixed-quality targets. Soil moisture and net photosynthesis are not equally trustworthy. The architecture therefore gives the isoprene end point more interpretive weight than the photosynthesis proxy alone.

### Tradeoff 3: Small-Sample Learning vs Model Capacity

Only a 375-step window is available. That is why the current surrogate enhancement remains moderate:

- **ResNet** for the most important surrogates
- **MLP** for lighter surrogates
- **Dropout Monte Carlo** spread through `get_prediction_uncertainty()`;
  the case sets `uncertainty_quantification=False` and publishes no
  `*_uncertainty` state or diagnostic outputs
- **Ordered training** rather than shuffled temporal batches

This is enough to improve representational flexibility without inventing a second training framework outside RELIC.

---

## RELIC Advantages Highlighted by This Case

> [!TIP]
> The most important conceptual advantage is not "ML everywhere" --- it is **"ML only where the archived data and model structure justify it"**.

1. Physics and ML paths can be **run together** without duplicating the whole model
2. The same run stores forcing, observations, checkpoints, monitors, optimized parameters, and figures **in one place**
3. Process replacement decisions are **transparent** and can be justified module by module
4. The simulator remains **differentiable** and compatible with gradient-based calibration
5. The parameter database remains the **single source of truth** rather than scattering constants into Python code

---

## Practical Interpretation

> [!IMPORTANT]
> That is the central reason this tutorial is a better fit for RELIC than a pure empirical emulator.

The most important conceptual advantage of the case is not "ML everywhere". It is **"ML only where the archived data and model structure justify it"**.
