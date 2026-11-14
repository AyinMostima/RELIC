# Tapajos BVOC Case Documentation Index

> [!NOTE]
> **Canonical Reference Run:** `dual_path_full_20260306_v1`
> | Item | Value |
> |---|---|
> | **Site** | Tapajos KM67 (`BR-Sa1`) |
> | **Window** | 2014-06-01 ~ 2014-06-16 (375 hourly steps) |
> | **Mode** | Physics + ML-hybrid dual-path optimization |
> | **Phases** | 4-phase sequential calibration |
> | **Wall time** | ~1.86 hours |

---

## Document Map

```mermaid
graph LR
    A["00 Index"]
    B["01 Functor<br/>Topology"]
    C["02 Input Data<br/>Catalog"]
    D["03 Theory &<br/>Tradeoffs"]
    E["04 Reference<br/>Run Results"]
    F["05 Literature<br/>Comparison"]
    G["06 Usage<br/>Guidance"]
    H["07 Outputs &<br/>Monitoring"]

    A --> B --> C --> D --> E --> F --> G --> H

    style A fill:#4a90d9,stroke:#2c5f8a,color:#fff,stroke-width:2px
    style B fill:#e8f4fd,stroke:#4a90d9,color:#1a3a5c
    style C fill:#e8f4fd,stroke:#4a90d9,color:#1a3a5c
    style D fill:#fff3e0,stroke:#e8a317,color:#5a3e00
    style E fill:#e8f5e9,stroke:#4caf50,color:#1b5e20
    style F fill:#fff3e0,stroke:#e8a317,color:#5a3e00
    style G fill:#f3e5f5,stroke:#9c27b0,color:#4a148c
    style H fill:#f3e5f5,stroke:#9c27b0,color:#4a148c
```

---

## Chapter Structure

|  #  | Chapter                   | Category     | Description                                                                                       |
| :-: | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------- |
| 00  | **Case Index**            | Navigation   | This index and quick navigation page                                                              |
| 01  | **Functor Topology**      | Architecture | Physics and ML-hybrid functor graph; why each process is kept physical or replaced by a surrogate |
| 02  | **Input Data Catalog**    | Data         | Detailed source inventory, metadata, units, coverage, and processing logic                        |
| 03  | **Theory & Tradeoffs**    | Science      | Scientific interpretation of the workflow and why the hybrid design matches RELIC                 |
| 04  | **Reference Run Results** | Results      | Actual outputs, optimization summaries, monitors, and metrics from the `v1` reference             |
| 05  | **Literature Comparison** | Validation   | Comparison with canonical literature and site-scale BVOC evaluation studies                       |
| 06  | **Usage Guidance**        | Template     | Practical recommendations and code/config templates for building new RELIC cases                  |
| 07  | **Outputs & Monitoring**  | Operations   | Detailed explanation of NetCDF outputs, CSV monitors, checkpoints, and RELIC core diagnostics     |

---

## Recommended Reading Order

> [!TIP]
> If you are **new to the case**, read chapters 01-07 sequentially.
> If you are **building a new case**, start with **06** (templates), then read **01** (topology) and **03** (theory).

---

## Key Files in the Executable Case Directory

```text
tutorial/case/
  +-- README.md                  # Case overview
  +-- run_case.py                # Main entry point
  +-- plot_results.py            # Publication figure generator
  +-- analyze_results.py         # Post-run analysis
  +-- config/                    # YAML configuration files
  +-- data/                      # Raw and processed input data
  +-- functors/                  # Custom RELIC functors for this case
  +-- output/                    # All run outputs
```

---

## Reference Run Snapshot

The reference run `dual_path_full_20260306_v1` provides:

- A full **375-step** optimization and final forward pass
- Phase loss histories for **phase1** through **phase4**
- Split **physics** and **ML** time series outputs
- Split **physics** and **ML** process monitors
- Archived **ML checkpoints** and training histories
- A full **input snapshot** and provenance report

> [!IMPORTANT]
> The documentation uses `dual_path_full_20260306_v1` as the **canonical best-result data source** for this tutorial. All core descriptions, result tables, and workflow recommendations are anchored to that run.
