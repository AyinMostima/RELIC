# Anthropogenic Emissions Chemical Mechanism and Interface Mapping

## 1. Process Definition and Scope

`SpeciesMapper` (`relic/scientific/anthroposphere/emissions/species_mapping.py`) maps sector-disaggregated inventory mass fluxes (`sector_emission_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$) across five canonical sectors (`power`, `transport`, `industry`, `buildings`, `agriculture`) and six inventory species (`NOx`, `CO`, `SO2`, `VOC`, `NH3`, `PM25`) into Carbon Bond 2005 (`CB05`) gas molar fluxes (`mechanism_gas_molar_flux`, $\mathrm{mol\,m^{-2}\,s^{-1}}$) and speciated $\mathrm{PM_{2.5}}$ aerosol mass fluxes (`mechanism_aerosol_mass_flux`, $\mathrm{kg\,m^{-2}\,s^{-1}}$).

`ChemicalMechanism` (`relic/scientific/anthroposphere/emissions/atmosphere_interface_contract.py`) enumerates mechanism identifiers: `CB05` is `[I]`, whereas `CB6R3`, `SAPRC07`, `MOZART4`, `RACM2`, and `MCM` raise `NotImplementedError` (`[N]`). Historical `AtmosphereInterfaceContract` has been removed (`[X]`). `AnthroposphereInterfaceMapping` (`relic/scientific/anthroposphere/interface_mapping.py`) is fail-closed `[N]` at IPHS **Level 0** (`[N] runtime_metadata_domain_unbound`) because canonical variable metadata and CF conventions belong exclusively to the SCM `ContractRegistry`. Deeper derivations appear in *Methods of Ecological Analysis*, Chapter 24 (Section 24.3.2).

## 2. Governing Equations

### 2.1 Sector-Specific Gas and Aerosol Mass Speciation

Let $F^{\mathrm{sec}}_{s,k} \ge 0$ ($\mathrm{kg\,m^{-2}\,s^{-1}}$) denote `sector_emission_flux` on leading shape `[5, 6]`. Inventory $\mathrm{NO}_x$ is reported on an $\mathrm{NO_2}$-equivalent mass basis ($M_{\mathrm{NOx}}$) and split by molar fraction $f_{\mathrm{NO}} \in [0, 1]$ while preserving total reactive nitrogen moles:

$$
F_{\mathrm{NO},s} = f_{\mathrm{NO}}\,F^{\mathrm{sec}}_{s,\mathrm{NOx}}\,\frac{M_{\mathrm{NO}}}{M_{\mathrm{NOx}}}, \qquad
F_{\mathrm{NO_2},s} = (1 - f_{\mathrm{NO}})\,F^{\mathrm{sec}}_{s,\mathrm{NOx}}\,\frac{M_{\mathrm{NO_2}}}{M_{\mathrm{NOx}}}, \qquad
F_{g,s} = F^{\mathrm{sec}}_{s,g} \quad (g \in \{\mathrm{CO}, \mathrm{SO_2}, \mathrm{NH_3}\})
$$

Sector VOC mass splits across six CB05 lumped species $v \in \{\mathrm{PAR}, \mathrm{OLE}, \mathrm{TOL}, \mathrm{XYL}, \mathrm{FORM}, \mathrm{ALD2}\}$ using sector key $\sigma_{\mathrm{voc}}(s) \in \{\text{power\_plant}, \text{transport}, \text{industrial}, \text{residential}, \text{agriculture}\}$ with $\sum_v f^{\mathrm{VOC}}_{\sigma_{\mathrm{voc}}(s),v} = 1$:

$$
F_{v,s} = f^{\mathrm{VOC}}_{\sigma_{\mathrm{voc}}(s),v}\,F^{\mathrm{sec}}_{s,\mathrm{VOC}}
$$

Sector $\mathrm{PM_{2.5}}$ mass splits across five parameterized components $a \in \{\mathrm{BC}, \mathrm{OC}, \mathrm{NO3}, \mathrm{SO4}, \mathrm{NH4}\}$ using sector key $\sigma_{\mathrm{pm}}(s) \in \{\text{power}, \text{transport}, \text{industrial}, \text{residential}, \text{agriculture}\}$ and closes the unspeciated `OTHER` residual by exact subtraction:

$$
F_{a,s} = f^{\mathrm{PM}}_{\sigma_{\mathrm{pm}}(s),a}\,F^{\mathrm{sec}}_{s,\mathrm{PM25}}, \qquad
F_{\mathrm{OTHER},s} = \left(1 - \sum_{a \in \{\mathrm{BC},\mathrm{OC},\mathrm{NO3},\mathrm{SO4},\mathrm{NH4}\}} f^{\mathrm{PM}}_{\sigma_{\mathrm{pm}}(s),a}\right) F^{\mathrm{sec}}_{s,\mathrm{PM25}}
$$

### 2.2 Positive Inverse Molar-Mass Hodge Conversion and Auxiliary Mixing-Ratio Helper

Summing over sectors $s$ and applying the positive diagonal inverse molar-mass Hodge operator $\star_{M^{-1}} = \operatorname{diag}(1 / M_g)$ with $M_g = 10^{-3}\tilde{M}_g$ ($\mathrm{kg\,mol^{-1}}$) yields the 11-species CB05 gas molar vector (`NO`, `NO2`, `CO`, `SO2`, `NH3`, `PAR`, `OLE`, `TOL`, `XYL`, `FORM`, `ALD2`) and 6-species aerosol mass vector (`BC`, `OC`, `NO3`, `SO4`, `NH4`, `OTHER`):

$$
F^{\mathrm{mol}}_g = \frac{1}{M_g}\sum_{s=1}^{5} F_{g,s}, \qquad
F^{\mathrm{aer}}_a = \sum_{s=1}^{5} F_{a,s}
$$

Standalone helper functions `flux_to_mixing_ratio` and `mixing_ratio_to_flux` evaluate ideal-gas boundary-layer tendency $\Delta \chi = F^{\mathrm{mol}}\,\Delta t\,R\,T / (P\,H)$ using universal gas constant $R$, air temperature $T$, pressure $P$, and mixing-layer depth $H$, outside `SpeciesMapper.forward()`.

## 3. Variable Table

| Symbol | Meaning | Units | Code Name | Role |
|---|---|---|---|---|
| $F^{\mathrm{sec}}_{s,k}$ | Sector-by-species anthropogenic mass flux (`[5, 6, ...]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `sector_emission_flux` | `flux` |
| $F^{\mathrm{mol}}_g$ | Speciated CB05 gas molar emission flux (`[11, ...]`) | $\mathrm{mol\,m^{-2}\,s^{-1}}$ | `mechanism_gas_molar_flux` | `flux` |
| $F^{\mathrm{aer}}_a$ | Speciated $\mathrm{PM_{2.5}}$ aerosol mass emission flux (`[6, ...]`) | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `mechanism_aerosol_mass_flux` | `flux` |
| $F_{\mathrm{in}}, F_{\mathrm{out}}$ | Total inventory input and physical output mass fluxes | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.species_mapping.total_input_mass_flux`, `diagnostics.species_mapping.total_output_mass_flux` | `diagnostic` |
| $F_{\mathrm{eq}}, \Delta F_{\mathrm{NOx}}$ | $\mathrm{NO_2}$-equivalent output mass flux and basis adjustment | $\mathrm{kg\,m^{-2}\,s^{-1}}$ | `diagnostics.species_mapping.inventory_equivalent_output_mass_flux`, `diagnostics.species_mapping.inventory_basis_adjustment_flux` | `diagnostic` |
| $\epsilon_{\mathrm{rel}}, c_{\mathrm{iphs}}$ | Pointwise relative mass-balance residual and canonical runtime flag | $\mathrm{1}$ | `diagnostics.species_mapping.relative_error`, `diagnostics.species_mapping.canonical_iphs_runtime` | `diagnostic` |

## 4. Parameter Table

| Symbol | ParameterManager / Constant Key | Units | Default Source | Reference |
|---|---|---|---|---|
| $m_{\mathrm{chem}}, \tau_{\mathrm{mb}}$ | `anthroposphere.emissions.species_mapping.default_mechanism`, `anthroposphere.emissions.species_mapping.mass_balance_tolerance` | `str`, $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Yarwood et al. (2005) |
| $f_{\mathrm{NO}}$ | `anthroposphere.emissions.partition_factors.no_fraction` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Primary $\mathrm{NO}/\mathrm{NO}_x$ molar split |
| $\tilde{M}_g$ | `anthroposphere.emissions.molecular_weights.*` | $\mathrm{g\,mol^{-1}}$ | `anthroposphere/emissions_parameters.yaml` | CB05 species molar weights |
| $\tilde{M}_{\mathrm{CO}}, \tilde{M}_{\mathrm{FORM}}$ | `relic.physical.constants.molecular_weights.co`, `relic.physical.constants.molecular_weights.hcho` | $\mathrm{g\,mol^{-1}}$ | `physical/constants.yaml` | IUPAC molar weights |
| $f^{\mathrm{VOC}}_{\sigma,v}$ | `anthroposphere.emissions.speciation.voc_split_fractions.*` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Carter (2015); Yarwood et al. (2005) |
| $f^{\mathrm{PM}}_{\sigma,a}$ | `anthroposphere.emissions.pm_composition.*` | $\mathrm{1}$ | `anthroposphere/emissions_parameters.yaml` | Reff et al. (2009) |
| $R$ | `relic.physical.constants.gas_constant` | $\mathrm{J\,mol^{-1}\,K^{-1}}$ | `physical/constants.yaml` | CODATA universal gas constant |

## 5. Numerical Methods and Discretization

`SpeciesMapper.forward()` executes vectorized tensor contractions over arbitrary trailing spatial dimensions and emits `mechanism_gas_molar_flux`, `mechanism_aerosol_mass_flux`, and all six `diagnostics.species_mapping.*` fields via `UpdateItem(UpdateMode.REPLACE, per_timestep=False)`. Optional dependency `sector_allocation` orders `SectorAllocation` (`StateWriteMode.DERIVED`) ahead of `SpeciesMapper` whenever both are registered. Calling `upscale()` or `downscale()` raises `RuntimeError` because spatial remapping is separate from mechanism speciation.

## 6. Conservation, Boundary Conditions, and Ledgers

Physical mass differs from inventory mass solely by the oxygen mass difference between emitted $\mathrm{NO}$ and its $\mathrm{NO_2}$-equivalent reporting basis. Reconstructing inventory-equivalent output mass flux

$$
F_{\mathrm{eq}} = \sum_{g=1}^{11} F^{\mathrm{mol}}_g M_g + \sum_{a=1}^{6} F^{\mathrm{aer}}_a - \sum_{s=1}^{5}(F_{\mathrm{NO},s} + F_{\mathrm{NO_2},s}) + \left(F^{\mathrm{mol}}_{\mathrm{NO}} + F^{\mathrm{mol}}_{\mathrm{NO_2}}\right) M_{\mathrm{NOx}}
$$

yields relative ledger residual $\epsilon_{\mathrm{rel}} = \lvert F_{\mathrm{eq}} - F_{\mathrm{in}}\rvert / \max(F_{\mathrm{in}}, 10^{-30}) \le \tau_{\mathrm{mb}}$, verified on every step inside `forward()` and `check_conservation_laws()`.

## 7. Code Mapping

| Functor Class | File Path | Abstract Contract | Bridge Class | Execution Phase |
|---|---|---|---|---|
| `SpeciesMapper` | `relic/scientific/anthroposphere/emissions/species_mapping.py` | `ScaleCoupling` (`source_scale="inventory"`, `target_scale="mechanism"`, `coupling_type="hierarchical"`) | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | `ExecutionPhase.MAIN_PROCESSES` (`SpatialScale.GRID_CELL`) |
| `AnthroposphereInterfaceMapping` | `relic/scientific/anthroposphere/interface_mapping.py` | `ScaleCoupling` | `ScaleCouplingFunctor` (`relic/core/functor/scale_coupling_bridge.py`) | Fail-closed (`[N]`, unbound metadata domain) |

## 8. Implementation Status

- `SpeciesMapper`: `[I]`, IPHS **Level 3** (`scale_coupling`, certificate `anthroposphere_emissions_species_mapping_species_mapper.yaml`). Fixed CB05 typed chain map with positive diagonal inverse molar-mass Hodge conversion and exact $\mathrm{NO_2}$-equivalent mass ledger. Non-CB05 `ChemicalMechanism` enum members (`CB6R3`, `SAPRC07`, `MOZART4`, `RACM2`, `MCM`) are `[N]`.
- `AnthroposphereInterfaceMapping`: `[N]`, IPHS **Level 0** (`scale_coupling`, certificate `anthroposphere_interface_mapping_anthroposphere_interface_mapping.yaml`). Fail-closed (`runtime_metadata_domain_unbound`); variable metadata ownership resides in SCM `ContractRegistry`.

Frozen certificate expressions:

```text
Dynamic dictionary mapping with an implicit industry profile and mixed mass/molar payloads.
Fixed typed chain maps consume sector_emission_flux and emit separate gas-molar and aerosol-mass matter ports.
The removed metadata path did not establish a source-domain to target-domain chain map or a governed StateManager update contract.
[N] runtime_metadata_domain_unbound; no executable scientific expression is exposed.
```

## 9. References

- Carter, W. P. L. (2015). Development of a database for chemical mechanism assignments for volatile organic emissions. *Journal of the Air & Waste Management Association*, 65(10), 1171-1184.
- Reff, A., Bhave, P. V., Simon, H., et al. (2009). Emissions inventory of $\mathrm{PM_{2.5}}$ trace elements across the United States. *Environmental Science & Technology*, 43(15), 5790-5796.
- Yarwood, G., Rao, S., Yocke, M., and Whitten, G. Z. (2005). *Updates to the Carbon Bond Chemical Mechanism: CB05* (Final Report RT-0400675). US Environmental Protection Agency.
