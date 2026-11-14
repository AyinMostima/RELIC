"""Registry/bootstrap helpers for CrossDomainValidator."""

from __future__ import annotations

from relic.data.standards.cf_conventions import CFConventions


def infer_domain_from_name(var_name: str, domain_type) -> object:
    """Infer scientific domain from variable name heuristics."""
    name = var_name.lower()
    if any(token in name for token in ("air_", "atmos", "wind", "humidity", "pressure", "radiation", "vapor")):
        return domain_type.ATMOSPHERE
    if any(token in name for token in ("soil_", "decomposition", "nitrification", "leaching", "pedo")):
        return domain_type.PEDOSPHERE
    if any(token in name for token in ("evapo", "runoff", "infiltration", "moisture", "hydro", "water", "precip")):
        return domain_type.HYDROSPHERE
    if any(token in name for token in ("gpp", "npp", "stomatal", "leaf", "biomass", "transpiration", "photo")):
        return domain_type.BIOSPHERE
    if any(token in name for token in ("anthrop", "fertilizer", "irrigation", "land_use", "management", "emission")):
        return domain_type.ANTHROPOSPHERE
    return domain_type.TRANSPORT


def initialize_standard_variables(validator, domain_type, variable_spec) -> None:
    """Initialize validator variable registry from StateManager schema."""
    if validator.state_manager is None or not hasattr(validator.state_manager, "schema"):
        return

    schema = getattr(validator.state_manager, "schema", {}) or {}
    if not hasattr(schema, "items"):
        return
    try:
        schema_entries = list(schema.items())
    except (TypeError, ValueError, AttributeError):
        return
    for var_name, spec in schema_entries:
        if not isinstance(var_name, str):
            continue

        domain = infer_domain_from_name(var_name, domain_type)
        units = getattr(spec, "units", None)
        if (
            not isinstance(units, str)
            or not units
            or units != units.strip()
        ):
            raise ValueError(
                f"State variable '{var_name}' requires one exact unit string"
            )
        bounds = getattr(spec, "bounds", None)
        description = getattr(spec, "description", "")
        aliases = list(getattr(spec, "aliases", []) or [])

        cf_standard_name = getattr(spec, "cf_standard_name", None)
        if cf_standard_name is not None:
            if not CFConventions.is_valid_standard_name(cf_standard_name):
                raise ValueError(
                    f"State variable '{var_name}' declares a non-v94 "
                    f"cf_standard_name: {cf_standard_name!r}"
                )
            canonical_units = CFConventions.get_standard_units(cf_standard_name)
            if (
                canonical_units is not None
                and not CFConventions.are_standard_name_units_equivalent(
                    cf_standard_name,
                    units,
                )
            ):
                raise ValueError(
                    f"State variable '{var_name}' units {units!r} are not "
                    f"compatible with v94 canonical units {canonical_units!r}"
                )

        validator.register_variable(
            variable_spec(
                name=var_name,
                units=units,
                domain=domain,
                cf_standard_name=cf_standard_name,
                bounds=bounds,
                description=description,
                aliases=aliases,
            )
        )


def initialize_interface_contracts(validator, domain_type, interface_contract, variable_spec) -> None:
    """Initialize baseline interface contracts between major domains."""
    contracts = [
        interface_contract(
            domain_type.ATMOSPHERE, domain_type.HYDROSPHERE,
            variables=[
                variable_spec("net_radiation", "W m-2", domain_type.ATMOSPHERE),
                variable_spec("aerodynamic_resistance", "s m-1", domain_type.ATMOSPHERE),
                variable_spec("vapor_pressure_deficit", "Pa", domain_type.ATMOSPHERE),
            ],
            validation_rules=["aerodynamic_resistance > 0", "net_radiation > -500"],
        ),
        interface_contract(
            domain_type.HYDROSPHERE, domain_type.PEDOSPHERE,
            variables=[
                variable_spec("infiltration_rate", "kg m-2 s-1", domain_type.HYDROSPHERE),
                variable_spec("drainage_flux", "kg m-2 s-1", domain_type.HYDROSPHERE),
            ],
            conversion_factors={"water_to_volume": 0.001},
            validation_rules=["infiltration_rate >= 0", "drainage_flux >= 0"],
        ),
        interface_contract(
            domain_type.BIOSPHERE, domain_type.ATMOSPHERE,
            variables=[
                variable_spec("co2_flux", "g C m-2 s-1", domain_type.BIOSPHERE),
                variable_spec("bvoc_emissions", "g C m-2 s-1", domain_type.BIOSPHERE),
                variable_spec("transpiration", "kg m-2 s-1", domain_type.BIOSPHERE),
            ],
            validation_rules=["co2_flux can be negative (uptake)", "bvoc_emissions >= 0"],
        ),
        interface_contract(
            domain_type.PEDOSPHERE, domain_type.ATMOSPHERE,
            variables=[
                variable_spec("soil_co2_flux", "g C m-2 s-1", domain_type.PEDOSPHERE),
                variable_spec("n2o_emissions", "g N m-2 s-1", domain_type.PEDOSPHERE),
                variable_spec("no_emissions", "g N m-2 s-1", domain_type.PEDOSPHERE),
            ],
            validation_rules=["all emissions >= 0"],
        ),
        interface_contract(
            domain_type.ANTHROPOSPHERE, domain_type.PEDOSPHERE,
            variables=[
                variable_spec("fertilizer_input", "g N m-2 s-1", domain_type.ANTHROPOSPHERE),
                variable_spec("tillage_depth", "m", domain_type.ANTHROPOSPHERE),
            ],
            validation_rules=["fertilizer_input >= 0", "0 <= tillage_depth <= 0.5"],
        ),
    ]

    for contract in contracts:
        validator.register_interface_contract(contract)


def integrate_hydrosphere_mappings(validator, domain_type, variable_spec) -> None:
    """Register hydrosphere interface mappings into variable registry."""
    from relic.scientific.hydrosphere.interface_mapping import ALL_MAPPINGS

    for mapping in ALL_MAPPINGS:
        validator.register_variable(
            variable_spec(
                name=mapping.standard_name,
                units=mapping.units,
                domain=domain_type.HYDROSPHERE,
                cf_standard_name=mapping.cf_standard_name,
                description=mapping.description,
                aliases=list(mapping.aliases),
            )
        )


__all__ = [
    "infer_domain_from_name",
    "initialize_standard_variables",
    "initialize_interface_contracts",
    "integrate_hydrosphere_mappings",
]
