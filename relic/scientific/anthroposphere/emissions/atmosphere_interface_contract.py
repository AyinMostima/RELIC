"""Recognized atmospheric chemistry mechanism identifiers.

Only ``ChemicalMechanism.CB05`` currently has an executable SpeciesMapper
contract. The remaining identifiers are configuration-domain values whose
public SpeciesMapper paths fail closed until complete species, unit,
conservation, and runtime bindings exist.

Variable metadata remains owned by the SCM ContractRegistry. Speciation and
mass-to-molar conversion remain owned by SpeciesMapper. This module defines no
second variable schema, validator, conversion runtime, DEC structure, or port.
"""

from enum import Enum


class ChemicalMechanism(Enum):
    """Recognized mechanism identifiers; only CB05 is executable."""

    CB05 = "cb05"
    CB6R3 = "cb6r3"
    SAPRC07 = "saprc07"
    MOZART4 = "mozart4"
    RACM2 = "racm2"
    MCM = "mcm"


__all__ = ["ChemicalMechanism"]
