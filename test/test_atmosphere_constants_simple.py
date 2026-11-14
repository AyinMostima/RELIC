"""
Simple test for atmosphere module parameter loading.

This simplified test focuses on testing parameter loading without
complex dependencies.
"""

def test_turbulence_parameter_loading():
    """Test that turbulence module can load parameters."""
    from relic.scientific.atmosphere import turbulence

    von_karman = turbulence.get_physical_constant("von_karman_constant")
    gravity = turbulence.get_physical_constant("gravity_acceleration")
    air_density = turbulence.get_physical_constant("relic.physical.constants.air.density_reference")
    specific_heat = turbulence.get_physical_constant("air_specific_heat_cp")

    assert 0.35 <= von_karman <= 0.45
    assert 9.7 <= gravity <= 9.9
    assert 1.0 <= air_density <= 1.4
    assert 1000 <= specific_heat <= 1010

def test_stability_parameters_loading():
    """Test that stability_parameters module can load parameters."""
    from relic.scientific.atmosphere.most import stability_parameters

    von_karman = stability_parameters.get_physical_constant("von_karman_constant")
    gravity = stability_parameters.get_physical_constant("gravity_acceleration")

    assert 0.35 <= von_karman <= 0.45
    assert 9.7 <= gravity <= 9.9

def test_chemistry_parameter_loading():
    """Test that chemistry module can load parameters."""
    from relic.scientific.atmosphere import chemistry

    avogadro = chemistry.get_physical_constant("avogadro_number")
    boltzmann = chemistry.get_physical_constant("boltzmann_constant")
    ref_pressure = chemistry.get_physical_constant("reference_pressure")
    ref_temp = chemistry.get_physical_constant("reference_temperature")

    assert 6.0e23 <= avogadro <= 6.1e23
    assert 1.3e-23 <= boltzmann <= 1.4e-23
    assert 100000 <= ref_pressure <= 102000
    assert 298.0 <= ref_temp <= 299.0

def test_parameter_consistency():
    """Test that MOST operators use the physical-constant SSOT."""
    from relic.scientific.atmosphere.most import stability_parameters, surface_fluxes
    from relic.utils.physical_constants import get_physical_constant

    assert stability_parameters.VON_KARMAN_CONSTANT == get_physical_constant("von_karman_constant")
    assert surface_fluxes.VON_KARMAN_CONSTANT == get_physical_constant("von_karman_constant")
    assert stability_parameters.GRAVITY_ACCELERATION == get_physical_constant("gravity_acceleration")
    # The MOST fluxes take the caller's moist-air density; there is no module density.
    assert not hasattr(surface_fluxes, "AIR_DENSITY_STANDARD")
    assert surface_fluxes.SPECIFIC_HEAT_AIR == get_physical_constant("air_specific_heat_cp")

