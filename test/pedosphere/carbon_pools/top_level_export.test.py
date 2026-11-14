"""Top-level export checks for pedosphere carbon pools."""

from relic.scientific import pedosphere


def test_carbon_pools_functor_exported_from_pedosphere_package() -> None:
    """CarbonPoolsFunctor should be available at pedosphere top-level API."""
    assert "CarbonPoolsFunctor" in pedosphere.__all__
    assert hasattr(pedosphere, "CarbonPoolsFunctor")

