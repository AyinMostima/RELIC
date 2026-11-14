"""Export governance tests for BVOC synthesis package."""

from relic.scientific.biosphere.bvoc import synthesis
from relic.scientific.biosphere.bvoc.synthesis import sesquiterpene


def test_placeholder_functors_not_in_default_exports() -> None:
    """Placeholder sesquiterpene functors must not be publicly exported by default."""
    assert "SesquiterpeneSynthesis" not in synthesis.__all__
    assert "SesquiterpeneStressModel" not in synthesis.__all__


def test_placeholder_module_remains_directly_importable() -> None:
    """Direct module import remains available for explicit experimental usage."""
    assert hasattr(sesquiterpene, "SesquiterpeneSynthesis")
    assert hasattr(sesquiterpene, "SesquiterpeneStressModel")

