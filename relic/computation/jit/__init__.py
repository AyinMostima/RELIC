"""Explicit PyTorch callable-compilation adapter surface.

GraphIR compilation remains owned by ``relic.orchestration.compile_graph``.
This package exposes no RELIC compiler object, callable cache, global policy,
scientific-kernel wrapper, decorator, context, or compilation statistics.
"""

from .compile import maybe_compile


__all__ = ["maybe_compile"]
