"""Stateless PyTorch callable-compilation adapter.

GraphIR compilation is owned by ``relic.orchestration.compile_graph``. This
module only forwards one explicit request to ``torch.compile``. PyTorch owns
its code-object cache and guards; RELIC maintains no callable cache, compiler
singleton, global policy, decorator, context, warmup, or compilation stats.

Scientific numerical-equivalence validation remains ``[N]``. A generic Python
callable cannot be proven pure or transactionally replayed from declarations,
closure scans, or stochastic-operation blacklists. Requests for such validation
therefore fail before compilation or callable execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


def _normalize_mode(mode: Any) -> str:
    """Normalize a string or existing enum value without owning a taxonomy."""
    raw_mode = getattr(mode, "value", mode)
    normalized = str(raw_mode).strip().lower()
    if normalized not in {"auto", "compile", "off"}:
        raise ValueError(
            "mode must be one of 'auto', 'compile', or 'off'; "
            "TorchScript modes are not supported"
        )
    return normalized


def maybe_compile(
    func_or_module: Callable[..., Any],
    mode: Any = "auto",
    example_inputs: tuple[Any, ...] | None = None,
    allow_uncompiled: bool = False,
    *,
    backend: str | Callable[..., Any] = "inductor",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    compile_mode: str | None = "default",
    options: Mapping[str, Any] | None = None,
    validate_numerics: bool = False,
) -> Callable[..., Any]:
    """Return an explicitly configured ``torch.compile`` callable.

    Parameters are local to this invocation. The adapter never executes the
    original or compiled callable and never performs warmup. ``example_inputs``
    is accepted for benchmark-call compatibility but is not executed or used as
    scientific evidence.

    ``allow_uncompiled`` applies only to explicit ``mode='off'``. Compilation
    failures in active modes propagate and never fall back to eager execution.
    """
    if not callable(func_or_module):
        raise TypeError("func_or_module must be callable")
    normalized_mode = _normalize_mode(mode)
    if not isinstance(allow_uncompiled, bool):
        raise TypeError("allow_uncompiled must be bool")
    if not isinstance(fullgraph, bool):
        raise TypeError("fullgraph must be bool")
    if dynamic is not None and not isinstance(dynamic, bool):
        raise TypeError("dynamic must be bool or None")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping or None")
    if not isinstance(validate_numerics, bool):
        raise TypeError("validate_numerics must be bool")

    if validate_numerics:
        raise RuntimeError(
            "[N] scientific callable numerical-equivalence validation is not "
            "implemented"
        )

    if normalized_mode == "off":
        if allow_uncompiled:
            return func_or_module
        raise RuntimeError("JIT compilation is disabled")

    _ = example_inputs
    compile_kwargs: dict[str, Any] = {
        "backend": backend,
        "fullgraph": fullgraph,
    }
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    if options is not None:
        compile_kwargs["options"] = dict(options)

    return torch.compile(func_or_module, **compile_kwargs)


__all__ = ["maybe_compile"]
