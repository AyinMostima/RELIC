# RELIC Extension Interface

This package provides a non-invasive extension layer for RELIC.
All extension logic is isolated under `relic\extensions\` and can be used
without modifying `relic\core\`, `relic\scientific\`, or `relic\orchestration\`.

## Capabilities

- Declare extension functors with `relic_extension.yaml`.
- Discover extensions from:
  - explicit directories (`extra_dirs`)
  - environment variable `RELIC_EXTENSION_DIRS` (semicolon-separated)
  - Python entry points group `relic.extensions`
- Validate extension contracts before registration.
- Dynamically load extension functors.
- Optionally register loaded functors into:
  - `RELICSimulator` via existing `register_functor(...)`
  - orchestration `NodeRegistry` via existing `register_functor_instance(...)`
- Scaffold a new extension project with CLI templates.

## Manifest Format

Each extension root must contain `relic_extension.yaml`.

```yaml
name: "my_custom_extension"
version: "0.1.0"
author: "Researcher"
description: "Custom transport process"
min_relic_version: "1.0.0"
functors:
  - class: "functor.CustomTransportFunctor"
    bridge: "TransportMechanismFunctor"
    category: "transport_mechanism"
    execution_group: "fast"
    config: {}
parameters:
  - path: "parameters.yaml"
dependencies:
  required: []
  optional: []
```

## Python API

```python
from pathlib import Path

from relic.extensions import discover_extensions, validate_extension, load_extensions
from relic.extensions.hooks import load_and_register_extensions
from relic.core.simulator import RELICSimulator
from relic.parameters import get_global_manager

infos = discover_extensions(extra_dirs=[Path(r".\my_extension")])
for info in infos:
    report = validate_extension(info)
    print(info.name, report.valid, report.errors, report.warnings)

loaded = load_extensions(extra_dirs=[Path(r".\my_extension")], validate=True)

sim = RELICSimulator(config={})
summary = load_and_register_extensions(
    sim,
    parameter_manager=get_global_manager(),
    extra_dirs=[Path(r".\my_extension")],
)
print(summary)
```

## CLI Usage

Run as standalone module:

```powershell
python -m relic.extensions.cli scaffold my_extension --bridge transport_mechanism --output .\my_extension
python -m relic.extensions.cli list
python -m relic.extensions.cli validate .\my_extension
python -m relic.extensions.cli info my_extension
```

## Notes

- Keep extension module names unique to avoid Python import collisions.
- Prefer package-style extension modules for large projects.
- Validation enforces bridge inheritance, metadata quality, forward signature,
  dependency declaration, and parameter file checks.

