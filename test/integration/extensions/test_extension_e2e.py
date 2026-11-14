from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import torch
import yaml

from relic.core.simulator import RELICSimulator
from relic.extensions.discovery import discover_extensions
from relic.extensions.hooks import (
    load_and_register_extensions,
    register_with_orchestration,
    unload_from_runtime,
)
from relic.extensions.loader import load_extensions
from relic.extensions.protocol import MANIFEST_FILENAME
from relic.extensions.validator import validate_extension
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog
from relic.parameters import get_global_manager


def _create_temp_extension(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    (root / "functor.py").write_text(
        dedent(
            """
            import torch
            from relic.core.functor import (
                FunctorMetadata,
                ProcessCategory,
                SpatialScale,
                TransportMechanismFunctor,
                VariableKind,
            )

            class CustomTransport(TransportMechanismFunctor):
                TRANSPORT_TYPE = "diffusion"

                def __init__(self, config=None, name="custom_transport", **kwargs):
                    metadata = FunctorMetadata(
                        name=name,
                        category=ProcessCategory.TRANSPORT_MECHANISM,
                        description="custom extension transport functor",
                        references=["Fick (1855)."],
                        input_variables=["concentration_gradient"],
                        output_variables=["custom_extension_flux"],
                        spatial_scale=SpatialScale.GRID_CELL,
                        variable_kind=VariableKind.FLUX,
                    )
                    super().__init__(name=name, config=config or {}, metadata=metadata, **kwargs)

                def get_dependencies(self):
                    return []

                def forward(self, state, time, dt, **kwargs):
                    gradient = state.get("concentration_gradient")
                    if gradient is None:
                        gradient = torch.tensor(0.0)
                    return {"custom_extension_flux": gradient}
            """
        ),
        encoding="utf-8",
    )

    manifest = {
        "name": "custom_e2e_extension",
        "version": "0.1.0",
        "author": "integration-test",
        "description": "e2e extension test",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": "functor.CustomTransport",
                "bridge": "TransportMechanismFunctor",
                "category": "transport_mechanism",
                "execution_group": "fast",
            }
        ],
        "parameters": [{"path": "parameters.yaml"}],
        "dependencies": {"required": [], "optional": []},
    }
    (root / MANIFEST_FILENAME).write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (root / "parameters.yaml").write_text(
        "schema_version: 2.0.0\nparameters: {}\n",
        encoding="utf-8",
    )
    return root


def test_extension_e2e(tmp_path: Path) -> None:
    extension_root = _create_temp_extension(tmp_path / "custom_e2e_extension")

    discovered = discover_extensions(
        extra_dirs=[extension_root],
        use_entry_points=False,
        use_env_dirs=False,
    )
    assert len(discovered) == 1
    assert discovered[0].name == "custom_e2e_extension"

    validation_result = validate_extension(discovered[0])
    assert validation_result.valid is True

    loaded = load_extensions(
        extra_dirs=[extension_root],
        validate=True,
        strict=True,
    )
    assert "custom_transport" in loaded

    functor = loaded["custom_transport"]
    outputs = functor.forward(
        {"concentration_gradient": torch.tensor(2.5)},
        time=0.0,
        dt=1.0,
    )
    assert "custom_extension_flux" in outputs

    simulator = RELICSimulator(config={})
    register_summary = load_and_register_extensions(
        simulator,
        parameter_manager=get_global_manager(),
        extra_dirs=[extension_root],
        validate=True,
        strict=True,
    )
    assert register_summary["registered_count"] == 1
    assert "custom_transport" in simulator.functors

    registry = NodeRegistry(ssot_catalog=SSOTCatalog.from_parameter_manager())
    node_type_ids = register_with_orchestration(
        registry,
        extra_dirs=[extension_root],
        validate=True,
        strict=True,
    )
    assert len(node_type_ids) == 1
    assert node_type_ids[0].startswith("Functor:")

    unload_summary = unload_from_runtime(
        simulator=simulator,
        extension_name="custom_e2e_extension",
        registry=registry,
        strict=True,
    )
    assert "custom_transport" in unload_summary["removed_functors"]
    assert "custom_transport" not in simulator.functors
    assert len(unload_summary["removed_node_types"]) == 1
