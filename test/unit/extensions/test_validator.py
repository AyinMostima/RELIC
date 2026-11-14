from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import yaml

from relic.extensions.discovery import build_extension_info
from relic.extensions.protocol import MANIFEST_FILENAME
from relic.extensions.validator import validate_extension


def _write_manifest(root: Path, class_path: str, bridge: str, category: str) -> None:
    payload = {
        "name": "validator_demo",
        "version": "0.1.0",
        "author": "tester",
        "description": "validator demo",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": class_path,
                "bridge": bridge,
                "category": category,
                "execution_group": "fast",
            }
        ],
        "parameters": [{"path": "parameters.yaml"}],
        "dependencies": {"required": [], "optional": []},
    }
    (root / MANIFEST_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (root / "parameters.yaml").write_text(
        "schema_version: 2.0.0\nparameters: {}\n",
        encoding="utf-8",
    )


def _write_module(root: Path, content: str) -> None:
    (root / "functor_impl.py").write_text(dedent(content), encoding="utf-8")


def test_validator_accepts_valid_functor(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        import torch
        from relic.core.functor import (
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            TransportMechanismFunctor,
            VariableKind,
        )

        class ValidTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self, config=None, name="valid_transport", **kwargs):
                metadata = FunctorMetadata(
                    name=name,
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="valid transport",
                    references=["Fick (1855)."],
                    input_variables=["gradient"],
                    output_variables=["custom_flux"],
                    spatial_scale=SpatialScale.GRID_CELL,
                    variable_kind=VariableKind.FLUX,
                )
                super().__init__(name=name, config=config or {}, metadata=metadata, **kwargs)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"custom_flux": torch.tensor(0.0)}
        """,
    )
    _write_manifest(
        tmp_path,
        class_path="functor_impl.ValidTransport",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    info = build_extension_info(tmp_path, source="directory")
    result = validate_extension(info)

    assert result.valid is True
    assert result.errors == []


def test_validator_reports_forward_signature_error(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        from relic.core.functor import (
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            TransportMechanismFunctor,
            VariableKind,
        )

        class BadForward(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self, config=None, name="bad_forward", **kwargs):
                metadata = FunctorMetadata(
                    name=name,
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="bad forward",
                    references=["Fick (1855)."],
                    input_variables=["gradient"],
                    output_variables=["custom_flux"],
                    spatial_scale=SpatialScale.GRID_CELL,
                    variable_kind=VariableKind.FLUX,
                )
                super().__init__(name=name, config=config or {}, metadata=metadata, **kwargs)

            def get_dependencies(self):
                return []

            def forward(self, state):
                return {"custom_flux": state}
        """,
    )
    _write_manifest(
        tmp_path,
        class_path="functor_impl.BadForward",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    info = build_extension_info(tmp_path, source="directory")
    result = validate_extension(info)

    assert result.valid is False
    assert any("forward()" in error for error in result.errors)


def test_validator_reports_bridge_inheritance_error(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        import torch
        from relic.core.functor import (
            FunctorMetadata,
            PhysicalFieldFunctor,
            ProcessCategory,
            SpatialScale,
            VariableKind,
        )

        class WrongBridge(PhysicalFieldFunctor):
            def __init__(self, config=None, name="wrong_bridge", **kwargs):
                metadata = FunctorMetadata(
                    name=name,
                    category=ProcessCategory.PHYSICAL_FIELD,
                    description="wrong bridge",
                    references=["Arakawa and Lamb (1977)."],
                    input_variables=["field"],
                    output_variables=["custom_flux"],
                    spatial_scale=SpatialScale.GRID_CELL,
                    variable_kind=VariableKind.STATE,
                )
                super().__init__(name=name, config=config or {}, metadata=metadata, **kwargs)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"custom_flux": torch.tensor(0.0)}
        """,
    )
    _write_manifest(
        tmp_path,
        class_path="functor_impl.WrongBridge",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    info = build_extension_info(tmp_path, source="directory")
    result = validate_extension(info)

    assert result.valid is False
    assert any("does not inherit" in error for error in result.errors)


def test_validator_warns_when_scale_or_kind_missing(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        import torch
        from relic.core.functor import (
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            TransportMechanismFunctor,
            VariableKind,
        )

        class MissingScale(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self, config=None, name="missing_scale", **kwargs):
                metadata = FunctorMetadata(
                    name=name,
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="missing scale",
                    references=["Fick (1855)."],
                    input_variables=["gradient"],
                    output_variables=["custom_flux"],
                    spatial_scale=SpatialScale.UNKNOWN,
                    variable_kind=VariableKind.UNKNOWN,
                )
                super().__init__(name=name, config=config or {}, metadata=metadata, **kwargs)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"custom_flux": torch.tensor(0.0)}
        """,
    )
    _write_manifest(
        tmp_path,
        class_path="functor_impl.MissingScale",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    info = build_extension_info(tmp_path, source="directory")
    result = validate_extension(info)

    assert result.valid is True
    assert any("spatial_scale" in warning for warning in result.warnings)
    assert any("variable_kind" in warning for warning in result.warnings)


def test_validator_reports_non_importable_class(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        class_path="missing_module.MissingFunctor",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    info = build_extension_info(tmp_path, source="directory")
    result = validate_extension(info)

    assert result.valid is False
    assert any("Import failed" in error for error in result.errors)


def test_validator_reports_constructor_type_error_without_retry(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        """
        import torch
        from relic.core.functor import (
            FunctorMetadata,
            ProcessCategory,
            SpatialScale,
            TransportMechanismFunctor,
            VariableKind,
        )

        class StrictConfigTransport(TransportMechanismFunctor):
            TRANSPORT_TYPE = "diffusion"

            def __init__(self, config=None, name="strict_config_transport", **kwargs):
                if config is not None:
                    raise TypeError("rate_constant must be a float, got str")
                metadata = FunctorMetadata(
                    name=name,
                    category=ProcessCategory.TRANSPORT_MECHANISM,
                    description="functor whose declared config is rejected",
                    references=["Fick (1855)."],
                    input_variables=["gradient"],
                    output_variables=["strict_config_flux"],
                    spatial_scale=SpatialScale.GRID_CELL,
                    variable_kind=VariableKind.FLUX,
                )
                super().__init__(name=name, config={}, metadata=metadata, **kwargs)

            def get_dependencies(self):
                return []

            def forward(self, state, time, dt, **kwargs):
                return {"strict_config_flux": torch.tensor(0.0)}
        """,
    )
    _write_manifest(
        tmp_path,
        class_path="functor_impl.StrictConfigTransport",
        bridge="TransportMechanismFunctor",
        category="transport_mechanism",
    )

    result = validate_extension(build_extension_info(tmp_path, source="directory"))

    assert result.valid is False
    assert any(
        "Instantiation failed: rate_constant must be a float" in error
        for error in result.errors
    )
