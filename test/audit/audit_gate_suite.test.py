"""
Audit gate suite for high-priority atmosphere and metadata fixes.
"""

import ast
import importlib.util
import json
from pathlib import Path

import yaml

from test.audit.collect_functor_inventory import (
    _extract_metadata,
    scan as collect_functor_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_functor_inventory_resolves_metadata_name_from_constructor_default() -> None:
    source = """
class ExampleFunctor:
    def __init__(self, name="example_writer"):
        functor_name = name
        metadata = self._create_metadata(functor_name)

    def _create_metadata(self, functor_name):
        return FunctorMetadata(
            name=functor_name,
            output_variables=["example_output"],
        )
"""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    metadata = _extract_metadata(cls, tree)

    assert metadata["name"] == "example_writer"


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8", errors="ignore")


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _collect_non_diagnostic_output_duplicates() -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    roots = (REPO_ROOT / "relic" / "scientific", REPO_ROOT / "relic" / "coupling")

    for root in roots:
        for path in root.rglob("*.py"):
            src = path.read_text(encoding="utf-8-sig", errors="ignore")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue

            for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                metadata = _extract_metadata(cls, tree)
                metadata_outputs = [
                    value
                    for value in metadata.get("output_variables", "").split("|")
                    if value
                ]
                if not metadata_outputs:
                    continue

                ref = f"{cls.name}@{path.as_posix()}"
                for output_name in metadata_outputs:
                    if output_name.startswith("diagnostics."):
                        continue
                    outputs.setdefault(output_name, []).append(ref)

    return {k: v for k, v in outputs.items() if len(v) > 1}


def _scm_increment_contract_matches_live_writers(
    variable: str,
    writer_refs: list[str],
) -> bool:
    """Return whether SCM is the complete SSOT for a shared state/flux ledger."""
    contract_paths = list(
        (REPO_ROOT / "relic" / "contracts" / "variables").rglob(
            f"{variable}.yaml"
        )
    )
    if len(contract_paths) != 1:
        return False
    payload = yaml.safe_load(contract_paths[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("incrementable") is not True:
        return False

    units = payload.get("dimension", {}).get("udunits")
    increment_writers = payload.get("writers", {}).get("increments", [])
    if not isinstance(increment_writers, list) or not increment_writers:
        return False

    declared_modules: set[str] = set()
    for writer in increment_writers:
        if not isinstance(writer, dict):
            return False
        module_name = writer.get("module")
        provides = writer.get("provides", [])
        if not isinstance(module_name, str) or variable not in provides:
            return False
        if writer.get("unit") != units:
            return False
        declared_modules.add(module_name)

    live_modules: set[str] = set()
    for ref in writer_refs:
        _, raw_path = ref.split("@", 1)
        path = Path(raw_path).resolve().relative_to(REPO_ROOT.resolve())
        live_modules.add(".".join(path.with_suffix("").parts))
    return declared_modules == live_modules


def test_stability_parameters_are_parameterized() -> None:
    text = _read_text("relic/scientific/atmosphere/most/stability_parameters.py")
    assert "H_min = 1e-6" not in text
    assert "min=-1000.0, max=1000.0" not in text
    assert "h_min_safeguard" in text
    assert "monin_obukhov_length_clamp_min" in text
    assert "monin_obukhov_length_clamp_max" in text


def test_eddy_diffusion_clamps_and_background_are_parameterized() -> None:
    text = _read_text("relic/scientific/atmosphere/transport/eddy_diffusion_blm.py")
    assert "torch.clamp(pbl_height, min=50.0, max=3000.0)" not in text
    assert "torch.clamp(aerodynamic_resistance, min=10.0, max=500.0)" not in text
    assert "background_ratio = mixing_ratio * 0.5" not in text
    assert "background_ratio = mixing_ratio * 0.1" not in text
    assert "background_default_mixing_ratio" not in text
    for species in ("no", "no2", "o3", "isoprene"):
        assert f"background_{species}_mixing_ratio" in text


def test_isotope_heartbeat_is_fail_closed_without_output_contract() -> None:
    text = _read_text("relic/scientific/isotopes/functor.py")
    assert 'output_variables=["diagnostics.isotope_heartbeat"]' not in text
    assert "diagnostics.isotope_heartbeat" not in text
    assert "runtime_unbound" in text


def test_atmospheric_advection_diffusion_metadata_and_theory_doc() -> None:
    text = _read_text(
        "relic/scientific/atmosphere/transport/AtmosphericAdvectionDiffusionFunctor.py"
    )
    assert 'version="1.0.0"' in text
    assert "spatial_scale=SpatialScale.GRID_CELL" in text
    assert "variable_kind=VariableKind.FLUX" in text
    assert (REPO_ROOT / "docs/source/theory/atmosphere_advection_diffusion.md").exists()


def test_units_use_parseable_dimensionless_forms() -> None:
    """Every chemistry parameter states units the dimensionality owner derives.

    The database loader reads only the ``parameters`` section, so the file holds no
    other section of parameter-like entries (the former top-level ``validation``
    pseudo-parameters were never loaded).
    """
    from relic.data.standards.dimensionality import dimensionality_from_unit

    chemistry = yaml.safe_load(
        _read_text("relic/parameters/database/relic/atmosphere/chemistry.yaml")
    )
    assert set(chemistry) == {
        "schema_version", "tier", "namespace", "description", "metadata", "parameters",
    }
    unparseable = {
        key: entry.get("units")
        for key, entry in chemistry["parameters"].items()
        if dimensionality_from_unit(entry.get("units")) is None
    }
    assert unparseable == {}


def test_diffusion_boundary_flux_not_placeholder() -> None:
    text = _read_text("relic/scientific/transport/diffusion.py")
    assert "Return boundary flux placeholder (zero flux by default)." not in text
    assert "get_boundary_flux" not in text
    assert "pack_outward_boundary_flux_cochain" in text
    assert 'positive_power_convention="out_of_system"' in text
    assert 'result.get("boundary_fluxes")' in text


def test_output_contract_scanner_is_pure_and_resolves_finite_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The existing scanner must be import-safe and resolve static output builders."""
    scanner_path = REPO_ROOT / "test" / "audit" / "scan_output_contract.py"
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "relic_audit_scan_output_contract",
        scanner_path,
    )
    assert spec is not None and spec.loader is not None
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)

    assert not (tmp_path / "analysis_results").exists()

    source = '''
class SyntheticFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            output_variables=[
                "a",
                "b",
                "c",
                "dynamic_x",
                "dynamic_y",
            ]
        )

    def forward(self, state):
        output = {"a": state}
        output.update({"b": state})
        output["c"] = state
        for species in ("x", "y"):
            output[f"dynamic_{species}"] = state
        return output
'''
    records = scanner.analyze_source(source, Path("synthetic.py"))
    assert len(records) == 1
    assert records[0]["class"] == "SyntheticFunctor"
    assert records[0]["metadata_outputs"] == [
        "a",
        "b",
        "c",
        "dynamic_x",
        "dynamic_y",
    ]
    assert records[0]["forward_literal_keys"] == [
        "a",
        "b",
        "c",
        "dynamic_x",
        "dynamic_y",
    ]
    assert records[0]["unresolved_dynamic_keys"] == []
    assert records[0]["missing_in_forward_literal"] == []
    assert records[0]["extra_not_in_metadata"] == []

    inherited_source = '''
class BaseFunctor(DemoFunctor):
    def forward(self, state):
        return {"inherited_output": state}

class ChildFunctor(BaseFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["inherited_output"])
'''
    inherited_records = scanner.analyze_source(
        inherited_source,
        Path("inherited.py"),
    )
    child_record = next(
        record for record in inherited_records if record["class"] == "ChildFunctor"
    )
    assert child_record["forward_owner"] == "BaseFunctor"
    assert child_record["resolution_status"] == "inherited_static"
    assert child_record["unresolved_dynamic_keys"] == []
    assert child_record["missing_in_forward_literal"] == []
    assert child_record["extra_not_in_metadata"] == []

    helper_source = '''
def identity_output(outputs):
    return dict(outputs)

class HelperFunctor(DemoFunctor):
    OUTPUT_KEYS = ("dynamic_x", "dynamic_y")
    OPTIONAL_OUTPUTS = {"optional_output": "1"}

    def __init__(self):
        metadata = FunctorMetadata(
            output_variables=["a", "b", "diagnostic", "dynamic_x", "dynamic_y"]
        )
        if self.enabled:
            self.metadata.output_variables.extend(list(self.OPTIONAL_OUTPUTS))

    def _finalize(self, outputs):
        outputs["b"] = outputs["a"]
        return identity_output(outputs)

    def _append_diagnostic(self, outputs):
        outputs["diagnostic"] = outputs["a"]

    def forward(self, state):
        outputs = {"a": state}
        for key in self.OUTPUT_KEYS:
            outputs[key] = state
        if self.enabled:
            outputs["optional_output"] = state
        self._append_diagnostic(outputs)
        return self._finalize(outputs)

class DispatchBaseFunctor(DemoFunctor):
    def forward(self, state):
        return self._compute_outputs(state)

class DispatchChildFunctor(DispatchBaseFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["child_output"])

    def _compute_outputs(self, state):
        return {"child_output": state}

class TransportFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["tracer_state"])

    def forward(self, state):
        fluxes = {"tracer_flux": state}
        return self.finalize_transport_updates(
            outputs=fluxes,
            dt=1.0,
            state=state,
            mapping={"tracer_flux": "tracer_state"},
            drop_diagnostics=True,
        )

class ConstructorDefaultOutputFunctor(DemoFunctor):
    def __init__(self, tracer_key: str = "tracer_concentration"):
        metadata = FunctorMetadata(output_variables=[tracer_key])
        self.tracer_key = tracer_key

    def forward(self, state):
        return {self.tracer_key: state}

class FilteredFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["public_a", "public_b"])

    def _raw_outputs(self, state):
        return {
            "public_a": state,
            "internal_a": state,
            "public_b": state,
            "internal_b": state,
        }

    def forward(self, state):
        raw_outputs = self._raw_outputs(state)
        internal_keys = {"internal_a", "internal_b"}
        public_outputs = {
            key: value
            for key, value in raw_outputs.items()
            if key not in internal_keys
        }
        return public_outputs

class TupleDictionaryFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            output_variables=["left_a", "left_b", "right_c"]
        )

    def _output_groups(self, state):
        left = {}
        for key in ("a", "b"):
            left[key] = state
        right = {"c": state}
        return left, right

    def forward(self, state):
        left, right = self._output_groups(state)
        output = {}
        for key, value in left.items():
            output[f"left_{key}"] = value
        for key, value in right.items():
            output[f"right_{key}"] = value
        return output

class TensorTupleFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["result"])

    def _tensor_pair(self, state):
        left = torch.clamp(state, min=0.0)
        right = torch.zeros_like(state)
        return left, right

    def forward(self, state):
        left, right = self._tensor_pair(state)
        return {"result": left + right}

class AttributeDictionaryFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(
            output_variables=["base", "diagnostics.alpha", "diagnostics.beta"]
        )
        self._diagnostics = {}

    def _update_diagnostics(self, state):
        self._diagnostics = {
            "diagnostics.alpha": state,
            "diagnostics.beta": state,
        }

    def forward(self, state):
        output = {"base": state}
        self._update_diagnostics(state)
        output.update(self._diagnostics)
        return output

class TensorAttributeHelperFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["result"])

    def _build_output(self, state):
        self.current_tensor = torch.clamp(state, min=0.0)
        return {"result": self.current_tensor}

    def forward(self, state):
        return self._build_output(state)

class ValidatedSequenceFunctor(DemoFunctor):
    CANONICAL_NAMES = {"alpha": "a", "beta": "b"}

    def __init__(self):
        metadata = FunctorMetadata(
            output_variables=["dynamic_alpha", "dynamic_beta"]
        )
        self.names = self._load_names()

    def _load_names(self):
        names = self._configured_names()
        if names != list(self.CANONICAL_NAMES):
            raise ValueError("names must use the canonical finite schema")
        return tuple(names)

    def forward(self, state):
        output = {}
        for name in self.names:
            output[f"dynamic_{name}"] = state
        return output

class DictionaryTransformer:
    def transform_outputs(self, data, context):
        transformed = data.copy()
        for key, value in data.items():
            transformed[key] = value
        return transformed

class ExternalMethodFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["a_flux", "b_state"])
        self.transformer = DictionaryTransformer()

    def forward(self, state):
        output = {"a_flux": state, "b_state": state}
        flux_outputs = {
            key: value
            for key, value in output.items()
            if "flux" in key
        }
        output.update(
            self.transformer.transform_outputs(
                flux_outputs,
                {"context": state},
            )
        )
        normalized = {}
        for key, value in list(output.items()):
            normalized[key] = value
        return normalized
'''
    helper_records = scanner.analyze_source(
        helper_source,
        Path("helpers.py"),
    )
    helper_record = next(
        record for record in helper_records if record["class"] == "HelperFunctor"
    )
    assert helper_record["forward_literal_keys"] == [
        "a",
        "b",
        "diagnostic",
        "dynamic_x",
        "dynamic_y",
        "optional_output",
    ]
    assert helper_record["metadata_outputs"] == [
        "a",
        "b",
        "diagnostic",
        "dynamic_x",
        "dynamic_y",
        "optional_output",
    ]
    assert helper_record["unresolved_dynamic_keys"] == []
    assert helper_record["extra_not_in_metadata"] == []
    dispatch_record = next(
        record
        for record in helper_records
        if record["class"] == "DispatchChildFunctor"
    )
    assert dispatch_record["forward_owner"] == "DispatchBaseFunctor"
    assert dispatch_record["forward_literal_keys"] == ["child_output"]
    assert dispatch_record["unresolved_dynamic_keys"] == []
    transport_record = next(
        record
        for record in helper_records
        if record["class"] == "TransportFunctor"
    )
    assert transport_record["forward_literal_keys"] == ["tracer_state"]
    assert transport_record["unresolved_dynamic_keys"] == []
    constructor_default_record = next(
        record
        for record in helper_records
        if record["class"] == "ConstructorDefaultOutputFunctor"
    )
    assert constructor_default_record["metadata_outputs"] == [
        "tracer_concentration"
    ]
    assert constructor_default_record["forward_literal_keys"] == [
        "tracer_concentration"
    ]
    assert constructor_default_record["unresolved_dynamic_keys"] == []
    assert constructor_default_record["missing_in_forward_literal"] == []
    assert constructor_default_record["extra_not_in_metadata"] == []
    filtered_record = next(
        record
        for record in helper_records
        if record["class"] == "FilteredFunctor"
    )
    assert filtered_record["forward_literal_keys"] == ["public_a", "public_b"]
    assert filtered_record["unresolved_dynamic_keys"] == []
    assert filtered_record["extra_not_in_metadata"] == []
    tuple_record = next(
        record
        for record in helper_records
        if record["class"] == "TupleDictionaryFunctor"
    )
    assert tuple_record["forward_literal_keys"] == [
        "left_a",
        "left_b",
        "right_c",
    ]
    assert tuple_record["unresolved_dynamic_keys"] == []
    assert tuple_record["missing_in_forward_literal"] == []
    assert tuple_record["extra_not_in_metadata"] == []
    tensor_tuple_record = next(
        record
        for record in helper_records
        if record["class"] == "TensorTupleFunctor"
    )
    assert tensor_tuple_record["forward_literal_keys"] == ["result"]
    assert tensor_tuple_record["unresolved_dynamic_keys"] == []
    assert tensor_tuple_record["missing_in_forward_literal"] == []
    assert tensor_tuple_record["extra_not_in_metadata"] == []
    attribute_record = next(
        record
        for record in helper_records
        if record["class"] == "AttributeDictionaryFunctor"
    )
    assert attribute_record["forward_literal_keys"] == [
        "base",
        "diagnostics.alpha",
        "diagnostics.beta",
    ]
    assert attribute_record["unresolved_dynamic_keys"] == []
    assert attribute_record["missing_in_forward_literal"] == []
    assert attribute_record["extra_not_in_metadata"] == []
    tensor_attribute_record = next(
        record
        for record in helper_records
        if record["class"] == "TensorAttributeHelperFunctor"
    )
    assert tensor_attribute_record["forward_literal_keys"] == ["result"]
    assert tensor_attribute_record["unresolved_dynamic_keys"] == []
    assert tensor_attribute_record["missing_in_forward_literal"] == []
    assert tensor_attribute_record["extra_not_in_metadata"] == []
    validated_record = next(
        record
        for record in helper_records
        if record["class"] == "ValidatedSequenceFunctor"
    )
    assert validated_record["forward_literal_keys"] == [
        "dynamic_alpha",
        "dynamic_beta",
    ]
    assert validated_record["unresolved_dynamic_keys"] == []
    assert validated_record["missing_in_forward_literal"] == []
    assert validated_record["extra_not_in_metadata"] == []
    external_record = next(
        record
        for record in helper_records
        if record["class"] == "ExternalMethodFunctor"
    )
    assert external_record["forward_literal_keys"] == ["a_flux", "b_state"]
    assert external_record["unresolved_dynamic_keys"] == []
    assert external_record["missing_in_forward_literal"] == []
    assert external_record["extra_not_in_metadata"] == []

    abstract_source = '''
from abc import abstractmethod

class AbstractPipelineFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["abstract_output"])

    @abstractmethod
    def forward(self, state):
        raise NotImplementedError

class ConcretePipelineFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["concrete_output"])

    def forward(self, state):
        return {"concrete_output": state}
'''
    abstract_records = scanner.analyze_source(
        abstract_source,
        Path("abstract_pipeline.py"),
    )
    assert [record["class"] for record in abstract_records] == [
        "ConcretePipelineFunctor"
    ]

    declared_output_source = '''
class DeclaredOutputBaseFunctor(DemoFunctor):
    COMMON_OUTPUTS = ("common_state",)

    def __init__(self, metadata):
        metadata.output_variables = list(dict.fromkeys([
            *metadata.output_variables,
            *self.COMMON_OUTPUTS,
        ]))

    def _raw_outputs(self, state):
        return {
            "specific_output": state,
            "common_state": state,
        }

    def forward(self, state):
        raw_outputs = self._raw_outputs(state)
        updates = {}
        for key, value in raw_outputs.items():
            updates[key] = UpdateItem(value=value)
        return {
            key: updates[key]
            for key in self.metadata.output_variables
        }

class DeclaredOutputChildFunctor(DeclaredOutputBaseFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["specific_output"])
        super().__init__(metadata)
'''
    declared_output_records = scanner.analyze_source(
        declared_output_source,
        Path("declared_output.py"),
    )
    declared_output_record = next(
        record
        for record in declared_output_records
        if record["class"] == "DeclaredOutputChildFunctor"
    )
    assert declared_output_record["metadata_outputs"] == [
        "specific_output",
        "common_state",
    ]
    assert declared_output_record["forward_literal_keys"] == [
        "common_state",
        "specific_output",
    ]
    assert declared_output_record["unresolved_dynamic_keys"] == []
    assert declared_output_record["missing_in_forward_literal"] == []
    assert declared_output_record["extra_not_in_metadata"] == []

    # An ExternalIntent value is a delivery request, never a scientific output
    # (the runtime refuses an intent that is also a declared output), so the
    # scanner must not report its key as undeclared.
    intent_source = '''
class IntentFunctor(DemoFunctor):
    def __init__(self):
        metadata = FunctorMetadata(output_variables=["a"])

    def forward(self, state):
        outputs = {"a": state, "request_b": ExternalIntent("checkpoint", "p", "r")}
        outputs["request_c"] = ExternalIntent("topology", "t", "r", {"transition": "t"})
        return outputs
'''
    intent_record = scanner.analyze_source(intent_source, Path("intent.py"))[0]
    assert intent_record["forward_literal_keys"] == ["a"]
    assert intent_record["unresolved_dynamic_keys"] == []
    assert intent_record["extra_not_in_metadata"] == []

    report = scanner.scan()
    inventory_rows = collect_functor_inventory()
    implemented_pairs = {
        (row["file_path"], row["class_name"])
        for row in inventory_rows
        if row.get("process_status") == "process"
        and row.get("implementation_status") == "implemented"
    }
    unavailable_pairs = {
        (row["file_path"], row["class_name"])
        for row in inventory_rows
        if row.get("process_status") == "process"
        and row.get("implementation_status") == "not_implemented_fail_closed"
    }
    scanner_pairs = {
        (
            str(Path(record["file"]).resolve().relative_to(REPO_ROOT)).replace(
                "/",
                "\\",
            ),
            record["class"],
        )
        for record in report["records"]
    }
    assert scanner_pairs == implemented_pairs
    assert scanner_pairs.isdisjoint(unavailable_pairs)
    assert report["record_count"] == len(implemented_pairs)
    assert report["issue_count"] == 0

    persisted_rows = json.loads(
        (
            REPO_ROOT / "analysis_results" / "scientific_functor_inventory.json"
        ).read_text(encoding="utf-8")
    )
    identity_fields = ("file_path", "class_name", "process_id", "process_status")
    live_identities = sorted(
        tuple(str(row.get(field, "")) for field in identity_fields)
        for row in inventory_rows
    )
    persisted_identities = sorted(
        tuple(str(row.get(field, "")) for field in identity_fields)
        for row in persisted_rows
    )
    assert persisted_identities == live_identities

    abstract_pair = (
        "relic\\scientific\\biosphere\\phenology\\base.py",
        "PhenologyBase",
    )
    assert abstract_pair not in implemented_pairs
    assert abstract_pair not in scanner_pairs


def test_non_diagnostics_output_overlaps_must_be_whitelisted() -> None:
    """All non-diagnostic overlap outputs must be explicitly governed."""
    whitelist = json.loads(
        _read_text("relic/core/simulator/output_overlap_whitelist.json")
    )
    duplicates = _collect_non_diagnostic_output_duplicates()

    missing_policy = sorted([name for name in duplicates if name not in whitelist])
    assert not missing_policy, (
        "Found non-diagnostic output overlaps without governance policy: "
        + ", ".join(missing_policy)
    )

    stale_policy = sorted([name for name in whitelist if name not in duplicates])
    assert not stale_policy, (
        "Overlap whitelist contains entries without multiple live writers: "
        + ", ".join(stale_policy)
    )

    invalid_policy = []
    allowed = {"mutually_exclusive", "aggregated"}
    for name, entry in whitelist.items():
        policy = entry.get("policy")
        reason = entry.get("reason")
        if policy not in allowed or not isinstance(reason, str) or not reason.strip():
            invalid_policy.append(name)
        if policy == "aggregated":
            is_explicit_tendency_port = name.startswith("delta_")
            is_scm_increment_ledger = _scm_increment_contract_matches_live_writers(
                name,
                duplicates[name],
            )
            if not (is_explicit_tendency_port or is_scm_increment_ledger):
                invalid_policy.append(name)
        reason_lower = str(reason).lower()
        if any(
            marker in reason_lower
            for marker in (
                "co-maintained",
                "legacy",
                "migration",
                "pending",
                "staged refactor",
                "transitional",
                "until ",
                "current architecture",
            )
        ):
            invalid_policy.append(name)
        assert "legacy_shared" not in json.dumps(entry).lower()
    assert not invalid_policy, (
        "Overlap whitelist contains invalid policy entries: "
        + ", ".join(sorted(invalid_policy))
    )


def test_alternative_closure_ports_are_mutually_exclusive() -> None:
    """Alternative physical closures must never be accumulated at runtime."""
    policies = json.loads(
        _read_text("relic/core/simulator/output_overlap_whitelist.json")
    )
    for variable in (
        "drainage_flux",
        "monoterpenes_emission",
        "sesquiterpenes_emission",
        "soil_moisture",
        "target_grid_fields",
    ):
        assert policies.get(variable, {}).get("policy") == "mutually_exclusive"
