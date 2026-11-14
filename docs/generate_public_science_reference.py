"""Generate the public scientific class reference from RELIC SSOT sources.

The generator reads live Python classes, IPHS certificates, State Contract
Model files, and ParameterManager metadata. It deliberately omits tests,
migration artifacts, and build evidence from the public output.
"""

from __future__ import annotations

import argparse
import fnmatch
from html import escape
import importlib
import inspect
import logging
from pathlib import Path
import re
import sys
import types
import typing
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_ROOT = REPO_ROOT / "docs" / "source" / "theory" / "iphs_certificates"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "source" / "public" / "science"
DOMAIN_ORDER = (
    "anthroposphere",
    "atmosphere",
    "biosphere",
    "hydrosphere",
    "pedosphere",
    "isotopes",
    "transport",
)
DOMAIN_TITLES = {
    "anthroposphere": "Anthroposphere Scientific Classes",
    "atmosphere": "Atmosphere Scientific Classes",
    "biosphere": "Biosphere Scientific Classes",
    "hydrosphere": "Hydrosphere Scientific Classes",
    "pedosphere": "Pedosphere Scientific Classes",
    "isotopes": "Stable-Isotope Scientific Classes",
    "transport": "General Transport Scientific Classes",
}
THEORY_TARGETS = {
    "anthroposphere": "../theory/anthroposphere.md",
    "atmosphere": "../theory/atmosphere.md",
    "biosphere": "../theory/biosphere.md",
    "hydrosphere": "../theory/hydrosphere.md",
    "pedosphere": "../theory/pedosphere.md",
    "isotopes": "../theory/isotopes_transport.md",
    "transport": "../theory/isotopes_transport.md",
}
STANDARD_ARGUMENT_MEANINGS = {
    "name": "Functor identity used by registration, provenance, and dependency resolution.",
    "config": (
        "Structural and runtime configuration mapping. Scientific numeric "
        "parameters remain owned by ParameterManager unless the signature "
        "explicitly exposes an override."
    ),
    "device": (
        "Optional construction device. The simulator assigns the effective "
        "runtime device when the Functor is registered."
    ),
    "metadata": "Optional explicit FunctorMetadata object for advanced composition.",
    "pft_name": "Optional plant functional type namespace used for PFT parameter overrides.",
    "validate_inputs": "Enables fail-fast validation of required runtime inputs.",
    "enable_gpu": (
        "Compatibility control accepted by this constructor; effective device "
        "placement is still governed by simulator device policy."
    ),
    "kwargs": "Additional canonical bridge or FunctorBase options.",
    "args": "Additional positional arguments accepted by the inherited constructor.",
}
PUBLIC_TEXT_REPLACEMENTS = {
    "艩im暖nek": "Simunek",
    "掳C": "degC",
    "胃_a": "theta_a",
}


FORMULA_MARKER = re.compile(
    r"(?:[A-Za-z0-9_{}\]\)])\s*(?:=|:=|<=|>=|<|>)"
    r"|(?:\bd[A-Za-z0-9_{}\[\]]+\s*/\s*d[A-Za-z0-9_{}]+)"
    r"|(?:\b[A-Za-z][A-Za-z0-9_]*_dot\b)"
    r"|(?:--[A-Za-z0-9_]+-->|->)"
    r"|(?:\\(?:begin|frac|dfrac|max|min|sum|partial|nabla|dot|Delta)\b)"
    r"|(?:\\[A-Za-z]+)"
    r"|(?:\$[^$\n]+\$)"
)
FORMULA_TOKEN = re.compile(
    r"--[A-Za-z0-9_]+-->"
    r"|-{2,}>"
    r"|>=|<=|:=|->|\\\\"
    r"|\\[A-Za-z]+"
    r"|d(?:\[[^\]]+\]|[A-Za-z][A-Za-z0-9_,]*)/d[A-Za-z][A-Za-z0-9_]*"
    r"|partial\s*[A-Za-z][A-Za-z0-9_]*\s*/\s*partial\s*[A-Za-z][A-Za-z0-9_]*"
    r"|(?:u|theta)\*"
    r"|[A-Za-z][A-Za-z0-9_]*(?:,[A-Za-z0-9_]+)+(?:\.T)?"
    r"|[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+"
    r"|[A-Za-z][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)?"
    r"|[^\s]"
)
FORMULA_RELATION = re.compile(r"(?<![<>=:])(?:>=|<=|:=|=|<|>)(?![=])")
FORMULA_LHS_ATOM = re.compile(
    r"(?:u|theta)\*"
    r"|d(?:\[[^\]]+\]|[A-Za-z][A-Za-z0-9_,]*)/d[A-Za-z][A-Za-z0-9_]*"
    r"|partial\s*[A-Za-z][A-Za-z0-9_]*\s*/\s*partial\s*[A-Za-z][A-Za-z0-9_]*"
    r"|[A-Za-z][A-Za-z0-9_]*(?:,[A-Za-z0-9_]+)*(?:\.[A-Za-z][A-Za-z0-9_]*)*"
    r"(?:\([^()\n]*\))?"
    r"|\d+(?:\.\d+)?"
)
FORMULA_PROSE_BOUNDARIES = frozenset(
    {
        "after",
        "and",
        "are",
        "as",
        "at",
        "before",
        "by",
        "declared",
        "equal",
        "equals",
        "evaluated",
        "following",
        "for",
        "from",
        "give",
        "gives",
        "has",
        "have",
        "is",
        "mapping",
        "on",
        "only",
        "or",
        "over",
        "plus",
        "then",
        "through",
        "to",
        "use",
        "uses",
        "using",
        "was",
        "were",
        "when",
        "where",
        "while",
        "whose",
        "with",
    }
)
GREEK_IDENTIFIERS = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "Delta": r"\Delta",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "eps": r"\varepsilon",
    "eta": r"\eta",
    "Gamma": r"\Gamma",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "phi": r"\phi",
    "psi": r"\psi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "theta": r"\theta",
    "zeta": r"\zeta",
    "chi": r"\chi",
}
MATH_FUNCTIONS = {
    "clamp": r"\operatorname{clamp}",
    "clip": r"\operatorname{clip}",
    "cos": r"\cos",
    "diag": r"\operatorname{diag}",
    "div": r"\operatorname{div}",
    "dot": r"\cdot",
    "exp": r"\exp",
    "grad": r"\nabla",
    "integral": r"\int",
    "ln": r"\ln",
    "max": r"\max",
    "mean": r"\operatorname{mean}",
    "min": r"\min",
    "nabla": r"\nabla",
    "sqrt": r"\sqrt",
    "sum": r"\sum",
    "tanh": r"\tanh",
}
IDENTIFIER_LATEX_OVERRIDES = {
    "Cstar": r"C^\star",
    "fh": r"f_h",
    "fm": r"f_m",
    "ISmax": r"\mathrm{IS}_{\max}",
    "Jz": r"J_z",
    "Kc": r"K_c",
    "Km": r"K_m",
    "Ko": r"K_o",
    "psih": r"\psi_h",
    "psim": r"\psi_m",
    "rah": r"r_{ah}",
    "Rn": r"R_n",
    "Vcmax": r"V_{c,\max}",
    "Vmax": r"V_{\max}",
}
FORMULA_LATEX_OVERRIDES = {
    "V = (MIC * Vmax(T,theta) * [S]) / (Km(T) + MIC)": (
        r"V = \frac{\mathrm{MIC}\,V_{\max}(T,\theta)\,[S]}"
        r"{K_m(T)+\mathrm{MIC}}"
    ),
}
MAPPING_LITERAL_ASSIGNMENT = re.compile(
    r"\b(?:open_system_route|species|reactions)\s*=\s*"
    r"(?:'[^']*'|\"[^\"]*\"|\([^)]*\)|[A-Za-z][A-Za-z0-9_]*)"
)
MAPPING_FORMULA_FRAGMENTS = (
    "torch.matrix_exp(Q * dt_years)",
    "[f_leaf*(NPP-F_BVOC,C), f_stem*(NPP-F_BVOC,C), "
    "f_root*(NPP-F_BVOC,C)]",
    "[SLA * growth_rate_leaf, k_leaf * leaf_area_index]",
    "DDF max(0, T_air - T_melt) / seconds_per_day",
    "K_sat * f_drainage * (theta / theta_sat)^b",
    "(infiltration_flux - evapotranspiration_flux) / water_density",
    "[O3] / [O3]_threshold",
    "min(R_den, NO3 / dt)",
    "min(R_nit, NH4 / dt)",
    "min(x_i / -f_i)",
    "X_h.T star_K X_h",
    "rho_w*D_surf",
    "theta * Z * rho_w",
    "B*a/Z^b",
    "[NO3, N2O, N2]^T",
    "[NH4, NO3, N2O]^T",
    "[1, 0, 0]^T",
    "[1,1,1,1,1]",
    "[1,-1,-1]",
    "[0,1,1]",
    "[1,-1]",
)
GENERIC_REACTION_FORM = (
    "x_dot = N_h r_h with ParameterManager-backed process rates r_h "
    "and declared stoichiometric/output map N_h"
)


# Explicit constructor arguments used to instantiate each documented class for
# metadata introspection, keyed by the certificate process_id. Documentation never
# relies on an implicit constructor default: a certificate that declares a
# constructor_selection is constructed with exactly that selection (its section
# documents that runtime point), and for the other certificates a class whose
# constructor requires a
# configuration is listed with the configuration its page documents, and a class
# whose runtime interface depends on a configuration mode lists one
# (label, keyword arguments) entry per documented mode; the page then states which
# variables each mode declares. Classes absent from the table are constructed
# without arguments. A construction failure stops the generator and names this
# table (test/unit/publication/generate_public_science_reference.test.py).
DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS: dict[str, tuple[tuple[str, dict[str, Any]], ...]] = {
    "relic.scientific.biosphere.growth.phenology_growth.PhenologyGrowth": (
        ("phenology_model_owner='gsi_phenology'", {"phenology_model_owner": "gsi_phenology"}),
    ),
    "relic.scientific.biosphere.phenology.phenology_functor.PhenologyFunctor": (
        ("canonical_model_owner='gsi_phenology'", {"config": {"canonical_model_owner": "gsi_phenology"}}),
    ),
    # Stochastic owners require an explicit substream seed (certificate
    # constructor_selection: config={'seed': integer}).
    "relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor": (
        ("seed=7", {"config": {"seed": 7}}),
    ),
    "relic.scientific.atmosphere.stochastic_temperature.OrnsteinUhlenbeckTemperatureFunctor": (
        ("seed=7", {"config": {"seed": 7}}),
    ),
    "relic.scientific.anthroposphere.management.fire_management.FireManagement": (
        ("seed=7", {"config": {"seed": 7}}),
    ),
    # physical_boundary has no default. The open mechanism is documented here;
    # the closed constant-volume box has its own certificate, whose
    # constructor_selection builds that section.
    "relic.scientific.atmosphere.chemistry.ChemistryFunctor": (
        ("physical_boundary='open_mechanism'", {"physical_boundary": "open_mechanism"}),
    ),
    # surface_conditions has no default; the only implemented mode substitutes canopy air
    # for leaf-surface CO2 and VPD (docs/source/theory/photosynthesis.md, Medlyn et al. 2011).
    "relic.scientific.biosphere.photosynthesis.stomatal.MedlynStomatalConductance": (
        ("surface_conditions='leaf_surface_equals_canopy_air'",
         {"config": {"surface_conditions": "leaf_surface_equals_canopy_air"}}),
    ),
    # temporal_support has no default; both declared supports are documented
    # (docs/source/theory/biosphere_leaf_energy_balance.md).
    "relic.scientific.biosphere.photosynthesis.leaf_energy_balance.LeafEnergyBalance": (
        ("temporal_support='daily'", {"config": {"temporal_support": "daily"}}),
        ("temporal_support='instantaneous'", {"config": {"temporal_support": "instantaneous"}}),
    ),
    # spatial_dims and stability_scheme have no default; the stability scheme
    # selects the eddy-diffusivity closure and therefore the declared inputs.
    "relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor.AtmosphericAdvectionDiffusionFunctor": (
        ("spatial_dims=3, stability_scheme='monin-obukhov'", {"spatial_dims": 3, "stability_scheme": "monin-obukhov"}),
        ("spatial_dims=3, stability_scheme='richardson'", {"spatial_dims": 3, "stability_scheme": "richardson"}),
        ("spatial_dims=3, stability_scheme='constant'", {"spatial_dims": 3, "stability_scheme": "constant"}),
    ),
    "relic.scientific.hydrosphere.runoff.surface_ponding.SurfacePondingFunctor": (
        ("event_provider='ponding_event'", {"config": {"event_provider": "ponding_event"}}),
    ),
    # canopy_resistance_source has no default; the state mode declares the
    # surface_resistance and canopy_conductance inputs, the FAO-56 reference-surface
    # mode reads the reference resistance parameter instead.
    "relic.scientific.hydrosphere.evapotranspiration.penman_monteith.PenmanMonteithET": (
        ("canopy_resistance_source='state'", {"config": {"canopy_resistance_source": "state"}}),
        (
            "canopy_resistance_source='fao56_reference_surface'",
            {"config": {"canopy_resistance_source": "fao56_reference_surface"}},
        ),
    ),
}


def _load_certificates() -> list[dict[str, Any]]:
    certificates: list[dict[str, Any]] = []
    for path in sorted(CERTIFICATE_ROOT.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"Certificate is not a mapping: {path}")
        data["_certificate_path"] = path
        certificates.append(data)
    # One MyST target per rendered section. The first certificate of a
    # process keeps the class anchor; a further certificate for the same
    # process (for example a strict runtime-point selection) is anchored by
    # its certificate file stem so that Sphinx never sees a duplicate target.
    anchored: set[str] = set()
    for data in certificates:
        process_id = str(data["process_id"])
        if process_id in anchored:
            data["_anchor"] = _anchor_for(str(data["_certificate_path"].stem))
        else:
            data["_anchor"] = _anchor_for(process_id)
            anchored.add(process_id)
    return certificates


def _domain_for(process_id: str) -> str:
    parts = process_id.split(".")
    if len(parts) < 4 or parts[:2] != ["relic", "scientific"]:
        raise ValueError(f"Unexpected scientific process path: {process_id}")
    return parts[2]


def _anchor_for(process_id: str) -> str:
    return "class-" + re.sub(r"[^a-z0-9]+", "-", process_id.lower()).strip("-")


def _inline(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    for malformed, replacement in PUBLIC_TEXT_REPLACEMENTS.items():
        text = text.replace(malformed, replacement)
    text = _normalize_embedded_math(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def _annotation_text(annotation: Any) -> str:
    """Render a constructor annotation independently of the interpreter version.

    repr() and inspect.formatannotation() changed between CPython 3.11 and 3.14
    (typing.Optional/Union render PEP 604 style and the ``typing.`` prefix is
    stripped recursively in 3.14), so the page text is built from the
    annotation structure: unions always use ``a | b``, ``None`` is spelled
    ``None``, typing generics keep their ``typing.`` prefix and classes are
    module-qualified unless they are builtins.
    """
    if annotation is inspect.Parameter.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    if annotation is typing.Any:
        return "typing.Any"
    if isinstance(annotation, list):
        return "[" + ", ".join(_annotation_text(item) for item in annotation) + "]"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(_annotation_text(item) for item in args)
    if origin is typing.Literal:
        return "typing.Literal[" + ", ".join(repr(item) for item in args) + "]"
    if args:
        head = repr(annotation).split("[", 1)[0]
        return head + "[" + ", ".join(_annotation_text(item) for item in args) + "]"
    if isinstance(annotation, type):
        module = annotation.__module__
        if module == "builtins":
            return annotation.__qualname__
        return f"{module}.{annotation.__qualname__}"
    return repr(annotation)


def _signature_text(cls: type) -> str:
    """Interpreter-independent constructor signature (same layout as inspect)."""
    signature = inspect.signature(cls)
    rendered: list[str] = []
    saw_positional_only = False
    saw_keyword_marker = False
    for parameter in signature.parameters.values():
        kind = parameter.kind
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            saw_positional_only = True
        elif saw_positional_only:
            rendered.append("/")
            saw_positional_only = False
        if kind is inspect.Parameter.VAR_POSITIONAL:
            saw_keyword_marker = True
        elif kind is inspect.Parameter.KEYWORD_ONLY and not saw_keyword_marker:
            rendered.append("*")
            saw_keyword_marker = True
        text = parameter.name
        if kind is inspect.Parameter.VAR_POSITIONAL:
            text = "*" + text
        elif kind is inspect.Parameter.VAR_KEYWORD:
            text = "**" + text
        has_annotation = parameter.annotation is not inspect.Parameter.empty
        if has_annotation:
            text += ": " + re.sub(r"\btyping\.", "", _annotation_text(parameter.annotation))
        if parameter.default is not inspect.Parameter.empty:
            text += (" = " if has_annotation else "=") + repr(parameter.default)
        rendered.append(text)
    if saw_positional_only:
        rendered.append("/")
    result = "(" + ", ".join(rendered) + ")"
    if signature.return_annotation is not inspect.Signature.empty:
        result += " -> " + re.sub(r"\btyping\.", "", _annotation_text(signature.return_annotation))
    return result


def _code(value: Any) -> str:
    text = _inline(value).replace("`", r"\`")
    if "|" in text:
        html_text = escape(re.sub(r"\\+\|", "|", text)).replace("|", "&#124;")
        return f"<code>{html_text}</code>"
    return f"`{text}`"


def _looks_like_formula(value: str) -> bool:
    return bool(FORMULA_MARKER.search(value))


def _tex_subscript(value: str) -> str:
    if "," in value:
        return ",".join(_tex_subscript(part) for part in value.split(","))
    if value in GREEK_IDENTIFIERS:
        return GREEK_IDENTIFIERS[value]
    if len(value) == 1 and value.isalpha():
        return value
    if value == "star":
        return r"\star"
    return rf"\mathrm{{{value}}}"


def _tex_identifier(value: str) -> str:
    if value in {"u*", "theta*"}:
        base = value[:-1]
        return rf"{_tex_identifier(base)}_*"
    if "," in value:
        head, *tail = value.split(",")
        if "_" not in head or tail == ["theta"] or any("_" in part for part in tail):
            return ", ".join(_tex_identifier(part) for part in (head, *tail))
    if value.endswith(".T"):
        base = value[:-2]
        return rf"{_tex_identifier(base)}^{{\mathsf{{T}}}}"
    if "." in value:
        escaped = value.replace("_", r"\_")
        return rf"\mathtt{{{escaped}}}"
    if value in IDENTIFIER_LATEX_OVERRIDES:
        return IDENTIFIER_LATEX_OVERRIDES[value]
    if value in GREEK_IDENTIFIERS:
        return GREEK_IDENTIFIERS[value]
    if value in MATH_FUNCTIONS:
        return MATH_FUNCTIONS[value]
    dot_match = re.fullmatch(
        r"(?P<base>[A-Za-z][A-Za-z0-9_]*?)_dot"
        r"(?P<suffix>(?:_[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)*)",
        value,
    )
    if dot_match:
        rendered = rf"\dot{{{_tex_identifier(dot_match.group('base'))}}}"
        suffix = dot_match.group("suffix").lstrip("_")
        if suffix:
            rendered += rf"_{{{_tex_subscript(suffix)}}}"
        return rendered
    if value.startswith("partial") and len(value) > len("partial"):
        return rf"\partial {_tex_identifier(value[len('partial'):])}"
    if value.startswith("grad") and len(value) > len("grad"):
        return rf"\nabla {_tex_identifier(value[len('grad'):])}"
    if value.startswith("d") and len(value) == 2 and value[1].isalpha():
        return rf"\mathrm{{d}}{_tex_identifier(value[1])}"
    if "_" in value:
        base, *subscripts = value.split("_")
        if base == "star":
            rendered_base = r"\star"
        else:
            rendered_base = _tex_identifier(base)
        rendered_subscript = ",".join(
            _tex_subscript(subscript)
            for subscript in subscripts
            if subscript
        )
        if not rendered_subscript:
            return rendered_base
        return rf"{rendered_base}_{{{rendered_subscript}}}"
    chemical = re.fullmatch(r"([A-Z]{1,4})(\d+)", value)
    if chemical:
        return rf"\mathrm{{{chemical.group(1)}}}_{{{chemical.group(2)}}}"
    if len(value) == 1:
        return value
    if value.isupper() or any(character.isdigit() for character in value):
        return rf"\mathrm{{{value}}}"
    return rf"\mathrm{{{value}}}"


def _tex_derivative_target(value: str) -> str:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return rf"\left[{_tex_identifier(inner)}\right]"
    return _tex_identifier(value)


def _normalize_latex_notation(value: str) -> str:
    def replace_derivative(match: re.Match[str]) -> str:
        numerator = _tex_derivative_target(match.group("numerator"))
        denominator = _tex_identifier(match.group("denominator"))
        return (
            rf"\frac{{\mathrm{{d}}{numerator}}}"
            rf"{{\mathrm{{d}}{denominator}}}"
        )

    normalized = re.sub(
        r"(?<![A-Za-z0-9_\\])"
        r"d(?P<numerator>[A-Za-z][A-Za-z0-9_,]*)"
        r"/d(?P<denominator>[A-Za-z][A-Za-z0-9_]*)",
        replace_derivative,
        value,
    )
    return re.sub(
        r"(?<!\\)\|(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\|",
        lambda match: (
            rf"\lvert {_tex_identifier(match.group('symbol'))}\rvert"
        ),
        normalized,
    )


def _normalize_embedded_math(value: str) -> str:
    return re.sub(
        r"\$(?P<formula>[^$\n]+)\$",
        lambda match: (
            "$" + _normalize_latex_notation(match.group("formula")) + "$"
        ),
        value,
    )


def _ascii_formula_to_latex(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    override = FORMULA_LATEX_OVERRIDES.get(compact)
    if override is not None:
        return override

    tokens = FORMULA_TOKEN.findall(compact)
    rendered: list[str] = []
    absolute_open = False
    skip_through = -1
    for index, token in enumerate(tokens):
        if index <= skip_through:
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith("\\"):
            rendered.append(token)
            continue
        derivative = re.fullmatch(
            r"d(?P<numerator>\[[^\]]+\]|[A-Za-z][A-Za-z0-9_,]*)"
            r"/d(?P<denominator>[A-Za-z][A-Za-z0-9_]*)",
            token,
        )
        if derivative:
            numerator = _tex_derivative_target(derivative.group("numerator"))
            denominator = _tex_identifier(derivative.group("denominator"))
            rendered.append(
                rf"\frac{{\mathrm{{d}}{numerator}}}"
                rf"{{\mathrm{{d}}{denominator}}}"
            )
            continue
        partial = re.fullmatch(
            r"partial\s*(?P<numerator>[A-Za-z][A-Za-z0-9_]*)"
            r"\s*/\s*partial\s*(?P<denominator>[A-Za-z][A-Za-z0-9_]*)",
            token,
        )
        if partial:
            numerator = _tex_identifier(partial.group("numerator"))
            denominator = _tex_identifier(partial.group("denominator"))
            rendered.append(
                rf"\frac{{\partial {numerator}}}{{\partial {denominator}}}"
            )
            continue
        if re.fullmatch(r"--[A-Za-z0-9_]+-->", token):
            label = token[2:-3]
            rendered.append(rf"\xrightarrow{{{_tex_identifier(label)}}}")
            continue
        if re.fullmatch(r"-{2,}>", token):
            rendered.append(r"\longrightarrow")
            continue
        if token in {">=", "<=", ":=", "->"}:
            rendered.append(
                {
                    ">=": r"\geq",
                    "<=": r"\leq",
                    ":=": r"\coloneqq",
                    "->": r"\rightarrow",
                }[token]
            )
            continue
        if token in {"u*", "theta*"}:
            rendered.append(_tex_identifier(token))
            continue
        if token == "^" and following == "T":
            rendered.append(r"^{\mathsf{T}}")
            skip_through = index + 1
            continue
        if token == "sqrt" and following == "(":
            depth = 0
            closing_index: int | None = None
            for nested_index in range(index + 1, len(tokens)):
                nested = tokens[nested_index]
                if nested == "(":
                    depth += 1
                elif nested == ")":
                    depth -= 1
                    if depth == 0:
                        closing_index = nested_index
                        break
            if closing_index is None:
                raise ValueError(f"Unbalanced sqrt call in formula: {value!r}")
            inner = " ".join(tokens[index + 2:closing_index])
            rendered.append(rf"\sqrt{{{_ascii_formula_to_latex(inner)}}}")
            skip_through = closing_index
            continue
        if token == "*":
            rendered.append(r"\cdot")
            continue
        if token == "|":
            rendered.append(r"\lvert" if not absolute_open else r"\rvert")
            absolute_open = not absolute_open
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            rendered.append(token)
            continue
        if re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*(?:,[A-Za-z0-9_]+)*"
            r"(?:\.[A-Za-z][A-Za-z0-9_]*)*",
            token,
        ):
            rendered.append(_tex_identifier(token))
            continue
        rendered.append(token)
    if absolute_open:
        raise ValueError(f"Unbalanced absolute-value delimiter in formula: {value!r}")
    return " ".join(rendered)


def _display_math(value: str) -> list[str]:
    return ["$$", value.strip(), "$$"]


def _public_prose(value: str) -> str:
    text = value
    for malformed, replacement in PUBLIC_TEXT_REPLACEMENTS.items():
        text = text.replace(malformed, replacement)
    return re.sub(r"\s+", " ", text).strip()


def _top_level_formula_candidates(value: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    start = 0
    round_depth = 0
    square_depth = 0
    brace_depth = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\":
            index += 2
            continue
        if character == "(":
            round_depth += 1
        elif character == ")":
            round_depth = max(0, round_depth - 1)
        elif character == "[":
            square_depth += 1
        elif character == "]":
            square_depth = max(0, square_depth - 1)
        elif character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth = max(0, brace_depth - 1)

        at_top_level = round_depth == square_depth == brace_depth == 0
        if at_top_level and (
            character == ";"
            or (
                character == ","
                and index + 1 < len(value)
                and value[index + 1].isspace()
            )
        ):
            candidates.append((start, index))
            start = index + 1
            index += 1
            continue
        if (
            at_top_level
            and character == ":"
            and (index + 1 >= len(value) or value[index + 1] != "=")
        ):
            candidates.append((start, index))
            start = index + 1
            index += 1
            continue
        if at_top_level and character.isalpha():
            word_end = index + 1
            while word_end < len(value) and value[word_end].isalpha():
                word_end += 1
            word = value[index:word_end]
            before = value[index - 1] if index else ""
            after = value[word_end] if word_end < len(value) else ""
            is_standalone = (
                (not before or not (before.isalnum() or before == "_"))
                and (not after or not (after.isalnum() or after == "_"))
            )
            if is_standalone and word in FORMULA_PROSE_BOUNDARIES:
                candidates.append((start, index))
                start = word_end
                index = word_end
                continue
            index = word_end
            continue
        index += 1
    candidates.append((start, len(value)))
    return candidates


def _lhs_atom_is_prependable(value: str) -> bool:
    base = value.split("(", maxsplit=1)[0]
    if re.fullmatch(r"\d+(?:\.\d+)?", base):
        return True
    if base.startswith(("d", "partial")) and "/" in base:
        return True
    if base in GREEK_IDENTIFIERS or base in MATH_FUNCTIONS:
        return True
    if "_" in base or "." in base:
        return True
    return len(base) == 1 or base.isupper()


def _formula_lhs_start(value: str, operator_start: int) -> int | None:
    prefix = value[:operator_start]
    trimmed_end = len(prefix.rstrip())
    atoms = [
        match
        for match in FORMULA_LHS_ATOM.finditer(prefix[:trimmed_end])
        if match.end() <= trimmed_end
    ]
    if not atoms or atoms[-1].end() != trimmed_end:
        return None

    selected = atoms[-1]
    start = selected.start()
    for previous in reversed(atoms[:-1]):
        between = prefix[previous.end():start]
        if not between or not between.isspace():
            break
        if not _lhs_atom_is_prependable(previous.group()):
            break
        start = previous.start()
    return start


def _find_formula_spans(value: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for candidate_start, candidate_end in _top_level_formula_candidates(value):
        candidate = value[candidate_start:candidate_end]
        leading = len(candidate) - len(candidate.lstrip())
        trailing = len(candidate.rstrip())
        if leading >= trailing:
            continue
        stripped = candidate[leading:trailing]
        absolute_start = candidate_start + leading
        absolute_end = candidate_start + trailing

        if re.search(r"--[A-Za-z0-9_]+-->|-{2,}>|(?<!-)->", stripped):
            spans.append((absolute_start, absolute_end))
            continue

        relation = FORMULA_RELATION.search(stripped)
        if relation is None:
            continue
        lhs_start = _formula_lhs_start(stripped, relation.start())
        if lhs_start is None:
            continue
        spans.append((absolute_start + lhs_start, absolute_end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _formula_span_to_latex(value: str) -> str:
    text = value.strip()
    if re.search(r"\\[A-Za-z]+|_\{", text):
        return _normalize_latex_notation(text)
    return _ascii_formula_to_latex(text)


def _render_prose_symbols(value: str) -> str:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"(?P<symbol>"
        r"[A-Za-z]_[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*"
        r"|[A-Za-z]\([A-Za-z](?:,[A-Za-z])+\)"
        r")"
        r"(?![A-Za-z0-9_])"
    )
    output: list[str] = []
    cursor = 0
    for match in pattern.finditer(value):
        output.append(value[cursor:match.start()])
        symbol = match.group("symbol")
        latex = (
            _ascii_formula_to_latex(symbol)
            if "(" in symbol
            else _tex_identifier(symbol)
        )
        output.append(f"${latex}$")
        cursor = match.end()
    output.append(value[cursor:])
    return "".join(output)


def _render_mixed_formula(
    value: str,
    *,
    force_inline: bool = False,
) -> list[str]:
    spans = _find_formula_spans(value)
    if not spans:
        if not _looks_like_formula(value):
            return [_public_prose(value)]
        latex = _formula_span_to_latex(value)
        return [f"${latex}$"] if force_inline else _display_math(latex)

    gaps: list[str] = []
    cursor = 0
    for start, end in spans:
        gaps.append(value[cursor:start])
        cursor = end
    gaps.append(value[cursor:])
    has_prose = any(re.search(r"[A-Za-z]", gap) for gap in gaps)

    if not has_prose and not force_inline:
        output: list[str] = []
        cursor = 0
        for start, end in spans:
            output.append(value[cursor:start])
            output.append(_formula_span_to_latex(value[start:end]))
            cursor = end
        output.append(value[cursor:])
        return _display_math(re.sub(r"\s+", " ", "".join(output)).strip())

    output = []
    cursor = 0
    for start, end in spans:
        output.append(_render_prose_symbols(value[cursor:start]))
        output.append(f"${_formula_span_to_latex(value[start:end])}$")
        cursor = end
    output.append(_render_prose_symbols(value[cursor:]))
    return [_public_prose("".join(output))]


def _render_formula_line(value: str, *, bullet: bool = False) -> list[str]:
    text = value.strip()
    if not text:
        return []
    if "$" in text:
        rendered = _public_prose(_normalize_embedded_math(text))
        return [("- " if bullet else "") + rendered]
    if not _looks_like_formula(text):
        rendered = _public_prose(text)
        return [("- " if bullet else "") + rendered]

    if bullet:
        return ["- " + _render_mixed_formula(text, force_inline=True)[0]]

    colon = text.find(":")
    if (
        colon > 0
        and not text.startswith(":=", colon)
    ):
        label = text[:colon].strip()
        expression = text[colon + 1 :].strip()
        if expression and not _looks_like_formula(label):
            lines = [_inline(label) + ":", ""]
            lines.extend(_render_mixed_formula(expression))
            return lines

    if re.search(r"\\[A-Za-z]+|_\{", text):
        return _display_math(text)

    return _render_mixed_formula(text)


def _render_formula_field(value: Any) -> list[str]:
    text = inspect.cleandoc(str(value)).strip()
    if not text:
        return ["Not declared."]
    if text == GENERIC_REACTION_FORM:
        return [
            "$$",
            r"\dot{x}=N_h r_h",
            "$$",
            "",
            (
                "with ParameterManager-backed process rates $r_h$ "
                "and declared stoichiometric/output map $N_h$"
            ),
        ]
    if not _looks_like_formula(text):
        return [_inline(text)]

    output: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    for paragraph in paragraphs:
        source_lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not source_lines:
            continue
        if any(r"\begin{" in line for line in source_lines):
            first_line = source_lines[0]
            colon = first_line.find(":")
            if colon > 0:
                label = first_line[:colon].strip()
                expression = first_line[colon + 1 :].strip()
                if expression and not _looks_like_formula(label):
                    output.extend([_inline(label) + ":", ""])
                    source_lines[0] = expression
            leading: list[str] = []
            equation: list[str] = []
            for line in source_lines:
                if not equation and not _looks_like_formula(line):
                    leading.append(line)
                else:
                    equation.append(line)
            for line in leading:
                output.extend(_render_formula_line(line))
                output.append("")
            output.extend(_display_math("\n".join(equation)))
            output.append("")
            continue
        for line in source_lines:
            is_bullet = line.startswith("- ")
            content = line[2:] if is_bullet else line
            output.extend(_render_formula_line(content, bullet=is_bullet))
            output.append("")
    while output and not output[-1]:
        output.pop()
    return output


def _display_value(value: Any, limit: int = 180) -> str:
    if value is None:
        return "None"
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().tolist()
    text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _enum_value(value: Any) -> str:
    if value is None:
        return "Not declared"
    return str(getattr(value, "value", value))


def _first_paragraph(docstring: str | None) -> str:
    if not docstring:
        return "Scientific Functor participating in the canonical RELIC lifecycle."
    return re.split(r"\n\s*\n", inspect.cleandoc(docstring), maxsplit=1)[0].replace("\n", " ")


def _numpy_doc_parameters(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    lines = inspect.cleandoc(docstring).splitlines()
    result: dict[str, str] = {}
    start = None
    for index in range(len(lines) - 1):
        if lines[index].strip() == "Parameters" and set(lines[index + 1].strip()) == {"-"}:
            start = index + 2
            break
    if start is None:
        return result

    current_names: list[str] = []
    description: list[str] = []

    def flush() -> None:
        if not current_names:
            return
        meaning = " ".join(part.strip() for part in description if part.strip()).strip()
        if not meaning:
            meaning = "Constructor argument documented by the class interface."
        for name in current_names:
            result[name] = meaning

    for line in lines[start:]:
        if line and not line.startswith((" ", "\t")):
            if re.match(r"^[A-Za-z][A-Za-z0-9 _-]*$", line.strip()):
                flush()
                break
            match = re.match(r"^([*A-Za-z_][A-Za-z0-9_, *]*)\s*:\s*(.+)$", line)
            if match:
                flush()
                current_names = [
                    name.strip().lstrip("*")
                    for name in match.group(1).split(",")
                    if name.strip()
                ]
                description = []
                continue
        if current_names:
            description.append(line)
    else:
        flush()
    return result


def _constructor_rows(cls: type) -> list[tuple[str, str, str, str]]:
    signature = inspect.signature(cls)
    doc_parameters = _numpy_doc_parameters(inspect.getdoc(cls))
    rows: list[tuple[str, str, str, str]] = []
    for parameter in signature.parameters.values():
        annotation = _inline(_annotation_text(parameter.annotation))
        default = (
            "required"
            if parameter.default is inspect.Parameter.empty
            else _display_value(parameter.default)
        )
        name = parameter.name
        meaning = doc_parameters.get(name)
        if not meaning:
            meaning = STANDARD_ARGUMENT_MEANINGS.get(name)
        if not meaning and name.startswith("enable_"):
            meaning = f"Enables or disables the {_inline(name[7:].replace('_', ' '))} pathway."
        if not meaning and name.endswith("_owner"):
            meaning = (
                "Selects the canonical upstream owner used for dependency and "
                "single-writer composition."
            )
        if not meaning:
            meaning = (
                f"Constructor control for {_inline(name.replace('_', ' '))}; "
                "see the class runtime contract and ParameterManager table below."
            )
        rows.append((name, annotation, default, meaning))
    return rows


def _documented_instances(
    cls: type,
    process_id: str,
    selection: Mapping[str, Any] | None = None,
) -> list[tuple[str, Any]]:
    """Instantiate a documented class once per declared documentation configuration.

    A certificate's constructor_selection is the one configuration of its section;
    otherwise DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS applies. Returns (label, instance)
    pairs; the label is empty for a class constructed without arguments. Abstract
    classes are not instantiated.
    """
    if inspect.isabstract(cls):
        return []
    if selection:
        label = ", ".join(f"{key}={value!r}" for key, value in selection.items())
        configurations: tuple[tuple[str, dict[str, Any]], ...] = ((label, dict(selection)),)
    else:
        configurations = DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS.get(process_id, (("", {}),))
    instances: list[tuple[str, Any]] = []
    for label, kwargs in configurations:
        try:
            instance = cls(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with the documentation context
            raise RuntimeError(
                f"{process_id}: construction for documentation failed with {kwargs!r} "
                f"({type(exc).__name__}: {exc}). Declare its documented constructor "
                "configuration in DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS."
            ) from exc
        instances.append((label, instance))
    return instances


def _load_contracts() -> Any:
    from relic.contracts.contract_loader import ContractRegistry

    registry = ContractRegistry(strict=False)
    registry.load_contracts_dir(REPO_ROOT / "relic" / "contracts" / "variables")
    return registry


def _load_parameter_manager() -> Any:
    from relic.parameters import get_global_manager

    return get_global_manager()


def _declared_parameter_key(name: str, info: Mapping[str, Any] | None) -> str:
    """The full ParameterManager key one FunctorMetadata.parameters entry declares.

    The entry name is the key itself, or, when the entry declares a namespace, the
    namespace joined with the name. A name that already starts with its namespace is
    the full key, and so is a namespace that already ends with the name. No key is
    inferred from the loaded parameter database.
    """
    namespace = str(info.get("namespace", "")).strip() if info else ""
    if not namespace or name.startswith(namespace + "."):
        return name
    if namespace.rsplit(".", 1)[-1] == name:
        return namespace
    return f"{namespace}.{name}"


def _resolve_parameter_key(raw_key: str, manager_keys: set[str]) -> list[str]:
    """Exact full-key resolution against the loaded parameter database.

    A declared key resolves only to itself. A declared pattern, a ``*`` wildcard or a
    ``<pft>`` / ``<pft_name>`` placeholder, names a family of keys and resolves to every
    key that matches the whole pattern from its first character. A key that is not
    declared yields no match; it is never replaced by a key that merely ends with the
    same name.
    """
    if raw_key in manager_keys:
        return [raw_key]
    pattern = raw_key.replace("<pft>", "*").replace("<pft_name>", "*")
    if "*" in pattern:
        return sorted(fnmatch.filter(manager_keys, pattern))
    return []


NOT_DECLARED_IN_DATABASE = "not declared in the parameter database"


def _parameter_rows(
    certificate: Mapping[str, Any],
    metadata_parameters: Mapping[str, Any],
    manager: Any,
    manager_keys: set[str],
) -> tuple[
    list[tuple[str, str, str, str, str, str]],
    list[tuple[str, str, str, str, str, str]],
]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    constant_rows: list[tuple[str, str, str, str, str, str]] = []
    seen: set[str] = set()
    parameter_map = certificate.get("parameter_map") or {}
    management_map = certificate.get("parameter_management_map") or {}

    # Each metadata entry declares one key (its name, or its namespace and name).
    metadata_keys: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for name, entry in metadata_parameters.items():
        info = entry if isinstance(entry, Mapping) else {"description": str(entry)}
        metadata_keys[str(name)] = (_declared_parameter_key(str(name), info), info)

    # Certificate keys first, then the metadata keys the certificate does not list; a
    # certificate key equal to a metadata entry name takes that entry's declared key.
    declared: list[tuple[str, Mapping[str, Any] | None]] = []
    for raw in certificate.get("parameter_keys") or []:
        declared.append(metadata_keys.get(str(raw), (str(raw), None)))
    for key, info in metadata_keys.values():
        if all(item[0] != key for item in declared):
            declared.append((key, info))

    for raw_key, declared_info in declared:
        matches = _resolve_parameter_key(raw_key, manager_keys)
        if matches:
            for exact_key in matches:
                if exact_key in seen:
                    continue
                seen.add(exact_key)
                spec = manager.get_parameter(exact_key)
                if spec is None:
                    raise ValueError(
                        f"{certificate['process_id']}: {exact_key} is listed by the "
                        "parameter database but has no ParameterSpec"
                    )
                units = _inline(getattr(spec, "units", "dimensionless"))
                default = _display_value(getattr(spec, "value", None))
                bounds = _display_value(getattr(spec, "bounds", None))
                meaning = _inline(getattr(spec, "description", "")).strip()
                if not meaning:
                    meaning = "ParameterManager-backed scientific parameter."
                if declared_info and declared_info.get("required") is False:
                    meaning += " This runtime metadata binding is optional."
                requires_grad = bool(getattr(spec, "requires_grad", False))
                gradient = "learnable" if requires_grad else "fixed"
                references = list(getattr(spec, "references", None) or [])
                provenance = getattr(spec, "provenance", None)
                evidence_parts: list[str] = []
                if isinstance(provenance, Mapping):
                    source = provenance.get("source")
                    if source:
                        source_text = _inline(source)
                        if any(
                            marker in source_text.lower()
                            for marker in (
                                "audit",
                                "migration",
                                "autofill",
                                "phase8",
                            )
                        ):
                            source_text = "RELIC parameter database"
                        evidence_parts.append(f"Source: {source_text}")
                elif provenance:
                    evidence_parts.append(f"Source: {_inline(provenance)}")
                public_references: list[str] = []
                missing_reference = False
                for item in references:
                    reference = _inline(item)
                    if reference == "[NEEDS_CITATION]":
                        missing_reference = True
                        continue
                    if re.match(
                        r"^RELIC\b.*\baudit\b.*\d{4}",
                        reference,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    public_references.append(reference)
                evidence_parts.extend(public_references[:2])
                if missing_reference and not public_references:
                    evidence_parts.append(
                        "Scientific reference not recorded in "
                        "ParameterManager metadata"
                    )
                evidence = "; ".join(evidence_parts) or "ParameterManager YAML"
                target = (
                    constant_rows
                    if exact_key.startswith("relic.physical.constants.")
                    else rows
                )
                target.append(
                    (
                        exact_key,
                        units,
                        f"{default}; bounds={bounds}",
                        gradient,
                        meaning,
                        evidence,
                    )
                )
            continue
        if raw_key in seen:
            continue
        seen.add(raw_key)
        # The declaration names no key of the parameter database: it is reported as
        # declared, with the meaning and units its declaration states, and no value.
        mapped = parameter_map.get(raw_key) or management_map.get(raw_key)
        if isinstance(mapped, Mapping):
            mapped = mapped.get("description") or mapped.get("source")
        units = "not declared"
        if mapped:
            meaning = _inline(mapped)
        elif declared_info:
            meaning = _inline(
                declared_info.get("description")
                or declared_info.get("meaning")
                or "FunctorMetadata declaration without a description."
            )
            declared_units = declared_info.get("units")
            if not declared_units:
                match = re.search(r"\[([^\[\]]+)\]\s*$", meaning)
                declared_units = match.group(1) if match else None
            units = _inline(declared_units or units)
        else:
            meaning = "Certificate declaration without a description."
        rows.append(
            (
                raw_key,
                units,
                NOT_DECLARED_IN_DATABASE,
                "not declared",
                meaning,
                "Declared by FunctorMetadata or the certificate; no key of this exact "
                "name exists in the parameter database",
            )
        )
    return rows, constant_rows


DIAGNOSTICS_CHANNEL = "diagnostics"
RUNTIME_WRITE_MODE_CHANNEL = "decided at run time by the returned write mode"


def _output_channels(variables: Iterable[str]) -> dict[str, str]:
    """Channel of each declared output under the runtime channel classifier.

    ``partition_outputs`` is the only runtime channel classifier. Applied to a
    name alone it sees the reserved diagnostic syntax only: the DERIVED write
    mode travels with the value that ``forward()`` returns, and the generator
    does not run ``forward()``.
    """
    from relic.core.state.types import partition_outputs

    names = [str(variable) for variable in variables]
    _, diagnostics, _ = partition_outputs(dict.fromkeys(names), declared_outputs=names)
    return {
        name: DIAGNOSTICS_CHANNEL if name in diagnostics else RUNTIME_WRITE_MODE_CHANNEL
        for name in names
    }


def _variable_rows(
    variables: Iterable[str],
    certificate: Mapping[str, Any],
    contract_registry: Any,
    direction: str,
) -> list[tuple[str, ...]]:
    state_map = certificate.get("state_map") or {}
    unit_map = certificate.get("unit_map") or {}
    io_map = certificate.get("input_output_map") or {}
    variables = list(variables)
    channels = _output_channels(variables) if direction == "output" else {}
    rows: list[tuple[str, ...]] = []
    for variable in variables:
        contract = contract_registry.get_contract(variable)
        units = (
            getattr(contract, "udunits", None)
            or unit_map.get(variable)
            or "not declared"
        )
        kind = (
            getattr(contract, "kind", None)
            or ("input port" if direction == "input" else "output port")
        )
        meaning = state_map.get(variable) or io_map.get(variable)
        if not meaning:
            meaning = (
                "Required or optional value consumed by forward()."
                if direction == "input"
                else "Output returned by forward(); the certificate gives no description."
            )
        if direction == "output":
            rows.append(
                (
                    str(variable),
                    _inline(kind),
                    _inline(units),
                    channels[str(variable)],
                    _inline(meaning),
                )
            )
        else:
            rows.append((str(variable), _inline(kind), _inline(units), _inline(meaning)))
    return rows


def _declares_fail_closed_nonimplementation(
    certificate: Mapping[str, Any],
) -> bool:
    """Reuse the certificate owner's exact executable-availability predicate."""

    from relic.computation.operators.iphs_certificate import (
        load_iphs_equivalence_certificate,
    )

    certificate_model = load_iphs_equivalence_certificate(certificate)
    return certificate_model.declares_fail_closed_nonimplementation()


def _metadata_snapshot(
    cls: type,
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    unavailable = _declares_fail_closed_nonimplementation(certificate)
    process_id = str(certificate["process_id"])
    instances = (
        []
        if unavailable
        else _documented_instances(
            cls, process_id, certificate.get("constructor_selection")
        )
    )
    instance = instances[0][1] if instances else None
    metadata = getattr(instance, "metadata", None) if instance is not None else None
    if metadata is None:
        boundary_label = "runtime_unbound" if unavailable else "Abstract contract"
        return {
            "name": (
                "not_applicable_runtime_unbound" if unavailable else cls.__name__
            ),
            "spatial_scale": boundary_label,
            "variable_kind": boundary_label,
            "inputs": list(certificate.get("input_variables") or []),
            "outputs": list(certificate.get("output_variables") or []),
            "dependencies": [],
            "parameters": {},
            "references": [],
            "port_owner": None,
            "ports": [],
            "implementation_status": (
                "not_implemented_fail_closed" if unavailable else "abstract_contract"
            ),
        }
    port_owner = next(
        (
            base
            for base in cls.__mro__
            if base.__module__.startswith("relic.scientific")
            and "ports" in base.__dict__
        ),
        None,
    )
    ports: list[dict[str, str]] = []
    if port_owner is not None and instance is not None:
        for port in instance.ports():
            effort_space = getattr(port, "effort_space", None)
            flow_space = getattr(port, "flow_space", None)
            ports.append(
                {
                    "name": str(getattr(port, "name", "")),
                    "effort": str(getattr(port, "effort_var", "")),
                    "effort_units": str(getattr(effort_space, "units", "")),
                    "flow": str(getattr(port, "flow_var", "")),
                    "flow_units": str(getattr(flow_space, "units", "")),
                    "power_units": str(getattr(port, "power_units", "")),
                    "convention": str(
                        getattr(port, "positive_power_convention", "")
                    ),
                    "kind": str(getattr(port, "port_kind", "")),
                }
            )
    modes = [
        {
            "label": label,
            "inputs": list(getattr(mode_instance.metadata, "input_variables", None) or []),
            "outputs": list(getattr(mode_instance.metadata, "output_variables", None) or []),
        }
        for label, mode_instance in instances
    ]
    # A page renders one metadata block; documented modes may differ only in their
    # declared runtime interface, which the page lists per mode.
    for label, mode_instance in instances[1:]:
        for field in ("name", "spatial_scale", "variable_kind", "dependencies", "parameters", "references"):
            if getattr(mode_instance.metadata, field, None) != getattr(metadata, field, None):
                raise ValueError(
                    f"{process_id}: documented mode {label!r} changes metadata field "
                    f"{field!r}; only the runtime interface may differ between modes"
                )
    return {
        "name": getattr(metadata, "name", cls.__name__),
        "spatial_scale": _enum_value(getattr(metadata, "spatial_scale", None)),
        "variable_kind": _enum_value(getattr(metadata, "variable_kind", None)),
        # Union over the documented modes, in declaration order.
        "inputs": list(dict.fromkeys(name for mode in modes for name in mode["inputs"])),
        "outputs": list(dict.fromkeys(name for mode in modes for name in mode["outputs"])),
        "modes": modes,
        "dependencies": list(getattr(metadata, "dependencies", None) or []),
        "parameters": dict(getattr(metadata, "parameters", None) or {}),
        "references": list(getattr(metadata, "references", None) or []),
        "port_owner": port_owner.__name__ if port_owner is not None else None,
        "ports": ports,
        "implementation_status": "implemented",
    }


def _live_bridge(cls: type) -> str:
    canonical = {
        "PhysicalFieldFunctor",
        "BiologicalProcessFunctor",
        "TransportMechanismFunctor",
        "ScaleCouplingFunctor",
    }
    for base in cls.__mro__:
        if base.__name__ in canonical:
            return base.__name__
    return "FunctorBase"


def _source_location(cls: type) -> tuple[str, int]:
    source_path = Path(inspect.getsourcefile(cls) or "")
    try:
        relative = source_path.resolve().relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError):
        relative = source_path
    try:
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        line = 1
    return str(relative).replace("/", "\\"), line


def _render_simple_table(
    headers: tuple[str, ...],
    rows: Iterable[tuple[Any, ...]],
    code_columns: set[int] | None = None,
) -> list[str]:
    code_columns = code_columns or set()
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for index, value in enumerate(row):
            cells.append(_code(value) if index in code_columns else _inline(value))
        output.append("| " + " | ".join(cells) + " |")
    return output


def _render_mapping_plain_segment(value: str) -> str:
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    core = value.strip()
    if not core:
        return value
    protected: dict[str, str] = {}

    def protect_literal(match: re.Match[str]) -> str:
        token = f"RELICCONFIGTOKEN{len(protected)}"
        protected[token] = match.group(0)
        return token

    prepared = MAPPING_LITERAL_ASSIGNMENT.sub(protect_literal, core)
    rendered = _render_mixed_formula(prepared, force_inline=True)[0]
    for token, literal in protected.items():
        rendered = rendered.replace(token, f"`{literal}`")

    markup_parts = re.split(r"(\$[^$\n]+\$|`[^`\n]+`)", rendered)
    output: list[str] = []
    for part in markup_parts:
        if not part:
            continue
        if part.startswith(("$", "`")):
            output.append(_normalize_embedded_math(part))
            continue
        for fragment in MAPPING_FORMULA_FRAGMENTS:
            if fragment in part:
                latex = _ascii_formula_to_latex(fragment)
                part = part.replace(fragment, f"${latex}$")
        formula_parts = re.split(r"(\$[^$\n]+\$)", part)
        output.extend(
            formula_part
            if formula_part.startswith("$")
            else _render_prose_symbols(formula_part)
            for formula_part in formula_parts
            if formula_part
        )
    return leading + "".join(output) + trailing


def _render_mapping_value(value: Any) -> str:
    text = _public_prose(_normalize_embedded_math(str(value)))
    parts = re.split(r"(\$[^$\n]+\$|`[^`\n]+`)", text)
    rendered = [
        _normalize_embedded_math(part)
        if part.startswith(("$", "`"))
        else _render_mapping_plain_segment(part)
        for part in parts
        if part
    ]
    return _public_prose("".join(rendered))


def _render_mapping(mapping: Mapping[str, Any]) -> list[str]:
    if not mapping:
        return ["No additional mapping is declared."]
    return _render_simple_table(
        ("Entry", "Meaning"),
        ((key, _render_mapping_value(value)) for key, value in mapping.items()),
        {0},
    )


def _render_class(
    certificate: Mapping[str, Any],
    cls: type,
    metadata: Mapping[str, Any],
    manager: Any,
    manager_keys: set[str],
    contract_registry: Any,
) -> list[str]:
    process_id = str(certificate["process_id"])
    class_name = str(certificate["module_name"])
    source_path, source_line = _source_location(cls)
    bridge = _live_bridge(cls)
    abstraction = (certificate.get("interface_map") or {}).get(
        "abstraction_contract",
        "Scientific contract",
    )
    role = certificate.get("canonical_role", "declared process")
    level = certificate.get("iphs_level", "not declared")
    is_abstract = inspect.isabstract(cls)
    implementation_status = str(
        metadata.get("implementation_status", "unknown")
    )
    is_unavailable = implementation_status == "not_implemented_fail_closed"
    lines = [
        f"({certificate['_anchor']})=",
        f"## {class_name}",
        "",
        _first_paragraph(inspect.getdoc(cls)),
        "",
        "### Identity and architecture",
        "",
    ]
    identity_rows = [
        ("Import path", process_id),
        ("Source", f"{source_path}:{source_line}"),
        ("Runtime name", metadata["name"]),
        ("Scientific abstraction", abstraction),
        ("Canonical bridge", bridge),
        ("Structural role", role),
        ("Spatial scale", metadata["spatial_scale"]),
        ("Variable kind", metadata["variable_kind"]),
        ("IPHS level", f"level_{level}"),
        ("Implementation status", implementation_status),
        (
            "Class status",
            "abstract contract"
            if is_abstract
            else (
                "fail-closed unavailable boundary"
                if is_unavailable
                else "concrete scientific Functor"
            ),
        ),
    ]
    lines.extend(_render_simple_table(("Field", "Current contract"), identity_rows, {1}))
    if is_unavailable:
        replacement_entrypoint = str(
            (certificate.get("runtime_replacement_map") or {}).get(
                "replacement_entrypoint", "none_runtime_unbound"
            )
        )
        runtime_reason = replacement_entrypoint.removeprefix("none_")
        lines.extend(
            [
                "",
                ":::{warning}",
                f"This class is `[N] {runtime_reason}`. Construction and `forward(...)` "
                "reject before state access or side effects. Empty runtime input, output, "
                "and port declarations below are the documented fail-closed boundary, not missing "
                "documentation.",
                ":::",
            ]
        )
    literature_formula = str(certificate.get("literature_formula", "")).strip()
    if is_abstract and literature_formula.startswith("self."):
        literature_formula = (
            "Abstract phenology contract: concrete StaticPhenology, GDDPhenology, "
            "GSIPhenology, and CropPhenology classes provide the executable process equations."
        )
    lines.extend(
        [
            "",
            "### Constructor and argument meanings",
            "",
            "```python",
            f"{class_name}{_signature_text(cls)}",
            "```",
            "",
        ]
    )
    lines.extend(
        _render_simple_table(
            ("Argument", "Type", "Default", "Purpose"),
            _constructor_rows(cls),
            {0, 1, 2},
        )
    )
    modes = list(metadata.get("modes") or [])
    if len(modes) == 1 and modes[0]["label"]:
        lines.extend(
            [
                "",
                "The runtime metadata below is introspected from the documented "
                f"configuration {_code(modes[0]['label'])}.",
            ]
        )
    elif len(modes) > 1:
        common_inputs = set.intersection(*(set(mode["inputs"]) for mode in modes))
        common_outputs = set.intersection(*(set(mode["outputs"]) for mode in modes))

        def _mode_only(names: list[str], common: set[str]) -> str:
            specific = [name for name in names if name not in common]
            return ", ".join(_code(name) for name in specific) if specific else "none"

        lines.extend(
            [
                "",
                "### Configuration modes",
                "",
                "The declared runtime interface depends on the constructor configuration; "
                "there is no default mode. Each documented mode was instantiated. The "
                "table lists the variables that only that mode declares; every other "
                "variable of the runtime contracts below is declared in all modes.",
                "",
            ]
        )
        lines.extend(
            _render_simple_table(
                ("Mode", "Inputs declared only in this mode", "Outputs declared only in this mode"),
                (
                    (
                        mode["label"],
                        _mode_only(mode["inputs"], common_inputs),
                        _mode_only(mode["outputs"], common_outputs),
                    )
                    for mode in modes
                ),
                {0},
            )
        )

    certificate_inputs = list(certificate.get("input_variables") or [])
    certificate_outputs = list(certificate.get("output_variables") or [])
    lines.extend(["", "### Runtime input contract", ""])
    lines.extend(
        _render_simple_table(
            ("Variable", "Kind", "Units", "Use"),
            _variable_rows(certificate_inputs, certificate, contract_registry, "input"),
            {0, 2},
        )
    )
    lines.extend(["", "### Runtime output contract", ""])
    lines.extend(
        _render_simple_table(
            ("Variable", "Kind", "Units", "Channel", "Meaning"),
            _variable_rows(certificate_outputs, certificate, contract_registry, "output"),
            {0, 2},
        )
    )

    metadata_inputs = set(metadata.get("inputs") or [])
    metadata_outputs = set(metadata.get("outputs") or [])
    certificate_input_set = set(certificate_inputs)
    certificate_output_set = set(certificate_outputs)
    input_delta = sorted(certificate_input_set.symmetric_difference(metadata_inputs))
    output_delta = sorted(certificate_output_set.symmetric_difference(metadata_outputs))
    if input_delta or output_delta:
        lines.extend(
            [
                "",
                ":::{note}",
                "The certificate interface and the FunctorMetadata of the documented "
                "constructor configuration (the union over all documented modes when "
                "there are several) are not identical for this class. The tables above "
                "report the complete certificate-backed interface, including any "
                "configuration-conditional ports. The difference is listed here so "
                "callers do not infer an unconditional interface.",
            ]
        )
        if input_delta:
            lines.append("Input declaration difference: " + ", ".join(_code(x) for x in input_delta) + ".")
        if output_delta:
            lines.append("Output declaration difference: " + ", ".join(_code(x) for x in output_delta) + ".")
        lines.append(":::")

    lines.extend(
        [
            "",
            "### Governing equations and canonical operator",
            "",
            "**Literature or process form**",
            "",
        ]
    )
    lines.extend(_render_formula_field(literature_formula))
    lines.extend(
        [
            "",
            "**Canonical RELIC structural form**",
            "",
        ]
    )
    lines.extend(
        _render_formula_field(
            str(certificate.get("iphs_formula", "")).strip()
        )
    )
    lines.append("")
    lines.extend(_render_mapping(certificate.get("operator_map") or {}))
    lines.extend(["", "### State advancement, conservation, and open boundaries", ""])
    lines.extend(_render_mapping(certificate.get("state_advancement_map") or {}))
    certificate_ports = [
        item.strip()
        for value in certificate.get("composition_ports") or []
        for item in str(value).split(",")
        if item.strip()
    ]
    lines.extend(["", "### Composition channels and executable PortSpec API", ""])
    lines.append(
        "Certificate composition channels: "
        + (
            ", ".join(_code(port) for port in certificate_ports)
            if certificate_ports
            else "none declared."
        )
    )
    executable_ports = list(metadata.get("ports") or [])
    port_owner = metadata.get("port_owner")
    if executable_ports:
        lines.extend(
            [
                "",
                f"Executable `ports()` owner: {_code(port_owner)}.",
                "",
            ]
        )
        lines.extend(
            _render_simple_table(
                (
                    "Port",
                    "Effort and units",
                    "Flow and units",
                    "Pairing units",
                    "Convention and kind",
                ),
                (
                    (
                        port["name"],
                        f"{port['effort']} [{port['effort_units']}]",
                        f"{port['flow']} [{port['flow_units']}]",
                        port["power_units"],
                        f"{port['convention']}; {port['kind']}",
                    )
                    for port in executable_ports
                ),
                {0, 1, 2, 3},
            )
        )
    elif port_owner:
        lines.extend(
            [
                "",
                f"Executable `ports()` owner: {_code(port_owner)}. The default "
                "instance returns no fixed PortSpec entries; its ports are "
                "configuration-conditional or observational.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "This class does not expose a class-level executable `ports()` schema. "
                "Certificate channels therefore describe composition evidence, not an "
                "instantiated PortSpec API.",
            ]
        )

    dependencies = list(metadata.get("dependencies") or [])
    if executable_ports:
        lines.append("")
    if is_unavailable:
        lines.append(
            "No executable dependency or composition path is declared while this "
            "capability remains fail-closed."
        )
    else:
        lines.append(
            "Metadata dependencies: "
            + (
                ", ".join(_code(item) for item in dependencies)
                if dependencies
                else "no direct dependency list is declared; composition uses ports, owners, or graph wiring."
            )
        )

    lines.extend(["", "### Scientific parameter and numeric binding contract", ""])
    parameter_rows, constant_rows = (
        ([], [])
        if is_abstract
        else _parameter_rows(
            certificate,
            metadata.get("parameters") or {},
            manager,
            manager_keys,
        )
    )
    if parameter_rows:
        lines.extend(
            _render_simple_table(
                (
                    "Parameter key",
                    "Units",
                    "Default and bounds",
                    "Gradient",
                    "Purpose",
                    "Provenance or reference",
                ),
                parameter_rows,
                {0, 1, 2, 3},
            )
        )
    elif is_unavailable:
        lines.append(
            "This fail-closed class does not resolve or execute scientific parameters."
        )
    else:
        lines.append(
            "This class has no numeric scientific parameter keys in its certificate; "
            "it acts as a structural observer or adapter over declared state and ports."
        )
    if constant_rows:
        lines.extend(
            [
                "",
                "### Physical constants (SSOT)",
                "",
                "These values are resolved from the physical-constants database. They "
                "are not domain parameters and must not be duplicated in scientific "
                "YAML or Python code.",
                "",
            ]
        )
        lines.extend(
            _render_simple_table(
                (
                    "Constant key",
                    "Units",
                    "Value and fixed bounds",
                    "Gradient",
                    "Meaning",
                    "Provenance or reference",
                ),
                constant_rows,
                {0, 1, 2, 3},
            )
        )

    lines.extend(["", "### Runtime flow", ""])
    interface_map = certificate.get("interface_map") or {}
    selected_interface = {
        key: interface_map[key]
        for key in (
            "constructor",
            "forward",
            "forward_signature",
            "state_write_mode",
            "abstraction_contract",
            "bridge_class",
        )
        if key in interface_map
    }
    selected_interface["bridge_class"] = bridge
    lines.extend(_render_mapping(selected_interface))
    lines.extend(
        [
            "",
            f"Detailed derivation: [{_domain_for(process_id).capitalize()} theory]"
            f"({THEORY_TARGETS[_domain_for(process_id)]}).",
            "",
            "Certificate theory owner: " + _code(certificate.get("theory_document", "")) + ".",
            "",
            "### References",
            "",
        ]
    )
    certificate_references = [
        str(reference)
        for reference in certificate.get("references") or []
        if not str(reference).startswith("self.")
    ]
    runtime_references = [str(reference) for reference in metadata.get("references") or []]
    reference_origins: dict[str, set[str]] = {}
    for reference in runtime_references:
        reference_origins.setdefault(reference, set()).add("FunctorMetadata")
    for reference in certificate_references:
        reference_origins.setdefault(reference, set()).add("certificate")
    if is_abstract and not reference_origins:
        reference_origins[
            "Concrete phenology classes provide model-specific references in their own entries."
        ] = {"abstract contract"}
    for reference, origins in reference_origins.items():
        lines.append(
            f"- {_inline(reference)} "
            f"({_inline(', '.join(sorted(origins)))})"
        )
    lines.append("")
    return lines


def _load_class(process_id: str) -> type:
    module_name, class_name = process_id.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"Certificate target is not a class: {process_id}")
    return cls


def _render_domain(
    domain: str,
    certificates: list[dict[str, Any]],
    class_map: Mapping[str, type],
    metadata_map: Mapping[str, Mapping[str, Any]],
    manager: Any,
    manager_keys: set[str],
    contract_registry: Any,
) -> str:
    abstract_count = sum(
        inspect.isabstract(class_map[str(item["process_id"])])
        for item in certificates
    )
    unavailable_count = sum(
        metadata_map[str(item["_anchor"])].get("implementation_status")
        == "not_implemented_fail_closed"
        for item in certificates
    )
    executable_count = len(certificates) - abstract_count - unavailable_count
    unavailable_label = "boundary" if unavailable_count == 1 else "boundaries"
    lines = [
        f"# {DOMAIN_TITLES[domain]}",
        "",
        f"This reference covers all {len(certificates)} certificate-backed scientific "
        f"classes in `relic.scientific.{domain}`: {executable_count} executable concrete "
        f"Functors, {unavailable_count} fail-closed unavailable {unavailable_label}, and "
        f"{abstract_count} abstract "
        f"{'contract' if abstract_count == 1 else 'contracts'}. Each entry is "
        "generated from the live class, its IPHS certificate, State Contract Model, "
        "and ParameterManager metadata. Fail-closed unavailable classes are not "
        "instantiated during documentation generation.",
        "",
        "The tables distinguish constructor controls, runtime variables, and "
        "ParameterManager scientific parameters. Values shown as package defaults "
        "belong to the documented RELIC architecture version and must not be copied "
        "as hardcoded constants into scientific code.",
        "",
        "Certificate wildcard parameter groups (`*`, `<pft>`) are expanded to their "
        "exact current YAML entries. Every other declared parameter is matched to the "
        "parameter database by its exact full key; a declared name is never mapped to "
        "a key that merely ends with the same name.",
        "",
        "Exact ParameterManager rows report the live YAML value, bounds, gradient "
        "policy, description, and provenance. A row whose value reads "
        f"`{NOT_DECLARED_IN_DATABASE}` is intentionally not assigned an invented "
        "default: the class metadata or certificate declares the quantity, with the "
        "meaning and units it states, but the parameter database has no key of that "
        "exact name.",
        "",
        "IPHS levels are read from the sole certificate owner. Levels 1 to 3 "
        "describe compatibility, wrapping, or declared structure; they are not "
        "strict IPHS/GENERIC claims.",
        "",
        "The Channel column of each runtime output contract applies "
        "`relic.core.state.types.partition_outputs`, the only runtime output "
        "channel classifier, to the declared output name. A name that the "
        "classifier reserves for diagnostics, such as `diagnostics.*`, goes to "
        "the diagnostics channel and is never committed as state. For any other "
        "name the channel is decided at run time by the write mode of the "
        "returned value: `StateWriteMode.DERIVED` routes the output to the "
        "diagnostics channel, and any other write mode sends it to the "
        "StateManager commit (a value without a write mode takes the run's "
        "semantic-mode rules or the StateManager schema). The generator does "
        "not run `forward()`, so it does not predict that write mode. The Kind "
        "column is the State Contract Model kind of the variable and does not "
        "set its channel.",
        "",
    ]
    for certificate in certificates:
        process_id = str(certificate["process_id"])
        lines.extend(
            _render_class(
                certificate,
                class_map[process_id],
                metadata_map[str(certificate["_anchor"])],
                manager,
                manager_keys,
                contract_registry,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_index(
    by_domain: Mapping[str, list[dict[str, Any]]],
    class_map: Mapping[str, type],
    metadata_map: Mapping[str, Mapping[str, Any]],
) -> str:
    total = sum(len(items) for items in by_domain.values())
    abstract = sum(
        inspect.isabstract(class_map[str(item["process_id"])])
        for items in by_domain.values()
        for item in items
    )
    unavailable = sum(
        metadata_map[str(item["_anchor"])].get("implementation_status")
        == "not_implemented_fail_closed"
        for items in by_domain.values()
        for item in items
    )
    executable = total - abstract - unavailable
    abstract_label = "contract" if abstract == 1 else "contracts"
    unavailable_label = "boundary" if unavailable == 1 else "boundaries"
    lines = [
        "Core Scientific Class Reference",
        "===============================",
        "",
        f"The scientific reference covers {total} certificate-backed classes: "
        f"{executable} executable concrete Functors, {unavailable} fail-closed "
        f"unavailable {unavailable_label}, and {abstract} abstract scientific "
        f"{abstract_label}. The class-to-certificate relationship is one-to-one.",
        "",
        "Every executable concrete process follows the current ``abstraction -> bridge -> "
        "FunctorBase -> ParameterManager -> StateManager`` lifecycle. The pages below "
        "report unavailable classes separately; their constructors are not invoked. "
        "The pages document exact signatures, variable contracts, governing equations, "
        "parameters, units, structural roles, ports, and references.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :caption: Scientific domains",
        "",
    ]
    for domain in DOMAIN_ORDER:
        if by_domain.get(domain):
            lines.append(f"   {domain}")
    lines.extend(["", "Complete coverage matrix", "------------------------", ""])
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for domain in DOMAIN_ORDER:
        for certificate in by_domain.get(domain, []):
            process_id = str(certificate["process_id"])
            class_name = str(certificate["module_name"])
            metadata = metadata_map[str(certificate["_anchor"])]
            bridge = _live_bridge(class_map[process_id])
            anchor = str(certificate["_anchor"])
            class_link = f":ref:`{class_name} <{anchor}>`"
            rows.append(
                (
                    domain,
                    class_link,
                    bridge,
                    certificate.get("canonical_role", ""),
                    metadata["spatial_scale"],
                    f"level_{certificate.get('iphs_level')}",
                    metadata.get("implementation_status", "unknown"),
                )
            )
    lines.extend(
        [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 12 24 20 14 14 10 20",
            "",
            "   * - Domain",
            "     - Class",
            "     - Bridge",
            "     - Role",
            "     - Spatial scale",
            "     - IPHS level",
            "     - Implementation status",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"   * - {row[0]}",
                f"     - {row[1]}",
                f"     - ``{row[2]}``",
                f"     - ``{row[3]}``",
                f"     - ``{row[4]}``",
                f"     - ``{row[5]}``",
                f"     - ``{row[6]}``",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _expected_outputs() -> dict[str, str]:
    logging.disable(logging.CRITICAL)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    certificates = _load_certificates()
    stale_configurations = sorted(
        set(DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS)
        - {str(certificate["process_id"]) for certificate in certificates}
    )
    if stale_configurations:
        raise ValueError(
            "DOCUMENTED_CONSTRUCTOR_CONFIGURATIONS names classes without a certificate: "
            f"{stale_configurations}"
        )
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    for certificate in certificates:
        domain = _domain_for(str(certificate["process_id"]))
        if domain not in by_domain:
            raise ValueError(f"Undocumented scientific domain: {domain}")
        by_domain[domain].append(certificate)

    class_map: dict[str, type] = {}
    metadata_map: dict[str, dict[str, Any]] = {}
    for certificate in certificates:
        process_id = str(certificate["process_id"])
        cls = _load_class(process_id)
        class_map[process_id] = cls
        metadata_map[str(certificate["_anchor"])] = _metadata_snapshot(cls, certificate)

    manager = _load_parameter_manager()
    manager_keys = set(manager.list())
    contract_registry = _load_contracts()

    outputs = {
        "index.rst": _render_index(by_domain, class_map, metadata_map),
    }
    for domain in DOMAIN_ORDER:
        outputs[f"{domain}.md"] = _render_domain(
            domain,
            by_domain[domain],
            class_map,
            metadata_map,
            manager,
            manager_keys,
            contract_registry,
        )
    return outputs


def _write_outputs(output_root: Path, outputs: Mapping[str, str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    expected_names = set(outputs)
    for existing in output_root.glob("*"):
        if existing.is_file() and existing.name not in expected_names:
            raise RuntimeError(f"Unexpected generated science source: {existing}")
    for name, content in outputs.items():
        (output_root / name).write_text(content, encoding="utf-8", newline="\n")


def _check_outputs(output_root: Path, outputs: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    expected_names = set(outputs)
    actual_names = {path.name for path in output_root.glob("*") if path.is_file()}
    if actual_names != expected_names:
        errors.append(
            "Generated page set differs: "
            f"expected={sorted(expected_names)} actual={sorted(actual_names)}"
        )
    for name, content in outputs.items():
        path = output_root / name
        if not path.exists():
            errors.append(f"Missing generated page: {path}")
            continue
        if path.read_text(encoding="utf-8") != content:
            errors.append(f"Generated page is stale: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the public RELIC scientific class reference."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Write generated source pages.")
    mode.add_argument("--check", action="store_true", help="Verify generated source pages.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Target directory for generated source pages.",
    )
    args = parser.parse_args()

    outputs = _expected_outputs()
    if args.write:
        _write_outputs(args.output_root, outputs)
        print(f"[OK] Generated {len(outputs)} scientific reference pages.")
        return 0

    errors = _check_outputs(args.output_root, outputs)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    print(f"[OK] Verified {len(outputs)} scientific reference pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
