"""Module structure tests for the current tutorial.case package layout."""

from __future__ import annotations

import importlib

def test_subpackages_importable() -> None:
    modules = [
        "tutorial.case.run_case",
        "tutorial.case.plot_results",
        "tutorial.case.plot_megan_comparison",
        "tutorial.case.functors.case_relic_wrappers",
        "tutorial.case.functors.infiltration_bridge",
        "tutorial.case.functors.irrigation_precip_adapter",
        "tutorial.case.functors.ml_surrogates",
        "tutorial.case.functors.stomatal_proxy",
        "tutorial.case.functors.stress_key_adapter",
        "tutorial.case.utils.diagnostics_writer",
        "tutorial.case.utils.netcdf_loader",
        "tutorial.case.utils.optimization_loop",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_legacy_real_case_package_removed() -> None:
    legacy_modules = [
        "tutorial.relic_real_case",
        "tutorial.relic_real_case.hydrology_demo.objective",
        "tutorial.relic_real_case.bvoc_demo.objective",
        "tutorial.relic_real_case.simulation",
    ]
    for module_name in legacy_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"Legacy tutorial module should not be importable: {module_name}")
