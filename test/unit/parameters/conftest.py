from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _fresh_parameter_manager_for_parameter_unit_tests(
    fresh_global_parameter_manager: None,
) -> Iterator[None]:
    """Every parameter unit test runs on a fresh process-wide manager
    (test/conftest.py::fresh_global_parameter_manager puts the previous one back)."""
    yield
