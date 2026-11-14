import pytest



@pytest.fixture(autouse=True)
def _reset_parameter_manager(fresh_global_parameter_manager):
    """Each parameter test runs on a fresh manager; the previous one is put back."""
    yield
