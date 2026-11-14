"""
Integration tests for 15N isotope tracking module

This package contains integration tests that validate the isotope tracking
module's interaction with core RELIC components.

Test Structure:
- test_state_manager_integration.py: Tests with StateManager callbacks
- test_simulator_integration.py: Tests with full simulation integration

To run all integration tests:
    python -m pytest test/integration/isotopes/ -v

To run specific test files:
    python -m pytest test/integration/isotopes/test_state_manager_integration.py -v
    python -m pytest test/integration/isotopes/test_simulator_integration.py -v
"""

__version__ = "1.0.0"
__author__ = "Yuchong Long"

