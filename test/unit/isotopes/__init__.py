"""
Unit tests for 15N isotope tracking module

This package contains comprehensive unit tests for all components of the
15N isotope tracking module in RELIC.

Test Structure:
- test_fractionation.py: Tests for isotope fractionation calculations
- test_config.py: Tests for configuration models and validation  
- test_tracker_service.py: Tests for core tracking service
- test_integration.py: Integration tests between components

To run all isotope unit tests:
    python -m pytest test/unit/isotopes/ -v

To run specific test files:
    python -m pytest test/unit/isotopes/test_fractionation.py -v
    python -m pytest test/unit/isotopes/test_config.py -v
    python -m pytest test/unit/isotopes/test_tracker_service.py -v
"""

__version__ = "1.0.0"
__author__ = "Yuchong Long"

