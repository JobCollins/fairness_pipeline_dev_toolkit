"""
Tests for environment variable configuration.
"""

from __future__ import annotations

import os

from fairness_pipeline_dev_toolkit.config.env import get_env_bool, get_env_int


class TestGetEnvBool:
    """Test suite for get_env_bool() function."""

    def test_true_values(self):
        """Test that all true values return True."""
        true_values = ["true", "1", "yes", "on"]
        for val in true_values:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL", default=False) is True
            # Clean up
            del os.environ["TEST_BOOL"]

    def test_true_values_case_insensitive(self):
        """Test that true values are case insensitive."""
        true_values = ["TRUE", "True", "TrUe", "1", "YES", "Yes", "ON", "On"]
        for val in true_values:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL", default=False) is True
            # Clean up
            del os.environ["TEST_BOOL"]

    def test_false_values(self):
        """Test that false values return False."""
        false_values = ["false", "0", "no", "off", "invalid", "random", ""]
        for val in false_values:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL", default=True) is False
            # Clean up
            del os.environ["TEST_BOOL"]

    def test_false_values_case_insensitive(self):
        """Test that false values are case insensitive."""
        false_values = ["FALSE", "False", "FaLsE", "NO", "No", "OFF", "Off"]
        for val in false_values:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL", default=True) is False
            # Clean up
            del os.environ["TEST_BOOL"]

    def test_none_returns_default(self):
        """Test that None (unset variable) returns default."""
        # Ensure variable is not set
        if "TEST_BOOL" in os.environ:
            del os.environ["TEST_BOOL"]

        assert get_env_bool("TEST_BOOL", default=False) is False
        assert get_env_bool("TEST_BOOL", default=True) is True

    def test_default_value(self):
        """Test that default value is used when variable is not set."""
        if "TEST_BOOL" in os.environ:
            del os.environ["TEST_BOOL"]

        assert get_env_bool("NONEXISTENT_VAR", default=False) is False
        assert get_env_bool("NONEXISTENT_VAR", default=True) is True

    def test_invalid_values_return_false(self):
        """Test that invalid values return False."""
        invalid_values = ["maybe", "2", "truee", "fals", "y", "n", "t", "f"]
        for val in invalid_values:
            os.environ["TEST_BOOL"] = val
            assert get_env_bool("TEST_BOOL", default=True) is False
            # Clean up
            del os.environ["TEST_BOOL"]

    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        os.environ["TEST_BOOL"] = ""
        assert get_env_bool("TEST_BOOL", default=True) is False
        del os.environ["TEST_BOOL"]

    def test_whitespace_handling(self):
        """Test that whitespace is not trimmed (should return False)."""
        # The function uses .lower() but doesn't strip, so whitespace values should return False
        os.environ["TEST_BOOL"] = " true "
        assert get_env_bool("TEST_BOOL", default=True) is False
        del os.environ["TEST_BOOL"]


class TestGetEnvInt:
    """Test suite for get_env_int() function."""

    def test_valid_integers(self):
        """Test that valid integers are parsed correctly."""
        test_cases = [
            ("0", 0),
            ("1", 1),
            ("-1", -1),
            ("42", 42),
            ("-42", -42),
            ("1000", 1000),
            ("-1000", -1000),
            ("2147483647", 2147483647),  # Max 32-bit int
            ("-2147483648", -2147483648),  # Min 32-bit int
        ]
        for val, expected in test_cases:
            os.environ["TEST_INT"] = val
            assert get_env_int("TEST_INT", default=None) == expected
            # Clean up
            del os.environ["TEST_INT"]

    def test_none_returns_default(self):
        """Test that None (unset variable) returns default."""
        # Ensure variable is not set
        if "TEST_INT" in os.environ:
            del os.environ["TEST_INT"]

        assert get_env_int("TEST_INT", default=None) is None
        assert get_env_int("TEST_INT", default=42) == 42
        assert get_env_int("TEST_INT", default=0) == 0
        assert get_env_int("TEST_INT", default=-1) == -1

    def test_invalid_strings_return_default(self):
        """Test that invalid strings return default."""
        invalid_values = [
            "not_a_number",
            "abc",
            "12.34",  # Float string
            "12.0",  # Float string (int() doesn't accept decimal points)
            "true",
            "false",
            "",
            "42.5",  # Float string
            "42abc",  # Contains non-numeric characters
            "abc42",  # Contains non-numeric characters
        ]
        for val in invalid_values:
            os.environ["TEST_INT"] = val
            assert get_env_int("TEST_INT", default=99) == 99
            # Clean up
            del os.environ["TEST_INT"]

    def test_default_value(self):
        """Test that default value is used when variable is not set."""
        if "TEST_INT" in os.environ:
            del os.environ["TEST_INT"]

        assert get_env_int("NONEXISTENT_VAR", default=None) is None
        assert get_env_int("NONEXISTENT_VAR", default=42) == 42
        assert get_env_int("NONEXISTENT_VAR", default=0) == 0

    def test_edge_case_zero(self):
        """Test edge case with zero."""
        os.environ["TEST_INT"] = "0"
        assert get_env_int("TEST_INT", default=99) == 0
        del os.environ["TEST_INT"]

    def test_edge_case_negative_zero(self):
        """Test edge case with negative zero (parsed as 0)."""
        os.environ["TEST_INT"] = "-0"
        assert get_env_int("TEST_INT", default=99) == 0
        del os.environ["TEST_INT"]

    def test_edge_case_very_large_numbers(self):
        """Test edge case with very large numbers."""
        # Python int can handle arbitrarily large numbers
        os.environ["TEST_INT"] = "999999999999999999999"
        result = get_env_int("TEST_INT", default=None)
        assert result == 999999999999999999999
        del os.environ["TEST_INT"]

    def test_edge_case_very_small_numbers(self):
        """Test edge case with very small (negative) numbers."""
        os.environ["TEST_INT"] = "-999999999999999999999"
        result = get_env_int("TEST_INT", default=None)
        assert result == -999999999999999999999
        del os.environ["TEST_INT"]

    def test_float_strings_return_default(self):
        """Test that float strings return default (not converted)."""
        float_strings = ["3.14", "2.0", "-1.5", "0.0", "1e10", "1.5e5"]
        for val in float_strings:
            os.environ["TEST_INT"] = val
            assert get_env_int("TEST_INT", default=99) == 99
            # Clean up
            del os.environ["TEST_INT"]

    def test_whitespace_handling(self):
        """Test that whitespace is stripped by int() (parses successfully)."""
        # int() actually strips whitespace, so " 42 " will parse as 42
        os.environ["TEST_INT"] = " 42 "
        assert get_env_int("TEST_INT", default=99) == 42
        del os.environ["TEST_INT"]

    def test_hex_strings_return_default(self):
        """Test that hex strings return default (not converted)."""
        # int() can parse hex with base parameter, but get_env_int doesn't use it
        hex_strings = ["0xFF", "0x10", "0x0"]
        for val in hex_strings:
            os.environ["TEST_INT"] = val
            assert get_env_int("TEST_INT", default=99) == 99
            # Clean up
            del os.environ["TEST_INT"]
