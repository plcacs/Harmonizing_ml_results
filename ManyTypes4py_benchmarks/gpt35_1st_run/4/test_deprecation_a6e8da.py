from enum import Enum
import logging
import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers.deprecation import DeprecatedAlias, DeprecatedConstant, DeprecatedConstantEnum, EnumWithDeprecatedMembers, check_if_deprecated_constant, deprecated_class, deprecated_function, deprecated_substitute, dir_with_deprecated_constants, get_deprecated
from homeassistant.helpers.frame import MissingIntegrationFrame
from tests.common import MockModule, extract_stack_to_frame, mock_integration

class MockBaseClassDeprecatedProperty:
    @property
    @deprecated_substitute('old_property')
    def new_property(self) -> str:
        return 'default_new'

@patch('logging.getLogger')
def test_deprecated_substitute_old_class(mock_get_logger: MagicMock) -> None:
    ...

@patch('logging.getLogger')
def test_deprecated_substitute_default_class(mock_get_logger: MagicMock) -> None:
    ...

@patch('logging.getLogger')
def test_deprecated_substitute_new_class(mock_get_logger: MagicMock) -> None:
    ...

@patch('logging.getLogger')
def test_config_get_deprecated_old(mock_get_logger: MagicMock) -> None:
    ...

@patch('logging.getLogger')
def test_config_get_deprecated_new(mock_get_logger: MagicMock) -> None:
    ...

@deprecated_class('homeassistant.blah.NewClass')
class MockDeprecatedClass:
    ...

@patch('logging.getLogger')
def test_deprecated_class(mock_get_logger: MagicMock) -> None:
    ...

def test_deprecated_function(caplog: Any, breaks_in_ha_version: str, extra_msg: str) -> None:
    ...

def test_deprecated_function_called_from_built_in_integration(caplog: Any, breaks_in_ha_version: str, extra_msg: str) -> None:
    ...

def test_deprecated_function_called_from_custom_integration(hass: HomeAssistant, caplog: Any, breaks_in_ha_version: str, extra_msg: str) -> None:
    ...

class TestDeprecatedConstantEnum(Enum):
    ...

def _get_value(obj: Any) -> Any:
    ...

def test_check_if_deprecated_constant(caplog: Any, deprecated_constant: Any, extra_msg: str, description: str) -> None:
    ...

def test_check_if_deprecated_constant_integration_not_found(caplog: Any, deprecated_constant: Any, extra_msg: str, module_name: str, description: str) -> None:
    ...

def test_test_check_if_deprecated_constant_invalid(caplog: Any) -> None:
    ...

def test_dir_with_deprecated_constants(module_globals: dict, expected: list) -> None:
    ...

def test_enum_with_deprecated_members(caplog: Any, module_name: str, extra_extra_msg: str) -> None:
    ...

def test_enum_with_deprecated_members_integration_not_found(caplog: Any) -> None:
    ...
