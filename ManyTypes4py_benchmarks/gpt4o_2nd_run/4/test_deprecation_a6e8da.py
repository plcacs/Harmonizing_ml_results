"""Test deprecation helpers."""
from enum import StrEnum
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers.deprecation import (
    DeprecatedAlias,
    DeprecatedConstant,
    DeprecatedConstantEnum,
    EnumWithDeprecatedMembers,
    check_if_deprecated_constant,
    deprecated_class,
    deprecated_function,
    deprecated_substitute,
    dir_with_deprecated_constants,
    get_deprecated,
)
from homeassistant.helpers.frame import MissingIntegrationFrame
from tests.common import MockModule, extract_stack_to_frame, mock_integration


class MockBaseClassDeprecatedProperty:
    """Mock base class for deprecated testing."""

    @property
    @deprecated_substitute('old_property')
    def new_property(self) -> str:
        """Test property to fetch."""
        return 'default_new'


@patch('logging.getLogger')
def test_deprecated_substitute_old_class(mock_get_logger: MagicMock) -> None:
    """Test deprecated class object."""

    class MockDeprecatedClass(MockBaseClassDeprecatedProperty):
        """Mock deprecated class object."""

        @property
        def old_property(self) -> str:
            """Test property to fetch."""
            return 'old'

    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_object = MockDeprecatedClass()
    assert mock_object.new_property == 'old'
    assert mock_logger.warning.called
    assert len(mock_logger.warning.mock_calls) == 1


@patch('logging.getLogger')
def test_deprecated_substitute_default_class(mock_get_logger: MagicMock) -> None:
    """Test deprecated class object."""

    class MockDefaultClass(MockBaseClassDeprecatedProperty):
        """Mock updated class object."""

    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_object = MockDefaultClass()
    assert mock_object.new_property == 'default_new'
    assert not mock_logger.warning.called


@patch('logging.getLogger')
def test_deprecated_substitute_new_class(mock_get_logger: MagicMock) -> None:
    """Test deprecated class object."""

    class MockUpdatedClass(MockBaseClassDeprecatedProperty):
        """Mock updated class object."""

        @property
        def new_property(self) -> str:
            """Test property to fetch."""
            return 'new'

    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_object = MockUpdatedClass()
    assert mock_object.new_property == 'new'
    assert not mock_logger.warning.called


@patch('logging.getLogger')
def test_config_get_deprecated_old(mock_get_logger: MagicMock) -> None:
    """Test deprecated config."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    config: Dict[str, bool] = {'old_name': True}
    assert get_deprecated(config, 'new_name', 'old_name') is True
    assert mock_logger.warning.called
    assert len(mock_logger.warning.mock_calls) == 1


@patch('logging.getLogger')
def test_config_get_deprecated_new(mock_get_logger: MagicMock) -> None:
    """Test deprecated config."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    config: Dict[str, bool] = {'new_name': True}
    assert get_deprecated(config, 'new_name', 'old_name') is True
    assert not mock_logger.warning.called


@deprecated_class('homeassistant.blah.NewClass')
class MockDeprecatedClass:
    """Mock class for deprecated testing."""


@patch('logging.getLogger')
def test_deprecated_class(mock_get_logger: MagicMock) -> None:
    """Test deprecated class."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    MockDeprecatedClass()
    assert mock_logger.warning.called
    assert len(mock_logger.warning.mock_calls) == 1


@pytest.mark.parametrize(
    ('breaks_in_ha_version', 'extra_msg'),
    [(None, ''), ('2099.1', ' which will be removed in HA Core 2099.1')],
)
def test_deprecated_function(
    caplog: Any, breaks_in_ha_version: Optional[str], extra_msg: str
) -> None:
    """Test deprecated_function decorator.

    This tests the behavior when the calling integration is not known.
    """

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass

    mock_deprecated_function()
    assert (
        f'mock_deprecated_function is a deprecated function{extra_msg}. Use new_function instead'
        in caplog.text
    )


@pytest.mark.parametrize(
    ('breaks_in_ha_version', 'extra_msg'),
    [(None, ''), ('2099.1', ' which will be removed in HA Core 2099.1')],
)
def test_deprecated_function_called_from_built_in_integration(
    caplog: Any, breaks_in_ha_version: Optional[str], extra_msg: str
) -> None:
    """Test deprecated_function decorator.

    This tests the behavior when the calling integration is built-in.
    """

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass

    with patch(
        'homeassistant.helpers.frame.linecache.getline',
        return_value='await session.close()',
    ), patch(
        'homeassistant.helpers.frame.get_current_frame',
        return_value=extract_stack_to_frame(
            [
                Mock(
                    filename='/home/paulus/homeassistant/core.py',
                    lineno='23',
                    line='do_something()',
                ),
                Mock(
                    filename='/home/paulus/homeassistant/components/hue/light.py',
                    lineno='23',
                    line='await session.close()',
                ),
                Mock(
                    filename='/home/paulus/aiohue/lights.py',
                    lineno='2',
                    line='something()',
                ),
            ]
        ),
    ):
        mock_deprecated_function()
    assert (
        f'mock_deprecated_function was called from hue, this is a deprecated function{extra_msg}. Use new_function instead'
        in caplog.text
    )


@pytest.mark.parametrize(
    ('breaks_in_ha_version', 'extra_msg'),
    [(None, ''), ('2099.1', ' which will be removed in HA Core 2099.1')],
)
def test_deprecated_function_called_from_custom_integration(
    hass: HomeAssistant, caplog: Any, breaks_in_ha_version: Optional[str], extra_msg: str
) -> None:
    """Test deprecated_function decorator.

    This tests the behavior when the calling integration is custom.
    """
    mock_integration(hass, MockModule('hue'), built_in=False)

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass

    with patch(
        'homeassistant.helpers.frame.linecache.getline',
        return_value='await session.close()',
    ), patch(
        'homeassistant.helpers.frame.get_current_frame',
        return_value=extract_stack_to_frame(
            [
                Mock(
                    filename='/home/paulus/homeassistant/core.py',
                    lineno='23',
                    line='do_something()',
                ),
                Mock(
                    filename='/home/paulus/config/custom_components/hue/light.py',
                    lineno='23',
                    line='await session.close()',
                ),
                Mock(
                    filename='/home/paulus/aiohue/lights.py',
                    lineno='2',
                    line='something()',
                ),
            ]
        ),
    ):
        mock_deprecated_function()
    assert (
        f"mock_deprecated_function was called from hue, this is a deprecated function{extra_msg}. Use new_function instead, please report it to the author of the 'hue' custom integration"
        in caplog.text
    )


class TestDeprecatedConstantEnum(StrEnum):
    """Test deprecated constant enum."""

    __test__ = False
    TEST = 'value'


def _get_value(obj: Any) -> Any:
    if isinstance(obj, DeprecatedConstant):
        return obj.value
    if isinstance(obj, DeprecatedConstantEnum):
        return obj.enum
    if isinstance(obj, DeprecatedAlias):
        return obj.value
    if len(obj) == 2:
        return obj[0].value
    return obj[0]


@pytest.mark.parametrize(
    ('deprecated_constant', 'extra_msg', 'description'),
    [
        (DeprecatedConstant('value', 'NEW_CONSTANT', None), '. Use NEW_CONSTANT instead', 'constant'),
        (DeprecatedConstant(1, 'NEW_CONSTANT', '2099.1'), ' which will be removed in HA Core 2099.1. Use NEW_CONSTANT instead', 'constant'),
        (DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, None), '. Use TestDeprecatedConstantEnum.TEST instead', 'constant'),
        (DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, '2099.1'), ' which will be removed in HA Core 2099.1. Use TestDeprecatedConstantEnum.TEST instead', 'constant'),
        (DeprecatedAlias(1, 'new_alias', None), '. Use new_alias instead', 'alias'),
        (DeprecatedAlias(1, 'new_alias', '2099.1'), ' which will be removed in HA Core 2099.1. Use new_alias instead', 'alias'),
    ],
)
@pytest.mark.parametrize(
    ('module_name', 'extra_extra_msg'),
    [
        ('homeassistant.components.hue.light', ''),
        ('config.custom_components.hue.light', ", please report it to the author of the 'hue' custom integration"),
    ],
)
def test_check_if_deprecated_constant(
    caplog: Any,
    deprecated_constant: Union[DeprecatedConstant, DeprecatedConstantEnum, DeprecatedAlias],
    extra_msg: str,
    module_name: str,
    extra_extra_msg: str,
    description: str,
) -> None:
    """Test check_if_deprecated_constant."""
    module_globals: Dict[str, Any] = {
        '__name__': module_name,
        '_DEPRECATED_TEST_CONSTANT': deprecated_constant,
    }
    filename = f'/home/paulus/{module_name.replace(".", "/")}.py'
    with patch.dict(
        sys.modules, {module_name: Mock(__file__=filename)}
    ), patch(
        'homeassistant.helpers.frame.linecache.getline',
        return_value='await session.close()',
    ), patch(
        'homeassistant.helpers.frame.get_current_frame',
        return_value=extract_stack_to_frame(
            [
                Mock(
                    filename='/home/paulus/homeassistant/core.py',
                    lineno='23',
                    line='do_something()',
                ),
                Mock(
                    filename=filename,
                    lineno='23',
                    line='await session.close()',
                ),
                Mock(
                    filename='/home/paulus/aiohue/lights.py',
                    lineno='2',
                    line='something()',
                ),
            ]
        ),
    ):
        value = check_if_deprecated_constant('TEST_CONSTANT', module_globals)
        assert value == _get_value(deprecated_constant)
    assert (
        module_name,
        logging.WARNING,
        f'TEST_CONSTANT was used from hue, this is a deprecated {description}{extra_msg}{extra_extra_msg}',
    ) in caplog.record_tuples


@pytest.mark.parametrize(
    ('deprecated_constant', 'extra_msg', 'description'),
    [
        (DeprecatedConstant('value', 'NEW_CONSTANT', None), '. Use NEW_CONSTANT instead', 'constant'),
        (DeprecatedConstant(1, 'NEW_CONSTANT', '2099.1'), ' which will be removed in HA Core 2099.1. Use NEW_CONSTANT instead', 'constant'),
        (DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, None), '. Use TestDeprecatedConstantEnum.TEST instead', 'constant'),
        (DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, '2099.1'), ' which will be removed in HA Core 2099.1. Use TestDeprecatedConstantEnum.TEST instead', 'constant'),
        (DeprecatedAlias(1, 'new_alias', None), '. Use new_alias instead', 'alias'),
        (DeprecatedAlias(1, 'new_alias', '2099.1'), ' which will be removed in HA Core 2099.1. Use new_alias instead', 'alias'),
    ],
)
@pytest.mark.parametrize(
    'module_name',
    ['homeassistant.components.hue.light', 'config.custom_components.hue.light'],
)
def test_check_if_deprecated_constant_integration_not_found(
    caplog: Any,
    deprecated_constant: Union[DeprecatedConstant, DeprecatedConstantEnum, DeprecatedAlias],
    extra_msg: str,
    module_name: str,
    description: str,
) -> None:
    """Test check_if_deprecated_constant."""
    module_globals: Dict[str, Any] = {
        '__name__': module_name,
        '_DEPRECATED_TEST_CONSTANT': deprecated_constant,
    }
    with patch(
        'homeassistant.helpers.frame.get_current_frame',
        side_effect=MissingIntegrationFrame,
    ):
        value = check_if_deprecated_constant('TEST_CONSTANT', module_globals)
        assert value == _get_value(deprecated_constant)
    assert (
        module_name,
        logging.WARNING,
        f'TEST_CONSTANT is a deprecated {description}{extra_msg}',
    ) not in caplog.record_tuples


def test_test_check_if_deprecated_constant_invalid(caplog: Any) -> None:
    """Test check_if_deprecated_constant error handling.

    Test check_if_deprecated_constant raises an attribute error and creates a log entry
    on an invalid deprecation type.
    """
    module_name = 'homeassistant.components.hue.light'
    module_globals: Dict[str, Any] = {
        '__name__': module_name,
        '_DEPRECATED_TEST_CONSTANT': 1,
    }
    name = 'TEST_CONSTANT'
    excepted_msg = (
        f"Value of _DEPRECATED_{name} is an instance of <class 'int'> but an instance of "
        "DeprecatedAlias, DeferredDeprecatedAlias, DeprecatedConstant or DeprecatedConstantEnum is required"
    )
    with pytest.raises(AttributeError, match=excepted_msg):
        check_if_deprecated_constant(name, module_globals)
    assert (module_name, logging.DEBUG, excepted_msg) in caplog.record_tuples


@pytest.mark.parametrize(
    ('module_globals', 'expected'),
    [
        ({'CONSTANT': 1}, ['CONSTANT']),
        ({'_DEPRECATED_CONSTANT': 1}, ['_DEPRECATED_CONSTANT', 'CONSTANT']),
        ({'_DEPRECATED_CONSTANT': 1, 'SOMETHING': 2}, ['_DEPRECATED_CONSTANT', 'SOMETHING', 'CONSTANT']),
    ],
)
def test_dir_with_deprecated_constants(
    module_globals: Dict[str, Any], expected: List[str]
) -> None:
    """Test dir() with deprecated constants."""
    assert dir_with_deprecated_constants([*module_globals.keys()]) == expected


@pytest.mark.parametrize(
    ('module_name', 'extra_extra_msg'),
    [
        ('homeassistant.components.hue.light', ''),
        ('config.custom_components.hue.light', ", please report it to the author of the 'hue' custom integration"),
    ],
)
def test_enum_with_deprecated_members(caplog: Any, module_name: str, extra_extra_msg: str) -> None:
    """Test EnumWithDeprecatedMembers."""
    filename = f'/home/paulus/{module_name.replace(".", "/")}.py'

    class TestEnum(
        StrEnum,
        metaclass=EnumWithDeprecatedMembers,
        deprecated={
            'CATS': ('TestEnum.CATS_PER_CM', '2025.11.0'),
            'DOGS': ('TestEnum.DOGS_PER_CM', None),
        },
    ):
        """Zoo units."""

        CATS_PER_CM = 'cats/cm'
        DOGS_PER_CM = 'dogs/cm'
        CATS = 'cats/cm'
        DOGS = 'dogs/cm'

    with patch.dict(
        sys.modules, {module_name: Mock(__file__=filename)}
    ), patch(
        'homeassistant.helpers.frame.linecache.getline',
        return_value='await session.close()',
    ), patch(
        'homeassistant.helpers.frame.get_current_frame',
        return_value=extract_stack_to_frame(
            [
                Mock(
                    filename='/home/paulus/homeassistant/core.py',
                    lineno='23',
                    line='do_something()',
                ),
                Mock(
                    filename=filename,
                    lineno='23',
                    line='await session.close()',
                ),
                Mock(
                    filename='/home/paulus/aiohue/lights.py',
                    lineno='2',
                    line='something()',
                ),
            ]
        ),
    ):
        TestEnum.CATS
        TestEnum.DOGS
    assert len(caplog.record_tuples) == 2
    assert (
        'tests.helpers.test_deprecation',
        logging.WARNING,
        f'TestEnum.CATS was used from hue, this is a deprecated enum member which will be removed in HA Core 2025.11.0. Use TestEnum.CATS_PER_CM instead{extra_extra_msg}',
    ) in caplog.record_tuples
    assert (
        'tests.helpers.test_deprecation',
        logging.WARNING,
        f'TestEnum.DOGS was used from hue, this is a deprecated enum member. Use TestEnum.DOGS_PER_CM instead{extra_extra_msg}',
    ) in caplog.record_tuples


def test_enum_with_deprecated_members_integration_not_found(caplog: Any) -> None:
    """Test check_if_deprecated_constant."""

    class TestEnum(
        StrEnum,
        metaclass=EnumWithDeprecatedMembers,
        deprecated={
            'CATS': ('TestEnum.CATS_PER_CM', '2025.11.0'),
            'DOGS': ('TestEnum.DOGS_PER_CM', None),
        },
    ):
        """Zoo units."""

        CATS_PER_CM = 'cats/cm'
        DOGS_PER_CM = 'dogs/cm'
        CATS = 'cats/cm'
        DOGS = 'dogs/cm'

    with patch(
        'homeassistant.helpers.frame.get_current_frame',
        side_effect=MissingIntegrationFrame,
    ):
        TestEnum.CATS
        TestEnum.DOGS
    assert len(caplog.record_tuples) == 0
