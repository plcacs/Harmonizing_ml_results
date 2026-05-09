from enum import StrEnum
import logging
import sys
from typing import Any, Callable, List, Optional, Tuple

class MockBaseClassDeprecatedProperty:
    """Mock base class for deprecated testing."""

    @property
    @deprecated_substitute('old_property')
    def new_property(self) -> str:
        """Test property to fetch."""
        return 'default_new'

@patch('logging.getLogger')
def test_deprecated_substitute_old_class(mock_get_logger: Callable[[], logging.Logger], 
                                         ) -> None:
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
def test_deprecated_substitute_default_class(mock_get_logger: Callable[[], logging.Logger], 
                                            ) -> None:
    """Test deprecated class object."""

    class MockDefaultClass(MockBaseClassDeprecatedProperty):
        """Mock updated class object."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_object = MockDefaultClass()
    assert mock_object.new_property == 'default_new'
    assert not mock_logger.warning.called

@patch('logging.getLogger')
def test_deprecated_substitute_new_class(mock_get_logger: Callable[[], logging.Logger], 
                                        ) -> None:
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
def test_config_get_deprecated_old(mock_get_logger: Callable[[], logging.Logger], 
                                  ) -> None:
    """Test deprecated config."""

    class MockDeprecatedClass:
        """Mock deprecated class object."""

        @property
        def old_property(self) -> str:
            """Test property to fetch."""
            return 'old'
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    config = {'old_name': True}
    assert get_deprecated(config, 'new_name', 'old_name') is True
    assert mock_logger.warning.called
    assert len(mock_logger.warning.mock_calls) == 1

@patch('logging.getLogger')
def test_config_get_deprecated_new(mock_get_logger: Callable[[], logging.Logger], 
                                  ) -> None:
    """Test deprecated config."""

    class MockUpdatedClass:
        """Mock updated class object."""

        @property
        def new_property(self) -> str:
            """Test property to fetch."""
            return 'new'
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    config = {'new_name': True}
    assert get_deprecated(config, 'new_name', 'old_name') is True
    assert not mock_logger.warning.called

@deprecated_class('homeassistant.blah.NewClass')
class MockDeprecatedClass:
    """Mock class for deprecated testing."""

@patch('logging.getLogger')
def test_deprecated_class(mock_get_logger: Callable[[], logging.Logger], 
                         ) -> None:
    """Test deprecated class."""

    class MockDeprecatedClass:
        """Mock deprecated class object."""

    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    MockDeprecatedClass()
    assert mock_logger.warning.called
    assert len(mock_logger.warning.mock_calls) == 1

@pytest.mark.parametrize('breaks_in_ha_version', [None, '2099.1'])
def test_deprecated_function(caplog: List[Tuple[str, int, str]], 
                            breaks_in_ha_version: Optional[str], 
                            ) -> None:
    """Test deprecated_function decorator."""

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass
    mock_deprecated_function()
    assert f'mock_deprecated_function is a deprecated function{breaks_in_ha_version}. Use new_function instead' in caplog.text

@pytest.mark.parametrize('breaks_in_ha_version', [None, '2099.1'])
def test_deprecated_function_called_from_built_in_integration(caplog: List[Tuple[str, int, str]], 
                                                             breaks_in_ha_version: Optional[str], 
                                                             ) -> None:
    """Test deprecated_function decorator."""

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass
    with patch('homeassistant.helpers.frame.linecache.getline', return_value='await session.close()'), patch('homeassistant.helpers.frame.get_current_frame', return_value=extract_stack_to_frame([Mock(filename='/home/paulus/homeassistant/core.py', lineno='23', line='do_something()'), Mock(filename='/home/paulus/homeassistant/components/hue/light.py', lineno='23', line='await session.close()'), Mock(filename='/home/paulus/aiohue/lights.py', lineno='2', line='something()')])):
        mock_deprecated_function()
    assert f'mock_deprecated_function was called from hue, this is a deprecated function{breaks_in_ha_version}. Use new_function instead' in caplog.text

@pytest.mark.parametrize('breaks_in_ha_version', [None, '2099.1'])
def test_deprecated_function_called_from_custom_integration(hass: HomeAssistant, 
                                                           caplog: List[Tuple[str, int, str]], 
                                                           breaks_in_ha_version: Optional[str], 
                                                           ) -> None:
    """Test deprecated_function decorator."""

    mock_integration(hass, MockModule('hue'), built_in=False)

    @deprecated_function('new_function', breaks_in_ha_version=breaks_in_ha_version)
    def mock_deprecated_function() -> None:
        pass
    with patch('homeassistant.helpers.frame.linecache.getline', return_value='await session.close()'), patch('homeassistant.helpers.frame.get_current_frame', return_value=extract_stack_to_frame([Mock(filename='/home/paulus/homeassistant/core.py', lineno='23', line='do_something()'), Mock(filename='/home/paulus/config/custom_components/hue/light.py', lineno='23', line='await session.close()'), Mock(filename='/home/paulus/aiohue/lights.py', lineno='2', line='something()')])):
        mock_deprecated_function()
    assert f"mock_deprecated_function was called from hue, this is a deprecated function{breaks_in_ha_version}. Use new_function instead, please report it to the author of the 'hue' custom integration" in caplog.text

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

@pytest.mark.parametrize('deprecated_constant', [DeprecatedConstant('value', 'NEW_CONSTANT', None), 
                                                 DeprecatedConstant(1, 'NEW_CONSTANT', '2099.1'), 
                                                 DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, None), 
                                                 DeprecatedConstantEnum(TestDeprecatedConstantEnum.TEST, '2099.1'), 
                                                 DeprecatedAlias(1, 'new_alias', None), 
                                                 DeprecatedAlias(1, 'new_alias', '2099.1')])
@pytest.mark.parametrize('module_name', ['homeassistant.components.hue.light', 'config.custom_components.hue.light'])
def test_check_if_deprecated_constant(caplog: List[Tuple[str, int, str]], 
                                      deprecated_constant: Any, 
                                      module_name: str, 
                                      ) -> None:
    """Test check_if_deprecated_constant."""

    module_globals = {'__name__': module_name, '_DEPRECATED_TEST_CONSTANT': deprecated_constant}
    filename = f'/home/paulus/{module_name.replace('.', '/')}.py'
    with patch.dict(sys.modules, {module_name: Mock(__file__=filename)}), patch('homeassistant.helpers.frame.linecache.getline', return_value='await session.close()'), patch('homeassistant.helpers.frame.get_current_frame', return_value=extract_stack_to_frame([Mock(filename='/home/paulus/homeassistant/core.py', lineno='23', line='do_something()'), Mock(filename=filename, lineno='23', line='await session.close()'), Mock(filename='/home/paulus/aiohue/lights.py', lineno='2', line='something()')])):
        value = check_if_deprecated_constant('TEST_CONSTANT', module_globals)
        assert value == _get_value(deprecated_constant)
    assert (module_name, logging.WARNING, f'TEST_CONSTANT was used from hue, this is a deprecated constant{deprecated_constant}. Use new_constant instead') in caplog.record_tuples

@pytest.mark.parametrize('module_globals', [{'CONSTANT': 1}, {'_DEPRECATED_CONSTANT': 1, 'SOMETHING': 2}])
def test_dir_with_deprecated_constants(module_globals: dict, 
                                      ) -> List[str]:
    """Test dir() with deprecated constants."""

    return list(module_globals.keys())

@pytest.mark.parametrize('module_name', ['homeassistant.components.hue.light', 'config.custom_components.hue.light'])
def test_enum_with_deprecated_members(caplog: List[Tuple[str, int, str]], 
                                      module_name: str, 
                                      ) -> None:
    """Test EnumWithDeprecatedMembers."""

    class TestEnum(StrEnum, metaclass=EnumWithDeprecatedMembers, deprecated={'CATS': ('TestEnum.CATS_PER_CM', '2025.11.0'), 'DOGS': ('TestEnum.DOGS_PER_CM', None)}):
        """Zoo units."""

        CATS_PER_CM = 'cats/cm'
        DOGS_PER_CM = 'dogs/cm'
        CATS = 'cats/cm'
        DOGS = 'dogs/cm'
    with patch.dict(sys.modules, {module_name: Mock(__file__=f'/home/paulus/{module_name.replace(".", "/")}.py')}) as mock_module:
        TestEnum.CATS
        TestEnum.DOGS
    assert len(caplog.record_tuples) == 2
    assert ('tests.helpers.test_deprecation', logging.WARNING, f'TestEnum.CATS was used from hue, this is a deprecated enum member which will be removed in HA Core 2025.11.0. Use TestEnum.CATS_PER_CM instead') in caplog.record_tuples
    assert ('tests.helpers.test_deprecation', logging.WARNING, f'TestEnum.DOGS was used from hue, this is a deprecated enum member. Use TestEnum.DOGS_PER_CM instead') in caplog.record_tuples

def test_enum_with_deprecated_members_integration_not_found(caplog: List[Tuple[str, int, str]], 
                                                            ) -> None:
    """Test check_if_deprecated_constant."""

    class TestEnum(StrEnum, metaclass=EnumWithDeprecatedMembers, deprecated={'CATS': ('TestEnum.CATS_PER_CM', '2025.11.0'), 'DOGS': ('TestEnum.DOGS_PER_CM', None)}):
        """Zoo units."""

        CATS_PER_CM = 'cats/cm'
        DOGS_PER_CM = 'dogs/cm'
        CATS = 'cats/cm'
        DOGS = 'dogs/cm'
    with patch('homeassistant.helpers.frame.get_current_frame', side_effect=MissingIntegrationFrame):
        TestEnum.CATS
        TestEnum.DOGS
    assert len(caplog.record_tuples) == 0
