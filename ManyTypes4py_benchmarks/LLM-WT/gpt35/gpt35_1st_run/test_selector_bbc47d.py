from enum import Enum
from typing import Any, List, Dict, Union
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID: str = 'a266a680b608c32770e6c45bfe6b8411'

def test_valid_base_schema(schema: Dict[str, Any]) -> None:
    """Test base schema validation."""
    selector.validate_selector(schema)

def test_invalid_base_schema(schema: Union[None, str, Dict[str, Any]]) -> None:
    """Test base schema validation."""
    with pytest.raises(vol.Invalid):
        selector.validate_selector(schema)

def _test_selector(selector_type: str, schema: Dict[str, Any], valid_selections: List[Any], invalid_selections: List[Any], converter: Any = None) -> None:
    """Help test a selector."""

    def default_converter(x: Any) -> Any:
        return x

    if converter is None:
        converter = default_converter

    config: Dict[str, Any] = {selector_type: schema}
    selector.validate_selector(config)
    selector_instance = selector.selector(config)
    assert selector_instance == selector.selector(config)
    assert selector_instance != 5
    assert not any((isinstance(val, Enum) for val in selector_instance.config.values()))
    vol_schema = vol.Schema({'selection': selector_instance})

    for selection in valid_selections:
        assert vol_schema({'selection': selection}) == {'selection': converter(selection)}

    for selection in invalid_selections:
        with pytest.raises(vol.Invalid):
            vol_schema({'selection': selection})

    selector_instance = selector.selector({selector_type: schema})
    assert selector_instance.serialize() == {'selector': {selector_type: selector_instance.config}}
    yaml_util.dump(selector_instance.serialize())

def test_device_selector_schema(schema: Dict[str, Any], valid_selections: List[Any], invalid_selections: List[Any]) -> None:
    """Test device selector."""
    _test_selector('device', schema, valid_selections, invalid_selections)

# Add similar type annotations for other test functions
