from enum import Enum
from typing import Any, Dict, List, Tuple
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID = 'a266a680b608c32770e6c45bfe6b8411'

def _test_selector(
    selector_type: str,
    schema: Dict[str, Any],
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...],
    converter: callable = None
) -> None:
    ...

@pytest.mark.parametrize(
    'schema', [
        {},
        {'device': None},
        {'entity': None},
        {'area': None},
        {'assist_pipeline': None},
        {'country': None},
        {'theme': None},
        {'language': None},
        {'location': None},
        {'file': None},
        {'constant': None},
        {'conversation_agent': None},
        {'condition': None},
        {'trigger': None},
        {'qr_code': None},
        {'label': None},
        {'floor': None}
    ]
)
def test_selector_schema_error(schema: Dict[str, Any]) -> None:
    with pytest.raises(vol.Invalid):
        selector.validate_selector({selector_type: schema})

@pytest.mark.parametrize(
    'schema', [
        {'value': True, 'label': 'Blah'},
        {'value': False},
        {'value': 0},
        {'value': 1},
        {'value': 4},
        {'value': 'dog'}
    ]
)
def test_constant_selector_schema_error(schema: Dict[str, Any]) -> None:
    with pytest.raises(vol.Invalid):
        selector.validate_selector({'constant': schema})
