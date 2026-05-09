"""Test selectors."""

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
import pytest
import voluptuous as vol
from homeassistant.helpers import selector
from homeassistant.util import yaml as yaml_util

FAKE_UUID: str = 'a266a680b608c32770e6c45bfe6b8411'

@pytest.mark.parametrize('schema', [{'device': None}, {'entity': None}])
def test_valid_base_schema(schema: Dict[str, Any]) -> None:
    """Test base schema validation."""
    ...

@pytest.mark.parametrize('schema', [None, 'not_a_dict', {}, {'non_existing': {}}, {'device': {}, 'entity': {}}])
def test_invalid_base_schema(schema: Union[None, str, Dict[str, Any]]) -> None:
    """Test base schema validation."""
    ...

def _test_selector(
    selector_type: str,
    schema: Dict[str, Any],
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
    converter: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Help test a selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        (None, ('abc123',), (None,)),
        ({}, ('abc123',), (None,)),
        # ... (other test cases)
    ],
)
def test_device_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test device selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('sensor.abc123', FAKE_UUID), (None, 'abc123')),
        # ... (other test cases)
    ],
)
def test_entity_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[Union[str, str], ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test entity selector."""
    ...

@pytest.mark.parametrize('schema', [{'filter': [{'supported_features': [1]}]}, {'filter': [{'supported_features': ['blah']}]}])
def test_entity_selector_schema_error(schema: Dict[str, Any]) -> None:
    """Test number selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('abc123',), (None,)),
        # ... (other test cases)
    ],
)
def test_area_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test area selector."""
    ...

# ... (other test functions with similar structure)

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('2022-03-24',), (None, 'abc', '00:00', '2022-03-24 00:00', '2022-03-32')),
    ],
)
def test_date_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test date selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('2022-03-24 00:00', '2022-03-24'), (None, 'abc', '00:00', '2022-03-24 24:01')),
    ],
)
def test_datetime_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test datetime selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('home_assistant', '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1)),
        ({'language': 'nl'}, ('home_assistant', '2j4hp3uy4p87wyrpiuhk34'), (None, True, 1)),
    ],
)
def test_conversation_agent_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test conversation agent selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({'data': 'test', 'scale': 5}, ('test',), (False, 0, [])),
        ({'data': 'test'}, ('test',), (True, 1, [])),
        ({'data': 'test', 'scale': 5, 'error_correction_level': selector.QrErrorCorrectionLevel.HIGH}, ('test',), (True, 1, [])),
    ],
)
def test_qr_code_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test QR code selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('abc123',), (None,)),
        ({'multiple': True}, (['abc123', 'def456'],), (None, 'abc123', ['abc123', None])),
    ],
)
def test_label_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Union[Tuple[str, ...], Tuple[List[str], ...]],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test label selector."""
    ...

@pytest.mark.parametrize(
    ('schema', 'valid_selections', 'invalid_selections'),
    [
        ({}, ('abc123',), (None,)),
        # ... (other test cases)
    ],
)
def test_floor_selector_schema(
    schema: Dict[str, Any],
    valid_selections: Tuple[str, ...],
    invalid_selections: Tuple[Any, ...],
) -> None:
    """Test floor selector."""
    ...