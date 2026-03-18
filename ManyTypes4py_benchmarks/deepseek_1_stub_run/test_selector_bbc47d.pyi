```python
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import voluptuous as vol

FAKE_UUID: str = ...

def validate_selector(schema: Any) -> None: ...

def selector(config: Dict[str, Any]) -> Any: ...

class QrErrorCorrectionLevel(Enum):
    LOW = ...
    MEDIUM = ...
    QUARTILE = ...
    HIGH = ...

class _Selector:
    config: Dict[str, Any]
    
    def __init__(self, config: Dict[str, Any]) -> None: ...
    
    def __eq__(self, other: Any) -> bool: ...
    
    def serialize(self) -> Dict[str, Any]: ...

def _test_selector(
    selector_type: str,
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...],
    converter: Optional[Any] = None
) -> None: ...

def test_valid_base_schema(schema: Dict[str, Any]) -> None: ...

def test_invalid_base_schema(schema: Any) -> None: ...

def test_device_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_entity_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_entity_selector_schema_error(schema: Any) -> None: ...

def test_area_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_assist_pipeline_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_number_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_number_selector_schema_error(schema: Any) -> None: ...

def test_addon_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_backup_location_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_boolean_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_config_entry_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_country_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_time_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_state_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_target_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_action_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_object_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_text_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_select_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_select_selector_schema_error(schema: Any) -> None: ...

def test_attribute_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_duration_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_icon_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_theme_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_media_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_language_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_location_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_rgb_color_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_color_tempselector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_date_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_datetime_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_template_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_file_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_constant_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_constant_selector_schema_error(schema: Any) -> None: ...

def test_conversation_agent_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_condition_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_trigger_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_qr_code_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_label_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...

def test_floor_selector_schema(
    schema: Any,
    valid_selections: Tuple[Any, ...],
    invalid_selections: Tuple[Any, ...]
) -> None: ...
```