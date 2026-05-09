"""Test selectors."""

from typing import Any, Callable, Optional, Union, Sequence, Iterable

FAKE_UUID: str

def test_valid_base_schema(schema: dict[str, Any]) -> None:
    """Test base schema validation."""
    ...

def test_invalid_base_schema(schema: Any) -> None:
    """Test base schema validation."""
    ...

def _test_selector(
    selector_type: str,
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
    converter: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Help test a selector."""
    ...

def test_device_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test device selector."""
    ...

def test_entity_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test entity selector."""
    ...

def test_entity_selector_schema_error(schema: dict[str, Any]) -> None:
    """Test number selector."""
    ...

def test_area_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test area selector."""
    ...

def test_assist_pipeline_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test assist pipeline selector."""
    ...

def test_number_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test number selector."""
    ...

def test_number_selector_schema_error(schema: dict[str, Any]) -> None:
    """Test number selector."""
    ...

def test_addon_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test add-on selector."""
    ...

def test_backup_location_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test backup location selector."""
    ...

def test_boolean_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test boolean selector."""
    ...

def test_config_entry_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test config entry selector."""
    ...

def test_country_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test country selector."""
    ...

def test_time_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test time selector."""
    ...

def test_state_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test state selector."""
    ...

def test_target_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test target selector."""
    ...

def test_action_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test action sequence selector."""
    ...

def test_object_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test object selector."""
    ...

def test_text_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test text selector."""
    ...

def test_select_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test select selector."""
    ...

def test_select_selector_schema_error(schema: dict[str, Any]) -> None:
    """Test select selector."""
    ...

def test_attribute_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test attribute selector."""
    ...

def test_duration_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test duration selector."""
    ...

def test_icon_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test icon selector."""
    ...

def test_theme_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test theme selector."""
    ...

def test_media_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test media selector."""
    ...

def test_language_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test language selector."""
    ...

def test_location_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test location selector."""
    ...

def test_rgb_color_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test color_rgb selector."""
    ...

def test_color_tempselector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test color_temp selector."""
    ...

def test_date_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test date selector."""
    ...

def test_datetime_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test datetime selector."""
    ...

def test_template_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test template selector."""
    ...

def test_file_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test file selector."""
    ...

def test_constant_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test constant selector."""
    ...

def test_constant_selector_schema_error(schema: dict[str, Any]) -> None:
    """Test constant selector."""
    ...

def test_conversation_agent_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test conversation agent selector."""
    ...

def test_condition_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test condition sequence selector."""
    ...

def test_trigger_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test trigger sequence selector."""
    ...

def test_qr_code_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test QR code selector."""
    ...

def test_label_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test label selector."""
    ...

def test_floor_selector_schema(
    schema: Any,
    valid_selections: Iterable[Any],
    invalid_selections: Iterable[Any],
) -> None:
    """Test floor selector."""
    ...