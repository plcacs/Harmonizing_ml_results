from typing import Any
from collections.abc import Callable

FAKE_UUID: str

def test_valid_base_schema(schema: dict[str, Any] | None) -> None: ...
def test_invalid_base_schema(schema: Any) -> None: ...
def _test_selector(
    selector_type: str,
    schema: dict[str, Any] | None,
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...] | str,
    converter: Callable[[Any], Any] | None = ...,
) -> None: ...
def test_device_selector_schema(
    schema: dict[str, Any] | None,
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_entity_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_entity_selector_schema_error(schema: dict[str, Any]) -> None: ...
def test_area_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_assist_pipeline_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_number_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_number_selector_schema_error(schema: dict[str, Any]) -> None: ...
def test_addon_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_backup_location_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_boolean_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_config_entry_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_country_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_time_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_state_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_target_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_action_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_object_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_text_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_select_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_select_selector_schema_error(schema: dict[str, Any]) -> None: ...
def test_attribute_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_duration_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_icon_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_theme_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_media_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_language_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_location_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_rgb_color_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_color_tempselector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_date_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_datetime_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_template_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_file_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_constant_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_constant_selector_schema_error(schema: dict[str, Any]) -> None: ...
def test_conversation_agent_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_condition_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: str,
) -> None: ...
def test_trigger_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: str,
) -> None: ...
def test_qr_code_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_label_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...
def test_floor_selector_schema(
    schema: dict[str, Any],
    valid_selections: tuple[Any, ...],
    invalid_selections: tuple[Any, ...],
) -> None: ...