from __future__ import annotations
import contextlib
from collections.abc import Generator
from functools import partial
from typing import Any, GeneratorType

from pydantic_core import ValidationError
from pydantic import BaseModel, TypeAdapter, create_model, dataclasses, field_validator, validate_call
from pydantic.plugin import PydanticPluginProtocol, SchemaTypePath, ValidateJsonHandlerProtocol, ValidatePythonHandlerProtocol, ValidateStringsHandlerProtocol
from pydantic.plugin._loader import _plugins

@contextlib.contextmanager
def install_plugin(plugin: PydanticPluginProtocol) -> GeneratorType:
    _plugins[plugin.__class__.__qualname__] = plugin
    try:
        yield
    finally:
        _plugins.clear()

def test_on_validate_json_on_success() -> None:
    # ... (rest of the function)

def test_on_validate_json_on_error() -> None:
    # ... (rest of the function)

def test_on_validate_python_on_success() -> None:
    # ... (rest of the function)

def test_on_validate_python_on_error() -> None:
    # ... (rest of the function)

def test_stateful_plugin() -> None:
    # ... (rest of the function)

def test_all_handlers() -> None:
    # ... (rest of the function)

def test_plugin_path_dataclass() -> None:
    # ... (rest of the function)

def test_plugin_path_type_adapter() -> None:
    # ... (rest of the function)

def test_plugin_path_type_adapter_with_module() -> None:
    # ... (rest of the function)

def test_plugin_path_type_adapter_without_name_in_globals() -> None:
    # ... (rest of the function)

def test_plugin_path_validate_call() -> None:
    # ... (rest of the function)

def test_plugin_path_create_model() -> None:
    # ... (rest of the function)

def test_plugin_path_complex() -> None:
    # ... (rest of the function)
