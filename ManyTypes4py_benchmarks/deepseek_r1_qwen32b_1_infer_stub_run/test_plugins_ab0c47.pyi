from __future__ import annotations
import contextlib
from collections.abc import Generator
from functools import partial
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    Callable,
    TypeVar,
    overload,
)
from pydantic_core import ValidationError
from pydantic import (
    BaseModel,
    TypeAdapter,
    create_model,
    dataclasses,
    field_validator,
    validate_call,
)
from pydantic.plugin import (
    PydanticPluginProtocol,
    SchemaTypePath,
    ValidateJsonHandlerProtocol,
    ValidatePythonHandlerProtocol,
    ValidateStringsHandlerProtocol,
)

T = TypeVar("T")
Self = TypeVar("Self")

@contextlib.contextmanager
def install_plugin(plugin: PydanticPluginProtocol) -> Generator[Any, None, None]:
    ...

class CustomOnValidateJson(ValidateJsonHandlerProtocol):
    def on_enter(self, input: str, *, strict: Optional[bool], context: Optional[Dict[str, Any]], self_instance: Optional[Any]) -> None:
        ...
    def on_success(self, result: BaseModel) -> None:
        ...
    def on_error(self, error: ValidationError) -> None:
        ...

class CustomOnValidatePython(ValidatePythonHandlerProtocol):
    def on_enter(self, input: Dict[str, Any], *, strict: Optional[bool], from_attributes: Optional[bool], context: Optional[Dict[str, Any]], self_instance: Optional[Any]) -> None:
        ...
    def on_success(self, result: BaseModel) -> None:
        ...
    def on_error(self, error: ValidationError) -> None:
        ...
    def on_exception(self, exception: Exception) -> None:
        ...

class Plugin(PydanticPluginProtocol):
    def new_schema_validator(
        self,
        schema: Any,
        schema_type: Union[Type[BaseModel], TypeAdapter[Any], Type[validate_call]],
        schema_type_path: SchemaTypePath,
        schema_kind: str,
        config: Dict[str, Any],
        plugin_settings: Dict[str, Any],
    ) -> Tuple[
        Optional[ValidatePythonHandlerProtocol],
        Optional[ValidateJsonHandlerProtocol],
        Optional[ValidateStringsHandlerProtocol],
    ]:
        ...

class MyException(Exception):
    ...

class Python(ValidatePythonHandlerProtocol):
    def on_enter(self, input: Dict[str, Any], **kwargs: Any) -> None:
        ...
    def on_success(self, result: BaseModel) -> None:
        ...
    def on_error(self, error: ValidationError) -> None:
        ...

class Json(ValidateJsonHandlerProtocol):
    def on_enter(self, input: str, **kwargs: Any) -> None:
        ...
    def on_success(self, result: BaseModel) -> None:
        ...
    def on_error(self, error: ValidationError) -> None:
        ...

class Strings(ValidateStringsHandlerProtocol):
    def on_enter(self, input: Dict[str, str], **kwargs: Any) -> None:
        ...
    def on_success(self, result: BaseModel) -> None:
        ...
    def on_error(self, error: ValidationError) -> None:
        ...

class Model(BaseModel):
    a: int
    ...

def test_on_validate_json_on_success() -> None:
    ...

def test_on_validate_json_on_error() -> None:
    ...

def test_on_validate_python_on_success() -> None:
    ...

def test_on_validate_python_on_error() -> None:
    ...

def test_stateful_plugin() -> None:
    ...

def test_all_handlers() -> None:
    ...

def test_plugin_path_dataclass() -> None:
    ...

def test_plugin_path_type_adapter() -> None:
    ...

def test_plugin_path_type_adapter_with_module() -> None:
    ...

def test_plugin_path_type_adapter_without_name_in_globals() -> None:
    ...

def test_plugin_path_validate_call() -> None:
    ...

def test_plugin_path_create_model() -> None:
    ...

def test_plugin_path_complex() -> None:
    ...