from __future__ import annotations
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, Optional, List, Tuple
from pydantic_core import CoreConfig, CoreSchema, SchemaValidator, ValidationError
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from . import BaseValidateHandlerProtocol, PydanticPluginProtocol, SchemaKind, SchemaTypePath

P = ParamSpec("P")
R = TypeVar("R")
Event = Literal["on_validate_python", "on_validate_json", "on_validate_strings"]
events: List[Event] = list(Event.__args__)

def create_schema_validator(
    schema: CoreSchema,
    schema_type: Any,
    schema_type_module: str,
    schema_type_name: str,
    schema_kind: Any,
    config: Optional[CoreConfig] = None,
    plugin_settings: Optional[dict[str, Any]] = None,
) -> Union[SchemaValidator, PluggableSchemaValidator]:
    from . import SchemaTypePath
    from ._loader import get_plugins
    plugins: List[Any] = get_plugins()
    if plugins:
        return PluggableSchemaValidator(
            schema,
            schema_type,
            SchemaTypePath(schema_type_module, schema_type_name),
            schema_kind,
            config,
            plugins,
            plugin_settings or {},
        )
    else:
        return SchemaValidator(schema, config)

class PluggableSchemaValidator:
    __slots__ = ("_schema_validator", "validate_json", "validate_python", "validate_strings")

    def __init__(
        self,
        schema: CoreSchema,
        schema_type: Any,
        schema_type_path: Any,  # Expected to be of type SchemaTypePath
        schema_kind: Any,
        config: Optional[CoreConfig],
        plugins: List[Any],  # Expected to be List[PydanticPluginProtocol]
        plugin_settings: dict[str, Any],
    ) -> None:
        self._schema_validator = SchemaValidator(schema, config)
        python_event_handlers: List[Any] = []
        json_event_handlers: List[Any] = []
        strings_event_handlers: List[Any] = []
        for plugin in plugins:
            try:
                p, j, s = plugin.new_schema_validator(
                    schema, schema_type, schema_type_path, schema_kind, config, plugin_settings
                )
            except TypeError as e:
                raise TypeError(
                    f"Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}"
                ) from e
            if p is not None:
                python_event_handlers.append(p)
            if j is not None:
                json_event_handlers.append(j)
            if s is not None:
                strings_event_handlers.append(s)
        self.validate_python = build_wrapper(self._schema_validator.validate_python, python_event_handlers)
        self.validate_json = build_wrapper(self._schema_validator.validate_json, json_event_handlers)
        self.validate_strings = build_wrapper(self._schema_validator.validate_strings, strings_event_handlers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._schema_validator, name)

def build_wrapper(
    func: Callable[..., R],
    event_handlers: Iterable[Any],  # Expected to be Iterable[BaseValidateHandlerProtocol]
) -> Callable[..., R]:
    if not event_handlers:
        return func
    else:
        on_enters: Tuple[Callable[..., Any], ...] = tuple(
            (h.on_enter for h in event_handlers if filter_handlers(h, "on_enter"))
        )
        on_successes: Tuple[Callable[[Any], Any], ...] = tuple(
            (h.on_success for h in event_handlers if filter_handlers(h, "on_success"))
        )
        on_errors: Tuple[Callable[[ValidationError], Any], ...] = tuple(
            (h.on_error for h in event_handlers if filter_handlers(h, "on_error"))
        )
        on_exceptions: Tuple[Callable[[Exception], Any], ...] = tuple(
            (h.on_exception for h in event_handlers if filter_handlers(h, "on_exception"))
        )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for on_enter_handler in on_enters:
                on_enter_handler(*args, **kwargs)
            try:
                result = func(*args, **kwargs)
            except ValidationError as error:
                for on_error_handler in on_errors:
                    on_error_handler(error)
                raise
            except Exception as exception:
                for on_exception_handler in on_exceptions:
                    on_exception_handler(exception)
                raise
            else:
                for on_success_handler in on_successes:
                    on_success_handler(result)
                return result

        return wrapper

def filter_handlers(handler_cls: Any, method_name: str) -> bool:
    handler = getattr(handler_cls, method_name, None)
    if handler is None:
        return False
    elif handler.__module__ == "pydantic.plugin":
        return False
    else:
        return True