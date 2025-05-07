"""Pluggable schema validator for pydantic."""
from __future__ import annotations
import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Optional, Dict
from pydantic_core import CoreConfig, CoreSchema, SchemaValidator, ValidationError
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from . import (
        BaseValidateHandlerProtocol,
        PydanticPluginProtocol,
        SchemaKind,
        SchemaTypePath,
    )

P = ParamSpec('P')
R = TypeVar('R')
Event = Literal['on_validate_python', 'on_validate_json', 'on_validate_strings']
events = list(Event.__args__)


def create_schema_validator(
    schema: CoreSchema,
    schema_type: str,
    schema_type_module: str,
    schema_type_name: str,
    schema_kind: SchemaKind,
    config: Optional[CoreConfig] = None,
    plugin_settings: Optional[Dict[str, Any]] = None,
) -> SchemaValidator | PluggableSchemaValidator:
    """Create a `SchemaValidator` or `PluggableSchemaValidator` if plugins are installed.

    Returns:
        If plugins are installed then return `PluggableSchemaValidator`, otherwise return `SchemaValidator`.
    """
    from . import SchemaTypePath
    from ._loader import get_plugins

    plugins: Iterable[PydanticPluginProtocol] = get_plugins()
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
    """Pluggable schema validator."""
    __slots__ = ('_schema_validator', 'validate_json', 'validate_python', 'validate_strings')

    def __init__(
        self,
        schema: CoreSchema,
        schema_type: str,
        schema_type_path: SchemaTypePath,
        schema_kind: SchemaKind,
        config: CoreConfig,
        plugins: Iterable[PydanticPluginProtocol],
        plugin_settings: Dict[str, Any],
    ) -> None:
        self._schema_validator: SchemaValidator = SchemaValidator(schema, config)
        python_event_handlers: list[BaseValidateHandlerProtocol] = []
        json_event_handlers: list[BaseValidateHandlerProtocol] = []
        strings_event_handlers: list[BaseValidateHandlerProtocol] = []
        for plugin in plugins:
            try:
                p: Optional[BaseValidateHandlerProtocol]
                j: Optional[BaseValidateHandlerProtocol]
                s: Optional[BaseValidateHandlerProtocol]
                p, j, s = plugin.new_schema_validator(
                    schema, schema_type, schema_type_path, schema_kind, config, plugin_settings
                )
            except TypeError as e:
                raise TypeError(
                    f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}'
                ) from e
            if p is not None:
                python_event_handlers.append(p)
            if j is not None:
                json_event_handlers.append(j)
            if s is not None:
                strings_event_handlers.append(s)
        self.validate_python: Callable[..., Any] = build_wrapper(
            self._schema_validator.validate_python, python_event_handlers
        )
        self.validate_json: Callable[..., Any] = build_wrapper(
            self._schema_validator.validate_json, json_event_handlers
        )
        self.validate_strings: Callable[..., Any] = build_wrapper(
            self._schema_validator.validate_strings, strings_event_handlers
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._schema_validator, name)


def build_wrapper(
    func: Callable[..., R],
    event_handlers: list[BaseValidateHandlerProtocol],
) -> Callable[..., R]:
    if not event_handlers:
        return func
    else:
        on_enters: tuple[Callable[..., Any], ...] = tuple(
            h.on_enter for h in event_handlers if filter_handlers(h, 'on_enter')
        )
        on_successes: tuple[Callable[[R], Any], ...] = tuple(
            h.on_success for h in event_handlers if filter_handlers(h, 'on_success')
        )
        on_errors: tuple[Callable[[ValidationError], Any], ...] = tuple(
            h.on_error for h in event_handlers if filter_handlers(h, 'on_error')
        )
        on_exceptions: tuple[Callable[[Exception], Any], ...] = tuple(
            h.on_exception for h in event_handlers if filter_handlers(h, 'on_exception')
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


def filter_handlers(
    handler_cls: BaseValidateHandlerProtocol,
    method_name: str,
) -> bool:
    """Filter out handler methods which are not implemented by the plugin directly - e.g. are missing
    or are inherited from the protocol.
    """
    handler = getattr(handler_cls, method_name, None)
    if handler is None:
        return False
    elif handler.__module__ == 'pydantic.plugin':
        return False
    else:
        return True
