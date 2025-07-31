from __future__ import annotations
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Optional, TypeVar, Union, Type
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator
from ..errors import PydanticErrorCodes, PydanticUserError
from ..plugin._schema_validator import PluggableSchemaValidator

if TYPE_CHECKING:
    from ..dataclasses import PydanticDataclass
    from ..main import BaseModel
    from ..type_adapter import TypeAdapter

ValSer = TypeVar('ValSer', bound=Union[SchemaValidator, PluggableSchemaValidator, SchemaSerializer])
T = TypeVar('T')


class MockCoreSchema(Mapping[str, Any]):
    """Mocker for `pydantic_core.CoreSchema` which optionally attempts to
    rebuild the thing it's mocking when one of its methods is accessed and raises an error if that fails.
    """
    __slots__ = ('_error_message', '_code', '_attempt_rebuild', '_built_memo')

    def __init__(
        self,
        error_message: str,
        *,
        code: str,
        attempt_rebuild: Optional[Callable[[], Optional[Mapping[str, Any]]]] = None
    ) -> None:
        self._error_message: str = error_message
        self._code: str = code
        self._attempt_rebuild: Optional[Callable[[], Optional[Mapping[str, Any]]]] = attempt_rebuild
        self._built_memo: Optional[Mapping[str, Any]] = None

    def __getitem__(self, key: str) -> Any:
        return self._get_built().__getitem__(key)

    def __len__(self) -> int:
        return self._get_built().__len__()

    def __iter__(self) -> Iterator[str]:
        return self._get_built().__iter__()

    def _get_built(self) -> Mapping[str, Any]:
        if self._built_memo is not None:
            return self._built_memo
        if self._attempt_rebuild:
            schema: Optional[Mapping[str, Any]] = self._attempt_rebuild()
            if schema is not None:
                self._built_memo = schema
                return schema
        raise PydanticUserError(self._error_message, code=self._code)

    def rebuild(self) -> Optional[Mapping[str, Any]]:
        self._built_memo = None
        if self._attempt_rebuild:
            schema: Optional[Mapping[str, Any]] = self._attempt_rebuild()
            if schema is not None:
                return schema
            else:
                raise PydanticUserError(self._error_message, code=self._code)
        return None


class MockValSer(Generic[ValSer]):
    """Mocker for `pydantic_core.SchemaValidator` or `pydantic_core.SchemaSerializer` which optionally attempts to
    rebuild the thing it's mocking when one of its methods is accessed and raises an error if that fails.
    """
    __slots__ = ('_error_message', '_code', '_val_or_ser', '_attempt_rebuild')

    def __init__(
        self,
        error_message: str,
        *,
        code: str,
        val_or_ser: Literal['validator', 'serializer'],
        attempt_rebuild: Optional[Callable[[], Optional[ValSer]]] = None
    ) -> None:
        self._error_message: str = error_message
        self._val_or_ser: Union[Type[SchemaValidator], Type[SchemaSerializer]] = (
            SchemaValidator if val_or_ser == 'validator' else SchemaSerializer
        )
        self._code: str = code
        self._attempt_rebuild: Optional[Callable[[], Optional[ValSer]]] = attempt_rebuild

    def __getattr__(self, item: str) -> Any:
        __tracebackhide__ = True
        if self._attempt_rebuild:
            val_ser: Optional[ValSer] = self._attempt_rebuild()
            if val_ser is not None:
                return getattr(val_ser, item)
        getattr(self._val_or_ser, item)
        raise PydanticUserError(self._error_message, code=self._code)

    def rebuild(self) -> Optional[ValSer]:
        if self._attempt_rebuild:
            val_ser: Optional[ValSer] = self._attempt_rebuild()
            if val_ser is not None:
                return val_ser
            else:
                raise PydanticUserError(self._error_message, code=self._code)
        return None


def set_type_adapter_mocks(adapter: TypeAdapter[Any]) -> None:
    """Set `core_schema`, `validator` and `serializer` to mock core types on a type adapter instance.

    Args:
        adapter: The type adapter instance to set the mocks on
    """
    type_repr: str = str(adapter._type)
    undefined_type_error_message: str = (
        f'`TypeAdapter[{type_repr}]` is not fully defined; you should define `{type_repr}` and all referenced types, '
        f'then call `.rebuild()` on the instance.'
    )

    def attempt_rebuild_fn(attr_fn: Callable[[Any], Any]) -> Callable[[], Optional[Any]]:
        def handler() -> Optional[Any]:
            if adapter.rebuild(raise_errors=False, _parent_namespace_depth=5) is not False:
                return attr_fn(adapter)
            return None
        return handler

    adapter.core_schema = MockCoreSchema(
        undefined_type_error_message,
        code='class-not-fully-defined',
        attempt_rebuild=attempt_rebuild_fn(lambda ta: ta.core_schema)
    )
    adapter.validator = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='validator',
        attempt_rebuild=attempt_rebuild_fn(lambda ta: ta.validator)
    )
    adapter.serializer = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='serializer',
        attempt_rebuild=attempt_rebuild_fn(lambda ta: ta.serializer)
    )


def set_model_mocks(cls: Type[BaseModel], undefined_name: str = 'all referenced types') -> None:
    """Set `__pydantic_core_schema__`, `__pydantic_validator__` and `__pydantic_serializer__` to mock core types on a model.

    Args:
        cls: The model class to set the mocks on
        undefined_name: Name of the undefined thing, used in error messages
    """
    undefined_type_error_message: str = (
        f'`{cls.__name__}` is not fully defined; you should define {undefined_name}, then call '
        f'`{cls.__name__}.model_rebuild()`.'
    )

    def attempt_rebuild_fn(attr_fn: Callable[[Any], Any]) -> Callable[[], Optional[Any]]:
        def handler() -> Optional[Any]:
            if cls.model_rebuild(raise_errors=False, _parent_namespace_depth=5) is not False:
                return attr_fn(cls)
            return None
        return handler

    cls.__pydantic_core_schema__ = MockCoreSchema(
        undefined_type_error_message,
        code='class-not-fully-defined',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_core_schema__)
    )
    cls.__pydantic_validator__ = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='validator',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_validator__)
    )
    cls.__pydantic_serializer__ = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='serializer',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_serializer__)
    )


def set_dataclass_mocks(cls: Type[PydanticDataclass], undefined_name: str = 'all referenced types') -> None:
    """Set `__pydantic_validator__` and `__pydantic_serializer__` to `MockValSer`s on a dataclass.

    Args:
        cls: The model class to set the mocks on
        undefined_name: Name of the undefined thing, used in error messages
    """
    from ..dataclasses import rebuild_dataclass
    undefined_type_error_message: str = (
        f'`{cls.__name__}` is not fully defined; you should define {undefined_name}, then call '
        f'`pydantic.dataclasses.rebuild_dataclass({cls.__name__})`.'
    )

    def attempt_rebuild_fn(attr_fn: Callable[[Any], Any]) -> Callable[[], Optional[Any]]:
        def handler() -> Optional[Any]:
            if rebuild_dataclass(cls, raise_errors=False, _parent_namespace_depth=5) is not False:
                return attr_fn(cls)
            return None
        return handler

    cls.__pydantic_core_schema__ = MockCoreSchema(
        undefined_type_error_message,
        code='class-not-fully-defined',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_core_schema__)
    )
    cls.__pydantic_validator__ = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='validator',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_validator__)
    )
    cls.__pydantic_serializer__ = MockValSer(
        undefined_type_error_message,
        code='class-not-fully-defined',
        val_or_ser='serializer',
        attempt_rebuild=attempt_rebuild_fn(lambda c: c.__pydantic_serializer__)
    )