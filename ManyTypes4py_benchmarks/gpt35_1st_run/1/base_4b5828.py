from faust.models import Record
from faust.types.models import ModelT
from faust.serializers.codecs import CodecArg, dumps, loads
from typing import Any, Callable, ClassVar, Iterable, List, MutableMapping, Optional, Tuple, Type

registry: MutableMapping[str, Type[ModelT]] = {}

def maybe_model(arg: Any) -> Any:
    ...

class Model(ModelT):
    __is_abstract__: ClassVar[bool] = True
    __validation_errors__: Optional[List[ValidationError]] = None
    _pending_finalizers: Optional[List[Callable[[], None]]] = None
    _blessed_key: ClassVar[str] = '__faust'

    @classmethod
    def _maybe_namespace(cls, data: Any, *, preferred_type: Optional[Type[ModelT]] = None, fast_types: Tuple[Type, Type] = (bytes, str), isinstance: Callable = isinstance) -> Optional[Type[ModelT]]:
        ...

    @classmethod
    def _maybe_reconstruct(cls, data: Any) -> Any:
        ...

    @classmethod
    def _from_data_field(cls, data: Any) -> Any:
        ...

    @classmethod
    def loads(cls, s: bytes, *, default_serializer: Optional[CodecArg] = None, serializer: Optional[CodecArg] = None) -> Any:
        ...

    def __init_subclass__(self, serializer: Optional[CodecArg] = None, namespace: Optional[str] = None, include_metadata: Optional[bool] = None, isodates: Optional[bool] = None, abstract: bool = False, allow_blessed_key: Optional[bool] = None, decimals: Optional[int] = None, coerce: Optional[bool] = None, coercions: Optional[CoercionMapping] = None, polymorphic_fields: Optional[bool] = None, validation: Optional[bool] = None, date_parser: Optional[Callable] = None, lazy_creation: bool = False, **kwargs: Any) -> None:
        ...

    @classmethod
    def make_final(cls) -> None:
        ...

    @classmethod
    def _init_subclass(cls, serializer: Optional[CodecArg] = None, namespace: Optional[str] = None, include_metadata: Optional[bool] = None, isodates: Optional[bool] = None, abstract: bool = False, allow_blessed_key: Optional[bool] = None, decimals: Optional[int] = None, coerce: Optional[bool] = None, coercions: Optional[CoercionMapping] = None, polymorphic_fields: Optional[bool] = None, validation: Optional[bool] = None, date_parser: Optional[Callable] = None) -> None:
        ...

    def __abstract_init__(self) -> None:
        ...

    @classmethod
    def _contribute_to_options(cls, options: ModelOptions) -> None:
        ...

    @classmethod
    def _contribute_methods(cls) -> None:
        ...

    @classmethod
    def _contribute_field_descriptors(cls, target: Type, options: ModelOptions, parent: Optional[Type] = None) -> FieldMap:
        ...

    @classmethod
    def _BUILD_init(cls) -> Callable[[], None]:
        ...

    @classmethod
    def _BUILD_hash(cls) -> Callable[[], None]:
        ...

    @classmethod
    def _BUILD_eq(cls) -> Callable[[], None]:
        ...

    def to_representation(self) -> Any:
        ...

    def _humanize(self) -> str:
        ...

    def is_valid(self) -> bool:
        ...

    def validate(self) -> List[ValidationError]:
        ...

    def validate_or_raise(self) -> None:
        ...

    def _itervalidate(self) -> Iterable[ValidationError]:
        ...

    @property
    def validation_errors(self) -> List[ValidationError]:
        ...

    def derive(self, *objects: Any, **fields: Any) -> Any:
        ...

    def _derive(self, *objects: Any, **fields: Any) -> Any:
        ...

    def dumps(self, *, serializer: Optional[CodecArg] = None) -> bytes:
        ...

    def __repr__(self) -> str:
        ...
