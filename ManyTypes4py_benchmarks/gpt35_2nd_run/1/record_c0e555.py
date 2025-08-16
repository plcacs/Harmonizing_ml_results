from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, MutableMapping, Set, Tuple, Type

class Record(Model, abstract=True):
    DATE_TYPES: Tuple[Type[datetime]] = (datetime,)
    DECIMAL_TYPES: Tuple[Type[Decimal]] = (Decimal,)
    ALIAS_FIELD_TYPES: Dict[type, Type] = {dict: Dict, tuple: Tuple, list: List, set: Set, frozenset: FrozenSet}
    E_NON_DEFAULT_FOLLOWS_DEFAULT: str = '\nNon-default {cls_name} field {field_name} cannot\nfollow default {fields} {default_names}\n'
    _ReconFun: Callable[..., Any]

    def __init_subclass__(cls, serializer=None, namespace=None, include_metadata=None, isodates=None, abstract=False, allow_blessed_key=None, decimals=None, coerce=None, coercions=None, polymorphic_fields=None, validation=None, date_parser=None, lazy_creation=False, **kwargs) -> None:
        ...

    @classmethod
    def _contribute_to_options(cls, options: ModelOptions) -> None:
        ...

    @classmethod
    def _contribute_methods(cls) -> None:
        ...

    @classmethod
    def _contribute_field_descriptors(cls, target, options, parent=None) -> Dict[str, FieldDescriptorT]:
        ...

    @classmethod
    def from_data(cls, data, *, preferred_type=None) -> Record:
        ...

    def __init__(self, *args, __strict__=True, __faust=None, **kwargs) -> None:
        ...

    @classmethod
    def _BUILD_input_translate_fields(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_init(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_hash(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_eq(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_ne(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_gt(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_ge(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_lt(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_le(cls) -> Callable:
        ...

    @classmethod
    def _BUILD_asdict(cls) -> Callable:
        ...

    def _prepare_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @classmethod
    def _BUILD_asdict_field(cls, name: str, field: FieldDescriptorT) -> str:
        ...

    def _derive(self, *objects, **fields) -> Record:
        ...

    def to_representation(self) -> Dict[str, Any]:
        ...

    def asdict(self) -> Dict[str, Any]:
        ...

    def _humanize(self) -> str:
        ...

    def __json__(self) -> Dict[str, Any]:
        ...

    def __eq__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __le__(self, other) -> bool:
        ...

    def __gt__(self, other) -> bool:
        ...

    def __ge__(self, other) -> bool:
        ...

def _kvrepr(d: Dict[str, Any], *, sep: str = ', ') -> str:
    ...
