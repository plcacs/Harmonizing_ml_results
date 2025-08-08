from __future__ import annotations
import dataclasses
import sys
from typing import TYPE_CHECKING, Any, cast, dataclass_transform, Tuple

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

def _class_fields(cls: type, kw_only: bool) -> List[Tuple[str, Any, Any]]:
    cls_annotations = cls.__dict__.get('__annotations__', {})
    cls_fields = []
    _dataclasses = sys.modules[dataclasses.__name__]
    for name, _type in cls_annotations.items():
        if dataclasses._is_kw_only(type, _dataclasses) or (isinstance(_type, str) and dataclasses._is_type(_type, cls, _dataclasses, dataclasses.KW_ONLY, dataclasses._is_kw_only)):
            kw_only = True
        else:
            cls_fields.append(dataclasses._get_field(cls, name, _type, kw_only))
    return [(field.name, field.type, field) for field in cls_fields]

@dataclass_transform(field_specifiers=(dataclasses.field, dataclasses.Field), frozen_default=True, kw_only_default=True)
class FrozenOrThawed(type):
    def _make_dataclass(cls: type, name: str, bases: Tuple[type, ...], kw_only: bool) -> None:
        class_fields = _class_fields(cls, kw_only)
        dataclass_bases = [getattr(base, '_dataclass', base) for base in bases]
        cls._dataclass = dataclasses.make_dataclass(name, class_fields, bases=tuple(dataclass_bases), frozen=True)

    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: dict, frozen_or_thawed: bool = False, **kwargs) -> FrozenOrThawed:
        namespace['_FrozenOrThawed__frozen_or_thawed'] = frozen_or_thawed
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name: str, bases: Tuple[type, ...], namespace: dict, **kwargs) -> None:
        if not namespace['_FrozenOrThawed__frozen_or_thawed']:
            if all((dataclasses.is_dataclass(base) for base in bases)):
                return
            annotations = {}
            for parent in cls.__mro__[::-1]:
                if parent is object:
                    continue
                annotations |= parent.__annotations__
            cls.__annotations__ = annotations
            return
        try:
            cls._make_dataclass(name, bases, False)
        except TypeError:
            cls._make_dataclass(name, bases, True)

        def __new__(*args, **kwargs):
            cls, *_args = args
            if dataclasses.is_dataclass(cls):
                if TYPE_CHECKING:
                    cls = cast(type[DataclassInstance], cls)
                return object.__new__(cls)
            return cls._dataclass(*_args, **kwargs)
        cls.__init__ = cls._dataclass.__init__
        cls.__new__ = __new__
