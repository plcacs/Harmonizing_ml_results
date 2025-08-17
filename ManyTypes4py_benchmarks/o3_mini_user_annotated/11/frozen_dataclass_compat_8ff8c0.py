from __future__ import annotations

import dataclasses
import sys
from typing import Any, Callable, Dict, List, Tuple, Type, cast, TYPE_CHECKING
from dataclass_transform import dataclass_transform

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _class_fields(cls: Type[Any], kw_only: bool) -> List[Tuple[str, Any, Any]]:
    cls_annotations: Dict[str, Any] = cls.__dict__.get("__annotations__", {})
    cls_fields: List[dataclasses.Field[Any]] = []

    _dataclasses = sys.modules[dataclasses.__name__]
    for name, _type in cls_annotations.items():
        if dataclasses._is_kw_only(_type, _dataclasses) or (
            isinstance(_type, str)
            and dataclasses._is_type(  # type: ignore[attr-defined]
                _type,
                cls,
                _dataclasses,
                dataclasses.KW_ONLY,
                dataclasses._is_kw_only,  # type: ignore[attr-defined]
            )
        ):
            kw_only = True
        else:
            cls_fields.append(dataclasses._get_field(cls, name, _type, kw_only))  # type: ignore[attr-defined]
    return [(field.name, field.type, field) for field in cls_fields]


@dataclass_transform(
    field_specifiers=(dataclasses.field, dataclasses.Field),
    frozen_default=True,
    kw_only_default=True,
)
class FrozenOrThawed(type):
    def _make_dataclass(cls: Type[Any], name: str, bases: Tuple[Type[Any], ...], kw_only: bool) -> None:
        class_fields: List[Tuple[str, Any, Any]] = _class_fields(cls, kw_only)
        dataclass_bases = [getattr(base, "_dataclass", base) for base in bases]
        cls._dataclass = dataclasses.make_dataclass(
            name, class_fields, bases=tuple(dataclass_bases), frozen=True
        )

    def __new__(
        mcs: Type[FrozenOrThawed],
        name: str,
        bases: Tuple[Type[Any], ...],
        namespace: Dict[Any, Any],
        frozen_or_thawed: bool = False,
        **kwargs: Any,
    ) -> Any:
        namespace["_FrozenOrThawed__frozen_or_thawed"] = frozen_or_thawed
        return super().__new__(mcs, name, bases, namespace)

    def __init__(
        cls: FrozenOrThawed,
        name: str,
        bases: Tuple[Type[Any], ...],
        namespace: Dict[Any, Any],
        **kwargs: Any,
    ) -> None:
        if not namespace["_FrozenOrThawed__frozen_or_thawed"]:
            if all(dataclasses.is_dataclass(base) for base in bases):
                return
            annotations: Dict[Any, Any] = {}
            for parent in cls.__mro__[::-1]:
                if parent is object:
                    continue
                annotations |= getattr(parent, "__annotations__", {})
            cls.__annotations__ = annotations
            return

        try:
            cls._make_dataclass(name, bases, False)
        except TypeError:
            cls._make_dataclass(name, bases, True)

        def __new__(*args: Any, **kwargs: Any) -> object:
            cls_inner, *_args = args
            if dataclasses.is_dataclass(cls_inner):
                if TYPE_CHECKING:
                    cls_inner = cast(Type[DataclassInstance], cls_inner)
                return object.__new__(cls_inner)
            return cls._dataclass(*_args, **kwargs)

        cls.__init__ = cls._dataclass.__init__  # type: ignore[misc]
        cls.__new__ = __new__  # type: ignore[assignment]