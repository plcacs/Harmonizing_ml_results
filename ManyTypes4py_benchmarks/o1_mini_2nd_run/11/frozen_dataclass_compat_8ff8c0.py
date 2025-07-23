"""Utility to  create classes from which frozen or mutable dataclasses can be derived.

This module enabled a non-breaking transition from mutable to frozen dataclasses
derived from EntityDescription and sub classes thereof.
"""
from __future__ import annotations
import dataclasses
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    dataclass_transform,
    Type,
    TypeVar,
    List,
    Tuple,
    Dict,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def _class_fields(cls: type, kw_only: bool) -> List[Tuple[str, Any, dataclasses.Field]]:
    """Return a list of dataclass fields.

    Extracted from dataclasses._process_class.
    """
    cls_annotations: Dict[str, Any] = cls.__dict__.get("__annotations__", {})
    cls_fields: List[dataclasses.Field] = []
    _dataclasses = sys.modules[dataclasses.__name__]
    for name, _type in cls_annotations.items():
        if dataclasses._is_kw_only(_type, _dataclasses) or (
            isinstance(_type, str)
            and dataclasses._is_type(
                _type,
                cls,
                _dataclasses,
                dataclasses.KW_ONLY,
                dataclasses._is_kw_only,
            )
        ):
            kw_only = True
        else:
            field: dataclasses.Field = dataclasses._get_field(cls, name, _type, kw_only)
            cls_fields.append(field)
    return [(field.name, field.type, field) for field in cls_fields]


@dataclass_transform(
    field_specifiers=(dataclasses.field, dataclasses.Field),
    frozen_default=True,
    kw_only_default=True,
)
class FrozenOrThawed(type):
    """Metaclass which which makes classes which behave like a dataclass.

    This allows child classes to be either mutable or frozen dataclasses.
    """

    def _make_dataclass(
        cls: Type[Any],
        name: str,
        bases: tuple[type, ...],
        kw_only: bool,
    ) -> None:
        class_fields: List[Tuple[str, Any, dataclasses.Field]] = _class_fields(cls, kw_only)
        dataclass_bases: List[type[Any]] = [
            getattr(base, "_dataclass", base) for base in bases
        ]
        cls._dataclass = dataclasses.make_dataclass(
            name,
            class_fields,
            bases=tuple(dataclass_bases),
            frozen=True,
        )

    def __new__(
        mcs: Type[type],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        frozen_or_thawed: bool = False,
        **kwargs: Any,
    ) -> type:
        """Pop frozen_or_thawed and store it in the namespace."""
        namespace["_FrozenOrThawed__frozen_or_thawed"] = frozen_or_thawed
        return super().__new__(mcs, name, bases, namespace)

    def __init__(
        cls: Type[Any],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Optionally create a dataclass and store it in cls._dataclass.

        A dataclass will be created if frozen_or_thawed is set, if not we assume the
        class will be a real dataclass, i.e. it's decorated with @dataclass.
        """
        if not namespace.get("_FrozenOrThawed__frozen_or_thawed", False):
            if all(dataclasses.is_dataclass(base) for base in bases):
                return
            annotations: Dict[str, Any] = {}
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

        def __new__(*args: Any, **kwargs: Any) -> Any:
            """Create a new instance.

            The function has no named arguments to avoid name collisions with dataclass
            field names.
            """
            cls_, *_args = args
            if dataclasses.is_dataclass(cls_):
                if TYPE_CHECKING:
                    cls_ = cast(Type[DataclassInstance], cls_)
                return object.__new__(cls_)
            return cls_._dataclass(*_args, **kwargs)

        cls.__init__ = cls._dataclass.__init__  # type: ignore
        cls.__new__ = __new__
