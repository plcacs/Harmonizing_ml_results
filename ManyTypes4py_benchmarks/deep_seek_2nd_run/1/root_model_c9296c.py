"""RootModel class and type definitions."""
from __future__ import annotations as _annotations
import typing
from copy import copy, deepcopy
from pydantic_core import PydanticUndefined
from . import PydanticUserError
from ._internal import _model_construction, _repr
from .main import BaseModel, _object_setattr
from typing import Any, Dict, Optional, Set, TypeVar, Union
from typing_extensions import Literal, Self, dataclass_transform

if typing.TYPE_CHECKING:
    from .fields import Field as PydanticModelField
    from .fields import PrivateAttr as PydanticModelPrivateAttr

    @dataclass_transform(kw_only_default=False, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr))
    class _RootModelMetaclass(_model_construction.ModelMetaclass):
        ...
else:
    _RootModelMetaclass = _model_construction.ModelMetaclass

__all__ = ('RootModel',)

RootModelRootType = TypeVar('RootModelRootType')

class RootModel(BaseModel, typing.Generic[RootModelRootType], metaclass=_RootModelMetaclass):
    __pydantic_root_model__: Literal[True] = True
    __pydantic_private__: Optional[Dict[str, Any]] = None
    __pydantic_extra__: Optional[Dict[str, Any]] = None
    root: RootModelRootType

    def __init_subclass__(cls, **kwargs: Any) -> None:
        extra = cls.model_config.get('extra')
        if extra is not None:
            raise PydanticUserError("`RootModel` does not support setting `model_config['extra']`", code='root-model-extra')
        super().__init_subclass__(**kwargs)

    def __init__(self, /, root: Any = PydanticUndefined, **data: Any) -> None:
        __tracebackhide__ = True
        if data:
            if root is not PydanticUndefined:
                raise ValueError('"RootModel.__init__" accepts either a single positional argument or arbitrary keyword arguments')
            root = data
        self.__pydantic_validator__.validate_python(root, self_instance=self)
    __init__.__pydantic_base_init__ = True

    @classmethod
    def model_construct(cls, root: RootModelRootType, _fields_set: Optional[Set[str]] = None) -> Self:
        return super().model_construct(root=root, _fields_set=_fields_set)

    def __getstate__(self) -> Dict[str, Any]:
        return {'__dict__': self.__dict__, '__pydantic_fields_set__': self.__pydantic_fields_set__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        _object_setattr(self, '__pydantic_fields_set__', state['__pydantic_fields_set__'])
        _object_setattr(self, '__dict__', state['__dict__'])

    def __copy__(self) -> Self:
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', copy(self.__dict__))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        return m

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Self:
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', deepcopy(self.__dict__, memo=memo))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        return m

    if typing.TYPE_CHECKING:
        def model_dump(
            self,
            *,
            mode: Literal['json', 'python'] = 'python',
            include: Optional[Any] = None,
            exclude: Optional[Any] = None,
            context: Optional[Dict[str, Any]] = None,
            by_alias: bool = False,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            round_trip: bool = False,
            warnings: bool = True,
            serialize_as_any: bool = False
        ) -> Any:
            ...

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RootModel):
            return NotImplemented
        return self.__pydantic_fields__['root'].annotation == other.__pydantic_fields__['root'].annotation and super().__eq__(other)

    def __repr_args__(self) -> typing.Iterator[tuple[str, RootModelRootType]]:
        yield ('root', self.root)
