from __future__ import annotations
import typing
from pydantic_core import PydanticUndefined
from . import PydanticUserError
from ._internal import _model_construction
from .main import BaseModel, _object_setattr

if typing.TYPE_CHECKING:
    from typing import Any, Literal
    from typing_extensions import Self, dataclass_transform
    from .fields import Field as PydanticModelField
    from .fields import PrivateAttr as PydanticModelPrivateAttr

    @dataclass_transform(kw_only_default=False, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr))
    class _RootModelMetaclass(_model_construction.ModelMetaclass):
        ...

else:
    _RootModelMetaclass = _model_construction.ModelMetaclass

__all__: tuple[str] = ('RootModel',)
RootModelRootType = typing.TypeVar('RootModelRootType')

class RootModel(BaseModel, typing.Generic[RootModelRootType], metaclass=_RootModelMetaclass):
    __pydantic_root_model__: bool = True
    __pydantic_private__: typing.Optional[typing.Any] = None
    __pydantic_extra__: typing.Optional[typing.Any] = None

    def __init_subclass__(cls, **kwargs) -> None:
        ...

    def __init__(self, /, root: typing.Union[PydanticUndefined, dict] = PydanticUndefined, **data: typing.Any) -> None:
        ...

    @classmethod
    def model_construct(cls, root: RootModelRootType, _fields_set: typing.Optional[typing.Any] = None) -> RootModel:
        ...

    def __getstate__(self) -> dict:
        ...

    def __setstate__(self, state: dict) -> None:
        ...

    def __copy__(self) -> RootModel:
        ...

    def __deepcopy__(self, memo: typing.Optional[dict] = None) -> RootModel:
        ...

    if typing.TYPE_CHECKING:

        def model_dump(self, *, mode: str = 'python', include: typing.Optional[typing.Any] = None, exclude: typing.Optional[typing.Any] = None, context: typing.Optional[typing.Any] = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, serialize_as_any: bool = False) -> typing.Any:
            ...

    def __eq__(self, other: typing.Any) -> typing.Union[NotImplemented, bool]:
        ...

    def __repr_args__(self) -> typing.Generator[tuple[str, typing.Any], None, None]:
        ...
