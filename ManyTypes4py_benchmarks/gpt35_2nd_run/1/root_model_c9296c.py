from __future__ import annotations
import typing
from pydantic_core import PydanticUndefined, PydanticUserError
from .main import BaseModel
from .fields import Field as PydanticModelField, PrivateAttr as PydanticModelPrivateAttr

RootModelRootType = typing.TypeVar('RootModelRootType')

class RootModel(BaseModel, typing.Generic[RootModelRootType]):
    root: RootModelRootType
    __pydantic_root_model__: bool
    __pydantic_private__: typing.Any
    __pydantic_extra__: typing.Any

    def __init_subclass__(cls, **kwargs) -> None:
        ...

    def __init__(self, /, root: RootModelRootType = PydanticUndefined, **data) -> None:
        ...

    @classmethod
    def model_construct(cls, root: RootModelRootType, _fields_set: typing.Optional[typing.Set[str]] = None) -> RootModel:
        ...

    def __getstate__(self) -> dict:
        ...

    def __setstate__(self, state: dict) -> None:
        ...

    def __copy__(self) -> RootModel:
        ...

    def __deepcopy__(self, memo: typing.Optional[dict] = None) -> RootModel:
        ...

    def model_dump(self, *, mode: str = 'python', include: typing.Optional[typing.Set[str]] = None, exclude: typing.Optional[typing.Set[str]] = None, context: typing.Any = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, serialize_as_any: bool = False) -> typing.Any:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __repr_args__(self) -> typing.Generator[tuple[str, RootModelRootType], None, None]:
        ...
