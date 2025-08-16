from __future__ import annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NoReturn, TypeVar, overload
from warnings import warn
from typing_extensions import TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _namespace_utils, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .errors import PydanticUserError
from .fields import Field, FieldInfo, PrivateAttr
if TYPE_CHECKING:
    from ._internal._dataclasses import PydanticDataclass
    from ._internal._namespace_utils import MappingNamespace
__all__: tuple[str] = ('dataclass', 'rebuild_dataclass')
_T = TypeVar('_T')
if sys.version_info >= (3, 10):

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(*, init: bool = False, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: ConfigDict | None = None, validate_on_init: Any = None, kw_only: Any = ..., slots: Any = ...):
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(_cls: Any, *, init: bool = False, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool | None = None, config: ConfigDict | None = None, validate_on_init: Any = None, kw_only: Any = ..., slots: Any = ...):
        ...
else:

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(*, init: bool = False, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool | None = None, config: ConfigDict | None = None, validate_on_init: Any = None):
        ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
    @overload
    def dataclass(_cls: Any, *, init: bool = False, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool | None = None, config: ConfigDict | None = None, validate_on_init: Any = None):
        ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
def dataclass(_cls: Any = None, *, init: bool = False, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool | None = None, config: ConfigDict | None = None, validate_on_init: Any = None, kw_only: bool = False, slots: bool = False) -> Callable[[Any], Any]:
    ...

def rebuild_dataclass(cls: Any, *, force: bool = False, raise_errors: bool = True, _parent_namespace_depth: int = 2, _types_namespace: dict | None = None) -> None | bool:
    ...

def is_pydantic_dataclass(class_: Any) -> bool:
    ...
