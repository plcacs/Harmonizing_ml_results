"""RootModel class and type definitions."""
from __future__ import annotations as _annotations
import typing
from copy import copy, deepcopy
from pydantic_core import PydanticUndefined
from . import PydanticUserError
from ._internal import _model_construction, _repr
from .main import BaseModel, _object_setattr
if typing.TYPE_CHECKING:
    from typing import Any, Dict, Literal, Optional, Set, Tuple, Type, Union
    from typing_extensions import Self, dataclass_transform
    from .fields import Field as PydanticModelField
    from .fields import PrivateAttr as PydanticModelPrivateAttr

    @dataclass_transform(kw_only_default=False, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr))
    class _RootModelMetaclass(_model_construction.ModelMetaclass):
        ...
else:
    _RootModelMetaclass = _model_construction.ModelMetaclass
__all__ = ('RootModel',)
RootModelRootType = typing.TypeVar('RootModelRootType')

class RootModel(BaseModel, typing.Generic[RootModelRootType], metaclass=_RootModelMetaclass):
    """!!! abstract "Usage Documentation"
        [`RootModel` and Custom Root Types](../concepts/models.md#rootmodel-and-custom-root-types)

    A Pydantic `BaseModel` for the root object of the model.

    Attributes:
        root: The root object of the model.
        __pydantic_root_model__: Whether the model is a RootModel.
        __pydantic_private__: Private fields in the model.
        __pydantic_extra__: Extra fields in the model.

    """
    __pydantic_root_model__: bool = True
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
        """Create a new model using the provided root object and update fields set.

        Args:
            root: The root object of the model.
            _fields_set: The set of fields to be updated.

        Returns:
            The new model.

        Raises:
            NotImplemented: If the model is not a subclass of `RootModel`.
        """
        return super().model_construct(root=root, _fields_set=_fields_set)

    def __getstate__(self) -> Dict[str, Any]:
        return {'__dict__': self.__dict__, '__pydantic_fields_set__': self.__pydantic_fields_set__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        _object_setattr(self, '__pydantic_fields_set__', state['__pydantic_fields_set__'])
        _object_setattr(self, '__dict__', state['__dict__'])

    def __copy__(self) -> Self:
        """Returns a shallow copy of the model."""
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', copy(self.__dict__))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        return m

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Self:
        """Returns a deep copy of the model."""
        cls = type(self)
        m = cls.__new__(cls)
        _object_setattr(m, '__dict__', deepcopy(self.__dict__, memo=memo))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        return m
    if typing.TYPE_CHECKING:

        def model_dump(
            self, 
            *, 
            mode: Literal['python', 'json'] = 'python', 
            include: Optional[Union[Set[str], Dict[str, Any]]] = None, 
            exclude: Optional[Union[Set[str], Dict[str, Any]]] = None, 
            context: Optional[Dict[str, Any]] = None, 
            by_alias: bool = False, 
            exclude_unset: bool = False, 
            exclude_defaults: bool = False, 
            exclude_none: bool = False, 
            round_trip: bool = False, 
            warnings: bool = True, 
            serialize_as_any: bool = False
        ) -> Any:
            """This method is included just to get a more accurate return type for type checkers.
            It is included in this `if TYPE_CHECKING:` block since no override is actually necessary.

            See the documentation of `BaseModel.model_dump` for more details about the arguments.

            Generally, this method will have a return type of `RootModelRootType`, assuming that `RootModelRootType` is
            not a `BaseModel` subclass. If `RootModelRootType` is a `BaseModel` subclass, then the return
            type will likely be `dict[str, Any]`, as `model_dump` calls are recursive. The return type could
            even be something different, in the case of a custom serializer.
            Thus, `Any` is used here to catch all of these cases.
            """
            ...

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RootModel):
            return NotImplemented
        return self.__pydantic_fields__['root'].annotation == other.__pydantic_fields__['root'].annotation and super().__eq__(other)

    def __repr_args__(self) -> Tuple[Tuple[str, Any], ...]:
        yield ('root', self.root)
