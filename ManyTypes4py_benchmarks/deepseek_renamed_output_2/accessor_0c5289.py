from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import final
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Callable as abcCallable
    from pandas._typing import TypeT
    from pandas import Index, Series, DataFrame
    from pandas.core.generic import NDFrame

T = TypeVar("T")
AccessorT = TypeVar("AccessorT", bound="Accessor")
DelegateT = TypeVar("DelegateT", bound="PandasDelegate")
KlassT = TypeVar("KlassT")


class DirNamesMixin:
    _accessors: ClassVar[Set[str]] = set()
    _hidden_attrs: ClassVar[FrozenSet[str]] = frozenset()

    @final
    def _dir_deletions(self) -> Set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> Set[str]:
        """
        Add additional __dir__ for this object.
        """
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> List[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """
        rv = set(super().__dir__())
        rv = rv - self._dir_deletions() | self._dir_additions()
        return sorted(rv)


class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    def _delegate_property_get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"You cannot access the property {name}")

    def _delegate_property_set(self, name: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"The property {name} cannot be set")

    def _delegate_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"You cannot call method {name}")

    @classmethod
    def _add_delegate_accessors(
        cls: Type[DelegateT],
        delegate: Type[Any],
        accessors: Sequence[str],
        typ: str,
        overwrite: bool = False,
        accessor_mapping: Callable[[str], str] = lambda x: x,
        raise_on_missing: bool = True,
    ) -> None:
        """
        Add accessors to cls from the delegate class.
        """
        def create_property(name: str) -> property:
            def _getter(self: DelegateT) -> Any:
                return self._delegate_property_get(name)

            def _setter(self: DelegateT, new_values: Any) -> Any:
                return self._delegate_property_set(name, new_values)

            _getter.__name__ = name
            _setter.__name__ = name
            return property(
                fget=_getter,
                fset=_setter,
                doc=getattr(delegate, accessor_mapping(name)).__doc__,
            )

        def create_method(name: str) -> Callable[..., Any]:
            method = getattr(delegate, accessor_mapping(name))

            @functools.wraps(method)
            def f(self: DelegateT, *args: Any, **kwargs: Any) -> Any:
                return self._delegate_method(name, *args, **kwargs)
            return f

        for name in accessors:
            if not raise_on_missing and getattr(delegate, accessor_mapping(name), None) is None:
                continue
            if typ == "property":
                f = create_property(name)
            else:
                f = create_method(name)
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)


def delegate_names(
    delegate: Type[Any],
    accessors: Sequence[str],
    typ: str,
    overwrite: bool = False,
    accessor_mapping: Callable[[str], str] = lambda x: x,
    raise_on_missing: bool = True,
) -> Callable[[Type[DelegateT]], Type[DelegateT]]:
    """
    Add delegated names to a class using a class decorator.
    """
    def wrapper(cls: Type[DelegateT]) -> Type[DelegateT]:
        cls._add_delegate_accessors(
            delegate,
            accessors,
            typ,
            overwrite=overwrite,
            accessor_mapping=accessor_mapping,
            raise_on_missing=raise_on_missing,
        )
        return cls
    return wrapper


class Accessor:
    """
    Custom property-like object.
    """
    def __init__(self, name: str, accessor: Type[AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: Optional[NDFrame], cls: Type[Any]) -> Union[Type[AccessorT], AccessorT]:
        if obj is None:
            return self._accessor
        return self._accessor(obj)


CachedAccessor = Accessor


@doc(klass="", examples="", others="")
def _register_accessor(name: str, cls: Type[KlassT]) -> Callable[[Type[AccessorT]], Type[AccessorT]]:
    """
    Register a custom accessor on {klass} objects.
    """
    def decorator(accessor: Type[AccessorT]) -> Type[AccessorT]:
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type {cls!r} is overriding a preexisting attribute with the same name.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)  # type: ignore
        return accessor
    return decorator


_register_df_examples = """
An accessor that only accepts integers could
have a class defined like this:
"""


@doc(_register_accessor, klass="DataFrame", examples=_register_df_examples)
def register_dataframe_accessor(name: str) -> Callable[[Type[AccessorT]], Type[AccessorT]]:
    from pandas import DataFrame
    return _register_accessor(name, DataFrame)


_register_series_examples = """
An accessor that only accepts integers could
have a class defined like this:
"""


@doc(_register_accessor, klass="Series", examples=_register_series_examples)
def register_series_accessor(name: str) -> Callable[[Type[AccessorT]], Type[AccessorT]]:
    from pandas import Series
    return _register_accessor(name, Series)


_register_index_examples = """
An accessor that only accepts integers could
have a class defined like this:
"""


@doc(_register_accessor, klass="Index", examples=_register_index_examples)
def register_index_accessor(name: str) -> Callable[[Type[AccessorT]], Type[AccessorT]]:
    from pandas import Index
    return _register_accessor(name, Index)
