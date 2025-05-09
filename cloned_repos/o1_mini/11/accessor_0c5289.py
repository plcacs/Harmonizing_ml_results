"""
accessor.py contains base classes for implementing accessor properties
that can be mixed into or pinned onto other pandas classes.
"""
from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    final,
    Any,
    Callable,
    TypeVar,
    Type,
    List,
    Set,
    FrozenSet,
    Dict,
    Union,
    Optional,
    Sequence,
    Literal,
)
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import TypeT
    from pandas import Index
    from pandas.core.generic import NDFrame

K = TypeVar('K', bound='Accessor')


class DirNamesMixin:
    _accessors: Set[str] = set()
    _hidden_attrs: FrozenSet[str] = frozenset()

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
        rv: Set[str] = set(super().__dir__())
        rv = rv - self._dir_deletions() | self._dir_additions()
        return sorted(rv)


class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    def _delegate_property_get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f'You cannot access the property {name}')

    def _delegate_property_set(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f'The property {name} cannot be set')

    def _delegate_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f'You cannot call method {name}')

    @classmethod
    def _add_delegate_accessors(
        cls: Type[K],
        delegate: Type[Any],
        accessors: Sequence[str],
        typ: Literal['property', 'method'],
        overwrite: bool = False,
        accessor_mapping: Callable[[str], str] = lambda x: x,
        raise_on_missing: bool = True,
    ) -> None:
        """
        Add accessors to cls from the delegate class.

        Parameters
        ----------
        cls : Type[K]
            Class to add the methods/properties to.
        delegate : Type[Any]
            Class to get methods/properties and doc-strings.
        accessors : Sequence[str]
            List of accessors to add.
        typ : {'property', 'method'}
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable[[str], str], default lambda x: x
            Callable to map the delegate's function to the cls' function.
        raise_on_missing: bool, default True
            Raise if an accessor does not exist on delegate.
            False skips the missing accessor.
        """

        def _create_delegator_property(name: str) -> property:
            def _getter(self: Any) -> Any:
                return self._delegate_property_get(name)

            def _setter(self: Any, new_values: Any) -> None:
                return self._delegate_property_set(name, new_values)

            _getter.__name__ = name
            _setter.__name__ = name
            return property(
                fget=_getter,
                fset=_setter,
                doc=getattr(delegate, accessor_mapping(name)).__doc__,
            )

        def _create_delegator_method(name: str) -> Callable[..., Any]:
            method = getattr(delegate, accessor_mapping(name))

            @functools.wraps(method)
            def f(self: Any, *args: Any, **kwargs: Any) -> Any:
                return self._delegate_method(name, *args, **kwargs)

            return f

        for name in accessors:
            if not raise_on_missing and getattr(delegate, accessor_mapping(name), None) is None:
                continue
            if typ == 'property':
                f: Union[property, Callable[..., Any]] = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)


def delegate_names(
    delegate: Type[Any],
    accessors: Sequence[str],
    typ: Literal['property', 'method'],
    overwrite: bool = False,
    accessor_mapping: Callable[[str], str] = lambda x: x,
    raise_on_missing: bool = True,
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : Type[Any]
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {'property', 'method'}
    overwrite : bool, default False
        Overwrite the method/property in the target class if it exists.
    accessor_mapping: Callable[[str], str], default lambda x: x
        Callable to map the delegate's function to the cls' function.
    raise_on_missing: bool, default True
        Raise if an accessor does not exist on delegate.
        False skips the missing accessor.

    Returns
    -------
    Callable[[Type[Any]], Type[Any]]
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """

    def add_delegate_accessors(cls: Type[Any]) -> Type[Any]:
        cls._add_delegate_accessors(
            delegate,
            accessors,
            typ,
            overwrite=overwrite,
            accessor_mapping=accessor_mapping,
            raise_on_missing=raise_on_missing,
        )
        return cls

    return add_delegate_accessors


class Accessor:
    """
    Custom property-like object.

    A descriptor for accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : Type[Any]
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor: Type[Any]) -> None:
        self._name: str = name
        self._accessor: Type[Any] = accessor

    def __get__(self, obj: Optional[Any], cls: Optional[Type[Any]] = None) -> Any:
        if obj is None:
            return self._accessor
        return self._accessor(obj)


CachedAccessor = Accessor


@doc(klass='', examples='', others='')
def _register_accessor(name: str, cls: Type[Any]) -> Callable[[Type[Any]], Type[Any]]:
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.
    
    Returns
    -------
    callable
        A class decorator.
    
    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.
    
    Notes
    -----
    This function allows you to register a custom-defined accessor class for {klass}.
    The requirements for the accessor class are as follows:
    
    * Must contain an init method that:
    
      * accepts a single {klass} object
    
      * raises an AttributeError if the {klass} object does not have correctly
        matching inputs for the accessor
    
    * Must contain a method for each access pattern.
    
      * The methods should be able to take any argument signature.
    
      * Accessible using the @property decorator if no additional arguments are
        needed.
    
    Examples
    --------
    {examples}
    """

    def decorator(accessor: Type[Any]) -> Type[Any]:
        if hasattr(cls, name):
            warnings.warn(
                f'registration of accessor {accessor!r} under name {name!r} for type {cls!r} is overriding a preexisting attribute with the same name.',
                UserWarning,
                stacklevel=find_stack_level(),
            )
        setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator


_register_df_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_dataframe_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not all(pandas_obj[col].dtype == 'int64' for col in pandas_obj.columns):
...             raise AttributeError("All columns must contain integer values only")
...         self._obj = pandas_obj
...
...     def sum(self):
...         return self._obj.sum()
...
>>> df = pd.DataFrame([[1, 2], ['x', 'y']])
>>> df.int_accessor
Traceback (most recent call last):
...
AttributeError: All columns must contain integer values only.
>>> df = pd.DataFrame([[1, 2], [3, 4]])
>>> df.int_accessor.sum()
0    4
1    6
dtype: int64
"""


@doc(_register_accessor, klass='DataFrame', examples=_register_df_examples)
def register_dataframe_accessor(name: str) -> Callable[[Type[Any]], Type[Any]]:
    from pandas import DataFrame
    return _register_accessor(name, DataFrame)


_register_series_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_series_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not pandas_obj.dtype == 'int64':
...             raise AttributeError("The series must contain integer data only")
...         self._obj = pandas_obj
...
...     def sum(self):
...         return self._obj.sum()
...
>>> df = pd.Series([1, 2, 'x'])
>>> df.int_accessor
Traceback (most recent call last):
...
AttributeError: The series must contain integer data only.
>>> df = pd.Series([1, 2, 3])
>>> df.int_accessor.sum()
6
"""


@doc(_register_accessor, klass='Series', examples=_register_series_examples)
def register_series_accessor(name: str) -> Callable[[Type[Any]], Type[Any]]:
    from pandas import Series
    return _register_accessor(name, Series)


_register_index_examples = """
An accessor that only accepts integers could
have a class defined like this:

>>> @pd.api.extensions.register_index_accessor("int_accessor")
... class IntAccessor:
...     def __init__(self, pandas_obj):
...         if not all(isinstance(x, int) for x in pandas_obj):
...             raise AttributeError("The index must only be an integer value")
...         self._obj = pandas_obj
...
...     def even(self):
...         return [x for x in self._obj if x % 2 == 0]
>>> df = pd.DataFrame.from_dict(
...     {"row1": {"1": 1, "2": "a"}, "row2": {"1": 2, "2": "b"}}, orient="index"
... )
>>> df.index.int_accessor
Traceback (most recent call last):
...
AttributeError: The index must only be an integer value.
>>> df = pd.DataFrame(
...     {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[1, 2, 5, 8]
... )
>>> df.index.int_accessor.even()
[2, 8]
"""


@doc(_register_accessor, klass='Index', examples=_register_index_examples)
def register_index_accessor(name: str) -> Callable[[Type[Any]], Type[Any]]:
    from pandas import Index
    return _register_accessor(name, Index)
