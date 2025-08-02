from __future__ import annotations
import functools
from typing import TYPE_CHECKING, final, Set, FrozenSet, List, Callable, TypeVar, Type, Any, cast, Sequence, Optional
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import TypeT
    from pandas import Index
    from pandas.core.generic import NDFrame

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
        rv = set(super().__dir__())
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
        cls: Type[Any],
        delegate: Type[Any],
        accessors: List[str],
        typ: str,
        overwrite: bool = False,
        accessor_mapping: Callable[[str], str] = lambda x: x,
        raise_on_missing: bool = True
    ) -> None:
        """
        Add accessors to cls from the delegate class.

        Parameters
        ----------
        cls
            Class to add the methods/properties to.
        delegate
            Class to get methods/properties and doc-strings.
        accessors : list of str
            List of accessors to add.
        typ : {'property', 'method'}
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable, default lambda x: x
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
            return property(fget=_getter, fset=_setter, doc=getattr(delegate, accessor_mapping(name)).__doc__)

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
                f = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)

def delegate_names(
    delegate: Type[Any],
    accessors: Sequence[str],
    typ: str,
    overwrite: bool = False,
    accessor_mapping: Callable[[str], str] = lambda x: x,
    raise_on_missing: bool = True
) -> Callable[[Type[TypeT]], Type[TypeT]]:
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {'property', 'method'}
    overwrite : bool, default False
       Overwrite the method/property in the target class if it exists.
    accessor_mapping: Callable, default lambda x: x
        Callable to map the delegate's function to the cls' function.
    raise_on_missing: bool, default True
        Raise if an accessor does not exist on delegate.
        False skips the missing accessor.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """

    def add_delegate_accessors(cls: Type[TypeT]) -> Type[TypeT]:
        cls._add_delegate_accessors(delegate, accessors, typ, overwrite=overwrite, accessor_mapping=accessor_mapping, raise_on_missing=raise_on_missing)
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
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor: Type[Any]) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj: Optional[Any], cls: Type[Any]) -> Any:
        if obj is None:
            return self._accessor
        return self._accessor(obj)

CachedAccessor = Accessor

@doc(klass='', examples='', others='')
def _register_accessor(name: str, cls: Type[Any]) -> Callable[[Type[TypeT]], Type[TypeT]]:
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

    def decorator(accessor: Type[TypeT]) -> Type[TypeT]:
        if hasattr(cls, name):
            warnings.warn(f'registration of accessor {accessor!r} under name {name!r} for type {cls!r} is overriding a preexisting attribute with the same name.', UserWarning, stacklevel=find_stack_level())
        setattr(cls, name, Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor
    return decorator

_register_df_examples = '\nAn accessor that only accepts integers could\nhave a class defined like this:\n\n>>> @pd.api.extensions.register_dataframe_accessor("int_accessor")\n... class IntAccessor:\n...     def __init__(self, pandas_obj):\n...         if not all(pandas_obj[col].dtype == \'int64\' for col in pandas_obj.columns):\n...             raise AttributeError("All columns must contain integer values only")\n...         self._obj = pandas_obj\n...\n...     def sum(self):\n...         return self._obj.sum()\n...\n>>> df = pd.DataFrame([[1, 2], [\'x\', \'y\']])\n>>> df.int_accessor\nTraceback (most recent call last):\n...\nAttributeError: All columns must contain integer values only.\n>>> df = pd.DataFrame([[1, 2], [3, 4]])\n>>> df.int_accessor.sum()\n0    4\n1    6\ndtype: int64'

@doc(_register_accessor, klass='DataFrame', examples=_register_df_examples)
def register_dataframe_accessor(name: str) -> Callable[[Type[TypeT]], Type[TypeT]]:
    from pandas import DataFrame
    return _register_accessor(name, DataFrame)

_register_series_examples = '\nAn accessor that only accepts integers could\nhave a class defined like this:\n\n>>> @pd.api.extensions.register_series_accessor("int_accessor")\n... class IntAccessor:\n...     def __init__(self, pandas_obj):\n...         if not pandas_obj.dtype == \'int64\':\n...             raise AttributeError("The series must contain integer data only")\n...         self._obj = pandas_obj\n...\n...     def sum(self):\n...         return self._obj.sum()\n...\n>>> df = pd.Series([1, 2, \'x\'])\n>>> df.int_accessor\nTraceback (most recent call last):\n...\nAttributeError: The series must contain integer data only.\n>>> df = pd.Series([1, 2, 3])\n>>> df.int_accessor.sum()\n6'

@doc(_register_accessor, klass='Series', examples=_register_series_examples)
def register_series_accessor(name: str) -> Callable[[Type[TypeT]], Type[TypeT]]:
    from pandas import Series
    return _register_accessor(name, Series)

_register_index_examples = '\nAn accessor that only accepts integers could\nhave a class defined like this:\n\n>>> @pd.api.extensions.register_index_accessor("int_accessor")\n... class IntAccessor:\n...     def __init__(self, pandas_obj):\n...         if not all(isinstance(x, int) for x in pandas_obj):\n...             raise AttributeError("The index must only be an integer value")\n...         self._obj = pandas_obj\n...\n...     def even(self):\n...         return [x for x in self._obj if x % 2 == 0]\n>>> df = pd.DataFrame.from_dict(\n...     {"row1": {"1": 1, "2": "a"}, "row2": {"1": 2, "2": "b"}}, orient="index"\n... )\n>>> df.index.int_accessor\nTraceback (most recent call last):\n...\nAttributeError: The index must only be an integer value.\n>>> df = pd.DataFrame(\n...     {"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[1, 2, 5, 8]\n... )\n>>> df.index.int_accessor.even()\n[2, 8]'

@doc(_register_accessor, klass='Index', examples=_register_index_examples)
def register_index_accessor(name: str) -> Callable[[Type[TypeT]], Type[TypeT]]:
    from pandas import Index
    return _register_accessor(name, Index)
