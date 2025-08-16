from __future__ import annotations
import functools
from typing import TYPE_CHECKING, final, Set, Callable, Sequence
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
if TYPE_CHECKING:
    from pandas._typing import TypeT
    from pandas import Index
    from pandas.core.generic import NDFrame

class DirNamesMixin:
    _accessors: Set[str] = set()
    _hidden_attrs: frozenset = frozenset()

    @final
    def _dir_deletions(self) -> Set[str]:
        ...

    def _dir_additions(self) -> Set[str]:
        ...

    def __dir__(self) -> list[str]:
        ...

class PandasDelegate:
    ...

    @classmethod
    def _add_delegate_accessors(cls, delegate, accessors, typ, overwrite=False, accessor_mapping: Callable = lambda x: x, raise_on_missing=True):
        ...

def delegate_names(delegate, accessors, typ, overwrite=False, accessor_mapping: Callable = lambda x: x, raise_on_missing=True) -> Callable:
    ...

class Accessor:
    ...

@doc(klass='', examples='', others='')
def _register_accessor(name, cls):
    ...

@doc(_register_accessor, klass='DataFrame', examples=_register_df_examples)
def register_dataframe_accessor(name) -> Callable:
    ...

@doc(_register_accessor, klass='Series', examples=_register_series_examples)
def register_series_accessor(name) -> Callable:
    ...

@doc(_register_accessor, klass='Index', examples=_register_index_examples)
def register_index_accessor(name) -> Callable:
    ...
