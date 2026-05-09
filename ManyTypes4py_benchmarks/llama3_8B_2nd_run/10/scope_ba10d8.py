from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar

class DeepChainMap(ChainMap[_KT, _VT]):
    # ...

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class Scope:
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
    global_dict : dict or None, optional, default None
    local_dict : dict or Scope or None, optional, default None
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
    scope : DeepChainMap
    target : object
    temps : dict
    """
    __slots__ = ['level', 'resolvers', 'scope', 'target', 'temps']

    def __init__(self, 
                level: int, 
                global_dict: dict or None = None, 
                local_dict: dict or Scope or None = None, 
                resolvers: list-like or None = (), 
                target: object = None
                ) -> None:
        # ...

    def __repr__(self) -> str:
        # ...

    @property
    def has_resolvers(self) -> bool:
        # ...

    def resolve(self, key: str, is_local: bool) -> object:
        # ...

    def swapkey(self, old_key: str, new_key: str, new_value: object = None) -> None:
        # ...

    def _get_vars(self, stack: list, scopes: sequence) -> None:
        # ...

    def _update(self, level: int) -> None:
        # ...

    def add_tmp(self, value: object) -> str:
        # ...

    @property
    def ntemps(self) -> int:
        # ...

    @property
    def full_scope(self) -> DeepChainMap:
        # ...
