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
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class DeepChainMap(ChainMap[_KT, _VT]):
    def __setitem__(self, key: _KT, value: _VT) -> None:
        ...

    def __delitem__(self, key: _KT) -> None:
        ...

def ensure_scope(level: int, global_dict=None, local_dict=None, resolvers=(), target=None) -> Scope:
    ...

def _replacer(x: str) -> str:
    ...

def _raw_hex_id(obj: object) -> str:
    ...

DEFAULT_GLOBALS: dict = {'Timestamp': Timestamp, 'datetime': datetime.datetime, 'True': True, 'False': False, 'list': list, 'tuple': tuple, 'inf': np.inf, 'Inf': np.inf}

def _get_pretty_string(obj: object) -> str:
    ...

class Scope:
    def __init__(self, level: int, global_dict=None, local_dict=None, resolvers=(), target=None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def has_resolvers(self) -> bool:
        ...

    def resolve(self, key: str, is_local: bool) -> object:
        ...

    def swapkey(self, old_key: str, new_key: str, new_value=None) -> None:
        ...

    def _get_vars(self, stack, scopes) -> None:
        ...

    def _update(self, level: int) -> None:
        ...

    def add_tmp(self, value: object) -> str:
        ...

    @property
    def ntemps(self) -> int:
        ...

    @property
    def full_scope(self) -> DeepChainMap:
        ...
