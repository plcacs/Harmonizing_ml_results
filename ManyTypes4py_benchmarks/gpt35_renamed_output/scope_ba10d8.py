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


def func_e2maxzg9(level: int, global_dict=None, local_dict=None, resolvers=(),
    target=None) -> Scope:
    ...


def func_cwkimi40(x: str) -> str:
    ...


def func_hk3d7uvv(obj: object) -> str:
    ...


DEFAULT_GLOBALS: dict[str, object] = {'Timestamp': Timestamp, 'datetime': datetime.datetime,
    'True': True, 'False': False, 'list': list, 'tuple': tuple, 'inf': np.
    inf, 'Inf': np.inf}


def func_4hil1l2k(obj: object) -> str:
    ...


class Scope:
    def __init__(self, level: int, global_dict=None, local_dict=None, resolvers=
        (), target=None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def func_2i1h64nq(self) -> bool:
        ...

    def func_5mttxj0l(self, key: str, is_local: bool) -> object:
        ...

    def func_klgbsd9b(self, old_key: str, new_key: str, new_value=None) -> None:
        ...

    def func_3kc0gxeg(self, stack, scopes) -> None:
        ...

    def func_vbgocz63(self, level: int) -> None:
        ...

    def func_0yqvchvo(self, value: object) -> str:
        ...

    @property
    def func_gi98rph9(self) -> int:
        ...

    @property
    def func_uub62sln(self) -> DeepChainMap:
        ...
