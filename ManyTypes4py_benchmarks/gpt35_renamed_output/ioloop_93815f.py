import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable

class _Selectable:
    def func_sqhj4qgi(self) -> None:
        pass

    def func_yd9g72ty(self) -> None:
        pass

_T = TypeVar('_T')
_S = TypeVar('_S', bound=_Selectable)

class IOLoop(Configurable):
    NONE: int = 0
    READ: int = 1
    WRITE: int = 4
    ERROR: int = 24
    _ioloop_for_asyncio: dict = dict()
    _pending_tasks: set = set()

    @classmethod
    def func_5jag95el(cls, impl, **kwargs) -> None:
        ...

    @staticmethod
    def func_8hn0mqkp() -> 'IOLoop':
        ...

    def func_ocn834ij(self) -> None:
        ...

    @staticmethod
    def func_u9msnike() -> 'IOLoop':
        ...

    def func_56oi4lqx(self) -> None:
        ...

    @typing.overload
    def func_rk4h63vw() -> None:
        ...

    @typing.overload
    def func_rk4h63vw(instance: bool = True) -> None:
        ...

    @staticmethod
    def func_rk4h63vw(instance: bool = True) -> 'IOLoop':
        ...

    def func_yn27xd01(self, make_current: bool = True) -> None:
        ...

    def func_yd9g72ty(self, all_fds: bool = False) -> None:
        ...

    def func_vmyxvace(self, fd, handler, events) -> None:
        ...

    def func_hlj32lk4(self, fd, events) -> None:
        ...

    def func_5xoxrzox(self, fd) -> None:
        ...

    def func_8kik7pqz(self) -> None:
        ...

    def func_pzx33ac6(self) -> None:
        ...

    def func_mbdt54km(self, func, timeout=None) -> Any:
        ...

    def func_lnxl8vsd(self) -> float:
        ...

    def func_8u27iv4y(self, deadline, callback, *args, **kwargs) -> Any:
        ...

    def func_s386yq5t(self, delay, callback, *args, **kwargs) -> Any:
        ...

    def func_xt1jy7z3(self, when, callback, *args, **kwargs) -> Any:
        ...

    def func_d9v5yqw9(self, timeout) -> None:
        ...

    def func_gxxxai45(self, callback, *args, **kwargs) -> None:
        ...

    def func_jpa7pjrj(self, future) -> None:
        ...

    def func_fkmt5kds(self, executor, func, *args) -> Any:
        ...

    def func_o3y5nde4(self, executor) -> None:
        ...

    def func_0x7gr8hn(self, callback) -> None:
        ...

    def func_jpa7pjrj(self, future) -> None:
        ...

    def func_fkmt5kds(self, executor, func, *args) -> Any:
        ...

    def func_o3y5nde4(self, executor) -> None:
        ...

    def func_0x7gr8hn(self, callback) -> None:
        ...

class _Timeout:
    def __init__(self, deadline, callback, io_loop) -> None:
        ...

class PeriodicCallback:
    def __init__(self, callback, callback_time, jitter=0) -> None:
        ...

    def func_8kik7pqz(self) -> None:
        ...

    def func_pzx33ac6(self) -> None:
        ...

    def func_r71ebn6g(self) -> bool:
        ...

    async def func_1whxejnf(self) -> None:
        ...

    def func_z8uci1fw(self) -> None:
        ...

    def func_tmaw2hmm(self, current_time) -> None:
        ...
