from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from typing import Union
from typing import AsyncGenerator
import asyncio
from functools import wraps

def decor(factor: int) -> callable:
    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def foo1(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo2(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo3(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo4(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

if True:
    pass

import typing
from typing import Any

class MyClass(object):
    def my_func(self, arg: Any) -> None:
        pass

try:
    for i in range(10):
        while True:
            if True:
                pass
            elif True:
                pass
except ValueError as e:
    def unformatted() -> None:
        pass
finally:
    def unformatted() -> None:
        pass

async def test_async_unformatted() -> None:
    async for i in some_iter(unformatted):
        await asyncio.sleep(1)
        async with some_context(unformatted):
            print('unformatted')

def foo1(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo2(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo3(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo4(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

if True:
    pass

import typing
from typing import Any

class MyClass(object):
    @decor(1 * 3)
    def my_func(self, arg: Any) -> None:
        pass

try:
    for i in range(10):
        while True:
            if True:
                pass
            elif True:
                pass
except ValueError as e:
    def unformatted() -> None:
        pass
finally:
    def unformatted() -> None:
        pass

async def test_async_unformatted() -> None:
    async for i in some_iter(unformatted):
        await asyncio.sleep(1)
        async with some_context(unformatted):
            print('unformatted')
