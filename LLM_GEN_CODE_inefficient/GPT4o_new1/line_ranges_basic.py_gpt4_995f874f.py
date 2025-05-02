def foo1(
    parameter_1: int,
    parameter_2: str,
    parameter_3: float,
    parameter_4: bool,
    parameter_5: list,
    parameter_6: dict,
    parameter_7: tuple,
) -> None:
    pass


def foo2(
    parameter_1: int,
    parameter_2: str,
    parameter_3: float,
    parameter_4: bool,
    parameter_5: list,
    parameter_6: dict,
    parameter_7: tuple,
) -> None:
    pass


def foo3(
    parameter_1: int,
    parameter_2: str,
    parameter_3: float,
    parameter_4: bool,
    parameter_5: list,
    parameter_6: dict,
    parameter_7: tuple,
) -> None:
    pass


def foo4(
    parameter_1: int,
    parameter_2: str,
    parameter_3: float,
    parameter_4: bool,
    parameter_5: list,
    parameter_6: dict,
    parameter_7: tuple,
) -> None:
    pass


import typing
from typing import Any


class MyClass(object):
    @decor(1 * 3)
    def my_func(self, arg: Any) -> None:
        pass


try:
    for i in range(10):
        while condition:
            if something:
                then_something()
            elif something_else:
                then_something_else()
except ValueError as e:
    unformatted()
finally:
    unformatted()


async def test_async_unformatted() -> None:
    async for i in some_iter(unformatted):
        await asyncio.sleep(1)
        async with some_context(unformatted):
            print("unformatted")
