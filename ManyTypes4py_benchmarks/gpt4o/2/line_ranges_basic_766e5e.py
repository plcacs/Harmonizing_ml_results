from typing import Any, Callable, Coroutine, Generator

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

async def test_async_unformatted() -> Coroutine[Any, Any, None]:
    async for i in some_iter(unformatted):
        await asyncio.sleep(1)
        async with some_context(unformatted):
            print('unformatted')
