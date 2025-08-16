from typing import Any, AsyncGenerator

def foo1(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo2(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo3(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

def foo4(parameter_1: Any, parameter_2: Any, parameter_3: Any, parameter_4: Any, parameter_5: Any, parameter_6: Any, parameter_7: Any) -> None:
    pass

class MyClass(object):

    @decor(1 * 3)
    def my_func(self, arg: Any) -> None:
        pass

async def test_async_unformatted() -> AsyncGenerator:
    async for i in some_iter(unformatted):
        await asyncio.sleep(1)
        async with some_context(unformatted):
            print('unformatted')
