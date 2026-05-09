from typing import Any, Tuple, List, Dict, Callable, Optional
from itertools import chain

def f(a: Any) -> None:
    pass

def f(a: Any, b: Any, c: Any, d: Any, e: Any, f: Any, g: Any, h: Any, i: Any) -> None:
    pass

def f(a: Any, b: Any, c: Any, d: Any, e: Any, f: Any, g: Any, h: Any, i: Any) -> None:
    pass

def f(arg: Any, *args: Any, default: bool = False, **kwargs: Dict[str, Any]) -> None:
    pass

def f(a: int, b: int, c: int, d: int) -> None:
    element: int = 0
    another_element: int = 1
    another_element_with_long_name: int = 2
    another_really_really_long_element_with_a_unnecessarily_long_name_to_describe_what_it_does_enterprise_style: int = 3
    an_element_with_a_long_value: int = (calls() or (more_calls() and more()))  # type: ignore
    tup: Tuple[int, int] = (another_element, another_really_really_long_element_with_a_unnecessarily_long_name_to_describe_what_it_does_enterprise_style)
    a = element + another_element + another_element_with_long_name + element + another_element + another_element_with_long_name

def f(x: Any, y: Any) -> None:
    pass

def f(x: Any) -> None:
    pass

def func(a: Any = some_list[0]) -> None:
    c: List[float] = call(0.0123, 0.0456, 0.0789, 0.0123, 0.0456, 0.0789, 0.0123, 0.0456, 0.0789, a[-1])
    c: List[str] = call('aaaaaaaa', 'aaaaaaaa', 'aaaaaaaa', 'aaaaaaaa', 'aaaaaaaa', 'aaaaaaaa', 'aaaaaaaa')
result: str = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
AAAAAAAAAAAAA: List[Any] = [AAAAAAAAAAAAA] + SHARED_AAAAAAAAAAAAA + USER_AAAAAAAAAAAAA + AAAAAAAAAAAAA
call_to_some_function_asdf: Callable[[str, List[Any]], None] = call_to_some_function_asdf
call_to_some_function_asdf(foo, [AAAAAAAAAAAAAAAAAAAAAAA, AAAAAAAAAAAAAAAAAAAAAAA, AAAAAAAAAAAAAAAAAAAAAAA, BBBBBBBBBBBB])
aaaaaaaaaaaaa, bbbbbbbbb: List[List[Any]] = map(list, map(chain.from_iterable, zip(*items)))
