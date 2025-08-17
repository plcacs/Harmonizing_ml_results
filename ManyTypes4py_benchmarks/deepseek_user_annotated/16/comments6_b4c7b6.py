from typing import Any, Tuple, List


def f(
    a: int,
) -> None:
    pass


# test type comments
def f(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> None:
    pass


def f(
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
    g: int,
    h: int,
    i: int,
) -> None:
    pass


def f(
    arg: int,
    *args: Any,
    default: bool = False,
    **kwargs: Any,
) -> None:
    pass


def f(
    a: int,
    b: int,
    c: int,
    d: int,
) -> None:
    element: int = 0
    another_element: float = 1
    another_element_with_long_name: int = 2
    another_really_really_long_element_with_a_unnecessarily_long_name_to_describe_what_it_does_enterprise_style: int = 3
    an_element_with_a_long_value: bool = calls() or more_calls() and more()

    tup: Tuple[int, int] = (
        another_element,
        another_really_really_long_element_with_a_unnecessarily_long_name_to_describe_what_it_does_enterprise_style,
    )

    a: int = (
        element
        + another_element
        + another_element_with_long_name
        + element
        + another_element
        + another_element_with_long_name
    )


def f(
    x,  # not a type comment
    y: int,
) -> None:
    pass


def f(
    x,  # not a type comment
) -> None:
    pass


def func(
    a: int = some_list[0],
) -> int:
    c = call(
        0.0123,
        0.0456,
        0.0789,
        0.0123,
        0.0456,
        0.0789,
        0.0123,
        0.0456,
        0.0789,
        a[-1],  # type: ignore
    )

    c = call(
        "aaaaaaaa", "aaaaaaaa", "aaaaaaaa", "aaaaaaaa", "aaaaaaaa", "aaaaaaaa", "aaaaaaaa"  # type: ignore
    )


result = (  # aaa
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

AAAAAAAAAAAAA: List[Any] = [AAAAAAAAAAAAA] + SHARED_AAAAAAAAAAAAA + USER_AAAAAAAAAAAAA + AAAAAAAAAAAAA  # type: ignore

call_to_some_function_asdf(
    foo,
    [AAAAAAAAAAAAAAAAAAAAAAA, AAAAAAAAAAAAAAAAAAAAAAA, AAAAAAAAAAAAAAAAAAAAAAA, BBBBBBBBBBBB],  # type: ignore
)

aaaaaaaaaaaaa: List[Any]
bbbbbbbbb: List[Any]
aaaaaaaaaaaaa, bbbbbbbbb = map(list, map(itertools.chain.from_iterable, zip(*items)))  # type: ignore[arg-type]
