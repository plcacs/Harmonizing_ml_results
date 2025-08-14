from typing import Callable, Tuple

def positional_only_arg(a: int, /) -> None:
    pass


def all_markers(a: int, b: int, /, c: int, d: int, *, e: int, f: int) -> None:
    pass


def all_markers_with_args_and_kwargs(
    a_long_one: int,
    b_long_one: int,
    /,
    c_long_one: int,
    d_long_one: int,
    *args: int,
    e_long_one: int,
    f_long_one: int,
    **kwargs: int,
) -> None:
    pass


def all_markers_with_defaults(a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5) -> None:
    pass


def long_one_with_long_parameter_names(
    but_all_of_them: int,
    are_positional_only: int,
    arguments_mmmmkay: int,
    so_this_is_only_valid_after: int,
    three_point_eight: int,
    /,
) -> None:
    pass


annotated_lambda1: Callable[[int], int] = lambda a, /: a

annotated_lambda2: Callable[[int, int, int, int, int, int], int] = lambda a, b, /, c, d, *, e, f: a

annotated_lambda3: Callable[..., Tuple[int, ...]] = lambda a, b, /, c, d, *args, e, f, **kwargs: args

annotated_lambda4: Callable[[int, int, int, int, int, int], int] = lambda a, b=1, /, c=2, d=3, *, e=4, f=5: 1