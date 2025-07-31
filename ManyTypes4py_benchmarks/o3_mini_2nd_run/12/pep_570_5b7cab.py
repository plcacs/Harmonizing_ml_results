from typing import Any, Callable, Tuple

def positional_only_arg(a: Any, /) -> None:
    pass

def all_markers(a: Any, b: Any, /, c: Any, d: Any, *, e: Any, f: Any) -> None:
    pass

def all_markers_with_args_and_kwargs(
    a_long_one: Any, 
    b_long_one: Any, 
    /, 
    c_long_one: Any, 
    d_long_one: Any, 
    *args: Any, 
    e_long_one: Any, 
    f_long_one: Any, 
    **kwargs: Any
) -> None:
    pass

def all_markers_with_defaults(a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5) -> None:
    pass

def long_one_with_long_parameter_names(
    but_all_of_them: Any, 
    are_positional_only: Any, 
    arguments_mmmmkay: Any, 
    so_this_is_only_valid_after: Any, 
    three_point_eight: Any, 
    /
) -> None:
    pass

f1: Callable[[Any], Any] = lambda a, /: a
f2: Callable[[Any, Any, Any, Any, Any, Any], Any] = lambda a, b, /, c, d, *, e, f: a
f3: Callable[..., Tuple[Any, ...]] = lambda a, b, /, c, d, *args, e, f, **kwargs: args
f4: Callable[[int, int, int, int, int, int], int] = lambda a, b=1, /, c=2, d=3, *, e=4, f=5: 1