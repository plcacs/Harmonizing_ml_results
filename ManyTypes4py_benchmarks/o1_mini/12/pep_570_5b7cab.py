from typing import Any, Callable, Tuple, Dict

def positional_only_arg(a: Any, /) -> Any:
    pass

def all_markers(a: Any, b: Any, /, c: Any, d: Any, *, e: Any, f: Any) -> Any:
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
) -> Any:
    pass

def all_markers_with_defaults(
    a: Any,
    b: int = 1,
    /,
    c: int = 2,
    d: int = 3,
    *,
    e: int = 4,
    f: int = 5
) -> Any:
    pass

def long_one_with_long_parameter_names(
    but_all_of_them: Any,
    are_positional_only: Any,
    arguments_mmmmkay: Any,
    so_this_is_only_valid_after: Any,
    three_point_eight: Any,
    /,
) -> Any:
    pass

lambda_a: Callable[[Any], Any] = lambda a, /: a
lambda_b: Callable[[Any, Any, Any, Any, Any, Any], Any] = lambda a, b, /, c, d, *, e, f: a
lambda_c: Callable[[Any, Any, Any, Any, Any, Any, Any], Any] = lambda a, b, /, c, d, *args, e, f, **kwargs: args
lambda_d: Callable[[Any, Any, Any, Any, Any, Any], Any] = lambda a, b=1, /, c=2, d=3, *, e=4, f=5: 1
