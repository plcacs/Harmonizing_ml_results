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


lambda a: int, /: a  # type: ignore

lambda a: int, b: int, /, c: int, d: int, *, e: int, f: int: a  # type: ignore

lambda a: int, b: int, /, c: int, d: int, *args: int, e: int, f: int, **kwargs: int: args  # type: ignore

lambda a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5: 1  # type: ignore
