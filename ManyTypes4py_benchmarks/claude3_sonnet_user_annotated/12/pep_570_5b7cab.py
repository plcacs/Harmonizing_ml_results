def positional_only_arg(a: int, /) -> None:
    pass


def all_markers(a: str, b: int, /, c: float, d: bool, *, e: list, f: dict) -> None:
    pass


def all_markers_with_args_and_kwargs(
    a_long_one: str,
    b_long_one: int,
    /,
    c_long_one: float,
    d_long_one: bool,
    *args: tuple,
    e_long_one: list,
    f_long_one: dict,
    **kwargs: dict,
) -> None:
    pass


def all_markers_with_defaults(a: str, b: int = 1, /, c: float = 2, d: bool = 3, *, e: list = 4, f: dict = 5) -> None:
    pass


def long_one_with_long_parameter_names(
    but_all_of_them: str,
    are_positional_only: int,
    arguments_mmmmkay: float,
    so_this_is_only_valid_after: bool,
    three_point_eight: list,
    /,
) -> None:
    pass


lambda a: int, /: a

lambda a: str, b: int, /, c: float, d: bool, *, e: list, f: dict: a

lambda a: str, b: int, /, c: float, d: bool, *args: tuple, e: list, f: dict, **kwargs: dict: args

lambda a: str, b: int = 1, /, c: float = 2, d: bool = 3, *, e: list = 4, f: dict = 5: 1
