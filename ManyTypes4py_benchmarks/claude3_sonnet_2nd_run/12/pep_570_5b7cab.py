def positional_only_arg(a: int, /) -> None:
    pass

def all_markers(a: int, b: str, /, c: float, d: list, *, e: dict, f: bool) -> None:
    pass

def all_markers_with_args_and_kwargs(a_long_one: int, b_long_one: str, /, c_long_one: float, d_long_one: list, *args: tuple, e_long_one: dict, f_long_one: bool, **kwargs: dict) -> None:
    pass

def all_markers_with_defaults(a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5) -> None:
    pass

def long_one_with_long_parameter_names(but_all_of_them: str, are_positional_only: str, arguments_mmmmkay: str, so_this_is_only_valid_after: str, three_point_eight: str, /) -> None:
    pass

lambda a: int, /: a
lambda a: int, b: str, /, c: float, d: list, *, e: dict, f: bool: a
lambda a: int, b: str, /, c: float, d: list, *args: tuple, e: dict, f: bool, **kwargs: dict: args
lambda a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5: 1
