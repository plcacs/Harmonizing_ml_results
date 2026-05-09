def positional_only_arg(a: int, /) -> None:
    pass

def all_markers(a: int, b: int, /, c: int, d: int, *, e: str, f: bool) -> None:
    pass

def all_markers_with_args_and_kwargs(a_long_one: int, b_long_one: int, /, c_long_one: int, d_long_one: int, *args: list, e_long_one: str, f_long_one: bool, **kwargs: dict) -> None:
    pass

def all_markers_with_defaults(a: int, b: int = 1, /, c: int = 2, d: int = 3, *, e: str = 'default', f: bool = True) -> None:
    pass

def long_one_with_long_parameter_names(but_all_of_them: int, are_positional_only: int, arguments_mmmmkay: int, so_this_is_only_valid_after: int, three_point_eight: int, /) -> None:
    pass

lambda_a = lambda a: a
lambda_a_b = lambda a: a
lambda_a_b_c = lambda a, b, c: a
lambda_a_b_c_args_kwargs = lambda a, b, c, *args, d, e, **kwargs: args
lambda_a_b_defaults = lambda a, b=1, c=2: 1
