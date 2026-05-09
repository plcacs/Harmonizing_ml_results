def positional_only_arg(a: object) -> None:
    pass

def all_markers(a: object, b: object, /, c: object, d: object, *, e: object, f: object) -> None:
    pass

def all_markers_with_args_and_kwargs(a_long_one: object, b_long_one: object, /, c_long_one: object, d_long_one: object, *args: object, e_long_one: object, f_long_one: object, **kwargs: object) -> None:
    pass

def all_markers_with_defaults(a: object, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5) -> None:
    pass

def long_one_with_long_parameter_names(but_all_of_them: object, are_positional_only: object, arguments_mmmmkay: object, so_this_is_only_valid_after: object, three_point_eight: object, /) -> None:
    pass

positional_only_lambda = lambda a: object, /: a
all_markers_lambda = lambda a: object, b: object, /, c: object, d: object, *, e: object, f: object: a
all_markers_with_args_and_kwargs_lambda = lambda a: object, b: object, /, c: object, d: object, *args: object, e: object, f: object, **kwargs: object: args
all_markers_with_defaults_lambda = lambda a: object, b: int = 1, /, c: int = 2, d: int = 3, *, e: int = 4, f: int = 5: 1
