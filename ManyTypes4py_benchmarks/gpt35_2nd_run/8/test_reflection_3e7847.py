from typing import Any, Tuple, Dict, Set, Callable, Union, List, Optional

def do_conversion_test(f: Callable, args: Tuple, kwargs: Dict) -> None:
    result = f(*args, **kwargs)
    cargs, ckwargs = convert_keyword_arguments(f, args, kwargs)
    assert result == f(*cargs, **ckwargs)
    cargs2, ckwargs2 = convert_positional_arguments(f, args, kwargs)
    assert result == f(*cargs2, **ckwargs2)

def test_simple_conversion() -> None:

    def foo(a: int, b: int, c: int) -> Tuple[int, int, int]:
        return (a, b, c)
    assert convert_keyword_arguments(foo, (1, 2, 3), {}) == ((1, 2, 3), {})
    assert convert_keyword_arguments(foo, (), {'a': 3, 'b': 2, 'c': 1}) == ((3, 2, 1), {})
    do_conversion_test(foo, (1, 0), {'c': 2})
    do_conversion_test(foo, (1,), {'c': 2, 'b': 'foo'})

def test_leaves_unknown_kwargs_in_dict() -> None:

    def bar(x: int, **kwargs: Any) -> None:
        pass
    assert convert_keyword_arguments(bar, (1,), {'foo': 'hi'}) == ((1,), {'foo': 'hi'})
    assert convert_keyword_arguments(bar, (), {'x': 1, 'foo': 'hi'}) == ((1,), {'foo': 'hi'})
    do_conversion_test(bar, (1,), {})
    do_conversion_test(bar, (), {'x': 1, 'y': 1})

def test_errors_on_bad_kwargs() -> None:

    def bar() -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(bar, (), {'foo': 1})

def test_passes_varargs_correctly() -> None:

    def foo(*args: Any) -> None:
        pass
    assert convert_keyword_arguments(foo, (1, 2, 3), {}) == ((1, 2, 3), {})
    do_conversion_test(foo, (1, 2, 3), {})

def test_errors_if_keyword_precedes_positional() -> None:

    def foo(x: int, y: int) -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(foo, (1,), {'x': 2})

def test_errors_if_not_enough_args() -> None:

    def foo(a: int, b: int, c: int, d: int = 1) -> None:
        pass
    with raises(TypeError):
        convert_keyword_arguments(foo, (1, 2), {'d': 4})

def test_errors_on_extra_kwargs() -> None:

    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='keyword'):
        convert_keyword_arguments(foo, (1,), {'b': 1})
    with raises(TypeError, match='keyword'):
        convert_keyword_arguments(foo, (1,), {'b': 1, 'c': 2})

def test_positional_errors_if_too_many_args() -> None:

    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='too many positional arguments'):
        convert_positional_arguments(foo, (1, 2), {})

def test_positional_errors_if_too_few_args() -> None:

    def foo(a: int, b: int, c: int) -> None:
        pass
    with raises(TypeError):
        convert_positional_arguments(foo, (1, 2), {})

def test_positional_does_not_error_if_extra_args_are_kwargs() -> None:

    def foo(a: int, b: int, c: int) -> None:
        pass
    convert_positional_arguments(foo, (1, 2), {'c': 3})

def test_positional_errors_if_given_bad_kwargs() -> None:

    def foo(a: int) -> None:
        pass
    with raises(TypeError, match="missing a required argument: 'a'"):
        convert_positional_arguments(foo, (), {'b': 1})

def test_positional_errors_if_given_duplicate_kwargs() -> None:

    def foo(a: int) -> None:
        pass
    with raises(TypeError, match='multiple values'):
        convert_positional_arguments(foo, (2,), {'a': 1})

def test_names_of_functions_are_pretty() -> None:
    assert get_pretty_function_description(test_names_of_functions_are_pretty) == 'test_names_of_functions_are_pretty'

def test_class_names_are_not_included_in_class_method_prettiness() -> None:
    assert get_pretty_function_description(Foo.bar) == 'bar'

def test_repr_is_included_in_bound_method_prettiness() -> None:
    assert get_pretty_function_description(Foo().baz) == 'SoNotFoo().baz'

def test_class_is_not_included_in_unbound_method() -> None:
    assert get_pretty_function_description(Foo.baz) == 'baz'

def test_does_not_error_on_confused_sources() -> None:

    def ed(f: Callable, *args: Any) -> Callable:
        return f
    x = ed(lambda x, y: (x * y).conjugate() == x.conjugate() * y.conjugate(), complex, complex)
    get_pretty_function_description(x)

def test_digests_are_reasonably_unique() -> None:
    assert function_digest(test_simple_conversion) != function_digest(test_does_not_error_on_confused_sources)

def test_digest_returns_the_same_value_for_two_calls() -> None:
    assert function_digest(test_simple_conversion) == function_digest(test_simple_conversion)

def test_can_digest_a_built_in_function() -> None:
    import math
    assert function_digest(math.isnan) != function_digest(range)

def test_can_digest_a_unicode_lambda() -> None:
    function_digest(lambda x: '☃' in str(x))

def test_can_digest_a_function_with_no_name() -> None:

    def foo(x: int, y: int) -> None:
        pass
    function_digest(partial(foo, 1))

def test_arg_string_is_in_order() -> None:

    def foo(c: int, a: int, b: int, f: int, a1: int) -> None:
        pass
    assert repr_call(foo, (1, 2, 3, 4, 5), {}) == 'foo(c=1, a=2, b=3, f=4, a1=5)'
    assert repr_call(foo, (1, 2), {'b': 3, 'f': 4, 'a1': 5}) == 'foo(c=1, a=2, b=3, f=4, a1=5)'

def test_varkwargs_are_sorted_and_after_real_kwargs() -> None:

    def foo(d: int, e: int, f: int, **kwargs: Any) -> None:
        pass
    assert repr_call(foo, (), {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}) == 'foo(d=4, e=5, f=6, a=1, b=2, c=3)'

def test_varargs_come_without_equals() -> None:

    def foo(a: int, *args: int) -> None:
        pass
    assert repr_call(foo, (1, 2, 3, 4), {}) == 'foo(2, 3, 4, a=1)'

def test_can_mix_varargs_and_varkwargs() -> None:

    def foo(*args: int, **kwargs: Any) -> None:
        pass
    assert repr_call(foo, (1, 2, 3), {'c': 7}) == 'foo(1, 2, 3, c=7)'

def test_arg_string_does_not_include_unprovided_defaults() -> None:

    def foo(a: int, b: int, c: int = 9, d: int = 10) -> None:
        pass
    assert repr_call(foo, (1,), {'b': 1, 'd': 11}) == 'foo(a=1, b=1, d=11)'

def universal_acceptor(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:

def has_one_arg(hello: Any) -> None:

def has_two_args(hello: Any, world: Any) -> None:

def has_a_default(x: int, y: int, z: int = 1) -> None:

def has_varargs(*args: Any) -> None:

def has_kwargs(**kwargs: Any) -> None:

def test_copying_preserves_signature(f: Callable) -> None:

def test_name_does_not_clash_with_function_names() -> None:

def test_copying_sets_name() -> None:

def test_copying_sets_docstring() -> None:

def test_uses_defaults() -> None:

def test_uses_varargs() -> None:

DEFINE_FOO_FUNCTION: str = '\ndef foo(x):\n    return x\n'

def test_exec_as_module_execs() -> None:

def test_exec_as_module_caches() -> None:

def test_exec_leaves_sys_path_unchanged() -> None:

def test_define_function_signature_works_with_conflicts() -> None:

def test_define_function_signature_validates_function_name() -> None:

def test_copying_sets_name() -> None:

def test_copying_sets_docstring() -> None:

def test_uses_defaults() -> None:

def test_uses_varargs() -> None:

def test_inline_given_handles_self() -> None:

def logged(f: Callable) -> Callable:

def test_issue_2495_regression(_) -> None:

def test_error_on_keyword_parameter_name() -> None:

def test_param_is_called_within_func() -> None:

def test_param_is_called_within_subfunc() -> None:

def test_param_is_not_called_within_func() -> None:

def test_param_called_within_defaults_on_error() -> None:

def _prep_source(*pairs: Tuple[str, str]) -> List[pytest.param]:

def test_clean_source(src: str, clean: bytes) -> None:

def test_overlong_repr_warns() -> None:
