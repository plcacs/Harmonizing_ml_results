import pytest
from typing import Any, Callable, List, Tuple, Union
from pydantic import PydanticUserError
from pydantic._internal._decorators import inspect_annotated_serializer, inspect_validator

def _two_pos_required_args(a: Any, b: Any) -> None:
    pass

def _two_pos_required_args_extra_optional(a: Any, b: Any, c: int = 1, d: int = 2, *, e: int = 3) -> None:
    pass

def _three_pos_required_args(a: Any, b: Any, c: Any) -> None:
    pass

def _one_pos_required_arg_one_optional(a: Any, b: int = 1) -> None:
    pass

@pytest.mark.parametrize(['obj', 'mode', 'expected'], [
    (str, 'plain', False),
    (float, 'plain', False),
    (int, 'plain', False),
    (lambda a: str(a), 'plain', False),
    (lambda a='': str(a), 'plain', False),
    (_two_pos_required_args, 'plain', True),
    (_two_pos_required_args, 'wrap', False),
    (_two_pos_required_args_extra_optional, 'plain', True),
    (_two_pos_required_args_extra_optional, 'wrap', False),
    (_three_pos_required_args, 'wrap', True),
    (_one_pos_required_arg_one_optional, 'plain', False)
])
def test_inspect_validator(obj: Any, mode: str, expected: bool) -> None:
    assert inspect_validator(obj, mode=mode) == expected

def test_inspect_validator_error_wrap() -> None:

    def validator1(arg1: Any) -> None:
        pass

    def validator4(arg1: Any, arg2: Any, arg3: Any, arg4: Any) -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator1, mode='wrap')
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator4, mode='wrap')
    assert e.value.code == 'validator-signature'

@pytest.mark.parametrize('mode', ['before', 'after', 'plain'])
def test_inspect_validator_error(mode: str) -> None:

    def validator() -> None:
        pass

    def validator3(arg1: Any, arg2: Any, arg3: Any) -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator, mode=mode)
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator3, mode=mode)
    assert e.value.code == 'validator-signature'

@pytest.mark.parametrize(['obj', 'mode', 'expected'], [
    (str, 'plain', False),
    (float, 'plain', False),
    (int, 'plain', False),
    (lambda a: str(a), 'plain', False),
    (lambda a='': str(a), 'plain', False),
    (_two_pos_required_args, 'plain', True),
    (_two_pos_required_args, 'wrap', False),
    (_two_pos_required_args_extra_optional, 'plain', True),
    (_two_pos_required_args_extra_optional, 'wrap', False),
    (_three_pos_required_args, 'wrap', True),
    (_one_pos_required_arg_one_optional, 'plain', False)
])
def test_inspect_annotated_serializer(obj: Any, mode: str, expected: bool) -> None:
    assert inspect_annotated_serializer(obj, mode=mode) == expected

@pytest.mark.parametrize('mode', ['plain', 'wrap'])
def test_inspect_annotated_serializer_invalid_number_of_arguments(mode: str) -> None:

    def serializer() -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_annotated_serializer(serializer, mode=mode)
    assert e.value.code == 'field-serializer-signature'
