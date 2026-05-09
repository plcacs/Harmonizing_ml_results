import pytest
from pydantic import PydanticUserError
from pydantic._internal._decorators import inspect_annotated_serializer, inspect_validator

def _two_pos_required_args(a: object, b: object) -> None:
    pass

def _two_pos_required_args_extra_optional(a: object, b: object, c: int = 1, d: int = 2, *, e: int = 3) -> None:
    pass

def _three_pos_required_args(a: object, b: object, c: object) -> None:
    pass

def _one_pos_required_arg_one_optional(a: object, b: int = 1) -> None:
    pass

def test_inspect_validator(obj: callable, mode: str, expected: bool) -> None:
    """Test inspect_validator function."""
    assert inspect_validator(obj, mode=mode) == expected

def test_inspect_validator_error_wrap() -> None:
    """Test inspect_validator function with wrap mode and incorrect validator signature."""
    def validator1(arg1: object) -> None:
        pass

    def validator4(arg1: object, arg2: object, arg3: object, arg4: object) -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator1, mode='wrap')
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator4, mode='wrap')
    assert e.value.code == 'validator-signature'

def test_inspect_validator_error(mode: str) -> None:
    """Test inspect_validator function with incorrect validator signature."""
    def validator() -> None:
        pass

    def validator3(arg1: object, arg2: object, arg3: object) -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator, mode=mode)
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator3, mode=mode)
    assert e.value.code == 'validator-signature'

def test_inspect_annotated_serializer(obj: callable, mode: str, expected: bool) -> None:
    """Test inspect_annotated_serializer function."""
    assert inspect_annotated_serializer(obj, mode=mode) == expected

def test_inspect_annotated_serializer_invalid_number_of_arguments(mode: str) -> None:
    """Test inspect_annotated_serializer function with invalid number of arguments."""
    def serializer() -> None:
        pass
    with pytest.raises(PydanticUserError) as e:
        inspect_annotated_serializer(serializer, mode=mode)
    assert e.value.code == 'field-serializer-signature'
