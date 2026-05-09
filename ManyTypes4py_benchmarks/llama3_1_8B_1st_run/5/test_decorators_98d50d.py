import pytest
from pydantic import PydanticUserError
from pydantic._internal._decorators import (
    inspect_annotated_serializer,
    inspect_validator,
)

def _two_pos_required_args(a: object, b: object) -> None:
    pass

def _two_pos_required_args_extra_optional(
    a: object, b: object, c: int = 1, d: int = 2, *, e: int = 3
) -> None:
    pass

def _three_pos_required_args(a: object, b: object, c: object) -> None:
    pass

def _one_pos_required_arg_one_optional(a: object, b: int = 1) -> None:
    pass

def test_inspect_validator(
    obj: object,
    mode: str,
    expected: bool,
) -> None:
    assert inspect_validator(obj, mode=mode) == expected

def test_inspect_validator_error_wrap(
    validator1: callable,
    validator4: callable,
) -> None:
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator1, mode='wrap')
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator4, mode='wrap')
    assert e.value.code == 'validator-signature'

def test_inspect_validator_error(
    mode: str,
    validator: callable,
    validator3: callable,
) -> None:
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator, mode=mode)
    assert e.value.code == 'validator-signature'
    with pytest.raises(PydanticUserError) as e:
        inspect_validator(validator3, mode=mode)
    assert e.value.code == 'validator-signature'

def test_inspect_annotated_serializer(
    obj: object,
    mode: str,
    expected: bool,
) -> None:
    assert inspect_annotated_serializer(obj, mode=mode) == expected

def test_inspect_annotated_serializer_invalid_number_of_arguments(
    serializer: callable,
    mode: str,
) -> None:
    with pytest.raises(PydanticUserError) as e:
        inspect_annotated_serializer(serializer, mode=mode)
    assert e.value.code == 'field-serializer-signature'
