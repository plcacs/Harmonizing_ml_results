import pytest
from typing import Any, List, Tuple, Union, cast
from eth_utils import ValidationError
from eth.exceptions import FullStack, InsufficientStack
from eth.vm.stack import Stack

@pytest.fixture
def stack() -> Stack:
    return Stack()

@pytest.mark.parametrize('value,is_valid', ((b'abcde', True), (b'100100100100100100100100100100100', False)))
def test_push_only_pushes_valid_stack_bytes(stack: Stack, value: bytes, is_valid: bool) -> None:
    if is_valid:
        stack.push_bytes(value)
        assert stack.pop1_any() == value
    else:
        with pytest.raises(ValidationError):
            stack.push_bytes(value)

@pytest.mark.parametrize('value,is_valid', ((-1, False), (0, True), (1, True), (2 ** 256 - 1, True), (2 ** 256, False)))
def test_push_only_pushes_valid_stack_ints(stack: Stack, value: int, is_valid: bool) -> None:
    if is_valid:
        stack.push_int(value)
        assert stack.pop1_any() == value
    else:
        with pytest.raises(ValidationError):
            stack.push_int(value)

def test_push_does_not_allow_stack_to_exceed_1024_items(stack: Stack) -> None:
    for num in range(1024):
        stack.push_int(num)
    assert len(stack.values) == 1024
    with pytest.raises(FullStack):
        stack.push_int(1025)

def test_dup_does_not_allow_stack_to_exceed_1024_items(stack: Stack) -> None:
    stack.push_int(1)
    for _ in range(1023):
        stack.dup(1)
    assert len(stack.values) == 1024
    with pytest.raises(FullStack):
        stack.dup(1)

@pytest.mark.parametrize('items, stack_method', (([1], 'push_int'), ([1, 2, 3], 'push_int'), ([b'1', b'10', b'101', b'1010'], 'push_bytes')))
def test_pop_returns_latest_stack_item(stack: Stack, items: Union[List[int], List[bytes]], stack_method: str) -> None:
    method = getattr(stack, stack_method)
    for each in items:
        method(each)
    assert stack.pop1_any() == items[-1]

@pytest.mark.parametrize('value, push_method, pop_method, expect_result', ((1, 'push_int', 'pop_ints', (1,)), (1, 'push_int', 'pop_any', (1,)), (1, 'push_int', 'pop_bytes', (b'\x01',)), (1, 'push_int', 'pop1_int', 1), (1, 'push_int', 'pop1_any', 1), (1, 'push_int', 'pop1_bytes', b'\x01'), (b'\t', 'push_bytes', 'pop_ints', (9,)), (b'\t', 'push_bytes', 'pop_any', (b'\t',)), (b'\t', 'push_bytes', 'pop_bytes', (b'\t',)), (b'\t', 'push_bytes', 'pop1_int', 9), (b'\t', 'push_bytes', 'pop1_any', b'\t'), (b'\t', 'push_bytes', 'pop1_bytes', b'\t')))
def test_pop_different_types(stack: Stack, value: Union[int, bytes], push_method: str, pop_method: str, expect_result: Union[int, bytes, Tuple[int, ...], Tuple[bytes, ...]]) -> None:
    push = getattr(stack, push_method)
    push(value)
    pop = getattr(stack, pop_method)
    if '1' in pop_method:
        assert pop() == expect_result
    else:
        assert pop(1) == expect_result

def _validate_stack_integers(stack: Stack, expected_values: List[int]) -> None:
    popped = stack.pop_any(len(stack))
    assert popped == tuple(reversed(expected_values))
    for val in reversed(popped):
        stack.push_int(cast(int, val))

def test_swap_operates_correctly(stack: Stack) -> None:
    for num in range(5):
        stack.push_int(num)
    _validate_stack_integers(stack, [0, 1, 2, 3, 4])
    stack.swap(3)
    _validate_stack_integers(stack, [0, 4, 2, 3, 1])
    stack.swap(1)
    _validate_stack_integers(stack, [0, 4, 2, 1, 3])

def test_dup_operates_correctly(stack: Stack) -> None:
    for num in range(5):
        stack.push_int(num)
    _validate_stack_integers(stack, [0, 1, 2, 3, 4])
    stack.dup(1)
    _validate_stack_integers(stack, [0, 1, 2, 3, 4, 4])
    stack.dup(5)
    _validate_stack_integers(stack, [0, 1, 2, 3, 4, 4, 1])

def test_pop_raises_InsufficientStack_appropriately(stack: Stack) -> None:
    with pytest.raises(InsufficientStack):
        stack.pop1_int()

def test_swap_raises_InsufficientStack_appropriately(stack: Stack) -> None:
    with pytest.raises(InsufficientStack):
        stack.swap(0)

def test_dup_raises_InsufficientStack_appropriately(stack: Stack) -> None:
    with pytest.raises(InsufficientStack):
        stack.dup(0)
