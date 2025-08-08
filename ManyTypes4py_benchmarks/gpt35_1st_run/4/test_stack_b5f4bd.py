from eth.vm.stack import Stack
import pytest
from eth_utils import ValidationError
from eth.exceptions import FullStack, InsufficientStack
from typing import Tuple, Union, List

def test_push_only_pushes_valid_stack_bytes(stack: Stack, value: bytes, is_valid: bool) -> None:
def test_push_only_pushes_valid_stack_ints(stack: Stack, value: int, is_valid: bool) -> None:
def test_push_does_not_allow_stack_to_exceed_1024_items(stack: Stack) -> None:
def test_dup_does_not_allow_stack_to_exceed_1024_items(stack: Stack) -> None:
def test_pop_returns_latest_stack_item(stack: Stack, items: List[Union[int, bytes]], stack_method: str) -> None:
def test_pop_different_types(stack: Stack, value: Union[int, bytes], push_method: str, pop_method: str, expect_result: Union[int, bytes, Tuple[Union[int, bytes]]]) -> None:
def _validate_stack_integers(stack: Stack, expected_values: List[int]) -> None:
def test_swap_operates_correctly(stack: Stack) -> None:
def test_dup_operates_correctly(stack: Stack) -> None:
def test_pop_raises_InsufficientStack_appropriately(stack: Stack) -> None:
def test_swap_raises_InsufficientStack_appropriately(stack: Stack) -> None:
def test_dup_raises_InsufficientStack_appropriately(stack: Stack) -> None:
