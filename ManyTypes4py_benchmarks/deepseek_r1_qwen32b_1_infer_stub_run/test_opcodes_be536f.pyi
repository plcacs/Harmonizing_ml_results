from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)
from pytest import MarkDecorator
from eth_utils import ValidationError
from eth.vm.forks import Fork
from eth.vm.computation import BaseComputation
from eth._utils.address import Address
from eth._utils.padding import PaddedBytes32
from eth.consensus import ConsensusContext
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.vm.chain_context import ChainContext
from eth.vm.spoof import SpoofTransaction
from eth.vm.message import Message
from eth.vm.forks.cancun.computation import CancunComputation

NORMALIZED_ADDRESS_A: str = ...
NORMALIZED_ADDRESS_B: str = ...
ADDRESS_WITH_CODE: Tuple[str, bytes] = ...
EMPTY_ADDRESS_IN_STATE: str = ...
ADDRESS_NOT_IN_STATE: str = ...
ADDRESS_WITH_JUST_BALANCE: str = ...
CANONICAL_ADDRESS_A: bytes = ...
CANONICAL_ADDRESS_B: bytes = ...
CANONICAL_ADDRESS_C: bytes = ...
CANONICAL_ZERO_ADDRESS: bytes = ...

def assemble(*codes: Union[str, bytes]) -> bytes: ...

def setup_vm(vm_class: Fork, chain_id: Optional[int] = None) -> Fork: ...

def run_computation(
    vm: Fork,
    create_address: Optional[bytes],
    code: bytes,
    gas: int = ...,
    to: bytes = ...,
    transaction_sender: bytes = ...,
    data: bytes = ...,
    access_list: Optional[List[Tuple[bytes, List[int]]]] = ...,
) -> BaseComputation: ...

def run_general_computation(
    vm_class: Fork,
    create_address: Optional[bytes] = ...,
    code: bytes = ...,
    chain_id: Optional[int] = ...,
) -> BaseComputation: ...

@pytest.mark.parametrize('vm_class, val1, val2, expected', [...])
def test_add(vm_class: Fork, val1: int, val2: int, expected: int) -> None: ...

def test_base_fee() -> None: ...

@pytest.mark.parametrize('opcode_value, expected', [...])
def test_nullary_opcodes(VM: Fork, opcode_value: int, expected: Union[int, bytes]) -> None: ...

@pytest.mark.parametrize('val1, expected', [...])
def test_blockhash(VM: Fork, val1: int, expected: bytes) -> None: ...

@pytest.mark.parametrize('vm_class, val1, val2, expected', [...])
def test_mul(vm_class: Fork, val1: int, val2: int, expected: int) -> None: ...

@pytest.mark.parametrize('vm_class, base, exponent, expected', [...])
def test_exp(vm_class: Fork, base: int, exponent: int, expected: int) -> None: ...

@pytest.mark.parametrize('vm_class, val1, val2, expected', [...])
def test_shl(vm_class: Fork, val1: str, val2: str, expected: str) -> None: ...

@pytest.mark.parametrize('vm_class, val1, val2, expected', [...])
def test_shr(vm_class: Fork, val1: str, val2: str, expected: str) -> None: ...

@pytest.mark.parametrize('vm_class, val1, val2, expected', [...])
def test_sar(vm_class: Fork, val1: str, val2: str, expected: str) -> None: ...

@pytest.mark.parametrize('vm_class, address, expected', [...])
def test_extcodehash(vm_class: Fork, address: str, expected: str) -> None: ...

@pytest.mark.parametrize('vm_class, code, gas_used, refund, original', [...])
def test_sstore(vm_class: Fork, code: str, gas_used: int, refund: int, original: Union[int, bytes]) -> None: ...

@pytest.mark.parametrize('gas_supplied, success, gas_used, refund', [...])
def test_sstore_limit_2300(gas_supplied: int, success: bool, gas_used: int, refund: int) -> None: ...

@pytest.mark.parametrize('vm_class', [...])
@pytest.mark.parametrize('chain_id, expected_result', [...])
def test_chainid(vm_class: Fork, chain_id: int, expected_result: Union[int, ValidationError]) -> None: ...

@pytest.mark.parametrize('vm_class, code, expect_exception, expect_gas_used', [...])
def test_balance(vm_class: Fork, code: bytes, expect_exception: Optional[Exception], expect_gas_used: int) -> None: ...

@pytest.mark.parametrize('vm_class, code, expect_gas_used', [...])
def test_gas_costs(vm_class: Fork, code: bytes, expect_gas_used: int) -> None: ...

@pytest.mark.parametrize('vm_class, code, expect_gas_used, access_list', [...])
def test_access_list_gas_costs(vm_class: Fork, code: bytes, expect_gas_used: int, access_list: List[Tuple[bytes, List[int]]]) -> None: ...

@pytest.mark.parametrize('vm_class', [...])
@pytest.mark.parametrize('bytecode_hex, expect_gas_used', [...])
def test_eip2929_gas_by_cache_warmth(vm_class: Fork, bytecode_hex: str, expect_gas_used: int) -> None: ...

@pytest.mark.parametrize('vm_class, input_hex, output_hex, expect_exception', [...])
def test_blake2b_f_compression(vm_class: Fork, input_hex: str, output_hex: str, expect_exception: Optional[Exception]) -> None: ...

@pytest.mark.parametrize('vm_class', [...])
def test_selfdestruct_does_not_issue_deprecation_warning_pre_shanghai(vm_class: Fork) -> None: ...

@pytest.mark.parametrize('vm_class', [...])
def test_selfdestruct_issues_deprecation_warning_for_shanghai_and_later(vm_class: Fork) -> None: ...