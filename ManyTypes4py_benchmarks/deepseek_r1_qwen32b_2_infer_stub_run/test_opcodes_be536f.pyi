from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    TYPE_CHECKING,
)
from eth_utils import ValidationError
from eth.vm.forks.cancun.computation import CancunComputation
from eth.vm.message import Message
from eth.vm.spoof import SpoofTransaction

if TYPE_CHECKING:
    from eth.vm.base import BaseVM
    from eth._utils.address import force_bytes_to_address

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

def assemble(*codes: Any) -> bytes: ...

def setup_vm(vm_class: Type[BaseVM], chain_id: Optional[int] = None) -> vm_class: ...

def run_computation(
    vm: BaseVM,
    create_address: bytes,
    code: bytes,
    gas: int = ...,
    to: bytes = ...,
    transaction_sender: bytes = ...,
    data: bytes = ...,
    access_list: Optional[List[Tuple[bytes, List[int]]]] = ...,
) -> CancunComputation: ...

def run_general_computation(
    vm_class: Type[BaseVM],
    create_address: Optional[bytes] = ...,
    code: bytes = ...,
    chain_id: Optional[int] = ...,
) -> CancunComputation: ...

def test_add(
    vm_class: Type[BaseVM],
    val1: int,
    val2: int,
    expected: int,
) -> None: ...

def test_base_fee() -> None: ...

def test_nullary_opcodes(
    VM: Type[BaseVM],
    opcode_value: int,
    expected: Union[int, bytes],
) -> None: ...

def test_blockhash(
    VM: Type[BaseVM],
    val1: int,
    expected: bytes,
) -> None: ...

def test_mul(
    vm_class: Type[BaseVM],
    val1: int,
    val2: int,
    expected: int,
) -> None: ...

def test_exp(
    vm_class: Type[BaseVM],
    base: int,
    exponent: int,
    expected: int,
) -> None: ...

def test_shl(
    vm_class: Type[BaseVM],
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_shr(
    vm_class: Type[BaseVM],
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_sar(
    vm_class: Type[BaseVM],
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_extcodehash(
    vm_class: Type[BaseVM],
    address: str,
    expected: str,
) -> None: ...

def test_sstore(
    vm_class: Type[BaseVM],
    code: str,
    gas_used: int,
    refund: int,
    original: int,
) -> None: ...

def test_sstore_limit_2300(
    gas_supplied: int,
    success: bool,
    gas_used: int,
    refund: int,
) -> None: ...

def test_chainid(
    vm_class: Type[BaseVM],
    chain_id: int,
    expected_result: Union[int, Type[ValidationError]],
) -> None: ...

def test_balance(
    vm_class: Type[BaseVM],
    code: str,
    expect_exception: Optional[Type[Exception]],
    expect_gas_used: int,
) -> None: ...

def test_gas_costs(
    vm_class: Type[BaseVM],
    code: str,
    expect_gas_used: int,
) -> None: ...

def test_access_list_gas_costs(
    vm_class: Type[BaseVM],
    code: str,
    expect_gas_used: int,
    access_list: List[Tuple[bytes, List[int]]],
) -> None: ...

def test_eip2929_gas_by_cache_warmth(
    vm_class: Type[BaseVM],
    bytecode_hex: str,
    expect_gas_used: int,
) -> None: ...

def test_blake2b_f_compression(
    vm_class: Type[BaseVM],
    input_hex: str,
    output_hex: str,
    expect_exception: Optional[Type[Exception]],
) -> None: ...

def test_selfdestruct_does_not_issue_deprecation_warning_pre_shanghai(
    vm_class: Type[BaseVM],
) -> None: ...

def test_selfdestruct_issues_deprecation_warning_for_shanghai_and_later(
    vm_class: Type[BaseVM],
) -> None: ...