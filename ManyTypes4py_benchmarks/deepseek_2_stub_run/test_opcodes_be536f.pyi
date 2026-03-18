```python
from typing import Any, Optional, Tuple, List, Union
from eth.vm.forks import (
    BerlinVM,
    ByzantiumVM,
    ConstantinopleVM,
    FrontierVM,
    HomesteadVM,
    IstanbulVM,
    LondonVM,
    MuirGlacierVM,
    PetersburgVM,
    SpuriousDragonVM,
    TangerineWhistleVM,
)
from eth.vm.message import Message
from eth.vm.spoof import SpoofTransaction
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.vm.chain_context import ChainContext
from eth.consensus import ConsensusContext
from eth.exceptions import InvalidInstruction, VMError
from eth_utils import ValidationError

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

def setup_vm(
    vm_class: Any,
    chain_id: Optional[int] = None,
) -> Any: ...

def run_computation(
    vm: Any,
    create_address: Any,
    code: bytes,
    gas: int = 1000000,
    to: bytes = ...,
    transaction_sender: bytes = ...,
    data: bytes = ...,
    access_list: Optional[List[Any]] = None,
) -> Any: ...

def run_general_computation(
    vm_class: Any,
    create_address: Any = None,
    code: bytes = ...,
    chain_id: Optional[int] = None,
) -> Any: ...

def test_add(
    vm_class: Any,
    val1: int,
    val2: int,
    expected: int,
) -> None: ...

def test_base_fee() -> None: ...

def test_nullary_opcodes(
    VM: Any,
    opcode_value: int,
    expected: Any,
) -> None: ...

def test_blockhash(
    VM: Any,
    val1: int,
    expected: bytes,
) -> None: ...

def test_mul(
    vm_class: Any,
    val1: int,
    val2: int,
    expected: int,
) -> None: ...

def test_exp(
    vm_class: Any,
    base: int,
    exponent: int,
    expected: int,
) -> None: ...

def test_shl(
    vm_class: Any,
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_shr(
    vm_class: Any,
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_sar(
    vm_class: Any,
    val1: str,
    val2: str,
    expected: str,
) -> None: ...

def test_extcodehash(
    vm_class: Any,
    address: str,
    expected: str,
) -> None: ...

def test_sstore(
    vm_class: Any,
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
    vm_class: Any,
    chain_id: int,
    expected_result: Union[int, ValidationError],
) -> None: ...

def test_balance(
    vm_class: Any,
    code: bytes,
    expect_exception: Any,
    expect_gas_used: int,
) -> None: ...

def test_gas_costs(
    vm_class: Any,
    code: bytes,
    expect_gas_used: int,
) -> None: ...

def test_access_list_gas_costs(
    vm_class: Any,
    code: bytes,
    expect_gas_used: int,
    access_list: List[Any],
) -> None: ...

def test_eip2929_gas_by_cache_warmth(
    vm_class: Any,
    bytecode_hex: str,
    expect_gas_used: int,
) -> None: ...

def test_blake2b_f_compression(
    vm_class: Any,
    input_hex: str,
    output_hex: str,
    expect_exception: Any,
) -> None: ...

def test_selfdestruct_does_not_issue_deprecation_warning_pre_shanghai(
    vm_class: Any,
) -> None: ...

def test_selfdestruct_issues_deprecation_warning_for_shanghai_and_later(
    vm_class: Any,
) -> None: ...
```