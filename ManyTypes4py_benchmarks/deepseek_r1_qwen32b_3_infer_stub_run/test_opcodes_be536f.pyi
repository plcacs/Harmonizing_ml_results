from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from pytest import (
    fixture,
    mark,
    raises,
)
from eth_utils import (
    ValidationError,
    decode_hex,
    encode_hex,
    hexstr_if_str,
    int_to_big_endian,
    to_bytes,
    to_canonical_address,
)
from eth import constants
from eth._utils.address import force_bytes_to_address
from eth._utils.padding import pad32
from eth.chains.mainnet import MAINNET_VMS, POS_MAINNET_VMS
from eth.consensus import ConsensusContext
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.exceptions import InvalidInstruction, VMError
from eth.vm import opcode_values
from eth.vm.chain_context import ChainContext
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
from eth.vm.forks.cancun.computation import CancunComputation
from eth.vm.message import Message
from eth.vm.spoof import SpoofTransaction

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
def setup_vm(vm_class: Type, chain_id: Optional[int] = None) -> vm_class: ...
def run_computation(
    vm: Any,
    create_address: bytes,
    code: bytes,
    gas: int = 1000000,
    to: bytes = CANONICAL_ADDRESS_A,
    transaction_sender: bytes = b'\x11' * 20,
    data: bytes = b'',
    access_list: Optional[List[Tuple[bytes, List[int]]]] = None,
) -> Any: ...
def run_general_computation(
    vm_class: Type,
    create_address: Optional[bytes] = None,
    code: bytes = b'',
    chain_id: Optional[int] = None,
) -> Any: ...

@mark.parametrize('vm_class, val1, val2, expected', ((ByzantiumVM, 2, 4, 6), (SpuriousDragonVM, 2, 4, 6), (TangerineWhistleVM, 2, 4, 6), (HomesteadVM, 2, 4, 6), (FrontierVM, 2, 4, 6)))
def test_add(vm_class: Type, val1: int, val2: int, expected: int) -> None: ...

def test_base_fee() -> None: ...

@mark.parametrize('opcode_value, expected', ((opcode_values.COINBASE, b'\x00' * 20), (opcode_values.NUMBER, 0), (opcode_values.DIFFICULTY, 17179869184), (opcode_values.GASLIMIT, 5000)))
def test_nullary_opcodes(VM: Type, opcode_value: int, expected: Union[bytes, int]) -> None: ...

@mark.parametrize('val1, expected', ((0, b''), (1, b''), (255, b''), (256, b'')))
def test_blockhash(VM: Type, val1: int, expected: bytes) -> None: ...

@mark.parametrize('vm_class, val1, val2, expected', ((ByzantiumVM, 2, 2, 4), (SpuriousDragonVM, 2, 2, 4), (TangerineWhistleVM, 2, 2, 4), (HomesteadVM, 2, 2, 4), (FrontierVM, 2, 2, 4)))
def test_mul(vm_class: Type, val1: int, val2: int, expected: int) -> None: ...

@mark.parametrize('vm_class, base, exponent, expected', ((ByzantiumVM, 0, 1, 0), (ByzantiumVM, 0, 0, 1), (SpuriousDragonVM, 0, 1, 0), (SpuriousDragonVM, 0, 0, 1), (TangerineWhistleVM, 0, 1, 0), (TangerineWhistleVM, 0, 0, 1), (HomesteadVM, 0, 1, 0), (HomesteadVM, 0, 0, 1), (FrontierVM, 0, 1, 0), (FrontierVM, 0, 0, 1)))
def test_exp(vm_class: Type, base: int, exponent: int, expected: int) -> None: ...

@mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), ...))
def test_shl(vm_class: Type, val1: str, val2: str, expected: str) -> None: ...

@mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), ...))
def test_shr(vm_class: Type, val1: str, val2: str, expected: str) -> None: ...

@mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), ...))
def test_sar(vm_class: Type, val1: str, val2: str, expected: str) -> None: ...

@mark.parametrize('vm_class, address, expected', ((ConstantinopleVM, ADDRESS_NOT_IN_STATE, '0x0000000000000000000000000000000000000000000000000000000000000000'), ...))
def test_extcodehash(vm_class: Type, address: str, expected: str) -> None: ...

@mark.parametrize('vm_class, code, gas_used, refund, original', ((ByzantiumVM, '0x60006000556000600055', 10012, 0, 0), ...))
def test_sstore(vm_class: Type, code: str, gas_used: int, refund: int, original: int) -> None: ...

@mark.parametrize('gas_supplied, success, gas_used, refund', ((2306, False, 2306, 0), (2307, True, 806, 0)))
def test_sstore_limit_2300(gas_supplied: int, success: bool, gas_used: int, refund: int) -> None: ...

@mark.parametrize('vm_class', (IstanbulVM, MuirGlacierVM, BerlinVM, LondonVM))
@mark.parametrize('chain_id, expected_result', ((86, 86), (0, 0), (-1, ValidationError), (2 ** 256 - 1, 2 ** 256 - 1), (2 ** 256, ValidationError)))
def test_chainid(vm_class: Type, chain_id: int, expected_result: Union[int, Type[ValidationError]]) -> None: ...

@mark.parametrize('vm_class, code, expect_exception, expect_gas_used', ((ConstantinopleVM, assemble(opcode_values.PUSH20, CANONICAL_ADDRESS_B, opcode_values.BALANCE), None, 3 + 400), ...))
def test_balance(vm_class: Type, code: bytes, expect_exception: Optional[Type[Exception]], expect_gas_used: int) -> None: ...

@mark.parametrize('vm_class, code, expect_gas_used', ((ConstantinopleVM, assemble(opcode_values.PUSH1, 0, opcode_values.SLOAD), 3 + 200), ...))
def test_gas_costs(vm_class: Type, code: bytes, expect_gas_used: int) -> None: ...

@mark.parametrize('vm_class, code, expect_gas_used, access_list', ((BerlinVM, assemble(opcode_values.PUSH20, CANONICAL_ADDRESS_C, opcode_values.BALANCE), 3 + 2600, []), ...))
def test_access_list_gas_costs(vm_class: Type, code: bytes, expect_gas_used: int, access_list: List[Tuple[bytes, List[int]]]) -> None: ...

@mark.parametrize('vm_class', (BerlinVM, LondonVM))
@mark.parametrize('bytecode_hex, expect_gas_used', (('0x60013f5060023b506003315060f13f5060f23b5060f3315060f23f5060f33b5060f1315032315030315000', 8653), ...))
def test_eip2929_gas_by_cache_warmth(vm_class: Type, bytecode_hex: str, expect_gas_used: int) -> None: ...

@mark.parametrize('vm_class, input_hex, output_hex, expect_exception', ((PetersburgVM, '0000000048c9bdf267e6096a3ba7ca8485ae67bb2bf894fe72f36e3cf1361d5f3af54fa5d182e6ad7f520e511f6c3e2b8c68059b6bbd41fbabd9831f79217e1319cde05b61626300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000300000000000000000000000000000001', '', None), ...))
def test_blake2b_f_compression(vm_class: Type, input_hex: str, output_hex: str, expect_exception: Optional[Type[Exception]]) -> None: ...

@mark.parametrize('vm_class', MAINNET_VMS[:13])
def test_selfdestruct_does_not_issue_deprecation_warning_pre_shanghai(vm_class: Type) -> None: ...

@mark.parametrize('vm_class', MAINNET_VMS[13:])
def test_selfdestruct_issues_deprecation_warning_for_shanghai_and_later(vm_class: Type) -> None: ...