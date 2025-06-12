import pytest
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast
from eth_utils import ValidationError, decode_hex, encode_hex, hexstr_if_str, int_to_big_endian, to_bytes, to_canonical_address
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
from eth.vm.forks import BerlinVM, ByzantiumVM, ConstantinopleVM, FrontierVM, HomesteadVM, IstanbulVM, LondonVM, MuirGlacierVM, PetersburgVM, SpuriousDragonVM, TangerineWhistleVM
from eth.vm.forks.cancun.computation import CancunComputation
from eth.vm.message import Message
from eth.vm.spoof import SpoofTransaction
from eth.vm.computation import BaseComputation
from eth.vm.base import VM

NORMALIZED_ADDRESS_A: str = '0x0f572e5295c57f15886f9b263e2f6d2d6c7b5ec6'
NORMALIZED_ADDRESS_B: str = '0xcd1722f3947def4cf144679da39c4c32bdc35681'
ADDRESS_WITH_CODE: Tuple[str, bytes] = ('0xddd722f3947def4cf144679da39c4c32bdc35681', b'pseudocode')
EMPTY_ADDRESS_IN_STATE: str = NORMALIZED_ADDRESS_A
ADDRESS_NOT_IN_STATE: str = NORMALIZED_ADDRESS_B
ADDRESS_WITH_JUST_BALANCE: str = '0x0000000000000000000000000000000000000001'
CANONICAL_ADDRESS_A: bytes = to_canonical_address('0x0f572e5295c57f15886f9b263e2f6d2d6c7b5ec6')
CANONICAL_ADDRESS_B: bytes = to_canonical_address('0xcd1722f3947def4cf144679da39c4c32bdc35681')
CANONICAL_ADDRESS_C: bytes = b'\xee' * 20
CANONICAL_ZERO_ADDRESS: bytes = b'\x00' * 20

def assemble(*codes: Union[int, str, bytes]) -> bytes:
    return b''.join((hexstr_if_str(to_bytes, element) for element in codes))

def setup_vm(vm_class: Type[VM], chain_id: Optional[int] = None) -> VM:
    db = AtomicDB()
    chain_context = ChainContext(chain_id)
    genesis_header = vm_class.create_genesis_header(difficulty=constants.GENESIS_DIFFICULTY if vm_class not in POS_MAINNET_VMS else 0, timestamp=0)
    return vm_class(genesis_header, ChainDB(db), chain_context, ConsensusContext(db))

def run_computation(
    vm: VM,
    create_address: Optional[bytes],
    code: bytes,
    gas: int = 1000000,
    to: bytes = CANONICAL_ADDRESS_A,
    transaction_sender: bytes = b'\x11' * 20,
    data: bytes = b'',
    access_list: Optional[List[Tuple[bytes, List[int]]]] = None
) -> BaseComputation:
    executor = vm.state.get_transaction_executor()
    message = Message(to=to, sender=CANONICAL_ADDRESS_B, create_address=create_address, value=0, data=data, code=code, gas=gas)
    if access_list is not None:
        txn_builder = vm.get_transaction_builder()
        unsigned_transaction = txn_builder.new_unsigned_access_list_transaction(vm.chain_context.chain_id, nonce=2, gas_price=1, gas=gas, to=to, value=3, data=data, access_list=access_list)
    else:
        unsigned_transaction = vm.create_unsigned_transaction(nonce=2, gas_price=1, gas=gas, to=to, value=3, data=data)
    transaction = SpoofTransaction(unsigned_transaction, from_=transaction_sender)
    return executor.build_computation(message, transaction)

def run_general_computation(vm_class: Type[VM], create_address: Optional[bytes] = None, code: bytes = b'', chain_id: Optional[int] = None) -> BaseComputation:
    vm = setup_vm(vm_class, chain_id=chain_id)
    vm.state.touch_account(decode_hex(EMPTY_ADDRESS_IN_STATE))
    vm.state.set_code(decode_hex(ADDRESS_WITH_CODE[0]), ADDRESS_WITH_CODE[1])
    vm.state.set_balance(decode_hex(ADDRESS_WITH_JUST_BALANCE), 1)
    return run_computation(vm, create_address, code)

@pytest.mark.parametrize('vm_class, val1, val2, expected', ((ByzantiumVM, 2, 4, 6), (SpuriousDragonVM, 2, 4, 6), (TangerineWhistleVM, 2, 4, 6), (HomesteadVM, 2, 4, 6), (FrontierVM, 2, 4, 6)))
def test_add(vm_class: Type[VM], val1: int, val2: int, expected: int) -> None:
    computation = run_general_computation(vm_class)
    computation.stack_push_int(val1)
    computation.stack_push_int(val2)
    computation.opcodes[opcode_values.ADD](computation)
    result = computation.stack_pop1_int()
    assert result == expected

def test_base_fee() -> None:
    computation = run_general_computation(LondonVM)
    computation.opcodes[opcode_values.BASEFEE](computation)
    result = computation.stack_pop1_any()
    assert result == 10 ** 9

@pytest.mark.parametrize('opcode_value, expected', ((opcode_values.COINBASE, b'\x00' * 20), (opcode_values.NUMBER, 0), (opcode_values.DIFFICULTY, 17179869184), (opcode_values.GASLIMIT, 5000)))
def test_nullary_opcodes(VM: Type[VM], opcode_value: int, expected: Any) -> None:
    computation = run_general_computation(VM)
    computation.opcodes[opcode_value](computation)
    result = computation.stack_pop1_any()
    assert result == expected

@pytest.mark.parametrize('val1, expected', ((0, b''), (1, b''), (255, b''), (256, b'')))
def test_blockhash(VM: Type[VM], val1: int, expected: bytes) -> None:
    computation = run_general_computation(VM)
    computation.stack_push_int(val1)
    computation.opcodes[opcode_values.BLOCKHASH](computation)
    result = computation.stack_pop1_any()
    assert result == expected

@pytest.mark.parametrize('vm_class, val1, val2, expected', ((ByzantiumVM, 2, 2, 4), (SpuriousDragonVM, 2, 2, 4), (TangerineWhistleVM, 2, 2, 4), (HomesteadVM, 2, 2, 4), (FrontierVM, 2, 2, 4)))
def test_mul(vm_class: Type[VM], val1: int, val2: int, expected: int) -> None:
    computation = run_general_computation(vm_class)
    computation.stack_push_int(val1)
    computation.stack_push_int(val2)
    computation.opcodes[opcode_values.MUL](computation)
    result = computation.stack_pop1_int()
    assert result == expected

@pytest.mark.parametrize('vm_class, base, exponent, expected', ((ByzantiumVM, 0, 1, 0), (ByzantiumVM, 0, 0, 1), (SpuriousDragonVM, 0, 1, 0), (SpuriousDragonVM, 0, 0, 1), (TangerineWhistleVM, 0, 1, 0), (TangerineWhistleVM, 0, 0, 1), (HomesteadVM, 0, 1, 0), (HomesteadVM, 0, 0, 1), (FrontierVM, 0, 1, 0), (FrontierVM, 0, 0, 1)))
def test_exp(vm_class: Type[VM], base: int, exponent: int, expected: int) -> None:
    computation = run_general_computation(vm_class)
    computation.stack_push_int(exponent)
    computation.stack_push_int(base)
    computation.opcodes[opcode_values.EXP](computation)
    result = computation.stack_pop1_int()
    assert result == expected

@pytest.mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x01', '0x0000000000000000000000000000000000000000000000000000000000000002'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0xff', '0x8000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x0100', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x0101', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x00', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x01', '0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0xff', '0x8000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x0100', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000000', '0x01', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x01', '0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe')))
def test_shl(vm_class: Type[VM], val1: str, val2: str, expected: str) -> None:
    computation = run_general_computation(vm_class)
    computation.stack_push_bytes(decode_hex(val1))
    computation.stack_push_bytes(decode_hex(val2))
    computation.opcodes[opcode_values.SHL](computation)
    result = computation.stack_pop1_int()
    assert encode_hex(pad32(int_to_big_endian(result))) == expected

@pytest.mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x01', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x01', '0x4000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0xff', '0x0000000000000000000000000000000000000000000000000000000000000001'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x0100', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x0101', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x00', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x01', '0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0xff', '0x0000000000000000000000000000000000000000000000000000000000000001'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x0100', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000000', '0x01', '0x0000000000000000000000000000000000000000000000000000000000000000')))
def test_shr(vm_class: Type[VM], val1: str, val2: str, expected: str) -> None:
    computation = run_general_computation(vm_class)
    computation.stack_push_bytes(decode_hex(val1))
    computation.stack_push_bytes(decode_hex(val2))
    computation.opcodes[opcode_values.SHR](computation)
    result = computation.stack_pop1_int()
    assert encode_hex(pad32(int_to_big_endian(result))) == expected

@pytest.mark.parametrize('vm_class, val1, val2, expected', ((ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x00', '0x0000000000000000000000000000000000000000000000000000000000000001'), (ConstantinopleVM, '0x0000000000000000000000000000000000000000000000000000000000000001', '0x01', '0x0000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x01', '0xc000000000000000000000000000000000000000000000000000000000000000'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0xff', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x0100', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0x8000000000000000000000000000000000000000000000000000000000000000', '0x0101', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0x00', '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'), (ConstantinopleVM, '0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', '0