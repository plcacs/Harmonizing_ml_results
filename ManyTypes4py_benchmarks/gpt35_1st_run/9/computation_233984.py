import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from cached_property import cached_property
from eth_typing import Address
from eth_utils import encode_hex, get_extended_debug_logger
from eth._utils.datatypes import Configurable
from eth._utils.numeric import ceil32
from eth.abc import CodeStreamAPI, ComputationAPI, GasMeterAPI, MemoryAPI, MessageAPI, OpcodeAPI, StackAPI, StateAPI, TransactionContextAPI
from eth.constants import GAS_MEMORY, GAS_MEMORY_QUADRATIC_DENOMINATOR
from eth.exceptions import Halt, VMError
from eth.typing import BytesOrView
from eth.validation import validate_canonical_address, validate_is_bytes, validate_uint256
from eth.vm.code_stream import CodeStream
from eth.vm.gas_meter import GasMeter
from eth.vm.logic.invalid import InvalidOpcode
from eth.vm.memory import Memory
from eth.vm.message import Message
from eth.vm.stack import Stack

def NO_RESULT(computation: ComputationAPI) -> None:
    ...

def memory_gas_cost(size_in_bytes: int) -> int:
    ...

class BaseComputation(ComputationAPI, Configurable):
    logger: Any = get_extended_debug_logger('eth.vm.computation.BaseComputation')
    state: Optional[StateAPI] = None
    msg: Optional[MessageAPI] = None
    transaction_context: Optional[TransactionContextAPI] = None
    code: Optional[CodeStreamAPI] = None
    children: Optional[List[ComputationAPI]] = None
    return_data: bytes = b''
    accounts_to_delete: Optional[List[Address]] = None
    beneficiaries: Optional[List[Address]] = None
    _memory: Optional[MemoryAPI] = None
    _stack: Optional[StackAPI] = None
    _gas_meter: Optional[GasMeterAPI] = None
    _error: Optional[VMError] = None
    _output: bytes = b''
    _log_entries: Optional[List[Tuple[int, Address, List[int], bytes]]] = None
    opcodes: Optional[Dict[int, Callable[[ComputationAPI], None]]] = None
    _precompiles: Optional[Dict[Address, Callable[[ComputationAPI], None]]] = None

    def __init__(self, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI) -> None:
        ...

    @classmethod
    def apply_message(cls, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI, parent_computation: Optional[ComputationAPI] = None) -> None:
        ...

    @classmethod
    def apply_create_message(cls, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI, parent_computation: Optional[ComputationAPI] = None) -> None:
        ...

    @property
    def is_origin_computation(self) -> bool:
        ...

    def prepare_child_message(self, gas: int, to: Address, value: int, data: bytes, code: bytes, **kwargs: Any) -> MessageAPI:
        ...

    def apply_child_computation(self, child_msg: MessageAPI) -> ComputationAPI:
        ...

    def generate_child_computation(self, child_msg: MessageAPI) -> ComputationAPI:
        ...

    def add_child_computation(self, child_computation: ComputationAPI) -> None:
        ...

    def get_gas_refund(self) -> int:
        ...

    def register_account_for_deletion(self, beneficiary: Address) -> None:
        ...

    def get_accounts_for_deletion(self) -> List[Address]:
        ...

    def get_self_destruct_beneficiaries(self) -> List[Address]:
        ...

    def add_log_entry(self, account: Address, topics: List[int], data: bytes) -> None:
        ...

    def get_raw_log_entries(self) -> Tuple[Tuple[int, Address, List[int], bytes]]:
        ...

    def get_log_entries(self) -> Tuple[Tuple[Address, List[int], bytes]]:
        ...

    @classmethod
    def apply_computation(cls, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI, parent_computation: Optional[ComputationAPI] = None) -> ComputationAPI:
        ...

    @property
    def is_success(self) -> bool:
        ...

    @property
    def is_error(self) -> bool:
        ...

    @property
    def error(self) -> VMError:
        ...

    @error.setter
    def error(self, value: VMError) -> None:
        ...

    def raise_if_error(self) -> None:
        ...

    @property
    def should_burn_gas(self) -> bool:
        ...

    @property
    def should_return_gas(self) -> bool:
        ...

    @property
    def should_erase_return_data(self) -> bool:
        ...

    def extend_memory(self, start_position: int, size: int) -> None:
        ...

    def memory_write(self, start_position: int, size: int, value: bytes) -> None:
        ...

    def memory_read_bytes(self, start_position: int, size: int) -> bytes:
        ...

    def memory_copy(self, destination: int, source: int, length: int) -> None:
        ...

    def get_gas_meter(self) -> GasMeterAPI:
        ...

    def consume_gas(self, amount: int, reason: str) -> None:
        ...

    def return_gas(self, amount: int) -> None:
        ...

    def refund_gas(self, amount: int) -> None:
        ...

    def get_gas_used(self) -> int:
        ...

    def get_gas_remaining(self) -> int:
        ...

    @classmethod
    def consume_initcode_gas_cost(cls, computation: ComputationAPI) -> None:
        ...

    def stack_swap(self, position: int) -> None:
        ...

    def stack_dup(self, position: int) -> None:
        ...

    @cached_property
    def stack_pop_ints(self) -> Callable[[int], List[int]]:
        ...

    @cached_property
    def stack_pop_bytes(self) -> Callable[[int], List[bytes]]:
        ...

    @cached_property
    def stack_pop_any(self) -> Callable[[int], List[Union[int, bytes]]]:
        ...

    @cached_property
    def stack_pop1_int(self) -> int:
        ...

    @cached_property
    def stack_pop1_bytes(self) -> bytes:
        ...

    @cached_property
    def stack_pop1_any(self) -> Union[int, bytes]:
        ...

    @cached_property
    def stack_push_int(self) -> Callable[[int], None]:
        ...

    @cached_property
    def stack_push_bytes(self) -> Callable[[bytes], None]:
        ...

    @property
    def output(self) -> bytes:
        ...

    @output.setter
    def output(self, value: bytes) -> None:
        ...

    @property
    def precompiles(self) -> Dict[Address, Callable[[ComputationAPI], None]]:
        ...

    @classmethod
    def get_precompiles(cls) -> Dict[Address, Callable[[ComputationAPI], None]]:
        ...

    def get_opcode_fn(self, opcode: int) -> Callable[[ComputationAPI], None]:
        ...

    def __enter__(self) -> 'BaseComputation':
        ...

    def __exit__(self, exc_type: Type[BaseException], exc_value: BaseException, traceback: TracebackType) -> Optional[bool]:
        ...
