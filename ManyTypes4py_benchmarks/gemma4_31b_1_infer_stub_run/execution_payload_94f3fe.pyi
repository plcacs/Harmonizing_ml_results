from typing import Any, Optional, Protocol, Sequence, TypeVar, Union, overload
from hashlib import _sha256

# Define Protocols for the 'spec' and 'state' objects to provide concrete typing
# based on the usage patterns in the module.

class SpecProtocol(Protocol):
    ExecutionPayloadHeader: Any
    ExecutionPayload: Any
    SignedExecutionPayloadHeader: Any
    Root: Any
    ValidatorIndex: Any
    Slot: Any
    Gwei: Any
    Hash32: Any
    ExecutionAddress: Any
    Bytes32: Any
    ByteVector: Any
    ByteList: Any
    List: Any
    Transaction: Any
    MAX_TRANSACTIONS_PER_PAYLOAD: int
    BYTES_PER_LOGS_BLOOM: int
    MAX_EXTRA_DATA_BYTES: int
    def hash_tree_root(self, data: Any) -> Any: ...
    def get_execution_payload_header_signature(self, state: Any, message: Any, privkey: bytes) -> Any: ...
    def compute_timestamp_at_slot(self, state: Any, slot: int) -> int: ...
    def get_randao_mix(self, state: Any, epoch: int) -> bytes: ...
    def get_current_epoch(self, state: Any) -> int: ...
    def get_execution_requests_list(self, requests: Any) -> list[bytes]: ...
    def is_merge_transition_complete(self, state: Any) -> bool: ...
    def uint64(self, value: int) -> Any: ...

class StateProtocol(Protocol):
    latest_block_header: Any
    latest_block_hash: bytes
    slot: int
    latest_execution_payload_header: Any
    def copy(self) -> 'StateProtocol': ...
    def hash_tree_root(self) -> bytes: ...

class ExecutionPayloadProtocol(Protocol):
    parent_hash: bytes
    block_hash: bytes
    gas_limit: int
    fee_recipient: bytes
    state_root: bytes
    receipts_root: bytes
    logs_bloom: bytes
    prev_randao: bytes
    block_number: int
    gas_used: int
    timestamp: int
    extra_data: bytes
    base_fee_per_gas: int
    transactions: Sequence[Any]
    withdrawals: Optional[Sequence[Any]]
    blob_gas_used: Optional[int]
    excess_blob_gas: Optional[int]

class BlockProtocol(Protocol):
    parent_root: bytes
    body: Any

class BlockBodyProtocol(Protocol):
    execution_requests: Any
    execution_payload: ExecutionPayloadProtocol

class RNGProtocol(Protocol):
    def randint(self, a: int, b: int) -> int: ...

def get_execution_payload_header(spec: SpecProtocol, execution_payload: ExecutionPayloadProtocol) -> Any: ...

def compute_trie_root_from_indexed_data(data: Sequence[Any]) -> bytes: ...

def compute_requests_hash(block_requests: Sequence[bytes]) -> bytes: ...

def compute_el_header_block_hash(
    spec: SpecProtocol,
    payload_header: Any,
    transactions_trie_root: bytes,
    withdrawals_trie_root: Optional[bytes] = None,
    parent_beacon_block_root: Optional[bytes] = None,
    requests_hash: Optional[bytes] = None,
) -> Any: ...

def get_withdrawal_rlp(withdrawal: Any) -> bytes: ...

def get_deposit_request_rlp_bytes(deposit_request: Any) -> bytes: ...

def get_withdrawal_request_rlp_bytes(withdrawal_request: Any) -> bytes: ...

def get_consolidation_request_rlp_bytes(consolidation_request: Any) -> bytes: ...

def compute_el_block_hash_with_new_fields(
    spec: SpecProtocol,
    payload: ExecutionPayloadProtocol,
    parent_beacon_block_root: Optional[bytes],
    requests_hash: Optional[bytes],
) -> Any: ...

def compute_el_block_hash(spec: SpecProtocol, payload: ExecutionPayloadProtocol, pre_state: StateProtocol) -> Any: ...

def compute_el_block_hash_for_block(spec: SpecProtocol, block: BlockProtocol) -> Any: ...

def build_empty_post_eip7732_execution_payload_header(spec: SpecProtocol, state: StateProtocol) -> Optional[Any]: ...

def build_empty_signed_execution_payload_header(spec: SpecProtocol, state: StateProtocol) -> Optional[Any]: ...

def build_empty_execution_payload(
    spec: SpecProtocol, state: StateProtocol, randao_mix: Optional[bytes] = None
) -> ExecutionPayloadProtocol: ...

def build_randomized_execution_payload(spec: SpecProtocol, state: StateProtocol, rng: RNGProtocol) -> ExecutionPayloadProtocol: ...

def build_state_with_incomplete_transition(spec: SpecProtocol, state: StateProtocol) -> StateProtocol: ...

def build_state_with_complete_transition(spec: SpecProtocol, state: StateProtocol) -> StateProtocol: ...

def build_state_with_execution_payload_header(
    spec: SpecProtocol, state: StateProtocol, execution_payload_header: Any
) -> StateProtocol: ...

def get_random_tx(rng: RNGProtocol) -> bytes: ...