from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from eth2spec.base.types import (
    BeaconState,
    Bytes32,
    ByteList,
    ByteVector,
    Gwei,
    Hash32,
    List,
    Root,
    Slot,
    ValidatorIndex,
)
from eth2spec.test.helpers.keys import privkeys
from eth2spec.debug.random_value import get_random_bytes_list
from eth2spec.test.helpers.withdrawals import get_expected_withdrawals
from eth2spec.test.helpers.forks import (
    is_post_capella,
    is_post_deneb,
    is_post_electra,
    is_post_eip7732,
)

def get_execution_payload_header(spec: Any, execution_payload: Any) -> spec.ExecutionPayloadHeader: ...

def compute_trie_root_from_indexed_data(data: List[Any]) -> bytes: ...

def compute_requests_hash(block_requests: List[bytes]) -> bytes: ...

def compute_el_header_block_hash(
    spec: Any,
    payload_header: spec.ExecutionPayloadHeader,
    transactions_trie_root: bytes,
    withdrawals_trie_root: Optional[bytes] = None,
    parent_beacon_block_root: Optional[bytes] = None,
    requests_hash: Optional[bytes] = None,
) -> spec.Hash32: ...

def get_withdrawal_rlp(withdrawal: Any) -> bytes: ...

def get_deposit_request_rlp_bytes(deposit_request: Any) -> bytes: ...

def get_withdrawal_request_rlp_bytes(withdrawal_request: Any) -> bytes: ...

def get_consolidation_request_rlp_bytes(consolidation_request: Any) -> bytes: ...

def compute_el_block_hash_with_new_fields(
    spec: Any,
    payload: spec.ExecutionPayload,
    parent_beacon_block_root: Optional[bytes],
    requests_hash: Optional[bytes],
) -> spec.Hash32: ...

def compute_el_block_hash(spec: Any, payload: spec.ExecutionPayload, pre_state: BeaconState) -> spec.Hash32: ...

def compute_el_block_hash_for_block(spec: Any, block: Any) -> spec.Hash32: ...

def build_empty_post_eip7732_execution_payload_header(spec: Any, state: BeaconState) -> spec.ExecutionPayloadHeader: ...

def build_empty_signed_execution_payload_header(spec: Any, state: BeaconState) -> spec.SignedExecutionPayloadHeader: ...

def build_empty_execution_payload(
    spec: Any,
    state: BeaconState,
    randao_mix: Optional[Bytes32] = None,
) -> spec.ExecutionPayload: ...

def build_randomized_execution_payload(spec: Any, state: BeaconState, rng: Any) -> spec.ExecutionPayload: ...

def build_state_with_incomplete_transition(spec: Any, state: BeaconState) -> BeaconState: ...

def build_state_with_complete_transition(spec: Any, state: BeaconState) -> BeaconState: ...

def build_state_with_execution_payload_header(
    spec: Any,
    state: BeaconState,
    execution_payload_header: spec.ExecutionPayloadHeader,
) -> BeaconState: ...

def get_random_tx(rng: Any) -> spec.ByteList: ...