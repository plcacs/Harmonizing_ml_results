from typing import Any, Optional, List, Sequence, Union
from eth2spec.phase0 import spec as spec_phase0
from eth2spec.altair import spec as spec_altair
from eth2spec.bellatrix import spec as spec_bellatrix
from eth2spec.capella import spec as spec_capella
from eth2spec.deneb import spec as spec_deneb
from eth2spec.electra import spec as spec_electra
from trie import HexaryTrie
from rlp import sedes
from random import Random
from eth2spec.utils.ssz.ssz_typing import Bytes32, ByteVector, ByteList, Hash32, Root, Gwei, uint64, ValidatorIndex
from eth2spec.test.helpers.keys import privkeys

spec = Union[
    spec_phase0.BeaconState,
    spec_altair.BeaconState,
    spec_bellatrix.BeaconState,
    spec_capella.BeaconState,
    spec_deneb.BeaconState,
    spec_electra.BeaconState
]

def get_execution_payload_header(spec: Any, execution_payload: Any) -> Any: ...

def compute_trie_root_from_indexed_data(data: Sequence[bytes]) -> bytes: ...

def compute_requests_hash(block_requests: Sequence[bytes]) -> bytes: ...

def compute_el_header_block_hash(
    spec: Any,
    payload_header: Any,
    transactions_trie_root: bytes,
    withdrawals_trie_root: Optional[bytes] = None,
    parent_beacon_block_root: Optional[bytes] = None,
    requests_hash: Optional[bytes] = None
) -> Hash32: ...

def get_withdrawal_rlp(withdrawal: Any) -> bytes: ...

def get_deposit_request_rlp_bytes(deposit_request: Any) -> bytes: ...

def get_withdrawal_request_rlp_bytes(withdrawal_request: Any) -> bytes: ...

def get_consolidation_request_rlp_bytes(consolidation_request: Any) -> bytes: ...

def compute_el_block_hash_with_new_fields(
    spec: Any,
    payload: Any,
    parent_beacon_block_root: bytes,
    requests_hash: bytes
) -> Hash32: ...

def compute_el_block_hash(spec: Any, payload: Any, pre_state: Any) -> Hash32: ...

def compute_el_block_hash_for_block(spec: Any, block: Any) -> Hash32: ...

def build_empty_post_eip7732_execution_payload_header(spec: Any, state: Any) -> Optional[Any]: ...

def build_empty_signed_execution_payload_header(spec: Any, state: Any) -> Optional[Any]: ...

def build_empty_execution_payload(
    spec: Any,
    state: Any,
    randao_mix: Optional[bytes] = None
) -> Any: ...

def build_randomized_execution_payload(spec: Any, state: Any, rng: Random) -> Any: ...

def build_state_with_incomplete_transition(spec: Any, state: Any) -> Any: ...

def build_state_with_complete_transition(spec: Any, state: Any) -> Any: ...

def build_state_with_execution_payload_header(spec: Any, state: Any, execution_payload_header: Any) -> Any: ...

def get_random_tx(rng: Random) -> bytes: ...