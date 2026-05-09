from typing import (
    Any,
    Callable,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    Dict,
    Set,
)
from eth2spec.typing import (
    Bytes32,
    Slot,
    Epoch,
    Root,
    BLSSignature,
    Domain,
    Gwei,
    CommitteeIndex,
    ValidatorIndex,
    ParticipationFlags,
    ForkDigest,
    Version,
    DomainType,
)
from eth2spec.beacon.types.blocks import (
    BeaconBlock,
    SignedBeaconBlock,
    BeaconBlockBody,
)
from eth2spec.beacon.types.states import BeaconState
from eth2spec.beacon.types.attestations import Attestation
from eth2spec.beacon.types.attester_slashings import AttesterSlashing
from eth2spec.beacon.types.pow_chain_block import PowChainBlock

class BlobData(NamedTuple):
    blobs: List[bytes]
    proofs: List[bytes]

def with_blob_data(spec: Any, blob_data: BlobData, func: Callable) -> Generator[Any, None, None]: ...

def get_anchor_root(spec: Any, state: Any) -> bytes: ...

def tick_and_add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
    merge_block: bool = False,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Any: ...

def tick_and_add_block_with_data(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[Dict[str, Any]],
    blob_data: BlobData,
    valid: bool = True,
) -> Generator[Any, None, None]: ...

def add_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Attestation], None, None]: ...

def add_attestations(
    spec: Any,
    store: Any,
    attestations: Sequence[Attestation],
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Attestation], None, None]: ...

def tick_and_run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Any, None, None]: ...

def run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    is_from_block: bool = False,
    valid: bool = True,
) -> None: ...

def get_genesis_forkchoice_store(spec: Any, genesis_state: BeaconState) -> Any: ...

def get_genesis_forkchoice_store_and_block(
    spec: Any,
    genesis_state: BeaconState,
) -> Tuple[Any, SignedBeaconBlock]: ...

def get_block_file_name(block: SignedBeaconBlock) -> str: ...

def get_attestation_file_name(attestation: Attestation) -> str: ...

def get_attester_slashing_file_name(attester_slashing: AttesterSlashing) -> str: ...

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[Bytes32] = None) -> str: ...

def on_tick_and_append_step(
    spec: Any,
    store: Any,
    time: int,
    test_steps: List[Dict[str, Any]],
) -> None: ...

def run_on_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    valid: bool = True,
) -> None: ...

def add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Any: ...

def run_on_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: AttesterSlashing,
    valid: bool = True,
) -> None: ...

def add_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: AttesterSlashing,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
) -> Generator[Tuple[str, AttesterSlashing], None, None]: ...

def get_formatted_head_output(spec: Any, store: Any) -> Dict[str, Union[int, str]]: ...

def output_head_check(
    spec: Any,
    store: Any,
    test_steps: List[Dict[str, Any]],
) -> None: ...

def output_store_checks(
    spec: Any,
    store: Any,
    test_steps: List[Dict[str, Any]],
) -> None: ...

def apply_next_epoch_with_attestations(
    spec: Any,
    state: BeaconState,
    store: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable] = None,
    test_steps: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[BeaconState, Any, SignedBeaconBlock]: ...

def apply_next_slots_with_attestations(
    spec: Any,
    state: BeaconState,
    store: Any,
    slots: int,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    test_steps: List[Dict[str, Any]],
    participation_fn: Optional[Callable] = None,
) -> Tuple[BeaconState, Any, SignedBeaconBlock]: ...

def is_ready_to_justify(spec: Any, state: BeaconState) -> bool: ...

def find_next_justifying_slot(
    spec: Any,
    state: BeaconState,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable] = None,
) -> Tuple[List[SignedBeaconBlock], Slot]: ...

def get_pow_block_file_name(pow_block: PowChainBlock) -> str: ...

def add_pow_block(
    spec: Any,
    store: Any,
    pow_block: PowChainBlock,
    test_steps: List[Dict[str, Any]],
) -> Generator[Tuple[str, PowChainBlock], None, None]: ...