from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set, Generator, Any, List
from eth_utils import encode_hex
from eth2spec.utils.ssz.ssz_typing import Bytes32
from eth2spec.test.helpers.fork_choice import add_block


class PayloadStatusV1StatusAlias(Enum):
    NOT_VALIDATED = 'NOT_VALIDATED'
    INVALIDATED = 'INVALIDATED'


class PayloadStatusV1Status(Enum):
    VALID = 'VALID'
    INVALID = 'INVALID'
    SYNCING = 'SYNCING'
    ACCEPTED = 'ACCEPTED'
    INVALID_BLOCK_HASH = 'INVALID_BLOCK_HASH'

    @property
    def alias(self) -> PayloadStatusV1StatusAlias:
        if self.value in (self.SYNCING.value, self.ACCEPTED.value):
            return PayloadStatusV1StatusAlias.NOT_VALIDATED
        elif self.value in (self.INVALID.value, self.INVALID_BLOCK_HASH.value):
            return PayloadStatusV1StatusAlias.INVALIDATED


@dataclass
class PayloadStatusV1:
    status: PayloadStatusV1Status = PayloadStatusV1Status.VALID
    latest_valid_hash: Optional[Bytes32] = None
    validation_error: Optional[str] = None

    @property
    def formatted_output(self) -> Dict[str, Optional[str]]:
        return {
            'status': str(self.status.value),
            'latest_valid_hash': encode_hex(self.latest_valid_hash) if self.latest_valid_hash is not None else None,
            'validation_error': str(self.validation_error) if self.validation_error is not None else None
        }


class MegaStore:
    spec: Any
    fc_store: Any
    opt_store: Any
    block_payload_statuses: Dict[Bytes32, PayloadStatusV1]

    def __init__(self, spec: Any, fc_store: Any, opt_store: Any) -> None:
        self.spec = spec
        self.fc_store = fc_store
        self.opt_store = opt_store
        self.block_payload_statuses = {}


def get_optimistic_store(spec: Any, anchor_state: Any, anchor_block: Any) -> Any:
    assert anchor_block.state_root == anchor_state.hash_tree_root()
    opt_store = spec.OptimisticStore(optimistic_roots=set(), head_block_root=anchor_block.hash_tree_root())
    anchor_block_root = anchor_block.hash_tree_root()
    opt_store.blocks[anchor_block_root] = anchor_block.copy()
    opt_store.block_states[anchor_block_root] = anchor_state.copy()
    return opt_store


def get_valid_flag_value(status: PayloadStatusV1Status) -> bool:
    if status == PayloadStatusV1Status.VALID:
        return True
    elif status.alias == PayloadStatusV1StatusAlias.NOT_VALIDATED:
        return True
    else:
        return False


def add_optimistic_block(
    spec: Any,
    mega_store: MegaStore,
    signed_block: Any,
    test_steps: List[Any],
    payload_status: Optional[PayloadStatusV1] = None,
    status: PayloadStatusV1Status = PayloadStatusV1Status.SYNCING
) -> Generator[Any, None, None]:
    """
    Add a block with optimistic sync logic

    ``valid`` indicates if the given ``signed_block.message.body.execution_payload`` is valid/invalid
    from ``verify_and_notify_new_payload`` method response.
    """
    block = signed_block.message
    block_root = block.hash_tree_root()
    el_block_hash = block.body.execution_payload.block_hash
    if payload_status is None:
        payload_status = PayloadStatusV1(status=status)
        if payload_status.status == PayloadStatusV1Status.VALID:
            payload_status.latest_valid_hash = el_block_hash
    mega_store.block_payload_statuses[block_root] = payload_status
    test_steps.append({
        'block_hash': encode_hex(el_block_hash),
        'payload_status': payload_status.formatted_output
    })
    valid = get_valid_flag_value(payload_status.status)
    if payload_status.status == PayloadStatusV1Status.INVALID:
        assert payload_status.latest_valid_hash is not None
        current_block = block
        while el_block_hash != payload_status.latest_valid_hash and el_block_hash != spec.Bytes32():
            current_block_root = current_block.hash_tree_root()
            assert current_block_root in mega_store.block_payload_statuses
            mega_store.block_payload_statuses[current_block_root].status = PayloadStatusV1Status.INVALID
            current_block = mega_store.fc_store.blocks[current_block.parent_root]
            el_block_hash = current_block.body.execution_payload.block_hash
    yield from add_block(spec, mega_store.fc_store, signed_block, valid=valid, test_steps=test_steps, is_optimistic=True)
    is_optimistic_candidate = spec.is_optimistic_candidate_block(
        mega_store.opt_store,
        current_slot=spec.get_current_slot(mega_store.fc_store),
        block=signed_block.message
    )
    if is_optimistic_candidate:
        mega_store.opt_store.optimistic_roots.add(block_root)
        mega_store.opt_store.blocks[block_root] = signed_block.message.copy()
        if not is_invalidated(mega_store, block_root):
            mega_store.opt_store.block_states[block_root] = mega_store.fc_store.block_states[block_root].copy()
    clean_up_store(mega_store)
    mega_store.opt_store.head_block_root = get_opt_head_block_root(spec, mega_store)
    test_steps.append({
        'checks': {
            'head': get_formatted_optimistic_head_output(mega_store)
        }
    })


def get_opt_head_block_root(spec: Any, mega_store: MegaStore) -> Bytes32:
    """
    Copied and modified from fork-choice spec `get_head` function.
    """
    store = mega_store.fc_store
    blocks = spec.get_filtered_block_tree(store)
    head = store.justified_checkpoint.root
    while True:
        children = [
            root for root in blocks.keys()
            if blocks[root].parent_root == head and (not is_invalidated(mega_store, root))
        ]
        if len(children) == 0:
            return head
        head = max(children, key=lambda root: (spec.get_weight(store, root), root))


def is_invalidated(mega_store: MegaStore, block_root: Bytes32) -> bool:
    if block_root in mega_store.block_payload_statuses:
        return mega_store.block_payload_statuses[block_root].status.alias == PayloadStatusV1StatusAlias.INVALIDATED
    else:
        return False


def get_formatted_optimistic_head_output(mega_store: MegaStore) -> Dict[str, Any]:
    head = mega_store.opt_store.head_block_root
    slot = mega_store.fc_store.blocks[head].slot
    return {
        'slot': int(slot),
        'root': encode_hex(head)
    }


def clean_up_store(mega_store: MegaStore) -> None:
    """
    Remove invalidated blocks
    """
    ...
