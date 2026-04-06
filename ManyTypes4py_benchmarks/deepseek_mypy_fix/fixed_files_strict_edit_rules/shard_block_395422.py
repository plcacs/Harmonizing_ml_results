from typing import Dict, List, Optional, Sequence, TypeVar
from eth2spec.test.helpers.block import get_state_and_beacon_parent_root_at_slot
from eth2spec.test.helpers.keys import privkeys
from eth2spec.utils import bls
from eth2spec.utils.bls import only_with_bls
from eth2spec.typing import SpecObject, BeaconState, Shard, Slot, BLSSignature, ShardBlock, SignedShardBlock, ShardTransition, ShardBlockBody

T = TypeVar('T')

@only_with_bls()
def sign_shard_block(spec: SpecObject, beacon_state: BeaconState, shard: Shard, block: SignedShardBlock, proposer_index: Optional[int] = None) -> None:
    slot: Slot = block.message.slot
    if proposer_index is None:
        proposer_index = spec.get_shard_proposer_index(beacon_state, slot, shard)
    privkey: int = privkeys[proposer_index]
    domain: bytes = spec.get_domain(beacon_state, spec.DOMAIN_SHARD_PROPOSAL, spec.compute_epoch_at_slot(slot))
    signing_root: bytes = spec.compute_signing_root(block.message, domain)
    block.signature: BLSSignature = bls.Sign(privkey, signing_root)

def build_shard_block(spec: SpecObject, beacon_state: BeaconState, shard: Shard, slot: Optional[Slot] = None, body: Optional[ShardBlockBody] = None, shard_parent_state: Optional[BeaconState] = None, signed: bool = False) -> SignedShardBlock:
    if shard_parent_state is None:
        shard_parent_state = beacon_state.shard_states[shard]
    if slot is None:
        slot = shard_parent_state.slot + 1
    if body is None:
        body = get_sample_shard_block_body(spec)
    beacon_state, beacon_parent_root = get_state_and_beacon_parent_root_at_slot(spec, beacon_state, slot)
    proposer_index: int = spec.get_shard_proposer_index(beacon_state, slot, shard)
    block: ShardBlock = spec.ShardBlock(shard_parent_root=shard_parent_state.latest_block_root, beacon_parent_root=beacon_parent_root, slot=slot, shard=shard, proposer_index=proposer_index, body=body)
    signed_block: SignedShardBlock = spec.SignedShardBlock(message=block)
    if signed:
        sign_shard_block(spec, beacon_state, shard, signed_block, proposer_index=proposer_index)
    return signed_block

def get_shard_transitions(spec: SpecObject, parent_beacon_state: BeaconState, shard_block_dict: Dict[Shard, Sequence[SignedShardBlock]]) -> List[ShardTransition]:
    shard_transitions: List[ShardTransition] = [spec.ShardTransition()] * spec.MAX_SHARDS
    on_time_slot: Slot = parent_beacon_state.slot + 1
    for shard, blocks in shard_block_dict.items():
        shard_transition: ShardTransition = spec.get_shard_transition(parent_beacon_state, shard, blocks)
        offset_slots: List[Slot] = spec.compute_offset_slots(spec.get_latest_slot_for_shard(parent_beacon_state, shard), on_time_slot)
        len_offset_slots: int = len(offset_slots)
        shard_transition = spec.get_shard_transition(parent_beacon_state, shard, blocks)
        if len(blocks) > 0:
            shard_block_root: bytes = blocks[-1].message.hash_tree_root()
            assert shard_transition.shard_states[len_offset_slots - 1].latest_block_root == shard_block_root
            assert shard_transition.shard_states[len_offset_slots - 1].slot == offset_slots[-1]
        shard_transitions[shard] = shard_transition
    return shard_transitions

def get_committee_index_of_shard(spec: SpecObject, state: BeaconState, slot: Slot, shard: Shard) -> Optional[int]:
    active_shard_count: int = spec.get_active_shard_count(state)
    committee_count: int = spec.get_committee_count_per_slot(state, spec.compute_epoch_at_slot(slot))
    start_shard: Shard = spec.get_start_shard(state, slot)
    for committee_index in range(committee_count):
        if (start_shard + committee_index) % active_shard_count == shard:
            return committee_index
    return None

def get_sample_shard_block_body(spec: SpecObject, is_max: bool = False) -> ShardBlockBody:
    size: int = spec.MAX_SHARD_BLOCK_SIZE if is_max else 128
    return b'V' * size
