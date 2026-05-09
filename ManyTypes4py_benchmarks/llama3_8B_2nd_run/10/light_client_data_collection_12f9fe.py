from typing import Any, Dict, List, Set
from dataclasses import dataclass
from eth_utils import encode_hex
from eth2spec.test.helpers.constants import ALTAIR
from eth2spec.test.helpers.fork_transition import transition_across_forks
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.light_client import compute_start_slot_at_sync_committee_period, get_sync_aggregate, latest_current_sync_committee_gindex, latest_finalized_root_gindex, latest_normalize_merkle_branch, upgrade_lc_header_to_new_spec, upgrade_lc_update_to_new_spec

def _next_epoch_boundary_slot(spec: Any, slot: int) -> int:
    epoch = spec.compute_epoch_at_slot(slot + spec.SLOTS_PER_EPOCH - 1)
    return spec.compute_start_slot_at_epoch(epoch)

@dataclass(frozen=True)
class BlockID(object):
    pass

def _block_to_block_id(block: Any) -> BlockID:
    return BlockID(slot=block.message.slot, root=block.message.hash_tree_root())

def _state_to_block_id(state: Any) -> BlockID:
    parent_header = state.latest_block_header.copy()
    parent_header.state_root = state.hash_tree_root()
    return BlockID(slot=parent_header.slot, root=parent_header.hash_tree_root())

def get_lc_bootstrap_block_id(bootstrap: Any) -> BlockID:
    return BlockID(slot=bootstrap.header.beacon.slot, root=bootstrap.header.beacon.hash_tree_root())

def get_lc_update_attested_block_id(update: Any) -> BlockID:
    return BlockID(slot=update.attested_header.beacon.slot, root=update.attested_header.beacon.hash_tree_root())

@dataclass
class ForkedBeaconState(object):
    pass

@dataclass
class ForkedSignedBeaconBlock(object):
    pass

@dataclass
class ForkedLightClientHeader(object):
    pass

@dataclass
class ForkedLightClientBootstrap(object):
    pass

@dataclass
class ForkedLightClientUpdate(object):
    pass

@dataclass
class ForkedLightClientFinalityUpdate(object):
    pass

@dataclass
class ForkedLightClientOptimisticUpdate(object):
    pass

@dataclass
class CachedLightClientData(object):
    pass

@dataclass
class LightClientDataCache(object):
    pass

@dataclass
class LightClientDataDB(object):
    pass

@dataclass
class LightClientDataStore(object):
    pass

@dataclass
class LightClientDataCollectionTest(object):
    pass

def get_ancestor_of_block_id(test: Any, bid: BlockID, slot: int) -> BlockID:
    try:
        block = test.blocks[bid.root]
        while True:
            if block.data.message.slot <= slot:
                return _block_to_block_id(block.data)
            block = test.blocks[block.data.message.parent_root]
    except KeyError:
        return None

def _block_id_at_finalized_slot(test: Any, slot: int) -> BlockID:
    while slot >= test.historical_tail_slot:
        try:
            return BlockID(slot=slot, root=test.finalized_block_roots[slot])
        except KeyError:
            slot = slot - 1
    return None

def _get_current_sync_committee_for_finalized_period(test: Any, period: int) -> Any:
    low_slot = max(test.historical_tail_slot, test.lc_data_store.spec.compute_start_slot_at_epoch(test.lc_data_store.spec.config.ALTAIR_FORK_EPOCH))
    if period < test.lc_data_store.spec.compute_sync_committee_period_at_slot(low_slot):
        return None
    period_start_slot = compute_start_slot_at_sync_committee_period(test.lc_data_store.spec, period)
    sync_committee_slot = max(period_start_slot, low_slot)
    bid = _block_id_at_finalized_slot(test, sync_committee_slot)
    if bid is None:
        return None
    block = test.blocks[bid.root]
    state = test.finalized_checkpoint_states[block.data.message.state_root]
    if sync_committee_slot > state.data.slot:
        state.spec, state.data, _ = transition_across_forks(state.spec, state.data, sync_committee_slot, phases=test.phases)
    assert is_post_altair(state.spec)
    return state.data.current_sync_committee

def _light_client_header_for_block(test: Any, block: Any) -> ForkedLightClientHeader:
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _light_client_header_for_block_id(test: Any, bid: BlockID) -> ForkedLightClientHeader:
    block = test.blocks[bid.root]
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _sync_aggregate_for_block_id(test: Any, bid: BlockID) -> Any:
    block = test.blocks[bid.root]
    if not is_post_altair(block.spec):
        return None
    return block.data.message.body.sync_aggregate

def _get_light_client_data(lc_data_store: LightClientDataStore, bid: BlockID) -> CachedLightClientData:
    try:
        return lc_data_store.cache.data[bid]
    except KeyError:
        raise ValueError('Trying to get light client data that was not cached')

def _cache_lc_data(lc_data_store: LightClientDataStore, spec: Any, state: Any, bid: BlockID, current_period_best_update: ForkedLightClientUpdate, latest_signature_slot: int) -> None:
    cached_data = CachedLightClientData(current_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.current_sync_committee_gindex_at_slot(state.slot)), latest_current_sync_committee_gindex(lc_data_store.spec)), next_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.next_sync_committee_gindex_at_slot(state.slot)), latest_next_sync_committee_gindex(lc_data_store.spec)), finalized_slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch), finality_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.finalized_root_gindex_at_slot(state.slot)), latest_finalized_root_gindex(lc_data_store.spec)), current_period_best_update=current_period_best_update, latest_signature_slot=latest_signature_slot)
    if bid in lc_data_store.cache.data:
        raise ValueError('Redundant `_cache_lc_data` call')
    lc_data_store.cache