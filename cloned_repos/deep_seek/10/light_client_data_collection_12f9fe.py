from typing import Any, Dict, List, Set, Optional, Tuple, Generator, Union, Iterator
from dataclasses import dataclass
from eth_utils import encode_hex
from eth2spec.test.helpers.constants import ALTAIR
from eth2spec.test.helpers.fork_transition import transition_across_forks
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.light_client import compute_start_slot_at_sync_committee_period, get_sync_aggregate, latest_current_sync_committee_gindex, latest_finalized_root_gindex, latest_next_sync_committee_gindex, latest_normalize_merkle_branch, upgrade_lc_header_to_new_spec, upgrade_lc_update_to_new_spec

def _next_epoch_boundary_slot(spec: Any, slot: int) -> int:
    epoch = spec.compute_epoch_at_slot(slot + spec.SLOTS_PER_EPOCH - 1)
    return spec.compute_start_slot_at_epoch(epoch)

@dataclass(frozen=True)
class BlockID(object):
    slot: int
    root: bytes

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
    spec: Any
    data: Any

@dataclass
class ForkedSignedBeaconBlock(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientHeader(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientBootstrap(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientUpdate(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientFinalityUpdate(object):
    spec: Any
    data: Any

@dataclass
class ForkedLightClientOptimisticUpdate(object):
    spec: Any
    data: Any

@dataclass
class CachedLightClientData(object):
    current_sync_committee_branch: List[bytes]
    next_sync_committee_branch: List[bytes]
    finalized_slot: int
    finality_branch: List[bytes]
    current_period_best_update: Any
    latest_signature_slot: int

@dataclass
class LightClientDataCache(object):
    data: Dict[BlockID, CachedLightClientData]
    latest: ForkedLightClientFinalityUpdate
    tail_slot: int

@dataclass
class LightClientDataDB(object):
    headers: Dict[bytes, ForkedLightClientHeader]
    current_branches: Dict[int, List[bytes]]
    sync_committees: Dict[int, Any]
    best_updates: Dict[int, ForkedLightClientUpdate]

@dataclass
class LightClientDataStore(object):
    spec: Any
    cache: LightClientDataCache
    db: LightClientDataDB

@dataclass
class LightClientDataCollectionTest(object):
    steps: List[Dict[str, Any]]
    files: Set[str]
    phases: Dict[str, Any]
    blocks: Dict[bytes, ForkedSignedBeaconBlock]
    finalized_block_roots: Dict[int, bytes]
    states: Dict[bytes, ForkedBeaconState]
    finalized_checkpoint_states: Dict[bytes, ForkedBeaconState]
    latest_finalized_epoch: int
    latest_finalized_bid: BlockID
    historical_tail_slot: int
    lc_data_store: LightClientDataStore

def get_ancestor_of_block_id(test: LightClientDataCollectionTest, bid: BlockID, slot: int) -> Optional[BlockID]:
    try:
        block = test.blocks[bid.root]
        while True:
            if block.data.message.slot <= slot:
                return _block_to_block_id(block.data)
            block = test.blocks[block.data.message.parent_root]
    except KeyError:
        return None

def _block_id_at_finalized_slot(test: LightClientDataCollectionTest, slot: int) -> Optional[BlockID]:
    while slot >= test.historical_tail_slot:
        try:
            return BlockID(slot=slot, root=test.finalized_block_roots[slot])
        except KeyError:
            slot = slot - 1
    return None

def _get_current_sync_committee_for_finalized_period(test: LightClientDataCollectionTest, period: int) -> Optional[Any]:
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

def _light_client_header_for_block(test: LightClientDataCollectionTest, block: Any) -> ForkedLightClientHeader:
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _light_client_header_for_block_id(test: LightClientDataCollectionTest, bid: BlockID) -> ForkedLightClientHeader:
    block = test.blocks[bid.root]
    if not is_post_altair(block.spec):
        spec = test.phases[ALTAIR]
    else:
        spec = block.spec
    return ForkedLightClientHeader(spec=spec, data=spec.block_to_light_client_header(block.data))

def _sync_aggregate_for_block_id(test: LightClientDataCollectionTest, bid: BlockID) -> Optional[Any]:
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
    cached_data = CachedLightClientData(
        current_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.current_sync_committee_gindex_at_slot(state.slot)), latest_current_sync_committee_gindex(lc_data_store.spec)),
        next_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.next_sync_committee_gindex_at_slot(state.slot)), latest_next_sync_committee_gindex(lc_data_store.spec)),
        finalized_slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch),
        finality_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.finalized_root_gindex_at_slot(state.slot)), latest_finalized_root_gindex(lc_data_store.spec)),
        current_period_best_update=current_period_best_update,
        latest_signature_slot=latest_signature_slot
    )
    if bid in lc_data_store.cache.data:
        raise ValueError('Redundant `_cache_lc_data` call')
    lc_data_store.cache.data[bid] = cached_data

def _delete_light_client_data(lc_data_store: LightClientDataStore, bid: BlockID) -> None:
    del lc_data_store.cache.data[bid]

def _create_lc_finality_update_from_lc_data(test: LightClientDataCollectionTest, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any) -> ForkedLightClientFinalityUpdate:
    attested_header = _light_client_header_for_block_id(test, attested_bid)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    finalized_bid = _block_id_at_finalized_slot(test, attested_data.finalized_slot)
    if finalized_bid is not None:
        if finalized_bid.slot != attested_data.finalized_slot:
            attested_data.finalized_slot = finalized_bid.slot
        if finalized_bid.slot == attested_header.spec.GENESIS_SLOT:
            finalized_header = ForkedLightClientHeader(spec=attested_header.spec, data=attested_header.spec.LightClientHeader())
        else:
            finalized_header = _light_client_header_for_block_id(test, finalized_bid)
            finalized_header = ForkedLightClientHeader(spec=attested_header.spec, data=upgrade_lc_header_to_new_spec(finalized_header.spec, attested_header.spec, finalized_header.data))
        finality_branch = attested_data.finality_branch
    return ForkedLightClientFinalityUpdate(spec=attested_header.spec, data=attested_header.spec.LightClientFinalityUpdate(attested_header=attested_header.data, finalized_header=finalized_header.data, finality_branch=finality_branch, sync_aggregate=sync_aggregate, signature_slot=signature_slot))

def _create_lc_update_from_lc_data(test: LightClientDataCollectionTest, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any, next_sync_committee: Any) -> ForkedLightClientUpdate:
    finality_update = _create_lc_finality_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    return ForkedLightClientUpdate(spec=finality_update.spec, data=finality_update.spec.LightClientUpdate(attested_header=finality_update.data.attested_header, next_sync_committee=next_sync_committee, next_sync_committee_branch=attested_data.next_sync_committee_branch, finalized_header=finality_update.data.finalized_header, finality_branch=finality_update.data.finality_branch, sync_aggregate=finality_update.data.sync_aggregate, signature_slot=finality_update.data.signature_slot))

def _create_lc_update(test: LightClientDataCollectionTest, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
    attested_bid = parent_bid
    attested_slot = attested_bid.slot
    if attested_slot < test.lc_data_store.cache.tail_slot:
        _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=ForkedLightClientUpdate(spec=None, data=None), latest_signature_slot=spec.GENESIS_SLOT)
        return
    attested_period = spec.compute_sync_committee_period_at_slot(attested_slot)
    signature_slot = block.message.slot
    signature_period = spec.compute_sync_committee_period_at_slot(signature_slot)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    if attested_period != signature_period:
        best = ForkedLightClientUpdate(spec=None, data=None)
    else:
        best = attested_data.current_period_best_update
    sync_aggregate = block.message.body.sync_aggregate
    num_active_participants = sum(sync_aggregate.sync_committee_bits)
    if num_active_participants < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS:
        latest_signature_slot = attested_data.latest_signature_slot
    else:
        latest_signature_slot = signature_slot
    if num_active_participants < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS or attested_period != signature_period:
        _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=best, latest_signature_slot=latest_signature_slot)
        return
    update = _create_lc_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate, state.next_sync_committee)
    is_better = best.spec is None or spec.is_better_update(update.data, upgrade_lc_update_to_new_spec(best.spec, update.spec, best.data, test.phases))
    if is_better:
        best = update
    _cache_lc_data(test.lc_data_store, spec, state, _block_to_block_id(block), current_period_best_update=best, latest_signature_slot=latest_signature_slot)

def _create_lc_bootstrap(test: LightClientDataCollectionTest, spec: Any, bid: BlockID) -> None:
    block = test.blocks[bid.root]
    period = spec.compute_sync_committee_period_at_slot(bid.slot)
    if period not in test.lc_data_store.db.sync_committees:
        test.lc_data_store.db.sync_committees[period] = _get_current_sync_committee_for_finalized_period(test, period)
    test.lc_data_store.db.headers[bid.root] = ForkedLightClientHeader(spec=block.spec, data=block.spec.block_to_light_client_header(block.data))
    test.lc_data_store.db.current_branches[bid.slot] = _get_light_client_data(test.lc_data_store, bid).current_sync_committee_branch

def _process_new_block_for_light_client(test: LightClientDataCollectionTest, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
    if block.message.slot < test.lc_data_store.cache.tail_slot:
        return
    if is_post_altair(spec):
        _create_lc_update(test, spec, state, block, parent_bid)
    else:
        raise ValueError('`tail_slot` cannot be before Altair')

def _process_head_change_for_light_client(test: LightClientDataCollectionTest, spec: Any, head_bid: BlockID, old_finalized_bid: BlockID) -> None:
    if head_bid.slot < test.lc_data_store.cache.tail_slot:
        return
    head_period = spec.compute_sync_committee_period_at_slot(head_bid.slot)
    low_slot = max(test.lc_data_store.cache.tail_slot, old_finalized_bid.slot)
    low_period = spec.compute_sync_committee_period_at_slot(low_slot)
    bid = head_bid
    for period in reversed(range(low_period, head_period + 1)):
        period_end_slot = compute_start_slot_at_sync_committee_period(spec, period + 1) - 1
        bid = get_ancestor_of_block_id(test, bid, period_end_slot)
        if bid is None or bid.slot < low_slot:
            break
        best = _get_light_client_data(test.lc_data_store, bid).current_period_best_update
        if best.spec is None or sum(best.data.sync_aggregate.sync_committee_bits) < spec.MIN_SYNC_COMMITTEE_PARTICIPANTS:
            test.lc_data_store.db.best_updates.pop(period, None)
        else:
            test.lc_data_store.db.best_updates[period] = best
    head_data = _get_light_client_data(test.lc_data_store, head_bid)
    signature_slot = head_data.latest_signature_slot
    if signature_slot <= low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    signature_bid = get_ancestor_of_block_id(test, head_bid, signature_slot)
    if signature_bid is None or signature_bid.slot <= low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    attested_bid = get_ancestor_of_block_id(test, signature_bid, signature_bid.slot - 1)
    if attested_bid is None or attested_bid.slot < low_slot:
        test.lc_data_store.cache.latest = ForkedLightClientFinalityUpdate(spec=None, data=None)
        return
    sync_aggregate = _sync_aggregate_for_block_id(test, signature_bid