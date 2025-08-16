from typing import Any, Dict, List, Set
from dataclasses import dataclass
from eth_utils import encode_hex
from eth2spec.test.helpers.constants import ALTAIR
from eth2spec.test.helpers.fork_transition import transition_across_forks
from eth2spec.test.helpers.forks import is_post_altair
from eth2spec.test.helpers.light_client import compute_start_slot_at_sync_committee_period, get_sync_aggregate, latest_current_sync_committee_gindex, latest_finalized_root_gindex, latest_next_sync_committee_gindex, latest_normalize_merkle_branch, upgrade_lc_header_to_new_spec, upgrade_lc_update_to_new_spec

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

def _get_light_client_data(lc_data_store: Any, bid: BlockID) -> Any:
    try:
        return lc_data_store.cache.data[bid]
    except KeyError:
        raise ValueError('Trying to get light client data that was not cached')

def _cache_lc_data(lc_data_store: Any, spec: Any, state: Any, bid: BlockID, current_period_best_update: Any, latest_signature_slot: int) -> None:
    cached_data = CachedLightClientData(current_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.current_sync_committee_gindex_at_slot(state.slot)), latest_current_sync_committee_gindex(lc_data_store.spec)), next_sync_committee_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.next_sync_committee_gindex_at_slot(state.slot)), latest_next_sync_committee_gindex(lc_data_store.spec)), finalized_slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch), finality_branch=latest_normalize_merkle_branch(lc_data_store.spec, spec.compute_merkle_proof(state, spec.finalized_root_gindex_at_slot(state.slot)), latest_finalized_root_gindex(lc_data_store.spec)), current_period_best_update=current_period_best_update, latest_signature_slot=latest_signature_slot)
    if bid in lc_data_store.cache.data:
        raise ValueError('Redundant `_cache_lc_data` call')
    lc_data_store.cache.data[bid] = cached_data

def _delete_light_client_data(lc_data_store: Any, bid: BlockID) -> None:
    del lc_data_store.cache.data[bid]

def _create_lc_finality_update_from_lc_data(test: Any, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any) -> ForkedLightClientFinalityUpdate:
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

def _create_lc_update_from_lc_data(test: Any, attested_bid: BlockID, signature_slot: int, sync_aggregate: Any, next_sync_committee: Any) -> ForkedLightClientUpdate:
    finality_update = _create_lc_finality_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate)
    attested_data = _get_light_client_data(test.lc_data_store, attested_bid)
    return ForkedLightClientUpdate(spec=finality_update.spec, data=finality_update.spec.LightClientUpdate(attested_header=finality_update.data.attested_header, next_sync_committee=next_sync_committee, next_sync_committee_branch=attested_data.next_sync_committee_branch, finalized_header=finality_update.data.finalized_header, finality_branch=finality_update.data.finality_branch, sync_aggregate=finality_update.data.sync_aggregate, signature_slot=finality_update.data.signature_slot))

def _create_lc_update(test: Any, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
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

def _create_lc_bootstrap(test: Any, spec: Any, bid: BlockID) -> None:
    block = test.blocks[bid.root]
    period = spec.compute_sync_committee_period_at_slot(bid.slot)
    if period not in test.lc_data_store.db.sync_committees:
        test.lc_data_store.db.sync_committees[period] = _get_current_sync_committee_for_finalized_period(test, period)
    test.lc_data_store.db.headers[bid.root] = ForkedLightClientHeader(spec=block.spec, data=block.spec.block_to_light_client_header(block.data))
    test.lc_data_store.db.current_branches[bid.slot] = _get_light_client_data(test.lc_data_store, bid).current_sync_committee_branch

def _process_new_block_for_light_client(test: Any, spec: Any, state: Any, block: Any, parent_bid: BlockID) -> None:
    if block.message.slot < test.lc_data_store.cache.tail_slot:
        return
    if is_post_altair(spec):
        _create_lc_update(test, spec, state, block, parent_bid)
    else:
        raise ValueError('`tail_slot` cannot be before Altair')

def _process_head_change_for_light_client(test: Any, spec: Any, head_bid: BlockID, old_finalized_bid: BlockID) -> None:
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
    sync_aggregate = _sync_aggregate_for_block_id(test, signature_bid)
    assert sync_aggregate is not None
    test.lc_data_store.cache.latest = _create_lc_finality_update_from_lc_data(test, attested_bid, signature_slot, sync_aggregate)

def _process_finalization_for_light_client(test: Any, spec: Any, finalized_bid: BlockID, old_finalized_bid: BlockID) -> None:
    finalized_slot = finalized_bid.slot
    if finalized_slot < test.lc_data_store.cache.tail_slot:
        return
    first_new_slot = old_finalized_bid.slot + 1
    low_slot = max(first_new_slot, test.lc_data_store.cache.tail_slot)
    boundary_slot = finalized_slot
    while boundary_slot >= low_slot:
        bid = _block_id_at_finalized_slot(test, boundary_slot)
        if bid is None:
            break
        if bid.slot >= low_slot:
            _create_lc_bootstrap(test, spec, bid)
        boundary_slot = _next_epoch_boundary_slot(spec, bid.slot)
        if boundary_slot < spec.SLOTS_PER_EPOCH:
            break
        boundary_slot = boundary_slot - spec.SLOTS_PER_EPOCH
    bids_to_delete = []
    for bid in test.lc_data_store.cache.data:
        if bid.slot >= finalized_bid.slot:
            continue
        bids_to_delete.append(bid)
    for bid in bids_to_delete:
        _delete_light_client_data(test.lc_data_store, bid)

def get_light_client_bootstrap(test: Any, block_root: bytes) -> ForkedLightClientBootstrap:
    try:
        header = test.lc_data_store.db.headers[block_root]
    except KeyError:
        return ForkedLightClientBootstrap(spec=None, data=None)
    slot = header.data.beacon.slot
    period = header.spec.compute_sync_committee_period_at_slot(slot)
    return ForkedLightClientBootstrap(spec=header.spec, data=header.spec.LightClientBootstrap(header=header.data, current_sync_committee=test.lc_data_store.db.sync_committees[period], current_sync_committee_branch=test.lc_data_store.db.current_branches[slot]))

def get_light_client_update_for_period(test: Any, period: int) -> ForkedLightClientUpdate:
    try:
        return test.lc_data_store.db.best_updates[period]
    except KeyError:
        return ForkedLightClientUpdate(spec=None, data=None)

def get_light_client_finality_update(test: Any) -> ForkedLightClientFinalityUpdate:
    return test.lc_data_store.cache.latest

def get_light_client_optimistic_update(test: Any) -> ForkedLightClientOptimisticUpdate:
    finality_update = get_light_client_finality_update(test)
    if finality_update.spec is None:
        return ForkedLightClientOptimisticUpdate(spec=None, data=None)
    return ForkedLightClientOptimisticUpdate(spec=finality_update.spec, data=finality_update.spec.LightClientOptimisticUpdate(attested_header=finality_update.data.attested_header, sync_aggregate=finality_update.data.sync_aggregate, signature_slot=finality_update.data.signature_slot))

def setup_lc_data_collection_test(spec: Any, state: Any, phases: Any = None) -> LightClientDataCollectionTest:
    assert spec.compute_slots_since_epoch_start(state.slot) == 0
    test = LightClientDataCollectionTest(steps=[], files=set(), phases=phases, blocks={}, finalized_block_roots={}, states={}, finalized_checkpoint_states={}, latest_finalized_epoch=state.finalized_checkpoint.epoch, latest_finalized_bid=BlockID(slot=spec.compute_start_slot_at_epoch(state.finalized_checkpoint.epoch), root=state.finalized_checkpoint.root), historical_tail_slot=state.slot, lc_data_store=LightClientDataStore(spec=spec, cache=LightClientDataCache(data={}, latest=ForkedLightClientFinalityUpdate(spec=None, data=None), tail_slot=max(state.slot, spec.compute_start_slot_at_epoch(spec.config.ALTAIR_FORK_EPOCH))), db=LightClientDataDB(headers={}, current_branches={}, sync_committees={}, best_updates={}))
    bid = _state_to_block_id(state)
    yield ('initial_state', state)
    test.blocks[bid.root] = ForkedSignedBeaconBlock(spec=spec, data=spec.SignedBeaconBlock(message=spec.BeaconBlock(state_root=state.hash_tree_root())))
    test.finalized_block_roots[bid.slot] = bid.root
    test.states[state.hash_tree_root()] = ForkedBeaconState(spec=spec, data=state)
    test.finalized_checkpoint_states[state.hash_tree_root()] = ForkedBeaconState(spec=spec, data=state)
    _cache_lc_data(test.lc_data_store, spec, state, bid, current_period_best_update=ForkedLightClientUpdate(spec=None, data=None), latest_signature_slot=