from typing import Any, Iterator, List, Tuple

from eth2spec.test.context import spec_state_test, with_altair_and_later, with_presets
from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.constants import MAINNET
from eth2spec.test.helpers.fork_choice import (
    get_genesis_forkchoice_store_and_block,
    on_tick_and_append_step,
    add_attestation,
    add_block,
    tick_and_add_block,
)
from eth2spec.test.helpers.state import state_transition_and_sign_block


def _apply_base_block_a(
    spec: Any, state: Any, store: Any, test_steps: List[Any]
) -> Iterator[Tuple[str, Any]]:
    block = build_empty_block(spec, state, slot=state.slot + 1)
    signed_block_a = state_transition_and_sign_block(spec, state, block)
    yield from tick_and_add_block(spec, store, signed_block_a, test_steps)
    assert spec.get_head(store) == signed_block_a.message.hash_tree_root()
    # No explicit yield here, the caller captures yields from tick_and_add_block.


@with_altair_and_later
@spec_state_test
def test_ex_ante_vanilla(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    With a single adversarial attestation
    Objects:
        Block A - slot N
        Block B (parent A) - slot N+1
        Block C (parent A) - slot N+2
        Attestation_1 (Block B); size `1` - slot N+1
    Steps:
        Block A received at N — A is head
        Block C received at N+2 — C is head
        Block B received at N+2 — C is head
        Attestation_1 received at N+2 — C is head
    """
    test_steps: List[Any] = []
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    yield ("anchor_state", state)
    yield ("anchor_block", anchor_block)
    current_time: int = state.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, current_time, test_steps)
    assert store.time == current_time
    yield from _apply_base_block_a(spec, state, store, test_steps)
    state_a = state.copy()
    state_b = state_a.copy()
    block = build_empty_block(spec, state_a, slot=state_a.slot + 1)
    signed_block_b = state_transition_and_sign_block(spec, state_b, block)
    state_c = state_a.copy()
    block = build_empty_block(spec, state_c, slot=state_a.slot + 2)
    signed_block_c = state_transition_and_sign_block(spec, state_c, block)

    def _filter_participant_set(participants: Any) -> List[Any]:
        return [next(iter(participants))]

    attestation = get_valid_attestation(
        spec, state_b, slot=state_b.slot, signed=False, filter_participant_set=_filter_participant_set
    )
    attestation.data.beacon_block_root = signed_block_b.message.hash_tree_root()
    assert len([i for i in attestation.aggregation_bits if i == 1]) == 1
    sign_attestation(spec, state_b, attestation)
    time: int = state_c.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_c, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_b, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_attestation(spec, store, attestation, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield ("steps", test_steps)


def _get_greater_than_proposer_boost_score(
    spec: Any, store: Any, state: Any, proposer_boost_root: Any, root: Any
) -> int:
    """
    Return the minimum attestation participant count such that attestation_score > proposer_score
    """
    block = store.blocks[root]
    proposer_score: int = 0
    if spec.get_ancestor(store, root, block.slot) == proposer_boost_root:
        num_validators = len(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))
        avg_balance = spec.get_total_active_balance(state) // num_validators
        committee_size = num_validators // spec.SLOTS_PER_EPOCH
        committee_weight = committee_size * avg_balance
        proposer_score = committee_weight * spec.config.PROPOSER_SCORE_BOOST // 100
    base_effective_balance = state.validators[0].effective_balance
    return proposer_score // base_effective_balance + 1


@with_altair_and_later
@with_presets([MAINNET], reason="to create non-duplicate committee")
@spec_state_test
def test_ex_ante_attestations_is_greater_than_proposer_boost_with_boost(
    spec: Any, state: Any
) -> Iterator[Tuple[str, Any]]:
    """
    Adversarial attestations > proposer boost
    Objects:
        Block A - slot N
        Block B (parent A) - slot N+1
        Block C (parent A) - slot N+2
        Attestation_set_1 (Block B); size `proposer_boost + 1` - slot N+1
    Steps:
        Block A received at N — A is head
        Block C received at N+2 — C is head
        Block B received at N+2 — C is head
        Attestation_1 received at N+2 — B is head
    """
    test_steps: List[Any] = []
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    yield ("anchor_state", state)
    yield ("anchor_block", anchor_block)
    current_time: int = state.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, current_time, test_steps)
    assert store.time == current_time
    yield from _apply_base_block_a(spec, state, store, test_steps)
    state_a = state.copy()
    state_b = state_a.copy()
    block = build_empty_block(spec, state_a, slot=state_a.slot + 1)
    signed_block_b = state_transition_and_sign_block(spec, state_b, block)
    state_c = state_a.copy()
    block = build_empty_block(spec, state_c, slot=state_a.slot + 2)
    signed_block_c = state_transition_and_sign_block(spec, state_c, block)
    time: int = state_c.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_c, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_b, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    proposer_boost_root: Any = signed_block_b.message.hash_tree_root()
    root: Any = signed_block_b.message.hash_tree_root()
    participant_num: int = _get_greater_than_proposer_boost_score(spec, store, state, proposer_boost_root, root)

    def _filter_participant_set(participants: Any) -> List[Any]:
        return [index for i, index in enumerate(participants) if i < participant_num]

    attestation = get_valid_attestation(
        spec, state_b, slot=state_b.slot, signed=False, filter_participant_set=_filter_participant_set
    )
    attestation.data.beacon_block_root = signed_block_b.message.hash_tree_root()
    assert len([i for i in attestation.aggregation_bits if i == 1]) == participant_num
    sign_attestation(spec, state_b, attestation)
    yield from add_attestation(spec, store, attestation, test_steps)
    assert spec.get_head(store) == signed_block_b.message.hash_tree_root()
    yield ("steps", test_steps)


@with_altair_and_later
@spec_state_test
def test_ex_ante_sandwich_without_attestations(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    Simple Sandwich test with boost and no attestations.
    Objects:
        Block A - slot N
        Block B (parent A) - slot N+1
        Block C (parent A) - slot N+2
        Block D (parent B) - slot N+3
    Steps:
        Block A received at N — A is head
        Block C received at N+2 — C is head
        Block B received at N+2 — C is head (with boost)
        Block D received at N+3 — D is head (with boost)
    """
    test_steps: List[Any] = []
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    yield ("anchor_state", state)
    yield ("anchor_block", anchor_block)
    current_time: int = state.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, current_time, test_steps)
    assert store.time == current_time
    yield from _apply_base_block_a(spec, state, store, test_steps)
    state_a = state.copy()
    state_b = state_a.copy()
    block = build_empty_block(spec, state_a, slot=state_a.slot + 1)
    signed_block_b = state_transition_and_sign_block(spec, state_b, block)
    state_c = state_a.copy()
    block = build_empty_block(spec, state_c, slot=state_a.slot + 2)
    signed_block_c = state_transition_and_sign_block(spec, state_c, block)
    state_d = state_b.copy()
    block = build_empty_block(spec, state_d, slot=state_a.slot + 3)
    signed_block_d = state_transition_and_sign_block(spec, state_d, block)
    time = state_c.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_c, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_b, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    time = state_d.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_d, test_steps)
    assert spec.get_head(store) == signed_block_d.message.hash_tree_root()
    yield ("steps", test_steps)


@with_altair_and_later
@spec_state_test
def test_ex_ante_sandwich_with_honest_attestation(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    Boosting necessary to sandwich attack.
    Objects:
        Block A - slot N
        Block B (parent A) - slot N+1
        Block C (parent A) - slot N+2
        Block D (parent B) - slot N+3
        Attestation_1 (Block C); size 1 - slot N+2 (honest)
    Steps:
        Block A received at N — A is head
        Block C received at N+2 — C is head
        Block B received at N+2 — C is head
        Attestation_1 received at N+3 — C is head
        Block D received at N+3 — D is head
    """
    test_steps: List[Any] = []
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    yield ("anchor_state", state)
    yield ("anchor_block", anchor_block)
    current_time: int = state.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, current_time, test_steps)
    assert store.time == current_time
    yield from _apply_base_block_a(spec, state, store, test_steps)
    state_a = state.copy()
    state_b = state_a.copy()
    block = build_empty_block(spec, state_a, slot=state_a.slot + 1)
    signed_block_b = state_transition_and_sign_block(spec, state_b, block)
    state_c = state_a.copy()
    block = build_empty_block(spec, state_c, slot=state_a.slot + 2)
    signed_block_c = state_transition_and_sign_block(spec, state_c, block)

    def _filter_participant_set(participants: Any) -> List[Any]:
        return [next(iter(participants))]

    attestation = get_valid_attestation(
        spec, state_c, slot=state_c.slot, signed=False, filter_participant_set=_filter_participant_set
    )
    attestation.data.beacon_block_root = signed_block_c.message.hash_tree_root()
    assert len([i for i in attestation.aggregation_bits if i == 1]) == 1
    sign_attestation(spec, state_c, attestation)
    state_d = state_b.copy()
    block = build_empty_block(spec, state_d, slot=state_a.slot + 3)
    signed_block_d = state_transition_and_sign_block(spec, state_d, block)
    time = state_c.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_c, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_b, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    time = state_d.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_attestation(spec, store, attestation, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_d, test_steps)
    assert spec.get_head(store) == signed_block_d.message.hash_tree_root()
    yield ("steps", test_steps)


@with_altair_and_later
@with_presets([MAINNET], reason="to create non-duplicate committee")
@spec_state_test
def test_ex_ante_sandwich_with_boost_not_sufficient(spec: Any, state: Any) -> Iterator[Tuple[str, Any]]:
    """
    Boost not sufficient to sandwich attack.
    Objects:
        Block A - slot N
        Block B (parent A) - slot N+1
        Block C (parent A) - slot N+2
        Block D (parent B) - slot N+3
        Attestation_set_1 (Block C); size proposer_boost + 1 - slot N+2
    Steps:
        Block A received at N — A is head
        Block C received at N+2 — C is head
        Block B received at N+2 — C is head
        Attestation_set_1 received — C is head
        Block D received at N+3 — C is head
    """
    test_steps: List[Any] = []
    store, anchor_block = get_genesis_forkchoice_store_and_block(spec, state)
    yield ("anchor_state", state)
    yield ("anchor_block", anchor_block)
    current_time: int = state.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, current_time, test_steps)
    assert store.time == current_time
    yield from _apply_base_block_a(spec, state, store, test_steps)
    state_a = state.copy()
    state_b = state_a.copy()
    block = build_empty_block(spec, state_a, slot=state_a.slot + 1)
    signed_block_b = state_transition_and_sign_block(spec, state_b, block)
    state_c = state_a.copy()
    block = build_empty_block(spec, state_c, slot=state_a.slot + 2)
    signed_block_c = state_transition_and_sign_block(spec, state_c, block)
    state_d = state_b.copy()
    block = build_empty_block(spec, state_d, slot=state_a.slot + 3)
    signed_block_d = state_transition_and_sign_block(spec, state_d, block)
    time = state_c.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_block(spec, store, signed_block_c, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_b, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    proposer_boost_root: Any = signed_block_c.message.hash_tree_root()
    root: Any = signed_block_c.message.hash_tree_root()
    participant_num: int = _get_greater_than_proposer_boost_score(spec, store, state, proposer_boost_root, root)

    def _filter_participant_set(participants: Any) -> List[Any]:
        return [index for i, index in enumerate(participants) if i < participant_num]

    attestation = get_valid_attestation(
        spec, state_c, slot=state_c.slot, signed=False, filter_participant_set=_filter_participant_set
    )
    attestation.data.beacon_block_root = signed_block_c.message.hash_tree_root()
    assert len([i for i in attestation.aggregation_bits if i == 1]) == participant_num
    sign_attestation(spec, state_c, attestation)
    time = state_d.slot * spec.config.SECONDS_PER_SLOT + store.genesis_time
    on_tick_and_append_step(spec, store, time, test_steps)
    yield from add_attestation(spec, store, attestation, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield from add_block(spec, store, signed_block_d, test_steps)
    assert spec.get_head(store) == signed_block_c.message.hash_tree_root()
    yield ("steps", test_steps)