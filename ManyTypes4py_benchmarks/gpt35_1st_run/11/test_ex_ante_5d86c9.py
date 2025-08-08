from eth2spec.test.context import spec_state_test, with_altair_and_later, with_presets
from eth2spec.test.helpers.attestations import get_valid_attestation, sign_attestation
from eth2spec.test.helpers.block import build_empty_block
from eth2spec.test.helpers.constants import MAINNET
from eth2spec.test.helpers.fork_choice import get_genesis_forkchoice_store_and_block, on_tick_and_append_step, add_attestation, add_block, tick_and_add_block
from eth2spec.test.helpers.state import state_transition_and_sign_block
from eth2spec.fork_choice.higher_level import get_head
from eth2spec.fork_choice.abc import Store

def _apply_base_block_a(spec, state, store, test_steps) -> None:
    block = build_empty_block(spec, state, slot=state.slot + 1)
    signed_block_a = state_transition_and_sign_block(spec, state, block)
    yield from tick_and_add_block(spec, store, signed_block_a, test_steps)
    assert get_head(store) == signed_block_a.message.hash_tree_root()

def _get_greater_than_proposer_boost_score(spec, store: Store, state, proposer_boost_root, root) -> int:
    block = store.blocks[root]
    proposer_score = 0
    if spec.get_ancestor(store, root, block.slot) == proposer_boost_root:
        num_validators = len(spec.get_active_validator_indices(state, spec.get_current_epoch(state)))
        avg_balance = spec.get_total_active_balance(state) // num_validators
        committee_size = num_validators // spec.SLOTS_PER_EPOCH
        committee_weight = committee_size * avg_balance
        proposer_score = committee_weight * spec.config.PROPOSER_SCORE_BOOST // 100
    base_effective_balance = state.validators[0].effective_balance
    return proposer_score // base_effective_balance + 1

@with_altair_and_later
@spec_state_test
def test_ex_ante_vanilla(spec, state) -> None:
    ...

@with_altair_and_later
@with_presets([MAINNET], reason='to create non-duplicate committee')
@spec_state_test
def test_ex_ante_attestations_is_greater_than_proposer_boost_with_boost(spec, state) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_ex_ante_sandwich_without_attestations(spec, state) -> None:
    ...

@with_altair_and_later
@spec_state_test
def test_ex_ante_sandwich_with_honest_attestation(spec, state) -> None:
    ...

@with_altair_and_later
@with_presets([MAINNET], reason='to create non-duplicate committee')
@spec_state_test
def test_ex_ante_sandwich_with_boost_not_sufficient(spec, state) -> None:
    ...
