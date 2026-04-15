import random
from typing import Any, Callable, Generator, List, Optional, Set, Union

from eth2spec.test.context import (
    with_altair_and_later,
    with_presets,
    spec_state_test,
    always_bls,
    single_phase,
    with_custom_state,
    spec_test,
    default_balances_electra,
    default_activation_threshold,
)

def test_invalid_signature_bad_domain(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_missing_participant(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_no_participants(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_infinite_signature_with_all_participants(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_infinite_signature_with_single_participant(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_extra_participant(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def is_duplicate_sync_committee(committee_indices: List[int]) -> bool: ...

def test_sync_committee_rewards_nonduplicate_committee(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_no_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_half_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_full_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def _run_sync_committee_selected_twice(
    spec: Any,
    state: Any,
    pre_balance: int,
    participate_first_position: bool,
    participate_second_position: bool,
    skip_reward_validation: bool = False,
) -> Generator[Any, None, int]: ...

def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_first_one(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_zero_balance_only_participate_second_one(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_first_one(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_duplicate_committee_max_effective_balance_only_participate_second_one(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_not_full_participants(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_rewards_empty_participants(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_past_block(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_invalid_signature_previous_committee(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_valid_signature_future_committee(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_proposer_in_committee_without_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_proposer_in_committee_with_participation(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def _exit_validator_from_committee_and_transition_state(
    spec: Any,
    state: Any,
    committee_indices: List[int],
    rng: random.Random,
    target_epoch_provider: Callable[[Any], int],
    withdrawable_offset: int = 1,
) -> int: ...

def test_sync_committee_with_participating_exited_member(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_with_nonparticipating_exited_member(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_with_participating_withdrawable_member(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...

def test_sync_committee_with_nonparticipating_withdrawable_member(
    spec: Any,
    state: Any,
) -> Generator[Any, None, None]: ...