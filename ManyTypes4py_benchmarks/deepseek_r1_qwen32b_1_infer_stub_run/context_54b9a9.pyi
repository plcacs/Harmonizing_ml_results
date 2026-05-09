import pytest
from copy import deepcopy
from dataclasses import dataclass
import importlib
from eth2spec.utils import bls
from .exceptions import SkippedTest
from .helpers.constants import (
    PHASE0, ALTAIR, BELLATRIX, CAPELLA, DENEB, ELECTRA, FULU, WHISK, MINIMAL,
    ALL_PHASES, POST_FORK_OF, ALLOWED_TEST_RUNNER_FORKS, LIGHT_CLIENT_TESTING_FORKS
)
from .helpers.forks import is_post_fork, is_post_electra
from .helpers.genesis import create_genesis_state
from .helpers.typing import Spec, SpecForks
from .helpers.specs import spec_targets
from .utils import vector_test, with_meta_tags
from random import Random
from typing import (
    Any, Callable, Sequence, Dict, List, Optional, Union, Tuple, Set, Iterator
)
from lru import LRU

DEFAULT_TEST_PRESET = MINIMAL
DEFAULT_PYTEST_FORKS = ALL_PHASES


@dataclass(frozen=True)
class ForkMeta:
    pass


def _prepare_state(
    balances_fn: Callable[[Spec], Sequence[int]],
    threshold_fn: Callable[[Spec], int],
    spec: Spec,
    phases: Sequence[str]
) -> Spec.BeaconState:
    ...


def with_custom_state(
    balances_fn: Callable[[Spec], Sequence[int]],
    threshold_fn: Callable[[Spec], int]
) -> Callable[[Callable], Callable]:
    ...


def default_activation_threshold(spec: Spec) -> int:
    ...


def zero_activation_threshold(spec: Spec) -> int:
    ...


def default_balances(spec: Spec) -> List[int]:
    ...


def default_balances_electra(spec: Spec) -> List[int]:
    ...


def scaled_churn_balances_min_churn_limit(spec: Spec) -> List[int]:
    ...


def scaled_churn_balances_equal_activation_churn_limit(spec: Spec) -> List[int]:
    ...


def scaled_churn_balances_exceed_activation_churn_limit(spec: Spec) -> List[int]:
    ...


def scaled_churn_balances_exceed_activation_exit_churn_limit(spec: Spec) -> List[int]:
    ...


def low_balances(spec: Spec) -> List[int]:
    ...


def misc_balances(spec: Spec) -> List[int]:
    ...


def misc_balances_electra(spec: Spec) -> List[int]:
    ...


def misc_balances_in_default_range_with_many_validators(spec: Spec) -> List[int]:
    ...


def low_single_balance(spec: Spec) -> List[int]:
    ...


def large_validator_set(spec: Spec) -> List[int]:
    ...


def single_phase(fn: Callable) -> Callable:
    ...


def dump_skipping_message(reason: str) -> None:
    ...


def description(case_description: str) -> Callable[[Callable], Callable]:
    ...


def spec_test(fn: Callable) -> Callable:
    ...


def spec_state_test(fn: Callable) -> Callable:
    ...


def spec_configured_state_test(conf: Dict[str, Any]) -> Callable[[Callable], Callable]:
    ...


def config_fork_epoch_overrides(spec: Spec, state: Spec.BeaconState) -> Dict[str, int]:
    ...


def with_matching_spec_config(emitted_fork: Optional[str] = None) -> Callable[[Callable], Callable]:
    ...


def spec_state_test_with_matching_config(fn: Callable) -> Callable:
    ...


def expect_assertion_error(fn: Callable) -> None:
    ...


def never_bls(fn: Callable) -> Callable:
    ...


def always_bls(fn: Callable) -> Callable:
    ...


def bls_switch(fn: Callable) -> Callable:
    ...


def disable_process_reveal_deadlines(fn: Callable) -> Callable:
    ...


def with_all_phases(fn: Callable) -> Callable:
    ...


def with_all_phases_from(
    earliest_phase: str,
    all_phases: Sequence[str] = ALL_PHASES
) -> Callable[[Callable], Callable]:
    ...


def with_all_phases_from_except(
    earliest_phase: str,
    except_phases: Optional[Sequence[str]] = None
) -> Callable[[Callable], Callable]:
    ...


def with_all_phases_from_to(
    from_phase: str,
    to_phase: str,
    other_phases: Optional[Sequence[str]] = None,
    all_phases: Sequence[str] = ALL_PHASES
) -> Callable[[Callable], Callable]:
    ...


def with_all_phases_except(
    exclusion_phases: Sequence[str]
) -> Callable[[Callable], Callable]:
    ...


def with_phases(
    phases: Sequence[str],
    other_phases: Optional[Sequence[str]] = None
) -> Callable[[Callable], Callable]:
    ...


def with_presets(
    preset_bases: Sequence[str],
    reason: Optional[str] = None
) -> Callable[[Callable], Callable]:
    ...


class quoted_str(str):
    ...


def _get_basic_dict(ssz_dict: Dict[str, Any]) -> Dict[str, Any]:
    ...


def get_copy_of_spec(spec: Spec) -> Spec:
    ...


def spec_with_config_overrides(
    spec: Spec,
    config_overrides: Dict[str, Any]
) -> Tuple[Spec, Dict[str, Any]]:
    ...


def with_config_overrides(
    config_overrides: Dict[str, Any],
    emitted_fork: Optional[str] = None,
    emit: bool = True
) -> Callable[[Callable], Callable]:
    ...


def only_generator(reason: str) -> Callable[[Callable], Callable]:
    ...


def with_test_suite_name(suite_name: str) -> Callable[[Callable], Callable]:
    ...


def set_fork_metas(fork_metas: Sequence[ForkMeta]) -> Callable[[Callable], Callable]:
    ...


def with_fork_metas(fork_metas: Sequence[ForkMeta]) -> Callable[[Callable], Callable]:
    ...


def yield_fork_meta(fork_metas: Sequence[ForkMeta]) -> Callable[[Callable], Callable]:
    ...