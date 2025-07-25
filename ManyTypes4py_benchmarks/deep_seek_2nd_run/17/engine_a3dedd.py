import importlib
import math
import textwrap
import time
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager, suppress
from datetime import timedelta
from enum import Enum
from random import Random, getrandbits
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    List,
    Literal,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
import attr
from hypothesis import HealthCheck, Phase, Verbosity, settings as Settings
from hypothesis._settings import local_settings
from hypothesis.database import ExampleDatabase, choices_from_bytes, choices_to_bytes
from hypothesis.errors import (
    BackendCannotProceed,
    FlakyReplay,
    HypothesisException,
    InvalidArgument,
    StopTest,
)
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import (
    NotRequired,
    TypeAlias,
    TypedDict,
    ceil,
    override,
)
from hypothesis.internal.conjecture.choice import (
    ChoiceKeyT,
    ChoiceKwargsT,
    ChoiceNode,
    ChoiceT,
    ChoiceTemplate,
    choices_key,
)
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    ConjectureResult,
    DataObserver,
    Overrun,
    Status,
    _Overrun,
)
from hypothesis.internal.conjecture.datatree import (
    DataTree,
    PreviouslyUnseenBehaviour,
    TreeRecordingObserver,
)
from hypothesis.internal.conjecture.junkdrawer import (
    ensure_free_stackframes,
    startswith,
)
from hypothesis.internal.conjecture.pareto import (
    NO_SCORE,
    ParetoFront,
    ParetoOptimiser,
)
from hypothesis.internal.conjecture.providers import (
    AVAILABLE_PROVIDERS,
    HypothesisProvider,
    PrimitiveProvider,
)
from hypothesis.internal.conjecture.shrinker import (
    Shrinker,
    ShrinkPredicateT,
    sort_key,
)
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.reporting import base_report, report

MAX_SHRINKS: Final[int] = 500
CACHE_SIZE: Final[int] = 10000
MUTATION_POOL_SIZE: Final[int] = 100
MIN_TEST_CALLS: Final[int] = 10
BUFFER_SIZE: Final[int] = 8 * 1024
MAX_SHRINKING_SECONDS: Final[int] = 300

Ls = List[Union["Ls", int]]

def shortlex(s: bytes) -> Tuple[int, bytes]:
    return (len(s), s)

@attr.s
class HealthCheckState:
    valid_examples: int = attr.ib(default=0)
    invalid_examples: int = attr.ib(default=0)
    overrun_examples: int = attr.ib(default=0)
    draw_times: Dict[str, List[float]] = attr.ib(factory=lambda: defaultdict(list))

    @property
    def total_draw_time(self) -> float:
        return math.fsum(sum(self.draw_times.values(), start=[]))

    def timing_report(self) -> str:
        """Return a terminal report describing what was slow."""
        if not self.draw_times:
            return ""
        width = max(
            (
                len(k.removeprefix("generate:").removesuffix(": "))
                for k in self.draw_times
            )
        )
        out = [f"\n  {'':^{width}}   count | fraction |    slowest draws (seconds)"]
        args_in_order = sorted(self.draw_times.items(), key=lambda kv: -sum(kv[1]))
        for i, (argname, times) in enumerate(args_in_order):
            if 5 <= i < len(self.draw_times) - 2 and math.fsum(times) * 20 < self.total_draw_time:
                out.append(
                    f"  (skipped {len(self.draw_times) - i} rows of fast draws)"
                )
                break
            reprs = [f"{t:>6.3f}," for t in sorted(times)[-5:] if t > 0.0005]
            desc = " ".join((["    -- "] * 5 + reprs)[-5:]).rstrip(",")
            arg = argname.removeprefix("generate:").removesuffix(": ")
            out.append(
                f"  {arg:^{width}} | {len(times):>4}  | {math.fsum(times) / self.total_draw_time:>7.0%}  |  {desc}"
            )
        return "\n".join(out)

class ExitReason(Enum):
    max_examples = "settings.max_examples={s.max_examples}"
    max_iterations = "settings.max_examples={s.max_examples}, but < 10% of examples satisfied assumptions"
    max_shrinks = f"shrunk example {MAX_SHRINKS} times"
    finished = "nothing left to do"
    flaky = "test was flaky"
    very_slow_shrinking = "shrinking was very slow"

    def describe(self, settings: Settings) -> str:
        return self.value.format(s=settings)

class RunIsComplete(Exception):
    pass

def _get_provider(backend: str) -> Union[HypothesisProvider, Type[HypothesisProvider]]:
    mname, cname = AVAILABLE_PROVIDERS[backend].rsplit(".", 1)
    provider_cls = getattr(importlib.import_module(mname), cname)
    if provider_cls.lifetime == "test_function":
        return provider_cls(None)
    elif provider_cls.lifetime == "test_case":
        return provider_cls
    else:
        raise InvalidArgument(
            f"invalid lifetime {provider_cls.lifetime} for provider {provider_cls.__name__}. Expected one of 'test_function', 'test_case'."
        )

class CallStats(TypedDict):
    pass

PhaseStatistics = TypedDict(
    "PhaseStatistics",
    {
        "duration-seconds": float,
        "test-cases": List[CallStats],
        "distinct-failures": int,
        "shrinks-successful": int,
    },
)

StatisticsDict = TypedDict(
    "StatisticsDict",
    {
        "generate-phase": NotRequired[PhaseStatistics],
        "reuse-phase": NotRequired[PhaseStatistics],
        "shrink-phase": NotRequired[PhaseStatistics],
        "stopped-because": NotRequired[str],
        "targets": NotRequired[Dict[str, float]],
        "nodeid": NotRequired[str],
    },
)

def choice_count(choices: Sequence[Union[ChoiceT, ChoiceTemplate]]) -> Optional[int]:
    count = 0
    for choice in choices:
        if isinstance(choice, ChoiceTemplate):
            if choice.count is None:
                return None
            count += choice.count
        else:
            count += 1
    return count

class DiscardObserver(DataObserver):
    @override
    def kill_branch(self) -> NoReturn:
        raise ContainsDiscard()

class ConjectureRunner:
    def __init__(
        self,
        test_function: Callable[[ConjectureData], None],
        *,
        settings: Optional[Settings] = None,
        random: Optional[Random] = None,
        database_key: Optional[bytes] = None,
        ignore_limits: bool = False,
    ):
        self._test_function = test_function
        self.settings = settings or Settings()
        self.shrinks: int = 0
        self.finish_shrinking_deadline: Optional[float] = None
        self.call_count: int = 0
        self.misaligned_count: int = 0
        self.valid_examples: int = 0
        self.invalid_examples: int = 0
        self.overrun_examples: int = 0
        self.random = random or Random(getrandbits(128))
        self.database_key = database_key
        self.ignore_limits = ignore_limits
        self._current_phase = "(not a phase)"
        self.statistics: Dict[str, Any] = {}
        self.stats_per_test_case: List[Dict[str, Any]] = []
        self.interesting_examples: Dict[InterestingOrigin, ConjectureResult] = {}
        self.first_bug_found_at: Optional[int] = None
        self.last_bug_found_at: Optional[int] = None
        self.shrunk_examples: Set[InterestingOrigin] = set()
        self.health_check_state: Optional[HealthCheckState] = None
        self.tree = DataTree()
        self.provider = _get_provider(self.settings.backend)
        self.best_observed_targets: Dict[str, float] = defaultdict(lambda: NO_SCORE)
        self.best_examples_of_observed_targets: Dict[str, ConjectureResult] = {}
        self.pareto_front: Optional[ParetoFront] = None
        if self.database_key is not None and self.settings.database is not None:
            self.pareto_front = ParetoFront(self.random)
            self.pareto_front.on_evict(self.on_pareto_evict)
        self.__data_cache = LRUReusedCache[
            Tuple[ChoiceKeyT, ...], Union[ConjectureResult, _Overrun]
        ](CACHE_SIZE)
        self.reused_previously_shrunk_test_case = False
        self.__pending_call_explanation: Optional[str] = None
        self._switch_to_hypothesis_provider = False
        self.__failed_realize_count = 0
        self._verified_by: Optional[str] = None
        self.exit_reason: Optional[ExitReason] = None

    @property
    def using_hypothesis_backend(self) -> bool:
        return self.settings.backend == "hypothesis" or self._switch_to_hypothesis_provider

    def explain_next_call_as(self, explanation: str) -> None:
        self.__pending_call_explanation = explanation

    def clear_call_explanation(self) -> None:
        self.__pending_call_explanation = None

    @contextmanager
    def _log_phase_statistics(self, phase: str) -> Generator[None, None, None]:
        self.stats_per_test_case.clear()
        start_time = time.perf_counter()
        try:
            self._current_phase = phase
            yield
        finally:
            self.statistics[phase + "-phase"] = {
                "duration-seconds": time.perf_counter() - start_time,
                "test-cases": list(self.stats_per_test_case),
                "distinct-failures": len(self.interesting_examples),
                "shrinks-successful": self.shrinks,
            }

    @property
    def should_optimise(self) -> bool:
        return Phase.target in self.settings.phases

    def __tree_is_exhausted(self) -> bool:
        return self.tree.is_exhausted and self.using_hypothesis_backend

    def __stoppable_test_function(self, data: ConjectureData) -> None:
        """Run ``self._test_function``, but convert a ``StopTest`` exception
        into a normal return and avoid raising anything flaky for RecursionErrors.
        """
        with ensure_free_stackframes():
            try:
                self._test_function(data)
            except StopTest as e:
                if e.testcounter == data.testcounter:
                    pass
                else:
                    raise

    def _cache_key(self, choices: Sequence[ChoiceT]) -> Tuple[ChoiceKeyT, ...]:
        return choices_key(choices)

    def _cache(self, data: ConjectureData) -> None:
        result = data.as_result()
        key = self._cache_key(data.choices)
        self.__data_cache[key] = result

    def cached_test_function_ir(
        self,
        choices: Sequence[Union[ChoiceT, ChoiceTemplate]],
        *,
        error_on_discard: bool = False,
        extend: Union[int, Literal["full"]] = 0,
    ) -> Union[ConjectureResult, _Overrun]:
        """
        If ``error_on_discard`` is set to True this will raise ``ContainsDiscard``
        in preference to running the actual test function. This is to allow us
        to skip test cases we expect to be redundant in some cases. Note that
        it may be the case that we don't raise ``ContainsDiscard`` even if the
        result has discards if we cannot determine from previous runs whether
        it will have a discard.
        """
        if not any((isinstance(choice, ChoiceTemplate) for choice in choices)):
            choices = cast(Sequence[ChoiceT], choices)
            key = self._cache_key(choices)
            try:
                cached = self.__data_cache[key]
                if extend == 0 or cached.status is not Status.OVERRUN:
                    return cached
            except KeyError:
                pass
        if extend == "full":
            max_length = None
        elif (count := choice_count(choices)) is None:
            max_length = None
        else:
            max_length = count + extend
        trial_observer = DataObserver()
        if error_on_discard:
            trial_observer = DiscardObserver()
        try:
            trial_data = self.new_conjecture_data_ir(
                choices, observer=trial_observer, max_choices=max_length
            )
            self.tree.simulate_test_function(trial_data)
        except PreviouslyUnseenBehaviour:
            pass
        else:
            trial_data.freeze()
            key = self._cache_key(trial_data.choices)
            if trial_data.status > Status.OVERRUN:
                try:
                    return self.__data_cache[key]
                except KeyError:
                    pass
            else:
                self.__data_cache[key] = Overrun
                return Overrun
            try:
                return self.__data_cache[key]
            except KeyError:
                pass
        data = self.new_conjecture_data_ir(choices, max_choices=max_length)
        self.test_function(data)
        return data.as_result()

    def test_function(self, data: ConjectureData) -> None:
        if self.__pending_call_explanation is not None:
            self.debug(self.__pending_call_explanation)
            self.__pending_call_explanation = None
        self.call_count += 1
        interrupted = False
        try:
            self.__stoppable_test_function(data)
        except KeyboardInterrupt:
            interrupted = True
            raise
        except BackendCannotProceed as exc:
            if exc.scope in ("verified", "exhausted"):
                self._switch_to_hypothesis_provider = True
                if exc.scope == "verified":
                    self._verified_by = self.settings.backend
            elif exc.scope == "discard_test_case":
                self.__failed_realize_count += 1
                if (
                    self.__failed_realize_count > 10
                    and self.__failed_realize_count / self.call_count > 0.2
                ):
                    self._switch_to_hypothesis_provider = True
            interrupted = True
            data.freeze()
            return
        except BaseException:
            self.save_choices(data.choices)
            raise
        finally:
            if not interrupted:
                data.freeze()
                call_stats = {
                    "status": data.status.name.lower(),
                    "runtime": data.finish_time - data.start_time,
                    "drawtime": math.fsum(data.draw_times.values()),
                    "gctime": data.gc_finish_time - data.gc_start_time,
                    "events": sorted(
                        (
                            k if v == "" else f"{k}: {v}"
                            for k, v in data.events.items()
                        )
                    ),
                }
                self.stats_per_test_case.append(call_stats)
                if self.settings.backend != "hypothesis":
                    for node in data.nodes:
                        value = data.provider.realize(node.value)
                        expected_type = {
                            "string": str,
                            "float": float,
                            "integer": int,
                            "boolean": bool,
                            "bytes": bytes,
                        }[node.type]
                        if type(value) is not expected_type:
                            raise HypothesisException(
                                f"expected {expected_type} from {data.provider.realize.__qualname__}, got {type(value)}"
                            )
                        kwargs = cast(
                            ChoiceKwargsT,
                            {k: data.provider.realize(v) for k, v in node.kwargs.items()},
                        )
                        node.value = value
                        node.kwargs = kwargs
                self._cache(data)
                if data.misaligned_at is not None:
                    self.misaligned_count += 1
        self.debug_data(data)
        if (
            data.target_observations
            and self.pareto_front is not None
            and self.pareto_front.add(data.as_result())
        ):
            self.save_choices(data.choices, sub_key=b"pareto")
        if data.status >= Status.VALID:
            for k, v in data.target_observations.items():
                self.best_observed_targets[k] = max(self.best_observed_targets[k], v)
                if k not in self.best_examples_of_observed_targets:
                    data_as_result = data.as_result()
                    assert not isinstance(data_as_result, _Overrun)
                    self.best_examples_of_observed_targets[k] = data_as_result
                    continue
                existing_example = self.best_examples_of_observed_targets[k]
                existing_score = existing_example.target_observations[k]
                if v < existing_score:
                    continue
                if v > existing_score or sort_key(data.nodes) < sort_key(
                    existing_example.nodes
                ):
                    data_as_result = data.as_result()
                    assert not isinstance(data_as_result, _Overrun)
                    self.best_examples_of_observed_targets[k] = data_as_result
        if data.status is Status.VALID:
            self.valid_examples += 1
        if data.status is Status.INVALID:
            self.invalid_examples += 1
        if data.status is Status.OVERRUN:
            self.overrun_examples += 1
        if data.status == Status.INTERESTING:
            if not self.using_hypothesis_backend:
                initial_origin = data.interesting_origin
                initial_traceback = data.expected_traceback
                data = ConjectureData.for_choices(data.choices)
                self.__stoppable_test_function(data)
                data.freeze()
                if data.status != Status.INTERESTING:
                    desc_new_status = {
                        data