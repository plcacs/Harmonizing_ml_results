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
from typing import Callable, Final, List, Literal, NoReturn, Optional, Union, cast

import attr
from hypothesis import HealthCheck, Phase, Verbosity, settings as Settings
from hypothesis._settings import local_settings
from hypothesis.database import ExampleDatabase, choices_from_bytes, choices_to_bytes
from hypothesis.errors import BackendCannotProceed, FlakyReplay, HypothesisException, InvalidArgument, StopTest
from hypothesis.internal.cache import LRUReusedCache
from hypothesis.internal.compat import NotRequired, TypeAlias, TypedDict, ceil, override
from hypothesis.internal.conjecture.choice import ChoiceKeyT, ChoiceKwargsT, ChoiceNode, ChoiceT, ChoiceTemplate, choices_key
from hypothesis.internal.conjecture.data import ConjectureData, ConjectureResult, DataObserver, Overrun, Status, _Overrun
from hypothesis.internal.conjecture.datatree import DataTree, PreviouslyUnseenBehaviour, TreeRecordingObserver
from hypothesis.internal.conjecture.junkdrawer import ensure_free_stackframes, startswith
from hypothesis.internal.conjecture.pareto import NO_SCORE, ParetoFront, ParetoOptimiser
from hypothesis.internal.conjecture.providers import AVAILABLE_PROVIDERS, HypothesisProvider, PrimitiveProvider
from hypothesis.internal.conjecture.shrinker import Shrinker, ShrinkPredicateT, sort_key
from hypothesis.internal.escalation import InterestingOrigin
from hypothesis.internal.healthcheck import fail_health_check
from hypothesis.reporting import base_report, report

MAX_SHRINKS: Final[int] = 500
CACHE_SIZE: Final[int] = 10000
MUTATION_POOL_SIZE: Final[int] = 100
MIN_TEST_CALLS: Final[int] = 10
BUFFER_SIZE: Final[int] = 8 * 1024
MAX_SHRINKING_SECONDS: Final[int] = 300
Ls = list['Ls | int']

def shortlex(s: str) -> tuple[int, str]:
    return (len(s), s)

@attr.s(auto_attribs=True)
class HealthCheckState:
    valid_examples: int = 0
    invalid_examples: int = 0
    overrun_examples: int = 0
    draw_times: dict[str, List[float]] = attr.ib(factory=lambda: defaultdict(list))

    @property
    def total_draw_time(self) -> float:
        return math.fsum(sum(self.draw_times.values(), start=[]))

    def timing_report(self) -> str:
        """Return a terminal report describing what was slow."""
        if not self.draw_times:
            return ''
        width: int = max((len(k.removeprefix('generate:').removesuffix(': ')) for k in self.draw_times))
        out: List[str] = [f'\n  {"":^{width}}   count | fraction |    slowest draws (seconds)']
        args_in_order = sorted(self.draw_times.items(), key=lambda kv: -sum(kv[1]))
        for i, (argname, times) in enumerate(args_in_order):
            if 5 <= i < len(self.draw_times) - 2 and math.fsum(times) * 20 < self.total_draw_time:
                out.append(f'  (skipped {len(self.draw_times) - i} rows of fast draws)')
                break
            reprs = [f'{t:>6.3f},' for t in sorted(times)[-5:] if t > 0.0005]
            desc = ' '.join((['    -- '] * 5 + reprs)[-5:]).rstrip(',')
            arg = argname.removeprefix('generate:').removesuffix(': ')
            out.append(f'  {arg:^{width}} | {len(times):>4}  | {math.fsum(times) / self.total_draw_time:>7.0%}  |  {desc}')
        return '\n'.join(out)

class ExitReason(Enum):
    max_examples = 'settings.max_examples={s.max_examples}'
    max_iterations = 'settings.max_examples={s.max_examples}, but < 10% of examples satisfied assumptions'
    max_shrinks = f'shrunk example {MAX_SHRINKS} times'
    finished = 'nothing left to do'
    flaky = 'test was flaky'
    very_slow_shrinking = 'shrinking was very slow'

    def describe(self, settings: Settings) -> str:
        return self.value.format(s=settings)

class RunIsComplete(Exception):
    pass

def _get_provider(backend: str) -> Union[HypothesisProvider, type]:
    mname, cname = AVAILABLE_PROVIDERS[backend].rsplit('.', 1)
    provider_cls = getattr(importlib.import_module(mname), cname)
    if provider_cls.lifetime == 'test_function':
        return provider_cls(None)
    elif provider_cls.lifetime == 'test_case':
        return provider_cls
    else:
        raise InvalidArgument(f"invalid lifetime {provider_cls.lifetime} for provider {provider_cls.__name__}. Expected one of 'test_function', 'test_case'.")

class CallStats(TypedDict):
    pass

PhaseStatistics = TypedDict('PhaseStatistics', {'duration-seconds': float, 'test-cases': list[CallStats], 'distinct-failures': int, 'shrinks-successful': int})
StatisticsDict = TypedDict('StatisticsDict', {'generate-phase': NotRequired[PhaseStatistics], 'reuse-phase': NotRequired[PhaseStatistics], 'shrink-phase': NotRequired[PhaseStatistics], 'stopped-because': NotRequired[str], 'targets': NotRequired[dict[str, float]], 'nodeid': NotRequired[str]})

def choice_count(choices: Sequence[Union[ChoiceTemplate, ChoiceT]]) -> Optional[int]:
    count: int = 0
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
        raise ContainsDiscard

class ConjectureRunner:
    _test_function: Callable[[ConjectureData], None]
    settings: Settings
    shrinks: int
    finish_shrinking_deadline: Optional[float]
    call_count: int
    misaligned_count: int
    valid_examples: int
    invalid_examples: int
    overrun_examples: int
    random: Random
    database_key: Optional[bytes]
    ignore_limits: bool
    _current_phase: str
    statistics: dict[str, object]
    stats_per_test_case: List[CallStats]
    interesting_examples: dict[object, ConjectureResult]
    first_bug_found_at: Optional[int]
    last_bug_found_at: Optional[int]
    shrunk_examples: set
    health_check_state: Optional[HealthCheckState]
    tree: DataTree
    provider: Union[HypothesisProvider, type]
    best_observed_targets: dict[str, float]
    best_examples_of_observed_targets: dict[str, ConjectureResult]
    pareto_front: Optional[ParetoFront]
    __data_cache: LRUReusedCache[tuple[ChoiceKeyT, ...], Union[ConjectureResult, _Overrun]]
    reused_previously_shrunk_test_case: bool
    __pending_call_explanation: Optional[str]
    _switch_to_hypothesis_provider: bool
    __failed_realize_count: int
    _verified_by: Optional[str]
    exit_reason: Optional[ExitReason]

    def __init__(
        self,
        test_function: Callable[[ConjectureData], None],
        *,
        settings: Optional[Settings] = None,
        random: Optional[Random] = None,
        database_key: Optional[bytes] = None,
        ignore_limits: bool = False,
    ) -> None:
        self._test_function = test_function
        self.settings = settings or Settings()
        self.shrinks = 0
        self.finish_shrinking_deadline = None
        self.call_count = 0
        self.misaligned_count = 0
        self.valid_examples = 0
        self.invalid_examples = 0
        self.overrun_examples = 0
        self.random = random or Random(getrandbits(128))
        self.database_key = database_key
        self.ignore_limits = ignore_limits
        self._current_phase = '(not a phase)'
        self.statistics = {}
        self.stats_per_test_case = []
        self.interesting_examples = {}
        self.first_bug_found_at = None
        self.last_bug_found_at = None
        self.shrunk_examples = set()
        self.health_check_state = None
        self.tree = DataTree()
        self.provider = _get_provider(self.settings.backend)
        self.best_observed_targets = defaultdict(lambda: NO_SCORE)
        self.best_examples_of_observed_targets = {}
        self.pareto_front = None
        if self.database_key is not None and self.settings.database is not None:
            self.pareto_front = ParetoFront(self.random)
            self.pareto_front.on_evict(self.on_pareto_evict)
        self.__data_cache = LRUReusedCache[tuple[ChoiceKeyT, ...], Union[ConjectureResult, _Overrun]](CACHE_SIZE)
        self.reused_previously_shrunk_test_case = False
        self.__pending_call_explanation = None
        self._switch_to_hypothesis_provider = False
        self.__failed_realize_count = 0
        self._verified_by = None
        self.exit_reason = None

    @property
    def using_hypothesis_backend(self) -> bool:
        return self.settings.backend == 'hypothesis' or self._switch_to_hypothesis_provider

    def explain_next_call_as(self, explanation: str) -> None:
        self.__pending_call_explanation = explanation

    def clear_call_explanation(self) -> None:
        self.__pending_call_explanation = None

    @contextmanager
    def _log_phase_statistics(self, phase: str) -> Generator[None, None, None]:
        self.stats_per_test_case.clear()
        start_time: float = time.perf_counter()
        try:
            self._current_phase = phase
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.statistics[phase + '-phase'] = {'duration-seconds': duration, 'test-cases': list(self.stats_per_test_case), 'distinct-failures': len(self.interesting_examples), 'shrinks-successful': self.shrinks}

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

    def _cache_key(self, choices: Sequence[ChoiceT]) -> bytes:
        return choices_key(choices)

    def _cache(self, data: ConjectureData) -> None:
        result: Union[ConjectureResult, _Overrun] = data.as_result()
        key: bytes = self._cache_key(data.choices)
        self.__data_cache[key] = result

    def cached_test_function_ir(
        self,
        choices: Sequence[ChoiceT],
        *,
        error_on_discard: bool = False,
        extend: Union[int, Literal['full']] = 0
    ) -> Union[ConjectureResult, _Overrun]:
        if not any((isinstance(choice, ChoiceTemplate) for choice in choices)):
            choices = cast(Sequence[ChoiceT], choices)
            key = self._cache_key(choices)
            try:
                cached = self.__data_cache[key]
                if extend == 0 or cached.status is not Status.OVERRUN:
                    return cached
            except KeyError:
                pass
        if extend == 'full':
            max_length: Optional[int] = None
        elif (count := choice_count(choices)) is None:
            max_length = None
        else:
            max_length = count + extend  # type: ignore
        trial_observer: DataObserver = DataObserver()
        if error_on_discard:
            trial_observer = DiscardObserver()
        try:
            trial_data: ConjectureData = self.new_conjecture_data_ir(choices, observer=trial_observer, max_choices=max_length)
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
        interrupted: bool = False
        try:
            self.__stoppable_test_function(data)
        except KeyboardInterrupt:
            interrupted = True
            raise
        except BackendCannotProceed as exc:
            if exc.scope in ('verified', 'exhausted'):
                self._switch_to_hypothesis_provider = True
                if exc.scope == 'verified':
                    self._verified_by = self.settings.backend
            elif exc.scope == 'discard_test_case':
                self.__failed_realize_count += 1
                if self.__failed_realize_count > 10 and self.__failed_realize_count / self.call_count > 0.2:
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
                call_stats: CallStats = {
                    'status': data.status.name.lower(),
                    'runtime': data.finish_time - data.start_time,
                    'drawtime': math.fsum(data.draw_times.values()),
                    'gctime': data.gc_finish_time - data.gc_start_time,
                    'events': sorted((k if v == '' else f'{k}: {v}' for k, v in data.events.items()))
                }
                self.stats_per_test_case.append(call_stats)
                if self.settings.backend != 'hypothesis':
                    for node in data.nodes:
                        value = data.provider.realize(node.value)
                        expected_type = {'string': str, 'float': float, 'integer': int, 'boolean': bool, 'bytes': bytes}[node.type]
                        if type(value) is not expected_type:
                            raise HypothesisException(f'expected {expected_type} from {data.provider.realize.__qualname__}, got {type(value)}')
                        kwargs: ChoiceKwargsT = {k: data.provider.realize(v) for k, v in node.kwargs.items()}
                        node.value = value
                        node.kwargs = kwargs
                self._cache(data)
                if data.misaligned_at is not None:
                    self.misaligned_count += 1
        self.debug_data(data)
        if data.target_observations and self.pareto_front is not None and self.pareto_front.add(data.as_result()):
            self.save_choices(data.choices, sub_key=b'pareto')
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
                if v > existing_score or sort_key(data.nodes) < sort_key(existing_example.nodes):
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
                    desc_new_status = {data.status.VALID: 'passed', data.status.INVALID: 'failed filters', data.status.OVERRUN: 'overran'}[data.status]
                    wrapped_tb = '' if initial_traceback is None else textwrap.indent(initial_traceback, '  | ')
                    raise FlakyReplay(f"Inconsistent results from replaying a failing test case!\n{wrapped_tb}on backend={self.settings.backend!r} but {desc_new_status} under backend='hypothesis'", interesting_origins=[initial_origin])
                self._cache(data)
            key = data.interesting_origin
            changed = False
            try:
                existing = self.interesting_examples[key]
            except KeyError:
                changed = True
                self.last_bug_found_at = self.call_count
                if self.first_bug_found_at is None:
                    self.first_bug_found_at = self.call_count
            else:
                if sort_key(data.nodes) < sort_key(existing.nodes):
                    self.shrinks += 1
                    self.downgrade_choices(existing.choices)
                    self.__data_cache.unpin(self._cache_key(existing.choices))
                    changed = True
            if changed:
                self.save_choices(data.choices)
                self.interesting_examples[key] = data.as_result()
                self.__data_cache.pin(self._cache_key(data.choices), data.as_result())
                self.shrunk_examples.discard(key)
            if self.shrinks >= MAX_SHRINKS:
                self.exit_with(ExitReason.max_shrinks)
        if not self.ignore_limits and self.finish_shrinking_deadline is not None and (self.finish_shrinking_deadline < time.perf_counter()):
            report('WARNING: Hypothesis has spent more than five minutes working to shrink a failing example, and stopped because it is making very slow progress.  When you re-run your tests, shrinking will resume and may take this long before aborting again.\nPLEASE REPORT THIS if you can provide a reproducing example, so that we can improve shrinking performance for everyone.')
            self.exit_with(ExitReason.very_slow_shrinking)
        if not self.interesting_examples:
            if self.valid_examples >= self.settings.max_examples:
                self.exit_with(ExitReason.max_examples)
            if self.call_count >= max(self.settings.max_examples * 10, 1000):
                self.exit_with(ExitReason.max_iterations)
        if self.__tree_is_exhausted():
            self.exit_with(ExitReason.finished)
        self.record_for_health_check(data)

    def on_pareto_evict(self, data: ConjectureResult) -> None:
        self.settings.database.delete(self.pareto_key, choices_to_bytes(data.choices))

    def generate_novel_prefix(self) -> Sequence[ChoiceT]:
        return self.tree.generate_novel_prefix(self.random)

    def record_for_health_check(self, data: ConjectureData) -> None:
        if data.status == Status.INTERESTING:
            self.health_check_state = None
        state = self.health_check_state
        if state is None:
            return
        for k, v in data.draw_times.items():
            state.draw_times[k].append(v)
        if data.status == Status.VALID:
            state.valid_examples += 1
        elif data.status == Status.INVALID:
            state.invalid_examples += 1
        else:
            assert data.status == Status.OVERRUN
            state.overrun_examples += 1
        max_valid_draws = 10
        max_invalid_draws = 50
        max_overrun_draws = 20
        assert state.valid_examples <= max_valid_draws
        if state.valid_examples == max_valid_draws:
            self.health_check_state = None
            return
        if state.overrun_examples == max_overrun_draws:
            fail_health_check(
                self.settings,
                f'Examples routinely exceeded the max allowable size. ({state.overrun_examples} examples overran while generating {state.valid_examples} valid ones). Generating examples this large will usually lead to bad results. You could try setting max_size parameters on your collections and turning max_leaves down on recursive() calls.',
                HealthCheck.data_too_large
            )
        if state.invalid_examples == max_invalid_draws:
            fail_health_check(
                self.settings,
                f'It looks like your strategy is filtering out a lot of data. Health check found {state.invalid_examples} filtered examples but only {state.valid_examples} good ones. This will make your tests much slower, and also will probably distort the data generation quite a lot. You should adapt your strategy to filter less. This can also be caused by a low max_leaves parameter in recursive() calls',
                HealthCheck.filter_too_much
            )
        draw_time: float = state.total_draw_time
        draw_time_limit: timedelta = 5 * (self.settings.deadline or timedelta(seconds=6))
        if draw_time > max(1.0, draw_time_limit.total_seconds()):
            fail_health_check(
                self.settings,
                f"Data generation is extremely slow: Only produced {state.valid_examples} valid examples in {draw_time:.2f} seconds ({state.invalid_examples} invalid ones and {state.overrun_examples} exceeded maximum size). Try decreasing size of the data you're generating (with e.g. max_size or max_leaves parameters)." + state.timing_report(),
                HealthCheck.too_slow
            )

    def save_choices(self, choices: Sequence[ChoiceT], sub_key: Optional[bytes] = None) -> None:
        if self.settings.database is not None:
            key = self.sub_key(sub_key)
            if key is None:
                return
            self.settings.database.save(key, choices_to_bytes(choices))

    def downgrade_choices(self, choices: Sequence[ChoiceT]) -> None:
        buffer: bytes = choices_to_bytes(choices)
        if self.settings.database is not None and self.database_key is not None:
            self.settings.database.move(self.database_key, self.secondary_key, buffer)

    def sub_key(self, sub_key: Optional[bytes]) -> Optional[bytes]:
        if self.database_key is None:
            return None
        if sub_key is None:
            return self.database_key
        return b'.'.join((self.database_key, sub_key))

    @property
    def secondary_key(self) -> Optional[bytes]:
        return self.sub_key(b'secondary')

    @property
    def pareto_key(self) -> Optional[bytes]:
        return self.sub_key(b'pareto')

    def debug(self, message: str) -> None:
        if self.settings.verbosity >= Verbosity.debug:
            base_report(message)

    @property
    def report_debug_info(self) -> bool:
        return self.settings.verbosity >= Verbosity.debug

    def debug_data(self, data: ConjectureData) -> None:
        if not self.report_debug_info:
            return
        status = repr(data.status)
        if data.status == Status.INTERESTING:
            status = f'{status} ({data.interesting_origin!r})'
        self.debug(f'{len(data.choices)} choices {data.choices} -> {status}{(", " + data.output if data.output else "")}')

    def run(self) -> None:
        with local_settings(self.settings):
            try:
                self._run()
            except RunIsComplete:
                pass
            for v in self.interesting_examples.values():
                self.debug_data(v)
            self.debug('Run complete after %d examples (%d valid) and %d shrinks' % (self.call_count, self.valid_examples, self.shrinks))

    @property
    def database(self) -> Optional[ExampleDatabase]:
        if self.database_key is None:
            return None
        return self.settings.database

    def has_existing_examples(self) -> bool:
        return self.database is not None and Phase.reuse in self.settings.phases

    def reuse_existing_examples(self) -> None:
        if self.has_existing_examples():
            self.debug('Reusing examples from database')
            corpus: List[bytes] = sorted(self.settings.database.fetch(self.database_key), key=shortlex)
            factor: float = 0.1 if Phase.generate in self.settings.phases else 1
            desired_size: int = max(2, ceil(factor * self.settings.max_examples))
            primary_corpus_size: int = len(corpus)
            if len(corpus) < desired_size:
                extra_corpus: List[bytes] = list(self.settings.database.fetch(self.secondary_key))
                shortfall: int = desired_size - len(corpus)
                if len(extra_corpus) <= shortfall:
                    extra = extra_corpus
                else:
                    extra = self.random.sample(extra_corpus, shortfall)
                extra.sort(key=shortlex)
                corpus.extend(extra)
            found_interesting_in_primary: bool = False
            all_interesting_in_primary_were_exact: bool = True
            for i, existing in enumerate(corpus):
                if i >= primary_corpus_size and found_interesting_in_primary:
                    break
                choices_obj = choices_from_bytes(existing)
                if choices_obj is None:
                    self.settings.database.delete(self.database_key, existing)
                    continue
                data = self.cached_test_function_ir(choices_obj, extend='full')
                if data.status != Status.INTERESTING:
                    self.settings.database.delete(self.database_key, existing)
                    self.settings.database.delete(self.secondary_key, existing)
                else:
                    if i < primary_corpus_size:
                        found_interesting_in_primary = True
                        assert not isinstance(data, _Overrun)
                        if choices_key(choices_obj) != choices_key(data.choices):
                            all_interesting_in_primary_were_exact = False
                    if not self.settings.report_multiple_bugs:
                        break
            if found_interesting_in_primary:
                if all_interesting_in_primary_were_exact:
                    self.reused_previously_shrunk_test_case = True
            assert self.pareto_front is not None
            if len(corpus) < desired_size and (not self.interesting_examples):
                desired_extra: int = desired_size - len(corpus)
                pareto_corpus: List[bytes] = list(self.settings.database.fetch(self.pareto_key))
                if len(pareto_corpus) > desired_extra:
                    pareto_corpus = self.random.sample(pareto_corpus, desired_extra)
                pareto_corpus.sort(key=shortlex)
                for existing in pareto_corpus:
                    choices_obj = choices_from_bytes(existing)
                    if choices_obj is None:
                        self.settings.database.delete(self.pareto_key, existing)
                        continue
                    data = self.cached_test_function_ir(choices_obj, extend='full')
                    if data not in self.pareto_front:
                        self.settings.database.delete(self.pareto_key, existing)
                    if data.status == Status.INTERESTING:
                        break

    def exit_with(self, reason: ExitReason) -> NoReturn:
        if self.ignore_limits:
            return
        self.statistics['stopped-because'] = reason.describe(self.settings)
        if self.best_observed_targets:
            self.statistics['targets'] = dict(self.best_observed_targets)
        self.debug(f'exit_with({reason.name})')
        self.exit_reason = reason
        raise RunIsComplete

    def should_generate_more(self) -> bool:
        if self.valid_examples >= self.settings.max_examples or self.call_count >= max(self.settings.max_examples * 10, 1000):
            return False
        if not self.interesting_examples:
            return True
        elif Phase.shrink not in self.settings.phases or not self.settings.report_multiple_bugs:
            return False
        assert isinstance(self.first_bug_found_at, int)
        assert isinstance(self.last_bug_found_at, int)
        assert self.first_bug_found_at <= self.last_bug_found_at <= self.call_count
        return self.call_count < MIN_TEST_CALLS or self.call_count < min(self.first_bug_found_at + 1000, self.last_bug_found_at * 2)

    def generate_new_examples(self) -> None:
        if Phase.generate not in self.settings.phases:
            return
        if self.interesting_examples:
            return
        self.debug('Generating new examples')
        assert self.should_generate_more()
        zero_data = self.cached_test_function_ir((ChoiceTemplate('simplest', count=None),))
        if zero_data.status > Status.OVERRUN:
            assert isinstance(zero_data, ConjectureResult)
            self.__data_cache.pin(self._cache_key(zero_data.choices), zero_data.as_result())
        if zero_data.status == Status.OVERRUN or (zero_data.status == Status.VALID and isinstance(zero_data, ConjectureResult) and (zero_data.length * 2 > BUFFER_SIZE)):
            fail_health_check(
                self.settings,
                'The smallest natural example for your test is extremely large. This makes it difficult for Hypothesis to generate good examples, especially when trying to reduce failing ones at the end. Consider reducing the size of your data if it is of a fixed size. You could also fix this by improving how your data shrinks (see https://hypothesis.readthedocs.io/en/latest/data.html#shrinking for details), or by introducing default values inside your strategy. e.g. could you replace some arguments with their defaults by using one_of(none(), some_complex_strategy)?',
                HealthCheck.large_base_example
            )
        self.health_check_state = HealthCheckState()
        consecutive_zero_extend_is_invalid: int = 0
        small_example_cap: int = min(self.settings.max_examples // 10, 50)
        optimise_at: int = max(self.settings.max_examples // 2, small_example_cap + 1, 10)
        ran_optimisations: bool = False
        while self.should_generate_more():
            if not self.using_hypothesis_backend:
                data = self.new_conjecture_data_ir([])
                with suppress(BackendCannotProceed):
                    self.test_function(data)
                continue
            self._current_phase = 'generate'
            prefix = self.generate_novel_prefix()
            if self.valid_examples <= small_example_cap and self.call_count <= 5 * small_example_cap and (not self.interesting_examples) and (consecutive_zero_extend_is_invalid < 5):
                minimal_example = self.cached_test_function_ir(prefix + (ChoiceTemplate('simplest', count=None),))
                if minimal_example.status < Status.VALID:
                    consecutive_zero_extend_is_invalid += 1
                    continue
                assert isinstance(minimal_example, ConjectureResult)
                consecutive_zero_extend_is_invalid = 0
                minimal_extension = len(minimal_example.choices) - len(prefix)
                max_length: int = len(prefix) + minimal_extension * 5
                trial_data = self.new_conjecture_data_ir(prefix, max_choices=max_length)
                try:
                    self.tree.simulate_test_function(trial_data)
                    continue
                except PreviouslyUnseenBehaviour:
                    pass
                assert isinstance(trial_data.observer, TreeRecordingObserver)
                if trial_data.observer.killed:
                    continue
                if not self.should_generate_more():
                    break
                prefix = trial_data.choices
            else:
                max_length = None
            data = self.new_conjecture_data_ir(prefix, max_choices=max_length)
            self.test_function(data)
            if data.status is Status.OVERRUN and max_length is not None and ('invalid because' not in data.events):
                data.events['invalid because'] = 'reduced max size for early examples (avoids flaky health checks)'
            self.generate_mutations_from(data)
            if self.valid_examples >= max(small_example_cap, optimise_at) and (not ran_optimisations):
                ran_optimisations = True
                self._current_phase = 'target'
                self.optimise_targets()

    def generate_mutations_from(self, data: ConjectureData) -> None:
        if data.status >= Status.INVALID and self.health_check_state is None:
            initial_calls = self.call_count
            failed_mutations: int = 0
            while self.should_generate_more() and self.call_count <= initial_calls + 5 and (failed_mutations <= 5):
                groups = data.examples.mutator_groups
                if not groups:
                    break
                group = self.random.choice(groups)
                (start1, end1), (start2, end2) = self.random.sample(sorted(group), 2)
                if start1 <= start2 <= end2 <= end1 or start2 <= start1 <= end1 <= end2:
                    failed_mutations += 1
                    continue
                if start1 > start2:
                    (start1, end1), (start2, end2) = ((start2, end2), (start1, end1))
                assert end1 <= start2
                choices_obj = data.choices
                start, end = self.random.choice([(start1, end1), (start2, end2)])
                replacement = choices_obj[start:end]
                try:
                    new_data = self.cached_test_function_ir(
                        choices_obj[:start1] + replacement + choices_obj[end1:start2] + replacement + choices_obj[end2:],
                        error_on_discard=True
                    )
                except ContainsDiscard:
                    failed_mutations += 1
                    continue
                if new_data is Overrun:
                    failed_mutations += 1
                else:
                    assert isinstance(new_data, ConjectureResult)
                    if new_data.status >= data.status and choices_key(data.choices) != choices_key(new_data.choices) and all((k in new_data.target_observations and new_data.target_observations[k] >= v for k, v in data.target_observations.items())):
                        data = new_data
                        failed_mutations = 0
                    else:
                        failed_mutations += 1

    def optimise_targets(self) -> None:
        if not self.should_optimise:
            return
        from hypothesis.internal.conjecture.optimiser import Optimiser
        max_improvements: int = 10
        while True:
            prev_calls: int = self.call_count
            any_improvements: bool = False
            for target, data in list(self.best_examples_of_observed_targets.items()):
                optimiser = Optimiser(self, data, target, max_improvements=max_improvements)
                optimiser.run()
                if optimiser.improvements > 0:
                    any_improvements = True
            if self.interesting_examples:
                break
            max_improvements *= 2
            if any_improvements:
                continue
            if self.best_observed_targets:
                self.pareto_optimise()
            if prev_calls == self.call_count:
                break

    def pareto_optimise(self) -> None:
        if self.pareto_front is not None:
            from hypothesis.internal.conjecture.optimiser import ParetoOptimiser
            ParetoOptimiser(self).run()

    def _run(self) -> None:
        self._switch_to_hypothesis_provider = True
        with self._log_phase_statistics('reuse'):
            self.reuse_existing_examples()
        if self.reused_previously_shrunk_test_case:
            self.exit_with(ExitReason.finished)
        self._switch_to_hypothesis_provider = False
        with self._log_phase_statistics('generate'):
            self.generate_new_examples()
            if Phase.generate not in self.settings.phases:
                self._current_phase = 'target'
                self.optimise_targets()
        self._switch_to_hypothesis_provider = True
        with self._log_phase_statistics('shrink'):
            self.shrink_interesting_examples()
        self.exit_with(ExitReason.finished)

    def new_conjecture_data_ir(
        self,
        prefix: Sequence[ChoiceT],
        *,
        observer: Optional[DataObserver] = None,
        max_choices: Optional[int] = None
    ) -> ConjectureData:
        provider = HypothesisProvider if self._switch_to_hypothesis_provider else self.provider
        obs: DataObserver = observer or self.tree.new_observer()
        if not self.using_hypothesis_backend:
            obs = DataObserver()
        return ConjectureData(prefix=prefix, observer=obs, provider=provider, max_choices=max_choices, random=self.random)

    def shrink_interesting_examples(self) -> None:
        if Phase.shrink not in self.settings.phases or not self.interesting_examples:
            return
        self.debug('Shrinking interesting examples')
        self.finish_shrinking_deadline = time.perf_counter() + MAX_SHRINKING_SECONDS
        for prev_data in sorted(self.interesting_examples.values(), key=lambda d: sort_key(d.nodes)):
            assert prev_data.status == Status.INTERESTING
            data = self.new_conjecture_data_ir(prev_data.choices)
            self.test_function(data)
            if data.status != Status.INTERESTING:
                self.exit_with(ExitReason.flaky)
        self.clear_secondary_key()
        while len(self.shrunk_examples) < len(self.interesting_examples):
            target, example = min(
                ((k, v) for k, v in self.interesting_examples.items() if k not in self.shrunk_examples),
                key=lambda kv: (sort_key(kv[1].nodes), shortlex(repr(kv[0])))
            )
            self.debug(f'Shrinking {target!r}: {example.choices}')
            if not self.settings.report_multiple_bugs:
                self.shrink(example, lambda d: d.status == Status.INTERESTING)
                return

            def predicate(d: ConjectureResult) -> bool:
                if d.status < Status.INTERESTING:
                    return False
                d = cast(ConjectureResult, d)
                return d.interesting_origin == target

            self.shrink(example, predicate)
            self.shrunk_examples.add(target)

    def clear_secondary_key(self) -> None:
        if self.has_existing_examples():
            corpus: List[bytes] = sorted(self.settings.database.fetch(self.secondary_key), key=shortlex)
            for c in corpus:
                choices_obj = choices_from_bytes(c)
                if choices_obj is None:
                    self.settings.database.delete(self.secondary_key, c)
                    continue
                primary = {choices_to_bytes(v.choices) for v in self.interesting_examples.values()}
                cap = max(map(shortlex, primary))
                if shortlex(c) > cap:
                    break
                else:
                    self.cached_test_function_ir(choices_obj)
                    self.settings.database.delete(self.secondary_key, c)

    def shrink(
        self,
        example: ConjectureResult,
        predicate: Optional[Callable[[ConjectureResult], bool]] = None,
        allow_transition: Optional[bool] = None
    ) -> ConjectureResult:
        s = self.new_shrinker(example, predicate, allow_transition=allow_transition)
        s.shrink()
        return s.shrink_target

    def new_shrinker(
        self,
        example: ConjectureResult,
        predicate: Optional[Callable[[ConjectureResult], bool]] = None,
        allow_transition: Optional[bool] = None
    ) -> Shrinker:
        return Shrinker(self, example, predicate, allow_transition=allow_transition, explain=Phase.explain in self.settings.phases, in_target_phase=self._current_phase == 'target')

    def passing_choice_sequences(self, prefix: Sequence[ChoiceT] = ()) -> frozenset[Sequence[ChoiceNode]]:
        return frozenset((cast(ConjectureResult, result).nodes for key in self.__data_cache if (result := self.__data_cache[key]).status is Status.VALID and startswith(cast(ConjectureResult, result).nodes, prefix)))

class ContainsDiscard(Exception):
    pass