from __future__ import annotations
import asyncio
import datetime
import sys
import traceback
import uuid
import warnings
from collections import Counter
from types import GeneratorType, TracebackType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Type, Union
import anyio
import httpx
from opentelemetry import propagate
from typing_extensions import TypeGuard
from prefect._internal.compatibility import deprecated
from prefect.client.schemas.objects import State, StateDetails, StateType
from prefect.exceptions import (
    CancelledRun,
    CrashedRun,
    FailedRun,
    MissingContextError,
    MissingResult,
    PausedRun,
    TerminationSignal,
    UnfinishedRun,
)
from prefect.logging.loggers import get_logger, get_run_logger
from prefect.types._datetime import DateTime, Duration, now
from prefect.utilities.annotations import BaseAnnotation
from prefect.utilities.asyncutils import in_async_main_thread, sync_compatible
from prefect.utilities.collections import ensure_iterable

if TYPE_CHECKING:
    import logging
    from prefect.client.schemas.actions import StateCreate
    from prefect.results import R, ResultStore

logger = get_logger('states')


def to_state_create(state: State) -> StateCreate:
    """
    Convert the state to a `StateCreate` type which can be used to set the state of
    a run in the API.

    This method will drop this state's `data` if it is not a result type. Only
    results should be sent to the API. Other data is only available locally.
    """
    from prefect.client.schemas.actions import StateCreate
    from prefect.results import ResultRecord, should_persist_result

    if isinstance(state.data, ResultRecord) and should_persist_result():
        data = state.data.metadata
    else:
        data = None
    return StateCreate(
        type=state.type,
        name=state.name,
        message=state.message,
        data=data,
        state_details=state.state_details,
    )


@deprecated.deprecated_parameter(
    'fetch',
    when=lambda fetch: fetch is not True,
    start_date='Oct 2024',
    end_date='Jan 2025',
    help='Please ensure you are awaiting the call to `result()` when calling in an async context.',
)
def get_state_result(
    state: State,
    raise_on_failure: bool = True,
    fetch: bool = True,
    retry_result_failure: bool = True,
) -> Any:
    """
    Get the result from a state.

    See `State.result()`
    """
    if not fetch and in_async_main_thread():
        warnings.warn(
            'State.result() was called from an async context but not awaited. This method will be updated to return a coroutine by default in the future. Pass `fetch=True` and `await` the call to get rid of this warning.',
            DeprecationWarning,
            stacklevel=2,
        )
        return state.data
    else:
        return _get_state_result(state, raise_on_failure=raise_on_failure, retry_result_failure=retry_result_failure)


RESULT_READ_MAXIMUM_ATTEMPTS: int = 10
RESULT_READ_RETRY_DELAY: float = 0.25


async def _get_state_result_data_with_retries(
    state: State, retry_result_failure: bool = True
) -> Any:
    from prefect._result_records import ResultRecordMetadata
    from prefect.results import ResultStore

    if not retry_result_failure:
        max_attempts = 1
    else:
        max_attempts = RESULT_READ_MAXIMUM_ATTEMPTS
    for i in range(1, max_attempts + 1):
        try:
            if isinstance(state.data, ResultRecordMetadata):
                record = await ResultStore._from_metadata(state.data)
                return record.result
            else:
                return await state.data.get()
        except Exception as e:
            if i == max_attempts:
                raise
            logger.debug(
                'Exception %r while reading result, retry %s/%s in %ss...',
                e,
                i,
                max_attempts,
                RESULT_READ_RETRY_DELAY,
            )
            await asyncio.sleep(RESULT_READ_RETRY_DELAY)


@sync_compatible
async def _get_state_result(
    state: State, raise_on_failure: bool, retry_result_failure: bool = True
) -> Any:
    """
    Internal implementation for `get_state_result` without async backwards compatibility
    """
    from prefect.results import ResultRecord, ResultRecordMetadata
    from prefect.results import ResultStore

    if state.is_paused():
        raise PausedRun('Run is paused, its result is not available.', state=state)
    if not state.is_final():
        raise UnfinishedRun(f'Run is in {state.type.name} state, its result is not available.')
    if raise_on_failure and (state.is_crashed() or state.is_failed() or state.is_cancelled()):
        raise await get_state_exception(state)
    if isinstance(state.data, ResultRecordMetadata):
        result = await _get_state_result_data_with_retries(state, retry_result_failure=retry_result_failure)
    elif isinstance(state.data, ResultRecord):
        result = state.data.result
    elif state.data is None:
        if state.is_failed() or state.is_crashed() or state.is_cancelled():
            return await get_state_exception(state)
        else:
            raise MissingResult(
                'State data is missing. Typically, this occurs when result persistence is disabled and the state has been retrieved from the API.'
            )
    else:
        result = state.data
    return result


def format_exception(exc: BaseException, tb: Optional[TracebackType] = None) -> str:
    exc_type = type(exc)
    if tb is not None:
        formatted = ''.join(list(traceback.format_exception(exc_type, exc, tb=tb)))
    else:
        formatted = f'{exc_type.__name__}: {exc}'
    if exc_type.__module__.startswith('prefect.'):
        formatted = formatted.replace(f'{exc_type.__module__}.{exc_type.__name__}', exc_type.__name__)
    return formatted


async def exception_to_crashed_state(
    exc: BaseException, result_store: Optional[ResultStore] = None
) -> CrashedRun:
    """
    Takes an exception that occurs _outside_ of user code and converts it to a
    'Crash' exception with a 'Crashed' state.
    """
    state_message: Optional[str] = None
    if isinstance(exc, anyio.get_cancelled_exc_class()):
        state_message = 'Execution was cancelled by the runtime environment.'
    elif isinstance(exc, KeyboardInterrupt):
        state_message = 'Execution was aborted by an interrupt signal.'
    elif isinstance(exc, TerminationSignal):
        state_message = 'Execution was aborted by a termination signal.'
    elif isinstance(exc, SystemExit):
        state_message = 'Execution was aborted by Python system exit call.'
    elif isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        try:
            request = exc.request
        except RuntimeError:
            state_message = f'Request failed while attempting to contact the server: {format_exception(exc)}'
        else:
            state_message = f'Request to {request.url} failed: {format_exception(exc)}.'
    else:
        state_message = f'Execution was interrupted by an unexpected exception: {format_exception(exc)}'
    if result_store:
        key = uuid.uuid4().hex
        data = result_store.create_result_record(exc, key=key)
    else:
        data = exc
    return CrashedRun(message=state_message, data=data)


async def exception_to_failed_state(
    exc: Optional[BaseException] = None,
    result_store: Optional[ResultStore] = None,
    write_result: bool = False,
    **kwargs: Any,
) -> FailedRun:
    """
    Convenience function for creating `Failed` states from exceptions
    """
    try:
        local_logger = get_run_logger()
    except MissingContextError:
        local_logger = logger
    if not exc:
        _, exc, _ = sys.exc_info()
        if exc is None:
            raise ValueError('Exception was not passed and no active exception could be found.')
    else:
        pass
    if result_store:
        key = uuid.uuid4().hex
        data = result_store.create_result_record(exc, key=key)
        if write_result:
            try:
                await result_store.apersist_result_record(data)
            except Exception as nested_exc:
                local_logger.warning(
                    'Failed to write result: %s Execution will continue, but the result has not been written',
                    nested_exc,
                )
    else:
        data = exc
    existing_message: str = kwargs.pop('message', '')
    if existing_message and (not existing_message.endswith(' ')):
        existing_message += ' '
    message = existing_message + format_exception(exc)
    state = FailedRun(data=data, message=message, **kwargs)
    state.state_details.retriable = False
    return state


async def return_value_to_state(
    retval: Any,
    result_store: ResultStore,
    key: Optional[str] = None,
    expiration: Optional[datetime.datetime] = None,
    write_result: bool = False,
) -> State:
    """
    Given a return value from a user's function, create a `State` the run should
    be placed in.

    - If data is returned, we create a 'COMPLETED' state with the data
    - If a single, manually created state is returned, we use that state as given
        (manual creation is determined by the lack of ids)
    - If an upstream state or iterable of upstream states is returned, we apply the
        aggregate rule

    The aggregate rule says that given multiple states we will determine the final state
    such that:

    - If any states are not COMPLETED the final state is FAILED
    - If all of the states are COMPLETED the final state is COMPLETED
    - The states will be placed in the final state `data` attribute

    Callers should resolve all futures into states before passing return values to this
    function.
    """
    from prefect.results import ResultRecord, ResultRecordMetadata

    try:
        local_logger = get_run_logger()
    except MissingContextError:
        local_logger = logger
    if isinstance(retval, State) and (not retval.state_details.flow_run_id) and (not retval.state_details.task_run_id):
        state: State = retval
        if not isinstance(state.data, (ResultRecord, ResultRecordMetadata)):
            result_record = result_store.create_result_record(state.data, key=key, expiration=expiration)
            if write_result:
                try:
                    await result_store.apersist_result_record(result_record)
                except Exception as exc:
                    local_logger.warning(
                        'Encountered an error while persisting result: %s Execution will continue, but the result has not been persisted',
                        exc,
                    )
            state.data = result_record
        return state
    if isinstance(retval, State) or is_state_iterable(retval):
        states = StateGroup(ensure_iterable(retval))
        if states.all_completed():
            new_state_type = StateType.COMPLETED
        elif states.any_cancelled():
            new_state_type = StateType.CANCELLED
        elif states.any_paused():
            new_state_type = StateType.PAUSED
        else:
            new_state_type = StateType.FAILED
        if states.all_completed():
            message = 'All states completed.'
        elif states.any_cancelled():
            message = f'{states.cancelled_count}/{states.total_count} states cancelled.'
        elif states.any_paused():
            message = f'{states.paused_count}/{states.total_count} states paused.'
        elif states.any_failed():
            message = f'{states.fail_count}/{states.total_count} states failed.'
        elif not states.all_final():
            message = f'{states.not_final_count}/{states.total_count} states are not final.'
        else:
            message = 'Given states: ' + states.counts_message()
        result_record = result_store.create_result_record(retval, key=key, expiration=expiration)
        if write_result:
            try:
                await result_store.apersist_result_record(result_record)
            except Exception as exc:
                local_logger.warning(
                    'Encountered an error while persisting result: %s Execution will continue, but the result has not been persisted',
                    exc,
                )
        return State(type=new_state_type, message=message, data=result_record)
    if isinstance(retval, GeneratorType):
        data = list(retval)
    else:
        data = retval
    if isinstance(data, ResultRecord):
        return Completed(data=data)
    else:
        result_record = result_store.create_result_record(data, key=key, expiration=expiration)
        if write_result:
            try:
                await result_store.apersist_result_record(result_record)
            except Exception as exc:
                local_logger.warning(
                    'Encountered an error while persisting result: %s Execution will continue, but the result has not been persisted',
                    exc,
                )
        return Completed(data=result_record)


@sync_compatible
async def get_state_exception(state: State) -> BaseException:
    """
    If not given a FAILED or CRASHED state, this raise a value error.

    If the state result is a state, its exception will be returned.

    If the state result is an iterable of states, the exception of the first failure
    will be returned.

    If the state result is a string, a wrapper exception will be returned with the
    string as the message.

    If the state result is null, a wrapper exception will be returned with the state
    message attached.

    If the state result is not of a known type, a `TypeError` will be returned.

    When a wrapper exception is returned, the type will be:
        - `FailedRun` if the state type is FAILED.
        - `CrashedRun` if the state type is CRASHED.
        - `CancelledRun` if the state type is CANCELLED.
    """
    from prefect._result_records import ResultRecord, ResultRecordMetadata
    from prefect.results import ResultStore

    if state.is_failed():
        wrapper: Type[BaseException] = FailedRun
        default_message: str = 'Run failed.'
    elif state.is_crashed():
        wrapper = CrashedRun
        default_message = 'Run crashed.'
    elif state.is_cancelled():
        wrapper = CancelledRun
        default_message = 'Run cancelled.'
    else:
        raise ValueError(f'Expected failed or crashed state got {state!r}.')

    if isinstance(state.data, ResultRecord):
        result = state.data.result
    elif isinstance(state.data, ResultRecordMetadata):
        record = await ResultStore._from_metadata(state.data)
        result = record.result
    elif state.data is None:
        result = None
    else:
        result = state.data

    if result is None:
        return wrapper(state.message or default_message)
    if isinstance(result, BaseException):
        return result
    elif isinstance(result, str):
        return wrapper(result)
    elif isinstance(result, State):
        return await get_state_exception(result)
    elif is_state_iterable(result):
        for state_item in result:
            if state_item.is_failed() or state_item.is_crashed() or state_item.is_cancelled():
                return await get_state_exception(state_item)
        raise ValueError('Failed state result was an iterable of states but none were failed.')
    else:
        raise TypeError(
            f'Unexpected result for failed state: {result!r} —— {type(result).__name__} cannot be resolved into an exception'
        )


@sync_compatible
async def raise_state_exception(state: State) -> None:
    """
    Given a FAILED or CRASHED state, raise the contained exception.
    """
    if not (state.is_failed() or state.is_crashed() or state.is_cancelled()):
        return
    raise await get_state_exception(state)


def is_state_iterable(obj: Any) -> TypeGuard[Iterable[State]]:
    """
    Check if a the given object is an iterable of states types

    Supported iterables are:
    - set
    - list
    - tuple

    Other iterables will return `False` even if they contain states.
    """
    if not isinstance(obj, BaseAnnotation) and isinstance(obj, (list, set, tuple)) and obj:
        return all(isinstance(o, State) for o in obj)
    else:
        return False


class StateGroup:
    states: Iterable[State]
    type_counts: Counter[StateType, int]
    total_count: int
    cancelled_count: int
    final_count: int
    not_final_count: int
    paused_count: int

    def __init__(self, states: Iterable[State]) -> None:
        self.states = states
        self.type_counts = self._get_type_counts(states)
        self.total_count = len(states)
        self.cancelled_count = self.type_counts[StateType.CANCELLED]
        self.final_count = sum(state.is_final() for state in states)
        self.not_final_count = self.total_count - self.final_count
        self.paused_count = self.type_counts[StateType.PAUSED]

    @property
    def fail_count(self) -> int:
        return self.type_counts[StateType.FAILED] + self.type_counts[StateType.CRASHED]

    def all_completed(self) -> bool:
        return self.type_counts[StateType.COMPLETED] == self.total_count

    def any_cancelled(self) -> bool:
        return self.cancelled_count > 0

    def any_failed(self) -> bool:
        return self.type_counts[StateType.FAILED] > 0 or self.type_counts[StateType.CRASHED] > 0

    def any_paused(self) -> bool:
        return self.paused_count > 0

    def all_final(self) -> bool:
        return self.final_count == self.total_count

    def counts_message(self) -> str:
        count_messages = [f'total={self.total_count}']
        if self.not_final_count:
            count_messages.append(f'not_final={self.not_final_count}')
        count_messages += [
            f'{state_type.value!r}={count}'
            for state_type, count in self.type_counts.items()
            if count
        ]
        return ', '.join(count_messages)

    @staticmethod
    def _get_type_counts(states: Iterable[State]) -> Counter[StateType, int]:
        return Counter(state.type for state in states)

    def __repr__(self) -> str:
        return f'StateGroup<{self.counts_message()}>'


def _traced(cls: Type[State], **kwargs: Any) -> State:
    state_details = StateDetails.model_validate(kwargs.pop('state_details', {}))
    carrier: Dict[str, Any] = {}
    propagate.inject(carrier)
    state_details.traceparent = carrier.get('traceparent')
    return cls(**kwargs, state_details=state_details)


def Scheduled(
    cls: Type[State] = State,
    scheduled_time: Optional[datetime.datetime] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `Scheduled` states.

    Returns:
        State: a Scheduled state
    """
    state_details = StateDetails.model_validate(kwargs.pop('state_details', {}))
    if scheduled_time is None:
        scheduled_time = now()
    elif state_details.scheduled_time:
        raise ValueError('An extra scheduled_time was provided in state_details')
    state_details.scheduled_time = scheduled_time
    return _traced(cls, type=StateType.SCHEDULED, state_details=state_details, **kwargs)


def Completed(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Completed` states.

    Returns:
        State: a Completed state
    """
    return _traced(cls, type=StateType.COMPLETED, **kwargs)


def Running(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Running` states.

    Returns:
        State: a Running state
    """
    return _traced(cls, type=StateType.RUNNING, **kwargs)


def Failed(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Failed` states.

    Returns:
        State: a Failed state
    """
    return _traced(cls, type=StateType.FAILED, **kwargs)


def Crashed(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Crashed` states.

    Returns:
        State: a Crashed state
    """
    return _traced(cls, type=StateType.CRASHED, **kwargs)


def Cancelling(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Cancelling` states.

    Returns:
        State: a Cancelling state
    """
    return _traced(cls, type=StateType.CANCELLING, **kwargs)


def Cancelled(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Cancelled` states.

    Returns:
        State: a Cancelled state
    """
    return _traced(cls, type=StateType.CANCELLED, **kwargs)


def Pending(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Pending` states.

    Returns:
        State: a Pending state
    """
    return _traced(cls, type=StateType.PENDING, **kwargs)


def Paused(
    cls: Type[State] = State,
    timeout_seconds: Optional[int] = None,
    pause_expiration_time: Optional[datetime.datetime] = None,
    reschedule: bool = False,
    pause_key: Optional[str] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `Paused` states.

    Returns:
        State: a Paused state
    """
    state_details = StateDetails.model_validate(kwargs.pop('state_details', {}))
    if state_details.pause_timeout:
        raise ValueError('An extra pause timeout was provided in state_details')
    if pause_expiration_time is not None and timeout_seconds is not None:
        raise ValueError('Cannot supply both a pause_expiration_time and timeout_seconds')
    if pause_expiration_time is None and timeout_seconds is None:
        pass
    else:
        state_details.pause_timeout = (
            DateTime.instance(pause_expiration_time)
            if pause_expiration_time
            else now() + Duration(seconds=timeout_seconds or 0)
        )
    state_details.pause_reschedule = reschedule
    state_details.pause_key = pause_key
    return _traced(cls, type=StateType.PAUSED, state_details=state_details, **kwargs)


def Suspended(
    cls: Type[State] = State,
    timeout_seconds: Optional[int] = None,
    pause_expiration_time: Optional[datetime.datetime] = None,
    pause_key: Optional[str] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `Suspended` states.

    Returns:
        State: a Suspended state
    """
    return Paused(
        cls=cls,
        name='Suspended',
        reschedule=True,
        timeout_seconds=timeout_seconds,
        pause_expiration_time=pause_expiration_time,
        pause_key=pause_key,
        **kwargs,
    )


def AwaitingRetry(
    cls: Type[State] = State,
    scheduled_time: Optional[datetime.datetime] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `AwaitingRetry` states.

    Returns:
        State: a AwaitingRetry state
    """
    return Scheduled(
        cls=cls,
        scheduled_time=scheduled_time,
        name='AwaitingRetry',
        **kwargs,
    )


def AwaitingConcurrencySlot(
    cls: Type[State] = State,
    scheduled_time: Optional[datetime.datetime] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `AwaitingConcurrencySlot` states.

    Returns:
        State: a AwaitingConcurrencySlot state
    """
    return Scheduled(
        cls=cls,
        scheduled_time=scheduled_time,
        name='AwaitingConcurrencySlot',
        **kwargs,
    )


def Retrying(cls: Type[State] = State, **kwargs: Any) -> State:
    """Convenience function for creating `Retrying` states.

    Returns:
        State: a Retrying state
    """
    return _traced(cls, type=StateType.RUNNING, name='Retrying', **kwargs)


def Late(
    cls: Type[State] = State,
    scheduled_time: Optional[datetime.datetime] = None,
    **kwargs: Any,
) -> State:
    """Convenience function for creating `Late` states.

    Returns:
        State: a Late state
    """
    return Scheduled(
        cls=cls,
        scheduled_time=scheduled_time,
        name='Late',
        **kwargs,
    )
