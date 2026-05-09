from __future__ import annotations
import asyncio
import datetime
import sys
import traceback
import uuid
import warnings
from collections import Counter
from types import GeneratorType, TracebackType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Type
import anyio
import httpx
from opentelemetry import propagate
from typing_extensions import TypeGuard
from prefect._internal.compatibility import deprecated
from prefect.client.schemas.objects import State, StateDetails, StateType
from prefect.exceptions import CancelledRun, CrashedRun, FailedRun, MissingContextError, MissingResult, PausedRun, TerminationSignal, UnfinishedRun
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
    # ...

@deprecated.deprecated_parameter('fetch', when=lambda fetch: fetch is not True, start_date='Oct 2024', end_date='Jan 2025', help='Please ensure you are awaiting the call to `result()` when calling in an async context.')
def get_state_result(state: State, raise_on_failure: bool = True, fetch: bool = True, retry_result_failure: bool = True) -> Any:
    """
    Get the result from a state.

    See `State.result()`
    """
    # ...

async def _get_state_result_data_with_retries(state: State, retry_result_failure: bool = True) -> Any:
    # ...

@sync_compatible
async def _get_state_result(state: State, raise_on_failure: bool, retry_result_failure: bool = True) -> Any:
    # ...

async def exception_to_crashed_state(exc: Exception, result_store: ResultStore = None) -> State:
    """
    Takes an exception that occurs _outside_ of user code and converts it to a
    'Crash' exception with a 'Crashed' state.
    """
    # ...

async def exception_to_failed_state(exc: Exception = None, result_store: ResultStore = None, write_result: bool = False, **kwargs) -> State:
    """
    Convenience function for creating `Failed` states from exceptions
    """
    # ...

async def return_value_to_state(retval: Any, result_store: ResultStore, key: str = None, expiration: datetime.datetime = None, write_result: bool = False) -> State:
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
    # ...

@sync_compatible
async def get_state_exception(state: State) -> Exception:
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
    # ...

@sync_compatible
async def raise_state_exception(state: State) -> None:
    """
    Given a FAILED or CRASHED state, raise the contained exception.
    """
    # ...

class StateGroup:
    # ...

def _traced(cls: Type[State], **kwargs) -> State:
    # ...

def Scheduled(cls: Type[State] = State, scheduled_time: datetime.datetime = None, **kwargs) -> State:
    # ...

def Completed(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Running(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Failed(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Crashed(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Cancelling(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Cancelled(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Pending(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Paused(cls: Type[State] = State, timeout_seconds: int = None, pause_expiration_time: datetime.datetime = None, reschedule: bool = False, pause_key: str = None, **kwargs) -> State:
    # ...

def Suspended(cls: Type[State] = State, timeout_seconds: int = None, pause_expiration_time: datetime.datetime = None, pause_key: str = None, **kwargs) -> State:
    # ...

def AwaitingRetry(cls: Type[State] = State, scheduled_time: datetime.datetime = None, **kwargs) -> State:
    # ...

def AwaitingConcurrencySlot(cls: Type[State] = State, scheduled_time: datetime.datetime = None, **kwargs) -> State:
    # ...

def Retrying(cls: Type[State] = State, **kwargs) -> State:
    # ...

def Late(cls: Type[State] = State, scheduled_time: datetime.datetime = None, **kwargs) -> State:
    # ...
