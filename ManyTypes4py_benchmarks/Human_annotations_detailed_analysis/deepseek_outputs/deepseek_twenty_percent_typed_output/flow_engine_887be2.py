from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.context
import os
import time
from contextlib import ExitStack, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    Generic,
    Iterable,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

from anyio import CancelScope
from opentelemetry import propagate, trace
from typing_extensions import ParamSpec

from prefect import Task
from prefect.client.orchestration import PrefectClient, SyncPrefectClient, get_client
from prefect.client.schemas import FlowRun, TaskRun
from prefect.client.schemas.filters import FlowRunFilter
from prefect.client.schemas.sorting import FlowRunSort
from prefect.concurrency.context import ConcurrencyContext
from prefect.concurrency.v1.context import ConcurrencyContext as ConcurrencyContextV1
from prefect.context import (
    AsyncClientContext,
    FlowRunContext,
    SettingsContext,
    SyncClientContext,
    TagsContext,
    get_settings_context,
    hydrated_context,
    serialize_context,
)
from prefect.exceptions import (
    Abort,
    MissingFlowError,
    Pause,
    PrefectException,
    TerminationSignal,
    UpstreamTaskError,
)
from prefect.flows import (
    Flow,
    load_flow_from_entrypoint,
    load_flow_from_flow_run,
    load_function_and_convert_to_flow,
)
from prefect.futures import PrefectFuture, resolve_futures_to_states
from prefect.logging.loggers import (
    flow_run_logger,
    get_logger,
    get_run_logger,
    patch_print,
)
from prefect.results import (
    ResultStore,
    get_result_store,
    should_persist_result,
)
from prefect.settings import PREFECT_DEBUG_MODE
from prefect.settings.context import get_current_settings
from prefect.settings.models.root import Settings
from prefect.states import (
    Failed,
    Pending,
    Running,
    State,
    exception_to_crashed_state,
    exception_to_failed_state,
    return_value_to_state,
)
from prefect.telemetry.run_telemetry import (
    LABELS_TRACEPARENT_KEY,
    TRACEPARENT_KEY,
    OTELSetter,
    RunTelemetry,
)
from prefect.types import KeyValueLabels
from prefect.utilities._engine import get_hook_name, resolve_custom_flow_run_name
from prefect.utilities.annotations import NotSet
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.callables import (
    call_with_parameters,
    cloudpickle_wrapped_call,
    get_call_parameters,
    parameters_to_args_kwargs,
)
from prefect.utilities.collections import visit_collection
from prefect.utilities.engine import (
    capture_sigterm,
    link_state_to_result,
    propose_state,
    propose_state_sync,
    resolve_to_final_result,
)
from prefect.utilities.timeout import timeout, timeout_async
from prefect.utilities.urls import url_for

P = ParamSpec("P")
R = TypeVar("R")


class FlowRunTimeoutError(TimeoutError):
    """Raised when a flow run exceeds its defined timeout."""


def load_flow_run(flow_run_id: UUID) -> FlowRun:
    client = get_client(sync_client=True)
    flow_run = client.read_flow_run(flow_run_id)
    return flow_run


def load_flow(flow_run: FlowRun) -> Flow[..., Any]:
    entrypoint = os.environ.get("PREFECT__FLOW_ENTRYPOINT")

    if entrypoint:
        # we should not accept a placeholder flow at runtime
        try:
            flow = load_flow_from_entrypoint(entrypoint, use_placeholder_flow=False)
        except MissingFlowError:
            flow = load_function_and_convert_to_flow(entrypoint)
    else:
        flow = run_coro_as_sync(
            load_flow_from_flow_run(flow_run, use_placeholder_flow=False)
        )
    return flow


def load_flow_and_flow_run(flow_run_id: UUID) -> tuple[FlowRun, Flow[..., Any]]:
    flow_run = load_flow_run(flow_run_id)
    flow = load_flow(flow_run)
    return flow_run, flow


@dataclass
class BaseFlowRunEngine(Generic[P, R]):
    flow: Union[Flow[P, R], Flow[P, Coroutine[Any, Any, R]]]
    parameters: Optional[Dict[str, Any]] = None
    flow_run: Optional[FlowRun] = None
    flow_run_id: Optional[UUID] = None
    logger: logging.Logger = field(default_factory=lambda: get_logger("engine"))
    wait_for: Optional[Iterable[PrefectFuture[Any]]] = None
    context: Optional[dict[str, Any]] = None
    # holds the return value from the user code
    _return_value: Union[R, Type[NotSet]] = NotSet
    # holds the exception raised by the user code, if any
    _raised: Union[Exception, Type[NotSet]] = NotSet
    _is_started: bool = False
    short_circuit: bool = False
    _flow_run_name_set: bool = False
    _telemetry: RunTelemetry = field(default_factory=RunTelemetry)

    def __post_init__(self) -> None:
        if self.flow is None and self.flow_run_id is None:
            raise ValueError("Either a flow or a flow_run_id must be provided.")

        if self.parameters is None:
            self.parameters = {}

    @property
    def state(self) -> State:
        return self.flow_run.state  # type: ignore

    def is_running(self) -> bool:
        if getattr(self, "flow_run", None) is None:
            return False
        return getattr(self, "flow_run").state.is_running()

    def is_pending(self) -> bool:
        if getattr(self, "flow_run", None) is None:
            return False  # TODO: handle this differently?
        return getattr(self, "flow_run").state.is_pending()

    def cancel_all_tasks(self) -> None:
        if hasattr(self.flow.task_runner, "cancel_all"):
            self.flow.task_runner.cancel_all()  # type: ignore

    def _update_otel_labels(
        self, span: trace.Span, client: Union[PrefectClient, SyncPrefectClient]
    ) -> None:
        parent_flow_run_ctx = FlowRunContext.get()

        if parent_flow_run_ctx and parent_flow_run_ctx.flow_run:
            if traceparent := parent_flow_run_ctx.flow_run.labels.get(
                LABELS_TRACEPARENT_KEY
            ):
                carrier: KeyValueLabels = {TRACEPARENT_KEY: traceparent}
                propagate.get_global_textmap().inject(
                    carrier={TRACEPARENT_KEY: traceparent},
                    setter=OTELSetter(),
                )

            else:
                carrier: KeyValueLabels = {}
                propagate.get_global_textmap().inject(
                    carrier,
                    context=trace.set_span_in_context(span),
                    setter=OTELSetter(),
                )
            if carrier.get(TRACEPARENT_KEY):
                if self.flow_run:
                    client.update_flow_run_labels(
                        flow_run_id=self.flow_run.id,
                        labels={LABELS_TRACEPARENT_KEY: carrier[TRACEPARENT_KEY]},
                    )
                else:
                    self.logger.info(
                        f"Tried to set traceparent {carrier[TRACEPARENT_KEY]} for flow run, but None was found"
                    )


@dataclass
class FlowRunEngine(BaseFlowRunEngine[P, R]):
    _client: Optional[SyncPrefectClient] = None
    flow_run: FlowRun | None = None
    parameters: dict[str, Any] | None = None

    @property
    def client(self) -> SyncPrefectClient:
        if not self._is_started or self._client is None:
            raise RuntimeError("Engine has not started.")
        return self._client

    def _resolve_parameters(self) -> None:
        if not self.parameters:
            return

        resolved_parameters = {}
        for parameter, value in self.parameters.items():
            try:
                resolved_parameters[parameter] = visit_collection(
                    value,
                    visit_fn=resolve_to_final_result,
                    return_data=True,
                    max_depth=-1,
                    remove_annotations=True,
                    context={"parameter_name": parameter},
                )
            except UpstreamTaskError:
                raise
            except Exception as exc:
                raise PrefectException(
                    f"Failed to resolve inputs in parameter {parameter!r}. If your"
                    " parameter type is not supported, consider using the `quote`"
                    " annotation to skip resolution of inputs."
                ) from exc

        self.parameters = resolved_parameters

    def _wait_for_dependencies(self) -> None:
        if not self.wait_for:
            return

        visit_collection(
            self.wait_for,
            visit_fn=resolve_to_final_result,
            return_data=False,
            max_depth=-1,
            remove_annotations=True,
            context={},
        )

    def begin_run(self) -> State:
        try:
            self._resolve_parameters()
            self._wait_for_dependencies()
        except UpstreamTaskError as upstream_exc:
            state = self.set_state(
                Pending(
                    name="NotReady",
                    message=str(upstream_exc),
                ),
                # if orchestrating a run already in a pending state, force orchestration to
                # update the state name
                force=self.state.is_pending(),
            )
            return state

        # validate prior to context so that context receives validated params
        if self.flow.should_validate_parameters:
            try:
                self.parameters = self.flow.validate_parameters(self.parameters or {})
            except Exception as exc:
                message = "Validation of flow parameters failed with error:"
                self.logger.error("%s %s", message, exc)
                self.handle_exception(
                    exc,
                    msg=message,
                    result_store=get_result_store().update_for_flow(
                        self.flow, _sync=True
                    ),
                )
                self.short_circuit = True

        new_state = Running()
        state = self.set_state(new_state)
        while state.is_pending():
            time.sleep(0.2)
            state = self.set_state(new_state)
        return state

    def set_state(self, state: State, force: bool = False) -> State:
        """ """
        # prevents any state-setting activity
        if self.short_circuit:
            return self.state

        state = propose_state_sync(
            self.client, state, flow_run_id=self.flow_run.id, force=force
        )  # type: ignore
        self.flow_run.state = state  # type: ignore
        self.flow_run.state_name = state.name  # type: ignore
        self.flow_run.state_type = state.type  # type: ignore

        self._telemetry.update_state(state)
        self.call_hooks(state)
        return state

    def result(self, raise_on_failure: bool = True) -> Union[R, State, None]:
        if self._return_value is not NotSet and not isinstance(
            self._return_value, State
        ):
            _result = self._return_value

            if asyncio.iscoroutine(_result):
                # getting the value for a BaseResult may return an awaitable
                # depending on whether the parent frame is sync or not
                _result = run_coro_as_sync(_result)
            return _result

        if self._raised is not NotSet:
            if raise_on_failure:
                raise self._raised
            return self._raised

        # This is a fall through case which leans on the existing state result mechanics to get the
        # return value. This is necessary because we currently will return a State object if the
        # the State was Prefect-created.
        # TODO: Remove the need to get the result from a State except in cases where the return value
        # is a State object.
        _result = self.state.result(raise_on_failure=raise_on_failure, fetch=True)  # type: ignore
        # state.result is a `sync_compatible` function that may or may not return an awaitable
        # depending on whether the parent frame is sync or not
        if asyncio.iscoroutine(_result):
            _result = run_coro_as_sync(_result)
        return _result

    def handle_success(self, result: R) -> R:
        result_store = getattr(FlowRunContext.get(), "result_store", None)
        if result_store is None:
            raise ValueError("Result store is not set")
        resolved_result = resolve_futures_to_states(result)
        terminal_state = run_coro_as_sync(
            return_value_to_state(
                resolved_result,
                result_store=result_store,
                write_result=should_persist_result(),
            )
        )
        self.set_state(terminal_state)
        self._return_value = resolved_result

        self._telemetry.end_span_on_success()

        return result

    def handle_exception(
        self,
        exc: Exception,
        msg: Optional[str] = None,
        result_store: Optional[ResultStore] = None,
    ) -> State:
        context = FlowRunContext.get()
        terminal_state = cast(
            State,
            run_coro_as_sync(
                exception_to_failed_state(
                    exc,
                    message=msg or "Flow run encountered an exception:",
                    result_store=result_store or getattr(context, "result_store", None),
                    write_result=True,
                )
            ),
        )
        state = self.set_state(terminal_state)
        if self.state.is_scheduled():
            self.logger.info(
                (
                    f"Received non-final state {state.name!r} when proposing final"
                    f" state {terminal_state.name!r} and will attempt to run again..."
                ),
            )
            state = self.set_state(Running())
        self._raised = exc
        self._telemetry.record_exception(exc)
        self._telemetry.end_span_on_failure(state.message)

        return state

    def handle_timeout(self, exc: TimeoutError) -> None:
        if isinstance(exc, FlowRunTimeoutError):
            message = (
                f"Flow run exceeded timeout of {self.flow.timeout_seconds} second(s)"
            )
        else:
            message = f"Flow run failed due to timeout: {exc!r}"
        self.logger.error(message)
        state = Failed(
            data=exc,
            message=message,
            name="TimedOut",
        )
        self.set_state(state)
        self._raised = exc
        self._telemetry.record_exception(exc)
        self._telemetry.end_span_on_failure(message)

    def handle_crash(self, exc: BaseException) -> None:
        state = run_coro_as_sync(exception_to_crashed_state(exc))
        self.logger.error(f"Crash detected! {state.message}")
        self.logger.debug("Crash details:", exc_info=exc)
        self.set_state(state, force=True)
        self._raised = exc
        self._telemetry.record_exception(exc)
        self._telemetry.end_span_on_failure(state.message if state else None)

    def load_subflow_run(
        self,
        parent_task_run: TaskRun,
        client: SyncPrefectClient,
        context: FlowRunContext,
    ) -> Union[FlowRun, None]:
        """
        This method attempts to load an existing flow run for a subflow task
        run, if appropriate.

        If the parent task run is in a final but not COMPLETED state, and not
        being rerun, then we attempt to load an existing flow run instead of
        creating a new one. This will prevent the engine from running the
        subflow again.

        If no existing flow run is found, or if the subflow should be rerun,
        then no flow run is returned.
        """

        # check if the parent flow run is rerunning
        rerunning = (
            context.flow_run.run_count > 1
            if getattr(context, "flow_run", None)
            and isinstance(context.flow_run, FlowRun)
            else False
        )

        # if the parent task run is in a final but not completed state, and
        # not rerunning, then retrieve the most recent flow run instead of
        # creating a new one. This effectively loads a cached flow run for
        # situations where we are confident the flow should not be run
        # again.
        assert isinstance(parent_task_run.state, State)
        if parent_task_run.state.is_final() and not (
            rerunning and not parent_task_run.state.is_completed()
        ):
            # return the most recent flow run, if it exists
            flow_runs = client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    parent_task_run_id={"any_": [parent_task_run.id]}
                ),
                sort=FlowRunSort.EXPECTED_START_TIME_ASC,
                limit=1,
            )
            if flow_runs:
                loaded_flow_run = flow_runs[-1]
                self._return_value = loaded_flow_run.state
                return loaded_flow_run
        return None

    def create_flow_run(self, client: SyncPrefectClient) -> FlowRun:
        flow_run_ctx = FlowRunContext.get()
        parameters = self.parameters or {}

        parent_task_run = None

        # this is a subflow run
        if flow_run_ctx:
            # add a task to a parent flow run that represents the execution of a subflow run
            parent_task = Task(
                name=self.flow.name, fn=self.flow.fn, version=self.flow.version
            )

            parent_task_run = run_coro_as_sync(
                parent_task.create_run(
                    flow_run_context=flow_run_ctx,
                    parameters=self.parameters,
                    wait_for=self.wait_for,
                )
            )

            # check if there is already a flow run for this subflow
            if subflow_run := self.load_subflow_run(
                parent_task_run=parent_task_run, client=client, context=flow_run_ctx
            ):
                return subflow_run

        return client.create