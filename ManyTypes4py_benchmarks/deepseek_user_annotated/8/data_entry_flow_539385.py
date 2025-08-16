"""Classes to help gather user submissions."""

from __future__ import annotations

import abc
import asyncio
from collections import defaultdict
from collections.abc import Callable, Container, Hashable, Iterable, Mapping
from contextlib import suppress
import copy
from dataclasses import dataclass
from enum import StrEnum
import logging
from types import MappingProxyType
from typing import Any, Generic, Optional, Required, TypedDict, TypeVar, cast, Union

import voluptuous as vol

from .core import HomeAssistant, callback
from .exceptions import HomeAssistantError
from .helpers.frame import ReportBehavior, report_usage
from .loader import async_suggest_report_issue
from .util import uuid as uuid_util

_LOGGER = logging.getLogger(__name__)


class FlowResultType(StrEnum):
    """Result type for a data entry flow."""

    FORM = "form"
    CREATE_ENTRY = "create_entry"
    ABORT = "abort"
    EXTERNAL_STEP = "external"
    EXTERNAL_STEP_DONE = "external_done"
    SHOW_PROGRESS = "progress"
    SHOW_PROGRESS_DONE = "progress_done"
    MENU = "menu"


# Event that is fired when a flow is progressed via external or progress source.
EVENT_DATA_ENTRY_FLOW_PROGRESSED = "data_entry_flow_progressed"

FLOW_NOT_COMPLETE_STEPS = {
    FlowResultType.FORM,
    FlowResultType.EXTERNAL_STEP,
    FlowResultType.EXTERNAL_STEP_DONE,
    FlowResultType.SHOW_PROGRESS,
    FlowResultType.SHOW_PROGRESS_DONE,
    FlowResultType.MENU,
}


STEP_ID_OPTIONAL_STEPS = {
    FlowResultType.EXTERNAL_STEP,
    FlowResultType.FORM,
    FlowResultType.MENU,
    FlowResultType.SHOW_PROGRESS,
}


_FlowContextT = TypeVar("_FlowContextT", bound="FlowContext", default="FlowContext")
_FlowResultT = TypeVar(
    "_FlowResultT", bound="FlowResult[Any, Any]", default="FlowResult"
)
_HandlerT = TypeVar("_HandlerT", default=str)


@dataclass(slots=True)
class BaseServiceInfo:
    """Base class for discovery ServiceInfo."""


class FlowError(HomeAssistantError):
    """Base class for data entry errors."""


class UnknownHandler(FlowError):
    """Unknown handler specified."""


class UnknownFlow(FlowError):
    """Unknown flow specified."""


class UnknownStep(FlowError):
    """Unknown step specified."""


class InvalidData(vol.Invalid):
    """Invalid data provided."""

    def __init__(
        self,
        message: str,
        path: Optional[list[Hashable]],
        error_message: Optional[str],
        schema_errors: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Initialize an invalid data exception."""
        super().__init__(message, path, error_message, **kwargs)
        self.schema_errors = schema_errors


class AbortFlow(FlowError):
    """Exception to indicate a flow needs to be aborted."""

    def __init__(
        self, reason: str, description_placeholders: Optional[Mapping[str, str]] = None
    ) -> None:
        """Initialize an abort flow exception."""
        super().__init__(f"Flow aborted: {reason}")
        self.reason = reason
        self.description_placeholders = description_placeholders


class FlowContext(TypedDict, total=False):
    """Typed context dict."""

    show_advanced_options: bool
    source: str


class FlowResult(TypedDict, Generic[_FlowContextT, _HandlerT], total=False):
    """Typed result dict."""

    context: _FlowContextT
    data_schema: Optional[vol.Schema]
    data: Mapping[str, Any]
    description_placeholders: Optional[Mapping[str, str]]
    description: Optional[str]
    errors: Optional[dict[str, str]]
    extra: str
    flow_id: Required[str]
    handler: Required[_HandlerT]
    last_step: Optional[bool]
    menu_options: Container[str]
    preview: Optional[str]
    progress_action: str
    progress_task: Optional[asyncio.Task[Any]]
    reason: str
    required: bool
    result: Any
    step_id: str
    title: str
    translation_domain: str
    type: FlowResultType
    url: str


def _map_error_to_schema_errors(
    schema_errors: dict[str, Any],
    error: vol.Invalid,
    data_schema: vol.Schema,
) -> None:
    """Map an error to the correct position in the schema_errors.

    Raises ValueError if the error path could not be found in the schema.
    Limitation: Nested schemas are not supported and a ValueError will be raised.
    """
    schema = data_schema.schema
    error_path = error.path
    if not error_path or (path_part := error_path[0]) not in schema:
        raise ValueError("Could not find path in schema")

    if len(error_path) > 1:
        raise ValueError("Nested schemas are not supported")

    # path_part can also be vol.Marker, but we need a string key
    path_part_str = str(path_part)
    schema_errors[path_part_str] = error.error_message


class FlowManager(abc.ABC, Generic[_FlowContextT, _FlowResultT, _HandlerT]):
    """Manage all the flows that are in progress."""

    _flow_result: type[_FlowResultT] = FlowResult  # type: ignore[assignment]

    def __init__(
        self,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the flow manager."""
        self.hass = hass
        self._preview: set[_HandlerT] = set()
        self._progress: dict[
            str, FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]
        ] = {}
        self._handler_progress_index: defaultdict[
            _HandlerT, set[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]
        ] = defaultdict(set)
        self._init_data_process_index: defaultdict[
            type, set[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]
        ] = defaultdict(set)

    @abc.abstractmethod
    async def async_create_flow(
        self,
        handler_key: _HandlerT,
        *,
        context: Optional[_FlowContextT] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]:
        """Create a flow for specified handler."""

    @abc.abstractmethod
    async def async_finish_flow(
        self,
        flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT],
        result: _FlowResultT,
    ) -> _FlowResultT:
        """Finish a data entry flow."""

    async def async_post_init(
        self,
        flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT],
        result: _FlowResultT,
    ) -> None:
        """Entry has finished executing its first step asynchronously."""

    @callback
    def async_get(self, flow_id: str) -> _FlowResultT:
        """Return a flow in progress as a partial FlowResult."""
        if (flow := self._progress.get(flow_id)) is None:
            raise UnknownFlow
        return self._async_flow_handler_to_flow_result([flow], False)[0]

    @callback
    def async_progress(self, include_uninitialized: bool = False) -> list[_FlowResultT]:
        """Return the flows in progress as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(
            self._progress.values(), include_uninitialized
        )

    @callback
    def async_progress_by_handler(
        self,
        handler: _HandlerT,
        include_uninitialized: bool = False,
        match_context: Optional[dict[str, Any]] = None,
    ) -> list[_FlowResultT]:
        """Return the flows in progress by handler as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(
            self._async_progress_by_handler(handler, match_context),
            include_uninitialized,
        )

    @callback
    def async_progress_by_init_data_type(
        self,
        init_data_type: type,
        matcher: Callable[[Any], bool],
        include_uninitialized: bool = False,
    ) -> list[_FlowResultT]:
        """Return flows in progress init matching by data type as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(
            [
                progress
                for progress in self._init_data_process_index.get(init_data_type, ())
                if matcher(progress.init_data)
            ],
            include_uninitialized,
        )

    @callback
    def _async_progress_by_handler(
        self, handler: _HandlerT, match_context: Optional[dict[str, Any]]
    ) -> list[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]:
        """Return the flows in progress by handler."""
        if not match_context:
            return list(self._handler_progress_index.get(handler, ()))
        match_context_items = match_context.items()
        return [
            progress
            for progress in self._handler_progress_index.get(handler, ())
            if match_context_items <= progress.context.items()
        ]

    async def async_init(
        self,
        handler: _HandlerT,
        *,
        context: Optional[_FlowContextT] = None,
        data: Any = None,
    ) -> _FlowResultT:
        """Start a data entry flow."""
        if context is None:
            context = cast(_FlowContextT, {})
        flow = await self.async_create_flow(handler, context=context, data=data)
        if not flow:
            raise UnknownFlow("Flow was not created")
        flow.hass = self.hass
        flow.handler = handler
        flow.flow_id = uuid_util.random_uuid_hex()
        flow.context = context
        flow.init_data = data
        self._async_add_flow_progress(flow)

        result = await self._async_handle_step(flow, flow.init_step, data)

        if result["type"] != FlowResultType.ABORT:
            await self.async_post_init(flow, result)

        return result

    async def async_configure(
        self, flow_id: str, user_input: Optional[dict] = None
    ) -> _FlowResultT:
        """Continue a data entry flow."""
        result: Optional[_FlowResultT] = None

        flow = self._progress.get(flow_id)
        if flow and flow.deprecated_show_progress:
            if (cur_step := flow.cur_step) and cur_step[
                "type"
            ] == FlowResultType.SHOW_PROGRESS:
                await asyncio.sleep(0)

        while not result or result["type"] == FlowResultType.SHOW_PROGRESS_DONE:
            result = await self._async_configure(flow_id, user_input)
            flow = self._progress.get(flow_id)
            if flow and flow.deprecated_show_progress:
                break
        return result

    async def _async_configure(
        self, flow_id: str, user_input: Optional[dict] = None
    ) -> _FlowResultT:
        """Continue a data entry flow."""
        if (flow := self._progress.get(flow_id)) is None:
            raise UnknownFlow

        cur_step = flow.cur_step
        assert cur_step is not None

        if (
            data_schema := cur_step.get("data_schema")
        ) is not None and user_input is not None:
            data_schema = cast(vol.Schema, data_schema)
            try:
                user_input = data_schema(user_input)
            except vol.Invalid as ex:
                raised_errors = [ex]
                if isinstance(ex, vol.MultipleInvalid):
                    raised_errors = ex.errors

                schema_errors: dict[str, Any] = {}
                for error in raised_errors:
                    try:
                        _map_error_to_schema_errors(schema_errors, error, data_schema)
                    except ValueError:
                        schema_errors.setdefault("base", []).append(str(error))
                raise InvalidData(
                    "Schema validation failed",
                    path=ex.path,
                    error_message=ex.error_message,
                    schema_errors=schema_errors,
                ) from ex

        if cur_step["type"] == FlowResultType.MENU and user_input:
            result = await self._async_handle_step(
                flow, user_input["next_step_id"], None
            )
        else:
            result = await self._async_handle_step(
                flow, cur_step["step_id"], user_input
            )

        if cur_step["type"] in (
            FlowResultType.EXTERNAL_STEP,
            FlowResultType.SHOW_PROGRESS,
        ):
            if cur_step["type"] == FlowResultType.EXTERNAL_STEP and result[
                "type"
            ] not in (
                FlowResultType.EXTERNAL_STEP,
                FlowResultType.EXTERNAL_STEP_DONE,
            ):
                raise ValueError(
                    "External step can only transition to "
                    "external step or external step done."
                )
            if cur_step["type"] == FlowResultType.SHOW_PROGRESS and result[
                "type"
            ] not in (
                FlowResultType.SHOW_PROGRESS,
                FlowResultType.SHOW_PROGRESS_DONE,
            ):
                raise ValueError(
                    "Show progress can only transition to show progress or show"
                    " progress done."
                )

            if cur_step["step_id"] != result.get("step_id") or (
                result["type"] == FlowResultType.SHOW_PROGRESS
                and (
                    cur_step["progress_action"] != result.get("progress_action")
                    or cur_step["description_placeholders"]
                    != result.get("description_placeholders")
                )
            ):
                self.hass.bus.async_fire_internal(
                    EVENT_DATA_ENTRY_FLOW_PROGRESSED,
                    {"handler": flow.handler, "flow_id": flow_id, "refresh": True},
                )

        return result

    @callback
    def async_abort(self, flow_id: str) -> None:
        """Abort a flow."""
        self._async_remove_flow_progress(flow_id)

    @callback
    def _async_add_flow_progress(
        self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]
    ) -> None:
        """Add a flow to in progress."""
        if flow.init_data is not None:
            self._init_data_process_index[type(flow.init_data)].add(flow)
        self._progress[flow.flow_id] = flow
        self._handler_progress_index[flow.handler].add(flow)

    @callback
    def _async_remove_flow_from_index(
        self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]
    ) -> None:
        """Remove a flow from in progress."""
        if flow.init_data is not None:
            init_data_type = type(flow.init_data)
            self._init_data_process_index[init_data_type].remove(flow)
            if not self._init_data_process_index[init_data_type]:
                del self._init_data_process_index[init_data_type]
        handler = flow.handler
        self._handler_progress_index[handler].remove(flow)
        if not self._handler_progress_index[handler]:
            del self._handler_progress_index[handler]

    @callback
    def _async_remove_flow_progress(self, flow_id: str) -> None:
        """Remove a flow from in progress."""
        if (flow := self._progress.pop(flow_id, None)) is None:
            raise UnknownFlow
        self._async_remove_flow_from_index(flow)
        flow.async_cancel_progress_task()
        try:
            flow.async_remove()
        except Exception:
            _LOGGER.exception("Error removing %s flow", flow.handler)

    async def _async_handle_step(
        self,
        flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT],
        step_id: str,
        user_input: Optional[Union[dict, BaseServiceInfo]],
    ) -> _FlowResultT:
        """Handle a step of a flow."""
        self._raise_if_step_does_not_exist(flow, step_id)

        method = f"async_step_{step_id}"
        try:
            result: _FlowResultT = await getattr(flow, method)(user_input)
        except AbortFlow as err:
            result = self._flow_result(
                type=FlowResultType.ABORT,
                flow_id=flow.flow_id,
                handler=flow.handler,
                reason=err.reason,
                description_placeholders=err.description_placeholders,
            )

        if result.get("preview") is not None:
            await self._async_setup_preview(flow)

        if not isinstance(result["type"], FlowResultType):
            result["type"] = FlowResultType(result["type"])  # type: ignore[unreachable]
            report_usage(
                "does not use FlowResultType enum for data entry flow result type",
                core_behavior=ReportBehavior.LOG,
                breaks_in_ha_version="2025.1",
            )

        if (
            result["type"] == FlowResultType.SHOW_PROGRESS
            and (progress_task := result.pop("progress_task", None))  # type: ignore[arg-type]
            and progress_task != flow.async_get_progress_task()
        ):
            async def call_configure() -> None:
                with suppress(UnknownFlow):
                    await self._async_configure(flow.flow_id)

            def schedule_configure(_: asyncio.Task) -> None:
                self.hass.async_create_task(call_configure())

            progress_task.add_done_callback(schedule_configure)  # type: ignore[attr-defined]
            flow.async_set_progress_task(progress_task)  # type: ignore[arg-type]

        elif result["type"] != FlowResultType.SHOW_PROGRESS:
            flow.async_cancel_progress_task()

        if result["type"] in STEP_ID_OPTIONAL_STEPS:
            if "step_id" not in result:
                result["step_id"] = step_id

        if result["type"] in FLOW_NOT_COMPL