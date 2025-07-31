from __future__ import annotations
import abc
import asyncio
from collections import defaultdict
from collections.abc import Callable, Container, Hashable, Iterable, Mapping as AbcMapping
from contextlib import suppress
import copy
from dataclasses import dataclass
from enum import StrEnum
import logging
from types import MappingProxyType
from typing import Any, Generic, Optional, Type, TypeVar, List, Dict, Mapping, cast, Awaitable
import voluptuous as vol
from .core import HomeAssistant, callback
from .exceptions import HomeAssistantError
from .helpers.frame import ReportBehavior, report_usage
from .loader import async_suggest_report_issue
from .util import uuid as uuid_util

_LOGGER: logging.Logger = logging.getLogger(__name__)

class FlowResultType(StrEnum):
    """Result type for a data entry flow."""
    FORM = 'form'
    CREATE_ENTRY = 'create_entry'
    ABORT = 'abort'
    EXTERNAL_STEP = 'external'
    EXTERNAL_STEP_DONE = 'external_done'
    SHOW_PROGRESS = 'progress'
    SHOW_PROGRESS_DONE = 'progress_done'
    MENU = 'menu'

EVENT_DATA_ENTRY_FLOW_PROGRESSED: str = 'data_entry_flow_progressed'
FLOW_NOT_COMPLETE_STEPS: Container[FlowResultType] = {
    FlowResultType.FORM,
    FlowResultType.EXTERNAL_STEP,
    FlowResultType.EXTERNAL_STEP_DONE,
    FlowResultType.SHOW_PROGRESS,
    FlowResultType.SHOW_PROGRESS_DONE,
    FlowResultType.MENU,
}
STEP_ID_OPTIONAL_STEPS: Container[FlowResultType] = {
    FlowResultType.EXTERNAL_STEP,
    FlowResultType.FORM,
    FlowResultType.MENU,
    FlowResultType.SHOW_PROGRESS,
}

_FlowContextT = TypeVar('_FlowContextT', bound='FlowContext', default='FlowContext')
_FlowResultT = TypeVar('_FlowResultT', bound='FlowResult[Any, Any]', default='FlowResult')
_HandlerT = TypeVar('_HandlerT', default=str)

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
    schema_errors: Dict[str, Any]

    def __init__(self, message: str, path: Any, error_message: str, schema_errors: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize an invalid data exception."""
        super().__init__(message, path, error_message, **kwargs)
        self.schema_errors = schema_errors

class AbortFlow(FlowError):
    """Exception to indicate a flow needs to be aborted."""
    reason: str
    description_placeholders: Optional[Mapping[str, Any]]

    def __init__(self, reason: str, description_placeholders: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize an abort flow exception."""
        super().__init__(f'Flow aborted: {reason}')
        self.reason = reason
        self.description_placeholders = description_placeholders

class FlowContext(TypedDict, total=False):
    """Typed context dict."""
    # Define expected keys as needed.
    ...

class FlowResult(TypedDict, Generic[_FlowContextT, _HandlerT], total=False):
    """Typed result dict."""
    # Define expected keys as needed.
    ...

def _map_error_to_schema_errors(schema_errors: Dict[str, Any], error: vol.Invalid, data_schema: vol.Schema) -> None:
    """Map an error to the correct position in the schema_errors.

    Raises ValueError if the error path could not be found in the schema.
    Limitation: Nested schemas are not supported and a ValueError will be raised.
    """
    schema = data_schema.schema
    error_path = error.path
    if not error_path or (path_part := error_path[0]) not in schema:
        raise ValueError('Could not find path in schema')
    if len(error_path) > 1:
        raise ValueError('Nested schemas are not supported')
    path_part_str = str(path_part)
    schema_errors[path_part_str] = error.error_message

class FlowManager(abc.ABC, Generic[_FlowContextT, _FlowResultT, _HandlerT]):
    """Manage all the flows that are in progress."""
    _flow_result = FlowResult

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the flow manager."""
        self.hass: HomeAssistant = hass
        self._preview: set[_HandlerT] = set()
        self._progress: Dict[str, FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]] = {}
        self._handler_progress_index: Dict[_HandlerT, set[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]] = defaultdict(set)
        self._init_data_process_index: Dict[Type[Any], set[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]] = defaultdict(set)

    @abc.abstractmethod
    async def async_create_flow(self, handler_key: _HandlerT, *, context: Optional[_FlowContextT] = None, data: Any = None) -> FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]:
        """Create a flow for specified handler.

        Handler key is the domain of the component that we want to set up.
        """

    @abc.abstractmethod
    async def async_finish_flow(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT], result: FlowResult[_FlowContextT, _HandlerT]) -> FlowResult[_FlowContextT, _HandlerT]:
        """Finish a data entry flow.

        This method is called when a flow step returns FlowResultType.ABORT or
        FlowResultType.CREATE_ENTRY.
        """

    async def async_post_init(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT], result: FlowResult[_FlowContextT, _HandlerT]) -> None:
        """Entry has finished executing its first step asynchronously."""
        return

    @callback
    def async_get(self, flow_id: str) -> FlowResult[_FlowContextT, _HandlerT]:
        """Return a flow in progress as a partial FlowResult."""
        flow = self._progress.get(flow_id)
        if flow is None:
            raise UnknownFlow
        return self._async_flow_handler_to_flow_result([flow], False)[0]

    @callback
    def async_progress(self, include_uninitialized: bool = False) -> List[FlowResult[_FlowContextT, _HandlerT]]:
        """Return the flows in progress as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(list(self._progress.values()), include_uninitialized)

    @callback
    def async_progress_by_handler(self, handler: _HandlerT, include_uninitialized: bool = False, match_context: Optional[Mapping[str, Any]] = None) -> List[FlowResult[_FlowContextT, _HandlerT]]:
        """Return the flows in progress by handler as a partial FlowResult.

        If match_context is specified, only return flows with a context that
        is a superset of match_context.
        """
        return self._async_flow_handler_to_flow_result(self._async_progress_by_handler(handler, match_context), include_uninitialized)

    @callback
    def async_progress_by_init_data_type(self, init_data_type: Type[Any], matcher: Callable[[Any], bool], include_uninitialized: bool = False) -> List[FlowResult[_FlowContextT, _HandlerT]]:
        """Return flows in progress init matching by data type as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(
            [progress for progress in self._init_data_process_index.get(init_data_type, set()) if matcher(progress.init_data)],
            include_uninitialized
        )

    @callback
    def _async_progress_by_handler(self, handler: _HandlerT, match_context: Optional[Mapping[str, Any]]) -> List[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]]:
        """Return the flows in progress by handler.

        If match_context is specified, only return flows with a context that
        is a superset of match_context.
        """
        if not match_context:
            return list(self._handler_progress_index.get(handler, set()))
        match_context_items = match_context.items()
        return [progress for progress in self._handler_progress_index.get(handler, set()) if match_context_items <= progress.context.items()]

    async def async_init(self, handler: _HandlerT, *, context: Optional[_FlowContextT] = None, data: Any = None) -> FlowResult[_FlowContextT, _HandlerT]:
        """Start a data entry flow."""
        if context is None:
            context = cast(_FlowContextT, {})
        flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT] = await self.async_create_flow(handler, context=context, data=data)
        if not flow:
            raise UnknownFlow('Flow was not created')
        flow.hass = self.hass
        flow.handler = handler
        flow.flow_id = uuid_util.random_uuid_hex()
        flow.context = context
        flow.init_data = data
        self._async_add_flow_progress(flow)
        result: FlowResult[_FlowContextT, _HandlerT] = await self._async_handle_step(flow, flow.init_step, data)
        if result['type'] != FlowResultType.ABORT:
            await self.async_post_init(flow, result)
        return result

    async def async_configure(self, flow_id: str, user_input: Optional[Mapping[str, Any]] = None) -> FlowResult[_FlowContextT, _HandlerT]:
        """Continue a data entry flow."""
        result: Optional[FlowResult[_FlowContextT, _HandlerT]] = None
        flow = self._progress.get(flow_id)
        if flow and flow.deprecated_show_progress:
            if (cur_step := flow.cur_step) and cur_step['type'] == FlowResultType.SHOW_PROGRESS:
                await asyncio.sleep(0)
        while not result or result['type'] == FlowResultType.SHOW_PROGRESS_DONE:
            result = await self._async_configure(flow_id, user_input)
            flow = self._progress.get(flow_id)
            if flow and flow.deprecated_show_progress:
                break
        return result

    async def _async_configure(self, flow_id: str, user_input: Optional[Mapping[str, Any]] = None) -> FlowResult[_FlowContextT, _HandlerT]:
        """Continue a data entry flow."""
        flow = self._progress.get(flow_id)
        if flow is None:
            raise UnknownFlow
        cur_step: Dict[str, Any] = flow.cur_step  # type: ignore
        assert cur_step is not None
        data_schema: Optional[Any] = cur_step.get('data_schema')
        if data_schema is not None and user_input is not None:
            data_schema = cast(vol.Schema, data_schema)
            try:
                user_input = data_schema(user_input)
            except vol.Invalid as ex:
                raised_errors: List[vol.Invalid] = [ex]
                if isinstance(ex, vol.MultipleInvalid):
                    raised_errors = ex.errors
                schema_errors: Dict[str, Any] = {}
                for error in raised_errors:
                    try:
                        _map_error_to_schema_errors(schema_errors, error, data_schema)
                    except ValueError:
                        schema_errors.setdefault('base', []).append(str(error))
                raise InvalidData('Schema validation failed', path=ex.path, error_message=ex.error_message, schema_errors=schema_errors) from ex
        if cur_step['type'] == FlowResultType.MENU and user_input:
            result = await self._async_handle_step(flow, user_input['next_step_id'], None)
        else:
            result = await self._async_handle_step(flow, cur_step['step_id'], user_input)
        if cur_step['type'] in (FlowResultType.EXTERNAL_STEP, FlowResultType.SHOW_PROGRESS):
            if cur_step['type'] == FlowResultType.EXTERNAL_STEP and result['type'] not in (FlowResultType.EXTERNAL_STEP, FlowResultType.EXTERNAL_STEP_DONE):
                raise ValueError('External step can only transition to external step or external step done.')
            if cur_step['type'] == FlowResultType.SHOW_PROGRESS and result['type'] not in (FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE):
                raise ValueError('Show progress can only transition to show progress or show progress done.')
            if cur_step['step_id'] != result.get('step_id') or (result['type'] == FlowResultType.SHOW_PROGRESS and (cur_step.get('progress_action') != result.get('progress_action') or cur_step.get('description_placeholders') != result.get('description_placeholders'))):
                self.hass.bus.async_fire_internal(EVENT_DATA_ENTRY_FLOW_PROGRESSED, {'handler': flow.handler, 'flow_id': flow_id, 'refresh': True})
        return result

    @callback
    def async_abort(self, flow_id: str) -> None:
        """Abort a flow."""
        self._async_remove_flow_progress(flow_id)

    @callback
    def _async_add_flow_progress(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]) -> None:
        """Add a flow to in progress."""
        if flow.init_data is not None:
            self._init_data_process_index[type(flow.init_data)].add(flow)
        self._progress[flow.flow_id] = flow  # type: ignore
        self._handler_progress_index[flow.handler].add(flow)

    @callback
    def _async_remove_flow_from_index(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]) -> None:
        """Remove a flow from in progress."""
        if flow.init_data is not None:
            init_data_type: Type[Any] = type(flow.init_data)
            self._init_data_process_index[init_data_type].remove(flow)
            if not self._init_data_process_index[init_data_type]:
                del self._init_data_process_index[init_data_type]
        handler: _HandlerT = flow.handler  # type: ignore
        self._handler_progress_index[handler].remove(flow)
        if not self._handler_progress_index[handler]:
            del self._handler_progress_index[handler]

    @callback
    def _async_remove_flow_progress(self, flow_id: str) -> None:
        """Remove a flow from in progress."""
        flow = self._progress.pop(flow_id, None)
        if flow is None:
            raise UnknownFlow
        self._async_remove_flow_from_index(flow)
        flow.async_cancel_progress_task()
        try:
            flow.async_remove()
        except Exception:
            _LOGGER.exception('Error removing %s flow', flow.handler)

    async def _async_handle_step(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT], step_id: str, user_input: Any) -> FlowResult[_FlowContextT, _HandlerT]:
        """Handle a step of a flow."""
        self._raise_if_step_does_not_exist(flow, step_id)
        method: str = f'async_step_{step_id}'
        try:
            result: FlowResult[_FlowContextT, _HandlerT] = await getattr(flow, method)(user_input)
        except AbortFlow as err:
            result = self._flow_result(type=FlowResultType.ABORT, flow_id=flow.flow_id, handler=flow.handler, reason=err.reason, description_placeholders=err.description_placeholders)
        if result.get('preview') is not None:
            await self._async_setup_preview(flow)
        if not isinstance(result['type'], FlowResultType):
            result['type'] = FlowResultType(result['type'])
            report_usage('does not use FlowResultType enum for data entry flow result type', core_behavior=ReportBehavior.LOG, breaks_in_ha_version='2025.1')
        if result['type'] == FlowResultType.SHOW_PROGRESS and (progress_task := result.pop('progress_task', None)) and (progress_task != flow.async_get_progress_task()):
            async def call_configure() -> None:
                with suppress(UnknownFlow):
                    await self._async_configure(flow.flow_id)
            def schedule_configure(_: Any) -> None:
                self.hass.async_create_task(call_configure())
            progress_task.add_done_callback(schedule_configure)
            flow.async_set_progress_task(progress_task)
        elif result['type'] != FlowResultType.SHOW_PROGRESS:
            flow.async_cancel_progress_task()
        if result['type'] in STEP_ID_OPTIONAL_STEPS:
            if 'step_id' not in result:
                result['step_id'] = step_id
        if result['type'] in FLOW_NOT_COMPLETE_STEPS:
            self._raise_if_step_does_not_exist(flow, result['step_id'])
            flow.cur_step = result  # type: ignore
            return result
        result = await self.async_finish_flow(flow, result.copy())
        if result['type'] == FlowResultType.FORM:
            flow.cur_step = result  # type: ignore
            return result
        self._async_remove_flow_progress(flow.flow_id)
        return result

    def _raise_if_step_does_not_exist(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT], step_id: str) -> None:
        """Raise if the step does not exist."""
        method: str = f'async_step_{step_id}'
        if not hasattr(flow, method):
            self._async_remove_flow_progress(flow.flow_id)
            raise UnknownStep(f"Handler {flow.__class__.__name__} doesn't support step {step_id}")

    async def _async_setup_preview(self, flow: FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]) -> None:
        """Set up preview for a flow handler."""
        if flow.handler not in self._preview:
            self._preview.add(flow.handler)
            await flow.async_setup_preview(self.hass)

    @callback
    def _async_flow_handler_to_flow_result(self, flows: Iterable[FlowHandler[_FlowContextT, _FlowResultT, _HandlerT]], include_uninitialized: bool) -> List[FlowResult[_FlowContextT, _HandlerT]]:
        """Convert a list of FlowHandler to a partial FlowResult that can be serialized."""
        results: List[FlowResult[_FlowContextT, _HandlerT]] = []
        for flow in flows:
            if flow.cur_step:
                results.append(self._flow_result(flow_id=flow.flow_id, handler=flow.handler, context=flow.context, step_id=flow.cur_step['step_id']))
            elif include_uninitialized:
                results.append(self._flow_result(flow_id=flow.flow_id, handler=flow.handler, context=flow.context))
        return results

class FlowHandler(Generic[_FlowContextT, _FlowResultT, _HandlerT]):
    """Handle a data entry flow."""
    _flow_result = FlowResult
    cur_step: Optional[Dict[str, Any]] = None
    flow_id: Optional[str] = None
    hass: Optional[HomeAssistant] = None
    handler: Optional[_HandlerT] = None
    context: Mapping[str, Any] = MappingProxyType({})
    init_step: str = 'init'
    init_data: Any = None
    VERSION: int = 1
    MINOR_VERSION: int = 1
    __progress_task: Optional[asyncio.Task] = None
    __no_progress_task_reported: bool = False
    deprecated_show_progress: bool = False

    @property
    def source(self) -> Optional[Any]:
        """Source that initialized the flow."""
        return self.context.get('source', None)

    @property
    def show_advanced_options(self) -> bool:
        """If we should show advanced options."""
        return self.context.get('show_advanced_options', False)

    def add_suggested_values_to_schema(self, data_schema: vol.Schema, suggested_values: Optional[Mapping[Any, Any]]) -> vol.Schema:
        """Make a copy of the schema, populated with suggested values.

        For each schema marker matching items in `suggested_values`,
        the `suggested_value` will be set. The existing `suggested_value` will
        be left untouched if there is no matching item.
        """
        schema: Dict[Any, Any] = {}
        for key, val in data_schema.schema.items():
            if isinstance(key, vol.Marker):
                if key.description and key.description.get('advanced') and (not self.show_advanced_options):
                    continue
            new_key = key
            if suggested_values and key in suggested_values and isinstance(key, vol.Marker):
                new_key = copy.copy(key)
                new_key.description = {'suggested_value': suggested_values[key.schema]}
            schema[new_key] = val
        return vol.Schema(schema)

    @callback
    def async_show_form(
        self,
        *,
        step_id: Optional[str] = None,
        data_schema: Optional[vol.Schema] = None,
        errors: Optional[Mapping[str, Any]] = None,
        description_placeholders: Optional[Mapping[str, Any]] = None,
        last_step: Optional[bool] = None,
        preview: Optional[Any] = None
    ) -> FlowResult[_FlowContextT, _HandlerT]:
        """Return the definition of a form to gather user input.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result: FlowResult[_FlowContextT, _HandlerT] = self._flow_result(
            type=FlowResultType.FORM,
            flow_id=self.flow_id,
            handler=self.handler,
            data_schema=data_schema,
            errors=errors,
            description_placeholders=description_placeholders,
            last_step=last_step,
            preview=preview
        )
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_create_entry(
        self,
        *,
        title: Optional[str] = None,
        data: Any,
        description: Optional[str] = None,
        description_placeholders: Optional[Mapping[str, Any]] = None
    ) -> FlowResult[_FlowContextT, _HandlerT]:
        """Finish flow."""
        flow_result: FlowResult[_FlowContextT, _HandlerT] = self._flow_result(
            type=FlowResultType.CREATE_ENTRY,
            flow_id=self.flow_id,
            handler=self.handler,
            data=data,
            description=description,
            description_placeholders=description_placeholders,
            context=self.context
        )
        if title is not None:
            flow_result['title'] = title
        return flow_result

    @callback
    def async_abort(self, *, reason: str, description_placeholders: Optional[Mapping[str, Any]] = None) -> FlowResult[_FlowContextT, _HandlerT]:
        """Abort the flow."""
        return self._flow_result(
            type=FlowResultType.ABORT,
            flow_id=self.flow_id,
            handler=self.handler,
            reason=reason,
            description_placeholders=description_placeholders
        )

    @callback
    def async_external_step(self, *, step_id: Optional[str] = None, url: str, description_placeholders: Optional[Mapping[str, Any]] = None) -> FlowResult[_FlowContextT, _HandlerT]:
        """Return the definition of an external step for the user to take.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result: FlowResult[_FlowContextT, _HandlerT] = self._flow_result(
            type=FlowResultType.EXTERNAL_STEP,
            flow_id=self.flow_id,
            handler=self.handler,
            url=url,
            description_placeholders=description_placeholders
        )
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_external_step_done(self, *, next_step_id: str) -> FlowResult[_FlowContextT, _HandlerT]:
        """Return the definition of an external step for the user to take."""
        return self._flow_result(
            type=FlowResultType.EXTERNAL_STEP_DONE,
            flow_id=self.flow_id,
            handler=self.handler,
            step_id=next_step_id
        )

    @callback
    def async_show_progress(
        self,
        *,
        step_id: Optional[str] = None,
        progress_action: Any,
        description_placeholders: Optional[Mapping[str, Any]] = None,
        progress_task: Optional[asyncio.Task] = None
    ) -> FlowResult[_FlowContextT, _HandlerT]:
        """Show a progress message to the user, without user input allowed.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        if progress_task is None and (not self.__no_progress_task_reported):
            self.__no_progress_task_reported = True
            cls = self.__class__
            report_issue = async_suggest_report_issue(self.hass, module=cls.__module__)
            _LOGGER.warning(
                '%s::%s calls async_show_progress without passing a progress task, this is not valid and will break in Home Assistant Core 2024.8. Please %s',
                cls.__module__,
                cls.__name__,
                report_issue
            )
        if progress_task is None:
            self.deprecated_show_progress = True
        flow_result: FlowResult[_FlowContextT, _HandlerT] = self._flow_result(
            type=FlowResultType.SHOW_PROGRESS,
            flow_id=self.flow_id,
            handler=self.handler,
            progress_action=progress_action,
            description_placeholders=description_placeholders,
            progress_task=progress_task
        )
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_show_progress_done(self, *, next_step_id: str) -> FlowResult[_FlowContextT, _HandlerT]:
        """Mark the progress done."""
        return self._flow_result(
            type=FlowResultType.SHOW_PROGRESS_DONE,
            flow_id=self.flow_id,
            handler=self.handler,
            step_id=next_step_id
        )

    @callback
    def async_show_menu(
        self,
        *,
        step_id: Optional[str] = None,
        menu_options: Mapping[Any, Any],
        description_placeholders: Optional[Mapping[str, Any]] = None
    ) -> FlowResult[_FlowContextT, _HandlerT]:
        """Show a navigation menu to the user.

        Options dict maps step_id => i18n label
        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result: FlowResult[_FlowContextT, _HandlerT] = self._flow_result(
            type=FlowResultType.MENU,
            flow_id=self.flow_id,
            handler=self.handler,
            data_schema=vol.Schema({'next_step_id': vol.In(menu_options)}),
            menu_options=menu_options,
            description_placeholders=description_placeholders
        )
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_remove(self) -> None:
        """Notification that the flow has been removed."""
        return

    @staticmethod
    async def async_setup_preview(hass: HomeAssistant) -> None:
        """Set up preview."""
        return

    @callback
    def async_cancel_progress_task(self) -> None:
        """Cancel in progress task."""
        if self.__progress_task and (not self.__progress_task.done()):
            self.__progress_task.cancel()
        self.__progress_task = None

    @callback
    def async_get_progress_task(self) -> Optional[asyncio.Task]:
        """Get in progress task."""
        return self.__progress_task

    @callback
    def async_set_progress_task(self, progress_task: asyncio.Task) -> None:
        """Set in progress task."""
        self.__progress_task = progress_task

class SectionConfig(TypedDict, total=False):
    """Class to represent a section config."""
    # Define configuration keys as needed.
    ...

class section:
    """Data entry flow section."""
    CONFIG_SCHEMA: vol.Schema = vol.Schema({vol.Optional('collapsed', default=False): bool})

    def __init__(self, schema: vol.Schema, options: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize."""
        self.schema: vol.Schema = schema
        self.options: Dict[str, Any] = self.CONFIG_SCHEMA(options or {})

    def __call__(self, value: Any) -> Any:
        """Validate input."""
        return self.schema(value)