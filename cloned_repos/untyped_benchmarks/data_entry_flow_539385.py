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
from typing import Any, Generic, Required, TypedDict, TypeVar, cast
import voluptuous as vol
from .core import HomeAssistant, callback
from .exceptions import HomeAssistantError
from .helpers.frame import ReportBehavior, report_usage
from .loader import async_suggest_report_issue
from .util import uuid as uuid_util
_LOGGER = logging.getLogger(__name__)

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
EVENT_DATA_ENTRY_FLOW_PROGRESSED = 'data_entry_flow_progressed'
FLOW_NOT_COMPLETE_STEPS = {FlowResultType.FORM, FlowResultType.EXTERNAL_STEP, FlowResultType.EXTERNAL_STEP_DONE, FlowResultType.SHOW_PROGRESS, FlowResultType.SHOW_PROGRESS_DONE, FlowResultType.MENU}
STEP_ID_OPTIONAL_STEPS = {FlowResultType.EXTERNAL_STEP, FlowResultType.FORM, FlowResultType.MENU, FlowResultType.SHOW_PROGRESS}
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

    def __init__(self, message, path, error_message, schema_errors, **kwargs):
        """Initialize an invalid data exception."""
        super().__init__(message, path, error_message, **kwargs)
        self.schema_errors = schema_errors

class AbortFlow(FlowError):
    """Exception to indicate a flow needs to be aborted."""

    def __init__(self, reason, description_placeholders=None):
        """Initialize an abort flow exception."""
        super().__init__(f'Flow aborted: {reason}')
        self.reason = reason
        self.description_placeholders = description_placeholders

class FlowContext(TypedDict, total=False):
    """Typed context dict."""

class FlowResult(TypedDict, Generic[_FlowContextT, _HandlerT], total=False):
    """Typed result dict."""

def _map_error_to_schema_errors(schema_errors, error, data_schema):
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

    def __init__(self, hass):
        """Initialize the flow manager."""
        self.hass = hass
        self._preview = set()
        self._progress = {}
        self._handler_progress_index = defaultdict(set)
        self._init_data_process_index = defaultdict(set)

    @abc.abstractmethod
    async def async_create_flow(self, handler_key, *, context=None, data=None):
        """Create a flow for specified handler.

        Handler key is the domain of the component that we want to set up.
        """

    @abc.abstractmethod
    async def async_finish_flow(self, flow, result):
        """Finish a data entry flow.

        This method is called when a flow step returns FlowResultType.ABORT or
        FlowResultType.CREATE_ENTRY.
        """

    async def async_post_init(self, flow, result):
        """Entry has finished executing its first step asynchronously."""

    @callback
    def async_get(self, flow_id):
        """Return a flow in progress as a partial FlowResult."""
        if (flow := self._progress.get(flow_id)) is None:
            raise UnknownFlow
        return self._async_flow_handler_to_flow_result([flow], False)[0]

    @callback
    def async_progress(self, include_uninitialized=False):
        """Return the flows in progress as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result(self._progress.values(), include_uninitialized)

    @callback
    def async_progress_by_handler(self, handler, include_uninitialized=False, match_context=None):
        """Return the flows in progress by handler as a partial FlowResult.

        If match_context is specified, only return flows with a context that
        is a superset of match_context.
        """
        return self._async_flow_handler_to_flow_result(self._async_progress_by_handler(handler, match_context), include_uninitialized)

    @callback
    def async_progress_by_init_data_type(self, init_data_type, matcher, include_uninitialized=False):
        """Return flows in progress init matching by data type as a partial FlowResult."""
        return self._async_flow_handler_to_flow_result([progress for progress in self._init_data_process_index.get(init_data_type, ()) if matcher(progress.init_data)], include_uninitialized)

    @callback
    def _async_progress_by_handler(self, handler, match_context):
        """Return the flows in progress by handler.

        If match_context is specified, only return flows with a context that
        is a superset of match_context.
        """
        if not match_context:
            return list(self._handler_progress_index.get(handler, ()))
        match_context_items = match_context.items()
        return [progress for progress in self._handler_progress_index.get(handler, ()) if match_context_items <= progress.context.items()]

    async def async_init(self, handler, *, context=None, data=None):
        """Start a data entry flow."""
        if context is None:
            context = cast(_FlowContextT, {})
        flow = await self.async_create_flow(handler, context=context, data=data)
        if not flow:
            raise UnknownFlow('Flow was not created')
        flow.hass = self.hass
        flow.handler = handler
        flow.flow_id = uuid_util.random_uuid_hex()
        flow.context = context
        flow.init_data = data
        self._async_add_flow_progress(flow)
        result = await self._async_handle_step(flow, flow.init_step, data)
        if result['type'] != FlowResultType.ABORT:
            await self.async_post_init(flow, result)
        return result

    async def async_configure(self, flow_id, user_input=None):
        """Continue a data entry flow."""
        result = None
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

    async def _async_configure(self, flow_id, user_input=None):
        """Continue a data entry flow."""
        if (flow := self._progress.get(flow_id)) is None:
            raise UnknownFlow
        cur_step = flow.cur_step
        assert cur_step is not None
        if (data_schema := cur_step.get('data_schema')) is not None and user_input is not None:
            data_schema = cast(vol.Schema, data_schema)
            try:
                user_input = data_schema(user_input)
            except vol.Invalid as ex:
                raised_errors = [ex]
                if isinstance(ex, vol.MultipleInvalid):
                    raised_errors = ex.errors
                schema_errors = {}
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
            if cur_step['step_id'] != result.get('step_id') or (result['type'] == FlowResultType.SHOW_PROGRESS and (cur_step['progress_action'] != result.get('progress_action') or cur_step['description_placeholders'] != result.get('description_placeholders'))):
                self.hass.bus.async_fire_internal(EVENT_DATA_ENTRY_FLOW_PROGRESSED, {'handler': flow.handler, 'flow_id': flow_id, 'refresh': True})
        return result

    @callback
    def async_abort(self, flow_id):
        """Abort a flow."""
        self._async_remove_flow_progress(flow_id)

    @callback
    def _async_add_flow_progress(self, flow):
        """Add a flow to in progress."""
        if flow.init_data is not None:
            self._init_data_process_index[type(flow.init_data)].add(flow)
        self._progress[flow.flow_id] = flow
        self._handler_progress_index[flow.handler].add(flow)

    @callback
    def _async_remove_flow_from_index(self, flow):
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
    def _async_remove_flow_progress(self, flow_id):
        """Remove a flow from in progress."""
        if (flow := self._progress.pop(flow_id, None)) is None:
            raise UnknownFlow
        self._async_remove_flow_from_index(flow)
        flow.async_cancel_progress_task()
        try:
            flow.async_remove()
        except Exception:
            _LOGGER.exception('Error removing %s flow', flow.handler)

    async def _async_handle_step(self, flow, step_id, user_input):
        """Handle a step of a flow."""
        self._raise_if_step_does_not_exist(flow, step_id)
        method = f'async_step_{step_id}'
        try:
            result = await getattr(flow, method)(user_input)
        except AbortFlow as err:
            result = self._flow_result(type=FlowResultType.ABORT, flow_id=flow.flow_id, handler=flow.handler, reason=err.reason, description_placeholders=err.description_placeholders)
        if result.get('preview') is not None:
            await self._async_setup_preview(flow)
        if not isinstance(result['type'], FlowResultType):
            result['type'] = FlowResultType(result['type'])
            report_usage('does not use FlowResultType enum for data entry flow result type', core_behavior=ReportBehavior.LOG, breaks_in_ha_version='2025.1')
        if result['type'] == FlowResultType.SHOW_PROGRESS and (progress_task := result.pop('progress_task', None)) and (progress_task != flow.async_get_progress_task()):

            async def call_configure():
                with suppress(UnknownFlow):
                    await self._async_configure(flow.flow_id)

            def schedule_configure(_):
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
            flow.cur_step = result
            return result
        result = await self.async_finish_flow(flow, result.copy())
        if result['type'] == FlowResultType.FORM:
            flow.cur_step = result
            return result
        self._async_remove_flow_progress(flow.flow_id)
        return result

    def _raise_if_step_does_not_exist(self, flow, step_id):
        """Raise if the step does not exist."""
        method = f'async_step_{step_id}'
        if not hasattr(flow, method):
            self._async_remove_flow_progress(flow.flow_id)
            raise UnknownStep(f"Handler {flow.__class__.__name__} doesn't support step {step_id}")

    async def _async_setup_preview(self, flow):
        """Set up preview for a flow handler."""
        if flow.handler not in self._preview:
            self._preview.add(flow.handler)
            await flow.async_setup_preview(self.hass)

    @callback
    def _async_flow_handler_to_flow_result(self, flows, include_uninitialized):
        """Convert a list of FlowHandler to a partial FlowResult that can be serialized."""
        return [self._flow_result(flow_id=flow.flow_id, handler=flow.handler, context=flow.context, step_id=flow.cur_step['step_id']) if flow.cur_step else self._flow_result(flow_id=flow.flow_id, handler=flow.handler, context=flow.context) for flow in flows if include_uninitialized or flow.cur_step is not None]

class FlowHandler(Generic[_FlowContextT, _FlowResultT, _HandlerT]):
    """Handle a data entry flow."""
    _flow_result = FlowResult
    cur_step = None
    flow_id = None
    hass = None
    handler = None
    context = MappingProxyType({})
    init_step = 'init'
    init_data = None
    VERSION = 1
    MINOR_VERSION = 1
    __progress_task = None
    __no_progress_task_reported = False
    deprecated_show_progress = False

    @property
    def source(self):
        """Source that initialized the flow."""
        return self.context.get('source', None)

    @property
    def show_advanced_options(self):
        """If we should show advanced options."""
        return self.context.get('show_advanced_options', False)

    def add_suggested_values_to_schema(self, data_schema, suggested_values):
        """Make a copy of the schema, populated with suggested values.

        For each schema marker matching items in `suggested_values`,
        the `suggested_value` will be set. The existing `suggested_value` will
        be left untouched if there is no matching item.
        """
        schema = {}
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
    def async_show_form(self, *, step_id=None, data_schema=None, errors=None, description_placeholders=None, last_step=None, preview=None):
        """Return the definition of a form to gather user input.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result = self._flow_result(type=FlowResultType.FORM, flow_id=self.flow_id, handler=self.handler, data_schema=data_schema, errors=errors, description_placeholders=description_placeholders, last_step=last_step, preview=preview)
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_create_entry(self, *, title=None, data, description=None, description_placeholders=None):
        """Finish flow."""
        flow_result = self._flow_result(type=FlowResultType.CREATE_ENTRY, flow_id=self.flow_id, handler=self.handler, data=data, description=description, description_placeholders=description_placeholders, context=self.context)
        if title is not None:
            flow_result['title'] = title
        return flow_result

    @callback
    def async_abort(self, *, reason, description_placeholders=None):
        """Abort the flow."""
        return self._flow_result(type=FlowResultType.ABORT, flow_id=self.flow_id, handler=self.handler, reason=reason, description_placeholders=description_placeholders)

    @callback
    def async_external_step(self, *, step_id=None, url, description_placeholders=None):
        """Return the definition of an external step for the user to take.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result = self._flow_result(type=FlowResultType.EXTERNAL_STEP, flow_id=self.flow_id, handler=self.handler, url=url, description_placeholders=description_placeholders)
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_external_step_done(self, *, next_step_id):
        """Return the definition of an external step for the user to take."""
        return self._flow_result(type=FlowResultType.EXTERNAL_STEP_DONE, flow_id=self.flow_id, handler=self.handler, step_id=next_step_id)

    @callback
    def async_show_progress(self, *, step_id=None, progress_action, description_placeholders=None, progress_task=None):
        """Show a progress message to the user, without user input allowed.

        The step_id parameter is deprecated and will be removed in a future release.
        """
        if progress_task is None and (not self.__no_progress_task_reported):
            self.__no_progress_task_reported = True
            cls = self.__class__
            report_issue = async_suggest_report_issue(self.hass, module=cls.__module__)
            _LOGGER.warning('%s::%s calls async_show_progress without passing a progress task, this is not valid and will break in Home Assistant Core 2024.8. Please %s', cls.__module__, cls.__name__, report_issue)
        if progress_task is None:
            self.deprecated_show_progress = True
        flow_result = self._flow_result(type=FlowResultType.SHOW_PROGRESS, flow_id=self.flow_id, handler=self.handler, progress_action=progress_action, description_placeholders=description_placeholders, progress_task=progress_task)
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_show_progress_done(self, *, next_step_id):
        """Mark the progress done."""
        return self._flow_result(type=FlowResultType.SHOW_PROGRESS_DONE, flow_id=self.flow_id, handler=self.handler, step_id=next_step_id)

    @callback
    def async_show_menu(self, *, step_id=None, menu_options, description_placeholders=None):
        """Show a navigation menu to the user.

        Options dict maps step_id => i18n label
        The step_id parameter is deprecated and will be removed in a future release.
        """
        flow_result = self._flow_result(type=FlowResultType.MENU, flow_id=self.flow_id, handler=self.handler, data_schema=vol.Schema({'next_step_id': vol.In(menu_options)}), menu_options=menu_options, description_placeholders=description_placeholders)
        if step_id is not None:
            flow_result['step_id'] = step_id
        return flow_result

    @callback
    def async_remove(self):
        """Notification that the flow has been removed."""

    @staticmethod
    async def async_setup_preview(hass):
        """Set up preview."""

    @callback
    def async_cancel_progress_task(self):
        """Cancel in progress task."""
        if self.__progress_task and (not self.__progress_task.done()):
            self.__progress_task.cancel()
        self.__progress_task = None

    @callback
    def async_get_progress_task(self):
        """Get in progress task."""
        return self.__progress_task

    @callback
    def async_set_progress_task(self, progress_task):
        """Set in progress task."""
        self.__progress_task = progress_task

class SectionConfig(TypedDict, total=False):
    """Class to represent a section config."""

class section:
    """Data entry flow section."""
    CONFIG_SCHEMA = vol.Schema({vol.Optional('collapsed', default=False): bool})

    def __init__(self, schema, options=None):
        """Initialize."""
        self.schema = schema
        self.options = self.CONFIG_SCHEMA(options or {})

    def __call__(self, value):
        """Validate input."""
        return self.schema(value)