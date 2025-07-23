from __future__ import annotations
import abc
import asyncio
import copy
from base64 import b64encode
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, ClassVar, Coroutine, Dict, List, Literal, MutableMapping, Optional, Tuple, Type, Union, cast
from uuid import UUID, uuid4
import jinja2
import orjson
from cachetools import TTLCache
from httpx import Response
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from typing_extensions import Self, TypeAlias
from prefect.blocks.abstract import NotificationBlock, NotificationError
from prefect.blocks.core import Block
from prefect.blocks.webhook import Webhook
from prefect.logging import get_logger
from prefect.server.events.clients import PrefectServerEventsAPIClient, PrefectServerEventsClient
from prefect.server.events.schemas.events import Event, RelatedResource, Resource
from prefect.server.events.schemas.labelling import LabelDiver
from prefect.server.schemas.actions import DeploymentFlowRunCreate, StateCreate
from prefect.server.schemas.core import BlockDocument, ConcurrencyLimitV2, Flow, TaskRun, WorkPool
from prefect.server.schemas.responses import DeploymentResponse, FlowRunResponse, OrchestrationResult, StateAcceptDetails, WorkQueueWithStatus
from prefect.server.schemas.states import Scheduled, State, StateType, Suspended
from prefect.server.utilities.http import should_redact_header
from prefect.server.utilities.messaging import Message, MessageHandler
from prefect.server.utilities.schemas import PrefectBaseModel
from prefect.server.utilities.user_templates import TemplateSecurityError, matching_types_in_templates, maybe_template, register_user_template_filters, render_user_template, validate_user_template
from prefect.types import StrictVariableValue
from prefect.types._datetime import DateTime, now, parse_datetime
from prefect.utilities.schema_tools.hydration import HydrationContext, HydrationError, Placeholder, ValidJinja, WorkspaceVariable, hydrate
from prefect.utilities.text import truncated_to

if TYPE_CHECKING:
    import logging
    from prefect.server.api.clients import OrchestrationClient
    from prefect.server.events.schemas.automations import TriggeredAction
    Parameters = Dict[str, Union[Any, Dict[str, Any], List[Union[Any, Dict[str, Any]]]]]

logger: logging.Logger = get_logger(__name__)

class ActionFailed(Exception):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason

class Action(PrefectBaseModel, abc.ABC):
    _result_details: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _resulting_related_resources: List[RelatedResource] = PrivateAttr(default_factory=list)

    @abc.abstractmethod
    async def act(self, triggered_action: TriggeredAction) -> None: ...

    async def fail(self, triggered_action: TriggeredAction, reason: str) -> None:
        from prefect.server.events.schemas.automations import EventTrigger
        automation = triggered_action.automation
        action = triggered_action.action
        action_index = triggered_action.action_index
        automation_resource_id = f'prefect.automation.{automation.id}'
        action_details = {'action_index': action_index, 'action_type': action.type, 'invocation': str(triggered_action.id)}
        resource = Resource({'prefect.resource.id': automation_resource_id, 'prefect.resource.name': automation.name, 'prefect.trigger-type': automation.trigger.type})
        if isinstance(automation.trigger, EventTrigger):
            resource['prefect.posture'] = automation.trigger.posture
        logger.warning('Action failed: %r', reason, extra={**self.logging_context(triggered_action)})
        async with PrefectServerEventsClient() as events:
            triggered_event_id = uuid4()
            await events.emit(Event(occurred=triggered_action.triggered, event='prefect.automation.action.triggered', resource=resource, related=self._resulting_related_resources, payload=action_details, id=triggered_event_id))
            await events.emit(Event(occurred=now('UTC'), event='prefect.automation.action.failed', resource=resource, related=self._resulting_related_resources, payload={**action_details, 'reason': reason, **self._result_details}, follows=triggered_event_id, id=uuid4()))

    async def succeed(self, triggered_action: TriggeredAction) -> None:
        from prefect.server.events.schemas.automations import EventTrigger
        automation = triggered_action.automation
        action = triggered_action.action
        action_index = triggered_action.action_index
        automation_resource_id = f'prefect.automation.{automation.id}'
        action_details = {'action_index': action_index, 'action_type': action.type, 'invocation': str(triggered_action.id)}
        resource = Resource({'prefect.resource.id': automation_resource_id, 'prefect.resource.name': automation.name, 'prefect.trigger-type': automation.trigger.type})
        if isinstance(automation.trigger, EventTrigger):
            resource['prefect.posture'] = automation.trigger.posture
        async with PrefectServerEventsClient() as events:
            triggered_event_id = uuid4()
            await events.emit(Event(occurred=triggered_action.triggered, event='prefect.automation.action.triggered', resource=Resource({'prefect.resource.id': automation_resource_id, 'prefect.resource.name': automation.name, 'prefect.trigger-type': automation.trigger.type}), related=self._resulting_related_resources, payload=action_details, id=triggered_event_id))
            await events.emit(Event(occurred=now('UTC'), event='prefect.automation.action.executed', resource=Resource({'prefect.resource.id': automation_resource_id, 'prefect.resource.name': automation.name, 'prefect.trigger-type': automation.trigger.type}), related=self._resulting_related_resources, payload={**action_details, **self._result_details}, id=uuid4(), follows=triggered_event_id))

    def logging_context(self, triggered_action: TriggeredAction) -> Dict[str, Any]:
        return {'automation': str(triggered_action.automation.id), 'action': self.model_dump(mode='json'), 'triggering_event': {'id': triggered_action.triggering_event.id, 'event': triggered_action.triggering_event.event} if triggered_action.triggering_event else None, 'triggering_labels': triggered_action.triggering_labels}

class DoNothing(Action):
    type: ClassVar[str] = 'do-nothing'

    async def act(self, triggered_action: TriggeredAction) -> None:
        logger.info('Doing nothing', extra={**self.logging_context(triggered_action)})

class EmitEventAction(Action):
    @abc.abstractmethod
    async def create_event(self, triggered_action: TriggeredAction) -> Event: ...

    async def act(self, triggered_action: TriggeredAction) -> None:
        event = await self.create_event(triggered_action)
        self._result_details['emitted_event'] = str(event.id)
        async with PrefectServerEventsClient() as events:
            await events.emit(event)

class ExternalDataAction(Action):
    async def orchestration_client(self, triggered_action: TriggeredAction) -> OrchestrationClient:
        from prefect.server.api.clients import OrchestrationClient
        return OrchestrationClient(additional_headers={'Prefect-Automation-ID': str(triggered_action.automation.id), 'Prefect-Automation-Name': b64encode(triggered_action.automation.name.encode()).decode()})

    async def events_api_client(self, triggered_action: TriggeredAction) -> PrefectServerEventsAPIClient:
        return PrefectServerEventsAPIClient(additional_headers={'Prefect-Automation-ID': str(triggered_action.automation.id), 'Prefect-Automation-Name': b64encode(triggered_action.automation.name.encode()).decode()})

    def reason_from_response(self, response: Response) -> str:
        error_detail = None
        if response.status_code in {409, 422}:
            try:
                error_detail = response.json().get('detail')
            except Exception:
                pass
            if response.status_code == 422 or error_detail:
                return f'Validation error occurred for {self.type!r}' + (f' - {error_detail}' if error_detail else '')
            else:
                return f'Conflict (409) occurred for {self.type!r} - {error_detail or response.text!r}'
        else:
            return f'Unexpected status from {self.type!r} action: {response.status_code}'

def _first_resource_of_kind(event: Event, expected_kind: str) -> Optional[Resource]:
    for resource in event.involved_resources:
        kind, _, _ = resource.id.rpartition('.')
        if kind == expected_kind:
            return resource
    return None

def _kind_and_id_from_resource(resource: Resource) -> Tuple[Optional[str], Optional[UUID]]:
    kind, _, id = resource.id.rpartition('.')
    try:
        return (kind, UUID(id))
    except ValueError:
        pass
    return (None, None)

def _id_from_resource_id(resource_id: str, expected_kind: str) -> Optional[UUID]:
    kind, _, id = resource_id.rpartition('.')
    if kind == expected_kind:
        try:
            return UUID(id)
        except ValueError:
            pass
    return None

def _id_of_first_resource_of_kind(event: Event, expected_kind: str) -> Optional[UUID]:
    resource = _first_resource_of_kind(event, expected_kind)
    if resource:
        if (id := _id_from_resource_id(resource.id, expected_kind)):
            return id
    return None

WorkspaceVariables = Dict[str, StrictVariableValue]
TemplateContextObject = Union[PrefectBaseModel, WorkspaceVariables, None]

class JinjaTemplateAction(ExternalDataAction):
    _object_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _registered_filters: bool = PrivateAttr(False)

    @classmethod
    def _register_filters_if_needed(cls) -> None:
        if not cls._registered_filters:
            from prefect.server.events.jinja_filters import all_filters
            register_user_template_filters(all_filters)
            cls._registered_filters = True

    @classmethod
    def validate_template(cls, template: str, field_name: str) -> str:
        cls._register_filters_if_needed()
        try:
            validate_user_template(template)
        except (jinja2.exceptions.TemplateSyntaxError, TemplateSecurityError) as exc:
            raise ValueError(f'{field_name!r} is not a valid template: {exc}')
        return template

    @classmethod
    def templates_in_dictionary(cls, dict_: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, str]]]:
        to_traverse: List[Dict[str, Any]] = []
        templates_at_layer: Dict[str, str] = {}
        for key, value in dict_.items():
            if isinstance(value, str) and maybe_template(value):
                templates_at_layer[key] = value
            elif isinstance(value, dict):
                to_traverse.append(value)
        templates: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
        if templates_at_layer:
            templates.append((dict_, templates_at_layer))
        for item in to_traverse:
            templates += cls.templates_in_dictionary(item)
        return templates

    def instantiate_object(self, model: Type[PrefectBaseModel], data: Dict[str, Any], triggered_action: TriggeredAction, resource: Optional[Resource] = None) -> Any:
        object = model.model_validate(data)
        if isinstance(object, (FlowRunResponse, TaskRun)):
            state_fields = ['prefect.state-message', 'prefect.state-name', 'prefect.state-timestamp', 'prefect.state-type']
            if resource and all((field in resource for field in state_fields)):
                try:
                    timestamp = parse_datetime(resource['prefect.state-timestamp'])
                    if TYPE_CHECKING:
                        assert isinstance(timestamp, DateTime)
                    object.state = State(message=resource['prefect.state-message'], name=resource['prefect.state-name'], timestamp=timestamp, type=StateType(resource['prefect.state-type']))
                except Exception:
                    logger.exception('Failed to parse state from event resource', extra={**self.logging_context(triggered_action)})
        return object

    async def _get_object_from_prefect_api(self, orchestration_client: OrchestrationClient, triggered_action: TriggeredAction, resource: Optional[Resource]) -> Optional[Any]:
        if not resource:
            return None
        kind, obj_id = _kind_and_id_from_resource(resource)
        if not obj_id:
            return None
        kind_to_model_and_methods: Dict[str, Tuple[Type[PrefectBaseModel], List[Callable[[UUID], Awaitable[Response]]]] = {
            'prefect.deployment': (DeploymentResponse, [orchestration_client.read_deployment_raw]),
            'prefect.flow': (Flow, [orchestration_client.read_flow_raw]),
            'prefect.flow-run': (FlowRunResponse, [orchestration_client.read_flow_run_raw]),
            'prefect.task-run': (TaskRun, [orchestration_client.read_task_run_raw]),
            'prefect.work-pool': (WorkPool, [orchestration_client.read_work_pool_raw]),
            'prefect.work-queue': (WorkQueueWithStatus, [orchestration_client.read_work_queue_raw, orchestration_client.read_work_queue_status_raw]),
            'prefect.concurrency-limit': (ConcurrencyLimitV2, [orchestration_client.read_concurrency_limit_v2_raw])
        }
        if kind not in kind_to_model_and_methods:
            return None
        model, client_methods = kind_to_model_and_methods[kind]
        responses = await asyncio.gather(*[client_method(obj_id) for client_method in client_methods])
        if any((response.status_code >= 300 for response in responses)):
            return None
        combined_response = {}
        for response in responses:
            data = response.json()
            if isinstance(data, list):
                if len(data) == 0:
                    return None
                data = data[0]
            combined_response.update(data)
        return self.instantiate_object(model, combined_response, triggered_action, resource=resource)

    async def _relevant_native_objects(self, templates: List[str], triggered_action: TriggeredAction) -> Dict[str, Any]:
        if not triggered_action.triggering_event:
            return {}
        orchestration_types = {'deployment', 'flow', 'flow_run', 'task_run', 'work_pool', 'work_queue', 'concurrency_limit'}
        special_types = {'variables'}
        types = matching_types_in_templates(templates, types=orchestration_types | special_types)
        if not types:
            return {}
        needed_types = list(set(types) - set(self._object_cache.keys()))
        async with await self.orchestration_client(triggered_action) as orchestration:
            calls = []
            for type_ in needed_types:
                if type_ in orchestration_types:
                    calls.append(self._get_object_from_prefect_api(orchestration, triggered_action, _first_resource_of_kind(triggered_action.triggering_event, f'prefect.{type_.replace('_', '-')}')))
                elif type_ == 'variables':
                    calls.append(orchestration.read_workspace_variables())
            objects = await asyncio.gather(*calls)
        self._object_cache.update(dict(zip(needed_types, objects)))
        return self._object_cache

    async def _template_context(self, templates: List[str], triggered_action: TriggeredAction) -> Dict[str, Any]:
        context = {'automation': triggered_action.automation, 'event': triggered_action.triggering_event, 'labels': LabelDiver(triggered_action.triggering_labels), 'firing': triggered_action.firing, 'firings': triggered_action.all_firings(), 'events': triggered_action.all_events()}
        context.update(await self._relevant_native_objects(templates, triggered_action))
        return context

    async def _render(self, templates: List[str], triggered_action: TriggeredAction) -> List[str]:
        self._register_filters_if_needed()
        context = await self._template_context(templates, triggered_action)
        return await asyncio.gather(*[render_user_template(template, context) for template in templates])

class DeploymentAction(Action):
    source: Literal['selected', 'inferred'] = Field('selected', description="Whether this Action applies to a specific selected deployment (given by `deployment_id`), or to a deployment that is inferred from the triggering event.  If the source is 'inferred', the `deployment_id` may not be set.  If the source is 'selected', the `deployment_id` must be set.")
    deployment_id: Optional[UUID] = Field(None, description='The identifier of the deployment')

    @model_validator(mode='after')
    def selected_deployment_requires_id(self) -> Self:
        wants_selected_deployment = self.source == 'selected'
        has_deployment_id = bool(self.deployment_id)
        if wants_selected_deployment != has_deployment_id:
            raise ValueError('deployment_id is ' + ('not allowed' if has_deployment_id else 'required'))
        return self

    async def deployment_id_to_use(self, triggered_action: TriggeredAction) -> UUID:
        if self.source == 'selected':
            assert self.deployment_id
            return self.deployment_id
        event = triggered_action.triggering_event
        if not event:
            raise ActionFailed('No event to infer the deployment')
        assert event
        if (id := _id_of_first_resource_of_kind(event, 'prefect.deployment')):
            return id
        raise ActionFailed('No deployment could be inferred')

class DeploymentCommandAction(DeploymentAction, ExternalDataAction):
    async def act(self, triggered_action: TriggeredAction) -> None:
        deployment_id = await self.deployment_id_to_use(triggered