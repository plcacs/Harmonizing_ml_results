#!/usr/bin/env python3
"""
The actions consumer watches for actions that have been triggered by Automations
and carries them out.  Also includes the various concrete subtypes of Actions
"""
from __future__ import annotations
import abc
import asyncio
import copy
from base64 import b64encode
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from uuid import UUID, uuid4
import jinja2
import orjson
from cachetools import TTLCache
from httpx import Response
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from typing_extensions import Self, TypeAlias

# TYPE_CHECKING imports
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
from prefect.server.utilities.user_templates import (
    TemplateSecurityError,
    matching_types_in_templates,
    maybe_template,
    register_user_template_filters,
    render_user_template,
    validate_user_template,
)
from prefect.types import StrictVariableValue
from prefect.types._datetime import DateTime, now, parse_datetime
from prefect.utilities.schema_tools.hydration import HydrationContext, HydrationError, Placeholder, ValidJinja, WorkspaceVariable, hydrate
from prefect.utilities.text import truncated_to

if False:
    # For type checking only
    from prefect.server.api.clients import OrchestrationClient
    from prefect.server.events.schemas.automations import TriggeredAction

Parameters = dict[str, Any | dict[str, Any] | list[Any | dict[str, Any]]]
logger = get_logger(__name__)


class ActionFailed(Exception):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason


class Action(PrefectBaseModel, abc.ABC):
    """An Action that may be performed when an Automation is triggered"""
    _result_details: dict[str, Any] = PrivateAttr(default_factory=dict)
    _resulting_related_resources: list[RelatedResource] = PrivateAttr(default_factory=list)

    @abc.abstractmethod
    async def act(self, triggered_action: "TriggeredAction") -> None:
        """Perform the requested Action"""

    async def fail(self, triggered_action: "TriggeredAction", reason: str) -> None:
        from prefect.server.events.schemas.automations import EventTrigger

        automation = triggered_action.automation
        action = triggered_action.action
        action_index: int = triggered_action.action_index
        automation_resource_id: str = f'prefect.automation.{automation.id}'
        action_details: dict[str, Any] = {
            'action_index': action_index,
            'action_type': action.type,
            'invocation': str(triggered_action.id),
        }
        resource: Resource = Resource({
            'prefect.resource.id': automation_resource_id,
            'prefect.resource.name': automation.name,
            'prefect.trigger-type': automation.trigger.type,
        })
        if isinstance(automation.trigger, EventTrigger):
            resource['prefect.posture'] = automation.trigger.posture
        logger.warning('Action failed: %r', reason, extra={**self.logging_context(triggered_action)})
        async with PrefectServerEventsClient() as events:
            triggered_event_id: UUID = uuid4()
            await events.emit(Event(
                occurred=triggered_action.triggered,
                event='prefect.automation.action.triggered',
                resource=resource,
                related=self._resulting_related_resources,
                payload=action_details,
                id=triggered_event_id,
            ))
            await events.emit(Event(
                occurred=now('UTC'),
                event='prefect.automation.action.failed',
                resource=resource,
                related=self._resulting_related_resources,
                payload={**action_details, 'reason': reason, **self._result_details},
                follows=triggered_event_id,
                id=uuid4(),
            ))

    async def succeed(self, triggered_action: "TriggeredAction") -> None:
        from prefect.server.events.schemas.automations import EventTrigger

        automation = triggered_action.automation
        action = triggered_action.action
        action_index: int = triggered_action.action_index
        automation_resource_id: str = f'prefect.automation.{automation.id}'
        action_details: dict[str, Any] = {
            'action_index': action_index,
            'action_type': action.type,
            'invocation': str(triggered_action.id),
        }
        resource: Resource = Resource({
            'prefect.resource.id': automation_resource_id,
            'prefect.resource.name': automation.name,
            'prefect.trigger-type': automation.trigger.type,
        })
        if isinstance(automation.trigger, EventTrigger):
            resource['prefect.posture'] = automation.trigger.posture
        async with PrefectServerEventsClient() as events:
            triggered_event_id: UUID = uuid4()
            await events.emit(Event(
                occurred=triggered_action.triggered,
                event='prefect.automation.action.triggered',
                resource=Resource({
                    'prefect.resource.id': automation_resource_id,
                    'prefect.resource.name': automation.name,
                    'prefect.trigger-type': automation.trigger.type,
                }),
                related=self._resulting_related_resources,
                payload=action_details,
                id=triggered_event_id,
            ))
            await events.emit(Event(
                occurred=now('UTC'),
                event='prefect.automation.action.executed',
                resource=Resource({
                    'prefect.resource.id': automation_resource_id,
                    'prefect.resource.name': automation.name,
                    'prefect.trigger-type': automation.trigger.type,
                }),
                related=self._resulting_related_resources,
                payload={**action_details, **self._result_details},
                id=uuid4(),
                follows=triggered_event_id,
            ))

    def logging_context(self, triggered_action: "TriggeredAction") -> dict[str, Any]:
        """Common logging context for all actions"""
        return {
            'automation': str(triggered_action.automation.id),
            'action': self.model_dump(mode='json'),
            'triggering_event': {
                'id': triggered_action.triggering_event.id,
                'event': triggered_action.triggering_event.event
            } if triggered_action.triggering_event else None,
            'triggering_labels': triggered_action.triggering_labels,
        }


class DoNothing(Action):
    """Do nothing when an Automation is triggered"""
    type: str = 'do-nothing'

    async def act(self, triggered_action: "TriggeredAction") -> None:
        logger.info('Doing nothing', extra={**self.logging_context(triggered_action)})


class EmitEventAction(Action):
    async def act(self, triggered_action: "TriggeredAction") -> None:
        event: Event = await self.create_event(triggered_action)
        self._result_details['emitted_event'] = str(event.id)
        async with PrefectServerEventsClient() as events:
            await events.emit(event)

    @abc.abstractmethod
    async def create_event(self, triggered_action: "TriggeredAction") -> Event:
        """Create an event from the TriggeredAction"""


class ExternalDataAction(Action):
    """Base class for Actions that require data from an external source such as
    the Orchestration API"""

    async def orchestration_client(self, triggered_action: "TriggeredAction") -> Any:
        # type: () -> Any
        from prefect.server.api.clients import OrchestrationClient
        return OrchestrationClient(additional_headers={
            'Prefect-Automation-ID': str(triggered_action.automation.id),
            'Prefect-Automation-Name': b64encode(triggered_action.automation.name.encode()).decode()
        })

    async def events_api_client(self, triggered_action: "TriggeredAction") -> PrefectServerEventsAPIClient:
        return PrefectServerEventsAPIClient(additional_headers={
            'Prefect-Automation-ID': str(triggered_action.automation.id),
            'Prefect-Automation-Name': b64encode(triggered_action.automation.name.encode()).decode()
        })

    def reason_from_response(self, response: Response) -> str:
        error_detail: Optional[str] = None
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
    kind, _, id_str = resource.id.rpartition('.')
    try:
        return (kind, UUID(id_str))
    except ValueError:
        pass
    return (None, None)


def _id_from_resource_id(resource_id: str, expected_kind: str) -> Optional[UUID]:
    kind, _, id_str = resource_id.rpartition('.')
    if kind == expected_kind:
        try:
            return UUID(id_str)
        except ValueError:
            pass
    return None


def _id_of_first_resource_of_kind(event: Event, expected_kind: str) -> Optional[UUID]:
    resource: Optional[Resource] = _first_resource_of_kind(event, expected_kind)
    if resource:
        if (id_val := _id_from_resource_id(resource.id, expected_kind)):
            return id_val
    return None


WorkspaceVariables: TypeAlias = Dict[str, StrictVariableValue]
TemplateContextObject: TypeAlias = Union[PrefectBaseModel, WorkspaceVariables, None]


class JinjaTemplateAction(ExternalDataAction):
    """Base class for Actions that use Jinja templates supplied by the user and
    are rendered with a context containing data from the triggered action,
    and the orchestration API."""
    _object_cache: dict[str, Any] = PrivateAttr(default_factory=dict)
    _registered_filters: bool = False

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

    def instantiate_object(
        self,
        model: Type[PrefectBaseModel],
        data: Any,
        triggered_action: "TriggeredAction",
        resource: Optional[Dict[str, Any]] = None,
    ) -> Any:
        obj: Any = model.model_validate(data)
        if isinstance(obj, FlowRunResponse) or isinstance(obj, TaskRun):
            state_fields = ['prefect.state-message', 'prefect.state-name', 'prefect.state-timestamp', 'prefect.state-type']
            if resource and all((field in resource for field in state_fields)):
                try:
                    timestamp: DateTime = parse_datetime(resource['prefect.state-timestamp'])
                    if TYPE_CHECKING:
                        assert isinstance(timestamp, DateTime)
                    obj.state = State(
                        message=resource['prefect.state-message'],
                        name=resource['prefect.state-name'],
                        timestamp=timestamp,
                        type=StateType(resource['prefect.state-type'])
                    )
                except Exception:
                    logger.exception('Failed to parse state from event resource', extra={**self.logging_context(triggered_action)})
        return obj

    async def _get_object_from_prefect_api(
        self,
        orchestration_client: Any,
        triggered_action: "TriggeredAction",
        resource: Optional[Resource],
    ) -> Optional[Any]:
        if not resource:
            return None
        kind, obj_id = _kind_and_id_from_resource(resource)
        if not obj_id:
            return None
        kind_to_model_and_methods: dict[str, Tuple[Type[PrefectBaseModel], List[Callable[[UUID], Awaitable[Response]]]]] = {
            'prefect.deployment': (DeploymentResponse, [orchestration_client.read_deployment_raw]),
            'prefect.flow': (Flow, [orchestration_client.read_flow_raw]),
            'prefect.flow-run': (FlowRunResponse, [orchestration_client.read_flow_run_raw]),
            'prefect.task-run': (TaskRun, [orchestration_client.read_task_run_raw]),
            'prefect.work-pool': (WorkPool, [orchestration_client.read_work_pool_raw]),
            'prefect.work-queue': (WorkQueueWithStatus, [orchestration_client.read_work_queue_raw, orchestration_client.read_work_queue_status_raw]),
            'prefect.concurrency-limit': (ConcurrencyLimitV2, [orchestration_client.read_concurrency_limit_v2_raw]),
        }
        if kind not in kind_to_model_and_methods:
            return None
        model, client_methods = kind_to_model_and_methods[kind]
        responses: List[Response] = await asyncio.gather(*[client_method(obj_id) for client_method in client_methods])
        if any((response.status_code >= 300 for response in responses)):
            return None
        combined_response: dict[str, Any] = {}
        for response in responses:
            data = response.json()
            if isinstance(data, list):
                if len(data) == 0:
                    return None
                data = data[0]
            combined_response.update(data)
        return self.instantiate_object(model, combined_response, triggered_action, resource=resource)

    async def _relevant_native_objects(self, templates: List[Tuple[Dict[str, Any], Dict[str, str]]], triggered_action: "TriggeredAction") -> dict[str, Any]:
        if not triggered_action.triggering_event:
            return {}
        orchestration_types = {'deployment', 'flow', 'flow_run', 'task_run', 'work_pool', 'work_queue', 'concurrency_limit'}
        special_types = {'variables'}
        types: set[str] = matching_types_in_templates(templates, types=orchestration_types | special_types)
        if not types:
            return {}
        needed_types: list[str] = list(set(types) - set(self._object_cache.keys()))
        async with await self.orchestration_client(triggered_action) as orchestration:
            calls: List[Awaitable[Any]] = []
            for type_ in needed_types:
                if type_ in orchestration_types:
                    calls.append(
                        self._get_object_from_prefect_api(
                            orchestration, 
                            triggered_action, 
                            _first_resource_of_kind(triggered_action.triggering_event, f'prefect.{type_.replace("_", "-")}')
                        )
                    )
                elif type_ == 'variables':
                    calls.append(orchestration.read_workspace_variables())
            objects: List[Any] = await asyncio.gather(*calls)
        self._object_cache.update(dict(zip(needed_types, objects)))
        return self._object_cache

    async def _template_context(self, templates: List[Tuple[Dict[str, Any], Dict[str, str]]], triggered_action: "TriggeredAction") -> dict[str, Any]:
        context: dict[str, Any] = {
            'automation': triggered_action.automation,
            'event': triggered_action.triggering_event,
            'labels': LabelDiver(triggered_action.triggering_labels),
            'firing': triggered_action.firing,
            'firings': triggered_action.all_firings(),
            'events': triggered_action.all_events(),
        }
        context.update(await self._relevant_native_objects(templates, triggered_action))
        return context

    async def _render(self, templates: List[str], triggered_action: "TriggeredAction") -> List[str]:
        self._register_filters_if_needed()
        context: dict[str, Any] = await self._template_context([], triggered_action)
        return await asyncio.gather(*[render_user_template(template, context) for template in templates])


class DeploymentAction(Action):
    """Base class for Actions that operate on Deployments and need to infer them from
    events"""
    source: str = Field('selected', description="Whether this Action applies to a specific selected deployment (given by `deployment_id`), or to a deployment that is inferred from the triggering event.  If the source is 'inferred', the `deployment_id` may not be set.  If the source is 'selected', the `deployment_id` must be set.")
    deployment_id: Optional[UUID] = Field(None, description='The identifier of the deployment')

    @model_validator(mode='after')
    def selected_deployment_requires_id(self) -> DeploymentAction:
        wants_selected_deployment: bool = self.source == 'selected'
        has_deployment_id: bool = bool(self.deployment_id)
        if wants_selected_deployment != has_deployment_id:
            raise ValueError('deployment_id is ' + ('not allowed' if has_deployment_id else 'required'))
        return self

    async def deployment_id_to_use(self, triggered_action: "TriggeredAction") -> UUID:
        if self.source == 'selected':
            assert self.deployment_id
            return self.deployment_id
        event: Optional[Event] = triggered_action.triggering_event
        if not event:
            raise ActionFailed('No event to infer the deployment')
        if (id_val := _id_of_first_resource_of_kind(event, 'prefect.deployment')):
            return id_val
        raise ActionFailed('No deployment could be inferred')


class DeploymentCommandAction(DeploymentAction, ExternalDataAction):
    """Executes a command against a matching deployment"""
    async def act(self, triggered_action: "TriggeredAction") -> None:
        deployment_id: UUID = await self.deployment_id_to_use(triggered_action)
        self._resulting_related_resources.append(RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.deployment.{deployment_id}',
            'prefect.resource.role': 'target'
        }))
        logger.info(self._action_description, extra={'deployment_id': deployment_id, **self.logging_context(triggered_action)})
        async with await self.orchestration_client(triggered_action) as orchestration:
            response: Response = await self.command(orchestration, deployment_id, triggered_action)
            self._result_details['status_code'] = response.status_code
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))

    @abc.abstractmethod
    async def command(self, orchestration: Any, deployment_id: UUID, triggered_action: "TriggeredAction") -> Response:
        """Execute the deployment command"""


class RunDeployment(JinjaTemplateAction, DeploymentCommandAction):
    """Runs the given deployment with the given parameters"""
    type: str = 'run-deployment'
    parameters: Optional[Dict[str, Any]] = Field(None, description="The parameters to pass to the deployment, or None to use the deployment's default parameters")
    job_variables: Optional[Dict[str, Any]] = Field(None, description="The job variables to pass to the created flow run, or None to use the deployment's default job variables")
    _action_description: str = 'Running deployment'

    async def command(self, orchestration: Any, deployment_id: UUID, triggered_action: "TriggeredAction") -> Response:
        state: Scheduled = Scheduled()
        try:
            flow_run_create: DeploymentFlowRunCreate = DeploymentFlowRunCreate(
                state=StateCreate(
                    type=state.type,
                    name=state.name,
                    message=state.message,
                    state_details=state.state_details
                ),
                parameters=await self.render_parameters(triggered_action),
                idempotency_key=triggered_action.idempotency_key(),
                job_variables=self.job_variables
            )
        except Exception as exc:
            raise ActionFailed(f'Unable to create flow run from deployment: {exc!r}')
        response: Response = await orchestration.create_flow_run(deployment_id, flow_run_create)
        if response.status_code < 300:
            flow_run: FlowRunResponse = FlowRunResponse.model_validate(response.json())
            self._resulting_related_resources.append(RelatedResource.model_validate({
                'prefect.resource.id': f'prefect.flow-run.{flow_run.id}',
                'prefect.resource.role': 'flow-run',
                'prefect.resource.name': flow_run.name
            }))
            logger.info('Started flow run', extra={'flow_run': {'id': str(flow_run.id), 'name': flow_run.name}, **self.logging_context(triggered_action)})
        if response.status_code == 409:
            self._result_details['validation_error'] = response.json().get('detail')
        return response

    @field_validator('parameters')
    def validate_parameters(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not value:
            return value
        for_testing: Dict[str, Any] = copy.deepcopy(value) or {}
        cls._upgrade_v1_templates(for_testing)
        problems = cls._collect_errors(hydrate(for_testing, HydrationContext(raise_on_error=False, render_workspace_variables=False, render_jinja=False)))
        if not problems:
            return value
        raise ValueError('Invalid parameters: \n' + '\n  '.join((f'{(k + ":" if k else "")} {e.message}' for k, e in problems.items())))

    @classmethod
    def _collect_errors(cls, hydrated: Any, prefix: str = '') -> Dict[str, HydrationError]:
        problems: Dict[str, HydrationError] = {}
        if isinstance(hydrated, HydrationError):
            problems[prefix] = hydrated
        if isinstance(hydrated, Placeholder):
            return problems
        if isinstance(hydrated, dict):
            for key, value in hydrated.items():
                if isinstance(value, dict):
                    problems.update(cls._collect_errors(value, f'{prefix}{key}.'))
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            problems.update(cls._collect_errors(item, f'{prefix}{key}[{index}].'))
                        elif isinstance(item, HydrationError):
                            problems[f'{prefix}{key}[{index}]'] = item
                elif isinstance(value, HydrationError):
                    problems[f'{prefix}{key}'] = value
        return problems

    async def render_parameters(self, triggered_action: "TriggeredAction") -> Dict[str, Any]:
        parameters: Dict[str, Any] = copy.deepcopy(self.parameters) or {}
        self._upgrade_v1_templates(parameters)
        placeholders: List[Placeholder] = self._collect_placeholders(
            hydrate(parameters, HydrationContext(raise_on_error=False, render_jinja=False, render_workspace_variables=False))
        )
        templates: List[str] = [p.template for p in placeholders if isinstance(p, ValidJinja)]
        template_context: dict[str, Any] = await self._template_context([], triggered_action)
        variable_names: List[str] = [p.variable_name for p in placeholders if isinstance(p, WorkspaceVariable)]
        workspace_variables: dict[str, Any] = {}
        if variable_names:
            async with await self.orchestration_client(triggered_action) as client:
                workspace_variables = await client.read_workspace_variables(variable_names)
        parameters = hydrate(parameters, HydrationContext(
            raise_on_error=True,
            render_jinja=True,
            jinja_context=template_context,
            render_workspace_variables=True,
            workspace_variables=workspace_variables
        ))
        return parameters

    @classmethod
    def _upgrade_v1_templates(cls, parameters: Dict[str, Any]) -> None:
        """
        Upgrades all v1-style template values from the parameters dictionary, changing
        the values in the given dictionary.  v1-style templates are any plain strings
        that include Jinja2 template syntax.
        """
        for key, value in parameters.items():
            if isinstance(value, dict):
                if '__prefect_kind' in value:
                    continue
                cls._upgrade_v1_templates(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        cls._upgrade_v1_templates(item)
                    elif isinstance(item, str) and maybe_template(item):
                        value[i] = {'__prefect_kind': 'jinja', 'template': item}
            elif isinstance(value, str) and maybe_template(value):
                parameters[key] = {'__prefect_kind': 'jinja', 'template': value}

    def _collect_placeholders(self, parameters: Any) -> List[Placeholder]:
        """
        Recursively collects all placeholder values embedded within the parameters
        dictionary, including templates and workspace variables
        """
        placeholders: List[Placeholder] = []
        if isinstance(parameters, Placeholder):
            return [parameters]
        if isinstance(parameters, dict):
            for _, value in parameters.items():
                if isinstance(value, dict):
                    placeholders += self._collect_placeholders(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            placeholders += self._collect_placeholders(item)
                        elif isinstance(item, Placeholder):
                            placeholders.append(item)
                elif isinstance(value, Placeholder):
                    placeholders.append(value)
        return placeholders


class PauseDeployment(DeploymentCommandAction):
    """Pauses the given Deployment"""
    type: str = 'pause-deployment'
    _action_description: str = 'Pausing deployment'

    async def command(self, orchestration: Any, deployment_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.pause_deployment(deployment_id)


class ResumeDeployment(DeploymentCommandAction):
    """Resumes the given Deployment"""
    type: str = 'resume-deployment'
    _action_description: str = 'Resuming deployment'

    async def command(self, orchestration: Any, deployment_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.resume_deployment(deployment_id)


class FlowRunAction(ExternalDataAction):
    """An action that operates on a flow run"""

    async def flow_run(self, triggered_action: "TriggeredAction") -> UUID:
        labels: dict[str, Any] = triggered_action.triggering_labels
        if (triggering_resource_id := labels.get('prefect.resource.id')):
            if (id_val := _id_from_resource_id(triggering_resource_id, 'prefect.flow-run')):
                return id_val
        event: Optional[Event] = triggered_action.triggering_event
        if event:
            if (id_val := _id_of_first_resource_of_kind(event, 'prefect.flow-run')):
                return id_val
        raise ActionFailed('No flow run could be inferred')


class FlowRunStateChangeAction(FlowRunAction):
    """Changes the state of a flow run associated with the trigger"""

    @abc.abstractmethod
    async def new_state(self, triggered_action: "TriggeredAction") -> StateCreate:
        """Return the new state for the flow run"""

    async def act(self, triggered_action: "TriggeredAction") -> None:
        flow_run_id: UUID = await self.flow_run(triggered_action)
        self._resulting_related_resources.append(RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.flow-run.{flow_run_id}',
            'prefect.resource.role': 'target'
        }))
        logger.info('Changing flow run state', extra={'flow_run_id': str(flow_run_id), **self.logging_context(triggered_action)})
        async with await self.orchestration_client(triggered_action) as orchestration:
            response: Response = await orchestration.set_flow_run_state(flow_run_id, await self.new_state(triggered_action=triggered_action))
            self._result_details['status_code'] = response.status_code
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))
            result: OrchestrationResult = OrchestrationResult.model_validate(response.json())
            if not isinstance(result.details, StateAcceptDetails):
                raise ActionFailed(f'Failed to set state: {result.details.reason}')


class ChangeFlowRunState(FlowRunStateChangeAction):
    """Changes the state of a flow run associated with the trigger"""
    type: str = 'change-flow-run-state'
    name: Optional[str] = Field(None, description='The name of the state to change the flow run to')
    state: str = Field(..., description='The type of the state to change the flow run to')
    message: Optional[str] = Field(None, description='An optional message to associate with the state change')

    async def new_state(self, triggered_action: "TriggeredAction") -> StateCreate:
        message: str = self.message or f'State changed by Automation {triggered_action.automation.id}'
        return StateCreate(name=self.name, type=self.state, message=message)


class CancelFlowRun(FlowRunStateChangeAction):
    """Cancels a flow run associated with the trigger"""
    type: str = 'cancel-flow-run'

    async def new_state(self, triggered_action: "TriggeredAction") -> StateCreate:
        return StateCreate(type=StateType.CANCELLING, message=f'Cancelled by Automation {triggered_action.automation.id}')


class SuspendFlowRun(FlowRunStateChangeAction):
    """Suspends a flow run associated with the trigger"""
    type: str = 'suspend-flow-run'

    async def new_state(self, triggered_action: "TriggeredAction") -> StateCreate:
        state: Suspended = Suspended(timeout_seconds=3600, message=f'Suspended by Automation {triggered_action.automation.id}')
        return StateCreate(type=state.type, name=state.name, message=state.message, state_details=state.state_details)


class ResumeFlowRun(FlowRunAction):
    """Resumes a paused or suspended flow run associated with the trigger"""
    type: str = 'resume-flow-run'

    async def act(self, triggered_action: "TriggeredAction") -> None:
        flow_run_id: UUID = await self.flow_run(triggered_action)
        self._resulting_related_resources.append(RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.flow-run.{flow_run_id}',
            'prefect.resource.role': 'target'
        }))
        logger.debug('Resuming flow run', extra={'flow_run_id': str(flow_run_id), **self.logging_context(triggered_action)})
        async with await self.orchestration_client(triggered_action) as orchestration:
            result: OrchestrationResult = await orchestration.resume_flow_run(flow_run_id)
            if not isinstance(result.details, StateAcceptDetails):
                raise ActionFailed(f'Failed to resume flow run: {result.details.reason}')


class CallWebhook(JinjaTemplateAction):
    """Call a webhook when an Automation is triggered."""
    type: str = 'call-webhook'
    block_document_id: str = Field(..., description='The identifier of the webhook block to use')
    payload: str = Field(default='', description='An optional templatable payload to send when calling the webhook.')

    @field_validator('payload', mode='before')
    @classmethod
    def ensure_payload_is_a_string(cls, value: Any) -> Any:
        """Temporary measure while we migrate payloads from being a dictionary to
        a string template.  This covers both reading from the database where values
        may currently be a dictionary, as well as the API, where older versions of the
        frontend may be sending a JSON object with the single `"message"` key."""
        if value is None:
            return value
        if isinstance(value, str):
            return value
        return orjson.dumps(value, option=orjson.OPT_INDENT_2).decode()

    @field_validator('payload')
    @classmethod
    def validate_payload_templates(cls, value: str) -> str:
        """
        Validate user-provided payload template.
        """
        if not value:
            return value
        cls.validate_template(value, 'payload')
        return value

    async def _get_webhook_block(self, triggered_action: "TriggeredAction") -> Webhook:
        async with await self.orchestration_client(triggered_action) as orchestration:
            response: Response = await orchestration.read_block_document_raw(self.block_document_id)
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))
            try:
                block_document: BlockDocument = BlockDocument.model_validate(response.json())
                block: Block = Block._from_block_document(block_document)
            except Exception as e:
                raise ActionFailed(f'The webhook block was invalid: {e!r}')
            if not isinstance(block, Webhook):
                raise ActionFailed('The referenced block was not a webhook block')
            self._resulting_related_resources += [
                RelatedResource.model_validate({
                    'prefect.resource.id': f'prefect.block-document.{self.block_document_id}',
                    'prefect.resource.role': 'block',
                    'prefect.resource.name': block_document.name
                }),
                RelatedResource.model_validate({
                    'prefect.resource.id': f'prefect.block-type.{block.get_block_type_slug()}',
                    'prefect.resource.role': 'block-type'
                })
            ]
            return block

    async def act(self, triggered_action: "TriggeredAction") -> None:
        block: Webhook = await self._get_webhook_block(triggered_action=triggered_action)
        rendered: List[str] = await self._render([self.payload], triggered_action)
        payload: str = rendered[0]
        try:
            response: Response = await block.call(payload=payload)
            ok_headers: dict[str, Any] = {k: v for k, v in response.headers.items() if not should_redact_header(k)}
            self._result_details.update({
                'status_code': response.status_code,
                'response_body': truncated_to(1000, response.text),
                'response_headers': {**(ok_headers or {})}
            })
        except Exception as e:
            raise ActionFailed(f'Webhook call failed: {e!r}')


class SendNotification(JinjaTemplateAction):
    """Send a notification when an Automation is triggered"""
    type: str = 'send-notification'
    block_document_id: str = Field(..., description='The identifier of the notification block to use')
    subject: str = Field('Prefect automated notification')
    body: str = Field(..., description='The text of the notification to send')

    @field_validator('subject', 'body')
    def is_valid_template(cls, value: str, info: Any) -> str:
        if TYPE_CHECKING:
            assert isinstance(info.field_name, str)
        return cls.validate_template(value, info.field_name)

    async def _get_notification_block(self, triggered_action: "TriggeredAction") -> NotificationBlock:
        async with await self.orchestration_client(triggered_action) as orion:
            response: Response = await orion.read_block_document_raw(self.block_document_id)
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))
            try:
                block_document: BlockDocument = BlockDocument.model_validate(response.json())
                block: Block = Block._from_block_document(block_document)
            except Exception as e:
                raise ActionFailed(f'The notification block was invalid: {e!r}')
            if 'notify' not in block.get_block_capabilities():
                raise ActionFailed('The referenced block was not a notification block')
            self._resulting_related_resources += [
                RelatedResource.model_validate({
                    'prefect.resource.id': f'prefect.block-document.{self.block_document_id}',
                    'prefect.resource.role': 'block',
                    'prefect.resource.name': block_document.name
                }),
                RelatedResource.model_validate({
                    'prefect.resource.id': f'prefect.block-type.{block.get_block_type_slug()}',
                    'prefect.resource.role': 'block-type'
                })
            ]
            return cast(NotificationBlock, block)

    async def act(self, triggered_action: "TriggeredAction") -> None:
        block: NotificationBlock = await self._get_notification_block(triggered_action=triggered_action)
        subject_str, body_str = await self.render(triggered_action)
        with block.raise_on_failure():
            try:
                await block.notify(subject=subject_str, body=body_str)
            except NotificationError as e:
                self._result_details['notification_log'] = e.log
                raise ActionFailed('Notification failed')

    async def render(self, triggered_action: "TriggeredAction") -> Tuple[str, str]:
        rendered: List[str] = await self._render([self.subject, self.body], triggered_action)
        return (rendered[0], rendered[1])


class WorkPoolAction(Action):
    """Base class for Actions that operate on Work Pools and need to infer them from
    events"""
    source: str = Field('selected', description="Whether this Action applies to a specific selected work pool (given by `work_pool_id`), or to a work pool that is inferred from the triggering event.  If the source is 'inferred', the `work_pool_id` may not be set.  If the source is 'selected', the `work_pool_id` must be set.")
    work_pool_id: Optional[UUID] = Field(None, description='The identifier of the work pool to pause')

    @model_validator(mode='after')
    def selected_work_pool_requires_id(self) -> WorkPoolAction:
        wants_selected_work_pool: bool = self.source == 'selected'
        has_work_pool_id: bool = bool(self.work_pool_id)
        if wants_selected_work_pool != has_work_pool_id:
            raise ValueError('work_pool_id is ' + ('not allowed' if has_work_pool_id else 'required'))
        return self

    async def work_pool_id_to_use(self, triggered_action: "TriggeredAction") -> UUID:
        if self.source == 'selected':
            assert self.work_pool_id
            return self.work_pool_id
        event: Optional[Event] = triggered_action.triggering_event
        if not event:
            raise ActionFailed('No event to infer the work pool')
        if (id_val := _id_of_first_resource_of_kind(event, 'prefect.work-pool')):
            return id_val
        raise ActionFailed('No work pool could be inferred')


class WorkPoolCommandAction(WorkPoolAction, ExternalDataAction):
    _target_work_pool: Optional[Any] = PrivateAttr(default=None)

    async def target_work_pool(self, triggered_action: "TriggeredAction") -> Any:
        if not self._target_work_pool:
            work_pool_id: UUID = await self.work_pool_id_to_use(triggered_action)
            async with await self.orchestration_client(triggered_action) as orchestration:
                work_pool = await orchestration.read_work_pool(work_pool_id)
                if not work_pool:
                    raise ActionFailed(f'Work pool {work_pool_id} not found')
                self._target_work_pool = work_pool
        return self._target_work_pool

    async def act(self, triggered_action: "TriggeredAction") -> None:
        work_pool: Any = await self.target_work_pool(triggered_action)
        self._resulting_related_resources += [RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.work-pool.{work_pool.id}',
            'prefect.resource.name': work_pool.name,
            'prefect.resource.role': 'target'
        })]
        logger.info(self._action_description, extra={'work_pool_id': work_pool.id, **self.logging_context(triggered_action)})
        async with await self.orchestration_client(triggered_action) as orchestration:
            response: Response = await self.command(orchestration, work_pool, triggered_action)
            self._result_details['status_code'] = response.status_code
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))

    @abc.abstractmethod
    async def command(self, orchestration: Any, work_pool: Any, triggered_action: "TriggeredAction") -> Response:
        """Issue the command to the Work Pool"""


class PauseWorkPool(WorkPoolCommandAction):
    """Pauses a Work Pool"""
    type: str = 'pause-work-pool'
    _action_description: str = 'Pausing work pool'

    async def command(self, orchestration: Any, work_pool: Any, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.pause_work_pool(work_pool.name)


class ResumeWorkPool(WorkPoolCommandAction):
    """Resumes a Work Pool"""
    type: str = 'resume-work-pool'
    _action_description: str = 'Resuming work pool'

    async def command(self, orchestration: Any, work_pool: Any, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.resume_work_pool(work_pool.name)


class WorkQueueAction(Action):
    """Base class for Actions that operate on Work Queues and need to infer them from
    events"""
    source: str = Field('selected', description="Whether this Action applies to a specific selected work queue (given by `work_queue_id`), or to a work queue that is inferred from the triggering event.  If the source is 'inferred', the `work_queue_id` may not be set.  If the source is 'selected', the `work_queue_id` must be set.")
    work_queue_id: Optional[UUID] = Field(None, description='The identifier of the work queue to pause')

    @model_validator(mode='after')
    def selected_work_queue_requires_id(self) -> WorkQueueAction:
        wants_selected_work_queue: bool = self.source == 'selected'
        has_work_queue_id: bool = bool(self.work_queue_id)
        if wants_selected_work_queue != has_work_queue_id:
            raise ValueError('work_queue_id is ' + ('not allowed' if has_work_queue_id else 'required'))
        return self

    async def work_queue_id_to_use(self, triggered_action: "TriggeredAction") -> UUID:
        if self.source == 'selected':
            assert self.work_queue_id
            return self.work_queue_id
        event: Optional[Event] = triggered_action.triggering_event
        if not event:
            raise ActionFailed('No event to infer the work queue')
        if (id_val := _id_of_first_resource_of_kind(event, 'prefect.work-queue')):
            return id_val
        raise ActionFailed('No work queue could be inferred')


class WorkQueueCommandAction(WorkQueueAction, ExternalDataAction):
    async def act(self, triggered_action: "TriggeredAction") -> None:
        work_queue_id: UUID = await self.work_queue_id_to_use(triggered_action)
        self._resulting_related_resources += [RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.work-queue.{work_queue_id}',
            'prefect.resource.role': 'target'
        })]
        logger.info(self._action_description, extra={'work_queue_id': work_queue_id, **self.logging_context(triggered_action)})
        async with await self.orchestration_client(triggered_action) as orchestration:
            response: Response = await self.command(orchestration, work_queue_id, triggered_action)
            self._result_details['status_code'] = response.status_code
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))

    @abc.abstractmethod
    async def command(self, orchestration: Any, work_queue_id: UUID, triggered_action: "TriggeredAction") -> Response:
        """Issue the command to the Work Queue"""


class PauseWorkQueue(WorkQueueCommandAction):
    """Pauses a Work Queue"""
    type: str = 'pause-work-queue'
    _action_description: str = 'Pausing work queue'

    async def command(self, orchestration: Any, work_queue_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.pause_work_queue(work_queue_id)


class ResumeWorkQueue(WorkQueueCommandAction):
    """Resumes a Work Queue"""
    type: str = 'resume-work-queue'
    _action_description: str = 'Resuming work queue'

    async def command(self, orchestration: Any, work_queue_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await orchestration.resume_work_queue(work_queue_id)


class AutomationAction(Action):
    """Base class for Actions that operate on Automations and need to infer them from
    events"""
    source: str = Field('selected', description="Whether this Action applies to a specific selected automation (given by `automation_id`), or to an automation that is inferred from the triggering event.  If the source is 'inferred', the `automation_id` may not be set.  If the source is 'selected', the `automation_id` must be set.")
    automation_id: Optional[UUID] = Field(None, description='The identifier of the automation to act on')

    @model_validator(mode='after')
    def selected_automation_requires_id(self) -> AutomationAction:
        wants_selected_automation: bool = self.source == 'selected'
        has_automation_id: bool = bool(self.automation_id)
        if wants_selected_automation != has_automation_id:
            raise ValueError('automation_id is ' + ('not allowed' if has_automation_id else 'required'))
        return self

    async def automation_id_to_use(self, triggered_action: "TriggeredAction") -> UUID:
        if self.source == 'selected':
            assert self.automation_id
            return self.automation_id
        event: Optional[Event] = triggered_action.triggering_event
        if not event:
            raise ActionFailed('No event to infer the automation')
        if (id_val := _id_of_first_resource_of_kind(event, 'prefect.automation')):
            return id_val
        raise ActionFailed('No automation could be inferred')


class AutomationCommandAction(AutomationAction, ExternalDataAction):
    async def act(self, triggered_action: "TriggeredAction") -> None:
        automation_id: UUID = await self.automation_id_to_use(triggered_action)
        self._resulting_related_resources += [RelatedResource.model_validate({
            'prefect.resource.id': f'prefect.automation.{automation_id}',
            'prefect.resource.role': 'target'
        })]
        logger.info(self._action_description, extra={'automation_id': automation_id, **self.logging_context(triggered_action)})
        async with await self.events_api_client(triggered_action) as events:
            response: Response = await self.command(events, automation_id, triggered_action)
            self._result_details['status_code'] = response.status_code
            if response.status_code >= 300:
                raise ActionFailed(self.reason_from_response(response))

    @abc.abstractmethod
    async def command(self, events: Any, automation_id: UUID, triggered_action: "TriggeredAction") -> Response:
        """Issue the command to the Work Queue"""


class PauseAutomation(AutomationCommandAction):
    """Pauses a Work Queue"""
    type: str = 'pause-automation'
    _action_description: str = 'Pausing automation'

    async def command(self, events: Any, automation_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await events.pause_automation(automation_id)


class ResumeAutomation(AutomationCommandAction):
    """Resumes a Work Queue"""
    type: str = 'resume-automation'
    _action_description: str = 'Resuming auitomation'

    async def command(self, events: Any, automation_id: UUID, triggered_action: "TriggeredAction") -> Response:
        return await events.resume_automation(automation_id)


ServerActionTypes = Union[
    DoNothing,
    RunDeployment,
    PauseDeployment,
    ResumeDeployment,
    CancelFlowRun,
    ChangeFlowRunState,
    PauseWorkQueue,
    ResumeWorkQueue,
    SendNotification,
    CallWebhook,
    PauseAutomation,
    ResumeAutomation,
    SuspendFlowRun,
    ResumeFlowRun,
    PauseWorkPool,
    ResumeWorkPool,
]
_recent_actions: TTLCache[UUID, bool] = TTLCache(maxsize=10000, ttl=3600)


async def record_action_happening(id: UUID) -> None:
    """Record that an action has happened, with an expiration of an hour."""
    _recent_actions[id] = True


async def action_has_already_happened(id: UUID) -> bool:
    """Check if the action has already happened"""
    return _recent_actions.get(id, False)


@asynccontextmanager
async def consumer() -> AsyncGenerator[Callable[[Message], Awaitable[None]], None]:
    from prefect.server.events.schemas.automations import TriggeredAction

    async def message_handler(message: Message) -> None:
        if not message.data:
            return
        triggered_action = TriggeredAction.model_validate_json(message.data)
        action: Action = triggered_action.action
        if await action_has_already_happened(triggered_action.id):
            logger.info('Action %s has already been executed, skipping', triggered_action.id)
            return
        try:
            await action.act(triggered_action)
        except ActionFailed as e:
            await action.fail(triggered_action, e.reason)
        else:
            await action.succeed(triggered_action)
            await record_action_happening(triggered_action.id)
    logger.info('Starting action message handler')
    yield message_handler