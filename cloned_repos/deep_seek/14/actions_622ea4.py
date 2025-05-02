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

    async def fail(self, triggered_action: TriggeredAction, reason: str) -> None: ...

    async def succeed(self, triggered_action: TriggeredAction) -> None: ...

    def logging_context(self, triggered_action: TriggeredAction) -> Dict[str, Any]: ...

class DoNothing(Action):
    type: ClassVar[str] = 'do-nothing'

    async def act(self, triggered_action: TriggeredAction) -> None: ...

class EmitEventAction(Action):
    async def act(self, triggered_action: TriggeredAction) -> None: ...

    @abc.abstractmethod
    async def create_event(self, triggered_action: TriggeredAction) -> Event: ...

class ExternalDataAction(Action):
    async def orchestration_client(self, triggered_action: TriggeredAction) -> OrchestrationClient: ...

    async def events_api_client(self, triggered_action: TriggeredAction) -> PrefectServerEventsAPIClient: ...

    def reason_from_response(self, response: Response) -> str: ...

def _first_resource_of_kind(event: Event, expected_kind: str) -> Optional[Resource]: ...

def _kind_and_id_from_resource(resource: Resource) -> Tuple[Optional[str], Optional[UUID]]: ...

def _id_from_resource_id(resource_id: str, expected_kind: str) -> Optional[UUID]: ...

def _id_of_first_resource_of_kind(event: Event, expected_kind: str) -> Optional[UUID]: ...

WorkspaceVariables = Dict[str, StrictVariableValue]
TemplateContextObject = Union[PrefectBaseModel, WorkspaceVariables, None]

class JinjaTemplateAction(ExternalDataAction):
    _object_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _registered_filters: bool = PrivateAttr(default=False)

    @classmethod
    def _register_filters_if_needed(cls) -> None: ...

    @classmethod
    def validate_template(cls, template: str, field_name: str) -> str: ...

    @classmethod
    def templates_in_dictionary(cls, dict_: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, str]]]: ...

    def instantiate_object(self, model: Type[PrefectBaseModel], data: Dict[str, Any], triggered_action: TriggeredAction, resource: Optional[Resource] = None) -> Any: ...

    async def _get_object_from_prefect_api(self, orchestration_client: OrchestrationClient, triggered_action: TriggeredAction, resource: Optional[Resource]) -> Optional[Any]: ...

    async def _relevant_native_objects(self, templates: List[str], triggered_action: TriggeredAction) -> Dict[str, Any]: ...

    async def _template_context(self, templates: List[str], triggered_action: TriggeredAction) -> Dict[str, Any]: ...

    async def _render(self, templates: List[str], triggered_action: TriggeredAction) -> List[str]: ...

class DeploymentAction(Action):
    source: Literal['selected', 'inferred'] = Field('selected')
    deployment_id: Optional[UUID] = Field(None)

    @model_validator(mode='after')
    def selected_deployment_requires_id(self) -> Self: ...

    async def deployment_id_to_use(self, triggered_action: TriggeredAction) -> UUID: ...

class DeploymentCommandAction(DeploymentAction, ExternalDataAction):
    _action_description: str

    async def act(self, triggered_action: TriggeredAction) -> None: ...

    @abc.abstractmethod
    async def command(self, orchestration: OrchestrationClient, deployment_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class RunDeployment(JinjaTemplateAction, DeploymentCommandAction):
    type: ClassVar[str] = 'run-deployment'
    parameters: Optional[Dict[str, Any]] = Field(None)
    job_variables: Optional[Dict[str, Any]] = Field(None)
    _action_description: ClassVar[str] = 'Running deployment'

    async def command(self, orchestration: OrchestrationClient, deployment_id: UUID, triggered_action: TriggeredAction) -> Response: ...

    @field_validator('parameters')
    def validate_parameters(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]: ...

    @classmethod
    def _collect_errors(cls, hydrated: Any, prefix: str = '') -> Dict[str, HydrationError]: ...

    async def render_parameters(self, triggered_action: TriggeredAction) -> Dict[str, Any]: ...

    @classmethod
    def _upgrade_v1_templates(cls, parameters: Dict[str, Any]) -> None: ...

    def _collect_placeholders(self, parameters: Dict[str, Any]) -> List[Placeholder]: ...

class PauseDeployment(DeploymentCommandAction):
    type: ClassVar[str] = 'pause-deployment'
    _action_description: ClassVar[str] = 'Pausing deployment'

    async def command(self, orchestration: OrchestrationClient, deployment_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class ResumeDeployment(DeploymentCommandAction):
    type: ClassVar[str] = 'resume-deployment'
    _action_description: ClassVar[str] = 'Resuming deployment'

    async def command(self, orchestration: OrchestrationClient, deployment_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class FlowRunAction(ExternalDataAction):
    async def flow_run(self, triggered_action: TriggeredAction) -> UUID: ...

class FlowRunStateChangeAction(FlowRunAction):
    @abc.abstractmethod
    async def new_state(self, triggered_action: TriggeredAction) -> StateCreate: ...

    async def act(self, triggered_action: TriggeredAction) -> None: ...

class ChangeFlowRunState(FlowRunStateChangeAction):
    type: ClassVar[str] = 'change-flow-run-state'
    name: Optional[str] = Field(None)
    state: StateType = Field(...)
    message: Optional[str] = Field(None)

    async def new_state(self, triggered_action: TriggeredAction) -> StateCreate: ...

class CancelFlowRun(FlowRunStateChangeAction):
    type: ClassVar[str] = 'cancel-flow-run'

    async def new_state(self, triggered_action: TriggeredAction) -> StateCreate: ...

class SuspendFlowRun(FlowRunStateChangeAction):
    type: ClassVar[str] = 'suspend-flow-run'

    async def new_state(self, triggered_action: TriggeredAction) -> StateCreate: ...

class ResumeFlowRun(FlowRunAction):
    type: ClassVar[str] = 'resume-flow-run'

    async def act(self, triggered_action: TriggeredAction) -> None: ...

class CallWebhook(JinjaTemplateAction):
    type: ClassVar[str] = 'call-webhook'
    block_document_id: UUID = Field(...)
    payload: str = Field(default='')

    @field_validator('payload', mode='before')
    @classmethod
    def ensure_payload_is_a_string(cls, value: Any) -> str: ...

    @field_validator('payload')
    @classmethod
    def validate_payload_templates(cls, value: str) -> str: ...

    async def _get_webhook_block(self, triggered_action: TriggeredAction) -> Webhook: ...

    async def act(self, triggered_action: TriggeredAction) -> None: ...

class SendNotification(JinjaTemplateAction):
    type: ClassVar[str] = 'send-notification'
    block_document_id: UUID = Field(...)
    subject: str = Field('Prefect automated notification')
    body: str = Field(...)

    @field_validator('subject', 'body')
    def is_valid_template(cls, value: str, info: ValidationInfo) -> str: ...

    async def _get_notification_block(self, triggered_action: TriggeredAction) -> NotificationBlock: ...

    async def act(self, triggered_action: TriggeredAction) -> None: ...

    async def render(self, triggered_action: TriggeredAction) -> Tuple[str, str]: ...

class WorkPoolAction(Action):
    source: Literal['selected', 'inferred'] = Field('selected')
    work_pool_id: Optional[UUID] = Field(None)

    @model_validator(mode='after')
    def selected_work_pool_requires_id(self) -> Self: ...

    async def work_pool_id_to_use(self, triggered_action: TriggeredAction) -> UUID: ...

class WorkPoolCommandAction(WorkPoolAction, ExternalDataAction):
    _target_work_pool: Optional[WorkPool] = PrivateAttr(default=None)
    _action_description: str

    async def target_work_pool(self, triggered_action: TriggeredAction) -> WorkPool: ...

    async def act(self, triggered_action: TriggeredAction) -> None: ...

    @abc.abstractmethod
    async def command(self, orchestration: OrchestrationClient, work_pool: WorkPool, triggered_action: TriggeredAction) -> Response: ...

class PauseWorkPool(WorkPoolCommandAction):
    type: ClassVar[str] = 'pause-work-pool'
    _action_description: ClassVar[str] = 'Pausing work pool'

    async def command(self, orchestration: OrchestrationClient, work_pool: WorkPool, triggered_action: TriggeredAction) -> Response: ...

class ResumeWorkPool(WorkPoolCommandAction):
    type: ClassVar[str] = 'resume-work-pool'
    _action_description: ClassVar[str] = 'Resuming work pool'

    async def command(self, orchestration: OrchestrationClient, work_pool: WorkPool, triggered_action: TriggeredAction) -> Response: ...

class WorkQueueAction(Action):
    source: Literal['selected', 'inferred'] = Field('selected')
    work_queue_id: Optional[UUID] = Field(None)

    @model_validator(mode='after')
    def selected_work_queue_requires_id(self) -> Self: ...

    async def work_queue_id_to_use(self, triggered_action: TriggeredAction) -> UUID: ...

class WorkQueueCommandAction(WorkQueueAction, ExternalDataAction):
    _action_description: str

    async def act(self, triggered_action: TriggeredAction) -> None: ...

    @abc.abstractmethod
    async def command(self, orchestration: OrchestrationClient, work_queue_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class PauseWorkQueue(WorkQueueCommandAction):
    type: ClassVar[str] = 'pause-work-queue'
    _action_description: ClassVar[str] = 'Pausing work queue'

    async def command(self, orchestration: OrchestrationClient, work_queue_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class ResumeWorkQueue(WorkQueueCommandAction):
    type: ClassVar[str] = 'resume-work-queue'
    _action_description: ClassVar[str] = 'Resuming work queue'

    async def command(self, orchestration: OrchestrationClient, work_queue_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class AutomationAction(Action):
    source: Literal['selected', 'inferred'] = Field('selected')
    automation_id: Optional[UUID] = Field(None)

    @model_validator(mode='after')
    def selected_automation_requires_id(self) -> Self: ...

    async def automation_id_to_use(self, triggered_action: TriggeredAction) -> UUID: ...

class AutomationCommandAction(AutomationAction, ExternalDataAction):
    _action_description: str

    async def act(self, triggered_action: TriggeredAction) -> None: ...

    @abc.abstractmethod
    async def command(self, events: PrefectServerEventsAPIClient, automation_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class PauseAutomation(AutomationCommandAction):
    type: ClassVar[str] = 'pause-automation'
    _action_description: ClassVar[str] = 'Pausing automation'

    async def command(self, events: PrefectServerEventsAPIClient, automation_id: UUID, triggered_action: TriggeredAction) -> Response: ...

class ResumeAutomation(AutomationCommandAction):
    type: ClassVar[str] = 'resume-automation'
    _action_description: ClassVar[str] = 'Resuming automation'

    async def command(self, events: PrefectServerEventsAPIClient, automation_id: UUID, triggered_action: TriggeredAction) -> Response: ...

ServerActionTypes = Union[
    DoNothing, RunDeployment, PauseDeployment, ResumeDeployment, 
    CancelFlowRun, ChangeFlowRunState, PauseWorkQueue, ResumeWorkQueue, 
    SendNotification, CallWebhook, PauseAutomation, ResumeAutomation, 
    SuspendFlowRun, ResumeFlowRun, PauseWorkPool, ResumeWorkPool
]

_recent_actions: TTLCache[UUID, bool] = TTLCache(maxsize=10000, ttl=3600)

async def record_action_happening(id: UUID) -> None: ...

async def action_has_already_happened(id: UUID) -> bool: ...

@asynccontextmanager
async def consumer() -> AsyncGenerator[MessageHandler, None]:
    async def message_handler(message: Message) -> None: ...
    logger.info('Starting action message handler')
    yield message_handler
