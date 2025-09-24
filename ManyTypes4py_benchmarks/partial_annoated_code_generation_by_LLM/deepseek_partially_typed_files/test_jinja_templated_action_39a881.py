import copy
from datetime import timedelta
from typing import Any, Dict, Generator, List, Literal, Optional
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4
import pendulum
import pytest
from pydantic import Field, ValidationInfo, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from prefect.server.api.clients import OrchestrationClient
from prefect.server.database.orm_models import ORMConcurrencyLimitV2, ORMDeployment, ORMFlow, ORMFlowRun, ORMTaskRun, ORMWorkPool, ORMWorkQueue
from prefect.server.events import actions
from prefect.server.events.schemas.automations import Automation, EventTrigger, Firing, Posture, ReceivedEvent, TriggeredAction, TriggerState
from prefect.server.events.schemas.events import Event, Resource
from prefect.server.models import deployments, flow_runs, flows, task_runs, variables, work_queues
from prefect.server.schemas.actions import VariableCreate, WorkQueueCreate, WorkQueueUpdate
from prefect.server.schemas.core import Deployment, Flow, FlowRun, TaskRun, WorkQueue
from prefect.server.schemas.responses import FlowRunResponse
from prefect.server.schemas.states import State, StateType
from prefect.settings import PREFECT_UI_URL, temporary_settings
from prefect.types import DateTime

@pytest.fixture(autouse=True)
def ui_url() -> Generator[str, None, None]:
    with temporary_settings(set_defaults={PREFECT_UI_URL: 'http://localhost:3000'}):
        yield PREFECT_UI_URL.value()

class DemoAction(actions.JinjaTemplateAction):
    type: Literal['test-action'] = 'test-action'
    template: str = Field()

    @field_validator('template')
    def is_valid_template(cls, value: str, info: ValidationInfo) -> str:
        return actions.JinjaTemplateAction.validate_template(value, info.field_name)

    async def act(self, triggered_action: TriggeredAction) -> None:
        return None

    async def render(self, triggered_action: TriggeredAction) -> List[str]:
        return await self._render([self.template], triggered_action)

@pytest.fixture
async def orchestration_client() -> OrchestrationClient:
    return OrchestrationClient()

@pytest.fixture
async def tell_me_about_the_culprit() -> Automation:
    return Automation(name='If my lilies get nibbled, tell me about it', description='Send an email notification whenever the lillies are nibbled', enabled=True, trigger=EventTrigger(expect={'animal.ingested'}, match_related={'prefect.resource.role': 'meal', 'genus': 'Hemerocallis', 'species': 'fulva'}, posture=Posture.Reactive, threshold=0, within=timedelta(seconds=30)), actions=[actions.DoNothing()])

@pytest.fixture
def woodchonk_triggered(tell_me_about_the_culprit: Automation, woodchonk_nibbled: ReceivedEvent) -> TriggeredAction:
    firing: Firing = Firing(trigger=tell_me_about_the_culprit.trigger, trigger_states={TriggerState.Triggered}, triggered=pendulum.now('UTC'), triggering_labels={'i.am.so': 'triggered'}, triggering_event=woodchonk_nibbled)
    return TriggeredAction(automation=tell_me_about_the_culprit, firing=firing, triggered=firing.triggered, triggering_labels=firing.triggering_labels, triggering_event=firing.triggering_event, action=tell_me_about_the_culprit.actions[0])

@pytest.fixture
async def snap_a_pic(session: AsyncSession) -> ORMFlow:
    flow: ORMFlow = await flows.create_flow(session=session, flow=Flow(name='snap-a-pic'))
    await session.commit()
    return flow

@pytest.fixture
async def take_a_picture(session: AsyncSession, snap_a_pic: ORMFlow) -> ORMFlowRun:
    now: DateTime = pendulum.now('UTC')
    flow_run: ORMFlowRun = await flow_runs.create_flow_run(session=session, flow_run=FlowRun(flow_id=snap_a_pic.id, flow_version='1.0'))
    scheduled_state: State = State(type=StateType.RUNNING, message="It's running!", timestamp=now)
    await flow_runs.set_flow_run_state(session=session, flow_run_id=flow_run.id, state=scheduled_state, force=True)
    await session.commit()
    return flow_run

@pytest.fixture
async def take_a_picture_task(session: AsyncSession, snap_a_pic: ORMFlow, take_a_picture: ORMFlowRun) -> ORMTaskRun:
    now: DateTime = pendulum.now('UTC')
    task_run: ORMTaskRun = await task_runs.create_task_run(session=session, task_run=TaskRun(flow_run_id=take_a_picture.id, name='the-task-run', task_key='task-123', dynamic_key='a'))
    assert task_run
    scheduled_state: State = State(type=StateType.RUNNING, message="It's running!", timestamp=now)
    await task_runs.set_task_run_state(session=session, task_run_id=task_run.id, state=scheduled_state, force=True)
    await session.commit()
    return task_run

@pytest.fixture
async def take_a_picture_deployment(session: AsyncSession, take_a_picture: ORMFlowRun) -> ORMDeployment:
    deployment: ORMDeployment = await deployments.create_deployment(session=session, deployment=Deployment(name='Take a picture on demand', flow_id=take_a_picture.flow_id, paused=False))
    await session.commit()
    return deployment

@pytest.fixture
async def take_a_picture_work_queue(session: AsyncSession) -> ORMWorkQueue:
    work_queue: ORMWorkQueue = await work_queues.create_work_queue(session=session, work_queue=WorkQueueCreate(name='camera-queue'))
    await session.commit()
    return work_queue

@pytest.fixture
def picture_taken(start_of_test: DateTime, take_a_picture: ORMFlowRun, take_a_picture_deployment: ORMDeployment, take_a_picture_work_queue: ORMWorkQueue) -> ReceivedEvent:
    return ReceivedEvent(occurred=start_of_test + timedelta(microseconds=2), event='prefect.flow-run.completed', resource={'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}', 'prefect.state-message': 'All states completed.', 'prefect.state-name': 'Completed', 'prefect.state-timestamp': pendulum.now('UTC').isoformat(), 'prefect.state-type': 'COMPLETED'}, related=[{'prefect.resource.id': f'prefect.flow.{take_a_picture.flow_id}', 'prefect.resource.role': 'flow'}, {'prefect.resource.id': f'prefect.deployment.{take_a_picture_deployment.id}', 'prefect.resource.role': 'deployment'}, {'prefect.resource.id': f'prefect.work-queue.{take_a_picture_work_queue.id}', 'prefect.resource.role': 'work-queue'}], id=uuid4())

@pytest.fixture
def picture_taken_by_task(start_of_test: DateTime, take_a_picture: ORMFlowRun, take_a_picture_task: ORMTaskRun, take_a_picture_deployment: ORMDeployment, take_a_picture_work_queue: ORMWorkQueue) -> ReceivedEvent:
    return ReceivedEvent(occurred=start_of_test + timedelta(microseconds=2), event='prefect.task-run.completed', resource={'prefect.resource.id': f'prefect.task-run.{take_a_picture_task.id}', 'prefect.state-message': 'All states completed.', 'prefect.state-name': 'Completed', 'prefect.state-timestamp': pendulum.now('UTC').isoformat(), 'prefect.state-type': 'COMPLETED'}, related=[{'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}', 'prefect.resource.role': 'flow'}, {'prefect.resource.id': f'prefect.flow.{take_a_picture.flow_id}', 'prefect.resource.role': 'flow'}, {'prefect.resource.id': f'prefect.deployment.{take_a_picture_deployment.id}', 'prefect.resource.role': 'deployment'}, {'prefect.resource.id': f'prefect.work-queue.{take_a_picture_work_queue.id}', 'prefect.resource.role': 'work-queue'}], id=uuid4())

@pytest.fixture
def took_a_picture(tell_me_about_the_culprit: Automation, picture_taken: ReceivedEvent) -> TriggeredAction:
    firing: Firing = Firing(trigger=tell_me_about_the_culprit.trigger, trigger_states={TriggerState.Triggered}, triggered=pendulum.now('UTC'), triggering_labels={}, triggering_event=picture_taken)
    return TriggeredAction(automation=tell_me_about_the_culprit, firing=firing, triggered=firing.triggered, triggering_labels=firing.triggering_labels, triggering_event=firing.triggering_event, action=tell_me_about_the_culprit.actions[0])

@pytest.fixture
def took_a_picture_by_task(tell_me_about_the_culprit: Automation, picture_taken_by_task: ReceivedEvent) -> TriggeredAction:
    firing: Firing = Firing(trigger=tell_me_about_the_culprit.trigger, trigger_states={TriggerState.Triggered}, triggered=pendulum.now('UTC'), triggering_labels={}, triggering_event=picture_taken_by_task)
    return TriggeredAction(automation=tell_me_about_the_culprit, firing=firing, triggered=firing.triggered, triggering_labels=firing.triggering_labels, triggering_event=firing.triggering_event, action=tell_me_about_the_culprit.actions[0])

async def test_filters_are_available_to_templates(took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ automation|ui_url }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert f'/automations/automation/{took_a_picture.automation.id}' in rendered[0]

async def test_flow_is_available_to_templates(snap_a_pic: ORMFlow, took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ flow.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == snap_a_pic.name

async def test_flow_run_is_available_to_templates(take_a_picture: ORMFlowRun, took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ flow_run.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == take_a_picture.name

async def test_flow_run_state_comes_from_event_resource(took_a_picture: TriggeredAction) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[ReceivedEvent] = took_a_picture.triggering_event
    assert event
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp.isoformat() }} {{ flow_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == ' '.join([event.resource['prefect.state-message'], event.resource['prefect.state-name'], event.resource['prefect.state-timestamp'], event.resource['prefect.state-type']])

async def test_flow_run_state_event_missing_state_data_uses_api_state(session: AsyncSession, took_a_picture: TriggeredAction, take_a_picture: ORMFlowRun) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[ReceivedEvent] = took_a_picture.triggering_event
    assert event
    event.resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}'})
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type }}')
    rendered: List[str] = await action.render(took_a_picture)
    flow_run: Optional[FlowRunResponse] = await flow_runs.read_flow_run(session, take_a_picture.id)
    assert flow_run is not None
    assert len(rendered) == 1
    assert rendered[0] == ' '.join([flow_run.state.message, flow_run.state.name, str(flow_run.state.timestamp), str(flow_run.state.type)])

async def test_flow_run_state_event_malformed_uses_api_state(session: AsyncSession, took_a_picture: TriggeredAction, take_a_picture: ORMFlowRun) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[ReceivedEvent] = took_a_picture.triggering_event
    assert event
    assert take_a_picture.state
    event.resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}', 'prefect.state-message': '', 'prefect.state-name': 'Buh-bye', 'prefect.state-timestamp': take_a_picture.state.timestamp.isoformat(), 'prefect.state-type': 'ANNIHILATED'})
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type }}')
    rendered: List[str] = await action.render(took_a_picture)
    flow_run: Optional[FlowRunResponse] = await flow_runs.read_flow_run(session, take_a_picture.id)
    assert flow_run is not None
    assert len(rendered) == 1
    assert rendered[0] == ' '.join([flow_run.state.message, flow_run.state.name, str(flow_run.state.timestamp), str(flow_run.state.type)])

async def test_flow_run_state_comes_from_event_resource_empty_message(took_a_picture: TriggeredAction, take_a_picture: ORMFlowRun) -> None:
    """Regression test for https://github.com/PrefectHQ/prefect/issues/9230
    where the flow run event had all of the correct state information but the
    message, as is the case in many events, was empty causing the state to not
    be rehydrated but instead use the state from the API"""
    event: Optional[ReceivedEvent] = took_a_picture.triggering_event
    assert event
    assert take_a_picture.state
    event.resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}', 'prefect.state-message': '', 'prefect.state-name': 'Pending', 'prefect.state-timestamp': take_a_picture.state.timestamp.isoformat(), 'prefect.state-type': 'PENDING'})
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == ' '.join(['', 'Pending', str(take_a_picture.state.timestamp), 'PENDING'])

async def test_task_run_is_available_to_templates(take_a_picture_task: ORMTaskRun, took_a_picture_by_task: TriggeredAction) -> None:
    assert took_a_picture_by_task.triggering_event
    action: DemoAction = DemoAction(template='{{ task_run.name }}')
    rendered: List[str] = await action.render(took_a_picture_by_task)
    assert len(rendered) == 1
    assert rendered[0] == take_a_picture_task.name

async def test_task_run_state_comes_from_event_resource(took_a_picture_by_task: TriggeredAction) -> None:
    event: Optional[ReceivedEvent] = took_a_picture_by_task.triggering_event
    assert event
    action: DemoAction = DemoAction(template='{{ task_run.state.message }} {{ task_run.state.name }} {{ task_run.state.timestamp.isoformat() }} {{ task_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture_by_task)
    assert len(rendered) == 1
    assert rendered[0] == ' '.join([event.resource['prefect.state-message'], event.resource['prefect.state-name'], event.resource['prefect.state-timestamp'], event.resource['prefect.state-type']])

async def test_deployment_is_available_to_templates(take_a_picture_deployment: ORMDeployment, took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='