#!/usr/bin/env python3
import copy
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Union
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
    type: str = 'test-action'
    template: str = Field()

    @field_validator('template')
    def is_valid_template(cls, value: str, info: ValidationInfo) -> str:
        return actions.JinjaTemplateAction.validate_template(value, info.field_name)

    async def act(self, triggered_action: TriggeredAction) -> Any:
        return None

    async def render(self, triggered_action: TriggeredAction) -> List[str]:
        return await self._render([self.template], triggered_action)


@pytest.fixture
async def orchestration_client() -> OrchestrationClient:
    return OrchestrationClient()


@pytest.fixture
async def tell_me_about_the_culprit() -> Automation:
    return Automation(
        name='If my lilies get nibbled, tell me about it',
        description='Send an email notification whenever the lillies are nibbled',
        enabled=True,
        trigger=EventTrigger(
            expect={'animal.ingested'},
            match_related={'prefect.resource.role': 'meal', 'genus': 'Hemerocallis', 'species': 'fulva'},
            posture=Posture.Reactive,
            threshold=0,
            within=timedelta(seconds=30)
        ),
        actions=[actions.DoNothing()]
    )


@pytest.fixture
def woodchonk_triggered(
    tell_me_about_the_culprit: Automation, woodchonk_nibbled: Any
) -> TriggeredAction:
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=pendulum.now('UTC'),
        triggering_labels={'i.am.so': 'triggered'},
        triggering_event=woodchonk_nibbled
    )
    return TriggeredAction(
        automation=tell_me_about_the_culprit,
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=tell_me_about_the_culprit.actions[0]
    )


@pytest.fixture
async def snap_a_pic(session: AsyncSession) -> Flow:
    flow: Flow = await flows.create_flow(session=session, flow=Flow(name='snap-a-pic'))
    await session.commit()
    return flow


@pytest.fixture
async def take_a_picture(session: AsyncSession, snap_a_pic: Flow) -> FlowRun:
    now: DateTime = pendulum.now('UTC')
    flow_run: FlowRun = await flow_runs.create_flow_run(
        session=session,
        flow_run=FlowRun(flow_id=snap_a_pic.id, flow_version='1.0')
    )
    scheduled_state: State = State(
        type=StateType.RUNNING,
        message="It's running!",
        timestamp=now
    )
    await flow_runs.set_flow_run_state(session=session, flow_run_id=flow_run.id, state=scheduled_state, force=True)
    await session.commit()
    return flow_run


@pytest.fixture
async def take_a_picture_task(session: AsyncSession, snap_a_pic: Flow, take_a_picture: FlowRun) -> TaskRun:
    now: DateTime = pendulum.now('UTC')
    task_run: TaskRun = await task_runs.create_task_run(
        session=session,
        task_run=TaskRun(flow_run_id=take_a_picture.id, name='the-task-run', task_key='task-123', dynamic_key='a')
    )
    assert task_run
    scheduled_state: State = State(
        type=StateType.RUNNING,
        message="It's running!",
        timestamp=now
    )
    await task_runs.set_task_run_state(session=session, task_run_id=task_run.id, state=scheduled_state, force=True)
    await session.commit()
    return task_run


@pytest.fixture
async def take_a_picture_deployment(session: AsyncSession, take_a_picture: FlowRun) -> Deployment:
    deployment: Deployment = await deployments.create_deployment(
        session=session,
        deployment=Deployment(name='Take a picture on demand', flow_id=take_a_picture.flow_id, paused=False)
    )
    await session.commit()
    return deployment


@pytest.fixture
async def take_a_picture_work_queue(session: AsyncSession) -> WorkQueue:
    work_queue: WorkQueue = await work_queues.create_work_queue(
        session=session,
        work_queue=WorkQueueCreate(name='camera-queue')
    )
    await session.commit()
    return work_queue


@pytest.fixture
def picture_taken(
    start_of_test: DateTime,
    take_a_picture: FlowRun,
    take_a_picture_deployment: Deployment,
    take_a_picture_work_queue: WorkQueue
) -> ReceivedEvent:
    return ReceivedEvent(
        occurred=start_of_test + timedelta(microseconds=2),
        event='prefect.flow-run.completed',
        resource={
            'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}',
            'prefect.state-message': 'All states completed.',
            'prefect.state-name': 'Completed',
            'prefect.state-timestamp': pendulum.now('UTC').isoformat(),
            'prefect.state-type': 'COMPLETED'
        },
        related=[
            {'prefect.resource.id': f'prefect.flow.{take_a_picture.flow_id}', 'prefect.resource.role': 'flow'},
            {'prefect.resource.id': f'prefect.deployment.{take_a_picture_deployment.id}', 'prefect.resource.role': 'deployment'},
            {'prefect.resource.id': f'prefect.work-queue.{take_a_picture_work_queue.id}', 'prefect.resource.role': 'work-queue'},
        ],
        id=uuid4()
    )


@pytest.fixture
def picture_taken_by_task(
    start_of_test: DateTime,
    take_a_picture: FlowRun,
    take_a_picture_task: TaskRun,
    take_a_picture_deployment: Deployment,
    take_a_picture_work_queue: WorkQueue
) -> ReceivedEvent:
    return ReceivedEvent(
        occurred=start_of_test + timedelta(microseconds=2),
        event='prefect.task-run.completed',
        resource={
            'prefect.resource.id': f'prefect.task-run.{take_a_picture_task.id}',
            'prefect.state-message': 'All states completed.',
            'prefect.state-name': 'Completed',
            'prefect.state-timestamp': pendulum.now('UTC').isoformat(),
            'prefect.state-type': 'COMPLETED'
        },
        related=[
            {'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}', 'prefect.resource.role': 'flow'},
            {'prefect.resource.id': f'prefect.flow.{take_a_picture.flow_id}', 'prefect.resource.role': 'flow'},
            {'prefect.resource.id': f'prefect.deployment.{take_a_picture_deployment.id}', 'prefect.resource.role': 'deployment'},
            {'prefect.resource.id': f'prefect.work-queue.{take_a_picture_work_queue.id}', 'prefect.resource.role': 'work-queue'},
        ],
        id=uuid4()
    )


@pytest.fixture
def took_a_picture(
    tell_me_about_the_culprit: Automation, picture_taken: ReceivedEvent
) -> TriggeredAction:
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=pendulum.now('UTC'),
        triggering_labels={},
        triggering_event=picture_taken
    )
    return TriggeredAction(
        automation=tell_me_about_the_culprit,
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=tell_me_about_the_culprit.actions[0]
    )


@pytest.fixture
def took_a_picture_by_task(
    tell_me_about_the_culprit: Automation, picture_taken_by_task: ReceivedEvent
) -> TriggeredAction:
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=pendulum.now('UTC'),
        triggering_labels={},
        triggering_event=picture_taken_by_task
    )
    return TriggeredAction(
        automation=tell_me_about_the_culprit,
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=tell_me_about_the_culprit.actions[0]
    )


async def test_filters_are_available_to_templates(took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ automation|ui_url }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert f'/automations/automation/{took_a_picture.automation.id}' in rendered[0]


async def test_flow_is_available_to_templates(snap_a_pic: Flow, took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ flow.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == snap_a_pic.name


async def test_flow_run_is_available_to_templates(take_a_picture: FlowRun, took_a_picture: TriggeredAction) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ flow_run.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    # Assuming that flow_run name is set to the flow run id if not explicitly set.
    assert rendered[0] == take_a_picture.name  # type: ignore


async def test_flow_run_state_comes_from_event_resource(took_a_picture: TriggeredAction) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[Event] = took_a_picture.triggering_event
    assert event
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp.isoformat() }} {{ flow_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture)
    state_text: str = ' '.join([
        event.resource['prefect.state-message'],
        event.resource['prefect.state-name'],
        event.resource['prefect.state-timestamp'],
        event.resource['prefect.state-type']
    ])
    assert len(rendered) == 1
    assert rendered[0] == state_text


async def test_flow_run_state_event_missing_state_data_uses_api_state(
    session: AsyncSession, took_a_picture: TriggeredAction, take_a_picture: FlowRun
) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[Event] = took_a_picture.triggering_event
    assert event
    event.resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}'})
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type }}')
    rendered: List[str] = await action.render(took_a_picture)
    flow_run: FlowRun = await flow_runs.read_flow_run(session, take_a_picture.id)
    expected_text: str = ' '.join([
        flow_run.state.message,
        flow_run.state.name,
        str(flow_run.state.timestamp),
        str(flow_run.state.type)
    ])
    assert len(rendered) == 1
    assert rendered[0] == expected_text


async def test_flow_run_state_event_malformed_uses_api_state(
    session: AsyncSession, took_a_picture: TriggeredAction, take_a_picture: FlowRun
) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3310,
    where the state of the flow run fetched from the API was updated after the
    state change that caused the automation to run and notifications that were
    sent contained the incorrect state information"""
    event: Optional[Event] = took_a_picture.triggering_event
    assert event
    assert take_a_picture.state
    event.resource = Resource.model_validate({
        'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}',
        'prefect.state-message': '',
        'prefect.state-name': 'Buh-bye',
        'prefect.state-timestamp': take_a_picture.state.timestamp.isoformat(),
        'prefect.state-type': 'ANNIHILATED'
    })
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type }}')
    rendered: List[str] = await action.render(took_a_picture)
    flow_run: FlowRun = await flow_runs.read_flow_run(session, take_a_picture.id)
    expected_text: str = ' '.join([
        flow_run.state.message,
        flow_run.state.name,
        str(flow_run.state.timestamp),
        str(flow_run.state.type)
    ])
    assert len(rendered) == 1
    assert rendered[0] == expected_text


async def test_flow_run_state_comes_from_event_resource_empty_message(
    took_a_picture: TriggeredAction, take_a_picture: FlowRun
) -> None:
    """Regression test for https://github.com/PrefectHQ/prefect/issues/9230
    where the flow run event had all of the correct state information but the
    message, as is the case in many events, was empty causing the state to not
    be rehydrated but instead use the state from the API"""
    event: Optional[Event] = took_a_picture.triggering_event
    assert event
    assert take_a_picture.state
    event.resource = Resource.model_validate({
        'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}',
        'prefect.state-message': '',
        'prefect.state-name': 'Pending',
        'prefect.state-timestamp': take_a_picture.state.timestamp.isoformat(),
        'prefect.state-type': 'PENDING'
    })
    action: DemoAction = DemoAction(template='{{ flow_run.state.message }} {{ flow_run.state.name }} {{ flow_run.state.timestamp }} {{ flow_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture)
    expected_text: str = ' '.join(['', 'Pending', str(take_a_picture.state.timestamp), 'PENDING'])
    assert len(rendered) == 1
    assert rendered[0] == expected_text


async def test_task_run_is_available_to_templates(
    take_a_picture_task: TaskRun, took_a_picture_by_task: TriggeredAction
) -> None:
    assert took_a_picture_by_task.triggering_event
    action: DemoAction = DemoAction(template='{{ task_run.name }}')
    rendered: List[str] = await action.render(took_a_picture_by_task)
    assert len(rendered) == 1
    assert rendered[0] == take_a_picture_task.name


async def test_task_run_state_comes_from_event_resource(took_a_picture_by_task: TriggeredAction) -> None:
    event: Optional[Event] = took_a_picture_by_task.triggering_event
    assert event
    action: DemoAction = DemoAction(template='{{ task_run.state.message }} {{ task_run.state.name }} {{ task_run.state.timestamp.isoformat() }} {{ task_run.state.type.value }}')
    rendered: List[str] = await action.render(took_a_picture_by_task)
    expected_text: str = ' '.join([
        event.resource['prefect.state-message'],
        event.resource['prefect.state-name'],
        event.resource['prefect.state-timestamp'],
        event.resource['prefect.state-type']
    ])
    assert len(rendered) == 1
    assert rendered[0] == expected_text


async def test_deployment_is_available_to_templates(
    take_a_picture_deployment: Deployment, took_a_picture: TriggeredAction
) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ deployment.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == take_a_picture_deployment.name


async def test_work_queue_is_available_to_templates(
    take_a_picture_work_queue: WorkQueue, took_a_picture: TriggeredAction
) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ work_queue.name }}')
    rendered: List[str] = await action.render(took_a_picture)
    assert len(rendered) == 1
    assert rendered[0] == take_a_picture_work_queue.name


async def test_all_objects_in_template(
    took_a_picture: TriggeredAction,
    snap_a_pic: Flow,
    take_a_picture_deployment: Deployment,
    take_a_picture_work_queue: WorkQueue,
    take_a_picture: FlowRun  # shadowing name intentionally for clarity
) -> None:
    assert took_a_picture.triggering_event
    action: DemoAction = DemoAction(template='{{ flow.id }} {{ flow_run.id }} {{ deployment.id }} {{ work_queue.id }}')
    rendered: List[str] = await action.render(took_a_picture)
    expected_text: str = ' '.join([str(snap_a_pic.id), str(take_a_picture.id), str(take_a_picture_deployment.id), str(take_a_picture_work_queue.id)])
    assert len(rendered) == 1
    assert rendered[0] == expected_text


async def test_caches_objects(took_a_picture: TriggeredAction, snap_a_pic: Flow) -> None:
    assert took_a_picture.triggering_event
    mock_get_object: AsyncMock = AsyncMock()
    mock_get_object.side_effect = [snap_a_pic]
    with patch('prefect.server.events.actions.JinjaTemplateAction._get_object_from_prefect_api', mock_get_object):
        action: DemoAction = DemoAction(template='{{ flow.id }}')
        await action.render(took_a_picture)
        assert mock_get_object.call_count == 1
        mock_get_object.reset_mock()
        await action.render(took_a_picture)
        assert mock_get_object.call_count == 0


async def test_retrieves_after_caching(
    took_a_picture: TriggeredAction,
    snap_a_pic: Flow,
    take_a_picture: FlowRun,
    take_a_picture_deployment: Deployment,
    take_a_picture_work_queue: WorkQueue
) -> None:
    assert took_a_picture.triggering_event

    def object_retriever(oriont_client: Any, triggered_action: TriggeredAction, resource: Resource) -> Any:
        if 'prefect.flow.' in resource.id:
            return snap_a_pic
        elif 'prefect.flow-run' in resource.id:
            return take_a_picture
        elif 'prefect.deployment' in resource.id:
            return take_a_picture_deployment
        elif 'prefect.work-queue' in resource.id:
            return take_a_picture_work_queue

    mock_get_object: AsyncMock = AsyncMock()
    mock_get_object.side_effect = object_retriever
    with patch('prefect.server.events.actions.JinjaTemplateAction._get_object_from_prefect_api', mock_get_object):
        action: DemoAction = DemoAction(template='')
        rendered: List[str] = await action._render(['{{ flow.id }} {{ flow_run.id}}'], took_a_picture)
        assert mock_get_object.call_count == 2
        assert rendered[0] == f'{snap_a_pic.id} {take_a_picture.id}'
        mock_get_object.reset_mock()
        rendered = await action._render(['{{ flow.id }} {{ flow_run.id }} {{ deployment.id }} {{ work_queue.id }}'], took_a_picture)
        assert mock_get_object.call_count == 2
        expected_text: str = f'{snap_a_pic.id} {take_a_picture.id} {take_a_picture_deployment.id} {take_a_picture_work_queue.id}'
        assert rendered[0] == expected_text
        mock_get_object.reset_mock()
        rendered = await action._render(['{{ flow.id }} {{ flow_run.id }} {{ deployment.id }} {{ work_queue.id }}'], took_a_picture)
        assert mock_get_object.call_count == 0
        assert rendered[0] == expected_text


async def test_lazy_objects_are_none_without_a_resource(woodchonk_triggered: TriggeredAction) -> None:
    assert woodchonk_triggered.triggering_event
    action: DemoAction = DemoAction(template='{{ flow_run.name }}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == ''


async def test_get_object_from_orion_null_resource(
    orchestration_client: OrchestrationClient, woodchonk_triggered: TriggeredAction
) -> None:
    resource: Optional[Resource] = None
    action: DemoAction = DemoAction(template='')
    result = await action._get_object_from_prefect_api(orchestration_client, woodchonk_triggered, resource)
    assert result is None


async def test_get_object_from_orion_resource_with_invalid_uuid(
    orchestration_client: OrchestrationClient, woodchonk_triggered: TriggeredAction
) -> None:
    resource: Resource = Resource.model_validate({'prefect.resource.id': 'prefect.flow-run.thing'})
    action: DemoAction = DemoAction(template='')
    result = await action._get_object_from_prefect_api(orchestration_client, woodchonk_triggered, resource)
    assert result is None


async def test_get_object_from_orion_resource_with_unknown_kind(
    orchestration_client: OrchestrationClient, woodchonk_triggered: TriggeredAction
) -> None:
    resource: Resource = Resource.model_validate({'prefect.resource.id': 'prefect.unknown.thing'})
    action: DemoAction = DemoAction(template='')
    result = await action._get_object_from_prefect_api(orchestration_client, woodchonk_triggered, resource)
    assert result is None


async def test_get_object_from_orion_resource_missing_from_api(
    orchestration_client: OrchestrationClient, woodchonk_triggered: TriggeredAction
) -> None:
    resource: Resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{uuid4()}'})
    action: DemoAction = DemoAction(template='')
    result = await action._get_object_from_prefect_api(orchestration_client, woodchonk_triggered, resource)
    assert result is None


async def test_get_object_returns_object(
    orchestration_client: OrchestrationClient, take_a_picture: FlowRun, woodchonk_triggered: TriggeredAction
) -> None:
    resource: Resource = Resource.model_validate({'prefect.resource.id': f'prefect.flow-run.{take_a_picture.id}'})
    action: DemoAction = DemoAction(template='')
    flow_run_obj: Any = await action._get_object_from_prefect_api(orchestration_client, woodchonk_triggered, resource)
    assert isinstance(flow_run_obj, FlowRunResponse)
    assert flow_run_obj.id == take_a_picture.id
    assert flow_run_obj.name == take_a_picture.name


async def test_triggering_labels_may_be_accessed_as_object(woodchonk_triggered: TriggeredAction) -> None:
    assert woodchonk_triggered.triggering_labels == {'i.am.so': 'triggered'}
    action: DemoAction = DemoAction(template='{{ labels.i.am.so }}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == 'triggered'


async def test_triggering_labels_may_be_accessed_as_dict(woodchonk_triggered: TriggeredAction) -> None:
    assert woodchonk_triggered.triggering_labels == {'i.am.so': 'triggered'}
    action: DemoAction = DemoAction(template="{{ labels['i.am.so'] }}")
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == 'triggered'


async def test_resource_labels_may_be_accessed_as_object(woodchonk_triggered: TriggeredAction) -> None:
    assert woodchonk_triggered.triggering_event
    assert woodchonk_triggered.triggering_event.resource.id == 'woodchonk'
    assert woodchonk_triggered.triggering_event.resource['kingdom'] == 'Animalia'
    action: DemoAction = DemoAction(template='{{ event.resource.labels.prefect.resource.id }} - {{ event.resource.labels.kingdom }}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == 'woodchonk - Animalia'


async def test_resource_labels_may_be_iterated(woodchonk_triggered: TriggeredAction) -> None:
    action: DemoAction = DemoAction(template='{% for label, value in event.resource.labels %}{{label}} - {{value}}\n{% endfor %}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    expected_labels = {
        'kingdom - Animalia',
        'phylum - Chordata',
        'class - Mammalia',
        'order - Rodentia',
        'family - Sciuridae',
        'genus - Marmota',
        'species - monax',
        'prefect.resource.id - woodchonk'
    }
    assert set(rendered[0].splitlines()) == expected_labels


async def test_resource_labels_may_be_sorted(woodchonk_triggered: TriggeredAction) -> None:
    action: DemoAction = DemoAction(template='{% for label, value in event.resource.labels|sort %}{{label}} - {{value}}\n{% endfor %}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0].splitlines() == [
        'class - Mammalia',
        'family - Sciuridae',
        'genus - Marmota',
        'kingdom - Animalia',
        'order - Rodentia',
        'phylum - Chordata',
        'prefect.resource.id - woodchonk',
        'species - monax'
    ]


async def test_related_labels_may_be_accessed_as_object(woodchonk_triggered: TriggeredAction) -> None:
    assert woodchonk_triggered.triggering_event
    assert len(woodchonk_triggered.triggering_event.related) == 2
    first, second = woodchonk_triggered.triggering_event.related
    assert first.role == 'meal'
    assert first['genus'] == 'Hemerocallis'
    assert second.role == 'meal'
    assert second['genus'] == 'Tulipa'
    action: DemoAction = DemoAction(template='{% for related in event.related %}{{ related.labels.prefect.resource.role }} - {{ related.labels.genus }}, {% endfor %}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == 'meal - Hemerocallis, meal - Tulipa, '


@pytest.fixture
async def workspace_variables(session: AsyncSession) -> Dict[str, Any]:
    to_create: Dict[str, Any] = {'hello': 'world!', 'goodbye': 'moon!'}
    for name, value in to_create.items():
        await variables.create_variable(session, VariableCreate(name=name, value=value))
    await session.commit()
    return to_create


async def test_workspace_variables_may_be_accessed_as_an_object(
    workspace_variables: Dict[str, Any], woodchonk_triggered: TriggeredAction
) -> None:
    action: DemoAction = DemoAction(template='{{ variables.hello }} {{ variables.goodbye }}')
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert rendered[0] == 'world! moon!'


async def test_workspace_variables_may_be_accessed_as_a_dict(
    workspace_variables: Dict[str, Any], woodchonk_triggered: TriggeredAction
) -> None:
    action: DemoAction = DemoAction(template="{{ variables['hello'] }}")
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert rendered[0] == 'world!'


@pytest.mark.parametrize('value', ['string-value', '"string-value"', 123, 12.3, True, False, None, {'key': 'value'}, ['value1', 'value2'], {'key': ['value1', 'value2']}])
async def test_json_workspace_variables(
    session: AsyncSession, woodchonk_triggered: TriggeredAction, value: Union[str, int, float, bool, None, Dict[Any, Any], List[Any]]
) -> None:
    await variables.create_variable(session, VariableCreate(name='my_var', value=value))
    await session.commit()
    action: DemoAction = DemoAction(template="{{ variables['my_var'] }} {{ variables.my_var }}")
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert rendered[0] == f'{value} {value}'


async def test_environment_is_immutable(woodchonk_triggered: TriggeredAction) -> None:
    template: str = '{{ event.related.append(42) }}'
    before: Any = copy.deepcopy(woodchonk_triggered)
    action: DemoAction = DemoAction(template=template)
    rendered: List[str] = await action.render(woodchonk_triggered)
    expected_error: str = f"""Failed to render template due to the following error: SecurityError("access to attribute 'append' of 'list' object is unsafe.")\nTemplate source:\n{template}"""
    assert len(rendered) == 1
    assert rendered[0] == expected_error
    assert woodchonk_triggered == before


async def test_ranges_are_limited(woodchonk_triggered: TriggeredAction) -> None:
    template: str = '{% for i in range(101) %}{{ i }} {% endfor %}'
    action: DemoAction = DemoAction(template=template)
    rendered: List[str] = await action.render(woodchonk_triggered)
    expected_error: str = f"Failed to render template due to the following error: OverflowError('Range too big. The sandbox blocks ranges larger than MAX_RANGE (100).')\nTemplate source:\n{template}"
    assert len(rendered) == 1
    assert rendered[0] == expected_error


@pytest.mark.parametrize('template', ["{% extends 'base.html' %}", "{% import 'anything.html' as forms %}"])
async def test_loading_templates_is_disabled(woodchonk_triggered: TriggeredAction, template: str) -> None:
    action: DemoAction = DemoAction(template=template)
    rendered: List[str] = await action.render(woodchonk_triggered)
    expected_error: str = f"Failed to render template due to the following error: TypeError('no loader for this environment specified')\nTemplate source:\n{template}"
    assert len(rendered) == 1
    assert rendered[0] == expected_error


@pytest.mark.parametrize('template', ['{{ event.__type__ }}', "{{ event|attr('__type__') }}"])
async def test_unsafe_attribute_is_verboten(woodchonk_triggered: TriggeredAction, template: str) -> None:
    full_template: str = 'I did not get this attribute: (' + template + '), no, Sir, I did not'
    action: DemoAction = DemoAction(template=full_template)
    rendered: List[str] = await action.render(woodchonk_triggered)
    assert len(rendered) == 1
    assert rendered[0] == 'I did not get this attribute: (), no, Sir, I did not'


async def test_deeply_nested_for_loops_prohibited() -> None:
    template: str = '{% for i in range(100) %}{% for j in range(100) %}{% for l in range(100) %}{% endfor %}{% endfor %}{% endfor %}'
    with pytest.raises(ValueError, match="'template' is not a valid template: Contains nested for loops at a depth of 3. Templates can nest for loops no more than 2 loops deep."):
        DemoAction(template=template)


async def test_many_loops_prohibited() -> None:
    template: str = (
        '{% for a in range(100) %}{% endfor %}'
        '{% for b in range(100) %}{% endfor %}'
        '{% for c in range(100) %}{% endfor %}'
        '{% for d in range(100) %}{% endfor %}'
        '{% for e in range(100) %}{% endfor %}'
        '{% for f in range(100) %}{% endfor %}'
        '{% for g in range(100) %}{% endfor %}'
        '{% for h in range(100) %}{% endfor %}'
        '{% for i in range(100) %}{% endfor %}'
        '{% for j in range(100) %}{% endfor %}'
        '{% for k in range(100) %}{% endfor %}'
    )
    with pytest.raises(ValueError, match="'template' is not a valid template: Contains 11 for loops. Templates can contain no more than 10 for loops."):
        DemoAction(template=template)


async def test_complex_valid_template() -> None:
    template: str = '''
    {% if flow_run %}
        {% set n = 10 %}
        {% for i in range(n) %}
            {% if i % 2 == 0 %}
                {% for j in range(n * 2) %}
                    {{ j }}
                {% endfor %}
            {% else %}
                {% for k in range(n * 3) %}
                    {{ k }}
                {% endfor %}
            {% endif %}
        {% endfor %}
    {% else %}
        {% for l in range(100) %}
            {{ l }}
        {% endfor %}
    {% endif %}
    '''
    # Should not raise
    DemoAction(template=template)


async def test_complex_invalid_template() -> None:
    template: str = '''
    {% if flow_run %}
        {% set n = 10 %}
        {% for i in range(n) %}
            {% if i % 2 == 0 %}
                {% for j in range(n * 2) %}
                    {% for k in range(n * 2) %}
                        {{ k }}
                    {% endfor %}
                {% endfor %}
            {% else %}
                {% for l in range(n * 3) %}
                    {{ l }}
                {% endfor %}
            {% endif %}
        {% endfor %}
    {% else %}
        {% for m in range(100) %}
            {{ m }}
        {% endfor %}
    {% endif %}
    '''
    with pytest.raises(ValueError, match="'template' is not a valid template: Contains nested for loops at a depth of 3. Templates can nest for loops no more than 2 loops deep."):
        DemoAction(template=template)


async def test_work_queue_health_late_run_count(
    session: AsyncSession,
    tell_me_about_the_culprit: Automation,
    take_a_picture_work_queue: WorkQueue,
    start_of_test: DateTime,
    monkeypatch: Any
) -> None:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/3635, where
    the late run count was not rendering in our work queue health template
    """
    await work_queues.update_work_queue(
        session,
        take_a_picture_work_queue.id,
        work_queue=WorkQueueUpdate(last_polled='2023-01-02T00:00:00.000000+00:00')
    )
    await session.commit()
    monkeypatch.setattr('prefect.server.models.flow_runs.count_flow_runs', AsyncMock(return_value=42))
    template: str = '''
    Name: {{ work_queue.name }}
    Last polled: {{ work_queue.last_polled.isoformat() }}
    Late run count: {{ work_queue.late_runs_count }}
    URL: {{ work_queue|ui_url }}
    '''
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=start_of_test + timedelta(seconds=1),
        triggering_event=Event(
            occurred=start_of_test,
            event='prefect.work-queue.healthy',
            resource={'prefect.resource.id': f'prefect.work-queue.{take_a_picture_work_queue.id}'},
            id=uuid4()
        ).receive(),
        triggering_labels={}
    )
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    rendered: str = rendered_list[0]
    assert '\n    Name: camera-queue\n    Last polled: 2023-01-02T00:00:00+00:00\n    Late run count: 42\n    ' in rendered
    assert 'URL:' in rendered
    assert f'work-queues/work-queue/{take_a_picture_work_queue.id}' in rendered


async def test_related_resource_by_role_in_templates(
    work_pool: ORMWorkPool,
    work_queue: ORMWorkQueue,
    tell_me_about_the_culprit: Automation,
    start_of_test: DateTime
) -> None:
    """
    Regression test for https://github.com/PrefectHQ/nebula/issues/5215, where we want
    to add {{ work_pool }} as a template shortcut.  This is an alternative and more
    general approach that can operate on just the event.
    """
    template: str = '''
    Name: {{ event.resource.name }}
    Pool: {{ event.resource_in_role['work-pool'].name }}
    Pools: {% for resource in event.resources_in_role['work-pool'] %}{{ resource.name }},{% endfor %}
    '''
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=start_of_test + timedelta(seconds=1),
        triggering_event=Event(
            occurred=start_of_test,
            event='prefect.work-queue.healthy',
            resource={'prefect.resource.id': f'prefect.work-queue.{uuid4()}', 'prefect.resource.name': 'some-queue'},
            related=[
                {'prefect.resource.id': f'prefect.work-pool.{uuid4()}', 'prefect.resource.role': 'work-pool', 'prefect.resource.name': 'this-pool'},
                {'prefect.resource.id': f'prefect.work-pool.{uuid4()}', 'prefect.resource.role': 'work-pool', 'prefect.resource.name': 'that-pool'}
            ],
            id=uuid4()
        ).receive(),
        triggering_labels={}
    )
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    assert rendered_list[0] == '\n    Name: some-queue\n    Pool: this-pool\n    Pools: this-pool,that-pool,\n    '


async def test_work_pool_is_available_to_templates(
    work_pool: ORMWorkPool,
    work_queue: ORMWorkQueue,
    tell_me_about_the_culprit: Automation,
    start_of_test: DateTime
) -> None:
    """
    Regression test for https://github.com/PrefectHQ/nebula/issues/5215, where we want
    to add {{ work_pool }} as a template shortcut
    """
    template: str = '''
    Name: {{ work_queue.name }}
    Pool: {{ work_pool.name }}
    '''
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=start_of_test + timedelta(seconds=1),
        triggering_event=Event(
            occurred=start_of_test,
            event='prefect.work-queue.healthy',
            resource={'prefect.resource.id': f'prefect.work-queue.{work_queue.id}'},
            related=[{'prefect.resource.id': f'prefect.work-pool.{work_pool.id}', 'prefect.resource.role': 'work-pool', 'prefect.resource.name': 'test-pool'}],
            id=uuid4()
        ).receive(),
        triggering_labels={}
    )
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    assert rendered_list[0] == f'\n    Name: {work_queue.name}\n    Pool: {work_pool.name}\n    '


async def test_concurrency_limit_is_available_in_templates(
    concurrency_limit_v2: ORMConcurrencyLimitV2,
    tell_me_about_the_culprit: Automation,
    start_of_test: DateTime
) -> None:
    """
    Regression test for https://github.com/PrefectHQ/nebula/issues/5120, where we want
    to add {{ concurrency_limit }} as a template shortcut (for v2 limits)
    """
    template: str = '''
    Name: {{ concurrency_limit.name }}
    Limit: {{ concurrency_limit.limit }}
    '''
    firing = Firing(
        trigger=tell_me_about_the_culprit.trigger,
        trigger_states={TriggerState.Triggered},
        triggered=start_of_test + timedelta(seconds=1),
        triggering_event=Event(
            occurred=start_of_test,
            event='prefect.concurrency-limit.acquired',
            resource={'prefect.resource.id': f'prefect.concurrency-limit.{concurrency_limit_v2.id}'},
            id=uuid4()
        ).receive(),
        triggering_labels={}
    )
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        firing=firing,
        triggered=firing.triggered,
        triggering_labels=firing.triggering_labels,
        triggering_event=firing.triggering_event,
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    assert rendered_list[0] == '\n    Name: my-limit\n    Limit: 42\n    '


async def test_composite_firings_are_available_in_templates(
    tell_me_about_the_culprit: Automation, start_of_test: DateTime
) -> None:
    template: str = '''
    Firings: {{ firings|length }}
    Outer: {{ firings[0].trigger.id }}
    Child 1: {{ firings[1].trigger.id }}
    Child 2: {{ firings[2].trigger.id }}
    '''
    first_child = tell_me_about_the_culprit.trigger.model_copy(update={'id': uuid4()})
    second_child = tell_me_about_the_culprit.trigger.model_copy(update={'id': uuid4()})
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        triggered=start_of_test + timedelta(seconds=1),
        firing=Firing(
            trigger=tell_me_about_the_culprit.trigger,
            trigger_states={TriggerState.Triggered},
            triggered=start_of_test + timedelta(seconds=1),
            triggering_firings=[
                Firing(
                    trigger=first_child,
                    trigger_states={TriggerState.Triggered},
                    triggered=start_of_test + timedelta(seconds=1)
                ),
                Firing(
                    trigger=second_child,
                    trigger_states={TriggerState.Triggered},
                    triggered=start_of_test + timedelta(seconds=1)
                )
            ]
        ),
        triggering_event=None,
        triggering_labels={},
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    expected_text: str = f'\n    Firings: 3\n    Outer: {tell_me_about_the_culprit.trigger.id}\n    Child 1: {first_child.id}\n    Child 2: {second_child.id}\n    '
    assert rendered_list[0] == expected_text


async def test_composite_triggering_events_are_available_in_templates(
    picture_taken: ReceivedEvent, tell_me_about_the_culprit: Automation, start_of_test: DateTime
) -> None:
    template: str = '''
    Events: {{ events|length }}
    Event 1: {{ events[0].id }}
    Event 2: {{ events[1].id }}
    '''
    first_event = picture_taken.model_copy(update={'id': uuid4()})
    second_event = picture_taken.model_copy(update={'id': uuid4()})
    triggered_action = TriggeredAction(
        automation=tell_me_about_the_culprit,
        id=uuid4(),
        triggered=start_of_test + timedelta(seconds=1),
        firing=Firing(
            trigger=tell_me_about_the_culprit.trigger,
            trigger_states={TriggerState.Triggered},
            triggered=start_of_test + timedelta(seconds=1),
            triggering_firings=[
                Firing(
                    trigger=tell_me_about_the_culprit.trigger,
                    trigger_states={TriggerState.Triggered},
                    triggered=start_of_test + timedelta(seconds=1),
                    triggering_event=first_event
                ),
                Firing(
                    trigger=tell_me_about_the_culprit.trigger,
                    trigger_states={TriggerState.Triggered},
                    triggered=start_of_test + timedelta(seconds=1),
                    triggering_event=second_event
                )
            ]
        ),
        triggering_event=None,
        triggering_labels={},
        action=actions.DoNothing(),
        action_index=0
    )
    action: DemoAction = DemoAction(template=template)
    rendered_list: List[str] = await action.render(triggered_action)
    expected_text: str = f'\n    Events: 2\n    Event 1: {first_event.id}\n    Event 2: {second_event.id}\n    '
    assert rendered_list[0] == expected_text
