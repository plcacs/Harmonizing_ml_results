from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, Dict, Union, List
import httpx
from typing_extensions import TypeVar
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound

T = TypeVar('T')
R = TypeVar('R', infer_variance=True)

if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas import FlowRun, OrchestrationResult
    from prefect.client.schemas.filters import (
        DeploymentFilter,
        FlowFilter,
        FlowRunFilter,
        TaskRunFilter,
        WorkPoolFilter,
        WorkQueueFilter,
    )
    from prefect.client.schemas.objects import FlowRunInput, FlowRunPolicy
    from prefect.client.schemas.sorting import FlowRunSort
    from prefect.flows import Flow as FlowObject
    from prefect.states import State
    from prefect.types import KeyValueLabelsField

class FlowRunClient(BaseClient):

    def create_flow_run(
        self,
        flow: FlowObject,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        parent_task_run_id: Optional[str] = None,
        state: Optional[State] = None
    ) -> FlowRun:
        """
        Create a flow run for a flow.
        """
        from prefect.client.schemas.actions import FlowCreate, FlowRunCreate
        from prefect.client.schemas.objects import Flow, FlowRun, FlowRunPolicy
        from prefect.states import Pending, to_state_create
        parameters = parameters or {}
        context = context or {}
        if state is None:
            state = Pending()
        flow_data = FlowCreate(name=flow.name)
        response = self.request('POST', '/flows/', json=flow_data.model_dump(mode='json'))
        flow_id = Flow.model_validate(response.json()).id
        flow_run_create = FlowRunCreate(
            flow_id=flow_id,
            flow_version=flow.version,
            name=name,
            parameters=parameters,
            context=context,
            tags=list(tags or []),
            parent_task_run_id=parent_task_run_id,
            state=to_state_create(state),
            empirical_policy=FlowRunPolicy(retries=flow.retries, retry_delay=int(flow.retry_delay_seconds or 0))
        )
        flow_run_create_json = flow_run_create.model_dump(mode='json')
        response = self.request('POST', '/flow_runs/', json=flow_run_create_json)
        flow_run = FlowRun.model_validate(response.json())
        flow_run.parameters = parameters
        return flow_run

    def update_flow_run(
        self,
        flow_run_id: Union[str, "UUID"],
        flow_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        empirical_policy: Optional[FlowRunPolicy] = None,
        infrastructure_pid: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """
        Update a flow run's details.
        """
        params: Dict[str, Any] = {}
        if flow_version is not None:
            params['flow_version'] = flow_version
        if parameters is not None:
            params['parameters'] = parameters
        if name is not None:
            params['name'] = name
        if tags is not None:
            params['tags'] = tags
        if empirical_policy is not None:
            params['empirical_policy'] = empirical_policy
        if infrastructure_pid:
            params['infrastructure_pid'] = infrastructure_pid
        if job_variables is not None:
            params['job_variables'] = job_variables
        from prefect.client.schemas.actions import FlowRunUpdate
        flow_run_data = FlowRunUpdate(**params)
        return self.request(
            'PATCH',
            '/flow_runs/{id}',
            path_params={'id': flow_run_id},
            json=flow_run_data.model_dump(mode='json', exclude_unset=True)
        )

    def delete_flow_run(self, flow_run_id: Union[str, "UUID"]) -> None:
        """
        Delete a flow run by UUID.
        """
        try:
            self.request('DELETE', '/flow_runs/{id}', path_params={'id': flow_run_id})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def read_flow_run(self, flow_run_id: Union[str, "UUID"]) -> FlowRun:
        """
        Query the Prefect API for a flow run by id.
        """
        try:
            response = self.request('GET', '/flow_runs/{id}', path_params={'id': flow_run_id})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import FlowRun
        return FlowRun.model_validate(response.json())

    def resume_flow_run(
        self,
        flow_run_id: Union[str, "UUID"],
        run_input: Optional[Any] = None
    ) -> "OrchestrationResult":
        """
        Resumes a paused flow run.
        """
        try:
            response = self.request(
                'POST',
                '/flow_runs/{id}/resume',
                path_params={'id': flow_run_id},
                json={'run_input': run_input}
            )
        except httpx.HTTPStatusError:
            raise
        from prefect.client.schemas import OrchestrationResult
        result = OrchestrationResult.model_validate(response.json())
        return result

    def read_flow_runs(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        work_pool_filter: Optional[WorkPoolFilter] = None,
        work_queue_filter: Optional[WorkQueueFilter] = None,
        sort: Optional[FlowRunSort] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[FlowRun]:
        """
        Query the Prefect API for flow runs.
        """
        body = {
            'flows': flow_filter.model_dump(mode='json') if flow_filter else None,
            'flow_runs': flow_run_filter.model_dump(mode='json', exclude_unset=True) if flow_run_filter else None,
            'task_runs': task_run_filter.model_dump(mode='json') if task_run_filter else None,
            'deployments': deployment_filter.model_dump(mode='json') if deployment_filter else None,
            'work_pools': work_pool_filter.model_dump(mode='json') if work_pool_filter else None,
            'work_pool_queues': work_queue_filter.model_dump(mode='json') if work_queue_filter else None,
            'sort': sort,
            'limit': limit,
            'offset': offset
        }
        response = self.request('POST', '/flow_runs/filter', json=body)
        from prefect.client.schemas.objects import FlowRun
        return FlowRun.model_validate_list(response.json())

    def set_flow_run_state(
        self,
        flow_run_id: Union[str, "UUID"],
        state: State,
        force: bool = False
    ) -> "OrchestrationResult":
        """
        Set the state of a flow run.
        """
        from uuid import UUID, uuid4
        from prefect.states import to_state_create
        flow_run_id = flow_run_id if isinstance(flow_run_id, UUID) else UUID(flow_run_id)
        state_create = to_state_create(state)
        state_create.state_details.flow_run_id = flow_run_id
        state_create.state_details.transition_id = uuid4()
        try:
            response = self.request(
                'POST',
                '/flow_runs/{id}/set_state',
                path_params={'id': flow_run_id},
                json=dict(state=state_create.model_dump(mode='json', serialize_as_any=True), force=force)
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas import OrchestrationResult
        result = OrchestrationResult.model_validate(response.json())
        return result

    def read_flow_run_states(
        self,
        flow_run_id: Union[str, "UUID"]
    ) -> List[State]:
        """
        Query for the states of a flow run.
        """
        response = self.request('GET', '/flow_run_states/', params=dict(flow_run_id=str(flow_run_id)))
        from prefect.states import State
        return State.model_validate_list(response.json())

    def set_flow_run_name(
        self,
        flow_run_id: Union[str, "UUID"],
        name: str
    ) -> httpx.Response:
        from prefect.client.schemas.actions import FlowRunUpdate
        flow_run_data = FlowRunUpdate(name=name)
        return self.request(
            'PATCH',
            '/flow_runs/{id}',
            path_params={'id': flow_run_id},
            json=flow_run_data.model_dump(mode='json', exclude_unset=True)
        )

    def create_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str,
        value: Any,
        sender: Optional[str] = None
    ) -> None:
        """
        Creates a flow run input.
        """
        from prefect.client.schemas.objects import FlowRunInput
        FlowRunInput(flow_run_id=flow_run_id, key=key, value=value)
        response = self.request(
            'POST',
            '/flow_runs/{id}/input',
            path_params={'id': flow_run_id},
            json={'key': key, 'value': value, 'sender': sender}
        )
        response.raise_for_status()

    def filter_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key_prefix: str,
        limit: int,
        exclude_keys: Iterable[str]
    ) -> List[FlowRunInput]:
        response = self.request(
            'POST',
            '/flow_runs/{id}/input/filter',
            path_params={'id': flow_run_id},
            json={'prefix': key_prefix, 'limit': limit, 'exclude_keys': list(exclude_keys)}
        )
        response.raise_for_status()
        from prefect.client.schemas.objects import FlowRunInput
        return FlowRunInput.model_validate_list(response.json())

    def read_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str
    ) -> str:
        """
        Reads a flow run input.
        """
        response = self.request(
            'GET',
            '/flow_runs/{id}/input/{key}',
            path_params={'id': flow_run_id, 'key': key}
        )
        response.raise_for_status()
        return response.content.decode()

    def delete_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str
    ) -> None:
        """
        Deletes a flow run input.
        """
        response = self.request(
            'DELETE',
            '/flow_runs/{id}/input/{key}',
            path_params={'id': flow_run_id, 'key': key}
        )
        response.raise_for_status()

    def update_flow_run_labels(
        self,
        flow_run_id: Union[str, "UUID"],
        labels: KeyValueLabelsField
    ) -> None:
        """
        Updates the labels of a flow run.
        """
        response = self.request(
            'PATCH',
            '/flow_runs/{id}/labels',
            path_params={'id': flow_run_id},
            json=labels
        )
        response.raise_for_status()


class FlowRunAsyncClient(BaseAsyncClient):

    async def create_flow_run(
        self,
        flow: FlowObject,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        parent_task_run_id: Optional[str] = None,
        state: Optional[State] = None
    ) -> FlowRun:
        """
        Create a flow run for a flow.
        """
        from prefect.client.schemas.actions import FlowCreate, FlowRunCreate
        from prefect.client.schemas.objects import Flow, FlowRun, FlowRunPolicy
        from prefect.states import Pending, to_state_create
        parameters = parameters or {}
        context = context or {}
        if state is None:
            state = Pending()
        flow_data = FlowCreate(name=flow.name)
        response = await self.request('POST', '/flows/', json=flow_data.model_dump(mode='json'))
        flow_id = Flow.model_validate(response.json()).id
        flow_run_create = FlowRunCreate(
            flow_id=flow_id,
            flow_version=flow.version,
            name=name,
            parameters=parameters,
            context=context,
            tags=list(tags or []),
            parent_task_run_id=parent_task_run_id,
            state=to_state_create(state),
            empirical_policy=FlowRunPolicy(retries=flow.retries, retry_delay=int(flow.retry_delay_seconds or 0))
        )
        flow_run_create_json = flow_run_create.model_dump(mode='json')
        response = await self.request('POST', '/flow_runs/', json=flow_run_create_json)
        flow_run = FlowRun.model_validate(response.json())
        flow_run.parameters = parameters
        return flow_run

    async def update_flow_run(
        self,
        flow_run_id: Union[str, "UUID"],
        flow_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        empirical_policy: Optional[FlowRunPolicy] = None,
        infrastructure_pid: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """
        Update a flow run's details.
        """
        params: Dict[str, Any] = {}
        if flow_version is not None:
            params['flow_version'] = flow_version
        if parameters is not None:
            params['parameters'] = parameters
        if name is not None:
            params['name'] = name
        if tags is not None:
            params['tags'] = tags
        if empirical_policy is not None:
            params['empirical_policy'] = empirical_policy
        if infrastructure_pid:
            params['infrastructure_pid'] = infrastructure_pid
        if job_variables is not None:
            params['job_variables'] = job_variables
        from prefect.client.schemas.actions import FlowRunUpdate
        flow_run_data = FlowRunUpdate(**params)
        return await self.request(
            'PATCH',
            '/flow_runs/{id}',
            path_params={'id': flow_run_id},
            json=flow_run_data.model_dump(mode='json', exclude_unset=True)
        )

    async def delete_flow_run(
        self,
        flow_run_id: Union[str, "UUID"]
    ) -> None:
        """
        Delete a flow run by UUID.
        """
        try:
            await self.request('DELETE', '/flow_runs/{id}', path_params={'id': flow_run_id})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def read_flow_run(
        self,
        flow_run_id: Union[str, "UUID"]
    ) -> FlowRun:
        """
        Query the Prefect API for a flow run by id.
        """
        try:
            response = await self.request('GET', '/flow_runs/{id}', path_params={'id': flow_run_id})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas.objects import FlowRun
        return FlowRun.model_validate(response.json())

    async def resume_flow_run(
        self,
        flow_run_id: Union[str, "UUID"],
        run_input: Optional[Any] = None
    ) -> "OrchestrationResult":
        """
        Resumes a paused flow run.
        """
        try:
            response = await self.request(
                'POST',
                '/flow_runs/{id}/resume',
                path_params={'id': flow_run_id},
                json={'run_input': run_input}
            )
        except httpx.HTTPStatusError:
            raise
        from prefect.client.schemas import OrchestrationResult
        result = OrchestrationResult.model_validate(response.json())
        return result

    async def read_flow_runs(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        work_pool_filter: Optional[WorkPoolFilter] = None,
        work_queue_filter: Optional[WorkQueueFilter] = None,
        sort: Optional[FlowRunSort] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[FlowRun]:
        """
        Query the Prefect API for flow runs.
        """
        body = {
            'flows': flow_filter.model_dump(mode='json') if flow_filter else None,
            'flow_runs': flow_run_filter.model_dump(mode='json', exclude_unset=True) if flow_run_filter else None,
            'task_runs': task_run_filter.model_dump(mode='json') if task_run_filter else None,
            'deployments': deployment_filter.model_dump(mode='json') if deployment_filter else None,
            'work_pools': work_pool_filter.model_dump(mode='json') if work_pool_filter else None,
            'work_pool_queues': work_queue_filter.model_dump(mode='json') if work_queue_filter else None,
            'sort': sort,
            'limit': limit,
            'offset': offset
        }
        response = await self.request('POST', '/flow_runs/filter', json=body)
        from prefect.client.schemas.objects import FlowRun
        return FlowRun.model_validate_list(response.json())

    async def set_flow_run_state(
        self,
        flow_run_id: Union[str, "UUID"],
        state: State,
        force: bool = False
    ) -> "OrchestrationResult":
        """
        Set the state of a flow run.
        """
        from uuid import UUID, uuid4
        from prefect.states import to_state_create
        flow_run_id = flow_run_id if isinstance(flow_run_id, UUID) else UUID(flow_run_id)
        state_create = to_state_create(state)
        state_create.state_details.flow_run_id = flow_run_id
        state_create.state_details.transition_id = uuid4()
        try:
            response = await self.request(
                'POST',
                '/flow_runs/{id}/set_state',
                path_params={'id': flow_run_id},
                json=dict(state=state_create.model_dump(mode='json', serialize_as_any=True), force=force)
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        from prefect.client.schemas import OrchestrationResult
        result = OrchestrationResult.model_validate(response.json())
        return result

    async def read_flow_run_states(
        self,
        flow_run_id: Union[str, "UUID"]
    ) -> List[State]:
        """
        Query for the states of a flow run.
        """
        response = await self.request('GET', '/flow_run_states/', params=dict(flow_run_id=str(flow_run_id)))
        from prefect.states import State
        return State.model_validate_list(response.json())

    async def set_flow_run_name(
        self,
        flow_run_id: Union[str, "UUID"],
        name: str
    ) -> httpx.Response:
        from prefect.client.schemas.actions import FlowRunUpdate
        flow_run_data = FlowRunUpdate(name=name)
        return await self.request(
            'PATCH',
            '/flow_runs/{id}',
            path_params={'id': flow_run_id},
            json=flow_run_data.model_dump(mode='json', exclude_unset=True)
        )

    async def create_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str,
        value: Any,
        sender: Optional[str] = None
    ) -> None:
        """
        Creates a flow run input.
        """
        from prefect.client.schemas.objects import FlowRunInput
        FlowRunInput(flow_run_id=flow_run_id, key=key, value=value)
        response = await self.request(
            'POST',
            '/flow_runs/{id}/input',
            path_params={'id': flow_run_id},
            json={'key': key, 'value': value, 'sender': sender}
        )
        response.raise_for_status()

    async def filter_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key_prefix: str,
        limit: int,
        exclude_keys: Iterable[str]
    ) -> List[FlowRunInput]:
        response = await self.request(
            'POST',
            '/flow_runs/{id}/input/filter',
            path_params={'id': flow_run_id},
            json={'prefix': key_prefix, 'limit': limit, 'exclude_keys': list(exclude_keys)}
        )
        response.raise_for_status()
        from prefect.client.schemas.objects import FlowRunInput
        return FlowRunInput.model_validate_list(response.json())

    async def read_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str
    ) -> str:
        """
        Reads a flow run input.
        """
        response = await self.request(
            'GET',
            '/flow_runs/{id}/input/{key}',
            path_params={'id': flow_run_id, 'key': key}
        )
        response.raise_for_status()
        return response.content.decode()

    async def delete_flow_run_input(
        self,
        flow_run_id: Union[str, "UUID"],
        key: str
    ) -> None:
        """
        Deletes a flow run input.
        """
        response = await self.request(
            'DELETE',
            '/flow_runs/{id}/input/{key}',
            path_params={'id': flow_run_id, 'key': key}
        )
        response.raise_for_status()

    async def update_flow_run_labels(
        self,
        flow_run_id: Union[str, "UUID"],
        labels: KeyValueLabelsField
    ) -> None:
        """
        Updates the labels of a flow run.
        """
        response = await self.request(
            'PATCH',
            '/flow_runs/{id}/labels',
            path_params={'id': flow_run_id},
            json=labels
        )
        response.raise_for_status()