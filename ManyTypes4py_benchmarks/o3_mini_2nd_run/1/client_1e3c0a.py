from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union, Optional, List, Tuple, Dict
from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound

if TYPE_CHECKING:
    import datetime
    from uuid import UUID
    from prefect.client.schemas import FlowRun
    from prefect.client.schemas.actions import DeploymentCreate, DeploymentScheduleCreate, DeploymentScheduleUpdate, DeploymentFlowRunCreate
    from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
    from prefect.client.schemas.objects import ConcurrencyOptions, DeploymentSchedule
    from prefect.client.schemas.responses import DeploymentResponse, FlowRunResponse
    from prefect.client.schemas.sorting import DeploymentSort
    from prefect.states import State
    from prefect.types import KeyValueLabelsField


class DeploymentClient(BaseClient):

    def create_deployment(
        self,
        flow_id: Union[str, "UUID"],
        name: str,
        version: Optional[str] = None,
        schedules: Optional[Iterable[Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_options: Optional[ConcurrencyOptions] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        work_pool_name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        storage_document_id: Optional[str] = None,
        path: Optional[str] = None,
        entrypoint: Optional[str] = None,
        infrastructure_document_id: Optional[str] = None,
        parameter_openapi_schema: Optional[Dict[str, Any]] = None,
        paused: Optional[bool] = None,
        pull_steps: Optional[Any] = None,
        enforce_parameter_schema: Optional[bool] = None,
        job_variables: Optional[Dict[str, Any]] = None
    ) -> "UUID":
        """
        Create a deployment.
        """
        from uuid import UUID
        from prefect.client.schemas.actions import DeploymentCreate
        if parameter_openapi_schema is None:
            parameter_openapi_schema = {}
        deployment_create = DeploymentCreate(
            flow_id=flow_id,
            name=name,
            version=version,
            parameters=dict(parameters or {}),
            tags=list(tags or []),
            work_queue_name=work_queue_name,
            description=description,
            storage_document_id=storage_document_id,
            path=path,
            entrypoint=entrypoint,
            infrastructure_document_id=infrastructure_document_id,
            job_variables=dict(job_variables or {}),
            parameter_openapi_schema=parameter_openapi_schema,
            paused=paused,
            schedules=schedules or [],
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            pull_steps=pull_steps,
            enforce_parameter_schema=enforce_parameter_schema
        )
        if work_pool_name is not None:
            deployment_create.work_pool_name = work_pool_name
        exclude = {field for field in ['work_pool_name', 'work_queue_name'] if field not in deployment_create.model_fields_set}
        if deployment_create.paused is None:
            exclude.add('paused')
        if deployment_create.pull_steps is None:
            exclude.add('pull_steps')
        if deployment_create.enforce_parameter_schema is None:
            exclude.add('enforce_parameter_schema')
        json = deployment_create.model_dump(mode='json', exclude=exclude)
        response = self.request('POST', '/deployments/', json=json)
        deployment_id = response.json().get('id')
        if not deployment_id:
            raise RequestError(f'Malformed response: {response}')
        return UUID(deployment_id)

    def set_deployment_paused_state(
        self, deployment_id: Union[str, "UUID"], paused: bool
    ) -> None:
        self.request('PATCH', '/deployments/{id}', path_params={'id': deployment_id}, json={'paused': paused})

    def update_deployment(self, deployment_id: Union[str, "UUID"], deployment: Any) -> None:
        self.request(
            'PATCH',
            '/deployments/{id}',
            path_params={'id': deployment_id},
            json=deployment.model_dump(mode='json', exclude_unset=True, exclude={'name', 'flow_name', 'triggers'})
        )

    def _create_deployment_from_schema(self, schema: DeploymentCreate) -> "UUID":
        """
        Create a deployment from a prepared `DeploymentCreate` schema.
        """
        from uuid import UUID
        response = self.request('POST', '/deployments/', json=schema.model_dump(mode='json'))
        deployment_id = response.json().get('id')
        if not deployment_id:
            raise RequestError(f'Malformed response: {response}')
        return UUID(deployment_id)

    def read_deployment(self, deployment_id: Union[str, "UUID"]) -> DeploymentResponse:
        """
        Query the Prefect API for a deployment by id.
        """
        from uuid import UUID
        from prefect.client.schemas.responses import DeploymentResponse
        if not isinstance(deployment_id, UUID):
            try:
                deployment_id = UUID(deployment_id)
            except ValueError:
                raise ValueError(f'Invalid deployment ID: {deployment_id}')
        try:
            response = self.request('GET', '/deployments/{id}', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        return DeploymentResponse.model_validate(response.json())

    def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        """
        Query the Prefect API for a deployment by name.
        """
        from prefect.client.schemas.responses import DeploymentResponse
        try:
            flow_name, deployment_name = name.split('/')
            response = self.request(
                'GET',
                '/deployments/name/{flow_name}/{deployment_name}',
                path_params={'flow_name': flow_name, 'deployment_name': deployment_name}
            )
        except (HTTPStatusError, ValueError) as e:
            if isinstance(e, HTTPStatusError) and e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            elif isinstance(e, ValueError):
                raise ValueError(f'Invalid deployment name format: {name}. Expected format: <FLOW_NAME>/<DEPLOYMENT_NAME>') from e
            else:
                raise
        return DeploymentResponse.model_validate(response.json())

    def read_deployments(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        work_pool_filter: Optional[WorkPoolFilter] = None,
        work_queue_filter: Optional[WorkQueueFilter] = None,
        limit: Optional[int] = None,
        sort: Optional[DeploymentSort] = None,
        offset: int = 0
    ) -> List[DeploymentResponse]:
        """
        Query the Prefect API for deployments.
        """
        from prefect.client.schemas.responses import DeploymentResponse
        body = {
            'flows': flow_filter.model_dump(mode='json') if flow_filter else None,
            'flow_runs': flow_run_filter.model_dump(mode='json', exclude_unset=True) if flow_run_filter else None,
            'task_runs': task_run_filter.model_dump(mode='json') if task_run_filter else None,
            'deployments': deployment_filter.model_dump(mode='json') if deployment_filter else None,
            'work_pools': work_pool_filter.model_dump(mode='json') if work_pool_filter else None,
            'work_pool_queues': work_queue_filter.model_dump(mode='json') if work_queue_filter else None,
            'limit': limit,
            'offset': offset,
            'sort': sort
        }
        response = self.request('POST', '/deployments/filter', json=body)
        return DeploymentResponse.model_validate_list(response.json())

    def delete_deployment(self, deployment_id: Union[str, "UUID"]) -> None:
        """
        Delete deployment by id.
        """
        try:
            self.request('DELETE', '/deployments/{id}', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def create_deployment_schedules(
        self,
        deployment_id: Union[str, "UUID"],
        schedules: List[Tuple[Any, bool]]
    ) -> List[DeploymentSchedule]:
        """
        Create deployment schedules.
        """
        from prefect.client.schemas.actions import DeploymentScheduleCreate
        from prefect.client.schemas.objects import DeploymentSchedule
        deployment_schedule_create_list = [
            DeploymentScheduleCreate(schedule=schedule[0], active=schedule[1]) for schedule in schedules
        ]
        json_list = [dsc.model_dump(mode='json') for dsc in deployment_schedule_create_list]
        response = self.request('POST', f'/deployments/{deployment_id}/schedules', path_params={'id': deployment_id}, json=json_list)
        return DeploymentSchedule.model_validate_list(response.json())

    def read_deployment_schedules(self, deployment_id: Union[str, "UUID"]) -> List[DeploymentSchedule]:
        """
        Query the Prefect API for a deployment's schedules.
        """
        from prefect.client.schemas.objects import DeploymentSchedule
        try:
            response = self.request('GET', '/deployments/{id}/schedules', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        return DeploymentSchedule.model_validate_list(response.json())

    def update_deployment_schedule(
        self,
        deployment_id: Union[str, "UUID"],
        schedule_id: Union[str, "UUID"],
        active: Optional[bool] = None,
        schedule: Optional[Any] = None
    ) -> None:
        """
        Update a deployment schedule by ID.
        """
        from prefect.client.schemas.actions import DeploymentScheduleUpdate
        kwargs: Dict[str, Any] = {}
        if active is not None:
            kwargs['active'] = active
        if schedule is not None:
            kwargs['schedule'] = schedule
        deployment_schedule_update = DeploymentScheduleUpdate(**kwargs)
        json_payload = deployment_schedule_update.model_dump(mode='json', exclude_unset=True)
        try:
            self.request(
                'PATCH',
                '/deployments/{id}/schedules/{schedule_id}',
                path_params={'id': deployment_id, 'schedule_id': schedule_id},
                json=json_payload
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def delete_deployment_schedule(
        self,
        deployment_id: Union[str, "UUID"],
        schedule_id: Union[str, "UUID"]
    ) -> None:
        """
        Delete a deployment schedule.
        """
        try:
            self.request(
                'DELETE',
                '/deployments/{id}/schedules/{schedule_id}',
                path_params={'id': deployment_id, 'schedule_id': schedule_id}
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[str, "UUID"]],
        scheduled_before: Optional[Any] = None,
        limit: Optional[int] = None
    ) -> List[FlowRunResponse]:
        from prefect.client.schemas.responses import FlowRunResponse
        body: Dict[str, Any] = {'deployment_ids': [str(id) for id in deployment_ids]}
        if scheduled_before:
            body['scheduled_before'] = str(scheduled_before)
        if limit:
            body['limit'] = limit
        response = self.request('POST', '/deployments/get_scheduled_flow_runs', json=body)
        return FlowRunResponse.model_validate_list(response.json())

    def create_flow_run_from_deployment(
        self,
        deployment_id: Union[str, "UUID"],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        state: Optional[State] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        idempotency_key: Optional[str] = None,
        parent_task_run_id: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        labels: Optional[KeyValueLabelsField] = None
    ) -> FlowRun:
        """
        Create a flow run for a deployment.
        """
        from prefect.client.schemas.actions import DeploymentFlowRunCreate
        from prefect.client.schemas.objects import FlowRun
        from prefect.states import Scheduled, to_state_create
        parameters = parameters or {}
        context = context or {}
        state = state or Scheduled()
        tags = tags or []
        labels = labels or {}
        flow_run_create = DeploymentFlowRunCreate(
            parameters=parameters,
            context=context,
            state=to_state_create(state),
            tags=list(tags),
            name=name,
            idempotency_key=idempotency_key,
            parent_task_run_id=parent_task_run_id,
            job_variables=job_variables,
            labels=labels
        )
        if work_queue_name:
            flow_run_create.work_queue_name = work_queue_name
        response = self.request(
            'POST',
            '/deployments/{id}/create_flow_run',
            path_params={'id': deployment_id},
            json=flow_run_create.model_dump(mode='json', exclude_unset=True)
        )
        return FlowRun.model_validate(response.json())


class DeploymentAsyncClient(BaseAsyncClient):

    async def create_deployment(
        self,
        flow_id: Union[str, "UUID"],
        name: str,
        version: Optional[str] = None,
        schedules: Optional[Iterable[Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_options: Optional[ConcurrencyOptions] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        work_pool_name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        storage_document_id: Optional[str] = None,
        path: Optional[str] = None,
        entrypoint: Optional[str] = None,
        infrastructure_document_id: Optional[str] = None,
        parameter_openapi_schema: Optional[Dict[str, Any]] = None,
        paused: Optional[bool] = None,
        pull_steps: Optional[Any] = None,
        enforce_parameter_schema: Optional[bool] = None,
        job_variables: Optional[Dict[str, Any]] = None
    ) -> "UUID":
        """
        Create a deployment.
        """
        from uuid import UUID
        from prefect.client.schemas.actions import DeploymentCreate
        if parameter_openapi_schema is None:
            parameter_openapi_schema = {}
        deployment_create = DeploymentCreate(
            flow_id=flow_id,
            name=name,
            version=version,
            parameters=dict(parameters or {}),
            tags=list(tags or []),
            work_queue_name=work_queue_name,
            description=description,
            storage_document_id=storage_document_id,
            path=path,
            entrypoint=entrypoint,
            infrastructure_document_id=infrastructure_document_id,
            job_variables=dict(job_variables or {}),
            parameter_openapi_schema=parameter_openapi_schema,
            paused=paused,
            schedules=schedules or [],
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            pull_steps=pull_steps,
            enforce_parameter_schema=enforce_parameter_schema
        )
        if work_pool_name is not None:
            deployment_create.work_pool_name = work_pool_name
        exclude = {field for field in ['work_pool_name', 'work_queue_name'] if field not in deployment_create.model_fields_set}
        if deployment_create.paused is None:
            exclude.add('paused')
        if deployment_create.pull_steps is None:
            exclude.add('pull_steps')
        if deployment_create.enforce_parameter_schema is None:
            exclude.add('enforce_parameter_schema')
        json = deployment_create.model_dump(mode='json', exclude=exclude)
        response = await self.request('POST', '/deployments/', json=json)
        deployment_id = response.json().get('id')
        if not deployment_id:
            raise RequestError(f'Malformed response: {response}')
        return UUID(deployment_id)

    async def set_deployment_paused_state(
        self, deployment_id: Union[str, "UUID"], paused: bool
    ) -> None:
        await self.request('PATCH', '/deployments/{id}', path_params={'id': deployment_id}, json={'paused': paused})

    async def update_deployment(self, deployment_id: Union[str, "UUID"], deployment: Any) -> None:
        await self.request(
            'PATCH',
            '/deployments/{id}',
            path_params={'id': deployment_id},
            json=deployment.model_dump(mode='json', exclude_unset=True, exclude={'name', 'flow_name', 'triggers'})
        )

    async def _create_deployment_from_schema(self, schema: DeploymentCreate) -> "UUID":
        """
        Create a deployment from a prepared `DeploymentCreate` schema.
        """
        from uuid import UUID
        response = await self.request('POST', '/deployments/', json=schema.model_dump(mode='json'))
        deployment_id = response.json().get('id')
        if not deployment_id:
            raise RequestError(f'Malformed response: {response}')
        return UUID(deployment_id)

    async def read_deployment(self, deployment_id: Union[str, "UUID"]) -> DeploymentResponse:
        """
        Query the Prefect API for a deployment by id.
        """
        from uuid import UUID
        from prefect.client.schemas.responses import DeploymentResponse
        if not isinstance(deployment_id, UUID):
            try:
                deployment_id = UUID(deployment_id)
            except ValueError:
                raise ValueError(f'Invalid deployment ID: {deployment_id}')
        try:
            response = await self.request('GET', '/deployments/{id}', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        return DeploymentResponse.model_validate(response.json())

    async def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        """
        Query the Prefect API for a deployment by name.
        """
        from prefect.client.schemas.responses import DeploymentResponse
        try:
            flow_name, deployment_name = name.split('/')
            response = await self.request(
                'GET',
                '/deployments/name/{flow_name}/{deployment_name}',
                path_params={'flow_name': flow_name, 'deployment_name': deployment_name}
            )
        except (HTTPStatusError, ValueError) as e:
            if isinstance(e, HTTPStatusError) and e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            elif isinstance(e, ValueError):
                raise ValueError(f'Invalid deployment name format: {name}. Expected format: <FLOW_NAME>/<DEPLOYMENT_NAME>') from e
            else:
                raise
        return DeploymentResponse.model_validate(response.json())

    async def read_deployments(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        work_pool_filter: Optional[WorkPoolFilter] = None,
        work_queue_filter: Optional[WorkQueueFilter] = None,
        limit: Optional[int] = None,
        sort: Optional[DeploymentSort] = None,
        offset: int = 0
    ) -> List[DeploymentResponse]:
        """
        Query the Prefect API for deployments.
        """
        from prefect.client.schemas.responses import DeploymentResponse
        body = {
            'flows': flow_filter.model_dump(mode='json') if flow_filter else None,
            'flow_runs': flow_run_filter.model_dump(mode='json', exclude_unset=True) if flow_run_filter else None,
            'task_runs': task_run_filter.model_dump(mode='json') if task_run_filter else None,
            'deployments': deployment_filter.model_dump(mode='json') if deployment_filter else None,
            'work_pools': work_pool_filter.model_dump(mode='json') if work_pool_filter else None,
            'work_pool_queues': work_queue_filter.model_dump(mode='json') if work_queue_filter else None,
            'limit': limit,
            'offset': offset,
            'sort': sort
        }
        response = await self.request('POST', '/deployments/filter', json=body)
        return DeploymentResponse.model_validate_list(response.json())

    async def delete_deployment(self, deployment_id: Union[str, "UUID"]) -> None:
        """
        Delete deployment by id.
        """
        try:
            await self.request('DELETE', '/deployments/{id}', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def create_deployment_schedules(
        self,
        deployment_id: Union[str, "UUID"],
        schedules: List[Tuple[Any, bool]]
    ) -> List[DeploymentSchedule]:
        """
        Create deployment schedules.
        """
        from prefect.client.schemas.actions import DeploymentScheduleCreate
        from prefect.client.schemas.objects import DeploymentSchedule
        deployment_schedule_create_list = [
            DeploymentScheduleCreate(schedule=schedule[0], active=schedule[1]) for schedule in schedules
        ]
        json_list = [dsc.model_dump(mode='json') for dsc in deployment_schedule_create_list]
        response = await self.request('POST', f'/deployments/{deployment_id}/schedules', path_params={'id': deployment_id}, json=json_list)
        return DeploymentSchedule.model_validate_list(response.json())

    async def read_deployment_schedules(self, deployment_id: Union[str, "UUID"]) -> List[DeploymentSchedule]:
        """
        Query the Prefect API for a deployment's schedules.
        """
        from prefect.client.schemas.objects import DeploymentSchedule
        try:
            response = await self.request('GET', '/deployments/{id}/schedules', path_params={'id': deployment_id})
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise
        return DeploymentSchedule.model_validate_list(response.json())

    async def update_deployment_schedule(
        self,
        deployment_id: Union[str, "UUID"],
        schedule_id: Union[str, "UUID"],
        active: Optional[bool] = None,
        schedule: Optional[Any] = None
    ) -> None:
        """
        Update a deployment schedule by ID.
        """
        from prefect.client.schemas.actions import DeploymentScheduleUpdate
        kwargs: Dict[str, Any] = {}
        if active is not None:
            kwargs['active'] = active
        if schedule is not None:
            kwargs['schedule'] = schedule
        deployment_schedule_update = DeploymentScheduleUpdate(**kwargs)
        json_payload = deployment_schedule_update.model_dump(mode='json', exclude_unset=True)
        try:
            await self.request(
                'PATCH',
                '/deployments/{id}/schedules/{schedule_id}',
                path_params={'id': deployment_id, 'schedule_id': schedule_id},
                json=json_payload
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def delete_deployment_schedule(
        self,
        deployment_id: Union[str, "UUID"],
        schedule_id: Union[str, "UUID"]
    ) -> None:
        """
        Delete a deployment schedule.
        """
        try:
            await self.request(
                'DELETE',
                '/deployments/{id}/schedules/{schedule_id}',
                path_params={'id': deployment_id, 'schedule_id': schedule_id}
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[str, "UUID"]],
        scheduled_before: Optional[Any] = None,
        limit: Optional[int] = None
    ) -> List[FlowRunResponse]:
        from prefect.client.schemas.responses import FlowRunResponse
        body: Dict[str, Any] = {'deployment_ids': [str(id) for id in deployment_ids]}
        if scheduled_before:
            body['scheduled_before'] = str(scheduled_before)
        if limit:
            body['limit'] = limit
        response = await self.request('POST', '/deployments/get_scheduled_flow_runs', json=body)
        return FlowRunResponse.model_validate_list(response.json())

    async def create_flow_run_from_deployment(
        self,
        deployment_id: Union[str, "UUID"],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        state: Optional[State] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        idempotency_key: Optional[str] = None,
        parent_task_run_id: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        labels: Optional[KeyValueLabelsField] = None
    ) -> FlowRun:
        """
        Create a flow run for a deployment.
        """
        from prefect.client.schemas.actions import DeploymentFlowRunCreate
        from prefect.client.schemas.objects import FlowRun
        from prefect.states import Scheduled, to_state_create
        parameters = parameters or {}
        context = context or {}
        state = state or Scheduled()
        tags = tags or []
        labels = labels or {}
        flow_run_create = DeploymentFlowRunCreate(
            parameters=parameters,
            context=context,
            state=to_state_create(state),
            tags=list(tags),
            name=name,
            idempotency_key=idempotency_key,
            parent_task_run_id=parent_task_run_id,
            job_variables=job_variables,
            labels=labels
        )
        if work_queue_name:
            flow_run_create.work_queue_name = work_queue_name
        response = await self.request(
            'POST',
            '/deployments/{id}/create_flow_run',
            path_params={'id': deployment_id},
            json=flow_run_create.model_dump(mode='json', exclude_unset=True)
        )
        return FlowRun.model_validate(response.json())