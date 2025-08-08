from pydantic import BaseModel, Field
from typing import Any, ClassVar, Dict, List, Optional, Type, Union
from prefect.server.schemas.core import CreatedBy, FlowRunPolicy, UpdatedBy, WorkQueueStatusDetail
from prefect.server.utilities.schemas.bases import ORMBaseModel, PrefectBaseModel
from prefect.types import DateTime, KeyValueLabelsField
from prefect.types._datetime import create_datetime_instance
from prefect.utilities.collections import AutoEnum
from prefect.utilities.names import generate_slug

class SetStateStatus(AutoEnum):
    ACCEPT: int
    REJECT: int
    ABORT: int
    WAIT: int

class StateAcceptDetails(PrefectBaseModel):
    type: str = Field(default='accept_details', description='The type of state transition detail. Used to ensure pydantic does not coerce into a different type.')

class StateRejectDetails(PrefectBaseModel):
    type: str = Field(default='reject_details', description='The type of state transition detail. Used to ensure pydantic does not coerce into a different type.')
    reason: Optional[str] = Field(default=None, description='The reason why the state transition was rejected.')

class StateAbortDetails(PrefectBaseModel):
    type: str = Field(default='abort_details', description='The type of state transition detail. Used to ensure pydantic does not coerce into a different type.')
    reason: Optional[str] = Field(default=None, description='The reason why the state transition was aborted.')

class StateWaitDetails(PrefectBaseModel):
    type: str = Field(default='wait_details', description='The type of state transition detail. Used to ensure pydantic does not coerce into a different type.')
    delay_seconds: Any = Field(default=..., description='The length of time in seconds the client should wait before transitioning states.')
    reason: Optional[str] = Field(default=None, description='The reason why the state transition should wait.')

class HistoryResponseState(PrefectBaseModel):
    state_type: Any = Field(default=..., description='The state type.')
    state_name: Any = Field(default=..., description='The state name.')
    count_runs: Any = Field(default=..., description='The number of runs in the specified state during the interval.')
    sum_estimated_run_time: Any = Field(default=..., description='The total estimated run time of all runs during the interval.')
    sum_estimated_lateness: Any = Field(default=..., description='The sum of differences between actual and expected start time during the interval.')

class HistoryResponse(PrefectBaseModel):
    interval_start: Any = Field(default=..., description='The start date of the interval.')
    interval_end: Any = Field(default=..., description='The end date of the interval.')
    states: Any = Field(default=..., description='A list of state histories during the interval.')

StateResponseDetails = Union[StateAcceptDetails, StateWaitDetails, StateRejectDetails, StateAbortDetails]

class OrchestrationResult(PrefectBaseModel):
    pass

class WorkerFlowRunResponse(PrefectBaseModel):
    model_config: Dict[str, Any] = Field(default={}, description='Model configuration for worker flow run response.')

class FlowRunResponse(ORMBaseModel):
    name: str = Field(default_factory=lambda: generate_slug(2), description='The name of the flow run. Defaults to a random slug if not specified.', examples=['my-flow-run'])
    flow_id: Any = Field(default=..., description='The id of the flow being run.')
    state_id: Optional[Any] = Field(default=None, description="The id of the flow run's current state.")
    deployment_id: Optional[Any] = Field(default=None, description='The id of the deployment associated with this flow run, if available.')
    deployment_version: Optional[str] = Field(default=None, description='The version of the deployment associated with this flow run.', examples=['1.0'])
    work_queue_id: Optional[Any] = Field(default=None, description="The id of the run's work pool queue.")
    work_queue_name: Optional[str] = Field(default=None, description='The work queue that handled this flow run.')
    flow_version: Optional[str] = Field(default=None, description='The version of the flow executed in this flow run.', examples=['1.0'])
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for the flow run.')
    idempotency_key: Optional[str] = Field(default=None, description='An optional idempotency key for the flow run. Used to ensure the same flow run is not created multiple times.')
    context: Dict[str, Any] = Field(default_factory=dict, description='Additional context for the flow run.', examples=[{'my_var': 'my_val'}])
    empirical_policy: FlowRunPolicy = Field(default_factory=FlowRunPolicy)
    tags: List[str] = Field(default_factory=list, description='A list of tags on the flow run', examples=[['tag-1', 'tag-2']])
    parent_task_run_id: Optional[Any] = Field(default=None, description="If the flow run is a subflow, the id of the 'dummy' task in the parent flow used to track subflow state.")
    state_type: Optional[str] = Field(default=None, description='The type of the current flow run state.')
    state_name: Optional[str] = Field(default=None, description='The name of the current flow run state.')
    run_count: int = Field(default=0, description='The number of times the flow run was executed.')
    expected_start_time: Optional[DateTime] = Field(default=None, description="The flow run's expected start time.")
    next_scheduled_start_time: Optional[DateTime] = Field(default=None, description='The next time the flow run is scheduled to start.')
    start_time: Optional[DateTime] = Field(default=None, description='The actual start time.')
    end_time: Optional[DateTime] = Field(default=None, description='The actual end time.')
    total_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='Total run time. If the flow run was executed multiple times, the time of each run will be summed.')
    estimated_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='A real-time estimate of the total run time.')
    estimated_start_time_delta: datetime.timedelta = Field(default=datetime.timedelta(0), description='The difference between actual and expected start time.')
    auto_scheduled: bool = Field(default=False, description='Whether or not the flow run was automatically scheduled.')
    infrastructure_document_id: Optional[Any] = Field(default=None, description='The block document defining infrastructure to use this flow run.')
    infrastructure_pid: Optional[Any] = Field(default=None, description='The id of the flow run as returned by an infrastructure block.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this flow run.')
    work_pool_id: Optional[Any] = Field(default=None, description="The id of the flow run's work pool.")
    work_pool_name: Optional[str] = Field(default=None, description="The name of the flow run's work pool.", examples=['my-work-pool'])
    state: Optional[StateResponseDetails] = Field(default=None, description='The current state of the flow run.')
    job_variables: Optional[Dict[str, Any]] = Field(default=None, description='Variables used as overrides in the base job template')

class DeploymentResponse(ORMBaseModel):
    name: str = Field(default=..., description='The name of the deployment.')
    version: Optional[str] = Field(default=None, description='An optional version for the deployment.')
    description: Optional[str] = Field(default=None, description='A description for the deployment.')
    flow_id: Any = Field(default=..., description='The flow id associated with the deployment.')
    paused: bool = Field(default=False, description='Whether or not the deployment is paused.')
    schedules: List[Any] = Field(default_factory=list, description='A list of schedules for the deployment.')
    concurrency_limit: Optional[Any] = Field(default=None, description='DEPRECATED: Prefer `global_concurrency_limit`. Will always be None for backwards compatibility. Will be removed after December 2024.', deprecated=True)
    global_concurrency_limit: Optional[Any] = Field(default=None, description='The global concurrency limit object for enforcing the maximum number of flow runs that can be active at once.')
    concurrency_options: Optional[Any] = Field(default=None, description='The concurrency options for the deployment.')
    job_variables: Dict[str, Any] = Field(default_factory=dict, description='Overrides to apply to the base infrastructure block at runtime.')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for flow runs scheduled by the deployment.')
    tags: List[str] = Field(default_factory=list, description='A list of tags for the deployment', examples=[['tag-1', 'tag-2']])
    work_queue_name: Optional[str] = Field(default=None, description='The work queue for the deployment. If no work queue is set, work will not be scheduled.')
    last_polled: Optional[DateTime] = Field(default=None, description='The last time the deployment was polled for status updates.')
    parameter_openapi_schema: Optional[Any] = Field(default=None, description='The parameter schema of the flow, including defaults.', json_schema_extra={'additionalProperties': True})
    path: Optional[Any] = Field(default=None, description='The path to the working directory for the workflow, relative to remote storage or an absolute path.')
    pull_steps: Optional[Any] = Field(default=None, description='Pull steps for cloning and running this deployment.')
    entrypoint: Optional[Any] = Field(default=None, description='The path to the entrypoint for the workflow, relative to the `path`.')
    storage_document_id: Optional[Any] = Field(default=None, description='The block document defining storage used for this flow.')
    infrastructure_document_id: Optional[Any] = Field(default=None, description='The block document defining infrastructure to use for flow runs.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this deployment.')
    updated_by: Optional[UpdatedBy] = Field(default=None, description='Optional information about the updater of this deployment.')
    work_pool_name: Optional[str] = Field(default=None, description="The name of the deployment's work pool.")
    status: schemas.statuses.DeploymentStatus = Field(default=schemas.statuses.DeploymentStatus.NOT_READY, description='Whether the deployment is ready to run flows.')
    enforce_parameter_schema: bool = Field(default=True, description='Whether or not the deployment should enforce the parameter schema.')

class WorkQueueResponse(schemas.core.WorkQueue):
    work_pool_name: Optional[str] = Field(default=None, description='The name of the work pool the work pool resides within.')
    status: Optional[Any] = Field(default=None, description='The queue status.')

class WorkQueueWithStatus(WorkQueueResponse, WorkQueueStatusDetail):
    pass

class WorkerResponse(schemas.core.Worker):
    status: schemas.statuses.WorkerStatus = Field(schemas.statuses.WorkerStatus.OFFLINE, description='Current status of the worker.')

class GlobalConcurrencyLimitResponse(ORMBaseModel):
    active: bool = Field(default=True, description='Whether the global concurrency limit is active.')
    name: str = Field(default=..., description='The name of the global concurrency limit.')
    limit: Any = Field(default=..., description='The concurrency limit.')
    active_slots: Any = Field(default=..., description='The number of active slots.')
    slot_decay_per_second: float = Field(default=2.0, description='The decay rate for active slots when used as a rate limit.')

class FlowPaginationResponse(BaseModel):
    pass

class FlowRunPaginationResponse(BaseModel):
    pass

class DeploymentPaginationResponse(BaseModel):
    pass

class SchemaValuePropertyError(BaseModel):
    pass

class SchemaValueIndexError(BaseModel):
    pass

SchemaValueError = Union[str, SchemaValuePropertyError, SchemaValueIndexError]

class SchemaValuesValidationResponse(BaseModel):
    pass
