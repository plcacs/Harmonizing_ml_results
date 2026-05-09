class DeploymentClient(BaseClient):
    async def create_deployment(
        self, 
        flow_id: str, 
        name: str, 
        version: str = None, 
        schedules: list = None, 
        concurrency_limit: int = None, 
        concurrency_options: dict = None, 
        parameters: dict = None, 
        description: str = None, 
        work_queue_name: str = None, 
        work_pool_name: str = None, 
        tags: list = None, 
        storage_document_id: str = None, 
        path: str = None, 
        entrypoint: str = None, 
        infrastructure_document_id: str = None, 
        parameter_openapi_schema: dict = None, 
        paused: bool = None, 
        pull_steps: list = None, 
        enforce_parameter_schema: bool = None, 
        job_variables: dict = None
    ) -> UUID:
        ...

    async def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None:
        ...

    async def update_deployment(self, deployment_id: UUID, deployment: DeploymentUpdate) -> None:
        ...

    async def read_deployment(self, deployment_id: UUID) -> Deployment:
        ...

    async def read_deployment_by_name(self, name: str) -> Deployment:
        ...

    async def read_deployments(
        self, 
        *, 
        flow_filter: FlowFilter = None, 
        flow_run_filter: FlowRunFilter = None, 
        task_run_filter: TaskRunFilter = None, 
        deployment_filter: DeploymentFilter = None, 
        work_pool_filter: WorkPoolFilter = None, 
        work_queue_filter: WorkQueueFilter = None, 
        limit: int = None, 
        sort: DeploymentSort = None, 
        offset: int = 0
    ) -> list[Deployment]:
        ...

    async def delete_deployment(self, deployment_id: UUID) -> None:
        ...

    async def create_deployment_schedules(self, deployment_id: UUID, schedules: list) -> list[DeploymentSchedule]:
        ...

    async def read_deployment_schedules(self, deployment_id: UUID) -> list[DeploymentSchedule]:
        ...

    async def update_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID, active: bool = None, schedule: str = None) -> None:
        ...

    async def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None:
        ...

    async def get_scheduled_flow_runs_for_deployments(
        self, 
        deployment_ids: list[UUID], 
        scheduled_before: datetime = None, 
        limit: int = None
    ) -> list[FlowRun]:
        ...

    async def create_flow_run_from_deployment(
        self, 
        deployment_id: UUID, 
        *, 
        parameters: dict = None, 
        context: dict = None, 
        state: State = None, 
        name: str = None, 
        tags: list = None, 
        idempotency_key: str = None, 
        parent_task_run_id: UUID = None, 
        work_queue_name: str = None, 
        job_variables: dict = None, 
        labels: KeyValueLabelsField = None
    ) -> FlowRun:
        ...
