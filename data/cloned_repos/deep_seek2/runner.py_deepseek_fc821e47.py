from typing import List, Optional, Union, Dict, Any, Iterable
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from prefect.client.schemas.actions import DeploymentScheduleCreate, DeploymentUpdate
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyOptions
from prefect.client.schemas.schedules import SCHEDULE_TYPES, construct_schedule
from prefect.deployments.schedules import create_deployment_schedule_create
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import PREFECT_DEFAULT_WORK_POOL_NAME, PREFECT_UI_URL
from prefect.types import ListOfNonEmptyStrings
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.callables import ParameterSchema, parameter_schema
from prefect.utilities.collections import get_from_dict, isiterable
from prefect.utilities.dockerutils import parse_image_tag

class DeploymentApplyError(RuntimeError):
    pass

class RunnerDeployment(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="The name of the deployment.")
    flow_name: Optional[str] = Field(None, description="The name of the underlying flow; typically inferred.")
    description: Optional[str] = Field(default=None, description="An optional description of the deployment.")
    version: Optional[str] = Field(default=None, description="An optional version for the deployment.")
    tags: ListOfNonEmptyStrings = Field(default_factory=list, description="One of more tags to apply to this deployment.")
    schedules: Optional[List[DeploymentScheduleCreate]] = Field(default=None, description="The schedules that should cause this deployment to run.")
    concurrency_limit: Optional[int] = Field(default=None, description="The maximum number of concurrent runs of this deployment.")
    concurrency_options: Optional[ConcurrencyOptions] = Field(default=None, description="The concurrency limit config for the deployment.")
    paused: Optional[bool] = Field(default=None, description="Whether or not the deployment is paused.")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    entrypoint: Optional[str] = Field(default=None, description="The path to the entrypoint for the workflow, relative to the `path`.")
    triggers: List[Union[DeploymentTriggerTypes, TriggerTypes]] = Field(default_factory=list, description="The triggers that should cause this deployment to run.")
    enforce_parameter_schema: bool = Field(default=True, description="Whether or not the Prefect API should enforce the parameter schema for this deployment.")
    storage: Optional[RunnerStorage] = Field(default=None, description="The storage object used to retrieve flow code for this deployment.")
    work_pool_name: Optional[str] = Field(default=None, description="The name of the work pool to use for this deployment. Only used when the deployment is registered with a built runner.")
    work_queue_name: Optional[str] = Field(default=None, description="The name of the work queue to use for this deployment. Only used when the deployment is registered with a built runner.")
    job_variables: Dict[str, Any] = Field(default_factory=dict, description="Job variables used to override the default values of a work pool base job template. Only used when the deployment is registered with a built runner.")
    _sla: Optional[Union[SlaTypes, List[SlaTypes]]] = PrivateAttr(default=None)
    _entrypoint_type: EntrypointType = PrivateAttr(default=EntrypointType.FILE_PATH)
    _path: Optional[str] = PrivateAttr(default=None)
    _parameter_openapi_schema: ParameterSchema = PrivateAttr(default_factory=ParameterSchema)

    @property
    def entrypoint_type(self) -> EntrypointType:
        return self._entrypoint_type

    @property
    def full_name(self) -> str:
        return f"{self.flow_name}/{self.name}"

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value.endswith(".py"):
            return Path(value).stem
        return value

    @model_validator(mode="after")
    def validate_automation_names(self) -> "RunnerDeployment":
        for i, trigger in enumerate(self.triggers, start=1):
            if trigger.name is None:
                trigger.name = f"{self.name}__automation_{i}"
        return self

    @model_validator(mode="before")
    @classmethod
    def reconcile_paused(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return reconcile_paused_deployment(values)

    @model_validator(mode="before")
    @classmethod
    def reconcile_schedules(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return reconcile_schedules_runner(values)

    async def _create(self, work_pool_name: Optional[str] = None, image: Optional[str] = None) -> UUID:
        work_pool_name = work_pool_name or self.work_pool_name

        if image and not work_pool_name:
            raise ValueError("An image can only be provided when registering a deployment with a work pool.")

        if self.work_queue_name and not work_pool_name:
            raise ValueError("A work queue can only be provided when registering a deployment with a work pool.")

        if self.job_variables and not work_pool_name:
            raise ValueError("Job variables can only be provided when registering a deployment with a work pool.")

        async with get_client() as client:
            flow_id = await client.create_flow_from_name(self.flow_name)

            create_payload: Dict[str, Any] = dict(
                flow_id=flow_id,
                name=self.name,
                work_queue_name=self.work_queue_name,
                work_pool_name=work_pool_name,
                version=self.version,
                paused=self.paused,
                schedules=self.schedules,
                concurrency_limit=self.concurrency_limit,
                concurrency_options=self.concurrency_options,
                parameters=self.parameters,
                description=self.description,
                tags=self.tags,
                path=self._path,
                entrypoint=self.entrypoint,
                storage_document_id=None,
                infrastructure_document_id=None,
                parameter_openapi_schema=self._parameter_openapi_schema.model_dump(exclude_unset=True),
                enforce_parameter_schema=self.enforce_parameter_schema,
            )

            if work_pool_name:
                create_payload["job_variables"] = self.job_variables
                if image:
                    create_payload["job_variables"]["image"] = image
                create_payload["path"] = None if self.storage else self._path
                if self.storage:
                    pull_steps = self.storage.to_pull_step()
                    if isinstance(pull_steps, list):
                        create_payload["pull_steps"] = pull_steps
                    else:
                        create_payload["pull_steps"] = [pull_steps]
                else:
                    create_payload["pull_steps"] = []

            try:
                deployment_id = await client.create_deployment(**create_payload)
            except Exception as exc:
                if isinstance(exc, PrefectHTTPStatusError):
                    detail = exc.response.json().get("detail")
                    if detail:
                        raise DeploymentApplyError(detail) from exc
                raise DeploymentApplyError(f"Error while applying deployment: {str(exc)}") from exc

            await self._create_triggers(deployment_id, client)

            if self._sla or self._sla == []:
                await self._create_slas(deployment_id, client)

            return deployment_id

    async def _update(self, deployment_id: UUID, client: PrefectClient) -> UUID:
        parameter_openapi_schema = self._parameter_openapi_schema.model_dump(exclude_unset=True)
        await client.update_deployment(
            deployment_id,
            deployment=DeploymentUpdate(
                parameter_openapi_schema=parameter_openapi_schema,
                **self.model_dump(mode="json", exclude_unset=True, exclude={"storage", "name", "flow_name", "triggers"}),
            ),
        )

        await self._create_triggers(deployment_id, client)

        if self._sla or self._sla == []:
            await self._create_slas(deployment_id, client)

        return deployment_id

    async def _create_triggers(self, deployment_id: UUID, client: PrefectClient) -> UUID:
        try:
            await client.delete_resource_owned_automations(f"prefect.deployment.{deployment_id}")
        except PrefectHTTPStatusError as e:
            if e.response.status_code == 404:
                return deployment_id
            raise e

        for trigger in self.triggers:
            trigger.set_deployment_id(deployment_id)
            await client.create_automation(trigger.as_automation())

    @sync_compatible
    async def apply(self, work_pool_name: Optional[str] = None, image: Optional[str] = None) -> UUID:
        async with get_client() as client:
            try:
                deployment = await client.read_deployment_by_name(self.full_name)
            except ObjectNotFound:
                return await self._create(work_pool_name, image)
            else:
                if image:
                    self.job_variables["image"] = image
                if work_pool_name:
                    self.work_pool_name = work_pool_name
                return await self._update(deployment.id, client)

    async def _create_slas(self, deployment_id: UUID, client: PrefectClient) -> None:
        if not isinstance(self._sla, list):
            self._sla = [self._sla]

        if client.server_type == ServerType.CLOUD:
            await client.apply_slas_for_deployment(deployment_id, self._sla)
        else:
            raise ValueError("SLA configuration is currently only supported on Prefect Cloud.")

    @staticmethod
    def _construct_deployment_schedules(
        interval: Optional[Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]] = None,
        anchor_date: Optional[Union[datetime, str]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        timezone: Optional[str] = None,
        schedule: Union[SCHEDULE_TYPES, Schedule, None] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
    ) -> Union[List[DeploymentScheduleCreate], "FlexibleScheduleList"]:
        num_schedules = sum(1 for entry in (interval, cron, rrule, schedule, schedules) if entry is not None)
        if num_schedules > 1:
            raise ValueError("Only one of interval, cron, rrule, schedule, or schedules can be provided.")
        elif num_schedules == 0:
            return []

        if schedules is not None:
            return schedules
        elif interval or cron or rrule:
            parameters = [("interval", interval), ("cron", cron), ("rrule", rrule)]
            schedule_type, value = [param for param in parameters if param[1] is not None][0]

            if not isiterable(value):
                value = [value]

            return [
                create_deployment_schedule_create(
                    construct_schedule(
                        **{
                            schedule_type: v,
                            "timezone": timezone,
                            "anchor_date": anchor_date,
                        }
                    )
                )
                for v in value
            ]
        else:
            return [create_deployment_schedule_create(schedule)]

    def _set_defaults_from_flow(self, flow: "Flow[..., Any]") -> None:
        self._parameter_openapi_schema = parameter_schema(flow)

        if not self.version:
            self.version = flow.version
        if not self.description:
            self.description = flow.description

    @classmethod
    def from_flow(
        cls,
        flow: "Flow[..., Any]",
        name: str,
        interval: Optional[Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        entrypoint_type: EntrypointType = EntrypointType.FILE_PATH,
        _sla: Optional[Union[SlaTypes, List[SlaTypes]]] = None,
    ) -> "RunnerDeployment":
        constructed_schedules = cls._construct_deployment_schedules(
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedules=schedules,
            schedule=schedule,
        )

        job_variables = job_variables or {}

        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {"collision_strategy": concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None

        deployment = cls(
            name=name,
            flow_name=flow.name,
            schedules=constructed_schedules,
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            paused=paused,
            tags=tags or [],
            triggers=triggers or [],
            parameters=parameters or {},
            description=description,
            version=version,
            enforce_parameter_schema=enforce_parameter_schema,
            work_pool_name=work_pool_name,
            work_queue_name=work_queue_name,
            job_variables=job_variables,
        )
        deployment._sla = _sla

        if not deployment.entrypoint:
            no_file_location_error = (
                "Flows defined interactively cannot be deployed. Check out the"
                " quickstart guide for help getting started:"
                " https://docs.prefect.io/latest/get-started/quickstart"
            )
            flow_file = getattr(flow, "__globals__", {}).get("__file__")
            mod_name = getattr(flow, "__module__", None)
            if entrypoint_type == EntrypointType.MODULE_PATH:
                if mod_name:
                    deployment.entrypoint = f"{mod_name}.{flow.__name__}"
                else:
                    raise ValueError("Unable to determine module path for provided flow.")
            else:
                if not flow_file:
                    if not mod_name:
                        raise ValueError(no_file_location_error)
                    try:
                        module = importlib.import_module(mod_name)
                        flow_file = getattr(module, "__file__", None)
                    except ModuleNotFoundError as exc:
                        if "__prefect_loader_" in str(exc):
                            raise ValueError(
                                "Cannot create a RunnerDeployment from a flow that has been"
                                " loaded from an entrypoint. To deploy a flow via"
                                " entrypoint, use RunnerDeployment.from_entrypoint instead."
                            )
                        raise ValueError(no_file_location_error)
                    if not flow_file:
                        raise ValueError(no_file_location_error)

                entry_path = Path(flow_file).absolute().relative_to(Path.cwd().absolute())
                deployment.entrypoint = f"{entry_path}:{flow.fn.__name__}"

        if entrypoint_type == EntrypointType.FILE_PATH and not deployment._path:
            deployment._path = "."

        deployment._entrypoint_type = entrypoint_type

        cls._set_defaults_from_flow(deployment, flow)

        return deployment

    @classmethod
    def from_entrypoint(
        cls,
        entrypoint: str,
        name: str,
        flow_name: Optional[str] = None,
        interval: Optional[Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        _sla: Optional[Union[SlaTypes, List[SlaTypes]]] = None,
    ) -> "RunnerDeployment":
        from prefect.flows import load_flow_from_entrypoint

        job_variables = job_variables or {}
        flow = load_flow_from_entrypoint(entrypoint)

        constructed_schedules = cls._construct_deployment_schedules(
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedules=schedules,
            schedule=schedule,
        )

        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {"collision_strategy": concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None

        deployment = cls(
            name=Path(name).stem,
            flow_name=flow_name or flow.name,
            schedules=constructed_schedules,
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            paused=paused,
            tags=tags or [],
            triggers=triggers or [],
            parameters=parameters or {},
            description=description,
            version=version,
            entrypoint=entrypoint,
            enforce_parameter_schema=enforce_parameter_schema,
            work_pool_name=work_pool_name,
            work_queue_name=work_queue_name,
            job_variables=job_variables,
        )
        deployment._sla = _sla
        deployment._path = str(Path.cwd())

        cls._set_defaults_from_flow(deployment, flow)

        return deployment

    @classmethod
    async def afrom_storage(
        cls,
        storage: RunnerStorage,
        entrypoint: str,
        name: str,
        flow_name: Optional[str] = None,
        interval: Optional[Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        _sla: Optional[Union[SlaTypes, List[SlaTypes]]] = None,
    ) -> "RunnerDeployment":
        from prefect.flows import load_flow_from_entrypoint

        constructed_schedules = cls._construct_deployment_schedules(
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedules=schedules,
            schedule=schedule,
        )

        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {"collision_strategy": concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None

        job_variables = job_variables or {}

        with tempfile.TemporaryDirectory() as tmpdir:
            storage.set_base_path(Path(tmpdir))
            await storage.pull_code()

            full_entrypoint = str(storage.destination / entrypoint)
            flow = await from_async.wait_for_call_in_new_thread(
                create_call(load_flow_from_entrypoint, full_entrypoint)
            )

        deployment = cls(
            name=Path(name).stem,
            flow_name=flow_name or flow.name,
            schedules=constructed_schedules,
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            paused=paused,
            tags=tags or [],
            triggers=triggers or [],
            parameters=parameters or {},
            description=description,
            version=version,
            entrypoint=entrypoint,
            enforce_parameter_schema=enforce_parameter_schema,
            storage=storage,
            work_pool_name=work_pool_name,
            work_queue_name=work_queue_name,
            job_variables=job_variables,
        )
        deployment._sla = _sla
        deployment._path = str(storage.destination).replace(tmpdir, "$STORAGE_BASE_PATH")

        cls._set_defaults_from_flow(deployment, flow)

        return deployment

    @classmethod
    @async_dispatch(afrom_storage)
    def from_storage(
        cls,
        storage: RunnerStorage,
        entrypoint: str,
        name: str,
        flow_name: Optional[str] = None,
        interval: Optional[Union[Iterable[Union[int, float, timedelta]], int, float, timedelta]] = None,
        cron: Optional[Union[Iterable[str], str]] = None,
        rrule: Optional[Union[Iterable[str], str]] = None,
        paused: Optional[bool] = None,
        schedule: Optional[Schedule] = None,
        schedules: Optional["FlexibleScheduleList"] = None,
        concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig, None]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        triggers: Optional[List[Union[DeploymentTriggerTypes, TriggerTypes]]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        enforce_parameter_schema: bool = True,
        work_pool_name: Optional[str] = None,
        work_queue_name: Optional[str] = None,
        job_variables: Optional[Dict[str, Any]] = None,
        _sla: Optional[Union[SlaTypes, List[SlaTypes]]] = None,
    ) -> "RunnerDeployment":
        from prefect.flows import load_flow_from_entrypoint

        constructed_schedules = cls._construct_deployment_schedules(
            interval=interval,
            cron=cron,
            rrule=rrule,
            schedules=schedules,
            schedule=schedule,
        )

        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {"collision_strategy": concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None

        job_variables = job_variables or {}

        with tempfile.TemporaryDirectory() as tmpdir:
            storage.set_base_path(Path(tmpdir))
            run_coro_as_sync(storage.pull_code())

            full_entrypoint = str(storage.destination / entrypoint)
            flow = load_flow_from_entrypoint(full_entrypoint)

        deployment = cls(
            name=Path(name).stem,
            flow_name=flow_name or flow.name,
            schedules=constructed_schedules,
            concurrency_limit=concurrency_limit,
            concurrency_options=concurrency_options,
            paused=paused,
            tags=tags or [],
            triggers=triggers or [],
            parameters=parameters or {},
            description=description,
            version=version,
            entrypoint=entrypoint,
            enforce_parameter_schema=enforce_parameter_schema,
            storage=storage,
            work_pool_name=work_pool_name,
            work_queue_name=work_queue_name,
            job_variables=job_variables,
        )
        deployment._sla = _sla
        deployment._path = str(storage.destination).replace(tmpdir, "$STORAGE_BASE_PATH")

        cls._set_defaults_from_flow(deployment, flow)

        return deployment

@sync_compatible
async def deploy(
    *deployments: RunnerDeployment,
    work_pool_name: Optional[str] = None,
    image: Optional[Union[str, DockerImage]] = None,
    build: bool = True,
    push: bool = True,
    print_next_steps_message: bool = True,
    ignore_warnings: bool = False,
) -> List[UUID]:
    work_pool_name = work_pool_name or PREFECT_DEFAULT_WORK_POOL_NAME.value()

    if not image and not all(d.storage or d.entrypoint_type == EntrypointType.MODULE_PATH for d in deployments):
        raise ValueError("Either an image or remote storage location must be provided when deploying a deployment.")

    if not work_pool_name:
        raise ValueError("A work pool name must be provided when deploying a deployment. Either provide a work pool name when calling `deploy` or set `PREFECT_DEFAULT_WORK_POOL_NAME` in your profile.")

    if image and isinstance(image, str):
        image_name, image_tag = parse_image_tag(image)
        image = DockerImage(name=image_name, tag=image_tag)

    try:
        async with get_client() as client:
            work_pool = await client.read_work_pool(work_pool_name)
            active_workers = await client.read_workers_for_work_pool(
                work_pool_name,
                worker_filter=WorkerFilter(status=WorkerFilterStatus(any_=["ONLINE"])),
            )
    except ObjectNotFound as exc:
        raise ValueError(f"Could not find work pool {work_pool_name!r}. Please create it before deploying this flow.") from exc

    is_docker_based_work_pool = get_from_dict(work_pool.base_job_template, "variables.properties.image", False)
    is_block_based_work_pool = get_from_dict(work_pool.base_job_template, "variables.properties.block", False)
    console = Console()
    if not is_docker_based_work_pool and not is_block_based_work_pool:
        if image:
            raise ValueError(f"Work pool {work_pool_name!r} does not support custom Docker images. Please use a work pool with an `image` variable in its base job template or specify a remote storage location for the flow with `.from_source`. If you are attempting to deploy a flow to a local process work pool, consider using `flow.serve` instead. See the documentation for more information: https://docs.prefect.io/latest/deploy/run-flows-in-local-processes")
        elif work_pool.type == "process" and not ignore_warnings:
            console.print(
                "Looks like you're deploying to a process work pool. If you're creating a deployment for local development, calling `.serve` on your flow is a great way to get started. See the documentation for more information: https://docs.prefect.io/latest/deploy/run-flows-in-local-processes Set `ignore_warnings=True` to suppress this message.",
                style="yellow",
            )

    is_managed_pool = work_pool.is_managed_pool
    if is_managed_pool:
        build = False
        push = False

    if image and build:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"Building image {image.reference}..."),
            transient=True,
            console=console,
        ) as progress:
            docker_build_task = progress.add_task("docker_build", total=1)
            image.build()

            progress.update(docker_build_task, completed=1)
            console.print(f"Successfully built image {image.reference!r}", style="green")

    if image and build and push:
        with Progress(
            SpinnerColumn(),
            TextColumn("Pushing image..."),
            transient=True,
            console=console,
        ) as progress:
            docker_push_task = progress.add_task("docker_push", total=1)

            image.push()

            progress.update(docker_push_task, completed=1)

        console.print(f"Successfully pushed image {image.reference!r}", style="green")

    deployment_exceptions: List[Dict[str, Any]] = []
    deployment_ids: List[UUID] = []
    image_ref = image.reference if image else None
    for deployment in track(
        deployments,
        description="Creating/updating deployments...",
        console=console,
        transient=True,
    ):
        try:
            deployment_ids.append(await deployment.apply(image=image_ref, work_pool_name=work_pool_name))
        except Exception as exc:
            if len(deployments) == 1:
                raise
            deployment_exceptions.append({"deployment": deployment, "exc": exc})

    if deployment_exceptions:
        console.print("Encountered errors while creating/updating deployments:\n", style="orange_red1")
    else:
        console.print("Successfully created/updated all deployments!\n", style="green")

    complete_failure = len(deployment_exceptions) == len(deployments)

    table = Table(
        title="Deployments",
        show_lines=True,
    )

    table.add_column(header="Name", style="blue", no_wrap=True)
    table.add_column(header="Status", style="blue", no_wrap=True)
    table.add_column(header="Details", style="blue")

    for deployment in deployments:
        errored_deployment = next((d for d in deployment_exceptions if d["deployment"] == deployment), None)
        if errored_deployment:
            table.add_row(
                f"{deployment.flow_name}/{deployment.name}",
                "failed",
                str(errored_deployment["exc"]),
                style="red",
            )
        else:
            table.add_row(f"{deployment.flow_name}/{deployment.name}", "applied")
    console.print(table)

    if print_next_steps_message and not complete_failure:
        if not work_pool.is_push_pool and not work_pool.is_managed_pool and not active_workers:
            console.print(
                "\nTo execute flow runs from these deployments, start a worker in a"
                " separate terminal that pulls work from the"
                f" {work_pool_name!r} work pool:"
                f"\n\t[blue]$ prefect worker start --pool {work_pool_name!r}[/]",
            )
        console.print(
            "\nTo trigger any of these deployments, use the"
            " following command:\n[blue]\n\t$ prefect deployment run"
            " [DEPLOYMENT_NAME]\n[/]"
        )

        if PREFECT_UI_URL:
            console.print(
                "\nYou can also trigger your deployments via the Prefect UI:"
                f" [blue]{PREFECT_UI_URL.value()}/deployments[/]\n"
            )

    return deployment_ids
