class DeploymentApplyError(RuntimeError):
    """
    Raised when an error occurs while applying a deployment.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)

class RunnerDeployment(BaseModel):
    """
    A Prefect RunnerDeployment definition, used for specifying and building deployments.

    Attributes:
        name: A name for the deployment (required).
        version: An optional version for the deployment; defaults to the flow's version
        description: An optional description of the deployment; defaults to the flow's
            description
        tags: An optional list of tags to associate with this deployment; note that tags
            are used only for organizational purposes. For delegating work to workers,
            see `work_queue_name`.
        schedule: A schedule to run this deployment on, once registered
        parameters: A dictionary of parameter values to pass to runs created from this
            deployment
        path: The path to the working directory for the workflow, relative to remote
            storage or, if stored on a local filesystem, an absolute path
        entrypoint: The path to the entrypoint for the workflow, always relative to the
            `path`
        parameter_openapi_schema: The parameter schema of the flow, including defaults.
        enforce_parameter_schema: Whether or not the Prefect API should enforce the
            parameter schema for this deployment.
        work_pool_name: The name of the work pool to use for this deployment.
        work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
            If not provided the default work queue for the work pool will be used.
        job_variables: Settings used to override the values specified default base job template
            of the chosen work pool. Refer to the base job template of the chosen work pool for
            available settings.
        _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(..., description='The name of the deployment.')
    flow_name: str = Field(None, description='The name of the underlying flow; typically inferred.')
    description: str = Field(default=None, description='An optional description of the deployment.')
    version: str = Field(default=None, description='An optional version for the deployment.')
    tags: List[str] = Field(default_factory=list, description='One of more tags to apply to this deployment.')
    schedules: List[Any] = Field(default=None, description='The schedules that should cause this deployment to run.')
    concurrency_limit: int = Field(default=None, description='The maximum number of concurrent runs of this deployment.')
    concurrency_options: Any = Field(default=None, description='The concurrency limit config for the deployment.')
    paused: bool = Field(default=None, description='Whether or not the deployment is paused.')
    parameters: dict = Field(default_factory=dict)
    entrypoint: str = Field(default=None, description='The path to the entrypoint for the workflow, relative to the `path`.')
    triggers: List[Any] = Field(default_factory=list, description='The triggers that should cause this deployment to run.')
    enforce_parameter_schema: bool = Field(default=True, description='Whether or not the Prefect API should enforce the parameter schema for this deployment.')
    storage: Any = Field(default=None, description='The storage object used to retrieve flow code for this deployment.')
    work_pool_name: str = Field(default=None, description='The name of the work pool to use for this deployment. Only used when the deployment is registered with a built runner.')
    work_queue_name: str = Field(default=None, description='The name of the work queue to use for this deployment. Only used when the deployment is registered with a built runner.')
    job_variables: dict = Field(default_factory=dict, description='Job variables used to override the default values of a work pool base job template. Only used when the deployment is registered with a built runner.')
    _sla: Any = PrivateAttr(default=None)
    _entrypoint_type: EntrypointType = PrivateAttr(default=EntrypointType.FILE_PATH)
    _path: str = PrivateAttr(default=None)
    _parameter_openapi_schema: ParameterSchema = PrivateAttr(default_factory=ParameterSchema)

    @property
    def entrypoint_type(self) -> EntrypointType:
        return self._entrypoint_type

    @property
    def full_name(self) -> str:
        return f'{self.flow_name}/{self.name}'

    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value.endswith('.py'):
            return Path(value).stem
        return value

    @model_validator(mode='after')
    def validate_automation_names(self) -> 'RunnerDeployment':
        """Ensure that each trigger has a name for its automation if none is provided."""
        for i, trigger in enumerate(self.triggers, start=1):
            if trigger.name is None:
                trigger.name = f'{self.name}__automation_{i}'
        return self

    @model_validator(mode='before')
    @classmethod
    def reconcile_paused(cls, values: dict) -> dict:
        return reconcile_paused_deployment(values)

    @model_validator(mode='before')
    @classmethod
    def reconcile_schedules(cls, values: dict) -> dict:
        return reconcile_schedules_runner(values)

    async def _create(self, work_pool_name: str = None, image: str = None) -> UUID:
        work_pool_name = work_pool_name or self.work_pool_name
        if image and (not work_pool_name):
            raise ValueError('An image can only be provided when registering a deployment with a work pool.')
        if self.work_queue_name and (not work_pool_name):
            raise ValueError('A work queue can only be provided when registering a deployment with a work pool.')
        if self.job_variables and (not work_pool_name):
            raise ValueError('Job variables can only be provided when registering a deployment with a work pool.')
        async with get_client() as client:
            flow_id = await client.create_flow_from_name(self.flow_name)
            create_payload = dict(flow_id=flow_id, name=self.name, work_queue_name=self.work_queue_name, work_pool_name=work_pool_name, version=self.version, paused=self.paused, schedules=self.schedules, concurrency_limit=self.concurrency_limit, concurrency_options=self.concurrency_options, parameters=self.parameters, description=self.description, tags=self.tags, path=self._path, entrypoint=self.entrypoint, storage_document_id=None, infrastructure_document_id=None, parameter_openapi_schema=self._parameter_openapi_schema.model_dump(exclude_unset=True), enforce_parameter_schema=self.enforce_parameter_schema)
            if work_pool_name:
                create_payload['job_variables'] = self.job_variables
                if image:
                    create_payload['job_variables']['image'] = image
                create_payload['path'] = None if self.storage else self._path
                if self.storage:
                    pull_steps = self.storage.to_pull_step()
                    if isinstance(pull_steps, list):
                        create_payload['pull_steps'] = pull_steps
                    else:
                        create_payload['pull_steps'] = [pull_steps]
                else:
                    create_payload['pull_steps'] = []
            try:
                deployment_id = await client.create_deployment(**create_payload)
            except Exception as exc:
                if isinstance(exc, PrefectHTTPStatusError):
                    detail = exc.response.json().get('detail')
                    if detail:
                        raise DeploymentApplyError(detail) from exc
                raise DeploymentApplyError(f'Error while applying deployment: {str(exc)}') from exc
            await self._create_triggers(deployment_id, client)
            if self._sla or self._sla == []:
                await self._create_slas(deployment_id, client)
            return deployment_id

    async def _update(self, deployment_id: UUID, client: PrefectClient) -> UUID:
        parameter_openapi_schema = self._parameter_openapi_schema.model_dump(exclude_unset=True)
        await client.update_deployment(deployment_id, deployment=DeploymentUpdate(parameter_openapi_schema=parameter_openapi_schema, **self.model_dump(mode='json', exclude_unset=True, exclude={'storage', 'name', 'flow_name', 'triggers'})))
        await self._create_triggers(deployment_id, client)
        if self._sla or self._sla == []:
            await self._create_slas(deployment_id, client)
        return deployment_id

    async def _create_triggers(self, deployment_id: UUID, client: PrefectClient) -> None:
        try:
            await client.delete_resource_owned_automations(f'prefect.deployment.{deployment_id}')
        except PrefectHTTPStatusError as e:
            if e.response.status_code == 404:
                return
            raise e
        for trigger in self.triggers:
            trigger.set_deployment_id(deployment_id)
            await client.create_automation(trigger.as_automation())

    @sync_compatible
    async def apply(self, work_pool_name: str = None, image: str = None) -> UUID:
        """
        Registers this deployment with the API and returns the deployment's ID.

        Args:
            work_pool_name: The name of the work pool to use for this
                deployment.
            image: The registry, name, and tag of the Docker image to
                use for this deployment. Only used when the deployment is
                deployed to a work pool.

        Returns:
            The ID of the created deployment.
        """
        async with get_client() as client:
            try:
                deployment = await client.read_deployment_by_name(self.full_name)
            except ObjectNotFound:
                return await self._create(work_pool_name, image)
            else:
                if image:
                    self.job_variables['image'] = image
                if work_pool_name:
                    self.work_pool_name = work_pool_name
                return await self._update(deployment.id, client)

    async def _create_slas(self, deployment_id: UUID, client: PrefectClient) -> None:
        if not isinstance(self._sla, list):
            self._sla = [self._sla]
        if client.server_type == ServerType.CLOUD:
            await client.apply_slas_for_deployment(deployment_id, self._sla)
        else:
            raise ValueError('SLA configuration is currently only supported on Prefect Cloud.')

    @staticmethod
    def _construct_deployment_schedules(interval: Union[int, timedelta, List[Union[int, timedelta]]] = None, anchor_date: datetime = None, cron: Union[str, List[str]] = None, rrule: Union[str, List[str]] = None, timezone: str = None, schedule: Schedule = None, schedules: List[Schedule] = None) -> List[Schedule]:
        """
        Construct a schedule or schedules from the provided arguments.

        This method serves as a unified interface for creating deployment
        schedules. If `schedules` is provided, it is directly returned. If
        `schedule` is provided, it is encapsulated in a list and returned. If
        `interval`, `cron`, or `rrule` are provided, they are used to construct
        schedule objects.

        Args:
            interval: An interval on which to schedule runs, either as a single
              value or as a list of values. Accepts numbers (interpreted as
              seconds) or `timedelta` objects. Each value defines a separate
              scheduling interval.
            anchor_date: The anchor date from which interval schedules should
              start. This applies to all intervals if a list is provided.
            cron: A cron expression or a list of cron expressions defining cron
              schedules. Each expression defines a separate cron schedule.
            rrule: An rrule string or a list of rrule strings for scheduling.
              Each string defines a separate recurrence rule.
            timezone: The timezone to apply to the cron or rrule schedules.
              This is a single value applied uniformly to all schedules.
            schedule: A singular schedule object, used for advanced scheduling
              options like specifying a timezone. This is returned as a list
              containing this single schedule.
            schedules: A pre-defined list of schedule objects. If provided,
              this list is returned as-is, bypassing other schedule construction
              logic.
        """
        num_schedules = sum((1 for entry in (interval, cron, rrule, schedule, schedules) if entry is not None))
        if num_schedules > 1:
            raise ValueError('Only one of interval, cron, rrule, schedule, or schedules can be provided.')
        elif num_schedules == 0:
            return []
        if schedules is not None:
            return schedules
        elif interval or cron or rrule:
            parameters = [('interval', interval), ('cron', cron), ('rrule', rrule)]
            schedule_type, value = [param for param in parameters if param[1] is not None][0]
            if not isiterable(value):
                value = [value]
            return [create_deployment_schedule_create(construct_schedule(**{schedule_type: v, 'timezone': timezone, 'anchor_date': anchor_date})) for v in value]
        else:
            return [create_deployment_schedule_create(schedule)]

    def _set_defaults_from_flow(self, flow: Any) -> None:
        self._parameter_openapi_schema = parameter_schema(flow)
        if not self.version:
            self.version = flow.version
        if not self.description:
            self.description = flow.description

    @classmethod
    def from_flow(cls, flow: Any, name: str, interval: Union[int, timedelta, List[Union[int, timedelta]]] = None, cron: Union[str, List[str]] = None, rrule: Union[str, List[str]] = None, paused: bool = None, schedule: Schedule = None, schedules: List[Schedule] = None, concurrency_limit: int = None, parameters: dict = None, triggers: List[Any] = None, description: str = None, tags: List[str] = None, version: str = None, enforce_parameter_schema: bool = True, work_pool_name: str = None, work_queue_name: str = None, job_variables: dict = None, entrypoint_type: EntrypointType = EntrypointType.FILE_PATH, _sla: Any = None) -> 'RunnerDeployment':
        """
        Configure a deployment for a given flow.

        Args:
            flow: A flow function to deploy
            name: A name for the deployment
            interval: An interval on which to execute the current flow. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this flow.
            rrule: An rrule schedule of when to execute runs of this flow.
            paused: Whether or not to set this deployment as paused.
            schedule: A schedule object defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            schedules: A list of schedule objects defining when to execute runs of this deployment.
                Used to define multiple schedules or additional scheduling options like `timezone`.
            concurrency_limit: The maximum number of concurrent runs this deployment will allow.
            triggers: A list of triggers that should kick of a run of this flow.
            parameters: A dictionary of default parameter values to pass to runs of this flow.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            enforce_parameter_schema: Whether or not the Prefect API should enforce the
                parameter schema for this deployment.
            work_pool_name: The name of the work pool to use for this deployment.
            work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
                If not provided the default work queue for the work pool will be used.
            job_variables: Settings used to override the values specified default base job template
                of the chosen work pool. Refer to the base job template of the chosen work pool for
                available settings.
            _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
        """
        constructed_schedules = cls._construct_deployment_schedules(interval=interval, cron=cron, rrule=rrule, schedules=schedules, schedule=schedule)
        job_variables = job_variables or {}
        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {'collision_strategy': concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None
        deployment = cls(name=name, flow_name=flow.name, schedules=constructed_schedules, concurrency_limit=concurrency_limit, concurrency_options=concurrency_options, paused=paused, tags=tags or [], triggers=triggers or [], parameters=parameters or {}, description=description, version=version, enforce_parameter_schema=enforce_parameter_schema, work_pool_name=work_pool_name, work_queue_name=work_queue_name, job_variables=job_variables)
        deployment._sla = _sla
        if not deployment.entrypoint:
            no_file_location_error = 'Flows defined interactively cannot be deployed. Check out the quickstart guide for help getting started: https://docs.prefect.io/latest/get-started/quickstart'
            flow_file = getattr(flow, '__globals__', {}).get('__file__')
            mod_name = getattr(flow, '__module__', None)
            if entrypoint_type == EntrypointType.MODULE_PATH:
                if mod_name:
                    deployment.entrypoint = f'{mod_name}.{flow.__name__}'
                else:
                    raise ValueError('Unable to determine module path for provided flow.')
            else:
                if not flow_file:
                    if not mod_name:
                        raise ValueError(no_file_location_error)
                    try:
                        module = importlib.import_module(mod_name)
                        flow_file = getattr(module, '__file__', None)
                    except ModuleNotFoundError as exc:
                        if '__prefect_loader_' in str(exc):
                            raise ValueError('Cannot create a RunnerDeployment from a flow that has been loaded from an entrypoint. To deploy a flow via entrypoint, use RunnerDeployment.from_entrypoint instead.')
                        raise ValueError(no_file_location_error)
                    if not flow_file:
                        raise ValueError(no_file_location_error)
                entry_path = Path(flow_file).absolute().relative_to(Path.cwd().absolute())
                deployment.entrypoint = f'{entry_path}:{flow.fn.__name__}'
        if entrypoint_type == EntrypointType.FILE_PATH and (not deployment._path):
            deployment._path = '.'
        deployment._entrypoint_type = entrypoint_type
        cls._set_defaults_from_flow(deployment, flow)
        return deployment

    @classmethod
    def from_entrypoint(cls, entrypoint: str, name: str, flow_name: str = None, interval: Union[int, timedelta, List[Union[int, timedelta]]] = None, cron: Union[str, List[str]] = None, rrule: Union[str, List[str]] = None, paused: bool = None, schedule: Schedule = None, schedules: List[Schedule] = None, concurrency_limit: int = None, parameters: dict = None, triggers: List[Any] = None, description: str = None, tags: List[str] = None, version: str = None, enforce_parameter_schema: bool = True, work_pool_name: str = None, work_queue_name: str = None, job_variables: dict = None, _sla: Any = None) -> 'RunnerDeployment':
        """
        Configure a deployment for a given flow located at a given entrypoint.

        Args:
            entrypoint:  The path to a file containing a flow and the name of the flow function in
                the format `./path/to/file.py:flow_func_name`.
            name: A name for the deployment
            flow_name: The name of the flow to deploy
            interval: An interval on which to execute the current flow. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this flow.
            rrule: An rrule schedule of when to execute runs of this flow.
            paused: Whether or not to set this deployment as paused.
            schedules: A list of schedule objects defining when to execute runs of this deployment.
                Used to define multiple schedules or additional scheduling options like `timezone`.
            triggers: A list of triggers that should kick of a run of this flow.
            parameters: A dictionary of default parameter values to pass to runs of this flow.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            enforce_parameter_schema: Whether or not the Prefect API should enforce the
                parameter schema for this deployment.
            work_pool_name: The name of the work pool to use for this deployment.
            work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
                If not provided the default work queue for the work pool will be used.
            job_variables: Settings used to override the values specified default base job template
                of the chosen work pool. Refer to the base job template of the chosen work pool for
                available settings.
            _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
        """
        from prefect.flows import load_flow_from_entrypoint
        job_variables = job_variables or {}
        flow = load_flow_from_entrypoint(entrypoint)
        constructed_schedules = cls._construct_deployment_schedules(interval=interval, cron=cron, rrule=rrule, schedules=schedules, schedule=schedule)
        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {'collision_strategy': concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None
        deployment = cls(name=Path(name).stem, flow_name=flow_name or flow.name, schedules=constructed_schedules, concurrency_limit=concurrency_limit, concurrency_options=concurrency_options, paused=paused, tags=tags or [], triggers=triggers or [], parameters=parameters or {}, description=description, version=version, entrypoint=entrypoint, enforce_parameter_schema=enforce_parameter_schema, work_pool_name=work_pool_name, work_queue_name=work_queue_name, job_variables=job_variables)
        deployment._sla = _sla
        deployment._path = str(Path.cwd())
        cls._set_defaults_from_flow(deployment, flow)
        return deployment

    @classmethod
    async def afrom_storage(cls, storage: Any, entrypoint: str, name: str, flow_name: str = None, interval: Union[int, timedelta, List[Union[int, timedelta]]] = None, cron: Union[str, List[str]] = None, rrule: Union[str, List[str]] = None, paused: bool = None, schedule: Schedule = None, schedules: List[Schedule] = None, concurrency_limit: int = None, parameters: dict = None, triggers: List[Any] = None, description: str = None, tags: List[str] = None, version: str = None, enforce_parameter_schema: bool = True, work_pool_name: str = None, work_queue_name: str = None, job_variables: dict = None, _sla: Any = None) -> 'RunnerDeployment':
        """
        Create a RunnerDeployment from a flow located at a given entrypoint and stored in a
        local storage location.

        Args:
            entrypoint:  The path to a file containing a flow and the name of the flow function in
                the format `./path/to/file.py:flow_func_name`.
            name: A name for the deployment
            flow_name: The name of the flow to deploy
            storage: A storage object to use for retrieving flow code. If not provided, a
                URL must be provided.
            interval: An interval on which to execute the current flow. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this flow.
            rrule: An rrule schedule of when to execute runs of this flow.
            paused: Whether or not the deployment is paused.
            schedule: A schedule object defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            schedules: A list of schedule objects defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            triggers: A list of triggers that should kick of a run of this flow.
            parameters: A dictionary of default parameter values to pass to runs of this flow.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            enforce_parameter_schema: Whether or not the Prefect API should enforce the
                parameter schema for this deployment.
            work_pool_name: The name of the work pool to use for this deployment.
            work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
                If not provided the default work queue for the work pool will be used.
            job_variables: Settings used to override the values specified default base job template
                of the chosen work pool. Refer to the base job template of the chosen work pool for
                available settings.
            _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
        """
        from prefect.flows import load_flow_from_entrypoint
        constructed_schedules = cls._construct_deployment_schedules(interval=interval, cron=cron, rrule=rrule, schedules=schedules, schedule=schedule)
        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {'collision_strategy': concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None
        job_variables = job_variables or {}
        with tempfile.TemporaryDirectory() as tmpdir:
            storage.set_base_path(Path(tmpdir))
            await storage.pull_code()
            full_entrypoint = str(storage.destination / entrypoint)
            flow = await from_async.wait_for_call_in_new_thread(create_call(load_flow_from_entrypoint, full_entrypoint))
        deployment = cls(name=Path(name).stem, flow_name=flow_name or flow.name, schedules=constructed_schedules, concurrency_limit=concurrency_limit, concurrency_options=concurrency_options, paused=paused, tags=tags or [], triggers=triggers or [], parameters=parameters or {}, description=description, version=version, entrypoint=entrypoint, enforce_parameter_schema=enforce_parameter_schema, storage=storage, work_pool_name=work_pool_name, work_queue_name=work_queue_name, job_variables=job_variables)
        deployment._sla = _sla
        deployment._path = str(storage.destination).replace(tmpdir, '$STORAGE_BASE_PATH')
        cls._set_defaults_from_flow(deployment, flow)
        return deployment

    @classmethod
    @async_dispatch(afrom_storage)
    def from_storage(cls, storage: Any, entrypoint: str, name: str, flow_name: str = None, interval: Union[int, timedelta, List[Union[int, timedelta]]] = None, cron: Union[str, List[str]] = None, rrule: Union[str, List[str]] = None, paused: bool = None, schedule: Schedule = None, schedules: List[Schedule] = None, concurrency_limit: int = None, parameters: dict = None, triggers: List[Any] = None, description: str = None, tags: List[str] = None, version: str = None, enforce_parameter_schema: bool = True, work_pool_name: str = None, work_queue_name: str = None, job_variables: dict = None, _sla: Any = None) -> 'RunnerDeployment':
        """
        Create a RunnerDeployment from a flow located at a given entrypoint and stored in a
        local storage location.

        Args:
            entrypoint:  The path to a file containing a flow and the name of the flow function in
                the format `./path/to/file.py:flow_func_name`.
            name: A name for the deployment
            flow_name: The name of the flow to deploy
            storage: A storage object to use for retrieving flow code. If not provided, a
                URL must be provided.
            interval: An interval on which to execute the current flow. Accepts either a number
                or a timedelta object. If a number is given, it will be interpreted as seconds.
            cron: A cron schedule of when to execute runs of this flow.
            rrule: An rrule schedule of when to execute runs of this flow.
            paused: Whether or not the deployment is paused.
            schedule: A schedule object defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            schedules: A list of schedule objects defining when to execute runs of this deployment.
                Used to provide additional scheduling options like `timezone` or `parameters`.
            triggers: A list of triggers that should kick of a run of this flow.
            parameters: A dictionary of default parameter values to pass to runs of this flow.
            description: A description for the created deployment. Defaults to the flow's
                description if not provided.
            tags: A list of tags to associate with the created deployment for organizational
                purposes.
            version: A version for the created deployment. Defaults to the flow's version.
            enforce_parameter_schema: Whether or not the Prefect API should enforce the
                parameter schema for this deployment.
            work_pool_name: The name of the work pool to use for this deployment.
            work_queue_name: The name of the work queue to use for this deployment's scheduled runs.
                If not provided the default work queue for the work pool will be used.
            job_variables: Settings used to override the values specified default base job template
                of the chosen work pool. Refer to the base job template of the chosen work pool for
                available settings.
            _sla: (Experimental) SLA configuration for the deployment. May be removed or modified at any time. Currently only supported on Prefect Cloud.
        """
        from prefect.flows import load_flow_from_entrypoint
        constructed_schedules = cls._construct_deployment_schedules(interval=interval, cron=cron, rrule=rrule, schedules=schedules, schedule=schedule)
        if isinstance(concurrency_limit, ConcurrencyLimitConfig):
            concurrency_options = {'collision_strategy': concurrency_limit.collision_strategy}
            concurrency_limit = concurrency_limit.limit
        else:
            concurrency_options = None
        job_variables = job_variables or {}
        with tempfile.TemporaryDirectory() as tmpdir:
            storage.set_base_path(Path(tmpdir))
            run_coro_as_sync(storage.pull_code())
            full_entrypoint = str(storage.destination / entrypoint)
            flow = load_flow_from_entrypoint(full_entrypoint)
        deployment = cls(name=Path(name).stem, flow_name=flow_name or flow.name, schedules=constructed_schedules, concurrency_limit=concurrency_limit, concurrency_options=concurrency_options, paused=paused, tags=tags or [], triggers=triggers or [], parameters=parameters or {}, description=description, version=version, entrypoint=entrypoint, enforce_parameter_schema=enforce_parameter_schema, storage=storage, work_pool_name=work_pool_name, work_queue_name=work_queue_name, job_variables=job_variables)
        deployment._sla = _sla
        deployment._path = str(storage.destination).replace(tmpdir, '$STORAGE_BASE_PATH')
        cls._set_defaults_from_flow(deployment, flow)
        return deployment

@sync_compatible
async def deploy(*deployments: RunnerDeployment, work_pool_name: str = None, image: str = None, build: bool = True, push: bool = True, print_next_steps_message: bool = True, ignore_warnings: bool = False) -> List[UUID]:
    """
    Deploy the provided list of deployments to dynamic infrastructure via a
    work pool.

    By default, calling this function will build a Docker image for the deployments, push it to a
    registry, and create each deployment via the Prefect API that will run the corresponding
    flow on the given schedule.

    If you want to use an existing image, you can pass `build=False` to skip building and pushing
    an image.

    Args:
        *deployments: A list of deployments to deploy.
        work_pool_name: The name of the work pool to use for these deployments. Defaults to
            the value of `PREFECT_DEFAULT_WORK_POOL_NAME`.
        image: The name of the Docker image to build, including the registry and
            repository. Pass a DockerImage instance to customize the Dockerfile used
            and build arguments.
        build: Whether or not to build a new image for the flow. If False, the provided
            image will be used as-is and pulled at runtime.
        push: Whether or not to skip pushing the built image to a registry.
        print_next_steps_message: Whether or not to print a message with next steps
            after deploying the deployments.

    Returns:
        A list of deployment IDs for the created/updated deployments.

    Examples:
        Deploy a group of flows to a work pool:

        