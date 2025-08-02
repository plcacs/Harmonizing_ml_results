def create_plan_only_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, DisplayOnlyExecutor, NoopResultsRecorder)

def create_default_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, Executor, ResultsRecorder)

def _create_deployer(session: Session, config: Config, ui: UI, 
                    executor_cls: Type[BaseExecutor], 
                    recorder_cls: Type['ResultsRecorder']) -> 'Deployer':
    client = TypedAWSClient(session)
    osutils = OSUtils()
    return Deployer(
        application_builder=ApplicationGraphBuilder(),
        deps_builder=DependencyBuilder(),
        build_stage=create_build_stage(osutils, UI(), TemplatedSwaggerGenerator(), config),
        plan_stage=PlanStage(
            osutils=osutils,
            remote_state=RemoteState(client, config.deployed_resources(config.chalice_stage))
        ),
        sweeper=ResourceSweeper(),
        executor=executor_cls(client, ui),
        recorder=recorder_cls(osutils=osutils)
    )

def create_build_stage(osutils: OSUtils, ui: UI, 
                      swagger_gen: SwaggerGenerator, 
                      config: Config) -> 'BuildStage':
    pip_runner = PipRunner(pip=SubprocessPip(osutils=osutils), osutils=osutils)
    dependency_builder = PipDependencyBuilder(osutils=osutils, pip_runner=pip_runner)
    deployment_packager: BaseDeployStep = cast(BaseDeployStep, None)
    if config.automatic_layer:
        deployment_packager = ManagedLayerDeploymentPackager(
            lambda_packager=AppOnlyDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui
            ),
            layer_packager=LayerDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui
            )
        )
    else:
        deployment_packager = DeploymentPackager(
            packager=LambdaDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui
            )
        )
    build_stage = BuildStage(steps=[
        InjectDefaults(),
        deployment_packager,
        PolicyGenerator(policy_gen=AppPolicyGenerator(osutils=osutils), osutils=osutils),
        SwaggerBuilder(swagger_generator=swagger_gen),
        LambdaEventSourcePolicyInjector(),
        WebsocketPolicyInjector()
    ])
    return build_stage

def create_deletion_deployer(client: TypedAWSClient, ui: UI) -> 'Deployer':
    return Deployer(
        application_builder=ApplicationGraphBuilder(),
        deps_builder=DependencyBuilder(),
        build_stage=BuildStage(steps=[]),
        plan_stage=NoopPlanner(),
        sweeper=ResourceSweeper(),
        executor=Executor(client, ui),
        recorder=ResultsRecorder(osutils=OSUtils())
    )

class Deployer(object):
    BACKEND_NAME = 'api'

    def __init__(self, application_builder: ApplicationGraphBuilder, 
                deps_builder: DependencyBuilder, 
                build_stage: 'BuildStage', 
                plan_stage: PlanStage, 
                sweeper: ResourceSweeper, 
                executor: BaseExecutor, 
                recorder: 'ResultsRecorder') -> None:
        self._application_builder = application_builder
        self._deps_builder = deps_builder
        self._build_stage = build_stage
        self._plan_stage = plan_stage
        self._sweeper = sweeper
        self._executor = executor
        self._recorder = recorder

    def deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        try:
            return self._deploy(config, chalice_stage_name)
        except _AWSCLIENT_EXCEPTIONS as e:
            raise ChaliceDeploymentError(e)

    def _deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        self._validate_config(config)
        application = self._application_builder.build(config, chalice_stage_name)
        resources = self._deps_builder.build_dependencies(application)
        self._build_stage.execute(config, resources)
        resources = self._deps_builder.build_dependencies(application)
        plan = self._plan_stage.execute(resources)
        self._sweeper.execute(plan, config)
        self._executor.execute(plan)
        deployed_values = {
            'resources': self._executor.resource_values,
            'schema_version': '2.0',
            'backend': self.BACKEND_NAME
        }
        self._recorder.record_results(
            deployed_values, chalice_stage_name, config.project_dir)
        return deployed_values

    def _validate_config(self, config: Config) -> None:
        try:
            validate_configuration(config)
        except ValueError as e:
            raise ChaliceDeploymentError(e)

class BaseDeployStep(object):
    def handle(self, config: Config, resource: models.Model) -> None:
        name = 'handle_%s' % resource.__class__.__name__.lower()
        handler = getattr(self, name, None)
        if handler is not None:
            handler(config, resource)

class InjectDefaults(BaseDeployStep):
    def __init__(self, lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT, 
                lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE, 
                tls_version: str = DEFAULT_TLS_VERSION) -> None:
        self._lambda_timeout = lambda_timeout
        self._lambda_memory_size = lambda_memory_size
        self._tls_version = DEFAULT_TLS_VERSION

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        if resource.timeout is None:
            resource.timeout = self._lambda_timeout
        if resource.memory_size is None:
            resource.memory_size = self._lambda_memory_size

    def handle_domainname(self, config: Config, resource: models.DomainName) -> None:
        if resource.tls_version is None:
            resource.tls_version = models.TLSVersion.create(DEFAULT_TLS_VERSION)

class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None:
        self._packager = packager

    def handle_deploymentpackage(self, config: Config, resource: models.DeploymentPackage) -> None:
        if isinstance(resource.filename, models.Placeholder):
            zip_filename = self._packager.create_deployment_package(
                config.project_dir, config.lambda_python_version)
            resource.filename = zip_filename

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(self, lambda_packager: AppOnlyDeploymentPackager, 
                layer_packager: LayerDeploymentPackager) -> None:
        self._lambda_packager = lambda_packager
        self._layer_packager = layer_packager

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        if isinstance(resource.deployment_package.filename, models.Placeholder):
            zip_filename = self._lambda_packager.create_deployment_package(
                config.project_dir, config.lambda_python_version)
            resource.deployment_package.filename = zip_filename
        if resource.managed_layer is not None and resource.managed_layer.is_empty:
            resource.managed_layer = None

    def handle_lambdalayer(self, config: Config, resource: models.LambdaLayer) -> None:
        if isinstance(resource.deployment_package.filename, models.Placeholder):
            try:
                zip_filename = self._layer_packager.create_deployment_package(
                    config.project_dir, config.lambda_python_version)
                resource.deployment_package.filename = zip_filename
            except EmptyPackageError:
                resource.is_empty = True

class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        self._swagger_generator = swagger_generator

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        swagger_doc = self._swagger_generator.generate_swagger(
            config.chalice_app, resource)
        resource.swagger_doc = swagger_doc

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        self._sqs_policy_injected = False
        self._kinesis_policy_injected = False
        self._ddb_policy_injected = False

    def handle_sqseventsource(self, config: Config, resource: models.SQSEventSource) -> None:
        role = resource.lambda_function.role
        if not self._sqs_policy_injected and self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document, SQS_EVENT_SOURCE_POLICY.copy())
            self._sqs_policy_injected = True

    def handle_kinesiseventsource(self, config: Config, resource: models.KinesisEventSource) -> None:
        role = resource.lambda_function.role
        if not self._kinesis_policy_injected and self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document, KINESIS_EVENT_SOURCE_POLICY.copy())
            self._kinesis_policy_injected = True

    def handle_dynamodbeventsource(self, config: Config, resource: models.DynamoDBEventSource) -> None:
        role = resource.lambda_function.role
        if not self._ddb_policy_injected and self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document, DDB_EVENT_SOURCE_POLICY.copy())
            self._ddb_policy_injected = True

    def _needs_policy_injected(self, role: Optional[models.IAMRole]) -> bool:
        return isinstance(role, models.ManagedIAMRole) and isinstance(role.policy, models.AutoGenIAMPolicy) and (not isinstance(role.policy.document, models.Placeholder))

    def _inject_trigger_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        document['Statement'].append(policy)

class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        self._policy_injected = False

    def handle_websocketapi(self, config: Config, resource: models.WebsocketAPI) -> None:
        self._inject_into_function(config, resource.connect_function)
        self._inject_into_function(config, resource.message_function)
        self._inject_into_function(config, resource.disconnect_function)

    def _inject_into_function(self, config: Config, lambda_function: Optional[models.LambdaFunction]) -> None:
        if lambda_function is None:
            return
        role = lambda_function.role
        if role is None:
            return
        if not self._policy_injected and isinstance(role, models.ManagedIAMRole) and isinstance(role.policy, models.AutoGenIAMPolicy) and (not isinstance(role.policy.document, models.Placeholder)):
            self._inject_policy(role.policy.document, POST_TO_WEBSOCKET_CONNECTION_POLICY.copy())
        self._policy_injected = True

    def _inject_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        document['Statement'].append(policy)

class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        self._policy_gen = policy_gen
        self._osutils = osutils

    def _read_document_from_file(self, filename: str) -> Dict[str, Any]:
        try:
            return json.loads(self._osutils.get_file_contents(filename))
        except IOError as e:
            raise RuntimeError('Unable to load IAM policy file %s: %s' % (filename, e))

    def handle_filebasediampolicy(self, config: Config, resource: models.FileBasedIAMPolicy) -> None:
        resource.document = self._read_document_from_file(resource.filename)

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        if resource.policy and isinstance(resource.policy, models.FileBasedIAMPolicy):
            resource.policy.document = self._read_document_from_file(resource.policy.filename)

    def handle_autogeniampolicy(self, config: Config, resource: models.AutoGenIAMPolicy) -> None:
        if isinstance(resource.document, models.Placeholder):
            policy = self._policy_gen.generate_policy(config)
            if models.RoleTraits.VPC_NEEDED in resource.traits:
                policy['Statement'].append(VPC_ATTACH_POLICY)
            resource.document = policy

class BuildStage(object):
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        self._steps = steps

    def execute(self, config: Config, resources: List[models.Model]) -> None:
        for resource in resources:
            for step in self._steps:
                step.handle(config, resource)

class ResultsRecorder(object):
    def __init__(self, osutils: OSUtils) -> None:
        self._osutils = osutils

    def record_results(self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str) -> None:
        deployed_dir = self._osutils.joinpath(project_dir, '.chalice', 'deployed')
        deployed_filename = self._osutils.joinpath(deployed_dir, '%s.json' % chalice_stage_name)
        if not self._osutils.directory_exists(deployed_dir):
            self._osutils.makedirs(deployed_dir)
        serialized = serialize_to_json(results)
        self._osutils.set_file_contents(filename=deployed_filename, contents=serialized, binary=False)

class NoopResultsRecorder(ResultsRecorder):
    def record_results(self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str) -> None:
        return None

class DeploymentReporter(object):
    _SORT_ORDER = {'rest_api': 100, 'websocket_api': 100, 'domain_name': 100}
    _DEFAULT_ORDERING = 50

    def __init__(self, ui: UI) -> None:
        self._ui = ui

    def generate_report(self, deployed_values: Dict[str, Any]) -> str:
        report = ['Resources deployed:']
        ordered = sorted(deployed_values['resources'], 
                        key=lambda x: self._SORT_ORDER.get(x['resource_type'], self._DEFAULT_ORDERING))
        for resource in ordered:
            getattr(self, '_report_%s' % resource['resource_type'], 
                   self._default_report)(resource, report)
        report.append('')
        return '\n'.join(report)

    def _report_rest_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append('  - Rest API URL: %s' % resource['rest_api_url'])

    def _report_websocket_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append('  - Websocket API URL: %s' % resource['websocket_api_url'])

    def _report_domain_name(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append('  - Custom domain name:\n      HostedZoneId: %s\n      AliasDomainName: %s' % 
                     (resource['hosted_zone_id'], resource['alias_domain_name']))

    def _report_lambda_function(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append('  - Lambda ARN: %s' % resource['lambda_arn'])

    def _report_lambda_layer(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append('  - Lambda Layer ARN: %s' % resource['layer_version_arn'])

    def _default_report(self, resource: Dict[str, Any], report: List[str]) -> None:
        pass

    def display_report(self, deployed_values: Dict[str, Any]) -> None:
        report = self.generate_report(deployed_values)
        self._ui.write(report)
