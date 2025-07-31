import copy
import json
import logging
import shlex
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union
from uuid import UUID

import anyio
import anyio.abc
import yaml
from pydantic import BaseModel, Field, model_validator
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from typing_extensions import Literal, Self

from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.objects import FlowRun
from prefect.client.utilities import inject_client
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.workers.base import (
    BaseJobConfiguration,
    BaseVariables,
    BaseWorker,
    BaseWorkerResult,
    apply_values,
    resolve_block_document_references,
    resolve_variables,
)
from prefect_aws.credentials import AwsCredentials, ClientType

_ECSClient = Any
ECS_DEFAULT_CONTAINER_NAME: str = 'prefect'
ECS_DEFAULT_CPU: int = 1024
ECS_DEFAULT_COMMAND: str = 'python -m prefect.engine'
ECS_DEFAULT_MEMORY: int = 2048
ECS_DEFAULT_LAUNCH_TYPE: str = 'FARGATE'
ECS_DEFAULT_FAMILY: str = 'prefect'
ECS_POST_REGISTRATION_FIELDS: List[str] = [
    'compatibilities',
    'taskDefinitionArn',
    'revision',
    'status',
    'requiresAttributes',
    'registeredAt',
    'registeredBy',
    'deregisteredAt',
]
DEFAULT_TASK_DEFINITION_TEMPLATE: str = (
    "\ncontainerDefinitions:\n- image: \"{{ image }}\"\n  name: \"{{ container_name }}\"\n"
    "cpu: \"{{ cpu }}\"\nfamily: \"{{ family }}\"\nmemory: \"{{ memory }}\"\n"
    "executionRoleArn: \"{{ execution_role_arn }}\"\n"
)
DEFAULT_TASK_RUN_REQUEST_TEMPLATE: str = (
    "\nlaunchType: \"{{ launch_type }}\"\ncluster: \"{{ cluster }}\"\noverrides:\n  containerOverrides:\n"
    "    - name: \"{{ container_name }}\"\n      command: \"{{ command }}\"\n"
    "      environment: \"{{ env }}\"\n      cpu: \"{{ cpu }}\"\n      memory: \"{{ memory }}\"\n"
    "  cpu: \"{{ cpu }}\"\n  memory: \"{{ memory }}\"\n  taskRoleArn: \"{{ task_role_arn }}\"\n"
    "tags: \"{{ labels }}\"\n"
    "taskDefinition: \"{{ task_definition_arn }}\"\n"
    "capacityProviderStrategy: \"{{ capacity_provider_strategy }}\"\n"
)
MAX_CREATE_TASK_RUN_ATTEMPTS: int = 3
CREATE_TASK_RUN_MIN_DELAY_SECONDS: int = 1
CREATE_TASK_RUN_MIN_DELAY_JITTER_SECONDS: int = 0
CREATE_TASK_RUN_MAX_DELAY_JITTER_SECONDS: int = 3
_TASK_DEFINITION_CACHE: Dict[str, str] = {}
_TAG_REGEX: str = '[^a-zA-Z0-9_./=+:@-]'


class ECSIdentifier(NamedTuple):
    """
    The identifier for a running ECS task.
    """
    cluster: str
    task: str


def _default_task_definition_template() -> Dict[str, Any]:
    """
    The default task definition template for ECS jobs.
    """
    return yaml.safe_load(DEFAULT_TASK_DEFINITION_TEMPLATE)


def _default_task_run_request_template() -> Dict[str, Any]:
    """
    The default task run request template for ECS jobs.
    """
    return yaml.safe_load(DEFAULT_TASK_RUN_REQUEST_TEMPLATE)


def _drop_empty_keys_from_dict(taskdef: Dict[str, Any]) -> None:
    """
    Recursively drop keys with 'empty' values from a task definition dict.
    Mutates the task definition in place. Only supports recursion into dicts and lists.
    """
    for key, value in list(taskdef.items()):
        if not value:
            taskdef.pop(key)
        if isinstance(value, dict):
            _drop_empty_keys_from_dict(value)
        if isinstance(value, list) and key != 'capacity_provider_strategy':
            for v in value:
                if isinstance(v, dict):
                    _drop_empty_keys_from_dict(v)


def _get_container(containers: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """
    Extract a container from a list of containers or container definitions.
    If not found, `None` is returned.
    """
    for container in containers:
        if container.get('name') == name:
            return container
    return None


def _container_name_from_task_definition(task_definition: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Attempt to infer the container name from a task definition.
    If not found, `None` is returned.
    """
    if task_definition:
        container_definitions = task_definition.get('containerDefinitions', [])
    else:
        container_definitions = []
    if _get_container(container_definitions, ECS_DEFAULT_CONTAINER_NAME):
        return ECS_DEFAULT_CONTAINER_NAME
    elif container_definitions:
        return container_definitions[0].get('name')
    return None


def parse_identifier(identifier: str) -> ECSIdentifier:
    """
    Splits identifier into its cluster and task components, e.g.
    input "cluster_name::task_arn" outputs ("cluster_name", "task_arn").
    """
    cluster, task = identifier.split('::', maxsplit=1)
    return ECSIdentifier(cluster, task)


def mask_sensitive_env_values(
    task_run_request: Dict[str, Any], 
    values: List[str], 
    keep_length: int = 3, 
    replace_with: str = '***'
) -> Dict[str, Any]:
    for container in task_run_request.get('overrides', {}).get('containerOverrides', []):
        for env_var in container.get('environment', []):
            if 'name' not in env_var or 'value' not in env_var or env_var['name'] not in values:
                continue
            if len(env_var['value']) > keep_length:
                env_var['value'] = env_var['value'][:keep_length] + replace_with
    return task_run_request


def mask_api_key(task_run_request: Dict[str, Any]) -> Dict[str, Any]:
    return mask_sensitive_env_values(deepcopy(task_run_request), ['PREFECT_API_KEY'], keep_length=6)


class CapacityProvider(BaseModel):
    """
    The capacity provider strategy to use when running the task.
    """
    # Add relevant fields if needed
    pass


class ECSJobConfiguration(BaseJobConfiguration):
    """
    Job configuration for an ECS worker.
    """
    aws_credentials: AwsCredentials = Field(default_factory=AwsCredentials)
    task_definition: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"template": _default_task_definition_template()})
    task_run_request: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"template": _default_task_run_request_template()})
    configure_cloudwatch_logs: Optional[Any] = Field(default=None)
    cloudwatch_logs_options: Dict[str, Any] = Field(default_factory=dict)
    cloudwatch_logs_prefix: Optional[str] = Field(default=None)
    network_configuration: Dict[str, Any] = Field(default_factory=dict)
    stream_output: Optional[Any] = Field(default=None)
    task_start_timeout_seconds: int = Field(default=300)
    task_watch_poll_interval: float = Field(default=5.0)
    auto_deregister_task_definition: bool = Field(default=False)
    vpc_id: Optional[str] = Field(default=None)
    container_name: Optional[str] = Field(default=None)
    cluster: Optional[str] = Field(default=None)
    match_latest_revision_in_family: bool = Field(default=False)
    execution_role_arn: Optional[str] = Field(title='Execution Role ARN', default=None, description='An execution role to use for the task. This controls the permissions of the task when it is launching. If this value is not null, it will override the value in the task definition. An execution role must be provided to capture logs from the container.')

    @model_validator(mode='after')
    def task_run_request_requires_arn_if_no_task_definition_given(self) -> Self:
        """
        If no task definition is provided, a task definition ARN must be present on the
        task run request.
        """
        if not (self.task_run_request or {}).get('taskDefinition') and (not self.task_definition):
            raise ValueError('A task definition must be provided if a task definition ARN is not present on the task run request.')
        return self

    @model_validator(mode='after')
    def container_name_default_from_task_definition(self) -> Self:
        """
        Infers the container name from the task definition if not provided.
        """
        if self.container_name is None:
            self.container_name = _container_name_from_task_definition(self.task_definition)
        return self

    @model_validator(mode='after')
    def set_default_configure_cloudwatch_logs(self) -> Self:
        """
        Streaming output generally requires CloudWatch logs to be configured.
        To avoid entangled arguments in the simple case, `configure_cloudwatch_logs`
        defaults to matching the value of `stream_output`.
        """
        configure_cloudwatch_logs = self.configure_cloudwatch_logs
        if configure_cloudwatch_logs is None:
            self.configure_cloudwatch_logs = self.stream_output
        return self

    @model_validator(mode='after')
    def configure_cloudwatch_logs_requires_execution_role_arn(self) -> Self:
        """
        Enforces that an execution role arn is provided (or could be provided by a
        runtime task definition) when configuring logging.
        """
        if self.configure_cloudwatch_logs and (not self.execution_role_arn) and (not (self.task_run_request or {}).get('taskDefinition')) and (not (self.task_definition or {}).get('executionRoleArn')):
            raise ValueError('An `execution_role_arn` must be provided to use `configure_cloudwatch_logs` or `stream_logs`.')
        return self

    @model_validator(mode='after')
    def cloudwatch_logs_options_requires_configure_cloudwatch_logs(self) -> Self:
        """
        Enforces that an execution role arn is provided (or could be provided by a
        runtime task definition) when configuring logging.
        """
        if self.cloudwatch_logs_options and (not self.configure_cloudwatch_logs):
            raise ValueError('`configure_cloudwatch_log` must be enabled to use `cloudwatch_logs_options`.')
        return self

    @model_validator(mode='after')
    def network_configuration_requires_vpc_id(self) -> Self:
        """
        Enforces a `vpc_id` is provided when custom network configuration mode is
        enabled for network settings.
        """
        if self.network_configuration and (not self.vpc_id):
            raise ValueError('You must provide a `vpc_id` to enable custom `network_configuration`.')
        return self

    @classmethod
    @inject_client
    async def from_template_and_values(cls, base_job_template: Dict[str, Any], values: Dict[str, Any], client: Optional[PrefectClient] = None) -> "ECSJobConfiguration":
        """Creates a valid worker configuration object from the provided base
        configuration and overrides.
        Important: this method expects that the base_job_template was already
        validated server-side.
        """
        job_config = base_job_template['job_configuration']
        variables_schema = base_job_template['variables']
        variables = cls._get_base_config_defaults(variables_schema.get('properties', {}))
        variables.update(values)
        _drop_empty_keys_from_dict(variables)
        populated_configuration = apply_values(template=job_config, values=variables)
        populated_configuration = await resolve_block_document_references(template=populated_configuration, client=client)
        populated_configuration = await resolve_variables(template=populated_configuration, client=client)
        return cls(**populated_configuration)


class ECSVariables(BaseVariables):
    """
    Variables for templating an ECS job.
    """
    task_definition_arn: Optional[str] = Field(default=None, description='An identifier for an existing task definition to use. If set, options that require changes to the task definition will be ignored. All contents of the task definition in the job configuration will be ignored.')
    env: Dict[str, Any] = Field(title='Environment Variables', default_factory=dict, description='Environment variables to provide to the task run. These variables are set on the Prefect container at task runtime. These will not be set on the task definition.')
    aws_credentials: AwsCredentials = Field(title='AWS Credentials', default_factory=AwsCredentials, description="The AWS credentials to use to connect to ECS. If not provided, credentials will be inferred from the local environment following AWS's boto client's rules.")
    cluster: Optional[str] = Field(default=None, description='The ECS cluster to run the task in. An ARN or name may be provided. If not provided, the default cluster will be used.')
    family: Optional[str] = Field(default=None, description='A family for the task definition. If not provided, it will be inferred from the task definition. If the task definition does not have a family, the name will be generated. When flow and deployment metadata is available, the generated name will include their names. Values for this field will be slugified to match AWS character requirements.')
    launch_type: str = Field(default=ECS_DEFAULT_LAUNCH_TYPE, description="The type of ECS task run infrastructure that should be used. Note that 'FARGATE_SPOT' is not a formal ECS launch type, but we will configure the proper capacity provider strategy if set here.")
    capacity_provider_strategy: List[Any] = Field(default_factory=list, description='The capacity provider strategy to use when running the task. If a capacity provider strategy is specified, the selected launch type will be ignored.')
    image: Optional[str] = Field(default=None, description='The image to use for the Prefect container in the task. If this value is not null, it will override the value in the task definition. This value defaults to a Prefect base image matching your local versions.')
    cpu: Optional[Union[int, str]] = Field(title='CPU', default=None, description=f'The amount of CPU to provide to the ECS task. Valid amounts are specified in the AWS documentation. If not provided, a default value of {ECS_DEFAULT_CPU} will be used unless present on the task definition.')
    memory: Optional[Union[int, str]] = Field(default=None, description=f'The amount of memory to provide to the ECS task. Valid amounts are specified in the AWS documentation. If not provided, a default value of {ECS_DEFAULT_MEMORY} will be used unless present on the task definition.')
    container_name: Optional[str] = Field(default=None, description=f'The name of the container flow run orchestration will occur in. If not specified, a default value of {ECS_DEFAULT_CONTAINER_NAME} will be used and if that is not found in the task definition the first container will be used.')
    task_role_arn: Optional[str] = Field(title='Task Role ARN', default=None, description='A role to attach to the task run. This controls the permissions of the task while it is running.')
    execution_role_arn: Optional[str] = Field(title='Execution Role ARN', default=None, description='An execution role to use for the task. This controls the permissions of the task when it is launching. If this value is not null, it will override the value in the task definition. An execution role must be provided to capture logs from the container.')
    vpc_id: Optional[str] = Field(title='VPC ID', default=None, description="The AWS VPC to link the task run to. This is only applicable when using the 'awsvpc' network mode for your task. FARGATE tasks require this network  mode, but for EC2 tasks the default network mode is 'bridge'. If using the 'awsvpc' network mode and this field is null, your default VPC will be used. If no default VPC can be found, the task run will fail.")
    configure_cloudwatch_logs: Optional[Any] = Field(default=None, description='If enabled, the Prefect container will be configured to send its output to the AWS CloudWatch logs service. This functionality requires an execution role with logs:CreateLogStream, logs:CreateLogGroup, and logs:PutLogEvents permissions. The default for this field is `False` unless `stream_output` is set.')
    cloudwatch_logs_options: Dict[str, Any] = Field(default_factory=dict, description='When `configure_cloudwatch_logs` is enabled, this setting may be used to pass additional options to the CloudWatch logs configuration or override the default options. See the [AWS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_awslogs.html#create_awslogs_logdriver_options) for available options. ')
    cloudwatch_logs_prefix: Optional[str] = Field(default=None, description='When `configure_cloudwatch_logs` is enabled, this setting may be used to set a prefix for the log group. If not provided, the default prefix will be `prefect-logs_<work_pool_name>_<deployment_id>`. If `awslogs-stream-prefix` is present in `Cloudwatch logs options` this setting will be ignored.')
    network_configuration: Dict[str, Any] = Field(default_factory=dict, description="When `network_configuration` is supplied it will override ECS Worker'sawsvpcConfiguration that defined in the ECS task executing your workload. See the [AWS documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-awsvpcconfiguration.html) for available options.")
    stream_output: Optional[Any] = Field(default=None, description='If enabled, logs will be streamed from the Prefect container to the local console. Unless you have configured AWS CloudWatch logs manually on your task definition, this requires the same prerequisites outlined in `configure_cloudwatch_logs`.')
    task_start_timeout_seconds: int = Field(default=300, description='The amount of time to watch for the start of the ECS task before marking it as failed. The task must enter a RUNNING state to be considered started.')
    task_watch_poll_interval: float = Field(default=5.0, description='The amount of time to wait between AWS API calls while monitoring the state of an ECS task.')
    auto_deregister_task_definition: bool = Field(default=False, description='If enabled, any task definitions that are created by this block will be deregistered. Existing task definitions linked by ARN will never be deregistered. Deregistering a task definition does not remove it from your AWS account, instead it will be marked as INACTIVE.')
    match_latest_revision_in_family: bool = Field(default=False, description=f'If enabled, the most recent active revision in the task definition family will be compared against the desired ECS task configuration. If they are equal, the existing task definition will be used instead of registering a new one. If no family is specified the default family "{ECS_DEFAULT_FAMILY}" will be used.')


class ECSWorkerResult(BaseWorkerResult):
    """
    The result of an ECS job.
    """
    pass


class ECSWorker(BaseWorker):
    """
    A Prefect worker to run flow runs as ECS tasks.
    """
    type: str = 'ecs'
    job_configuration: Type[ECSJobConfiguration] = ECSJobConfiguration
    job_configuration_variables: Type[ECSVariables] = ECSVariables
    _description: str = 'Execute flow runs within containers on AWS ECS. Works with EC2 and Fargate clusters. Requires an AWS account.'
    _display_name: str = 'AWS Elastic Container Service'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-aws/'
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png'

    async def run(self, flow_run: FlowRun, configuration: ECSJobConfiguration, task_status: Optional[Any] = None) -> ECSWorkerResult:
        """
        Runs a given flow run on the current worker.
        """
        ecs_client: _ECSClient = await run_sync_in_worker_thread(self._get_client, configuration, 'ecs')
        logger: logging.Logger = self.get_flow_run_logger(flow_run)
        task_arn, cluster_arn, task_definition, is_new_task_definition = await run_sync_in_worker_thread(
            self._create_task_and_wait_for_start, logger, ecs_client, configuration, flow_run
        )
        identifier: str = (configuration.cluster if configuration.cluster else cluster_arn) + '::' + task_arn
        if task_status:
            task_status.started(identifier)
        status_code: Optional[int] = await run_sync_in_worker_thread(
            self._watch_task_and_get_exit_code,
            logger,
            configuration,
            task_arn,
            cluster_arn,
            task_definition,
            is_new_task_definition and configuration.auto_deregister_task_definition,
            ecs_client,
        )
        return ECSWorkerResult(identifier=identifier, status_code=status_code if status_code is not None else -1)

    def _get_client(self, configuration: ECSJobConfiguration, client_type: str) -> _ECSClient:
        """
        Get a boto3 client of client_type. Will use a cached client if one exists.
        """
        return configuration.aws_credentials.get_client(client_type)

    def _create_task_and_wait_for_start(
        self, 
        logger: logging.Logger, 
        ecs_client: _ECSClient, 
        configuration: ECSJobConfiguration, 
        flow_run: FlowRun
    ) -> Tuple[str, str, Dict[str, Any], bool]:
        """
        Register the task definition, create the task run, and wait for it to start.
        Returns a tuple of:
        - The task ARN
        - The task's cluster ARN
        - The task definition
        - A bool indicating if the task definition is newly registered
        """
        task_definition_arn: Optional[str] = configuration.task_run_request.get('taskDefinition')
        new_task_definition_registered: bool = False
        if not task_definition_arn:
            task_definition: Dict[str, Any] = self._prepare_task_definition(configuration, region=ecs_client.meta.region_name, flow_run=flow_run)
            task_definition_arn, new_task_definition_registered = self._get_or_register_task_definition(logger, ecs_client, configuration, flow_run, task_definition)
        else:
            task_definition = self._retrieve_task_definition(logger, ecs_client, task_definition_arn)
            if configuration.task_definition:
                logger.warning('Ignoring task definition in configuration since task definition ARN is provided on the task run request.')
        self._validate_task_definition(task_definition, configuration)
        _TASK_DEFINITION_CACHE[flow_run.deployment_id] = task_definition_arn  # type: ignore
        logger.info(f'Using ECS task definition {task_definition_arn!r}...')
        logger.debug(f'Task definition {json.dumps(task_definition, indent=2, default=str)}')
        task_run_request: Dict[str, Any] = self._prepare_task_run_request(configuration, task_definition, task_definition_arn)
        logger.info('Creating ECS task run...')
        logger.debug(f'Task run request{json.dumps(mask_api_key(task_run_request), indent=2, default=str)}')
        try:
            task: Dict[str, Any] = self._create_task_run(ecs_client, task_run_request)
            task_arn = task['taskArn']
            cluster_arn = task['clusterArn']
        except Exception as exc:
            self._report_task_run_creation_failure(configuration, task_run_request, exc)
            raise
        logger.info('Waiting for ECS task run to start...')
        self._wait_for_task_start(logger, configuration, task_arn, cluster_arn, ecs_client, timeout=configuration.task_start_timeout_seconds)
        return (task_arn, cluster_arn, task_definition, new_task_definition_registered)

    def _get_or_register_task_definition(
        self, 
        logger: logging.Logger, 
        ecs_client: _ECSClient, 
        configuration: ECSJobConfiguration, 
        flow_run: FlowRun, 
        task_definition: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """Get or register a task definition for the given flow run.
        Returns a tuple of the task definition ARN and a bool indicating if the task
        definition is newly registered.
        """
        cached_task_definition_arn: Optional[str] = _TASK_DEFINITION_CACHE.get(flow_run.deployment_id)
        new_task_definition_registered: bool = False
        if cached_task_definition_arn:
            try:
                cached_task_definition: Dict[str, Any] = self._retrieve_task_definition(logger, ecs_client, cached_task_definition_arn)
                if not cached_task_definition['status'] == 'ACTIVE' or not self._task_definitions_equal(task_definition, cached_task_definition):
                    cached_task_definition_arn = None
            except Exception:
                cached_task_definition_arn = None
        if not cached_task_definition_arn and configuration.match_latest_revision_in_family:
            family_name: str = task_definition.get('family', ECS_DEFAULT_FAMILY)
            try:
                task_definition_from_family: Dict[str, Any] = self._retrieve_task_definition(logger, ecs_client, family_name)
                if task_definition_from_family and self._task_definitions_equal(task_definition, task_definition_from_family):
                    cached_task_definition_arn = task_definition_from_family['taskDefinitionArn']
            except Exception:
                cached_task_definition_arn = None
        if not cached_task_definition_arn:
            task_definition_arn = self._register_task_definition(logger, ecs_client, task_definition)
            new_task_definition_registered = True
        else:
            task_definition_arn = cached_task_definition_arn
        return (task_definition_arn, new_task_definition_registered)

    def _watch_task_and_get_exit_code(
        self, 
        logger: logging.Logger, 
        configuration: ECSJobConfiguration, 
        task_arn: str, 
        cluster_arn: str, 
        task_definition: Dict[str, Any], 
        deregister_task_definition: bool, 
        ecs_client: _ECSClient
    ) -> Optional[int]:
        """
        Wait for the task run to complete and retrieve the exit code of the Prefect
        container.
        """
        task: Dict[str, Any] = self._wait_for_task_finish(logger, configuration, task_arn, cluster_arn, task_definition, ecs_client)
        if deregister_task_definition:
            ecs_client.deregister_task_definition(taskDefinition=task['taskDefinitionArn'])
        container_name: Optional[str] = configuration.container_name or _container_name_from_task_definition(task_definition) or ECS_DEFAULT_CONTAINER_NAME
        container: Optional[Dict[str, Any]] = _get_container(task['containers'], container_name)
        assert container is not None, f"'{container_name}' container missing from task: {task}"
        status_code: Optional[int] = container.get('exitCode')
        self._report_container_status_code(logger, container_name, status_code)
        return status_code

    def _report_container_status_code(self, logger: logging.Logger, name: str, status_code: Optional[int]) -> None:
        """
        Display a log for the given container status code.
        """
        if status_code is None:
            logger.error(f'Task exited without reporting an exit status for container {name!r}.')
        elif status_code == 0:
            logger.info(f'Container {name!r} exited successfully.')
        else:
            logger.warning(f'Container {name!r} exited with non-zero exit code {status_code}.')

    def _report_task_run_creation_failure(self, configuration: ECSJobConfiguration, task_run: Dict[str, Any], exc: Exception) -> None:
        """
        Wrap common AWS task run creation failures with nicer user-facing messages.
        """
        if 'ClusterNotFoundException' in str(exc):
            cluster = task_run.get('cluster', 'default')
            raise RuntimeError(f'Failed to run ECS task, cluster {cluster!r} not found. Confirm that the cluster is configured in your region.') from exc
        elif 'No Container Instances' in str(exc) and task_run.get('launchType') == 'EC2':
            cluster = task_run.get('cluster', 'default')
            raise RuntimeError(f'Failed to run ECS task, cluster {cluster!r} does not appear to have any container instances associated with it. Confirm that you have EC2 container instances available.') from exc
        elif 'failed to validate logger args' in str(exc) and 'AccessDeniedException' in str(exc) and configuration.configure_cloudwatch_logs:
            raise RuntimeError(f'Failed to run ECS task, the attached execution role does not appear to have sufficient permissions. Ensure that the execution role {configuration.execution_role_arn!r} has permissions logs:CreateLogStream, logs:CreateLogGroup, and logs:PutLogEvents.')
        else:
            raise

    def _validate_task_definition(self, task_definition: Dict[str, Any], configuration: ECSJobConfiguration) -> None:
        """
        Ensure that the task definition is compatible with the configuration.
        Raises `ValueError` on incompatibility. Returns `None` on success.
        """
        launch_type: str = configuration.task_run_request.get('launchType', ECS_DEFAULT_LAUNCH_TYPE)
        if launch_type != 'EC2' and 'FARGATE' not in task_definition['requiresCompatibilities']:
            raise ValueError(f"Task definition does not have 'FARGATE' in 'requiresCompatibilities' and cannot be used with launch type {launch_type!r}")
        if launch_type == 'FARGATE' or launch_type == 'FARGATE_SPOT':
            network_mode: Any = task_definition.get('networkMode')
            if network_mode != 'awsvpc':
                raise ValueError(f"Found network mode {network_mode!r} which is not compatible with launch type {launch_type!r}. Use either the 'EC2' launch type or the 'awsvpc' network mode.")
        if configuration.configure_cloudwatch_logs and (not task_definition.get('executionRoleArn')):
            raise ValueError('An execution role arn must be set on the task definition to use `configure_cloudwatch_logs` or `stream_logs` but no execution role was found on the task definition.')

    def _register_task_definition(self, logger: logging.Logger, ecs_client: _ECSClient, task_definition: Dict[str, Any]) -> str:
        """
        Register a new task definition with AWS.
        Returns the ARN.
        """
        logger.info('Registering ECS task definition...')
        logger.debug(f'Task definition request{json.dumps(task_definition, indent=2, default=str)}')
        response: Dict[str, Any] = ecs_client.register_task_definition(**task_definition)
        return response['taskDefinition']['taskDefinitionArn']

    def _retrieve_task_definition(self, logger: logging.Logger, ecs_client: _ECSClient, task_definition: str) -> Dict[str, Any]:
        """
        Retrieve an existing task definition from AWS.
        """
        if task_definition.startswith('arn:aws:ecs:'):
            logger.info(f'Retrieving ECS task definition {task_definition!r}...')
        else:
            logger.info(f'Retrieving most recent active revision from ECS task family {task_definition!r}...')
        response: Dict[str, Any] = ecs_client.describe_task_definition(taskDefinition=task_definition)
        return response['taskDefinition']

    def _wait_for_task_start(
        self, 
        logger: logging.Logger, 
        configuration: ECSJobConfiguration, 
        task_arn: str, 
        cluster_arn: str, 
        ecs_client: _ECSClient, 
        timeout: Optional[float]
    ) -> Dict[str, Any]:
        """
        Waits for an ECS task run to reach a RUNNING status.
        If a STOPPED status is reached instead, an exception is raised indicating the
        reason that the task run did not start.
        """
        for task in self._watch_task_run(logger, configuration, task_arn, cluster_arn, ecs_client, until_status='RUNNING', timeout=timeout):
            if task['lastStatus'] == 'STOPPED':
                code = task.get('stopCode')
                reason = task.get('stoppedReason')
                raise type(code, (RuntimeError,), {})(reason)
        return task

    def _wait_for_task_finish(
        self, 
        logger: logging.Logger, 
        configuration: ECSJobConfiguration, 
        task_arn: str, 
        cluster_arn: str, 
        task_definition: Dict[str, Any], 
        ecs_client: _ECSClient
    ) -> Dict[str, Any]:
        """
        Watch an ECS task until it reaches a STOPPED status.
        If configured, logs from the Prefect container are streamed to stderr.
        Returns a description of the task on completion.
        """
        can_stream_output: bool = False
        container_name: str = configuration.container_name or _container_name_from_task_definition(task_definition) or ECS_DEFAULT_CONTAINER_NAME
        if configuration.stream_output:
            container_def: Optional[Dict[str, Any]] = _get_container(task_definition['containerDefinitions'], container_name)
            if not container_def:
                logger.warning('Prefect container definition not found in task definition. Output cannot be streamed.')
            elif not container_def.get('logConfiguration'):
                logger.warning('Logging configuration not found on task. Output cannot be streamed.')
            elif not container_def['logConfiguration'].get('logDriver') == 'awslogs':
                logger.warning(f"Logging configuration uses unsupported  driver {container_def['logConfiguration'].get('logDriver')!r}. Output cannot be streamed.")
            else:
                log_config: Dict[str, Any] = container_def['logConfiguration']['options']
                logs_client: Any = self._get_client(configuration, 'logs')
                can_stream_output = True
                last_log_timestamp: Optional[int] = None
                stream_name: str = '/'.join([log_config['awslogs-stream-prefix'], container_name, task_arn.rsplit('/')[-1]])
                self._logger.info(f'Streaming output from container {container_name!r}...')
        for task in self._watch_task_run(logger, configuration, task_arn, cluster_arn, ecs_client, current_status='RUNNING'):
            if configuration.stream_output and can_stream_output:
                last_log_timestamp = self._stream_available_logs(logger, logs_client, log_group=log_config['awslogs-group'], log_stream=stream_name, last_log_timestamp=last_log_timestamp)
        return task

    def _stream_available_logs(
        self, 
        logger: logging.Logger, 
        logs_client: Any, 
        log_group: str, 
        log_stream: str, 
        last_log_timestamp: Optional[int] = None
    ) -> Optional[int]:
        """
        Stream logs from the given log group and stream since the last log timestamp.
        Will continue on paginated responses until all logs are returned.
        Returns the last log timestamp which can be used to call this method in the future.
        """
        last_log_stream_token: str = 'NO-TOKEN'
        next_log_stream_token: Optional[str] = None
        while last_log_stream_token != next_log_stream_token:
            last_log_stream_token = next_log_stream_token
            request: Dict[str, Any] = {'logGroupName': log_group, 'logStreamName': log_stream}
            if last_log_stream_token is not None:
                request['nextToken'] = last_log_stream_token
            if last_log_timestamp is not None:
                request['startTime'] = last_log_timestamp + 1
            try:
                response: Dict[str, Any] = logs_client.get_log_events(**request)
            except Exception:
                logger.error(f'Failed to read log events with request {request}', exc_info=True)
                return last_log_timestamp
            log_events: List[Dict[str, Any]] = response['events']
            for log_event in log_events:
                print(log_event['message'], file=sys.stderr)
                if last_log_timestamp is None or log_event['timestamp'] > last_log_timestamp:
                    last_log_timestamp = log_event['timestamp']
            next_log_stream_token = response.get('nextForwardToken')
            if not log_events:
                break
        return last_log_timestamp

    def _watch_task_run(
        self, 
        logger: logging.Logger, 
        configuration: ECSJobConfiguration, 
        task_arn: str, 
        cluster_arn: str, 
        ecs_client: _ECSClient, 
        current_status: str = 'UNKNOWN', 
        until_status: Optional[str] = None, 
        timeout: Optional[float] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Watches an ECS task run by querying every `poll_interval` seconds. After each
        query, the retrieved task is yielded. This function returns when the task run
        reaches a STOPPED status or the provided `until_status`.
        Emits a log each time the status changes.
        """
        last_status: str = current_status
        status: str = current_status
        t0: float = time.time()
        while status != until_status:
            tasks: List[Dict[str, Any]] = ecs_client.describe_tasks(tasks=[task_arn], cluster=cluster_arn, include=['TAGS'])['tasks']
            if tasks:
                task: Dict[str, Any] = tasks[0]
                status = task['lastStatus']
                if status != last_status:
                    logger.info(f'ECS task status is {status}.')
                yield task
                if status == 'STOPPED':
                    break
                last_status = status
            else:
                logger.debug('Task not found.')
            elapsed_time: float = time.time() - t0
            if timeout is not None and elapsed_time > timeout:
                raise RuntimeError(f"Timed out after {elapsed_time}s while watching task for status {until_status or 'STOPPED'}.")
            time.sleep(configuration.task_watch_poll_interval)

    def _get_or_generate_family(self, task_definition: Dict[str, Any], flow_run: FlowRun) -> str:
        """
        Gets or generate a family for the task definition.
        """
        family: Optional[str] = task_definition.get('family')
        if not family:
            assert self._work_pool_name and flow_run.deployment_id
            family = f'{ECS_DEFAULT_FAMILY}_{self._work_pool_name}_{flow_run.deployment_id}'
        slugify(family, max_length=255, regex_pattern='[^a-zA-Z0-9-_]+')
        return family

    def _prepare_task_definition(self, configuration: ECSJobConfiguration, region: str, flow_run: FlowRun) -> Dict[str, Any]:
        """
        Prepare a task definition by inferring any defaults and merging overrides.
        """
        task_definition: Dict[str, Any] = copy.deepcopy(configuration.task_definition)
        task_definition.setdefault('containerDefinitions', [])
        task_definition['containerDefinitions'] = [d for d in task_definition['containerDefinitions'] if d]
        container_name: Optional[str] = configuration.container_name
        if not container_name:
            container_name = _container_name_from_task_definition(task_definition) or ECS_DEFAULT_CONTAINER_NAME
        container: Optional[Dict[str, Any]] = _get_container(task_definition['containerDefinitions'], container_name)
        if container is None:
            if container_name != ECS_DEFAULT_CONTAINER_NAME:
                raise ValueError(f'Container {container_name!r} not found in task definition.')
            for container in task_definition['containerDefinitions']:
                if 'name' not in container:
                    container['name'] = container_name
                    break
            else:
                container = {'name': container_name}
                task_definition['containerDefinitions'].append(container)
        container.setdefault('image', get_prefect_image_name())
        unset_keys = {key for key, value in configuration.env.items() if value is None}
        for item in tuple(container.get('environment', [])):
            if item['name'] in unset_keys or item['value'] is None:
                container['environment'].remove(item)
        if configuration.configure_cloudwatch_logs:
            prefix: str = f'prefect-logs_{self._work_pool_name}_{flow_run.deployment_id}'
            container['logConfiguration'] = {
                'logDriver': 'awslogs',
                'options': {
                    'awslogs-create-group': 'true',
                    'awslogs-group': 'prefect',
                    'awslogs-region': region,
                    'awslogs-stream-prefix': configuration.cloudwatch_logs_prefix or prefix,
                    **configuration.cloudwatch_logs_options,
                },
            }
        task_definition['family'] = self._get_or_generate_family(task_definition, flow_run)
        cpu: Union[int, str] = task_definition.get('cpu') or ECS_DEFAULT_CPU
        memory: Union[int, str] = task_definition.get('memory') or ECS_DEFAULT_MEMORY
        launch_type: str = configuration.task_run_request.get('launchType', ECS_DEFAULT_LAUNCH_TYPE)
        if launch_type == 'FARGATE' or launch_type == 'FARGATE_SPOT':
            task_definition['cpu'] = str(cpu)
            task_definition['memory'] = str(memory)
            requires_compatibilities: List[str] = task_definition.setdefault('requiresCompatibilities', [])
            if 'FARGATE' not in requires_compatibilities:
                requires_compatibilities.append('FARGATE')
            task_definition.setdefault('networkMode', 'awsvpc')
        elif launch_type == 'EC2':
            container.setdefault('cpu', cpu)
            container.setdefault('memory', memory)
            container['cpu'] = int(container['cpu'])
            container['memory'] = int(container['memory'])
        if task_definition.get('cpu'):
            task_definition['cpu'] = str(task_definition['cpu'])
        if task_definition.get('memory'):
            task_definition['memory'] = str(task_definition['memory'])
        return task_definition

    def _load_network_configuration(self, vpc_id: Optional[str], configuration: ECSJobConfiguration) -> Dict[str, Any]:
        """
        Load settings from a specific VPC or the default VPC and generate a task
        run request's network configuration.
        """
        ec2_client: Any = self._get_client(configuration, 'ec2')
        vpc_message: str = 'the default VPC' if not vpc_id else f'VPC with ID {vpc_id}'
        if not vpc_id:
            describe: Dict[str, Any] = {'Filters': [{'Name': 'isDefault', 'Values': ['true']}]}
        else:
            describe = {'VpcIds': [vpc_id]}
        vpcs: List[Dict[str, Any]] = ec2_client.describe_vpcs(**describe)['Vpcs']
        if not vpcs:
            help_message: str = 'Pass an explicit `vpc_id` or configure a default VPC.' if not vpc_id else 'Check that the VPC exists in the current region.'
            raise ValueError(f'Failed to find {vpc_message}. Network configuration cannot be inferred. ' + help_message)
        vpc_id = vpcs[0]['VpcId']
        subnets: List[Dict[str, Any]] = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])['Subnets']
        if not subnets:
            raise ValueError(f'Failed to find subnets for {vpc_message}. Network configuration cannot be inferred.')
        return {'awsvpcConfiguration': {'subnets': [s['SubnetId'] for s in subnets], 'assignPublicIp': 'ENABLED', 'securityGroups': []}}

    def _custom_network_configuration(self, vpc_id: str, network_configuration: Dict[str, Any], configuration: ECSJobConfiguration) -> Dict[str, Any]:
        """
        Load settings from a specific VPC or the default VPC and generate a task
        run request's network configuration.
        """
        ec2_client: Any = self._get_client(configuration, 'ec2')
        vpc_message: str = f'VPC with ID {vpc_id}'
        vpcs: List[Dict[str, Any]] = ec2_client.describe_vpcs(VpcIds=[vpc_id]).get('Vpcs')
        if not vpcs:
            raise ValueError(f'Failed to find {vpc_message}. ' + 'Network configuration cannot be inferred. ' + 'Pass an explicit `vpc_id`.')
        vpc_id = vpcs[0]['VpcId']
        subnets: List[Dict[str, Any]] = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])['Subnets']
        if not subnets:
            raise ValueError(f'Failed to find subnets for {vpc_message}. ' + 'Network configuration cannot be inferred.')
        subnet_ids: List[str] = [subnet['SubnetId'] for subnet in subnets]
        config_subnets: List[str] = network_configuration.get('subnets', [])
        if not all((conf_sn in subnet_ids for conf_sn in config_subnets)):
            raise ValueError(f'Subnets {config_subnets} not found within {vpc_message}.' + 'Please check that VPC is associated with supplied subnets.')
        return {'awsvpcConfiguration': network_configuration}

    def _prepare_task_run_request(self, configuration: ECSJobConfiguration, task_definition: Dict[str, Any], task_definition_arn: str) -> Dict[str, Any]:
        """
        Prepare a task run request payload.
        """
        task_run_request: Dict[str, Any] = deepcopy(configuration.task_run_request)
        task_run_request.setdefault('taskDefinition', task_definition_arn)
        assert task_run_request['taskDefinition'] == task_definition_arn, f"Task definition ARN mismatch: {task_run_request['taskDefinition']!r} != {task_definition_arn!r}"
        capacityProviderStrategy = task_run_request.get('capacityProviderStrategy')
        if capacityProviderStrategy:
            self._logger.warning('Found capacityProviderStrategy. Removing launchType from task run request.')
            task_run_request.pop('launchType', None)
        elif task_run_request.get('launchType') == 'FARGATE_SPOT':
            task_run_request.pop('launchType', None)
            task_run_request['capacityProviderStrategy'] = [{'capacityProvider': 'FARGATE_SPOT', 'weight': 1}]
        overrides: Dict[str, Any] = task_run_request.get('overrides', {})
        container_overrides: List[Dict[str, Any]] = overrides.get('containerOverrides', [])
        if task_definition.get('networkMode') == 'awsvpc' and (not task_run_request.get('networkConfiguration')) and (not configuration.network_configuration):
            task_run_request['networkConfiguration'] = self._load_network_configuration(configuration.vpc_id, configuration)
        if task_definition.get('networkMode') == 'awsvpc' and configuration.network_configuration and configuration.vpc_id:
            task_run_request['networkConfiguration'] = self._custom_network_configuration(configuration.vpc_id, configuration.network_configuration, configuration)
        container_name: str = configuration.container_name or _container_name_from_task_definition(task_definition) or ECS_DEFAULT_CONTAINER_NAME
        if container_overrides and (not container_overrides[0].get('name')):
            container_overrides[0]['name'] = container_name
        orchestration_container: Optional[Dict[str, Any]] = _get_container(container_overrides, container_name)
        if orchestration_container:
            if getattr(configuration, "command", None):
                orchestration_container['command'] = configuration.command  # type: ignore
        for container in container_overrides:
            if isinstance(container.get('command'), str):
                container['command'] = shlex.split(container['command'])
            if isinstance(container.get('environment'), dict):
                container['environment'] = [{'name': k, 'value': v} for k, v in container['environment'].items()]
            container['environment'] = [item for item in container.get('environment', []) if item['value'] is not None]
        if isinstance(task_run_request.get('tags'), dict):
            task_run_request['tags'] = [{'key': k, 'value': v} for k, v in task_run_request['tags'].items()]
        if overrides.get('cpu'):
            overrides['cpu'] = str(overrides['cpu'])
        if overrides.get('memory'):
            overrides['memory'] = str(overrides['memory'])
        tags: List[Dict[str, str]] = [
            item for item in task_run_request.get('tags', [])
            if item['key'] not in getattr(configuration, "labels", {}).keys()
        ] + [{'key': k, 'value': v} for k, v in getattr(configuration, "labels", {}).items() if v is not None]
        tags = [{'key': slugify(item['key'], regex_pattern=_TAG_REGEX, allow_unicode=True, lowercase=False), 'value': slugify(item['value'], regex_pattern=_TAG_REGEX, allow_unicode=True, lowercase=False)} for item in tags]
        if tags:
            task_run_request['tags'] = tags
        if orchestration_container:
            environment = [item for item in orchestration_container.get('environment', []) if item['name'] not in configuration.env.keys()] + [{'name': k, 'value': v} for k, v in configuration.env.items() if v is not None]
            if environment:
                orchestration_container['environment'] = environment
        overrides['containerOverrides'] = [v for v in container_overrides if v]
        return task_run_request

    @retry(
        stop=stop_after_attempt(MAX_CREATE_TASK_RUN_ATTEMPTS), 
        wait=wait_fixed(CREATE_TASK_RUN_MIN_DELAY_SECONDS) + wait_random(CREATE_TASK_RUN_MIN_DELAY_JITTER_SECONDS, CREATE_TASK_RUN_MAX_DELAY_JITTER_SECONDS), 
        reraise=True
    )
    def _create_task_run(self, ecs_client: _ECSClient, task_run_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a run of a task definition.
        Returns the task run details.
        """
        task: Dict[str, Any] = ecs_client.run_task(**task_run_request)
        if task['failures']:
            raise RuntimeError(f"Failed to run ECS task: {task['failures'][0]['reason']}")
        elif not task['tasks']:
            raise RuntimeError('Failed to run ECS task: no tasks or failures were returned.')
        return task['tasks'][0]

    def _task_definitions_equal(self, taskdef_1: Optional[Dict[str, Any]], taskdef_2: Optional[Dict[str, Any]]) -> bool:
        """
        Compare two task definitions.
        Since one may come from the AWS API and have populated defaults, we do our best
        to homogenize the definitions without changing their meaning.
        """
        if taskdef_1 == taskdef_2:
            return True
        if taskdef_1 is None or taskdef_2 is None:
            return False
        taskdef_1 = copy.deepcopy(taskdef_1)
        taskdef_2 = copy.deepcopy(taskdef_2)
        for taskdef in (taskdef_1, taskdef_2):
            container_definitions = taskdef.get('containerDefinitions', [])
            essential = any((container.get('essential') for container in container_definitions))
            if not essential and container_definitions:
                container_definitions[0].setdefault('essential', True)
            taskdef.setdefault('networkMode', 'bridge')
        _drop_empty_keys_from_dict(taskdef_1)
        _drop_empty_keys_from_dict(taskdef_2)
        for field in ECS_POST_REGISTRATION_FIELDS:
            taskdef_1.pop(field, None)
            taskdef_2.pop(field, None)
        return taskdef_1 == taskdef_2
