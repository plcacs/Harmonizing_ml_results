from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union
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
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult, apply_values, resolve_block_document_references, resolve_variables
from prefect_aws.credentials import AwsCredentials, ClientType

_ECSClient = Any
ECS_DEFAULT_CONTAINER_NAME: str = 'prefect'
ECS_DEFAULT_CPU: int = 1024
ECS_DEFAULT_COMMAND: str = 'python -m prefect.engine'
ECS_DEFAULT_MEMORY: int = 2048
ECS_DEFAULT_LAUNCH_TYPE: str = 'FARGATE'
ECS_DEFAULT_FAMILY: str = 'prefect'
ECS_POST_REGISTRATION_FIELDS: List[str] = ['compatibilities', 'taskDefinitionArn', 'revision', 'status', 'requiresAttributes', 'registeredAt', 'registeredBy', 'deregisteredAt']
DEFAULT_TASK_DEFINITION_TEMPLATE: str = '\ncontainerDefinitions:\n- image: "{{ image }}"\n  name: "{{ container_name }}"\ncpu: "{{ cpu }}"\nfamily: "{{ family }}"\nmemory: "{{ memory }}"\nexecutionRoleArn: "{{ execution_role_arn }}"\n'
DEFAULT_TASK_RUN_REQUEST_TEMPLATE: str = '\nlaunchType: "{{ launch_type }}"\ncluster: "{{ cluster }}"\noverrides:\n  containerOverrides:\n    - name: "{{ container_name }}"\n      command: "{{ command }}"\n      environment: "{{ env }}"\n      cpu: "{{ cpu }}"\n      memory: "{{ memory }}"\n  cpu: "{{ cpu }}"\n  memory: "{{ memory }}"\n  taskRoleArn: "{{ task_role_arn }}"\ntags: "{{ labels }}"\ntaskDefinition: "{{ task_definition_arn }}"\ncapacityProviderStrategy: "{{ capacity_provider_strategy }}"\n'
MAX_CREATE_TASK_RUN_ATTEMPTS: int = 3
CREATE_TASK_RUN_MIN_DELAY_SECONDS: int = 1
CREATE_TASK_RUN_MIN_DELAY_JITTER_SECONDS: int = 0
CREATE_TASK_RUN_MAX_DELAY_JITTER_SECONDS: int = 3
_TASK_DEFINITION_CACHE: Dict[UUID, str] = {}
_TAG_REGEX: str = '[^a-zA-Z0-9_./=+:@-]'

class ECSIdentifier(NamedTuple):
    cluster: str
    task: str

def _default_task_definition_template() -> Dict[str, Any]:
    return yaml.safe_load(DEFAULT_TASK_DEFINITION_TEMPLATE)

def _default_task_run_request_template() -> Dict[str, Any]:
    return yaml.safe_load(DEFAULT_TASK_RUN_REQUEST_TEMPLATE)

def _drop_empty_keys_from_dict(taskdef: Dict[str, Any]) -> None:
    for key, value in tuple(taskdef.items()):
        if not value:
            taskdef.pop(key)
        if isinstance(value, dict):
            _drop_empty_keys_from_dict(value)
        if isinstance(value, list) and key != 'capacity_provider_strategy':
            for v in value:
                if isinstance(v, dict):
                    _drop_empty_keys_from_dict(v)

def _get_container(containers: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for container in containers:
        if container.get('name') == name:
            return container
    return None

def _container_name_from_task_definition(task_definition: Optional[Dict[str, Any]]) -> Optional[str]:
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
    cluster, task = identifier.split('::', maxsplit=1)
    return ECSIdentifier(cluster, task)

def mask_sensitive_env_values(task_run_request: Dict[str, Any], values: List[str], keep_length: int = 3, replace_with: str = '***') -> Dict[str, Any]:
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
    capacityProvider: str
    weight: int
    base: Optional[int] = None

class ECSJobConfiguration(BaseJobConfiguration):
    aws_credentials: AwsCredentials = Field(default_factory=AwsCredentials)
    task_definition: Dict[str, Any] = Field(default_factory=dict, json_schema_extra=dict(template=_default_task_definition_template()))
    task_run_request: Dict[str, Any] = Field(default_factory=dict, json_schema_extra=dict(template=_default_task_run_request_template()))
    configure_cloudwatch_logs: Optional[bool] = Field(default=None)
    cloudwatch_logs_options: Dict[str, Any] = Field(default_factory=dict)
    cloudwatch_logs_prefix: Optional[str] = Field(default=None)
    network_configuration: Dict[str, Any] = Field(default_factory=dict)
    stream_output: Optional[bool] = Field(default=None)
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
        if not (self.task_run_request or {}).get('taskDefinition') and (not self.task_definition):
            raise ValueError('A task definition must be provided if a task definition ARN is not present on the task run request.')
        return self

    @model_validator(mode='after')
    def container_name_default_from_task_definition(self) -> Self:
        if self.container_name is None:
            self.container_name = _container_name_from_task_definition(self.task_definition)
        return self

    @model_validator(mode='after')
    def set_default_configure_cloudwatch_logs(self) -> Self:
        configure_cloudwatch_logs = self.configure_cloudwatch_logs
        if configure_cloudwatch_logs is None:
            self.configure_cloudwatch_logs = self.stream_output
        return self

    @model_validator(mode='after')
    def configure_cloudwatch_logs_requires_execution_role_arn(self) -> Self:
        if self.configure_cloudwatch_logs and (not self.execution_role_arn) and (not (self.task_run_request or {}).get('taskDefinition')) and (not (self.task_definition or {}).get('executionRoleArn')):
            raise ValueError('An `execution_role_arn` must be provided to use `configure_cloudwatch_logs` or `stream_logs`.')
        return self

    @model_validator(mode='after')
    def cloudwatch_logs_options_requires_configure_cloudwatch_logs(self) -> Self:
        if self.cloudwatch_logs_options and (not self.configure_cloudwatch_logs):
            raise ValueError('`configure_cloudwatch_log` must be enabled to use `cloudwatch_logs_options`.')
        return self

    @model_validator(mode='after')
    def network_configuration_requires_vpc_id(self) -> Self:
        if self.network_configuration and (not self.vpc_id):
            raise ValueError('You must provide a `vpc_id` to enable custom `network_configuration`.')
        return self

    @classmethod
    @inject_client
    async def from_template_and_values(cls, base_job_template: Dict[str, Any], values: Dict[str, Any], client: Optional[PrefectClient] = None) -> Self:
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
    task_definition_arn: Optional[str] = Field(default=None, description='An identifier for an existing task definition to use. If set, options that require changes to the task definition will be ignored. All contents of the task definition in the job configuration will be ignored.')
    env: Dict[str, Optional[str]] = Field(title='Environment Variables', default_factory=dict, description='Environment variables to provide to the task run. These variables are set on the Prefect container at task runtime. These will not be set on the task definition.')
    aws_credentials: AwsCredentials = Field(title='AWS Credentials', default_factory=AwsCredentials, description="The AWS credentials to use to connect to ECS. If not provided, credentials will be inferred from the local environment following AWS's boto client's rules.")
    cluster: Optional[str] = Field(default=None, description='The ECS cluster to run the task in. An ARN or name may be provided. If not provided, the default cluster will be used.')
    family: Optional[str] = Field(default=None, description='A family for the task definition. If not provided, it will be inferred from the task definition. If the task definition does not have a family, the name will be generated. When flow and deployment metadata is available, the generated name will include their names. Values for this field will be slugified to match AWS character requirements.')
    launch_type: str = Field(default=ECS_DEFAULT_LAUNCH_TYPE, description="The type of ECS task run infrastructure that should be used. Note that 'FARGATE_SPOT' is not a formal ECS launch type, but we will configure the proper capacity provider strategy if set here.")
    capacity_provider_strategy: List[CapacityProvider] = Field(default_factory=list, description='The capacity provider strategy to use when running the task. If a capacity provider strategy is specified, the selected launch type will be ignored.')
    image: Optional[str] = Field(default=None, description='The image to use for the Prefect container in the task. If this value is not null, it will override the value in the task definition. This value defaults to a Prefect base image matching your local versions.')
    cpu: Optional[Union[int, str]] = Field(title='CPU', default=None, description=f'The amount of CPU to provide to the ECS task. Valid amounts are specified in the AWS documentation. If not provided, a default value of {ECS_DEFAULT_CPU} will be used unless present on the task definition.')
    memory: Optional[Union[int, str]] = Field(default=None, description=f'The amount of memory to provide to the ECS task. Valid amounts are specified in the AWS documentation. If not provided, a default value of {ECS_DEFAULT_MEMORY} will be used unless present on the task definition.')
    container_name: Optional[str] = Field(default=None, description=f'The name of the container flow run orchestration will occur in. If not specified, a default value of {ECS_DEFAULT_CONTAINER_NAME} will be used and if that is not found in the task definition the first container will be used.')
    task_role_arn: Optional[str] = Field(title='Task Role ARN', default=None, description='A role to attach to the task run. This controls the permissions of the task while it is running.')
    execution_role_arn: Optional[str] = Field(title='Execution Role ARN', default=None, description='An execution role to use for the task. This controls the permissions of the task when it is launching. If this value is not null, it will override the value in the task definition. An execution role must be provided to capture logs from the container.')
    vpc_id: Optional[str] = Field(title='VPC ID', default=None, description="The AWS VPC to link the task run to. This is only applicable when using the 'awsvpc' network mode for your task. FARGATE tasks require this network  mode, but for EC2 tasks the default network mode is 'bridge'. If using the 'awsvpc' network mode and this field is null, your default VPC will be used. If no default VPC can be found, the task run will fail.")
    configure_cloudwatch_logs: Optional[bool] = Field(default=None, description='If enabled, the Prefect container will be configured to send its output to the AWS CloudWatch logs service. This functionality requires an execution role with logs:CreateLogStream, logs:CreateLogGroup, and logs:PutLogEvents permissions. The default for this field is `False` unless `stream_output` is set.')
    cloudwatch_logs_options: Dict[str, Any] = Field(default_factory=dict, description='When `configure_cloudwatch_logs` is enabled, this setting may be used to pass additional options to the CloudWatch logs configuration or override the default options. See the [AWS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_awslogs.html#create_awslogs_logdriver_options) for available options. ')
    cloudwatch_logs_prefix: Optional[str] = Field(default=None, description='When `configure_cloudwatch_logs` is enabled, this setting may be used to set a prefix for the log group. If not provided, the default prefix will be `prefect-logs_<work_pool_name>_<deployment_id>`. If `awslogs-stream-prefix` is present in `Cloudwatch logs options` this setting will be ignored.')
    network_configuration: Dict[str, Any] = Field(default_factory=dict, description="When `network_configuration` is supplied it will override ECS Worker'sawsvpcConfiguration that defined in the ECS task executing your workload. See the [AWS documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-awsvpcconfiguration.html) for available options.")
    stream_output: Optional[bool] = Field(default=None, description='If enabled, logs will be streamed from the Prefect container to the local console. Unless you have configured AWS CloudWatch logs manually on your task definition, this requires the same prerequisites outlined in `configure_cloudwatch_logs`.')
    task_start_timeout_seconds: int = Field(default=300, description='The amount of time to watch for the start of the ECS task before marking it as failed. The task must enter a RUNNING state to be considered started.')
    task_watch_poll_interval: float = Field(default=5.0, description='The amount of time to wait between AWS API calls while monitoring the state of an ECS task.')
    auto_deregister_task_definition: bool = Field(default=False, description='If enabled, any task definitions that are created by this block will be deregistered. Existing task definitions linked by ARN will never be deregistered. Deregistering a task definition does not remove it from your AWS account, instead it will be marked as INACTIVE.')
    match_latest_revision_in_family: bool = Field(default=False, description=f'If enabled, the most recent active revision in the task definition family will be compared against the desired ECS task configuration. If they are equal, the existing task definition will be used instead of registering a new one. If no family is specified the default family "{ECS_DEFAULT_FAMILY}" will be used.')

class ECSWorkerResult(BaseWorkerResult):
    identifier: str
    status_code: int

class ECSWorker(BaseWorker):
    type: str = 'ecs'
    job_configuration: Type[ECSJobConfiguration] = ECSJobConfiguration
    job_configuration_variables: Type[ECSVariables] = ECSVariables
    _description: str = 'Execute flow runs within containers on AWS ECS. Works with EC2 and Fargate clusters. Requires an AWS account.'
    _display_name: str = 'AWS Elastic Container Service'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-aws/'
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png'

    async def run(self, flow_run: FlowRun, configuration: ECSJobConfiguration, task_status: Optional[anyio.abc.T