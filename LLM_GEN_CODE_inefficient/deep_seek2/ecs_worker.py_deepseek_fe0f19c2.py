from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Type, Union
from uuid import UUID
import copy
import json
import logging
import shlex
import sys
import time
from copy import deepcopy

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

# Internal type alias for ECS clients which are generated dynamically in botocore
_ECSClient = Any

ECS_DEFAULT_CONTAINER_NAME = "prefect"
ECS_DEFAULT_CPU = 1024
ECS_DEFAULT_COMMAND = "python -m prefect.engine"
ECS_DEFAULT_MEMORY = 2048
ECS_DEFAULT_LAUNCH_TYPE = "FARGATE"
ECS_DEFAULT_FAMILY = "prefect"
ECS_POST_REGISTRATION_FIELDS = [
    "compatibilities",
    "taskDefinitionArn",
    "revision",
    "status",
    "requiresAttributes",
    "registeredAt",
    "registeredBy",
    "deregisteredAt",
]

DEFAULT_TASK_DEFINITION_TEMPLATE = """
containerDefinitions:
- image: "{{ image }}"
  name: "{{ container_name }}"
cpu: "{{ cpu }}"
family: "{{ family }}"
memory: "{{ memory }}"
executionRoleArn: "{{ execution_role_arn }}"
"""

DEFAULT_TASK_RUN_REQUEST_TEMPLATE = """
launchType: "{{ launch_type }}"
cluster: "{{ cluster }}"
overrides:
  containerOverrides:
    - name: "{{ container_name }}"
      command: "{{ command }}"
      environment: "{{ env }}"
      cpu: "{{ cpu }}"
      memory: "{{ memory }}"
  cpu: "{{ cpu }}"
  memory: "{{ memory }}"
  taskRoleArn: "{{ task_role_arn }}"
tags: "{{ labels }}"
taskDefinition: "{{ task_definition_arn }}"
capacityProviderStrategy: "{{ capacity_provider_strategy }}"
"""

# Create task run retry settings
MAX_CREATE_TASK_RUN_ATTEMPTS = 3
CREATE_TASK_RUN_MIN_DELAY_SECONDS = 1
CREATE_TASK_RUN_MIN_DELAY_JITTER_SECONDS = 0
CREATE_TASK_RUN_MAX_DELAY_JITTER_SECONDS = 3

_TASK_DEFINITION_CACHE: Dict[UUID, str] = {}
_TAG_REGEX = r"[^a-zA-Z0-9_./=+:@-]"


class ECSIdentifier(NamedTuple):
    """
    The identifier for a running ECS task.
    """

    cluster: str
    task_arn: str


def _default_task_definition_template() -> dict:
    """
    The default task definition template for ECS jobs.
    """
    return yaml.safe_load(DEFAULT_TASK_DEFINITION_TEMPLATE)


def _default_task_run_request_template() -> dict:
    """
    The default task run request template for ECS jobs.
    """
    return yaml.safe_load(DEFAULT_TASK_RUN_REQUEST_TEMPLATE)


def _drop_empty_keys_from_dict(taskdef: dict) -> None:
    """
    Recursively drop keys with 'empty' values from a task definition dict.

    Mutates the task definition in place. Only supports recursion into dicts and lists.
    """
    for key, value in tuple(taskdef.items()):
        if not value:
            taskdef.pop(key)
        if isinstance(value, dict):
            _drop_empty_keys_from_dict(value)
        if isinstance(value, list) and key != "capacity_provider_strategy":
            for v in value:
                if isinstance(v, dict):
                    _drop_empty_keys_from_dict(v)


def _get_container(containers: List[dict], name: str) -> Optional[dict]:
    """
    Extract a container from a list of containers or container definitions.
    If not found, `None` is returned.
    """
    for container in containers:
        if container.get("name") == name:
            return container
    return None


def _container_name_from_task_definition(task_definition: dict) -> Optional[str]:
    """
    Attempt to infer the container name from a task definition.

    If not found, `None` is returned.
    """
    if task_definition:
        container_definitions = task_definition.get("containerDefinitions", [])
    else:
        container_definitions = []

    if _get_container(container_definitions, ECS_DEFAULT_CONTAINER_NAME):
        # Use the default container name if present
        return ECS_DEFAULT_CONTAINER_NAME
    elif container_definitions:
        # Otherwise, if there's at least one container definition try to get the
        # name from that
        return container_definitions[0].get("name")

    return None


def parse_identifier(identifier: str) -> ECSIdentifier:
    """
    Splits identifier into its cluster and task components, e.g.
    input "cluster_name::task_arn" outputs ("cluster_name", "task_arn").
    """
    cluster, task = identifier.split("::", maxsplit=1)
    return ECSIdentifier(cluster, task)


def mask_sensitive_env_values(
    task_run_request: dict, values: List[str], keep_length: int = 3, replace_with: str = "***"
) -> dict:
    for container in task_run_request.get("overrides", {}).get(
        "containerOverrides", []
    ):
        for env_var in container.get("environment", []):
            if (
                "name" not in env_var
                or "value" not in env_var
                or env_var["name"] not in values
            ):
                continue
            if len(env_var["value"]) > keep_length:
                # Replace characters beyond the keep length
                env_var["value"] = env_var["value"][:keep_length] + replace_with
    return task_run_request


def mask_api_key(task_run_request: dict) -> dict:
    return mask_sensitive_env_values(
        deepcopy(task_run_request), ["PREFECT_API_KEY"], keep_length=6
    )


class CapacityProvider(BaseModel):
    """
    The capacity provider strategy to use when running the task.
    """

    capacityProvider: str
    weight: int
    base: int


class ECSJobConfiguration(BaseJobConfiguration):
    """
    Job configuration for an ECS worker.
    """

    aws_credentials: Optional[AwsCredentials] = Field(default_factory=AwsCredentials)
    task_definition: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra=dict(template=_default_task_definition_template()),
    )
    task_run_request: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra=dict(template=_default_task_run_request_template()),
    )
    configure_cloudwatch_logs: Optional[bool] = Field(default=None)
    cloudwatch_logs_options: Dict[str, str] = Field(default_factory=dict)
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

    execution_role_arn: Optional[str] = Field(
        title="Execution Role ARN",
        default=None,
        description=(
            "An execution role to use for the task. This controls the permissions of "
            "the task when it is launching. If this value is not null, it will "
            "override the value in the task definition. An execution role must be "
            "provided to capture logs from the container."
        ),
    )

    @model_validator(mode="after")
    def task_run_request_requires_arn_if_no_task_definition_given(self) -> Self:
        """
        If no task definition is provided, a task definition ARN must be present on the
        task run request.
        """
        if (
            not (self.task_run_request or {}).get("taskDefinition")
            and not self.task_definition
        ):
            raise ValueError(
                "A task definition must be provided if a task definition ARN is not "
                "present on the task run request."
            )
        return self

    @model_validator(mode="after")
    def container_name_default_from_task_definition(self) -> Self:
        """
        Infers the container name from the task definition if not provided.
        """
        if self.container_name is None:
            self.container_name = _container_name_from_task_definition(
                self.task_definition
            )

            # We may not have a name here still; for example if someone is using a task
            # definition arn. In that case, we'll perform similar logic later to find
            # the name to treat as the "orchestration" container.

        return self

    @model_validator(mode="after")
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

    @model_validator(mode="after")
    def configure_cloudwatch_logs_requires_execution_role_arn(
        self,
    ) -> Self:
        """
        Enforces that an execution role arn is provided (or could be provided by a
        runtime task definition) when configuring logging.
        """
        if (
            self.configure_cloudwatch_logs
            and not self.execution_role_arn
            # TODO: Does not match
            # Do not raise if they've linked to another task definition or provided
            # it without using our shortcuts
            and not (self.task_run_request or {}).get("taskDefinition")
            and not (self.task_definition or {}).get("executionRoleArn")
        ):
            raise ValueError(
                "An `execution_role_arn` must be provided to use "
                "`configure_cloudwatch_logs` or `stream_logs`."
            )
        return self

    @model_validator(mode="after")
    def cloudwatch_logs_options_requires_configure_cloudwatch_logs(
        self,
    ) -> Self:
        """
        Enforces that an execution role arn is provided (or could be provided by a
        runtime task definition) when configuring logging.
        """
        if self.cloudwatch_logs_options and not self.configure_cloudwatch_logs:
            raise ValueError(
                "`configure_cloudwatch_log` must be enabled to use "
                "`cloudwatch_logs_options`."
            )
        return self

    @model_validator(mode="after")
    def network_configuration_requires_vpc_id(self) -> Self:
        """
        Enforces a `vpc_id` is provided when custom network configuration mode is
        enabled for network settings.
        """
        if self.network_configuration and not self.vpc_id:
            raise ValueError(
                "You must provide a `vpc_id` to enable custom `network_configuration`."
            )
        return self

    @classmethod
    @inject_client
    async def from_template_and_values(
        cls,
        base_job_template: dict,
        values: dict,
        client: Optional[PrefectClient] = None,
    ) -> Self:
        """Creates a valid worker configuration object from the provided base
        configuration and overrides.

        Important: this method expects that the base_job_template was already
        validated server-side.
        """

        job_config: Dict[str, Any] = base_job_template["job_configuration"]
        variables_schema = base_job_template["variables"]
        variables = cls._get_base_config_defaults(
            variables_schema.get("properties", {})
        )
        variables.update(values)

        _drop_empty_keys_from_dict(variables)  # TODO: investigate why this is necessary

        populated_configuration = apply_values(template=job_config, values=variables)
        populated_configuration = await resolve_block_document_references(
            template=populated_configuration, client=client
        )
        populated_configuration = await resolve_variables(
            template=populated_configuration, client=client
        )
        return cls(**populated_configuration)


class ECSVariables(BaseVariables):
    """
    Variables for templating an ECS job.
    """

    task_definition_arn: Optional[str] = Field(
        default=None,
        description=(
            "An identifier for an existing task definition to use. If set, options that"
            " require changes to the task definition will be ignored. All contents of "
            "the task definition in the job configuration will be ignored."
        ),
    )
    env: Dict[str, Optional[str]] = Field(
        title="Environment Variables",
        default_factory=dict,
        description=(
            "Environment variables to provide to the task run. These variables are set "
            "on the Prefect container at task runtime. These will not be set on the "
            "task definition."
        ),
    )
    aws_credentials: AwsCredentials = Field(
        title="AWS Credentials",
        default_factory=AwsCredentials,
        description=(
            "The AWS credentials to use to connect to ECS. If not provided, credentials"
            " will be inferred from the local environment following AWS's boto client's"
            " rules."
        ),
    )
    cluster: Optional[str] = Field(
        default=None,
        description=(
            "The ECS cluster to run the task in. An ARN or name may be provided. If "
            "not provided, the default cluster will be used."
        ),
    )
    family: Optional[str] = Field(
        default=None,
        description=(
            "A family for the task definition. If not provided, it will be inferred "
            "from the task definition. If the task definition does not have a family, "
            "the name will be generated. When flow and deployment metadata is "
            "available, the generated name will include their names. Values for this "
            "field will be slugified to match AWS character requirements."
        ),
    )
    launch_type: Literal["FARGATE", "EC2", "EXTERNAL", "FARGATE_SPOT"] = Field(
        default=ECS_DEFAULT_LAUNCH_TYPE,
        description=(
            "The type of ECS task run infrastructure that should be used. Note that"
            " 'FARGATE_SPOT' is not a formal ECS launch type, but we will configure"
            " the proper capacity provider strategy if set here."
        ),
    )
    capacity_provider_strategy: List[CapacityProvider] = Field(
        default_factory=list,
        description=(
            "The capacity provider strategy to use when running the task. "
            "If a capacity provider strategy is specified, the selected launch"
            " type will be ignored."
        ),
    )
    image: Optional[str] = Field(
        default=None,
        description=(
            "The image to use for the Prefect container in the task. If this value is "
            "not null, it will override the value in the task definition. This value "
            "defaults to a Prefect base image matching your local versions."
        ),
    )
    cpu: Optional[int] = Field(
        title="CPU",
        default=None,
        description=(
            "The amount of CPU to provide to the ECS task. Valid amounts are "
            "specified in the AWS documentation. If not provided, a default value of "
            f"{ECS_DEFAULT_CPU} will be used unless present on the task definition."
        ),
    )
    memory: Optional[int] = Field(
        default=None,
        description=(
            "The amount of memory to provide to the ECS task. Valid amounts are "
            "specified in the AWS documentation. If not provided, a default value of "
            f"{ECS_DEFAULT_MEMORY} will be used unless present on the task definition."
        ),
    )
    container_name: Optional[str] = Field(
        default=None,
        description=(
            "The name of the container flow run orchestration will occur in. If not "
            f"specified, a default value of {ECS_DEFAULT_CONTAINER_NAME} will be used "
            "and if that is not found in the task definition the first container will "
            "be used."
        ),
    )
    task_role_arn: Optional[str] = Field(
        title="Task Role ARN",
        default=None,
        description=(
            "A role to attach to the task run. This controls the permissions of the "
            "task while it is running."
        ),
    )
    execution_role_arn: Optional[str] = Field(
        title="Execution Role ARN",
        default=None,
        description=(
            "An execution role to use for the task. This controls the permissions of "
            "the task when it is launching. If this value is not null, it will "
            "override the value in the task definition. An execution role must be "
            "provided to capture logs from the container."
        ),
    )
    vpc_id: Optional[str] = Field(
        title="VPC ID",
        default=None,
        description=(
            "The AWS VPC to link the task run to. This is only applicable when using "
            "the 'awsvpc' network mode for your task. FARGATE tasks require this "
            "network  mode, but for EC2 tasks the default network mode is 'bridge'. "
            "If using the 'awsvpc' network mode and this field is null, your default "
            "VPC will be used. If no default VPC can be found, the task run will fail."
        ),
    )
    configure_cloudwatch_logs: Optional[bool] = Field(
        default=None,
        description=(
            "If enabled, the Prefect container will be configured to send its output "
            "to the AWS CloudWatch logs service. This functionality requires an "
            "execution role with logs:CreateLogStream, logs:CreateLogGroup, and "
            "logs:PutLogEvents permissions. The default for this field is `False` "
            "unless `stream_output` is set."
        ),
    )
    cloudwatch_logs_options: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "When `configure_cloudwatch_logs` is enabled, this setting may be used to"
            " pass additional options to the CloudWatch logs configuration or override"
            " the default options. See the [AWS"
            " documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/using_awslogs.html#create_awslogs_logdriver_options)"  # noqa
            " for available options. "
        ),
    )
    cloudwatch_logs_prefix: Optional[str] = Field(
        default=None,
        description=(
            "When `configure_cloudwatch_logs` is enabled, this setting may be used to"
            " set a prefix for the log group. If not provided, the default prefix will"
            " be `prefect-logs_<work_pool_name>_<deployment_id>`. If"
            " `awslogs-stream-prefix` is present in `Cloudwatch logs options` this"
            " setting will be ignored."
        ),
    )

    network_configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "When `network_configuration` is supplied it will override ECS Worker's"
            "awsvpcConfiguration that defined in the ECS task executing your workload. "
            "See the [AWS documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-awsvpcconfiguration.html)"  # noqa
            " for available options."
        ),
    )

    stream_output: Optional[bool] = Field(
        default=None,
        description=(
            "If enabled, logs will be streamed from the Prefect container to the local "
            "console. Unless you have configured AWS CloudWatch logs manually on your "
            "task definition, this requires the same prerequisites outlined in "
            "`configure_cloudwatch_logs`."
        ),
    )
    task_start_timeout_seconds: int = Field(
        default=300,
        description=(
            "The amount of time to watch for the start of the ECS task "
            "before marking it as failed. The task must enter a RUNNING state to be "
            "considered started."
        ),
    )
    task_watch_poll_interval: float = Field(
        default=5.0,
        description=(
            "The amount of time to wait between AWS API calls while monitoring the "
            "state of an ECS task."
        ),
    )
    auto_deregister_task_definition: bool = Field(
        default=False,
        description=(
            "If enabled, any task definitions that are created by this block will be "
            "deregistered. Existing task definitions linked by ARN will never be "
            "deregistered. Deregistering a task definition does not remove it from "
            "your AWS account, instead it will be marked as INACTIVE."
        ),
    )
    match_latest_revision_in_family: bool = Field(
        default=False,
        description=(
            "If enabled, the most recent active revision in the task definition "
            "family will be compared against the desired ECS task configuration. "
            "If they are equal, the existing task definition will be used instead "
            "of registering a new one. If no family is specified the default family "
            f'"{ECS_DEFAULT_FAMILY}" will be used.'
        ),
    )


class ECSWorkerResult(BaseWorkerResult):
    """
    The result of an ECS job.
    """


class ECSWorker(BaseWorker):
    """
    A Prefect worker to run flow runs as ECS tasks.
    """

    type: str = "ecs"
    job_configuration: Type[ECSJobConfiguration] = ECSJobConfiguration
    job_configuration_variables: Type[ECSVariables] = ECSVariables
    _description: str = (
        "Execute flow runs within containers on AWS ECS. Works with EC2 "
        "and Fargate clusters. Requires an AWS account."
    )
    _display_name = "AWS Elastic Container Service"
    _documentation_url = "https://docs.prefect.io/integrations/prefect-aws/"
    _logo_url = "https://cdn.sanity.io/images/3ugk85nk/production/d74b16fe84ce626345adf235a47008fea2869a60-225x225.png"  # noqa

    async def run(
        self,
        flow_run: "FlowRun",
        configuration: ECSJobConfiguration,
        task_status: Optional[anyio.abc.TaskStatus] = None,
    ) -> ECSWorkerResult:
        """
        Runs a given flow run on the current worker.
        """
        ecs_client = await run_sync_in_worker_thread(
            self._get_client, configuration, "ecs"
        )

        logger = self.get_flow_run_logger(flow_run)

        (
            task_arn,
            cluster_arn,
            task_definition,
            is_new_task_definition,
        ) = await run_sync_in_worker_thread(
            self._create_task_and_wait_for_start,
            logger,
            ecs_client,
            configuration,
            flow_run,
        )

        # The task identifier is "{cluster}::{task}" where we use the configured cluster
        # if set to preserve matching by name rather than arn
        # Note "::" is used despite the Prefect standard being ":" because ARNs contain
        # single colons.
        identifier = (
            (configuration.cluster if configuration.cluster else cluster_arn)
            + "::"
            + task_arn
        )

        if task_status:
            task_status.started(identifier)

        status_code = await run_sync_in_worker_thread(
            self._watch_task_and_get_exit_code,
            logger,
            configuration,
            task_arn,
            cluster_arn,
            task_definition,
            is_new_task_definition and configuration.auto_deregister_task_definition,
            ecs_client,
        )

        return ECSWorkerResult(
            identifier=identifier,
            # If the container does not start the exit code can be null but we must
            # still report a status code. We use a -1 to indicate a special code.
            status_code=status_code if status_code is not None else -1,
        )

    def _get_client(
        self, configuration: ECSJobConfiguration, client_type: Union[str, ClientType]
    ) -> _ECSClient:
        """
        Get a boto3 client of client_type. Will use a cached client if one exists.
        """
        return configuration.aws_credentials.get_client(client_type)

    def _create_task_and_wait_for_start(
        self,
        logger: logging.Logger,
        ecs_client: _ECSClient,
        configuration: ECSJobConfiguration,
        flow_run: FlowRun,
    ) -> Tuple[str, str, dict, bool]:
        """
        Register the task definition, create the task run, and wait for it to start.

        Returns a tuple of
        - The task ARN
        - The task's cluster ARN
        - The task definition
        - A bool indicating if the task definition is newly registered
        """
        task_definition_arn = configuration.task_run_request.get("taskDefinition")
        new_task_definition_registered = False

        if not task_definition_arn:
            task_definition = self._prepare_task_definition(
                configuration, region=ecs_client.meta.region_name, flow_run=flow_run
            )

            (
                task_definition_arn,
                new_task_definition_registered,
            ) = self._get_or_register_task_definition(
                logger, ecs_client, configuration, flow_run, task_definition
            )
        else:
            task_definition = self._retrieve_task_definition(
                logger, ecs_client, task_definition_arn
            )
            if configuration.task_definition:
                logger.warning(
                    "Ignoring task definition in configuration since task definition"
                    " ARN is provided on the task run request."
                )

        self._validate_task_definition(task_definition, configuration)

        _TASK_DEFINITION_CACHE[flow_run.deployment_id] = task_definition_arn

        logger.info(f"Using ECS task definition {task_definition_arn!r}...")
        logger.debug(
            f"Task definition {json.dumps(task_definition, indent=2, default=str)}"
        )

        task_run_request = self._prepare_task_run_request(
            configuration,
            task_definition,
            task_definition_arn,
        )

        logger.info("Creating ECS task run...")
        logger.debug(
            "Task run request"
            f"{json.dumps(mask_api_key(task_run_request), indent=2, default=str)}"
        )

        try:
            task = self._create_task_run(ecs_client, task_run_request)
            task_arn = task["taskArn"]
            cluster_arn = task["clusterArn"]
        except Exception as exc:
            self._report_task_run_creation_failure(configuration, task_run_request, exc)
            raise

        logger.info("Waiting for ECS task run to start...")
        self._wait_for_task_start(
            logger,
            configuration,
            task_arn,
            cluster_arn,
            ecs_client,
            timeout=configuration.task_start_timeout_seconds,
        )

        return task_arn, cluster_arn, task_definition, new_task_definition_registered

    def _get_or_register_task_definition(
        self,
        logger: logging.Logger,
        ecs_client: _ECSClient,
        configuration: ECSJobConfiguration,
        flow_run: FlowRun,
        task_definition: dict,
    ) -> Tuple[str, bool]:
        """Get or register a task definition for the given flow run.

        Returns a tuple of the task definition ARN and a bool indicating if the task
        definition is newly registered.
        """

        cached_task_definition_arn = _TASK_DEFINITION_CACHE.get(flow_run.deployment_id)
        new_task_definition_registered = False

        if cached_task_definition_arn:
            try:
                cached_task_definition = self._retrieve_task_definition(
                    logger, ecs_client, cached_task_definition_arn
                )
                if not cached_task_definition[
                    "status"
                ] == "ACTIVE" or not self._task_definitions_equal(
                    task_definition, cached_task_definition
                ):
                    cached_task_definition_arn = None
            except Exception:
                cached_task_definition_arn = None

        if (
            not cached_task_definition_arn
            and configuration.match_latest_revision_in_family
        ):
            family_name = task_definition.get("family", ECS_DEFAULT_FAMILY)
            try:
                task_definition_from_family = self._retrieve_task_definition(
                    logger, ecs_client, family_name
                )
                if task_definition_from_family and self._task_definitions_equal(
                    task_definition, task_definition_from_family
                ):
                    cached_task_definition_arn = task_definition_from_family[
                        "taskDefinitionArn"
                    ]
            except Exception:
                cached_task_definition_arn = None

        if not cached_task_definition_arn:
            task_definition_arn = self._register_task_definition(
                logger, ecs_client, task_definition
            )
            new_task_definition_registered = True
        else:
            task_definition_arn = cached_task_definition_arn

        return task_definition_arn, new_task_definition_registered

    def _watch_task_and_get_exit_code(
        self,
        logger: logging.Logger,
        configuration: ECSJobConfiguration,
        task_arn: str,
        cluster_arn: str,
        task_definition: dict,
        deregister_task_definition: bool,
        ecs_client: _ECSClient,
    ) -> Optional[int]:
        """
        Wait for the task run to complete and retrieve the exit code of the Prefect
        container.
        """

        # Wait for completion and stream logs
        task = self._wait_for_task_finish(
            logger,
            configuration,
            task_arn,
            cluster_arn,
            task_definition,
            ecs_client,
        )

        if deregister_task_definition:
            ecs_client.deregister_task_definition(
                taskDefinition=task["taskDefinitionArn"]
            )

        container_name = (
            configuration.container_name
            or _container_name_from_task_definition(task_definition)
            or ECS_DEFAULT_CONTAINER_NAME
        )

        # Check the status code of the Prefect container
        container = _get_container(task["containers"], container_name)
        assert container is not None, (
            f"'{container_name}' container missing from task: {task}"
        )
        status_code = container.get("exitCode")
        self._report_container_status_code(logger, container_name, status_code)

        return status_code

    def _report_container_status_code(
        self, logger: logging.Logger, name: str, status_code: Optional[int]
    ) -> None:
        """
        Display a log for the given container status code.
        """
        if status_code is None:
            logger.error(
                f"Task exited without reporting an exit status for container {name!r}."
            )
        elif status_code == 0:
            logger.info(f"Container {name!r} exited successfully.")
        else:
            logger.warning(
                f"Container {name!r} exited with non-zero exit code {status_code}."
            )

    def _report_task_run_creation_failure(
        self, configuration: ECSJobConfiguration, task_run: dict, exc: Exception
    ) -> None:
        """
        Wrap common AWS task run creation failures with nicer user-facing messages.
        """
        # AWS generates exception types at runtime so they must be captured a bit
        # differently than normal.
        if "ClusterNotFoundException" in str(exc):
            cluster = task_run.get("cluster", "default")
            raise RuntimeError(
                f"Failed to run ECS task, cluster {cluster!r} not found. "
                "Confirm that the cluster is configured in your region."
            ) from exc
        elif (
            "No Container Instances" in str(exc) and task_run.get("launchType") == "EC2"
        ):
            cluster = task_run.get("cluster", "default")
            raise RuntimeError(
                f"Failed to run ECS task, cluster {cluster!r} does not appear to "
                "have any container instances associated with it. Confirm that you "
                "have EC2 container instances available."
            ) from exc
        elif (
            "failed to validate logger args" in str(exc)
            and "AccessDeniedException" in str(exc)
            and configuration.configure_cloudwatch_logs
        ):
            raise RuntimeError(
                "Failed to run ECS task, the attached execution role does not appear"
                " to have sufficient permissions. Ensure that the execution role"
                f" {configuration.execution_role!r} has permissions"
                " logs:CreateLogStream, logs:CreateLogGroup, and logs:PutLogEvents."
            )
        else:
            raise

    def _validate_task_definition(
        self, task_definition: dict, configuration: ECSJobConfiguration
    ) -> None:
        """
        Ensure that the task definition is compatible with the configuration.

        Raises `ValueError` on incompatibility. Returns `None` on success.
        """
        launch_type = configuration.task_run_request.get(
            "launchType", ECS_DEFAULT_LAUNCH_TYPE
        )
        if (
            launch_type != "EC2"
            and "FARGATE" not in task_definition["requiresCompatibilities"]
        ):
            raise ValueError(
                "Task definition does not have 'FARGATE' in 'requiresCompatibilities'"
                f" and cannot be used with launch type {launch_type!r}"
            )

        if launch_type == "FARGATE" or launch_type == "FARGATE_SPOT":
            # Only the 'awsvpc' network mode is supported when using FARGATE
            network_mode = task_definition.get("networkMode")
            if network_mode != "awsvpc":
                raise ValueError(
                    f"Found network mode {network_mode!r} which is not compatible with "
                    f"launch type {launch_type!r}. Use either the 'EC2' launch "
                    "type or the 'awsvpc' network mode."
                )

        if configuration.configure_cloudwatch_logs and not task_definition.get(
            "executionRoleArn"
        ):
            raise ValueError(
                "An execution role arn must be set on the task definition to use "
                "`configure_cloudwatch_logs` or `stream_logs` but no execution role "
                "was found on the task definition."
            )

    def _register_task_definition(
        self,
        logger: logging.Logger,
        ecs_client: _ECSClient,
        task_definition: dict,
    ) -> str:
        """
        Register a new task definition with AWS.

        Returns the ARN.
        """
        logger.info("Registering ECS task definition...")
        logger.debug(
            "Task definition request"
            f"{json.dumps(task_definition, indent=2, default=str