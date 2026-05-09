"""Chalice deployer module."""

import json
import textwrap
import socket
import logging
import botocore.exceptions
from botocore.vendored.requests import ConnectionError as RequestsConnectionError
from botocore.session import Session
from typing import (
    Optional,
    Dict,
    List,
    Any,
    Type,
    cast,
    Union,
    Callable,
    Iterable,
    overload,
)
from chalice.config import Config
from chalice.compat import is_broken_pipe_error
from chalice.awsclient import (
    DeploymentPackageTooLargeError,
    LambdaClientError,
    AWSClientError,
    TypedAWSClient,
)
from chalice.constants import (
    MAX_LAMBDA_DEPLOYMENT_SIZE,
    VPC_ATTACH_POLICY,
    DEFAULT_LAMBDA_TIMEOUT,
    DEFAULT_LAMBDA_MEMORY_SIZE,
    DEFAULT_TLS_VERSION,
    SQS_EVENT_SOURCE_POLICY,
    KINESIS_EVENT_SOURCE_POLICY,
    DDB_EVENT_SOURCE_POLICY,
    POST_TO_WEBSOCKET_CONNECTION_POLICY,
)
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.executor import BaseExecutor, Executor, DisplayOnlyExecutor
from chalice.deploy.packager import (
    PipRunner,
    SubprocessPip,
    PipDependencyBuilder,
    LambdaDeploymentPackager,
    AppOnlyDeploymentPackager,
    LayerDeploymentPackager,
    BaseLambdaDeploymentPackager,
    EmptyPackageError,
)
from chalice.deploy.planner import PlanStage, RemoteState, NoopPlanner
from chalice.deploy.swagger import TemplatedSwaggerGenerator, SwaggerGenerator
from chalice.deploy.sweeper import ResourceSweeper
from chalice.deploy.validate import validate_configuration
from chalice.policy import AppPolicyGenerator
from chalice.utils import OSUtils, UI, serialize_to_json

OptStr = Optional[str]
LOGGER = logging.getLogger(__name__)
_AWSCLIENT_EXCEPTIONS = (botocore.exceptions.ClientError, AWSClientError)

class ChaliceDeploymentError(Exception):
    def __init__(self, error: Exception) -> None:
        ...

class Deployer:
    BACKEND_NAME: str = 'api'

    def __init__(
        self,
        application_builder: ApplicationGraphBuilder,
        deps_builder: DependencyBuilder,
        build_stage: BuildStage,
        plan_stage: PlanStage,
        sweeper: ResourceSweeper,
        executor: BaseExecutor,
        recorder: ResultsRecorder,
    ) -> None:
        ...

    def deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        ...

    def _deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        ...

    def _validate_config(self, config: Config) -> None:
        ...

class BaseDeployStep:
    def handle(self, config: Config, resource: models.Model) -> None:
        ...

class InjectDefaults(BaseDeployStep):
    def __init__(
        self,
        lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT,
        lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE,
        tls_version: str = DEFAULT_TLS_VERSION,
    ) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        ...

    def handle_domainname(self, config: Config, resource: models.DomainName) -> None:
        ...

class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None:
        ...

    def handle_deploymentpackage(self, config: Config, resource: models.DeploymentPackage) -> None:
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(
        self,
        lambda_packager: AppOnlyDeploymentPackager,
        layer_packager: LayerDeploymentPackager,
    ) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        ...

    def handle_lambdalayer(self, config: Config, resource: models.LambdaLayer) -> None:
        ...

class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        ...

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_sqseventsource(self, config: Config, resource: models.SQSEventSource) -> None:
        ...

    def handle_kinesiseventsource(self, config: Config, resource: models.KinesisEventSource) -> None:
        ...

    def handle_dynamodbeventsource(self, config: Config, resource: models.DynamoDBEventSource) -> None:
        ...

    def _needs_policy_injected(self, role: models.ManagedIAMRole) -> bool:
        ...

    def _inject_trigger_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_websocketapi(self, config: Config, resource: models.WebsocketAPI) -> None:
        ...

    def _inject_into_function(self, config: Config, lambda_function: models.LambdaFunction) -> None:
        ...

    def _inject_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        ...

class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        ...

    def _read_document_from_file(self, filename: str) -> Dict[str, Any]:
        ...

    def handle_filebasediampolicy(self, config: Config, resource: models.FileBasedIAMPolicy) -> None:
        ...

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        ...

    def handle_autogeniampolicy(self, config: Config, resource: models.AutoGenIAMPolicy) -> None:
        ...

class BuildStage:
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        ...

    def execute(self, config: Config, resources: List[models.Model]) -> None:
        ...

class ResultsRecorder:
    def __init__(self, osutils: OSUtils) -> None:
        ...

    def record_results(
        self,
        results: Dict[str, Any],
        chalice_stage_name: str,
        project_dir: str,
    ) -> None:
        ...

class NoopResultsRecorder(ResultsRecorder):
    def record_results(
        self,
        results: Dict[str, Any],
        chalice_stage_name: str,
        project_dir: str,
    ) -> None:
        ...

class DeploymentReporter:
    _SORT_ORDER: Dict[str, int] = {'rest_api': 100, 'websocket_api': 100, 'domain_name': 100}
    _DEFAULT_ORDERING: int = 50

    def __init__(self, ui: UI) -> None:
        ...

    def generate_report(self, deployed_values: Dict[str, Any]) -> str:
        ...

    def _report_rest_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def _report_websocket_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def _report_domain_name(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def _report_lambda_function(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def _report_lambda_layer(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def _default_report(self, resource: Dict[str, Any], report: List[str]) -> None:
        ...

    def display_report(self, deployed_values: Dict[str, Any]) -> None:
        ...

def create_plan_only_deployer(
    session: Session,
    config: Config,
    ui: UI,
) -> Deployer:
    ...

def create_default_deployer(
    session: Session,
    config: Config,
    ui: UI,
) -> Deployer:
    ...

def _create_deployer(
    session: Session,
    config: Config,
    ui: UI,
    executor_cls: Type[BaseExecutor],
    recorder_cls: Type[ResultsRecorder],
) -> Deployer:
    ...

def create_build_stage(
    osutils: OSUtils,
    ui: UI,
    swagger_gen: SwaggerGenerator,
    config: Config,
) -> BuildStage:
    ...