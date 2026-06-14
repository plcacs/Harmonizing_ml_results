import logging
from botocore.session import Session
from typing import Optional, Dict, List, Any, Type
from chalice.config import Config
from chalice.awsclient import TypedAWSClient
from chalice.awsclient import LambdaClientError
from chalice.awsclient import DeploymentPackageTooLargeError
from chalice.awsclient import AWSClientError
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.executor import BaseExecutor, Executor, DisplayOnlyExecutor
from chalice.deploy.packager import (
    BaseLambdaDeploymentPackager,
    LambdaDeploymentPackager,
    AppOnlyDeploymentPackager,
    LayerDeploymentPackager,
)
from chalice.deploy.planner import PlanStage, RemoteState, NoopPlanner
from chalice.deploy.swagger import TemplatedSwaggerGenerator, SwaggerGenerator
from chalice.deploy.sweeper import ResourceSweeper
from chalice.policy import AppPolicyGenerator
from chalice.utils import OSUtils, UI

OptStr = Optional[str]
LOGGER: logging.Logger
_AWSCLIENT_EXCEPTIONS: tuple[type[Exception], ...]


class ChaliceDeploymentError(Exception):
    original_error: Exception

    def __init__(self, error: Exception) -> None: ...
    def _get_error_location(self, error: Exception) -> str: ...
    def _get_error_message(self, error: Exception) -> str: ...
    def _get_error_message_for_connection_error(self, connection_error: Exception) -> str: ...
    def _get_error_suggestion(self, error: Exception) -> Optional[str]: ...
    def _wrap_text(self, text: str, indent: str = ...) -> str: ...
    def _get_verb_from_client_method(self, client_method_name: str) -> str: ...
    def _get_mb(self, value: int) -> str: ...


def create_plan_only_deployer(session: Session, config: Config, ui: UI) -> Deployer: ...
def create_default_deployer(session: Session, config: Config, ui: UI) -> Deployer: ...
def _create_deployer(
    session: Session,
    config: Config,
    ui: UI,
    executor_cls: Type[BaseExecutor],
    recorder_cls: Type[ResultsRecorder],
) -> Deployer: ...
def create_build_stage(
    osutils: OSUtils,
    ui: UI,
    swagger_gen: SwaggerGenerator,
    config: Config,
) -> BuildStage: ...
def create_deletion_deployer(client: TypedAWSClient, ui: UI) -> Deployer: ...


class Deployer:
    BACKEND_NAME: str
    _application_builder: ApplicationGraphBuilder
    _deps_builder: DependencyBuilder
    _build_stage: BuildStage
    _plan_stage: PlanStage
    _sweeper: ResourceSweeper
    _executor: BaseExecutor
    _recorder: ResultsRecorder

    def __init__(
        self,
        application_builder: ApplicationGraphBuilder,
        deps_builder: DependencyBuilder,
        build_stage: BuildStage,
        plan_stage: PlanStage,
        sweeper: ResourceSweeper,
        executor: BaseExecutor,
        recorder: ResultsRecorder,
    ) -> None: ...
    def deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]: ...
    def _deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]: ...
    def _validate_config(self, config: Config) -> None: ...


class BaseDeployStep:
    def handle(self, config: Config, resource: models.Model) -> None: ...


class InjectDefaults(BaseDeployStep):
    _lambda_timeout: int
    _lambda_memory_size: int
    _tls_version: str

    def __init__(
        self,
        lambda_timeout: int = ...,
        lambda_memory_size: int = ...,
        tls_version: str = ...,
    ) -> None: ...
    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None: ...
    def handle_domainname(self, config: Config, resource: Any) -> None: ...


class DeploymentPackager(BaseDeployStep):
    _packager: BaseLambdaDeploymentPackager

    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None: ...
    def handle_deploymentpackage(self, config: Config, resource: models.DeploymentPackage) -> None: ...


class ManagedLayerDeploymentPackager(BaseDeployStep):
    _lambda_packager: BaseLambdaDeploymentPackager
    _layer_packager: BaseLambdaDeploymentPackager

    def __init__(
        self,
        lambda_packager: BaseLambdaDeploymentPackager,
        layer_packager: BaseLambdaDeploymentPackager,
    ) -> None: ...
    def handle_lambdafunction(self, config: Config, resource: Any) -> None: ...
    def handle_lambdalayer(self, config: Config, resource: Any) -> None: ...


class SwaggerBuilder(BaseDeployStep):
    _swagger_generator: SwaggerGenerator

    def __init__(self, swagger_generator: SwaggerGenerator) -> None: ...
    def handle_restapi(self, config: Config, resource: Any) -> None: ...


class LambdaEventSourcePolicyInjector(BaseDeployStep):
    _sqs_policy_injected: bool
    _kinesis_policy_injected: bool
    _ddb_policy_injected: bool

    def __init__(self) -> None: ...
    def handle_sqseventsource(self, config: Config, resource: Any) -> None: ...
    def handle_kinesiseventsource(self, config: Config, resource: Any) -> None: ...
    def handle_dynamodbeventsource(self, config: Config, resource: Any) -> None: ...
    def _needs_policy_injected(self, role: Any) -> bool: ...
    def _inject_trigger_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None: ...


class WebsocketPolicyInjector(BaseDeployStep):
    _policy_injected: bool

    def __init__(self) -> None: ...
    def handle_websocketapi(self, config: Config, resource: Any) -> None: ...
    def _inject_into_function(self, config: Config, lambda_function: Optional[Any]) -> None: ...
    def _inject_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None: ...


class PolicyGenerator(BaseDeployStep):
    _policy_gen: AppPolicyGenerator
    _osutils: OSUtils

    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None: ...
    def _read_document_from_file(self, filename: str) -> Dict[str, Any]: ...
    def handle_filebasediampolicy(self, config: Config, resource: models.FileBasedIAMPolicy) -> None: ...
    def handle_restapi(self, config: Config, resource: Any) -> None: ...
    def handle_autogeniampolicy(self, config: Config, resource: models.AutoGenIAMPolicy) -> None: ...


class BuildStage:
    _steps: List[BaseDeployStep]

    def __init__(self, steps: List[BaseDeployStep]) -> None: ...
    def execute(self, config: Config, resources: List[models.Model]) -> None: ...


class ResultsRecorder:
    _osutils: OSUtils

    def __init__(self, osutils: OSUtils) -> None: ...
    def record_results(
        self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str
    ) -> None: ...


class NoopResultsRecorder(ResultsRecorder):
    def record_results(
        self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str
    ) -> None: ...


class DeploymentReporter:
    _SORT_ORDER: Dict[str, int]
    _DEFAULT_ORDERING: int
    _ui: UI

    def __init__(self, ui: UI) -> None: ...
    def generate_report(self, deployed_values: Dict[str, Any]) -> str: ...
    def _report_rest_api(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def _report_websocket_api(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def _report_domain_name(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def _report_lambda_function(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def _report_lambda_layer(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def _default_report(self, resource: Dict[str, Any], report: List[str]) -> None: ...
    def display_report(self, deployed_values: Dict[str, Any]) -> None: ...