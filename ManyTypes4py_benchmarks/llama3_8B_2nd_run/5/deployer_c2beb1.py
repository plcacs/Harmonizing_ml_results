from typing import Optional, Dict, List, Any, Type, cast
import json
import textwrap
import socket
import logging
import botocore.exceptions
from botocore.vendored.requests import ConnectionError as RequestsConnectionError
from botocore.session import Session
from chalice.config import Config
from chalice.awsclient import DeploymentPackageTooLargeError
from chalice.awsclient import LambdaClientError
from chalice.awsclient import AWSClientError
from chalice.awsclient import TypedAWSClient
from chalice.constants import MAX_LAMBDA_DEPLOYMENT_SIZE
from chalice.constants import VPC_ATTACH_POLICY
from chalice.constants import DEFAULT_LAMBDA_TIMEOUT
from chalice.constants import DEFAULT_LAMBDA_MEMORY_SIZE
from chalice.constants import DEFAULT_TLS_VERSION
from chalice.constants import SQS_EVENT_SOURCE_POLICY
from chalice.constants import KINESIS_EVENT_SOURCE_POLICY
from chalice.constants import DDB_EVENT_SOURCE_POLICY
from chalice.constants import POST_TO_WEBSOCKET_CONNECTION_POLICY
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.executor import BaseExecutor
from chalice.deploy.executor import Executor
from chalice.deploy.executor import DisplayOnlyExecutor
from chalice.deploy.packager import PipRunner
from chalice.deploy.packager import SubprocessPip
from chalice.deploy.packager import DependencyBuilder as PipDependencyBuilder
from chalice.deploy.packager import LambdaDeploymentPackager
from chalice.deploy.packager import AppOnlyDeploymentPackager
from chalice.deploy.packager import LayerDeploymentPackager
from chalice.deploy.packager import BaseLambdaDeploymentPackager
from chalice.deploy.packager import EmptyPackageError
from chalice.deploy.planner import PlanStage
from chalice.deploy.planner import RemoteState
from chalice.deploy.planner import NoopPlanner
from chalice.deploy.swagger import TemplatedSwaggerGenerator
from chalice.deploy.swagger import SwaggerGenerator
from chalice.deploy.sweeper import ResourceSweeper
from chalice.deploy.validate import validate_configuration
from chalice.policy import AppPolicyGenerator
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.utils import serialize_to_json
from chalice.utils import _get_error_location
from chalice.utils import _get_error_message
from chalice.utils import _get_error_suggestion
from chalice.utils import _wrap_text
from chalice.utils import _get_verb_from_client_method
from chalice.utils import _get_mb

class ChaliceDeploymentError(Exception):
    def __init__(self, error: Optional[AWSClientError]) -> None:
        ...

    def _get_error_location(self, error: AWSClientError) -> str:
        ...

    def _get_error_message(self, error: AWSClientError) -> str:
        ...

    def _get_error_suggestion(self, error: AWSClientError) -> Optional[str]:
        ...

class Deployer:
    BACKEND_NAME: str = 'api'

    def __init__(self,
                 application_builder: ApplicationGraphBuilder,
                 deps_builder: DependencyBuilder,
                 build_stage: BuildStage,
                 plan_stage: PlanStage,
                 sweeper: ResourceSweeper,
                 executor: BaseExecutor,
                 recorder: ResultsRecorder) -> None:
        ...

    def deploy(self,
               config: Config,
               chalice_stage_name: str) -> Dict[str, Any]:
        ...

    def _validate_config(self, config: Config) -> None:
        ...

class BaseDeployStep:
    def handle(self,
               config: Config,
               resource: models.Model) -> None:
        ...

class InjectDefaults(BaseDeployStep):
    def __init__(self,
                 lambda_timeout: Optional[int] = DEFAULT_LAMBDA_TIMEOUT,
                 lambda_memory_size: Optional[int] = DEFAULT_LAMBDA_MEMORY_SIZE,
                 tls_version: Optional[str] = DEFAULT_TLS_VERSION) -> None:
        ...

    def handle_lambdafunction(self,
                              config: Config,
                              resource: models.LambdaFunction) -> None:
        ...

    def handle_domainname(self,
                          config: Config,
                          resource: models.DomainName) -> None:
        ...

class DeploymentPackager(BaseDeployStep):
    def __init__(self,
                 packager: LambdaDeploymentPackager) -> None:
        ...

    def handle_deploymentpackage(self,
                                 config: Config,
                                 resource: models.DeploymentPackage) -> None:
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(self,
                 lambda_packager: LambdaDeploymentPackager,
                 layer_packager: LayerDeploymentPackager) -> None:
        ...

    def handle_lambdalayer(self,
                           config: Config,
                           resource: models.LambdaLayer) -> None:
        ...

class SwaggerBuilder(BaseDeployStep):
    def __init__(self,
                 swagger_generator: TemplatedSwaggerGenerator) -> None:
        ...

    def handle_restapi(self,
                       config: Config,
                       resource: models.RestApi) -> None:
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_sqseventsource(self,
                             config: Config,
                             resource: models.SQSEventSource) -> None:
        ...

    def handle_kinesiseventsource(self,
                                 config: Config,
                                 resource: models.KinesisEventSource) -> None:
        ...

    def handle_dynamodbeventsource(self,
                                   config: Config,
                                   resource: models.DynamoDBEventSource) -> None:
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_websocketapi(self,
                            config: Config,
                            resource: models.WebsocketApi) -> None:
        ...

class PolicyGenerator(BaseDeployStep):
    def __init__(self,
                 policy_gen: AppPolicyGenerator,
                 osutils: OSUtils) -> None:
        ...

    def _read_document_from_file(self,
                                 filename: str) -> Dict[str, Any]:
        ...

    def handle_filebasediampolicy(self,
                                  config: Config,
                                  resource: models.FileBasedIAMPolicy) -> None:
        ...

    def handle_restapi(self,
                       config: Config,
                       resource: models.RestApi) -> None:
        ...

    def handle_autogeniampolicy(self,
                                config: Config,
                                resource: models.AutoGenIAMPolicy) -> None:
        ...

class BuildStage:
    def __init__(self,
                 steps: List[BaseDeployStep]) -> None:
        ...

    def execute(self,
                config: Config,
                resources: List[models.Model]) -> None:
        ...

class ResultsRecorder:
    def __init__(self,
                 osutils: OSUtils) -> None:
        ...

    def record_results(self,
                       results: Dict[str, Any],
                       chalice_stage_name: str,
                       project_dir: str) -> None:
        ...

class DeploymentReporter:
    _SORT_ORDER: Dict[str, int] = {'rest_api': 100, 'websocket_api': 100, 'domain_name': 100}
    _DEFAULT_ORDERING: int = 50

    def __init__(self,
                 ui: UI) -> None:
        ...

    def generate_report(self,
                        deployed_values: Dict[str, Any]) -> str:
        ...

    def display_report(self,
                       deployed_values: Dict[str, Any]) -> None:
        ...

class NoopResultsRecorder(ResultsRecorder):
    def record_results(self,
                       results: Dict[str, Any],
                       chalice_stage_name: str,
                       project_dir: str) -> None:
        ...
