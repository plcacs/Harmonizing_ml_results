"""Chalice deployer module."""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    cast,
)
import json
import logging
from enum import Enum
from pathlib import Path
from botocore.exceptions import ClientError
from chalice.config import Config
from chalice.deploy.models import (
    Application,
    LambdaFunction,
    Layer,
    DeploymentPackage,
    ManagedIAMRole,
    AutoGenIAMPolicy,
    FileBasedIAMPolicy,
    TLSVersion,
)
from chalice.deploy.executor import BaseExecutor, Executor, DisplayOnlyExecutor
from chalice.deploy.planner import PlanStage, NoopPlanner, RemoteState
from chalice.deploy.sweeper import ResourceSweeper
from chalice.utils import UI, OSUtils

LOGGER: logging.Logger = ...

class ChaliceDeploymentError(Exception):
    original_error: Exception
    def __init__(self, error: Exception) -> None:
        ...

class Deployer:
    BACKEND_NAME: str = ...
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
    def handle(self, config: Config, resource: Any) -> None:
        ...

class InjectDefaults(BaseDeployStep):
    def __init__(
        self,
        lambda_timeout: int = ...,
        lambda_memory_size: int = ...,
        tls_version: TLSVersion = ...,
    ) -> None:
        ...

    def handle_lambdafunction(
        self,
        config: Config,
        resource: LambdaFunction,
    ) -> None:
        ...

    def handle_domainname(
        self,
        config: Config,
        resource: models.DomainName,
    ) -> None:
        ...

class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None:
        ...

    def handle_deploymentpackage(
        self,
        config: Config,
        resource: DeploymentPackage,
    ) -> None:
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(
        self,
        lambda_packager: AppOnlyDeploymentPackager,
        layer_packager: LayerDeploymentPackager,
    ) -> None:
        ...

    def handle_lambdafunction(
        self,
        config: Config,
        resource: LambdaFunction,
    ) -> None:
        ...

    def handle_lambdalayer(
        self,
        config: Config,
        resource: Layer,
    ) -> None:
        ...

class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        ...

    def handle_restapi(
        self,
        config: Config,
        resource: models.RestAPI,
    ) -> None:
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def handle_sqseventsource(
        self,
        config: Config,
        resource: models.SQSEventSource,
    ) -> None:
        ...

    def handle_kinesiseventsource(
        self,
        config: Config,
        resource: models.KinesisEventSource,
    ) -> None:
        ...

    def handle_dynamodbeventsource(
        self,
        config: Config,
        resource: models.DynamoDBEventSource,
    ) -> None:
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    def handle_websocketapi(
        self,
        config: Config,
        resource: models.WebsocketAPI,
    ) -> None:
        ...

class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        ...

    def handle_filebasediampolicy(
        self,
        config: Config,
        resource: FileBasedIAMPolicy,
    ) -> None:
        ...

    def handle_restapi(
        self,
        config: Config,
        resource: models.RestAPI,
    ) -> None:
        ...

    def handle_autogeniampolicy(
        self,
        config: Config,
        resource: AutoGenIAMPolicy,
    ) -> None:
        ...

class BuildStage:
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        ...

    def execute(
        self,
        config: Config,
        resources: List[Any],
    ) -> None:
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
    def __init__(self, ui: UI) -> None:
        ...

    def generate_report(
        self,
        deployed_values: Dict[str, Any],
    ) -> str:
        ...

    def display_report(
        self,
        deployed_values: Dict[str, Any],
    ) -> None:
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

def create_deletion_deployer(
    client: TypedAWSClient,
    ui: UI,
) -> Deployer:
    ...