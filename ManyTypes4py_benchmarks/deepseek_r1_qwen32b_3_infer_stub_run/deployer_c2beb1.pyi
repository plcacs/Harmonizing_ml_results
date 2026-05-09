"""Stub file for chalice.deploy.deployer module."""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
from chalice.config import Config
from chalice.deploy.models import (
    Application,
    Resource,
)
from chalice.deploy.executor import (
    BaseExecutor,
    DisplayOnlyExecutor,
    Executor,
)
from chalice.deploy.planner import PlanStage
from chalice.deploy.sweeper import ResourceSweeper
from chalice.deploy.appgraph import ApplicationGraphBuilder
from chalice.deploy.packager import (
    BaseLambdaDeploymentPackager,
    DeploymentPackager,
    ManagedLayerDeploymentPackager,
)
from chalice.deploy.swagger import SwaggerGenerator
from chalice.policy import AppPolicyGenerator
from chalice.utils import UI

class ChaliceDeploymentError(Exception):
    """Error raised during deployment process."""
    ...

class Deployer:
    """Main deployer class that coordinates the deployment process."""
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

    def deploy(
        self,
        config: Config,
        chalice_stage_name: str,
    ) -> dict[str, Any]:
        """Deploy the application and return deployed resources."""
        ...

    def _deploy(
        self,
        config: Config,
        chalice_stage_name: str,
    ) -> dict[str, Any]:
        """Internal deploy method."""
        ...

    def _validate_config(self, config: Config) -> None:
        """Validate the deployment configuration."""
        ...

class BaseDeployStep:
    """Base class for deployment steps."""
    def handle(self, config: Config, resource: Resource) -> None:
        """Handle a resource during deployment."""
        ...

class InjectDefaults(BaseDeployStep):
    """Inject default values for resources."""
    def __init__(
        self,
        lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT,
        lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE,
        tls_version: str = DEFAULT_TLS_VERSION,
    ) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: Resource) -> None:
        """Handle Lambda function defaults."""
        ...

    def handle_domainname(self, config: Config, resource: Resource) -> None:
        """Handle domain name defaults."""
        ...

class DeploymentPackager(BaseDeployStep):
    """Handle deployment package creation."""
    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None:
        ...

    def handle_deploymentpackage(self, config: Config, resource: Resource) -> None:
        """Handle deployment package."""
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    """Handle managed layer deployment."""
    def __init__(
        self,
        lambda_packager: BaseLambdaDeploymentPackager,
        layer_packager: BaseLambdaDeploymentPackager,
    ) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: Resource) -> None:
        """Handle Lambda function managed layers."""
        ...

    def handle_lambdalayer(self, config: Config, resource: Resource) -> None:
        """Handle Lambda layer deployment."""
        ...

class SwaggerBuilder(BaseDeployStep):
    """Generate Swagger documentation."""
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        ...

    def handle_restapi(self, config: Config, resource: Resource) -> None:
        """Handle REST API Swagger generation."""
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    """Inject policies for Lambda event sources."""
    def __init__(self) -> None:
        ...

    def handle_sqseventsource(self, config: Config, resource: Resource) -> None:
        """Handle SQS event source policies."""
        ...

    def handle_kinesiseventsource(self, config: Config, resource: Resource) -> None:
        """Handle Kinesis event source policies."""
        ...

    def handle_dynamodbeventsource(self, config: Config, resource: Resource) -> None:
        """Handle DynamoDB event source policies."""
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    """Inject policies for WebSocket APIs."""
    def __init__(self) -> None:
        ...

    def handle_websocketapi(self, config: Config, resource: Resource) -> None:
        """Handle WebSocket API policies."""
        ...

    def _inject_into_function(
        self,
        config: Config,
        lambda_function: Optional[Resource],
    ) -> None:
        """Inject policies into Lambda functions."""
        ...

class PolicyGenerator(BaseDeployStep):
    """Generate IAM policies."""
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        ...

    def _read_document_from_file(self, filename: str) -> Dict[str, Any]:
        """Read IAM policy document from file."""
        ...

    def handle_filebasediampolicy(self, config: Config, resource: Resource) -> None:
        """Handle file-based IAM policies."""
        ...

    def handle_restapi(self, config: Config, resource: Resource) -> None:
        """Handle REST API IAM policies."""
        ...

    def handle_autogeniampolicy(self, config: Config, resource: Resource) -> None:
        """Handle auto-generated IAM policies."""
        ...

class BuildStage:
    """Coordinate execution of deployment steps."""
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        ...

    def execute(self, config: Config, resources: List[Resource]) -> None:
        """Execute all deployment steps."""
        ...

class ResultsRecorder:
    """Record deployment results."""
    def __init__(self, osutils: OSUtils) -> None:
        ...

    def record_results(
        self,
        results: dict[str, Any],
        chalice_stage_name: str,
        project_dir: str,
    ) -> None:
        """Record deployment results to file."""
        ...

class NoopResultsRecorder(ResultsRecorder):
    """No-op implementation of ResultsRecorder."""
    def record_results(
        self,
        results: dict[str, Any],
        chalice_stage_name: str,
        project_dir: str,
    ) -> None:
        """No-op method."""
        ...

def create_plan_only_deployer(
    session: Session,
    config: Config,
    ui: UI,
) -> Deployer:
    """Create a deployer that only creates a deployment plan."""
    ...

def create_default_deployer(
    session: Session,
    config: Config,
    ui: UI,
) -> Deployer:
    """Create the default deployer."""
    ...

def _create_deployer(
    session: Session,
    config: Config,
    ui: UI,
    executor_cls: Type[BaseExecutor],
    recorder_cls: Type[ResultsRecorder],
) -> Deployer:
    """Helper function to create a deployer."""
    ...

def create_deletion_deployer(
    client: TypedAWSClient,
    ui: UI,
) -> Deployer:
    """Create a deployer for deleting resources."""
    ...

class DeploymentReporter:
    """Generate deployment reports."""
    def __init__(self, ui: UI) -> None:
        ...

    def generate_report(self, deployed_values: dict[str, Any]) -> str:
        """Generate a deployment report."""
        ...

    def _report_rest_api(self, resource: dict[str, Any], report: List[str]) -> None:
        """Report REST API deployment."""
        ...

    def _report_websocket_api(self, resource: dict[str, Any], report: List[str]) -> None:
        """Report WebSocket API deployment."""
        ...

    def _report_domain_name(self, resource: dict[str, Any], report: List[str]) -> None:
        """Report domain name deployment."""
        ...

    def _report_lambda_function(self, resource: dict[str, Any], report: List[str]) -> None:
        """Report Lambda function deployment."""
        ...

    def _report_lambda_layer(self, resource: dict[str, Any], report: List[str]) -> None:
        """Report Lambda layer deployment."""
        ...

    def _default_report(self, resource: dict[str, Any], report: List[str]) -> None:
        """Default report method."""
        ...

    def display_report(self, deployed_values: dict[str, Any]) -> None:
        """Display the deployment report."""
        ...