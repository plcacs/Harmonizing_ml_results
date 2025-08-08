from typing import Optional, Dict, List, Any, Type

class ChaliceDeploymentError(Exception):
    def __init__(self, error: Exception) -> None:
        ...

def create_plan_only_deployer(session: Session, config: Config, ui: UI) -> Deployer:
    ...

def create_default_deployer(session: Session, config: Config, ui: UI) -> Deployer:
    ...

def _create_deployer(session: Session, config: Config, ui: UI, executor_cls: Type, recorder_cls: Type) -> Deployer:
    ...

def create_build_stage(osutils: OSUtils, ui: UI, swagger_gen: SwaggerGenerator, config: Config) -> BuildStage:
    ...

def create_deletion_deployer(client: TypedAWSClient, ui: UI) -> Deployer:
    ...

class Deployer:
    BACKEND_NAME: str = 'api'

    def __init__(self, application_builder: ApplicationGraphBuilder, deps_builder: DependencyBuilder, build_stage: BuildStage, plan_stage: PlanStage, sweeper: ResourceSweeper, executor: BaseExecutor, recorder: ResultsRecorder) -> None:
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
    def __init__(self, lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT, lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE, tls_version: str = DEFAULT_TLS_VERSION) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: Any) -> None:
        ...

class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: Any) -> None:
        ...

    def handle_deploymentpackage(self, config: Config, resource: Any) -> None:
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(self, lambda_packager: Any, layer_packager: Any) -> None:
        ...

    def handle_lambdafunction(self, config: Config, resource: Any) -> None:
        ...

class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        ...

    def handle_restapi(self, config: Config, resource: Any) -> None:
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_sqseventsource(self, config: Config, resource: Any) -> None:
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

    def handle_websocketapi(self, config: Config, resource: Any) -> None:
        ...

class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: Any, osutils: OSUtils) -> None:
        ...

    def handle_filebasediampolicy(self, config: Config, resource: Any) -> None:
        ...

class BuildStage:
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        ...

    def execute(self, config: Config, resources: List[Any]) -> None:
        ...

class ResultsRecorder:
    def __init__(self, osutils: OSUtils) -> None:
        ...

    def record_results(self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str) -> None:
        ...

class NoopResultsRecorder(ResultsRecorder):
    def record_results(self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str) -> None:
        ...

class DeploymentReporter:
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
