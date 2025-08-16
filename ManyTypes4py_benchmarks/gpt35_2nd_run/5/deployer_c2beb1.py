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
    def __init__(self, application_builder: ApplicationGraphBuilder, deps_builder: DependencyBuilder, build_stage: BuildStage, plan_stage: PlanStage, sweeper: ResourceSweeper, executor: BaseExecutor, recorder: ResultsRecorder) -> None:
        ...

class BaseDeployStep:
    def handle(self, config: Config, resource: Any) -> None:
        ...

class InjectDefaults(BaseDeployStep):
    def __init__(self, lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT, lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE, tls_version: str = DEFAULT_TLS_VERSION) -> None:
        ...

class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: BaseLambdaDeploymentPackager) -> None:
        ...

class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(self, lambda_packager: AppOnlyDeploymentPackager, layer_packager: LayerDeploymentPackager) -> None:
        ...

class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        ...

class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        ...

class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        ...

class BuildStage:
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        ...

class ResultsRecorder:
    def __init__(self, osutils: OSUtils) -> None:
        ...

class NoopResultsRecorder(ResultsRecorder):
    def record_results(self, results: Dict[str, Any], chalice_stage_name: str, project_dir: str) -> None:
        ...

class DeploymentReporter:
    def __init__(self, ui: UI) -> None:
        ...
