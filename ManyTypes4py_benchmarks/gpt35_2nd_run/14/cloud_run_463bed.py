from typing import TYPE_CHECKING, Any, Dict, Optional
from prefect.client.schemas import FlowRun
from prefect.server.schemas.core import Flow
from prefect.server.schemas.responses import DeploymentResponse

if TYPE_CHECKING:
    from prefect.client.schemas import FlowRun
    from prefect.server.schemas.core import Flow
    from prefect.server.schemas.responses import DeploymentResponse

def _get_default_job_body_template() -> Dict[str, Any]:
    """Returns the default job body template used by the Cloud Run Job."""
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'name': '{{ name }}', 'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{'image': '{{ image }}', 'command': '{{ command }}', 'resources': {'limits': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}, 'requests': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}}}], 'timeoutSeconds': '{{ timeout }}', 'serviceAccountName': '{{ service_account_name }}'}}}, 'metadata': {'annotations': {'run.googleapis.com/vpc-access-connector': '{{ vpc_connector_name }}'}}}

def _get_base_job_body() -> Dict[str, Any]:
    """Returns a base job body to use for job body validation."""
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{}]}}}}}

class CloudRunWorkerJobConfiguration(BaseJobConfiguration):
    region: str
    credentials: GcpCredentials
    job_body: Dict[str, Any]
    timeout: int
    keep_job: bool

    @property
    def project(self) -> str:
        return self.credentials.project

    @property
    def job_name(self) -> str:
        return self.job_body['metadata']['name']

    def prepare_for_flow_run(self, flow_run: FlowRun, deployment: Optional[DeploymentResponse] = None, flow: Optional[Flow] = None) -> None:
        pass

    def _populate_envs(self) -> None:
        pass

    def _populate_name_if_not_present(self) -> None:
        pass

    def _populate_image_if_not_present(self) -> None:
        pass

    def _populate_or_format_command(self) -> None:
        pass

    def _format_args_if_present(self) -> None:
        pass

    @classmethod
    def _ensure_job_includes_all_required_components(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @classmethod
    def _ensure_job_has_compatible_values(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        pass

class CloudRunWorkerVariables(BaseVariables):
    region: str
    credentials: GcpCredentials
    image: Optional[str]
    cpu: Optional[str]
    memory: Optional[str]
    vpc_connector_name: Optional[str]
    service_account_name: Optional[str]
    keep_job: bool
    timeout: int

class CloudRunWorkerResult(BaseWorkerResult):
    pass

class CloudRunWorker(BaseWorker):
    type: str
    job_configuration: CloudRunWorkerJobConfiguration
    job_configuration_variables: CloudRunWorkerVariables
    _description: str
    _display_name: str
    _documentation_url: str
    _logo_url: str

    def _create_job_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
        pass

    def _job_run_submission_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
        pass

    async def run(self, flow_run: FlowRun, configuration: CloudRunWorkerJobConfiguration, task_status: Optional[TaskStatus] = None) -> CloudRunWorkerResult:
        pass

    def _get_client(self, configuration: CloudRunWorkerJobConfiguration) -> Resource:
        pass

    def _create_job_and_wait_for_registration(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> None:
        pass

    def _begin_job_execution(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> Execution:
        pass

    def _watch_job_execution_and_get_result(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, execution: Execution, logger: PrefectLogAdapter, poll_interval: int = 5) -> CloudRunWorkerResult:
        pass

    def _watch_job_execution(self, client: Resource, job_execution: Execution, poll_interval: int = 5) -> Execution:
        pass

    def _wait_for_job_creation(self, client: Resource, configuration: CloudRunWorkerJobConfiguration, logger: PrefectLogAdapter, poll_interval: int = 5) -> None:
        pass
