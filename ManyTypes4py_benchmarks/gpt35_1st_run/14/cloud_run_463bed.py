from typing import TYPE_CHECKING, Any, Dict, Optional
from prefect.client.schemas import FlowRun
from prefect.server.schemas.core import Flow
from prefect.server.schemas.responses import DeploymentResponse

if TYPE_CHECKING:
    from prefect.client.schemas import FlowRun
    from prefect.server.schemas.core import Flow
    from prefect.server.schemas.responses import DeploymentResponse

def _get_default_job_body_template() -> Dict[str, Any]:
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'name': '{{ name }}', 'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{'image': '{{ image }}', 'command': '{{ command }}', 'resources': {'limits': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}, 'requests': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}}}], 'timeoutSeconds': '{{ timeout }}', 'serviceAccountName': '{{ service_account_name }}'}}}, 'metadata': {'annotations': {'run.googleapis.com/vpc-access-connector': '{{ vpc_connector_name }}'}}}

def _get_base_job_body() -> Dict[str, Any]:
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{}]}}}}}

class CloudRunWorkerJobConfiguration(BaseJobConfiguration):
    def prepare_for_flow_run(self, flow_run: FlowRun, deployment: Optional[DeploymentResponse] = None, flow: Optional[Flow] = None) -> None:
    def _populate_envs(self) -> None:
    def _populate_name_if_not_present(self) -> None:
    def _populate_image_if_not_present(self) -> None:
    def _populate_or_format_command(self) -> None:
    def _format_args_if_present(self) -> None:
    @classmethod
    def _ensure_job_includes_all_required_components(cls, value: Dict[str, Any]) -> Dict[str, Any]:
    @classmethod
    def _ensure_job_has_compatible_values(cls, value: Dict[str, Any]) -> Dict[str, Any]:

class CloudRunWorkerVariables(BaseVariables):
class CloudRunWorkerResult(BaseWorkerResult):
class CloudRunWorker(BaseWorker):
    def _create_job_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
    def _job_run_submission_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
    async def run(self, flow_run: FlowRun, configuration: CloudRunWorkerJobConfiguration, task_status: Optional[TaskStatus] = None) -> CloudRunWorkerResult:
    def _get_client(self, configuration: CloudRunWorkerJobConfiguration) -> Resource:
    def _create_job_and_wait_for_registration(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> None:
    def _begin_job_execution(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> Execution:
    def _watch_job_execution_and_get_result(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, execution: Execution, logger: PrefectLogAdapter, poll_interval: int = 5) -> CloudRunWorkerResult:
    def _watch_job_execution(self, client: Resource, job_execution: Execution, poll_interval: int = 5) -> Execution:
    def _wait_for_job_creation(self, client: Resource, configuration: CloudRunWorkerJobConfiguration, logger: PrefectLogAdapter, poll_interval: int = 5) -> None:
