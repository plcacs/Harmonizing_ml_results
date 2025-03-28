from typing import Optional, List, Dict, Union
from enum import Enum
import datetime
import sys
import time
from azure.core.polling import LROPoller
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import Container, ContainerGroup, Logs
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import Deployment, DeploymentExtended, DeploymentMode, DeploymentProperties
from pydantic import Field, SecretStr
from prefect.client.orchestration import get_client
from prefect.client.schemas import FlowRun
from prefect.server.schemas.core import Flow
from prefect.server.schemas.responses import DeploymentResponse
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult
from prefect_azure.container_instance import ACRManagedIdentity
from prefect_azure.credentials import AzureContainerInstanceCredentials

class ContainerGroupProvisioningState(str, Enum):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"

class ContainerRunState(str, Enum):
    RUNNING = "Running"
    TERMINATED = "Terminated"

class AzureContainerJobConfiguration(BaseJobConfiguration):
    image: str = Field(default_factory=get_prefect_image_name)
    resource_group_name: str = Field(default=...)
    subscription_id: SecretStr = Field(default=...)
    identities: Optional[List[str]] = Field(default=None)
    entrypoint: Optional[str] = Field(default=DEFAULT_CONTAINER_ENTRYPOINT)
    image_registry: DockerRegistry = Field(default=None)
    cpu: float = Field(default=ACI_DEFAULT_CPU)
    gpu_count: Optional[int] = Field(default=None)
    gpu_sku: Optional[str] = Field(default=None)
    memory: float = Field(default=ACI_DEFAULT_MEMORY)
    subnet_ids: Optional[List[str]] = Field(default=None)
    dns_servers: Optional[List[str]] = Field(default=None)
    stream_output: bool = Field(default=False)
    aci_credentials: AzureContainerInstanceCredentials = Field(default_factory=AzureContainerInstanceCredentials)
    task_start_timeout_seconds: int = Field(default=240)
    task_watch_poll_interval: float = Field(default=5.0)
    arm_template: Dict[str, Any] = Field(json_schema_extra=dict(template=_get_default_arm_template()))
    keep_container_group: bool = Field(default=False)

class AzureContainerVariables(BaseVariables):
    image: str = Field(default_factory=get_prefect_image_name)
    resource_group_name: str = Field(default=...)
    subscription_id: SecretStr = Field(default=...)
    identities: Optional[List[str]] = Field(default=None)
    entrypoint: Optional[str] = Field(default=DEFAULT_CONTAINER_ENTRYPOINT)
    image_registry: DockerRegistry = Field(default=None)
    cpu: float = Field(default=ACI_DEFAULT_CPU)
    gpu_count: Optional[int] = Field(default=None)
    gpu_sku: Optional[str] = Field(default=None)
    memory: float = Field(default=ACI_DEFAULT_MEMORY)
    subnet_ids: Optional[List[str]] = Field(default=None)
    dns_servers: Optional[List[str]] = Field(default=None)
    aci_credentials: AzureContainerInstanceCredentials = Field(default_factory=AzureContainerInstanceCredentials)
    stream_output: bool = Field(default=False)
    task_start_timeout_seconds: int = Field(default=240)
    task_watch_poll_interval: float = Field(default=5.0)
    keep_container_group: bool = Field(default=False)

class AzureContainerWorkerResult(BaseWorkerResult):
    identifier: str
    status_code: int

class AzureContainerWorker(BaseWorker):
    type: str = "azure-container-instance"
    job_configuration = AzureContainerJobConfiguration
    job_configuration_variables = AzureContainerVariables
    _logo_url: str = "https://cdn.sanity.io/images/3ugk85nk/production/54e3fa7e00197a4fbd1d82ed62494cb58d08c96a-250x250.png"
    _display_name: str = "Azure Container Instances"
    _description: str = "Execute flow runs within containers on Azure's Container Instances service. Requires an Azure account."
    _documentation_url: str = "https://docs.prefect.io/integrations/prefect-azure"

    async def run(
        self,
        flow_run: FlowRun,
        configuration: AzureContainerJobConfiguration,
        task_status: Optional[anyio.abc.TaskStatus] = None,
    ) -> AzureContainerWorkerResult:
        pass

    def _wait_for_task_container_start(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str,
        creation_status_poller: LROPoller[DeploymentExtended],
    ) -> Optional[ContainerGroup]:
        pass

    async def _provision_container_group(
        self,
        aci_client: ContainerInstanceManagementClient,
        resource_client: ResourceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str,
    ) -> ContainerGroup:
        pass

    def _watch_task_and_get_exit_code(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        run_start_time: datetime.datetime,
    ) -> int:
        pass

    async def _wait_for_container_group_deletion(
        self,
        aci_client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str,
    ) -> None:
        pass

    def _get_container(self, container_group: ContainerGroup) -> Container:
        pass

    @staticmethod
    def _get_container_group(
        client: ContainerInstanceManagementClient,
        resource_group_name: str,
        container_group_name: str,
    ) -> ContainerGroup:
        pass

    def _get_and_stream_output(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        last_log_time: datetime.datetime,
    ) -> datetime.datetime:
        pass

    def _get_logs(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        max_lines: int = 100,
    ) -> str:
        pass

    def _stream_output(
        self, log_content: Union[str, None], last_log_time: datetime.datetime
    ) -> datetime.datetime:
        pass

    @property
    def _log_prefix(self) -> str:
        pass

    @staticmethod
    def _provisioning_succeeded(container_group: Union[ContainerGroup, None]) -> bool:
        pass

    @staticmethod
    def _write_output_line(line: str) -> None:
        pass
