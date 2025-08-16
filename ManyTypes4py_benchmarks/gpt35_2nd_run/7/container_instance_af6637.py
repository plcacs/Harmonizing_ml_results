from typing import Any, Dict, List, Optional, Union

class AzureContainerJobConfiguration(BaseJobConfiguration):
    image: str
    resource_group_name: str
    subscription_id: str
    identities: Optional[List[str]]
    entrypoint: str
    image_registry: Optional[Union[ACRManagedIdentity, Any, None]]
    cpu: float
    gpu_count: Optional[int]
    gpu_sku: Optional[str]
    memory: float
    subnet_ids: Optional[List[str]]
    dns_servers: Optional[List[str]]
    stream_output: bool
    aci_credentials: AzureContainerInstanceCredentials
    task_start_timeout_seconds: int
    task_watch_poll_interval: float
    arm_template: Dict[str, Any]
    keep_container_group: bool

    def prepare_for_flow_run(self, flow_run: FlowRun, deployment: Optional[Deployment], flow: Optional[Flow]) -> None:
        ...

class AzureContainerVariables(BaseVariables):
    image: str
    resource_group_name: str
    subscription_id: str
    identities: Optional[List[str]]
    entrypoint: str
    image_registry: Optional[Union[ACRManagedIdentity, Any, None]]
    cpu: float
    gpu_count: Optional[int]
    gpu_sku: Optional[str]
    memory: float
    subnet_ids: Optional[List[str]]
    dns_servers: Optional[List[str]]
    stream_output: bool
    aci_credentials: AzureContainerInstanceCredentials
    task_start_timeout_seconds: int
    task_watch_poll_interval: float
    keep_container_group: bool

class AzureContainerWorkerResult(BaseWorkerResult):
    identifier: str

class AzureContainerWorker(BaseWorker):
    type: str
    job_configuration: AzureContainerJobConfiguration
    job_configuration_variables: AzureContainerVariables

    async def run(self, flow_run: FlowRun, configuration: AzureContainerJobConfiguration, task_status: Optional[Any] = None) -> AzureContainerWorkerResult:
        ...

    async def _provision_container_group(self, aci_client: ContainerInstanceManagementClient, resource_client: ResourceManagementClient, configuration: AzureContainerJobConfiguration, container_group_name: str) -> ContainerGroup:
        ...

    def _watch_task_and_get_exit_code(self, client: ContainerInstanceManagementClient, configuration: AzureContainerJobConfiguration, container_group: ContainerGroup, run_start_time: datetime.datetime) -> int:
        ...

    async def _wait_for_container_group_deletion(self, aci_client: ContainerInstanceManagementClient, configuration: AzureContainerJobConfiguration, container_group_name: str) -> None:
        ...

    def _get_container(self, container_group: ContainerGroup) -> Container:
        ...

    @staticmethod
    def _get_container_group(client: ContainerInstanceManagementClient, resource_group_name: str, container_group_name: str) -> ContainerGroup:
        ...

    def _get_and_stream_output(self, client: ContainerInstanceManagementClient, configuration: AzureContainerJobConfiguration, container_group: ContainerGroup, last_log_time: datetime.datetime) -> datetime.datetime:
        ...

    def _get_logs(self, client: ContainerInstanceManagementClient, configuration: AzureContainerJobConfiguration, container_group: ContainerGroup, max_lines: int = 100) -> str:
        ...

    def _stream_output(self, log_content: str, last_log_time: datetime.datetime) -> datetime.datetime:
        ...

    @property
    def _log_prefix(self) -> str:
        ...

    @staticmethod
    def _provisioning_succeeded(container_group: ContainerGroup) -> bool:
        ...

    @staticmethod
    def _write_output_line(line: str) -> None:
        ...
