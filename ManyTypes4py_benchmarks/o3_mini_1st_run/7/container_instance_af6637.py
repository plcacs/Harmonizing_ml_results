#!/usr/bin/env python3
"""
Module containing the Azure Container Instances worker used for executing flow
runs in ACI containers.

To start an ACI worker, run the following command:

```bash
prefect worker start --pool 'my-work-pool' --type azure-container-instance
```

Replace `my-work-pool` with the name of the work pool you want the worker
to poll for flow runs.

!!! example "Using a custom ARM template"
    To facilitate easy customization, the Azure Container worker provisions a
    containing group using an ARM template. The default ARM template is represented
    in YAML as follows:
    ```yaml
    ---
    arm_template:
      "$schema": https://schema.management.azure.com/schemas/2019-08-01/deploymentTemplate.json#
      contentVersion: 1.0.0.0
      parameters:
        location:
          type: string
          defaultValue: "[resourceGroup().location]"
          metadata:
            description: Location for all resources.
        container_group_name:
          type: string
          defaultValue: "[uniqueString(resourceGroup().id)]"
          metadata:
            description: The name of the container group to create.
        container_name:
          type: string
          defaultValue: "[uniqueString(resourceGroup().id)]"
          metadata:
            description: The name of the container to create.
      resources:
      - type: Microsoft.ContainerInstance/containerGroups
        apiVersion: '2022-09-01'
        name: "[parameters('container_group_name')]"
        location: "[parameters('location')]"
        properties:
          containers:
          - name: "[parameters('container_name')]"
            properties:
              image: rpeden/my-aci-flow:latest
              command: "{{ command }}"
              resources:
                requests:
                  cpu: "{{ cpu }}"
                  memoryInGB: "{{ memory }}"
              environmentVariables: []
          osType: Linux
          restartPolicy: Never
    ```

    Each values enclosed in `{{ }}` is a placeholder that will be replaced with
    a value at runtime. The values that can be used a placeholders are defined
    by the `variables` schema defined in the base job template.

    The default job manifest and available variables can be customized on a work pool
    by work pool basis. These customizations can be made via the Prefect UI when
    creating or editing a work pool.

    Using an ARM template makes the worker flexible; you're not limited to using the
    features the worker provides out of the box. Instead, you can modify the ARM
    template to use any features available in Azure Container Instances.
"""
import datetime
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import anyio
import dateutil.parser
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.core.polling import LROPoller
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import Container, ContainerGroup, Logs
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import Deployment, DeploymentExtended, DeploymentMode, DeploymentProperties
from pydantic import Field, SecretStr
from slugify import slugify

from prefect.client.orchestration import get_client
from prefect.client.schemas import FlowRun
from prefect.server.schemas.core import Flow
from prefect.server.schemas.responses import DeploymentResponse
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult
from prefect_azure.container_instance import ACRManagedIdentity
from prefect_azure.credentials import AzureContainerInstanceCredentials

ACI_DEFAULT_CPU = 1.0
ACI_DEFAULT_MEMORY = 1.0
ACI_DEFAULT_GPU = 0.0
DEFAULT_CONTAINER_ENTRYPOINT = '/opt/prefect/entrypoint.sh'
ENV_SECRETS: List[str] = ['PREFECT_API_KEY']
CONTAINER_GROUP_DELETION_TIMEOUT_SECONDS = 30
DockerRegistry = Union[ACRManagedIdentity, Any, None]


def _get_default_arm_template() -> Dict[str, Any]:
    """Get the default ARM template for creating a container group."""
    return {
        '$schema': 'https://schema.management.azure.com/schemas/2019-08-01/deploymentTemplate.json#',
        'contentVersion': '1.0.0.0',
        'parameters': {
            'location': {
                'type': 'string',
                'defaultValue': '[resourceGroup().location]',
                'metadata': {'description': 'Location for all resources.'},
            },
            'container_group_name': {
                'type': 'string',
                'defaultValue': '[uniqueString(resourceGroup().id)]',
                'metadata': {'description': 'The name of the container group to create.'},
            },
            'container_name': {
                'type': 'string',
                'defaultValue': '[uniqueString(resourceGroup().id)]',
                'metadata': {'description': 'The name of the container to create.'},
            },
        },
        'resources': [
            {
                'type': 'Microsoft.ContainerInstance/containerGroups',
                'apiVersion': '2022-09-01',
                'name': "[parameters('container_group_name')]",
                'location': "[parameters('location')]",
                'properties': {
                    'containers': [
                        {
                            'name': "[parameters('container_name')]",
                            'properties': {
                                'image': '{{ image }}',
                                'command': '{{ command }}',
                                'resources': {'requests': {'cpu': '{{ cpu }}', 'memoryInGB': '{{ memory }}'}},
                                'environmentVariables': [],
                            },
                        }
                    ],
                    'osType': 'Linux',
                    'restartPolicy': 'Never',
                },
            }
        ],
    }


class ContainerGroupProvisioningState(str, Enum):
    """
    Terminal provisioning states for ACI container groups. Per the Azure docs,
    the states in this Enum are the only ones that can be relied on as dependencies.
    """
    SUCCEEDED = 'Succeeded'
    FAILED = 'Failed'


class ContainerRunState(str, Enum):
    """
    Terminal run states for ACI containers.
    """
    RUNNING = 'Running'
    TERMINATED = 'Terminated'


class AzureContainerJobConfiguration(BaseJobConfiguration):
    """
    Configuration for an Azure Container Instance flow run.
    """
    image: str = Field(default_factory=get_prefect_image_name)
    resource_group_name: str = Field(default=...)
    subscription_id: SecretStr = Field(default=...)
    identities: Optional[List[str]] = Field(default=None)
    entrypoint: str = Field(default=DEFAULT_CONTAINER_ENTRYPOINT)
    image_registry: Optional[DockerRegistry] = Field(default=None)
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

    def prepare_for_flow_run(
        self, flow_run: FlowRun, deployment: Optional[DeploymentResponse] = None, flow: Optional[Flow] = None
    ) -> None:
        """
        Prepares the job configuration for a flow run.
        """
        super().prepare_for_flow_run(flow_run, deployment, flow)
        container_group = self.arm_template['resources'][0]
        container = container_group['properties']['containers'][0]
        container['properties']['environmentVariables'] = self._get_arm_environment()
        if self.command:
            container['properties']['command'] = self.command.split(' ')
        self._add_image()
        if self.entrypoint:
            container['properties']['command'].insert(0, self.entrypoint)
        if self.image_registry:
            self._add_image_registry_credentials(self.image_registry)
        if self.identities:
            self._add_identities(self.identities)
        if self.subnet_ids:
            self._add_subnets(self.subnet_ids)
        if self.dns_servers:
            self._add_dns_servers(self.dns_servers)

    def _add_image(self) -> None:
        """
        Add the image to the arm template.
        """
        try:
            self.arm_template['resources'][0]['properties']['containers'][0]['properties']['image'] = self.image
        except KeyError:
            raise ValueError('Unable to add image due to invalid job ARM template.')

    def _add_image_registry_credentials(self, image_registry: Optional[DockerRegistry]) -> None:
        """
        Create image registry credentials based on the type of image_registry provided.

        Args:
            image_registry: An instance of a DockerRegistry or
            ACRManagedIdentity object.
        """
        if not image_registry:
            return
        if isinstance(image_registry, ACRManagedIdentity):
            self.arm_template['resources'][0]['properties']['imageRegistryCredentials'] = [{
                'server': image_registry.registry_url,
                'identity': image_registry.identity,
            }]
        elif hasattr(image_registry, 'username') and hasattr(image_registry, 'password') and hasattr(image_registry, 'registry_url'):
            self.arm_template['resources'][0]['properties']['imageRegistryCredentials'] = [{
                'server': image_registry.registry_url,
                'username': image_registry.username,
                'password': image_registry.password.get_secret_value(),  # type: ignore
            }]

    def _add_identities(self, identities: List[str]) -> None:
        """
        Add identities to the container group.

        Args:
            identities: A list of user-assigned identities to add to
            the container group.
        """
        self.arm_template['resources'][0]['identity'] = {
            'type': 'UserAssigned',
            'userAssignedIdentities': {identity: {} for identity in identities},
        }

    def _add_subnets(self, subnet_ids: List[str]) -> None:
        """
        Add subnets to the container group.

        Args:
            subnet_ids: A list of subnet ids to add to the container group.
        """
        self.arm_template['resources'][0]['properties']['subnetIds'] = [{'id': subnet_id} for subnet_id in subnet_ids]

    def _add_dns_servers(self, dns_servers: List[str]) -> None:
        """
        Add dns servers to the container group.

        Args:
            dns_servers: A list of dns servers to add to the container group.
        """
        self.arm_template['resources'][0]['properties']['dnsConfig'] = {'nameServers': dns_servers}

    def _get_arm_environment(self) -> List[Dict[str, Any]]:
        """
        Returns the environment variables to pass to the ARM template.
        """
        env: Dict[str, str] = {**self._base_environment(), **self.env}  # type: ignore
        azure_env: List[Dict[str, Any]] = [
            {'name': key, 'secureValue': value} if key in ENV_SECRETS else {'name': key, 'value': value}
            for key, value in env.items()
        ]
        return azure_env


class AzureContainerVariables(BaseVariables):
    """
    Variables for an Azure Container Instance flow run.
    """
    image: str = Field(default_factory=get_prefect_image_name, description='The image to use for the Prefect container in the task. This value defaults to a Prefect base image matching your local versions.')
    resource_group_name: str = Field(default=..., title='Azure Resource Group Name', description='The name of the Azure Resource Group in which to run Prefect ACI tasks.')
    subscription_id: SecretStr = Field(default=..., title='Azure Subscription ID', description='The ID of the Azure subscription to create containers under.')
    identities: Optional[List[str]] = Field(title='Identities', default=None, description="A list of user-assigned identities to associate with the container group. The identities should be an ARM resource IDs in the form: '/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{identityName}'.")
    entrypoint: str = Field(default=DEFAULT_CONTAINER_ENTRYPOINT, description='The entrypoint of the container you wish you run. This value defaults to the entrypoint used by Prefect images and should only be changed when using a custom image that is not based on an official Prefect image. Any commands set on deployments will be passed to the entrypoint as parameters.')
    image_registry: Optional[DockerRegistry] = Field(default=None, title='Image Registry (Optional)', description='To use any private container registry with a username and password, choose DockerRegistry. To use a private Azure Container Registry with a managed identity, choose ACRManagedIdentity.')
    cpu: float = Field(title='CPU', default=ACI_DEFAULT_CPU, description=f'The number of virtual CPUs to assign to the task container. If not provided, a default value of {ACI_DEFAULT_CPU} will be used.')
    gpu_count: Optional[int] = Field(title='GPU Count', default=None, description='The number of GPUs to assign to the task container. If not provided, no GPU will be used.')
    gpu_sku: Optional[str] = Field(title='GPU SKU', default=None, description='The Azure GPU SKU to use. See the ACI documentation for a list of GPU SKUs available in each Azure region.')
    memory: float = Field(default=ACI_DEFAULT_MEMORY, description=f'The amount of memory in gigabytes to provide to the ACI task. Valid amounts are specified in the Azure documentation. If not provided, a default value of  {ACI_DEFAULT_MEMORY} will be used unless present on the task definition.')
    subnet_ids: Optional[List[str]] = Field(title='Subnet IDs', default=None, description='A list of subnet IDs to associate with the container group.')
    dns_servers: Optional[List[str]] = Field(title='DNS Servers', default=None, description='A list of DNS servers to associate with the container group.')
    aci_credentials: AzureContainerInstanceCredentials = Field(default_factory=AzureContainerInstanceCredentials, description='The credentials to use to authenticate with Azure.')
    stream_output: bool = Field(default=False, description='If `True`, logs will be streamed from the Prefect container to the local console.')
    task_start_timeout_seconds: int = Field(default=240, description='The amount of time to watch for the start of the ACI container. before marking it as failed.')
    task_watch_poll_interval: float = Field(default=5.0, description='The number of seconds to wait between Azure API calls while monitoring the state of an Azure Container Instances task.')
    keep_container_group: bool = Field(default=False, title='Keep Container Group After Completion', description='Keep the completed container group on Azure.')


class AzureContainerWorkerResult(BaseWorkerResult):
    """Contains information about the final state of a completed process"""


class AzureContainerWorker(BaseWorker):
    """
    A Prefect worker that runs flows in an Azure Container Instance.
    """
    type: str = 'azure-container-instance'
    job_configuration = AzureContainerJobConfiguration
    job_configuration_variables = AzureContainerVariables
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/54e3fa7e00197a4fbd1d82ed62494cb58d08c96a-250x250.png'
    _display_name: str = 'Azure Container Instances'
    _description: str = "Execute flow runs within containers on Azure's Container Instances service. Requires an Azure account."
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-azure'

    async def run(
        self, flow_run: FlowRun, configuration: AzureContainerJobConfiguration, task_status: Optional[Any] = None
    ) -> AzureContainerWorkerResult:
        """
        Run a flow in an Azure Container Instance.

        Args:
            flow_run: The flow run to run.
            configuration: The configuration for the flow run.
            task_status: The task status object for the current task. Used
                to provide an identifier that can be used to cancel the task.

        Returns:
            The result of the flow run.
        """
        run_start_time: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        prefect_client = get_client()
        flow: Flow = await prefect_client.read_flow(flow_run.flow_id)
        slugified_flow_name: str = slugify(
            flow.name, max_length=55 - len(str(flow_run.id)), regex_pattern='[^a-zA-Z0-9-]+'
        )
        container_group_name: str = f'{slugified_flow_name}-{flow_run.id}'
        self._logger.info(f'{self._log_prefix}: Preparing to run command {configuration.command} in container  {configuration.image})...')
        aci_client: ContainerInstanceManagementClient = configuration.aci_credentials.get_container_client(configuration.subscription_id.get_secret_value())
        resource_client: ResourceManagementClient = configuration.aci_credentials.get_resource_client(configuration.subscription_id.get_secret_value())
        created_container_group: Optional[ContainerGroup] = None
        try:
            self._logger.info(f'{self._log_prefix}: Creating container group...')
            created_container_group = await self._provision_container_group(aci_client, resource_client, configuration, container_group_name)
            identifier: str = f'{flow_run.id}:{container_group_name}'
            if self._provisioning_succeeded(created_container_group):
                self._logger.info(f'{self._log_prefix}: Running command...')
                if task_status is not None:
                    task_status.started(value=identifier)
                status_code: int = await run_sync_in_worker_thread(
                    self._watch_task_and_get_exit_code, aci_client, configuration, created_container_group, run_start_time
                )
                self._logger.info(f'{self._log_prefix}: Completed command run.')
            else:
                raise RuntimeError(f'{self._log_prefix}: Container creation failed.')
        finally:
            if configuration.keep_container_group:
                self._logger.info(f'{self._log_prefix}: Stopping container group...')
                aci_client.container_groups.stop(
                    resource_group_name=configuration.resource_group_name, container_group_name=container_group_name
                )
            else:
                await self._wait_for_container_group_deletion(aci_client, configuration, container_group_name)
        return AzureContainerWorkerResult(identifier=created_container_group.name, status_code=status_code)

    def _wait_for_task_container_start(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str,
        creation_status_poller: LROPoller
    ) -> Optional[ContainerGroup]:
        """
        Wait for the result of group and container creation.

        Args:
            creation_status_poller: Poller returned by the Azure SDK.

        Raises:
            RuntimeError: Raised if the timeout limit is exceeded before the
                container starts.

        Returns:
            A `ContainerGroup` representing the current status of the group being
            watched, or None if creation failed.
        """
        t0: float = time.time()
        timeout: int = configuration.task_start_timeout_seconds
        while not creation_status_poller.done():
            elapsed_time: float = time.time() - t0
            if timeout and elapsed_time > timeout:
                raise RuntimeError(f'Timed out after {elapsed_time}s while watching waiting for container start.')
            time.sleep(configuration.task_watch_poll_interval)
        deployment: Deployment = creation_status_poller.result()
        provisioning_succeeded: bool = deployment.properties.provisioning_state == ContainerGroupProvisioningState.SUCCEEDED
        if provisioning_succeeded:
            return self._get_container_group(client, configuration.resource_group_name, container_group_name)
        else:
            return None

    async def _provision_container_group(
        self,
        aci_client: ContainerInstanceManagementClient,
        resource_client: ResourceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str
    ) -> ContainerGroup:
        """
        Create a container group and wait for it to start.

        Args:
            aci_client: An authenticated ACI client.
            resource_client: An authenticated resource client.
            configuration: The job configuration.
            container_group_name: The name of the container group to create.

        Returns:
            A `ContainerGroup` representing the container group that was created.
        """
        properties: DeploymentProperties = DeploymentProperties(
            mode=DeploymentMode.INCREMENTAL, template=configuration.arm_template, 
            parameters={'container_group_name': {'value': container_group_name}}
        )
        deployment: Deployment = Deployment(properties=properties)
        creation_status_poller: LROPoller = await run_sync_in_worker_thread(
            resource_client.deployments.begin_create_or_update,
            resource_group_name=configuration.resource_group_name,
            deployment_name=f'prefect-{container_group_name}',
            parameters=deployment
        )
        created_container_group: Optional[ContainerGroup] = await run_sync_in_worker_thread(
            self._wait_for_task_container_start, aci_client, configuration, container_group_name, creation_status_poller
        )
        # Assuming created_container_group is not None if provisioning succeeds.
        assert created_container_group is not None
        return created_container_group

    def _watch_task_and_get_exit_code(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        run_start_time: datetime.datetime
    ) -> int:
        """
        Waits until the container finishes running and obtains its exit code.

        Args:
            client: An initialized Azure `ContainerInstanceManagementClient`
            container_group: The `ContainerGroup` in which the container resides.
            run_start_time: The start time of the flow run.

        Returns:
            An `int` representing the container's exit code.
        """
        status_code: int = -1
        running_container: Container = self._get_container(container_group)
        current_state: str = running_container.instance_view.current_state.state  # type: ignore
        last_log_time: datetime.datetime = run_start_time
        if configuration.stream_output:
            last_log_time = self._get_and_stream_output(
                client=client, configuration=configuration, container_group=container_group, last_log_time=last_log_time
            )
        if current_state == ContainerRunState.TERMINATED:
            status_code = running_container.instance_view.current_state.exit_code  # type: ignore
        while current_state != ContainerRunState.TERMINATED:
            try:
                container_group = self._get_container_group(client, configuration.resource_group_name, container_group.name)
            except ResourceNotFoundError:
                self._logger.exception(
                    f'{self._log_prefix}: Container group was deleted before flow run completed, likely due to flow cancellation.'
                )
                return status_code
            container: Container = self._get_container(container_group)
            current_state = container.instance_view.current_state.state  # type: ignore
            if current_state == ContainerRunState.TERMINATED:
                status_code = container.instance_view.current_state.exit_code  # type: ignore
                break
            if configuration.stream_output:
                last_log_time = self._get_and_stream_output(
                    client=client, configuration=configuration, container_group=container_group, last_log_time=last_log_time
                )
            time.sleep(configuration.task_watch_poll_interval)
        return status_code

    async def _wait_for_container_group_deletion(
        self,
        aci_client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group_name: str
    ) -> None:
        """
        Wait for the container group to be deleted.

        Args:
            aci_client: An authenticated ACI client.
            configuration: The job configuration.
            container_group_name: The name of the container group to delete.
        """
        self._logger.info(f'{self._log_prefix}: Deleting container...')
        deletion_status_poller: LROPoller = await run_sync_in_worker_thread(
            aci_client.container_groups.begin_delete,
            resource_group_name=configuration.resource_group_name,
            container_group_name=container_group_name
        )
        t0: float = time.time()
        timeout: int = CONTAINER_GROUP_DELETION_TIMEOUT_SECONDS
        while not deletion_status_poller.done():
            elapsed_time: float = time.time() - t0
            if timeout and elapsed_time > timeout:
                raise RuntimeError(
                    f'Timed out after {elapsed_time}s while waiting for deletion of container group {container_group_name}. To verify the group has been deleted, check the Azure Portal or run az container show --name {container_group_name} --resource-group {configuration.resource_group_name}'
                )
            await anyio.sleep(configuration.task_watch_poll_interval)
        self._logger.info(f'{self._log_prefix}: Container deleted.')

    def _get_container(self, container_group: ContainerGroup) -> Container:
        """
        Extracts the job container from a container group.
        """
        return container_group.containers[0]

    @staticmethod
    def _get_container_group(
        client: ContainerInstanceManagementClient, resource_group_name: str, container_group_name: str
    ) -> ContainerGroup:
        """
        Gets the container group from Azure.
        """
        return client.container_groups.get(resource_group_name=resource_group_name, container_group_name=container_group_name)

    def _get_and_stream_output(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        last_log_time: datetime.datetime
    ) -> datetime.datetime:
        """
        Fetches logs output from the job container and writes all entries after
        a given time to stderr.

        Args:
            client: An initialized `ContainerInstanceManagementClient`
            container_group: The container group that holds the job container.
            last_log_time: The timestamp of the last output line already streamed.

        Returns:
            The time of the most recent output line written by this call.
        """
        logs: str = self._get_logs(client=client, configuration=configuration, container_group=container_group)
        return self._stream_output(logs, last_log_time)

    def _get_logs(
        self,
        client: ContainerInstanceManagementClient,
        configuration: AzureContainerJobConfiguration,
        container_group: ContainerGroup,
        max_lines: int = 100
    ) -> str:
        """
        Gets the most container logs up to a given maximum.

        Args:
            client: An initialized `ContainerInstanceManagementClient`
            container_group: The container group that holds the job container.
            max_lines: The number of log lines to pull. Defaults to 100.

        Returns:
            A string containing the requested log entries, one per line.
        """
        container: Container = self._get_container(container_group)
        logs_obj: Optional[Logs] = None
        try:
            logs_obj = client.containers.list_logs(
                resource_group_name=configuration.resource_group_name,
                container_group_name=container_group.name,
                container_name=container.name,
                tail=max_lines,
                timestamps=True
            )
        except HttpResponseError:
            self._logger.warning(
                f'{self._log_prefix}: Unable to retrieve logs from container {container.name}. Trying again in {configuration.task_watch_poll_interval}s'
            )
        return logs_obj.content if logs_obj else ''

    def _stream_output(self, log_content: str, last_log_time: datetime.datetime) -> datetime.datetime:
        """
        Writes each entry from a string of log lines to stderr.

        Args:
            log_content: A string containing Azure container logs.
            last_log_time: The timestamp of the last output line already streamed.

        Returns:
            The time of the most recent output line written by this call.
        """
        if not log_content:
            return last_log_time
        log_lines: List[str] = log_content.split('\n')
        last_written_time: datetime.datetime = last_log_time
        for log_line in log_lines:
            if not log_line.strip():
                continue
            line_parts: List[str] = log_line.split(' ')
            line_timestamp: str = line_parts[0]
            line: str = ' '.join(line_parts[1:])
            try:
                line_time: datetime.datetime = dateutil.parser.parse(line_timestamp)
                if line_time > last_written_time:
                    self._write_output_line(line)
                    last_written_time = line_time
            except dateutil.parser.ParserError as e:
                self._logger.debug(f'{self._log_prefix}: Unable to parse timestamp from Azure log line: %s', log_line, exc_info=e)
        return last_written_time

    @property
    def _log_prefix(self) -> str:
        """
        Internal property for generating a prefix for logs where `name` may be null
        """
        if self.name is not None:
            return f'AzureContainerInstanceJob {self.name!r}'
        else:
            return 'AzureContainerInstanceJob'

    @staticmethod
    def _provisioning_succeeded(container_group: Optional[ContainerGroup]) -> bool:
        """
        Determines whether ACI container group provisioning was successful.

        Args:
            container_group: a container group returned by the Azure SDK.

        Returns:
            True if provisioning was successful, False otherwise.
        """
        if not container_group:
            return False
        return container_group.provisioning_state == ContainerGroupProvisioningState.SUCCEEDED and len(container_group.containers) == 1

    @staticmethod
    def _write_output_line(line: str) -> None:
        """
        Writes a line of output to stderr.
        """
        print(line, file=sys.stderr)