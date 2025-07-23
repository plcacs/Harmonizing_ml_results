"""<!-- # noqa -->

Module containing the Cloud Run worker used for executing flow runs as Cloud Run jobs.
"""
import re
import shlex
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import anyio
import googleapiclient
from anyio.abc import TaskStatus
from google.api_core.client_options import ClientOptions
from googleapiclient import discovery
from googleapiclient.discovery import Resource
from jsonpatch import JsonPatch
from pydantic import Field, field_validator
from prefect.logging.loggers import PrefectLogAdapter
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.utilities import Execution, Job, slugify_name

if TYPE_CHECKING:
    from prefect.client.schemas import FlowRun
    from prefect.server.schemas.core import Flow
    from prefect.server.schemas.responses import DeploymentResponse

def _get_default_job_body_template() -> Dict[str, Any]:
    """Returns the default job body template used by the Cloud Run Job."""
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'name': '{{ name }}', 'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{'image': '{{ image }}', 'command': '{{ command }}', 'resources': {'limits': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}, 'requests': {'cpu': '{{ cpu }}', 'memory': '{{ memory }}'}}}], 'timeoutSeconds': '{{ timeout }}', 'serviceAccountName': '{{ service_account_name }}'}}}, 'metadata': {'annotations': {'run.googleapis.com/vpc-access-connector': '{{ vpc_connector_name }}'}}}}}

def _get_base_job_body() -> Dict[str, Any]:
    """Returns a base job body to use for job body validation."""
    return {'apiVersion': 'run.googleapis.com/v1', 'kind': 'Job', 'metadata': {'annotations': {'run.googleapis.com/launch-stage': 'BETA'}}, 'spec': {'template': {'spec': {'template': {'spec': {'containers': [{}]}}}}}

class CloudRunWorkerJobConfiguration(BaseJobConfiguration):
    """
    Configuration class used by the Cloud Run Worker to create a Cloud Run Job.
    """
    region: str = Field(default='us-central1', description='The region where the Cloud Run Job resides.')
    credentials: GcpCredentials = Field(title='GCP Credentials', default_factory=GcpCredentials, description='The GCP Credentials used to connect to Cloud Run. If not provided credentials will be inferred from the local environment.')
    job_body: Dict[str, Any] = Field(json_schema_extra=dict(template=_get_default_job_body_template()))
    timeout: int = Field(default=600, gt=0, le=3600, title='Job Timeout', description='Max allowed duration the Job may be active before Cloud Run will actively try to mark it failed and kill associated containers (maximum of 3600 seconds, 1 hour).')
    keep_job: bool = Field(default=False, title='Keep Job After Completion', description='Keep the completed Cloud Run Job on Google Cloud Platform.')

    @property
    def project(self) -> str:
        """property for accessing the project from the credentials."""
        return self.credentials.project

    @property
    def job_name(self) -> str:
        """property for accessing the name from the job metadata."""
        return self.job_body['metadata']['name']

    def prepare_for_flow_run(
        self,
        flow_run: "FlowRun",
        deployment: Optional["DeploymentResponse"] = None,
        flow: Optional["Flow"] = None
    ) -> None:
        """
        Prepares the job configuration for a flow run.
        """
        super().prepare_for_flow_run(flow_run, deployment, flow)
        self._populate_envs()
        self._populate_or_format_command()
        self._format_args_if_present()
        self._populate_image_if_not_present()
        self._populate_name_if_not_present()

    def _populate_envs(self) -> None:
        """Populate environment variables."""
        envs = [{'name': k, 'value': v} for k, v in self.env.items()]
        self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]['env'] = envs

    def _populate_name_if_not_present(self) -> None:
        """Adds the flow run name to the job if one is not already provided."""
        try:
            if 'name' not in self.job_body['metadata']:
                base_job_name = slugify_name(self.name)
                job_name = f'{base_job_name}-{uuid4().hex}'
                self.job_body['metadata']['name'] = job_name
        except KeyError:
            raise ValueError('Unable to verify name due to invalid job body template.')

    def _populate_image_if_not_present(self) -> None:
        """Adds the latest prefect image to the job if one is not already provided."""
        try:
            if 'image' not in self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]:
                self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]['image'] = f'docker.io/{get_prefect_image_name()}'
        except KeyError:
            raise ValueError('Unable to verify image due to invalid job body template.')

    def _populate_or_format_command(self) -> None:
        """
        Ensures that the command is present in the job manifest.
        """
        try:
            command = self.job_body['spec']['template']['spec']['template']['spec']['containers'][0].get('command')
            if command is None:
                self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]['command'] = shlex.split(self._base_flow_run_command())
            elif isinstance(command, str):
                self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]['command'] = shlex.split(command)
        except KeyError:
            raise ValueError('Unable to verify command due to invalid job body template.')

    def _format_args_if_present(self) -> None:
        try:
            args = self.job_body['spec']['template']['spec']['template']['spec']['containers'][0].get('args')
            if args is not None and isinstance(args, str):
                self.job_body['spec']['template']['spec']['template']['spec']['containers'][0]['args'] = shlex.split(args)
        except KeyError:
            raise ValueError('Unable to verify args due to invalid job body template.')

    @field_validator('job_body')
    @classmethod
    def _ensure_job_includes_all_required_components(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures that the job body includes all required components.
        """
        patch = JsonPatch.from_diff(value, _get_base_job_body())
        missing_paths = sorted([op['path'] for op in patch if op['op'] == 'add'])
        if missing_paths:
            raise ValueError(f'Job is missing required attributes at the following paths: {", ".join(missing_paths)}')
        return value

    @field_validator('job_body')
    @classmethod
    def _ensure_job_has_compatible_values(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure that the job body has compatible values."""
        patch = JsonPatch.from_diff(value, _get_base_job_body())
        incompatible = sorted([f"{op['path']} must have value {op['value']!r}" for op in patch if op['op'] == 'replace'])
        if incompatible:
            raise ValueError(f'Job has incompatible values for the following attributes: {", ".join(incompatible)}')
        return value

class CloudRunWorkerVariables(BaseVariables):
    """
    Default variables for the Cloud Run worker.
    """
    region: str = Field(default='us-central1', description='The region where the Cloud Run Job resides.', examples=['us-central1'])
    credentials: GcpCredentials = Field(title='GCP Credentials', default_factory=GcpCredentials, description='The GCP Credentials used to initiate the Cloud Run Job. If not provided credentials will be inferred from the local environment.')
    image: Optional[str] = Field(default=None, title='Image Name', description='The image to use for a new Cloud Run Job. If not set, the latest Prefect image will be used. See https://cloud.google.com/run/docs/deploying#images.', examples=['docker.io/prefecthq/prefect:3-latest'])
    cpu: Optional[str] = Field(default=None, title='CPU', description='The amount of compute allocated to the Cloud Run Job. (1000m = 1 CPU). See https://cloud.google.com/run/docs/configuring/cpu#setting-jobs.', examples=['1000m'], pattern='^(\\d*000)m$')
    memory: Optional[str] = Field(default=None, title='Memory', description="The amount of memory allocated to the Cloud Run Job. Must be specified in units of 'G', 'Gi', 'M', or 'Mi'. See https://cloud.google.com/run/docs/configuring/memory-limits#setting.", examples=['512Mi'], pattern='^\\d+(?:G|Gi|M|Mi)$')
    vpc_connector_name: Optional[str] = Field(default=None, title='VPC Connector Name', description='The name of the VPC connector to use for the Cloud Run Job.')
    service_account_name: Optional[str] = Field(default=None, title='Service Account Name', description='The name of the service account to use for the task execution of Cloud Run Job. By default Cloud Run jobs run as the default Compute Engine Service Account. ', examples=['service-account@example.iam.gserviceaccount.com'])
    keep_job: bool = Field(default=False, title='Keep Job After Completion', description='Keep the completed Cloud Run Job after it has run.')
    timeout: int = Field(title='Job Timeout', default=600, gt=0, le=3600, description='Max allowed duration the Job may be active before Cloud Run will actively try to mark it failed and kill associated containers (maximum of 3600 seconds, 1 hour).')

class CloudRunWorkerResult(BaseWorkerResult):
    """Contains information about the final state of a completed process"""

class CloudRunWorker(BaseWorker):
    """Prefect worker that executes flow runs within Cloud Run Jobs."""
    type: str = 'cloud-run'
    job_configuration: type = CloudRunWorkerJobConfiguration
    job_configuration_variables: type = CloudRunWorkerVariables
    _description: str = 'Execute flow runs within containers on Google Cloud Run. Requires a Google Cloud Platform account.'
    _display_name: str = 'Google Cloud Run'
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-gcp'
    _logo_url: str = 'https://cdn.sanity.io/images/3ugk85nk/production/10424e311932e31c477ac2b9ef3d53cefbaad708-250x250.png'

    def _create_job_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
        """Provides a nicer error for 404s when trying to create a Cloud Run Job."""
        if getattr(exc, 'status_code', None) == 404:
            raise RuntimeError(f"Failed to find resources at {getattr(exc, 'uri', '')}. Confirm that region '{configuration.region}' is the correct region for your Cloud Run Job and that {configuration.project} is the correct GCP project. If your project ID is not correct, you are using a Credentials block with permissions for the wrong project.") from exc
        raise exc

    def _job_run_submission_error(self, exc: Exception, configuration: CloudRunWorkerJobConfiguration) -> None:
        """Provides a nicer error for 404s when submitting job runs."""
        if getattr(exc, 'status_code', None) == 404:
            pat1 = 'The requested URL [^ ]+ was not found on this server'
            if re.findall(pat1, str(exc)):
                raise RuntimeError(f"Failed to find resources at {getattr(exc, 'uri', '')}. Confirm that region '{configuration.region}' is the correct region for your Cloud Run Job and that '{configuration.project}' is the correct GCP project. If your project ID is not correct, you are using a Credentials block with permissions for the wrong project.") from exc
            else:
                raise exc
        raise exc

    async def run(
        self,
        flow_run: "FlowRun",
        configuration: CloudRunWorkerJobConfiguration,
        task_status: Optional[TaskStatus] = None
    ) -> CloudRunWorkerResult:
        """
        Executes a flow run within a Cloud Run Job.
        """
        logger = self.get_flow_run_logger(flow_run)
        with self._get_client(configuration) as client:
            await run_sync_in_worker_thread(self._create_job_and_wait_for_registration, configuration, client, logger)
            job_execution = await run_sync_in_worker_thread(self._begin_job_execution, configuration, client, logger)
            if task_status:
                task_status.started(configuration.job_name)
            result = await run_sync_in_worker_thread(self._watch_job_execution_and_get_result, configuration, client, job_execution, logger)
            return result

    def _get_client(self, configuration: CloudRunWorkerJobConfiguration) -> Resource:
        """Get the base client needed for interacting with GCP APIs."""
        api_endpoint = f'https://{configuration.region}-run.googleapis.com'
        gcp_creds = configuration.credentials.get_credentials_from_service_account()
        options = ClientOptions(api_endpoint=api_endpoint)
        return discovery.build('run', 'v1', client_options=options, credentials=gcp_creds).namespaces()

    def _create_job_and_wait_for_registration(
        self,
        configuration: CloudRunWorkerJobConfiguration,
        client: Resource,
        logger: PrefectLogAdapter
    ) -> None:
        """Create a new job wait for it to finish registering."""
        try:
            logger.info(f'Creating Cloud Run Job {configuration.job_name}')
            Job.create(client=client, namespace=configuration.credentials.project, body=configuration.job_body)
        except googleapiclient.errors.HttpError as exc:
            self._create_job_error(exc, configuration)
        try:
            self._wait_for_job_creation(client=client, configuration=configuration, logger=logger)
        except Exception:
            logger.exception('Encountered an exception while waiting for job run creation')
            if not configuration.keep_job:
                logger.info(f'Deleting Cloud Run Job {configuration.job_name} from Google Cloud Run.')
                try:
                    Job.delete(client=client, namespace=configuration.credentials.project, job_name=configuration.job_name)
                except Exception:
                    logger.exception(f'Received an unexpected exception while attempting to delete Cloud Run Job {configuration.job_name!r}')
            raise

    def _begin_job_execution(
        self,
        configuration: CloudRunWorkerJobConfiguration,
        client: Resource,
        logger: PrefectLogAdapter
    ) -> Execution:
        """Submit a job run for execution and return the execution object."""
        try:
            logger.info(f'Submitting Cloud Run Job {configuration.job_name!r} for execution.')
            submission = Job.run(client=client, namespace=configuration.project, job_name=configuration.job_name)
            job_execution = Execution.get(client=client, namespace=submission['metadata']['namespace'], execution_name=submission['metadata']['name'])
        except Exception as exc:
            self._job_run_submission_error(exc, configuration)
        return job_execution

    def _watch_job_execution_and_get_result(
        self,
        configuration: CloudRunWorkerJobConfiguration,
        client: Resource,
        execution: Execution,
        logger: PrefectLogAdapter,
        poll_interval: int = 5
    ) -> CloudRunWorkerResult:
        """Wait for execution to complete and then return result."""
        try:
            job_execution = self._watch_job_execution(client=client, job_execution=execution, poll_interval=poll_interval)
        except Exception:
            logger.exception(f'Received an unexpected exception while monitoring Cloud Run Job {configuration.job_name!r}')
            raise
        if job_execution.succeeded():
            status_code = 0
            logger.info(f'Job Run {configuration.job_name} completed successfully')
        else:
            status_code = 1
            error_msg = job_execution.condition_after_completion()['message']
            logger.error(f'Job Run {configuration.job_name} did not complete successfully. {error_msg}')
        logger.info(f'Job Run logs can be found on GCP at: {job_execution.log_uri}')
        if not configuration.keep_job:
            logger.info(f'Deleting completed Cloud Run Job {configuration.job_name!r} from Google Cloud Run...')
            try:
                Job.delete(client=client, namespace=configuration.project, job_name=configuration.job_name)
            except Exception:
                logger.exception(f'Received an unexpected exception while attempting to delete Cloud Run Job {configuration.job_name}')
        return CloudRunWorkerResult(identifier=configuration.job_name, status_code=status_code)

    def _watch_job_execution(
        self,
        client: Resource,
        job_execution: Execution,
        poll_interval: int = 5
    ) -> Execution:
        """
        Update job_execution status until it is no longer running.
        """
        while job_execution.is_running():
            job_execution = Execution.get(client=client, namespace=job_execution.namespace, execution_name=job_execution.name)
            time.sleep(poll_interval)
        return job_execution

    def _wait_for_job_creation(
        self,
        client: Resource,
        configuration: CloudRunWorkerJobConfiguration,
        logger: PrefectLogAdapter,
        poll_interval: int = 5
    ) -> Job:
        """Give created job time to register."""
        job = Job.get(client=client, namespace=configuration.project, job_name=configuration