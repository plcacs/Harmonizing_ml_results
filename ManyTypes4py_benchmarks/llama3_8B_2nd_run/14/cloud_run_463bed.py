from typing import TYPE_CHECKING, Any, Dict, Optional
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
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.utilities import Execution, Job, slugify_name

class CloudRunWorkerJobConfiguration(BaseJobConfiguration):
    ...
    region: str = Field(default='us-central1', description='The region where the Cloud Run Job resides.')
    credentials: GcpCredentials = Field(title='GCP Credentials', default_factory=GcpCredentials, description='The GCP Credentials used to connect to Cloud Run.')
    job_body: Dict[str, Any] = Field(json_schema_extra=dict(template=_get_default_job_body_template()))
    timeout: int = Field(default=600, gt=0, le=3600, title='Job Timeout', description='Max allowed duration the Job may be active before Cloud Run will actively try to mark it failed and kill associated containers (maximum of 3600 seconds, 1 hour).')
    keep_job: bool = Field(default=False, title='Keep Job After Completion', description='Keep the completed Cloud Run Job after it has run.')

class CloudRunWorkerVariables(BaseVariables):
    ...
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
    ...

class CloudRunWorker(BaseWorker):
    ...
    def _get_client(self, configuration: CloudRunWorkerJobConfiguration) -> Resource:
        ...

    def _create_job_and_wait_for_registration(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> None:
        ...

    def _begin_job_execution(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, logger: PrefectLogAdapter) -> Execution:
        ...

    def _watch_job_execution_and_get_result(self, configuration: CloudRunWorkerJobConfiguration, client: Resource, execution: Execution, logger: PrefectLogAdapter, poll_interval: int = 5) -> CloudRunWorkerResult:
        ...
