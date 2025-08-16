import re
import shlex
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional
from uuid import uuid4
from anyio.abc import TaskStatus
from google.api_core.client_options import ClientOptions
from googleapiclient import discovery
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from jsonpatch import JsonPatch
from pydantic import Field, PrivateAttr, field_validator
from prefect.logging.loggers import PrefectLogAdapter
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import get_prefect_image_name
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker, BaseWorkerResult
from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.models.cloud_run_v2 import ExecutionV2, JobV2, SecretKeySelector
from prefect_gcp.utilities import slugify_name

if TYPE_CHECKING:
    from prefect.client.schemas import FlowRun
    from prefect.server.schemas.core import Flow
    from prefect.server.schemas.responses import DeploymentResponse

def _get_default_job_body_template() -> Dict[str, Any]:
    ...

def _get_base_job_body() -> Dict[str, Any]:
    ...

class CloudRunWorkerJobV2Configuration(BaseJobConfiguration):
    ...

    def _populate_timeout(self) -> None:
        ...

    def _populate_env(self) -> None:
        ...

    def _configure_cloudsql_volumes(self) -> None:
        ...

    def _populate_image_if_not_present(self) -> None:
        ...

    def _populate_or_format_command(self) -> None:
        ...

    def _format_args_if_present(self) -> None:
        ...

    def _remove_vpc_access_if_unset(self) -> None:
        ...

    @field_validator('job_body')
    @classmethod
    def _ensure_job_includes_all_required_components(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @field_validator('job_body')
    @classmethod
    def _ensure_job_has_compatible_values(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        ...

class CloudRunWorkerV2Variables(BaseVariables):
    ...

class CloudRunWorkerV2Result(BaseWorkerResult):
    ...

class CloudRunWorkerV2(BaseWorker):
    ...

    async def run(self, flow_run: 'FlowRun', configuration: CloudRunWorkerJobV2Configuration, task_status: Optional[TaskStatus] = None) -> CloudRunWorkerV2Result:
        ...

    @staticmethod
    def _get_client(configuration: CloudRunWorkerJobV2Configuration) -> Resource:
        ...

    def _create_job_and_wait_for_registration(self, configuration: CloudRunWorkerJobV2Configuration, cr_client: Resource, logger: PrefectLogAdapter) -> None:
        ...

    @staticmethod
    def _wait_for_job_creation(cr_client: Resource, configuration: CloudRunWorkerJobV2Configuration, logger: PrefectLogAdapter, poll_interval: int = 5) -> None:
        ...

    @staticmethod
    def _create_job_error(exc: HttpError, configuration: CloudRunWorkerJobV2Configuration) -> None:
        ...

    def _begin_job_execution(self, cr_client: Resource, configuration: CloudRunWorkerJobV2Configuration, logger: PrefectLogAdapter) -> ExecutionV2:
        ...

    def _watch_job_execution_and_get_result(self, cr_client: Resource, configuration: CloudRunWorkerJobV2Configuration, execution: ExecutionV2, logger: PrefectLogAdapter, poll_interval: int = 5) -> CloudRunWorkerV2Result:
        ...

    @staticmethod
    def _watch_job_execution(cr_client: Resource, configuration: CloudRunWorkerJobV2Configuration, execution: ExecutionV2, poll_interval: int) -> ExecutionV2:
        ...

    @staticmethod
    def _job_run_submission_error(exc: Exception, configuration: CloudRunWorkerJobV2Configuration) -> None:
        ...
