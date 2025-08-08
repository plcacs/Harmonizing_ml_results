from __future__ import annotations
import asyncio
import math
import shutil
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, TypeVar, Union, overload
import anyio
import readchar
from rich.console import Console, RenderableType
from rich.live import Live
from rich.prompt import Confirm, InvalidResponse, Prompt, PromptBase
from rich.table import Table
from prefect.client.schemas.actions import BlockDocumentCreate, DeploymentScheduleCreate, WorkPoolCreate
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule, RRuleSchedule
from prefect.client.utilities import client_injector
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound
from prefect.flows import load_flow_from_entrypoint
from prefect.logging.loggers import get_logger
from prefect.settings import PREFECT_DEBUG_MODE
from prefect.utilities import urls
from prefect.utilities.asyncutils import LazySemaphore
from prefect.utilities.filesystem import filter_files, get_open_file_limit
from prefect.utilities.processutils import get_sys_executable, run_process
from prefect.utilities.slugify import slugify
if TYPE_CHECKING:
    from prefect.client.orchestration import PrefectClient

T = TypeVar('T', bound=RenderableType)
STORAGE_PROVIDER_TO_CREDS_BLOCK: Dict[str, str] = {'s3': 'aws-credentials', 'gcs': 'gcp-credentials', 'azure_blob_storage': 'azure-blob-storage-credentials'}
REQUIRED_FIELDS_FOR_CREDS_BLOCK: Dict[str, List[str]] = {'aws-credentials': ['aws_access_key_id', 'aws_secret_access_key'], 'gcp-credentials': ['project', 'service_account_file'], 'azure-blob-storage-credentials': ['account_url', 'connection_string']}
OPEN_FILE_SEMAPHORE: LazySemaphore = LazySemaphore(lambda: math.floor(get_open_file_limit() * 0.5))
logger: Logger = get_logger(__name__)

async def find_flow_functions_in_file(path: str) -> List[Dict[str, str]]:
    ...

async def search_for_flow_functions(directory: str = '.', exclude_patterns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    ...

def prompt(message: str, **kwargs: Any) -> Any:
    ...

def confirm(message: str, **kwargs: Any) -> bool:
    ...

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, str]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: Optional[str] = None, opt_out_response: Optional[Any] = None) -> Dict[str, str]:
    ...

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, str]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: str = '', opt_out_response: Optional[Any] = None) -> Dict[str, str]:
    ...

def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, str]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: Optional[str] = None, opt_out_response: Optional[Any] = None) -> Dict[str, str]:
    ...

class IntervalValuePrompt(PromptBase[timedelta]):
    ...

def prompt_interval_schedule(console: Console) -> IntervalSchedule:
    ...

class CronStringPrompt(PromptBase[str]):
    ...

class CronTimezonePrompt(PromptBase[str]):
    ...

def prompt_cron_schedule(console: Console) -> CronSchedule:
    ...

class RRuleStringPrompt(PromptBase[str]):
    ...

class RRuleTimezonePrompt(PromptBase[str]):
    ...

def prompt_rrule_schedule(console: Console) -> RRuleSchedule:
    ...

def prompt_schedule_type(console: Console) -> str:
    ...

def prompt_schedules(console: Console) -> List[DeploymentScheduleCreate]:
    ...

@client_injector
async def prompt_select_work_pool(client: PrefectClient, console: Console, prompt: str = 'Which work pool would you like to deploy this flow to?') -> str:
    ...

async def prompt_build_custom_docker_image(console: Console, deployment_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ...

async def prompt_push_custom_docker_image(console: Console, deployment_config: Dict[str, Any], build_docker_image_step: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    ...

@client_injector
async def prompt_create_work_pool(client: PrefectClient, console: Console) -> str:
    ...

class EntrypointPrompt(PromptBase[str]):
    ...

async def prompt_entrypoint(console: Console) -> str:
    ...

@client_injector
async def prompt_select_remote_flow_storage(client: PrefectClient, console: Console) -> str:
    ...

@client_injector
async def prompt_select_blob_storage_credentials(client: PrefectClient, console: Console, storage_provider: str) -> str:
    ...
