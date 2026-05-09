from __future__ import annotations
import ast
import asyncio
import math
import os
import shutil
from datetime import timedelta
from getpass import GetPassWarning
from typing import TYPE_CHECKING, Any, Coroutine, Optional, TypeVar, Union, overload
import anyio
import readchar
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from prefect.cli._utilities import exit_with_error
from prefect.client.collections import get_collections_metadata_client
from prefect.client.schemas.actions import BlockDocumentCreate, DeploymentScheduleCreate, WorkPoolCreate
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule, RRuleSchedule, is_valid_timezone
from prefect.client.utilities import client_injector
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound
from prefect.flows import load_flow_from_entrypoint
from prefect.logging.loggers import get_logger
from prefect.settings import PREFECT_DEBUG_MODE
from prefect.utilities import urls
from prefect.utilities._git import get_git_remote_origin_url
from prefect.utilities.asyncutils import LazySemaphore
from prefect.utilities.filesystem import filter_files, get_open_file_limit
from prefect.utilities.slugify import slugify
if TYPE_CHECKING:
    from prefect.client.orchestration import PrefectClient
T = TypeVar('T', bound=RenderableType)

async def find_flow_functions_in_file(path: str) -> list[dict]:
    ...

async def search_for_flow_functions(directory: str = '.', exclude_patterns: Optional[list[str]] = None) -> list[dict]:
    ...

def prompt(message: str, **kwargs: Any) -> str:
    ...

def confirm(message: str, **kwargs: Any) -> bool:
    ...

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: list[dict], data: list[dict], table_kwargs: Optional[dict] = None, opt_out_message: str = '', opt_out_response: Optional[Any] = None) -> dict:
    ...

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: list[dict], data: list[dict], table_kwargs: Optional[dict] = None, opt_out_message: str, opt_out_response: Optional[Any] = None) -> dict:
    ...

def prompt_select_from_table(console: Console, prompt: str, columns: list[dict], data: list[dict], table_kwargs: Optional[dict] = None, opt_out_message: str = '', opt_out_response: Optional[Any] = None) -> dict:
    ...

class IntervalValuePrompt(PromptBase[timedelta]):
    ...

def prompt_interval_schedule(console: Console) -> IntervalSchedule:
    ...

class CronStringPrompt(PromptBase[str]):
    ...

def prompt_cron_schedule(console: Console) -> CronSchedule:
    ...

class RRuleStringPrompt(PromptBase[str]):
    ...

def prompt_rrule_schedule(console: Console) -> RRuleSchedule:
    ...

def prompt_schedule_type(console: Console) -> str:
    ...

def prompt_schedules(console: Console) -> list[DeploymentScheduleCreate]:
    ...

async def prompt_build_custom_docker_image(console: Console, deployment_config: Any) -> dict:
    ...

async def prompt_push_custom_docker_image(console: Console, deployment_config: Any, build_docker_image_step: dict) -> tuple[dict, dict]:
    ...

@client_injector
async def prompt_create_work_pool(client: PrefectClient, console: Console) -> WorkPoolCreate:
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
