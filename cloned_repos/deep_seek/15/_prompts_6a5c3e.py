"""
Utilities for prompting the user for input
"""
from __future__ import annotations
import ast
import asyncio
import math
import os
import shutil
from datetime import timedelta
from getpass import GetPassWarning
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union, overload
import anyio
import readchar
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, InvalidResponse, Prompt, PromptBase
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
from prefect.utilities.processutils import get_sys_executable, run_process
from prefect.utilities.slugify import slugify
if TYPE_CHECKING:
    from prefect.client.orchestration import PrefectClient
    from prefect.blocks.core import Block
    from prefect.client.schemas.objects import BlockDocument, BlockSchema, BlockType, WorkerMetadata

T = TypeVar('T', bound=RenderableType)
STORAGE_PROVIDER_TO_CREDS_BLOCK: Dict[str, str] = {'s3': 'aws-credentials', 'gcs': 'gcp-credentials', 'azure_blob_storage': 'azure-blob-storage-credentials'}
REQUIRED_FIELDS_FOR_CREDS_BLOCK: Dict[str, List[str]] = {'aws-credentials': ['aws_access_key_id', 'aws_secret_access_key'], 'gcp-credentials': ['project', 'service_account_file'], 'azure-blob-storage-credentials': ['account_url', 'connection_string']}
OPEN_FILE_SEMAPHORE: LazySemaphore = LazySemaphore(lambda: math.floor(get_open_file_limit() * 0.5))
logger = get_logger(__name__)

async def find_flow_functions_in_file(path: Union[str, anyio.Path]) -> List[Dict[str, str]]:
    decorator_name = 'flow'
    decorator_module = 'prefect'
    decorated_functions: List[Dict[str, str]] = []
    async with OPEN_FILE_SEMAPHORE:
        try:
            async with await anyio.open_file(path) as f:
                try:
                    tree = ast.parse(await f.read())
                except SyntaxError:
                    if PREFECT_DEBUG_MODE:
                        get_logger().debug(f'Could not parse {path} as a Python file. Skipping.')
                    return decorated_functions
        except Exception as exc:
            if PREFECT_DEBUG_MODE:
                get_logger().debug(f'Could not open {path}: {exc}. Skipping.')
            return decorated_functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                is_name_match = isinstance(decorator, ast.Name) and decorator.id == decorator_name
                is_func_name_match = isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and (decorator.func.id == decorator_name)
                is_module_attribute_match = isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name) and (decorator.value.id == decorator_module) and (decorator.attr == decorator_name)
                is_module_attribute_func_match = isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and (decorator.func.attr == decorator_name) and isinstance(decorator.func.value, ast.Name) and (decorator.func.value.id == decorator_module)
                if is_name_match or is_module_attribute_match:
                    decorated_functions.append({'flow_name': node.name, 'function_name': node.name, 'filepath': str(path)})
                if is_func_name_match or is_module_attribute_func_match:
                    name_kwarg_node = None
                    if TYPE_CHECKING:
                        assert isinstance(decorator, ast.Call)
                    for kw in decorator.keywords:
                        if kw.arg == 'name':
                            name_kwarg_node = kw
                            break
                    if name_kwarg_node is not None and isinstance(name_kwarg_node.value, ast.Constant):
                        flow_name = name_kwarg_node.value.value
                    else:
                        flow_name = node.name
                    decorated_functions.append({'flow_name': flow_name, 'function_name': node.name, 'filepath': str(path)})
    return decorated_functions

async def search_for_flow_functions(directory: str = '.', exclude_patterns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Search for flow functions in the provided directory. If no directory is provided,
    the current working directory is used.

    Args:
        directory: The directory to search in
        exclude_patterns: List of patterns to exclude from the search, defaults to
            ["**/site-packages/**"]

    Returns:
        List[Dict]: the flow name, function name, and filepath of all flow functions found
    """
    path = anyio.Path(directory)
    exclude_patterns = exclude_patterns or ['**/site-packages/**']
    coros: List[Coroutine[Any, Any, List[Dict[str, str]]]] = []
    try:
        for file in filter_files(root=str(path), ignore_patterns=['*', '!**/*.py', *exclude_patterns], include_dirs=False):
            coros.append(find_flow_functions_in_file(anyio.Path(str(path / file))))
    except (PermissionError, OSError) as e:
        logger.error(f'Error searching for flow functions: {e}')
        return []
    return [fn for file_fns in await asyncio.gather(*coros) for fn in file_fns]

def prompt(message: str, **kwargs: Any) -> str:
    """Utility to prompt the user for input with consistent styling"""
    return Prompt.ask(f'[bold][green]?[/] {message}[/]', **kwargs)

def confirm(message: str, **kwargs: Any) -> bool:
    """Utility to prompt the user for confirmation with consistent styling"""
    return Confirm.ask(f'[bold][green]?[/] {message}[/]', **kwargs)

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, Any]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: Optional[str] = None, opt_out_response: Optional[Any] = None) -> Any:
    ...

@overload
def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, Any]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: str = '', opt_out_response: Optional[Any] = None) -> Any:
    ...

def prompt_select_from_table(console: Console, prompt: str, columns: List[Dict[str, str]], data: List[Dict[str, Any]], table_kwargs: Optional[Dict[str, Any]] = None, opt_out_message: Optional[str] = None, opt_out_response: Optional[Any] = None) -> Any:
    """
    Given a list of columns and some data, display options to user in a table
    and prompt them to select one.

    Args:
        prompt: A prompt to display to the user before the table.
        columns: A list of dicts with keys `header` and `key` to display in
            the table. The `header` value will be displayed in the table header
            and the `key` value will be used to lookup the value for each row
            in the provided data.
        data: A list of dicts with keys corresponding to the `key` values in
            the `columns` argument.
        table_kwargs: Additional kwargs to pass to the `rich.Table` constructor.
    Returns:
        dict: Data representation of the selected row
    """
    current_idx = 0
    selected_row = None
    table_kwargs = table_kwargs or {}
    visible_rows = min(10, console.height - 4)
    scroll_offset = 0
    total_options = len(data) + (1 if opt_out_message else 0)

    def build_table() -> Table:
        nonlocal scroll_offset
        table = Table(**table_kwargs)
        table.add_column()
        for column in columns:
            table.add_column(column.get('header', ''))
        if current_idx < scroll_offset:
            scroll_offset = current_idx
        elif current_idx >= scroll_offset + visible_rows:
            scroll_offset = current_idx - visible_rows + 1
        for i, item in enumerate(data[scroll_offset:scroll_offset + visible_rows]):
            row = [item.get(column.get('key', '')) for column in columns]
            if i + scroll_offset == current_idx:
                table.add_row('[bold][blue]>', *[f'[bold][blue]{cell}[/]' for cell in row])
            else:
                table.add_row('  ', *row)
        if opt_out_message:
            opt_out_row = [''] * (len(columns) - 1) + [opt_out_message]
            if current_idx == len(data):
                table.add_row('[bold][blue]>', *[f'[bold][blue]{cell}[/]' for cell in opt_out_row])
            else:
                table.add_row('  ', *opt_out_row)
        return table
    with Live(build_table(), auto_refresh=False, console=console) as live:
        instructions_message = f'[bold][green]?[/] {prompt} [bright_blue][Use arrows to move; enter to select'
        if opt_out_message:
            instructions_message += '; n to select none'
        instructions_message += ']'
        live.console.print(instructions_message)
        while selected_row is None:
            key = readchar.readkey()
            if key == readchar.key.UP:
                current_idx = (current_idx - 1) % total_options
            elif key == readchar.key.DOWN:
                current_idx = (current_idx + 1) % total_options
            elif key == readchar.key.CTRL_C:
                exit_with_error('')
            elif key == readchar.key.ENTER or key == readchar.key.CR:
                if current_idx == len(data):
                    return opt_out_response
                else:
                    selected_row = data[current_idx]
            elif key == 'n' and opt_out_message:
                return opt_out_response
            live.update(build_table(), refresh=True)
        return selected_row

class IntervalValuePrompt(PromptBase[timedelta]):
    response_type = timedelta
    validate_error_message = '[prompt.invalid]Please enter a valid interval denoted in seconds'

    def process_response(self, value: str) -> timedelta:
        try:
            int_value = int(value)
            if int_value <= 0:
                raise InvalidResponse('[prompt.invalid]Interval must be greater than 0')
            return timedelta(seconds=int_value)
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

def prompt_interval_schedule(console: Console) -> IntervalSchedule:
    """
    Prompt the user for an interval in seconds.
    """
    default_seconds = 3600
    default_duration = timedelta(seconds=default_seconds)
    interval = IntervalValuePrompt.ask(f'[bold][green]?[/] Seconds between scheduled runs ({default_seconds})', console=console, default=default_duration, show_default=False)
    return IntervalSchedule(interval=interval)

class CronStringPrompt(PromptBase[str]):
    response_type = str
    validate_error_message = '[prompt.invalid]Please enter a valid cron string'

    def process_response(self, value: str) -> str:
        try:
            CronSchedule.valid_cron_string(value)
            return value
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

class CronTimezonePrompt(PromptBase[str]):
    response_type = str
    validate_error_message = '[prompt.invalid]Please enter a valid timezone.'

    def process_response(self, value: str) -> str:
        try:
            CronSchedule.valid_timezone(value)
            return value
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

def prompt_cron_schedule(console: Console) -> CronSchedule:
    """
    Prompt the user for a cron string and timezone.
    """
    cron = CronStringPrompt.ask('[bold][green]?[/] Cron string', console=console, default='0 0 * * *')
    timezone = CronTimezonePrompt.ask('[bold][green]?[/] Timezone', console=console, default='UTC')
    return CronSchedule(cron=cron, timezone=timezone)

class RRuleStringPrompt(PromptBase[str]):
    response_type = str
    validate_error_message = '[prompt.invalid]Please enter a valid RRule string'

    def process_response(self, value: str) -> str:
        try:
            RRuleSchedule.validate_rrule_str(value)
            return value
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

class RRuleTimezonePrompt(PromptBase[str]):
    response_type = str
    validate_error_message = '[prompt.invalid]Please enter a valid timezone.'

    def process_response(self, value: str) -> str:
        try:
            is_valid_timezone(value)
            return value
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

def prompt_rrule_schedule(console: Console) -> RRuleSchedule:
    """
    Prompts the user to enter an RRule string and timezone.
    """
    rrule = RRuleStringPrompt.ask('[bold][green]?[/] RRule string', console=console, default='RRULE:FREQ=DAILY;INTERVAL=1')
    timezone = CronTimezonePrompt.ask('[bold][green]?[/] Timezone', console=console, default='UTC')
    return RRuleSchedule(rrule=rrule, timezone=timezone)

def prompt_schedule_type(console: Console) -> str:
    """
    Prompts the user to select a schedule type from a list of options.
    """
    selection = prompt_select_from_table(console, 'What type of schedule would you like to use?', [{'header': 'Schedule Type', 'key': 'type'}, {'header': 'Description', 'key': 'description'}], [{'type': 'Interval', 'description': 'Allows you to set flow runs to be executed at fixed time intervals.'}, {'type': 'Cron', 'description': 'Allows you to define recurring flow runs based on a specified pattern using cron syntax.'}, {'type': 'RRule', 'description': 'Allows you to define recurring flow runs using RFC 2445 recurrence rules.'}])
    return selection['type']

def prompt_schedules(console: Console) -> List[DeploymentScheduleCreate]:
    """
    Prompt the user to configure schedules for a deployment.
    """
    schedules: List[DeploymentScheduleCreate] = []
    if confirm('Would you like to configure schedules for this deployment?', default=True):
        add_schedule = True
        while add_schedule:
            schedule_type = prompt_schedule_type(console)
            if schedule_type == 'Cron':
                schedule = prompt_cron_schedule(console)
            elif schedule_type == 'Interval':
                schedule = prompt_interval_schedule(console)
            elif schedule_type == 'RRule':
                schedule = prompt_rrule_schedule(console)
            else:
                raise Exception('Invalid schedule type')
            is_schedule_active = confirm('Would you like to activate this schedule?', default=True)
            schedules.append(DeploymentScheduleCreate(schedule=schedule, active=is_schedule_active))
            add_schedule = confirm('Would you like to add another schedule?', default=False)
    return schedules

@client_injector
async def prompt_select_work_pool(client: PrefectClient, console: Console, prompt: str = 'Which work pool would you like to deploy this flow to?') -> str:
    work_pools = await client.read_work_pools()
    work_pool_options = [work_pool.model_dump() for work_pool in work_pools if work_pool.type != 'prefect-agent']
    if not work_pool_options:
        work_pool = await prompt_create_work_pool(console)
        return work_pool.name
    else:
        selected_work_pool_row = prompt_select_from_table(console, prompt, [{'header': 'Work Pool Name', 'key': 'name'}, {'header': 'Infrastructure Type', 'key': 'type'}, {'header': 'Description', 'key': 'description'}], work_pool_options)
        return selected_work_pool_row['name']

async def prompt_build_custom_docker_image(console: Console, deployment_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not confirm('Would you like to build a custom Docker image for this deployment?', console=console, default=False):
        return None
    build_step: Dict[str, Any] = {'requires': 'prefect-docker>=0.3.1', 'id': 'build-image'}
    if os.path.exists('Dockerfile'):
        if confirm('Would you like to use the Dockerfile in the current directory?', console=console, default=True):
            build_step['dockerfile'] = 'Dockerfile'
        elif confirm("A Dockerfile exists. You chose not to use it. A temporary Dockerfile will be automatically built during the deployment build step. If another file named 'Dockerfile' already exists at that time, the build step will fail. Would you like to rename your existing Dockerfile?"):
            new_dockerfile_name = prompt('New Dockerfile name', default='Dockerfile.backup')
            shutil.move('Docker