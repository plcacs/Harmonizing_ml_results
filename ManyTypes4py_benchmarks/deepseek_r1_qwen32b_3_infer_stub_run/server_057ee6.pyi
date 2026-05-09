"""
Command line interface for working with the Prefect API and server.
"""

from __future__ import annotations
from pathlib import Path
from subprocess import Popen
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import asyncio
import signal
import subprocess
import typer
import uvicorn
from prefect.settings import (
    SettingsVar,
)
from prefect.server.services.base import Service

__all__: List[str] = [
    'generate_welcome_blurb',
    'prestart_check',
    'start',
    'stop',
    'reset',
    'upgrade',
    'downgrade',
    'revision',
    'stamp',
    'list_services',
    'start_services',
    'stop_services',
]

PREFECT_API_SERVICES_LATE_RUNS_ENABLED: SettingsVar[bool]
PREFECT_API_SERVICES_SCHEDULER_ENABLED: SettingsVar[bool]
PREFECT_API_URL: SettingsVar[str]
PREFECT_HOME: SettingsVar[Path]
PREFECT_SERVER_ANALYTICS_ENABLED: SettingsVar[bool]
PREFECT_SERVER_API_HOST: SettingsVar[str]
PREFECT_SERVER_API_KEEPALIVE_TIMEOUT: SettingsVar[int]
PREFECT_SERVER_API_PORT: SettingsVar[int]
PREFECT_SERVER_LOGGING_LEVEL: SettingsVar[str]
PREFECT_UI_ENABLED: SettingsVar[bool]

def generate_welcome_blurb(base_url: str, ui_enabled: bool) -> str:
    ...

def prestart_check(base_url: str) -> None:
    ...

@server_app.command()
def start(
    host: str = ...,
    port: int = ...,
    keep_alive_timeout: int = ...,
    log_level: str = ...,
    scheduler: bool = ...,
    analytics: bool = ...,
    late_runs: bool = ...,
    ui: bool = ...,
    no_services: bool = ...,
    background: bool = ...,
) -> None:
    ...

def _run_in_background(
    pid_file: Path,
    server_settings: Dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None:
    ...

def _run_in_foreground(
    server_settings: Dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None:
    ...

@server_app.command()
async def stop() -> None:
    ...

@database_app.command()
async def reset(yes: bool = ...) -> None:
    ...

@database_app.command()
async def upgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...,
) -> None:
    ...

@database_app.command()
async def downgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...,
) -> None:
    ...

@database_app.command()
async def revision(
    message: Optional[str] = ...,
    autogenerate: bool = ...,
) -> None:
    ...

@database_app.command()
async def stamp(revision: str) -> None:
    ...

def _is_process_running(pid: int) -> bool:
    ...

def _read_pid_file(path: Path) -> Optional[int]:
    ...

def _write_pid_file(path: Path, pid: int) -> None:
    ...

def _cleanup_pid_file(path: Path) -> None:
    ...

@services_app.command(hidden=True, name='manager')
def run_manager_process() -> None:
    ...

@services_app.command(aliases=['ls'])
def list_services() -> None:
    ...

@services_app.command(aliases=['start'])
def start_services(
    background: bool = ...,
) -> None:
    ...

@services_app.command(aliases=['stop'])
async def stop_services() -> None:
    ...