"""
Stub file for 'server_057ee6' module
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Literal, Union
from pathlib import Path
import typer
from prefect.cli._types import PrefectTyper

if TYPE_CHECKING:
    from prefect.settings import (
        PREFECT_API_SERVICES_SCHEDULER_ENABLED,
        PREFECT_SERVER_API_HOST,
        PREFECT_SERVER_API_PORT,
        PREFECT_SERVER_API_KEEPALIVE_TIMEOUT,
        PREFECT_SERVER_LOGGING_LEVEL,
        PREFECT_API_SERVICES_LATE_RUNS_ENABLED,
        PREFECT_UI_ENABLED,
    )

server_app: PrefectTyper = ...
database_app: PrefectTyper = ...
services_app: PrefectTyper = ...

def prestart_check(base_url: str) -> None: ...

@server_app.command()
def start(
    host: str = PREFECT_SERVER_API_HOST,
    port: int = PREFECT_SERVER_API_PORT,
    keep_alive_timeout: Optional[int] = PREFECT_SERVER_API_KEEPALIVE_TIMEOUT,
    log_level: Optional[str] = PREFECT_SERVER_LOGGING_LEVEL,
    scheduler: bool = PREFECT_API_SERVICES_SCHEDULER_ENABLED,
    analytics: bool = PREFECT_SERVER_ANALYTICS_ENABLED,
    late_runs: bool = PREFECT_API_SERVICES_LATE_RUNS_ENABLED,
    ui: bool = PREFECT_UI_ENABLED,
    no_services: bool = typer.Option(False, '--no-services'),
    background: bool = typer.Option(False, '--background', '-b'),
) -> None: ...

def _run_in_background(
    pid_file: Path,
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None: ...

def _run_in_foreground(
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool,
) -> None: ...

@server_app.command()
async def stop() -> None: ...

@database_app.command()
async def reset(yes: bool = typer.Option(False, '--yes', '-y')) -> None: ...

@database_app.command()
async def upgrade(
    yes: bool = typer.Option(False, '--yes', '-y'),
    revision: str = typer.Option('head', '-r'),
    dry_run: bool = typer.Option(False),
) -> None: ...

@database_app.command()
async def downgrade(
    yes: bool = typer.Option(False, '--yes', '-y'),
    revision: str = typer.Option('-1', '-r'),
    dry_run: bool = typer.Option(False),
) -> None: ...

@database_app.command()
async def revision(
    message: Optional[str] = typer.Option(None, '--message', '-m'),
    autogenerate: bool = typer.Option(False),
) -> None: ...

@database_app.command()
async def stamp(revision: str) -> None: ...

def _is_process_running(pid: int) -> bool: ...
def _read_pid_file(path: Path) -> Optional[int]: ...
def _write_pid_file(path: Path, pid: int) -> None: ...
def _cleanup_pid_file(path: Path) -> None: ...

@services_app.command(hidden=True, name='manager')
def run_manager_process() -> None: ...

@services_app.command(aliases=['ls'])
def list_services() -> None: ...

@services_app.command(aliases=['start'])
def start_services(
    background: bool = typer.Option(False, '--background', '-b'),
) -> None: ...

@services_app.command(aliases=['stop'])
async def stop_services() -> None: ...