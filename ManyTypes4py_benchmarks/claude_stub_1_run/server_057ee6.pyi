```pyi
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import typer
import uvicorn
from rich.table import Table
from rich.text import Text

import prefect
import prefect.settings
from prefect.cli._types import PrefectTyper, SettingsOption
from prefect.settings import (
    PREFECT_API_SERVICES_LATE_RUNS_ENABLED,
    PREFECT_API_SERVICES_SCHEDULER_ENABLED,
    PREFECT_API_URL,
    PREFECT_HOME,
    PREFECT_SERVER_ANALYTICS_ENABLED,
    PREFECT_SERVER_API_HOST,
    PREFECT_SERVER_API_KEEPALIVE_TIMEOUT,
    PREFECT_SERVER_API_PORT,
    PREFECT_SERVER_LOGGING_LEVEL,
    PREFECT_UI_ENABLED,
    Profile,
)

server_app: PrefectTyper
database_app: PrefectTyper
services_app: PrefectTyper
logger: logging.Logger
SERVER_PID_FILE_NAME: str
SERVICES_PID_FILE: Path

def generate_welcome_blurb(base_url: str, ui_enabled: bool) -> str: ...
def prestart_check(base_url: str) -> None: ...
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
async def stop() -> None: ...
async def reset(yes: bool = ...) -> None: ...
async def upgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...,
) -> None: ...
async def downgrade(
    yes: bool = ...,
    revision: str = ...,
    dry_run: bool = ...,
) -> None: ...
async def revision(
    message: str | None = ...,
    autogenerate: bool = ...,
) -> None: ...
async def stamp(revision: str) -> None: ...
def _is_process_running(pid: int) -> bool: ...
def _read_pid_file(path: Path) -> int | None: ...
def _write_pid_file(path: Path, pid: int) -> None: ...
def _cleanup_pid_file(path: Path) -> None: ...
def run_manager_process() -> None: ...
def list_services() -> None: ...
def start_services(background: bool = ...) -> None: ...
async def stop_services() -> None: ...
```