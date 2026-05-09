from __future__ import annotations
import asyncio
import inspect
import os
import shlex
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING
import typer
import uvicorn
from rich.table import Table
from rich.text import Text
import prefect
import prefect.settings
from prefect.cli._prompts import prompt
from prefect.cli._types import PrefectTyper, SettingsOption
from prefect.cli._utilities import exit_with_error, exit_with_success
from prefect.cli.cloud import prompt_select_from_list
from prefect.cli.root import app, is_interactive
from prefect.logging import get_logger
from prefect.server.services.base import Service
from prefect.settings import PREFECT_API_SERVICES_LATE_RUNS_ENABLED, PREFECT_API_SERVICES_SCHEDULER_ENABLED, PREFECT_API_URL, PREFECT_HOME, PREFECT_SERVER_ANALYTICS_ENABLED, PREFECT_SERVER_API_HOST, PREFECT_SERVER_API_KEEPALIVE_TIMEOUT, PREFECT_SERVER_LOGGING_LEVEL, PREFECT_UI_ENABLED, Profile, load_current_profile, load_profiles, save_profiles, update_current_profile
from prefect.settings.context import temporary_settings
from prefect.utilities.asyncutils import run_sync_in_worker_thread

if TYPE_CHECKING:
    import logging

server_app: PrefectTyper = PrefectTyper(name='server', help='Start a Prefect server instance and interact with the database')
database_app: PrefectTyper = PrefectTyper(name='database', help='Interact with the database.')
services_app: PrefectTyper = PrefectTyper(name='services', help='Interact with server loop services.')

def generate_welcome_blurb(base_url: str, ui_enabled: bool) -> str:
    # ...

def prestart_check(base_url: str) -> None:
    # ...

@server_app.command()
async def start(
    host: SettingsOption(PREFECT_SERVER_API_HOST),
    port: SettingsOption(PREFECT_SERVER_API_PORT),
    keep_alive_timeout: SettingsOption(PREFECT_SERVER_API_KEEPALIVE_TIMEOUT),
    log_level: SettingsOption(PREFECT_SERVER_LOGGING_LEVEL),
    scheduler: SettingsOption(PREFECT_API_SERVICES_SCHEDULER_ENABLED),
    analytics: SettingsOption(PREFECT_SERVER_ANALYTICS_ENABLED, '--analytics-on/--analytics-off'),
    late_runs: SettingsOption(PREFECT_API_SERVICES_LATE_RUNS_ENABLED),
    ui: SettingsOption(PREFECT_UI_ENABLED),
    no_services: typer.Option(False, '--no-services', help='Only run the webserver API and UI'),
    background: typer.Option(False, '--background', '-b', help='Run the server in the background')
) -> None:
    # ...

def _run_in_background(
    pid_file: Path,
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool
) -> None:
    # ...

def _run_in_foreground(
    server_settings: dict[str, str],
    host: str,
    port: int,
    keep_alive_timeout: int,
    no_services: bool
) -> None:
    # ...

@server_app.command()
async def stop() -> None:
    # ...

@database_app.command()
async def reset(yes: bool = typer.Option(False, '--yes', '-y')) -> None:
    # ...

@database_app.command()
async def upgrade(
    yes: bool = typer.Option(False, '--yes', '-y'),
    revision: str = typer.Option('head', '-r', help='The revision to pass to `alembic upgrade`. If not provided, runs all migrations.'),
    dry_run: bool = typer.Option(False, help='Flag to show what migrations would be made without applying them. Will emit sql statements to stdout.')
) -> None:
    # ...

@database_app.command()
async def downgrade(
    yes: bool = typer.Option(False, '--yes', '-y'),
    revision: str = typer.Option('-1', '-r', help="The revision to pass to `alembic downgrade`. If not provided, downgrades to the most recent revision. Use 'base' to run all migrations."),
    dry_run: bool = typer.Option(False, help='Flag to show what migrations would be made without applying them. Will emit sql statements to stdout.')
) -> None:
    # ...

@database_app.command()
async def revision(
    message: str | None = typer.Option(None, '--message', '-m', help='A message to describe the migration.'),
    autogenerate: bool = False
) -> None:
    # ...

@database_app.command(aliases=['ls'])
def list_services() -> None:
    # ...

@services_app.command(aliases=['start'])
def start_services(background: bool = typer.Option(False, '--background', '-b', help='Run the services in the background')) -> None:
    # ...

@services_app.command(aliases=['stop'])
async def stop_services() -> None:
    # ...
