"""Command line interface for chalice."""

from __future__ import annotations
import logging
import os
import sys
import tempfile
import shutil
import traceback
import functools
import json
import botocore.exceptions
import click
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
    Any,
    Callable,
    Iterable,
    Tuple,
    Type,
    TypeVar,
    overload,
)
from chalice import __version__
from chalice.app import Chalice
from chalice.awsclient import TypedAWSClient, ReadTimeout
from chalice.cli.factory import CLIFactory, NoSuchFunctionError
from chalice.config import Config
from chalice.logs import LogRetrieveOptions
from chalice.local import LocalDevServer
from chalice.invoke import UnhandledLambdaError
from chalice.deploy.planner import PlanEncoder
from chalice.deploy.appgraph import ApplicationGraphBuilder, GraphPrettyPrint
from chalice.utils import UI
from chalice.constants import DEFAULT_STAGE_NAME, DEFAULT_HANDLER_NAME

def _configure_logging(level: Union[int, str], format_string: Optional[str] = None) -> None:
    ...

def get_system_info() -> str:
    ...

@overload
def cli(ctx: click.Context, project_dir: str, debug: bool) -> click.Group:
    ...

@overload
def cli(ctx: click.Context, project_dir: Optional[str] = None, debug: bool = False) -> click.Group:
    ...

def cli(ctx: click.Context, project_dir: Optional[str] = None, debug: bool = False) -> click.Group:
    ...

def _configure_cli_env_vars() -> None:
    ...

def local(
    ctx: click.Context,
    host: str = '127.0.0.1',
    port: int = 8000,
    stage: str = DEFAULT_STAGE_NAME,
    autoreload: bool = True,
) -> None:
    ...

def create_local_server(
    factory: CLIFactory,
    host: str,
    port: int,
    stage: str,
) -> LocalDevServer:
    ...

def run_local_server(
    factory: CLIFactory,
    host: str,
    port: int,
    stage: str,
) -> None:
    ...

def deploy(
    ctx: click.Context,
    autogen_policy: Optional[bool] = None,
    profile: Optional[str] = None,
    api_gateway_stage: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
    connection_timeout: Optional[int] = None,
) -> None:
    ...

@overload
def plan(
    ctx: click.Context,
    autogen_policy: Optional[bool] = None,
    profile: Optional[str] = None,
    api_gateway_stage: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

@overload
def plan(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
) -> None:
    ...

def plan(
    ctx: click.Context,
    autogen_policy: Optional[bool] = None,
    profile: Optional[str] = None,
    api_gateway_stage: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def appgraph(
    ctx: click.Context,
    autogen_policy: Optional[bool] = None,
    profile: Optional[str] = None,
    api_gateway_stage: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def invoke(
    ctx: click.Context,
    name: str,
    profile: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def delete(
    ctx: click.Context,
    profile: Optional[str] = None,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def logs(
    ctx: click.Context,
    num_entries: Optional[int] = None,
    include_lambda_messages: bool = False,
    stage: str = DEFAULT_STAGE_NAME,
    name: str = DEFAULT_HANDLER_NAME,
    since: Optional[str] = None,
    follow: bool = False,
    profile: Optional[str] = None,
) -> None:
    ...

def gen_policy(
    ctx: click.Context,
    filename: Optional[str] = None,
) -> None:
    ...

def new_project(
    ctx: click.Context,
    project_name: Optional[str] = None,
    profile: Optional[str] = None,
    project_type: str = 'legacy',
) -> None:
    ...

def url(
    ctx: click.Context,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def generate_sdk(
    ctx: click.Context,
    sdk_type: str = 'javascript',
    stage: str = DEFAULT_STAGE_NAME,
    outdir: str,
) -> None:
    ...

def generate_models(
    ctx: click.Context,
    stage: str = DEFAULT_STAGE_NAME,
) -> None:
    ...

def package(
    ctx: click.Context,
    single_file: bool = False,
    stage: str = DEFAULT_STAGE_NAME,
    merge_template: Optional[str] = None,
    out: str,
    pkg_format: str = 'cloudformation',
    template_format: str = 'json',
    profile: Optional[str] = None,
) -> None:
    ...

def generate_pipeline(
    ctx: click.Context,
    pipeline_version: str = 'v1',
    codebuild_image: Optional[str] = None,
    source: str = 'codecommit',
    buildspec_file: Optional[str] = None,
    filename: str,
) -> None:
    ...

def main() -> int:
    ...