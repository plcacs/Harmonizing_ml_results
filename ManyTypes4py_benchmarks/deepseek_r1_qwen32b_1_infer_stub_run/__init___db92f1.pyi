"""Command line interface for chalice."""

from __future__ import annotations
import logging
import os
import platform
import sys
import tempfile
import shutil
import traceback
import functools
import json
import botocore.exceptions
import click
from typing import Dict, Any, Optional, cast, List, Tuple, Union
from chalice import __version__ as chalice_version
from chalice.app import Chalice
from chalice.awsclient import TypedAWSClient
from chalice.awsclient import ReadTimeout
from chalice.cli.factory import CLIFactory
from chalice.cli.factory import NoSuchFunctionError
from chalice.config import Config
from chalice.logs import display_logs, LogRetrieveOptions
from chalice.utils import create_zip_file
from chalice.deploy.validate import validate_routes, validate_python_version
from chalice.deploy.validate import ExperimentalFeatureError
from chalice.utils import UI, serialize_to_json
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.local import LocalDevServer
from chalice.constants import DEFAULT_HANDLER_NAME
from chalice.invoke import UnhandledLambdaError
from chalice.deploy.swagger import TemplatedSwaggerGenerator
from chalice.deploy.planner import PlanEncoder
from chalice.deploy.appgraph import ApplicationGraphBuilder, GraphPrettyPrint
from chalice.cli import newproj

def _configure_logging(level: int, format_string: Optional[str] = None) -> None: ...

def get_system_info() -> str: ...

def cli(ctx: click.Context, project_dir: Optional[str], debug: bool) -> click.Group: ...

def _configure_cli_env_vars() -> None: ...

def local(ctx: click.Context, host: str = '127.0.0.1', port: int = 8000, stage: str = DEFAULT_STAGE_NAME, autoreload: bool = True) -> None: ...

def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...

def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...

def deploy(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str = DEFAULT_STAGE_NAME, connection_timeout: Optional[int]) -> None: ...

class dev(click.Group):
    def plan(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str = DEFAULT_STAGE_NAME) -> None: ...
    def appgraph(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str = DEFAULT_STAGE_NAME) -> None: ...

def invoke(ctx: click.Context, name: str, profile: Optional[str], stage: str = DEFAULT_STAGE_NAME) -> None: ...

def delete(ctx: click.Context, profile: Optional[str], stage: str = DEFAULT_STAGE_NAME) -> None: ...

def logs(ctx: click.Context, num_entries: Optional[int], include_lambda_messages: bool, stage: str = DEFAULT_STAGE_NAME, name: str = DEFAULT_HANDLER_NAME, since: Optional[str], follow: bool, profile: Optional[str]) -> None: ...

def gen_policy(ctx: click.Context, filename: Optional[str]) -> None: ...

def new_project(ctx: click.Context, project_name: Optional[str], profile: Optional[str], project_type: str = 'legacy') -> None: ...

def url(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

def generate_sdk(ctx: click.Context, sdk_type: str, stage: str = DEFAULT_STAGE_NAME, outdir: str) -> None: ...

def generate_models(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

def package(ctx: click.Context, single_file: bool, stage: str = DEFAULT_STAGE_NAME, merge_template: Optional[str], out: str, pkg_format: str, template_format: str, profile: Optional[str]) -> None: ...

def generate_pipeline(ctx: click.Context, pipeline_version: str, codebuild_image: Optional[str], source: str, buildspec_file: Optional[str], filename: str) -> None: ...

def main() -> int: ...