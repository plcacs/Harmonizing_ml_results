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
from typing import Dict, Any, Optional, List, Union, cast
from chalice.app import Chalice
from chalice.awsclient import TypedAWSClient, ReadTimeout
from chalice.cli.factory import CLIFactory
from chalice.config import Config
from chalice.local import LocalDevServer
from chalice.logs import LogRetrieveOptions
from chalice.utils import UI
from chalice.constants import DEFAULT_STAGE_NAME, DEFAULT_HANDLER_NAME
from chalice.invoke import UnhandledLambdaError
from chalice.deploy.planner import PlanEncoder
from chalice.deploy.appgraph import ApplicationGraphBuilder, GraphPrettyPrint
from chalice.cli import newproj

def _configure_logging(level: int, format_string: Optional[str] = None) -> None: ...

def get_system_info() -> str: ...

@click.group()
@click.version_option(version: str, message: str)
@click.option('--project-dir', help: str)
@click.option('--debug/--no-debug', default: bool, help: str)
@click.pass_context
def cli(ctx: click.Context, project_dir: str, debug: bool = False) -> None: ...

def _configure_cli_env_vars() -> None: ...

@cli.command()
@click.option('--host', default: str)
@click.option('--port', default: int, type: click.INT)
@click.option('--stage', default: str, help: str)
@click.option('--autoreload/--no-autoreload', default: bool, help: str)
@click.pass_context
def local(ctx: click.Context, host: str = '127.0.0.1', port: int = 8000, stage: str = DEFAULT_STAGE_NAME, autoreload: bool = True) -> None: ...

def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...

def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...

@cli.command()
@click.option('--autogen-policy/--no-autogen-policy', default: bool, help: str)
@click.option('--profile', help: str)
@click.option('--api-gateway-stage', help: str)
@click.option('--stage', default: str, help: str)
@click.option('--connection-timeout', type: int, help: str)
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME, connection_timeout: Optional[int] = None) -> None: ...

@cli.group()
def dev() -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default: bool, help: str)
@click.option('--profile', help: str)
@click.option('--api-gateway-stage', help: str)
@click.option('--stage', default: str, help: str)
@click.pass_context
def plan(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default: bool, help: str)
@click.option('--profile', help: str)
@click.option('--api-gateway-stage', help: str)
@click.option('--stage', default: str, help: str)
@click.pass_context
def appgraph(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('invoke')
@click.option('-n', '--name', metavar: str, required: bool, help: str)
@click.option('--profile', metavar: str, help: str)
@click.option('--stage', metavar: str, default: str, help: str)
@click.pass_context
def invoke(ctx: click.Context, name: str, profile: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('delete')
@click.option('--profile', help: str)
@click.option('--stage', default: str, help: str)
@click.pass_context
def delete(ctx: click.Context, profile: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command()
@click.option('--num-entries', default: Optional[int] = None, type: int, help: str)
@click.option('--include-lambda-messages/--no-include-lambda-messages', default: bool, help: str)
@click.option('--stage', default: str, help: str)
@click.option('-n', '--name', help: str, default: str)
@click.option('-s', '--since', help: str, default: Optional[str] = None)
@click.option('-f', '--follow/--no-follow', default: bool, help: str)
@click.option('--profile', help: str)
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int] = None, include_lambda_messages: bool = False, stage: str = DEFAULT_STAGE_NAME, name: str = DEFAULT_HANDLER_NAME, since: Optional[str] = None, follow: bool = False, profile: Optional[str] = None) -> None: ...

@cli.command('gen-policy')
@click.option('--filename', help: str)
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str] = None) -> None: ...

@cli.command('new-project')
@click.argument('project_name', required: bool = False)
@click.option('--profile', required: bool = False)
@click.option('-t', '--project-type', required: bool = False, default: str, help: str)
@click.pass_context
def new_project(ctx: click.Context, project_name: Optional[str] = None, profile: Optional[str] = None, project_type: str = 'legacy') -> None: ...

@cli.command('url')
@click.option('--stage', default: str, help: str)
@click.pass_context
def url(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('generate-sdk')
@click.option('--sdk-type', default: str, type: click.Choice(['javascript']))
@click.option('--stage', default: str, help: str)
@click.argument('outdir')
@click.pass_context
def generate_sdk(ctx: click.Context, sdk_type: str = 'javascript', stage: str = DEFAULT_STAGE_NAME, outdir: str) -> None: ...

@cli.command('generate-models')
@click.option('--stage', default: str, help: str)
@click.pass_context
def generate_models(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('package')
@click.option('--pkg-format', default: str, help: str, type: click.Choice(['cloudformation', 'terraform']))
@click.option('--stage', default: str, help: str)
@click.option('--single-file', is_flag: bool, default: bool, help: str)
@click.option('--merge-template', help: str)
@click.option('--template-format', default: str, type: click.Choice(['json', 'yaml']), case_sensitive: bool = False, help: str)
@click.option('--profile', help: str)
@click.argument('out')
@click.pass_context
def package(ctx: click.Context, single_file: bool, stage: str = DEFAULT_STAGE_NAME, merge_template: Optional[str] = None, out: str, pkg_format: str = 'cloudformation', template_format: str = 'json', profile: Optional[str] = None) -> None: ...

@cli.command('generate-pipeline')
@click.option('--pipeline-version', default: str, type: click.Choice(['v1', 'v2']), help: str)
@click.option('-i', '--codebuild-image', help: str)
@click.option('-s', '--source', default: str, type: click.Choice(['codecommit', 'github']), help: str)
@click.option('-b', '--buildspec-file', help: str)
@click.argument('filename')
@click.pass_context
def generate_pipeline(ctx: click.Context, pipeline_version: str = 'v1', codebuild_image: Optional[str] = None, source: str = 'codecommit', buildspec_file: Optional[str] = None, filename: str) -> None: ...

def main() -> int: ...