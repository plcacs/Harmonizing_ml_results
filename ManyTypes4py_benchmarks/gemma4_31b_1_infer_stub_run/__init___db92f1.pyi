from __future__ import annotations
import logging
import os
import click
from typing import Any, Optional, overload
from chalice.awsclient import TypedAWSClient
from chalice.cli.factory import CLIFactory
from chalice.config import Config
from chalice.local import LocalDevServer

def _configure_logging(level: int, format_string: Optional[str] = None) -> None: ...

def get_system_info() -> str: ...

@click.group()
def cli(ctx: click.Context, project_dir: Optional[str], debug: bool = False) -> None: ...

def _configure_cli_env_vars() -> None: ...

@cli.command()
def local(
    ctx: click.Context,
    host: str = '127.0.0.1',
    port: int = 8000,
    stage: str = 'dev',
    autoreload: bool = True,
) -> None: ...

def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...

def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...

@cli.command()
def deploy(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
    connection_timeout: Optional[int],
) -> None: ...

@cli.group()
def dev() -> None: ...

@dev.command()
def plan(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
) -> None: ...

@dev.command()
def appgraph(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
) -> None: ...

@cli.command('invoke')
def invoke(ctx: click.Context, name: str, profile: Optional[str], stage: str) -> None: ...

@cli.command('delete')
def delete(ctx: click.Context, profile: Optional[str], stage: str) -> None: ...

@cli.command()
def logs(
    ctx: click.Context,
    num_entries: Optional[int],
    include_lambda_messages: bool,
    stage: str,
    name: str,
    since: Optional[str],
    follow: bool,
    profile: Optional[str],
) -> None: ...

@cli.command('gen-policy')
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None: ...

@cli.command('new-project')
def new_project(
    ctx: click.Context,
    project_name: Optional[str],
    profile: Optional[str],
    project_type: str,
) -> None: ...

@cli.command('url')
def url(ctx: click.Context, stage: str) -> None: ...

@cli.command('generate-sdk')
def generate_sdk(ctx: click.Context, sdk_type: str, stage: str, outdir: str) -> None: ...

@cli.command('generate-models')
def generate_models(ctx: click.Context, stage: str) -> None: ...

@cli.command('package')
def package(
    ctx: click.Context,
    single_file: bool,
    stage: str,
    merge_template: Optional[str],
    out: str,
    pkg_format: str,
    template_format: str,
    profile: Optional[str],
) -> None: ...

@cli.command('generate-pipeline')
def generate_pipeline(
    ctx: click.Context,
    pipeline_version: str,
    codebuild_image: Optional[str],
    source: str,
    buildspec_file: Optional[str],
    filename: str,
) -> None: ...

def main() -> Any: ...