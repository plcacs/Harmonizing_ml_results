from __future__ import annotations

import logging
from typing import Any, Optional

import click

from chalice.cli.factory import CLIFactory
from chalice.local import LocalDevServer


def _configure_logging(level: int, format_string: Optional[str] = None) -> None: ...

def get_system_info() -> str: ...

@click.group()
@click.version_option(version=..., message=...)
@click.option('--project-dir', help=...)
@click.option('--debug/--no-debug', default=False, help=...)
@click.pass_context
def cli(ctx: click.Context, project_dir: Optional[str], debug: bool = False) -> None: ...

def _configure_cli_env_vars() -> None: ...

@cli.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8000, type=click.INT)
@click.option('--stage', default=...)
@click.option('--autoreload/--no-autoreload', default=True, help=...)
@click.pass_context
def local(ctx: click.Context, host: str = '127.0.0.1', port: int = 8000, stage: str = ..., autoreload: bool = True) -> None: ...

def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...

def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...

@cli.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help=...)
@click.option('--profile', help=...)
@click.option('--api-gateway-stage', help=...)
@click.option('--stage', default=..., help=...)
@click.option('--connection-timeout', type=int, help=...)
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str, connection_timeout: Optional[int]) -> None: ...

@cli.group()
def dev() -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help=...)
@click.option('--profile', help=...)
@click.option('--api-gateway-stage', help=...)
@click.option('--stage', default=..., help=...)
@click.pass_context
def plan(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str) -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help=...)
@click.option('--profile', help=...)
@click.option('--api-gateway-stage', help=...)
@click.option('--stage', default=..., help=...)
@click.pass_context
def appgraph(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str) -> None: ...

@cli.command('invoke')
@click.option('-n', '--name', metavar='NAME', required=True, help=...)
@click.option('--profile', metavar='PROFILE', help=...)
@click.option('--stage', metavar='STAGE', default=..., help=...)
@click.pass_context
def invoke(ctx: click.Context, name: str, profile: Optional[str], stage: str) -> None: ...

@cli.command('delete')
@click.option('--profile', help=...)
@click.option('--stage', default=..., help=...)
@click.pass_context
def delete(ctx: click.Context, profile: Optional[str], stage: str) -> None: ...

@cli.command()
@click.option('--num-entries', default=None, type=int, help=...)
@click.option('--include-lambda-messages/--no-include-lambda-messages', default=False, help=...)
@click.option('--stage', default=..., help=...)
@click.option('-n', '--name', help=..., default=...)
@click.option('-s', '--since', help=..., default=None)
@click.option('-f', '--follow/--no-follow', default=False, help=...)
@click.option('--profile', help=...)
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int], include_lambda_messages: bool, stage: str, name: str, since: Optional[str], follow: bool, profile: Optional[str]) -> None: ...

@cli.command('gen-policy')
@click.option('--filename', help=...)
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None: ...

@cli.command('new-project')
@click.argument('project_name', required=False)
@click.option('--profile', required=False)
@click.option('-t', '--project-type', required=False, default='legacy')
@click.pass_context
def new_project(ctx: click.Context, project_name: Optional[str], profile: Optional[str], project_type: str) -> None: ...

@cli.command('url')
@click.option('--stage', default=..., help=...)
@click.pass_context
def url(ctx: click.Context, stage: str) -> None: ...

@cli.command('generate-sdk')
@click.option('--sdk-type', default='javascript', type=click.Choice(['javascript']))
@click.option('--stage', default=..., help=...)
@click.argument('outdir')
@click.pass_context
def generate_sdk(ctx: click.Context, sdk_type: str, stage: str, outdir: str) -> None: ...

@cli.command('generate-models')
@click.option('--stage', default=..., help=...)
@click.pass_context
def generate_models(ctx: click.Context, stage: str) -> None: ...

@cli.command('package')
@click.option('--pkg-format', default='cloudformation', help=..., type=click.Choice(['cloudformation', 'terraform']))
@click.option('--stage', default=..., help=...)
@click.option('--single-file', is_flag=True, default=False, help=...)
@click.option('--merge-template', help=...)
@click.option('--template-format', default='json', type=click.Choice(['json', 'yaml'], case_sensitive=False), help=...)
@click.option('--profile', help=...)
@click.argument('out')
@click.pass_context
def package(ctx: click.Context, single_file: bool, stage: str, merge_template: Optional[str], out: str, pkg_format: str, template_format: str, profile: Optional[str]) -> None: ...

@cli.command('generate-pipeline')
@click.option('--pipeline-version', default='v1', type=click.Choice(['v1', 'v2']), help=...)
@click.option('-i', '--codebuild-image', help=...)
@click.option('-s', '--source', default='codecommit', type=click.Choice(['codecommit', 'github']), help=...)
@click.option('-b', '--buildspec-file', help=...)
@click.argument('filename')
@click.pass_context
def generate_pipeline(ctx: click.Context, pipeline_version: str, codebuild_image: Optional[str], source: str, buildspec_file: Optional[str], filename: str) -> None: ...

def main() -> Optional[int]: ...