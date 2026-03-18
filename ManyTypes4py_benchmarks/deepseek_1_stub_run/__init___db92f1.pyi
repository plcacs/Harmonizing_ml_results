```python
from __future__ import annotations

import logging
import sys
import click
from typing import Any, Optional, Dict

def _configure_logging(level: int, format_string: Optional[str] = ...) -> None: ...

def get_system_info() -> str: ...

@click.group()
@click.version_option(version: Any, message: str)
@click.option('--project-dir', help: Optional[str])
@click.option('--debug/--no-debug', default: bool)
@click.pass_context
def cli(ctx: click.Context, project_dir: Optional[str], debug: bool = ...) -> None: ...

def _configure_cli_env_vars() -> None: ...

@cli.command()
@click.option('--host', default: str)
@click.option('--port', default: int, type: Any)
@click.option('--stage', default: str, help: str)
@click.option('--autoreload/--no-autoreload', default: bool, help: str)
@click.pass_context
def local(ctx: click.Context, host: str = ..., port: int = ..., stage: str = ..., autoreload: bool = ...) -> None: ...

def create_local_server(factory: Any, host: str, port: int, stage: str) -> Any: ...

def run_local_server(factory: Any, host: str, port: int, stage: str) -> None: ...

@cli.command()
@click.option('--autogen-policy/--no-autogen-policy', default: Optional[bool], help: str)
@click.option('--profile', help: Optional[str])
@click.option('--api-gateway-stage', help: Optional[str])
@click.option('--stage', default: str, help: str)
@click.option('--connection-timeout', type: Optional[int], help: str)
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str, connection_timeout: Optional[int]) -> None: ...

@cli.group()
def dev() -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default: Optional[bool], help: str)
@click.option('--profile', help: Optional[str])
@click.option('--api-gateway-stage', help: Optional[str])
@click.option('--stage', default: str, help: str)
@click.pass_context
def plan(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str) -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default: Optional[bool], help: str)
@click.option('--profile', help: Optional[str])
@click.option('--api-gateway-stage', help: Optional[str])
@click.option('--stage', default: str, help: str)
@click.pass_context
def appgraph(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str) -> None: ...

@cli.command('invoke')
@click.option('-n', '--name', metavar: str, required: bool, help: str)
@click.option('--profile', metavar: str, help: Optional[str])
@click.option('--stage', metavar: str, default: str, help: str)
@click.pass_context
def invoke(ctx: click.Context, name: str, profile: Optional[str], stage: str) -> None: ...

@cli.command('delete')
@click.option('--profile', help: Optional[str])
@click.option('--stage', default: str, help: str)
@click.pass_context
def delete(ctx: click.Context, profile: Optional[str], stage: str) -> None: ...

@cli.command()
@click.option('--num-entries', default: Optional[int], type: Any, help: str)
@click.option('--include-lambda-messages/--no-include-lambda-messages', default: bool, help: str)
@click.option('--stage', default: str, help: str)
@click.option('-n', '--name', help: str, default: str)
@click.option('-s', '--since', help: str, default: Optional[str])
@click.option('-f', '--follow/--no-follow', default: bool, help: str)
@click.option('--profile', help: Optional[str])
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int], include_lambda_messages: bool, stage: str, name: str, since: Optional[str], follow: bool, profile: Optional[str]) -> None: ...

@cli.command('gen-policy')
@click.option('--filename', help: Optional[str])
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None: ...

@cli.command('new-project')
@click.argument('project_name', required: bool)
@click.option('--profile', required: bool)
@click.option('-t', '--project-type', required: bool, default: str)
@click.pass_context
def new_project(ctx: click.Context, project_name: Optional[str], profile: Optional[str], project_type: str) -> None: ...

@cli.command('url')
@click.option('--stage', default: str, help: str)
@click.pass_context
def url(ctx: click.Context, stage: str) -> None: ...

@cli.command('generate-sdk')
@click.option('--sdk-type', default: str, type: Any)
@click.option('--stage', default: str, help: str)
@click.argument('outdir')
@click.pass_context
def generate_sdk(ctx: click.Context, sdk_type: str, stage: str, outdir: str) -> None: ...

@cli.command('generate-models')
@click.option('--stage', default: str, help: str)
@click.pass_context
def generate_models(ctx: click.Context, stage: str) -> None: ...

@cli.command('package')
@click.option('--pkg-format', default: str, help: str, type: Any)
@click.option('--stage', default: str, help: str)
@click.option('--single-file', is_flag: bool, default: bool, help: str)
@click.option('--merge-template', help: Optional[str])
@click.option('--template-format', default: str, type: Any, help: str)
@click.option('--profile', help: Optional[str])
@click.argument('out')
@click.pass_context
def package(ctx: click.Context, single_file: bool, stage: str, merge_template: Optional[str], out: str, pkg_format: str, template_format: str, profile: Optional[str]) -> None: ...

@cli.command('generate-pipeline')
@click.option('--pipeline-version', default: str, type: Any, help: str)
@click.option('-i', '--codebuild-image', help: Optional[str])
@click.option('-s', '--source', default: str, type: Any, help: str)
@click.option('-b', '--buildspec-file', help: Optional[str])
@click.argument('filename')
@click.pass_context
def generate_pipeline(ctx: click.Context, pipeline_version: str, codebuild_image: Optional[str], source: str, buildspec_file: Optional[str], filename: str) -> None: ...

def main() -> int: ...
```