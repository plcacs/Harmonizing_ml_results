from __future__ import annotations

from typing import Optional, Union

import click
from chalice.cli.factory import CLIFactory
from chalice.local import LocalDevServer


def _configure_logging(level: Union[int, str], format_string: Optional[str] = ...) -> None: ...
def get_system_info() -> str: ...
def cli(ctx: click.Context, project_dir: Optional[str], debug: bool = ...) -> None: ...
def _configure_cli_env_vars() -> None: ...
def local(
    ctx: click.Context,
    host: str = ...,
    port: int = ...,
    stage: str = ...,
    autoreload: bool = ...,
) -> None: ...
def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...
def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...
def deploy(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
    connection_timeout: Optional[int],
) -> None: ...
def dev() -> None: ...
def plan(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
) -> None: ...
def appgraph(
    ctx: click.Context,
    autogen_policy: Optional[bool],
    profile: Optional[str],
    api_gateway_stage: Optional[str],
    stage: str,
) -> None: ...
def invoke(ctx: click.Context, name: str, profile: Optional[str], stage: str) -> None: ...
def delete(ctx: click.Context, profile: Optional[str], stage: str) -> None: ...
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
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None: ...
def new_project(
    ctx: click.Context,
    project_name: Optional[str],
    profile: Optional[str],
    project_type: str,
) -> None: ...
def url(ctx: click.Context, stage: str) -> None: ...
def generate_sdk(ctx: click.Context, sdk_type: str, stage: str, outdir: str) -> None: ...
def generate_models(ctx: click.Context, stage: str) -> None: ...
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
def generate_pipeline(
    ctx: click.Context,
    pipeline_version: str,
    codebuild_image: Optional[str],
    source: str,
    buildspec_file: Optional[str],
    filename: str,
) -> None: ...
def main() -> int: ...