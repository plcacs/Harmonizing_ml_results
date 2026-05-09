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
from typing import Dict, Any, Optional, cast

@cli.group()
def cli(obj: Dict[str, Any]) -> None:
    ...

@cli.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8000, type=click.INT)
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage for the local server to use.')
@click.option('--autoreload/--no-autoreload', default=True, help='Automatically restart server when code changes.')
@click.pass_context
def local(ctx: click.Context, host: str, port: int, stage: str, autoreload: bool) -> None:
    ...

@cli.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage', help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to deploy to.')
@click.option('--connection-timeout', type=int, help='Overrides the default botocore connection timeout.')
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str], api_gateway_stage: Optional[str], stage: str, connection_timeout: Optional[int]) -> None:
    ...

@cli.command()
@click.option('--num-entries', default=None, type=int, help='Max number of log entries to show.')
@click.option('--include-lambda-messages/--no-include-lambda-messages', default=False, help='Controls whether or not lambda log messages are included.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to get logs for.')
@click.option('-n', '--name', help='The name of the lambda function to retrieve logs from.', default=DEFAULT_HANDLER_NAME)
@click.option('-s', '--since', help='Only display logs since the provided time.  If the -f/--follow option is specified, then this value will default to 10 minutes from the current time.  Otherwise by default all log messages are displayed.  This value can either be an ISO8601 formatted timestamp or a relative time.  For relative times provide a number and a single unit.  Units can be "s" for seconds, "m" for minutes, "h" for hours, "d" for days, and "w" for weeks.  For example "5m" would indicate to display logs starting five minutes in the past.')
@click.option('-f', '--follow/--no-follow', default=False, help='Continuously poll for new log messages.  Note that this is a best effort attempt, and in certain cases can miss log messages.  This option is intended for interactive usage only.')
@click.option('--profile', help='The profile to use for fetching logs.')
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int], include_lambda_messages: bool, stage: str, name: str, since: Optional[str], follow: bool, profile: Optional[str]) -> None:
    ...

@cli.command()
@click.option('--filename', help='The filename to analyze.  Otherwise app.py is assumed.')
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None:
    ...

@cli.command()
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Chalice Stage for which to generate models.')
@click.pass_context
def generate_models(ctx: click.Context, stage: str) -> None:
    ...

@cli.command()
@click.option('--pkg-format', default='cloudformation', help='Specify the provisioning engine to use for template output. Chalice supports both CloudFormation and Terraform. Default is CloudFormation.', type=click.Choice(['cloudformation', 'terraform']))
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Chalice Stage to package.')
@click.option('--single-file', is_flag=True, default=False, help="Create a single packaged file. By default, the 'out' argument specifies a directory in which the package assets will be placed.  If this argument is specified, a single zip file will be created instead. CloudFormation Only.")
@click.option('--merge-template', help='Specify a JSON or YAML template to be merged into the generated template. This is useful for adding resources to a Chalice template or modify values in the template. CloudFormation Only.')
@click.option('--template-format', default='json', type=click.Choice(['json', 'yaml'], case_sensitive=False), help='Specify if the generated template should be serialized as either JSON or YAML.  CloudFormation only.')
@click.option('--profile', help='Override profile at packaging time.')
@click.argument('out')
@click.pass_context
def package(ctx: click.Context, single_file: bool, stage: str, merge_template: Optional[str], out: str, pkg_format: str, template_format: str, profile: Optional[str]) -> None:
    ...

@cli.command()
@click.option('--pipeline-version', default='v1', type=click.Choice(['v1', 'v2']), help='Which version of the pipeline template to generate.')
@click.option('-i', '--codebuild-image', help='Specify default codebuild image to use.  This option must be provided when using a python version besides 2.7.')
@click.option('-s', '--source', default='codecommit', type=click.Choice(['codecommit', 'github']), help="Specify the input source.  The default value of 'codecommit' will create a CodeCommit repository for you.  The 'github' value allows you to reference an existing GitHub repository.")
@click.option('-b', '--buildspec-file', help="Specify path for buildspec.yml file. By default, the build steps are included in the generated cloudformation template.  If this option is provided, a buildspec.yml will be generated as a separate file and not included in the cfn template.  This file should be named 'buildspec.yml' and placed in the root directory of your app.")
@click.argument('filename')
@click.pass_context
def generate_pipeline(ctx: click.Context, pipeline_version: str, codebuild_image: Optional[str], source: str, buildspec_file: Optional[str], filename: str) -> None:
    ...
