from __future__ import annotations
import logging
import sys
import tempfile
import click
from typing import Dict, Any, Optional, cast
from chalice import __version__ as chalice_version
from chalice.app import Chalice
from chalice.awsclient import TypedAWSClient
from chalice.awsclient import ReadTimeout
from chalice.cli.factory import CLIFactory
from chalice.cli.factory import NoSuchFunctionError
from chalice.config import Config
from chalice.logs import LogRetrieveOptions
from chalice.utils import UI
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

@click.group()
@click.version_option(version=chalice_version, message='%(prog)s %(version)s, {}'.format(get_system_info()))
@click.option('--project-dir', help='The project directory path (absolute or relative).Defaults to CWD')
@click.option('--debug/--no-debug', default=False, help='Print debug logs to stderr.')
@click.pass_context
def cli(ctx: click.Context, project_dir: Optional[str] = None, debug: bool = False) -> None: ...

def _configure_cli_env_vars() -> None: ...

@cli.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8000, type=click.INT)
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage for the local server to use.')
@click.option('--autoreload/--no-autoreload', default=True, help='Automatically restart server when code changes.')
@click.pass_context
def local(ctx: click.Context, host: str = '127.0.0.1', port: int = 8000, stage: str = DEFAULT_STAGE_NAME, autoreload: bool = True) -> None: ...

def create_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> LocalDevServer: ...

def run_local_server(factory: CLIFactory, host: str, port: int, stage: str) -> None: ...

@cli.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage', help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to deploy to. Specifying a new chalice stage will create an entirely new set of AWS resources.')
@click.option('--connection-timeout', type=int, help='Overrides the default botocore connection timeout.')
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME, connection_timeout: Optional[int] = None) -> None: ...

@cli.group()
def dev() -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage', help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to deploy to. Specifying a new chalice stage will create an entirely new set of AWS resources.')
@click.pass_context
def plan(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@dev.command()
@click.option('--autogen-policy/--no-autogen-policy', default=None, help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage', help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to deploy to. Specifying a new chalice stage will create an entirely new set of AWS resources.')
@click.pass_context
def appgraph(ctx: click.Context, autogen_policy: Optional[bool] = None, profile: Optional[str] = None, api_gateway_stage: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('invoke')
@click.option('-n', '--name', metavar='NAME', required=True, help='The name of the function to invoke. This is the logical name of the function. If the function is decorated by app.route use the name api_handler instead.')
@click.option('--profile', metavar='PROFILE', help='Override profile at deploy time.')
@click.option('--stage', metavar='STAGE', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to deploy to. Specifying a new chalice stage will create an entirely new set of AWS resources.')
@click.pass_context
def invoke(ctx: click.Context, name: str, profile: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('delete')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to delete.')
@click.pass_context
def delete(ctx: click.Context, profile: Optional[str] = None, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command()
@click.option('--num-entries', default=None, type=int, help='Max number of log entries to show.')
@click.option('--include-lambda-messages/--no-include-lambda-messages', default=False, help='Controls whether or not lambda log messages are included.')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to get logs for.')
@click.option('-n', '--name', help='The name of the lambda function to retrieve logs from.', default=DEFAULT_HANDLER_NAME)
@click.option('-s', '--since', help='Only display logs since the provided time.  If the -f/--follow option is specified, then this value will default to 10 minutes from the current time.  Otherwise by default all log messages are displayed.  This value can either be an ISO8601 formatted timestamp or a relative time.  For relative times provide a number and a single unit.  Units can be "s" for seconds, "m" for minutes, "h" for hours, "d" for days, and "w" for weeks.  For example "5m" would indicate to display logs starting five minutes in the past.', default=None)
@click.option('-f', '--follow/--no-follow', default=False, help='Continuously poll for new log messages.  Note that this is a best effort attempt, and in certain cases can miss log messages.  This option is intended for interactive usage only.')
@click.option('--profile', help='The profile to use for fetching logs.')
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int] = None, include_lambda_messages: bool = False, stage: str = DEFAULT_STAGE_NAME, name: str = DEFAULT_HANDLER_NAME, since: Optional[str] = None, follow: bool = False, profile: Optional[str] = None) -> None: ...

@cli.command('gen-policy')
@click.option('--filename', help='The filename to analyze.  Otherwise app.py is assumed.')
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str] = None) -> None: ...

@cli.command('new-project')
@click.argument('project_name', required=False)
@click.option('--profile', required=False)
@click.option('-t', '--project-type', required=False, default='legacy')
@click.pass_context
def new_project(ctx: click.Context, project_name: Optional[str] = None, profile: Optional[str] = None, project_type: str = 'legacy') -> None: ...

@cli.command('url')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to get the deployed URL for.')
@click.pass_context
def url(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('generate-sdk')
@click.option('--sdk-type', default='javascript', type=click.Choice(['javascript']))
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Name of the Chalice stage to generate an SDK for.')
@click.argument('outdir')
@click.pass_context
def generate_sdk(ctx: click.Context, sdk_type: str = 'javascript', stage: str = DEFAULT_STAGE_NAME, outdir: str = ...) -> None: ...

@cli.command('generate-models')
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Chalice Stage for which to generate models.')
@click.pass_context
def generate_models(ctx: click.Context, stage: str = DEFAULT_STAGE_NAME) -> None: ...

@cli.command('package')
@click.option('--pkg-format', default='cloudformation', help='Specify the provisioning engine to use for template output. Chalice supports both CloudFormation and Terraform. Default is CloudFormation.', type=click.Choice(['cloudformation', 'terraform']))
@click.option('--stage', default=DEFAULT_STAGE_NAME, help='Chalice Stage to package.')
@click.option('--single-file', is_flag=True, default=False, help="Create a single packaged file. By default, the 'out' argument specifies a directory in which the package assets will be placed.  If this argument is specified, a single zip file will be created instead. CloudFormation Only.")
@click.option('--merge-template', help='Specify a JSON or YAML template to be merged into the generated template. This is useful for adding resources to a Chalice template or modify values in the template. CloudFormation Only.')
@click.option('--template-format', default='json', type=click.Choice(['json', 'yaml'], case_sensitive=False), help='Specify if the generated template should be serialized as either JSON or YAML.  CloudFormation only.')
@click.option('--profile', help='Override profile at packaging time.')
@click.argument('out')
@click.pass_context
def package(ctx: click.Context, single_file: bool = False, stage: str = DEFAULT_STAGE_NAME, merge_template: Optional[str] = None, out: str = ..., pkg_format: str = 'cloudformation', template_format: str = 'json', profile: Optional[str] = None) -> None: ...

@cli.command('generate-pipeline')
@click.option('--pipeline-version', default='v1', type=click.Choice(['v1', 'v2']), help='Which version of the pipeline template to generate.')
@click.option('-i', '--codebuild-image', help='Specify default codebuild image to use.  This option must be provided when using a python version besides 2.7.')
@click.option('-s', '--source', default='codecommit', type=click.Choice(['codecommit', 'github']), help="Specify the input source.  The default value of 'codecommit' will create a CodeCommit repository for you.  The 'github' value allows you to reference an existing GitHub repository.")
@click.option('-b', '--buildspec-file', help="Specify path for buildspec.yml file. By default, the build steps are included in the generated cloudformation template.  If this option is provided, a buildspec.yml will be generated as a separate file and not included in the cfn template.  This allows you to make changes to how the project is built without having to redeploy a CloudFormation template. This file should be named 'buildspec.yml' and placed in the root directory of your app.")
@click.argument('filename')
@click.pass_context
def generate_pipeline(ctx: click.Context, pipeline_version: str = 'v1', codebuild_image: Optional[str] = None, source: str = 'codecommit', buildspec_file: Optional[str] = None, filename: str = ...) -> None: ...

def main() -> int: ...