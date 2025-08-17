"""Command line interface for chalice.

Contains commands for deploying chalice.

"""
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
from typing import Dict, Any, Optional, cast, List, Union, IO  # noqa

from chalice import __version__ as chalice_version
from chalice.app import Chalice  # noqa
from chalice.awsclient import TypedAWSClient
from chalice.awsclient import ReadTimeout
from chalice.cli.factory import CLIFactory
from chalice.cli.factory import NoSuchFunctionError
from chalice.config import Config  # noqa
from chalice.logs import display_logs, LogRetrieveOptions
from chalice.utils import create_zip_file
from chalice.deploy.validate import validate_routes, validate_python_version
from chalice.deploy.validate import ExperimentalFeatureError
from chalice.utils import UI, serialize_to_json
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.local import LocalDevServer  # noqa
from chalice.constants import DEFAULT_HANDLER_NAME
from chalice.invoke import UnhandledLambdaError
from chalice.deploy.swagger import TemplatedSwaggerGenerator
from chalice.deploy.planner import PlanEncoder
from chalice.deploy.appgraph import ApplicationGraphBuilder, GraphPrettyPrint
from chalice.cli import newproj


def _configure_logging(level: int, format_string: Optional[str] = None) -> None:
    if format_string is None:
        format_string = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logger = logging.getLogger('')
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_system_info() -> str:
    python_info = "python {}.{}.{}".format(sys.version_info[0],
                                           sys.version_info[1],
                                           sys.version_info[2])
    platform_system = platform.system().lower()
    platform_release = platform.release()
    platform_info = "{} {}".format(platform_system, platform_release)
    return "{}, {}".format(python_info, platform_info)


@click.group()
@click.version_option(version=chalice_version,
                      message='%(prog)s %(version)s, {}'
                      .format(get_system_info()))
@click.option('--project-dir',
              help='The project directory path (absolute or relative).'
                   'Defaults to CWD')
@click.option('--debug/--no-debug',
              default=False,
              help='Print debug logs to stderr.')
@click.pass_context
def cli(ctx: click.Context, project_dir: Optional[str], debug: bool = False) -> None:
    if project_dir is None:
        project_dir = os.getcwd()
    elif not os.path.isabs(project_dir):
        project_dir = os.path.abspath(project_dir)
    if debug is True:
        _configure_logging(logging.DEBUG)
    _configure_cli_env_vars()
    ctx.obj['project_dir'] = project_dir
    ctx.obj['debug'] = debug
    ctx.obj['factory'] = CLIFactory(project_dir, debug, environ=os.environ)
    os.chdir(project_dir)


def _configure_cli_env_vars() -> None:
    os.environ['AWS_CHALICE_CLI_MODE'] = 'true'


@cli.command()
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8000, type=click.INT)
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help='Name of the Chalice stage for the local server to use.')
@click.option('--autoreload/--no-autoreload',
              default=True,
              help='Automatically restart server when code changes.')
@click.pass_context
def local(ctx: click.Context, host: str = '127.0.0.1', port: int = 8000,
          stage: str = DEFAULT_STAGE_NAME, autoreload: bool = True) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    from chalice.cli import reloader
    server_factory = functools.partial(
        create_local_server, factory, host, port, stage)
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format='%(message)s')
    if autoreload:
        project_dir = factory.create_config_obj(
            chalice_stage_name=stage).project_dir
        rc = reloader.run_with_reloader(
            server_factory, os.environ, project_dir)
        sys.exit(rc)
    run_local_server(factory, host, port, stage)


def create_local_server(factory: CLIFactory, host: str, port: int,
                       stage: str) -> LocalDevServer:
    config = factory.create_config_obj(
        chalice_stage_name=stage
    )
    app_obj = config.chalice_app
    routes = config.chalice_app.routes
    validate_routes(routes)
    server = factory.create_local_server(app_obj, config, host, port)
    return server


def run_local_server(factory: CLIFactory, host: str, port: int,
                     stage: str) -> None:
    server = create_local_server(factory, host, port, stage)
    server.serve_forever()


@cli.command()
@click.option('--autogen-policy/--no-autogen-policy',
              default=None,
              help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage',
              help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help=('Name of the Chalice stage to deploy to. '
                    'Specifying a new chalice stage will create '
                    'an entirely new set of AWS resources.'))
@click.option('--connection-timeout',
              type=int,
              help=('Overrides the default botocore connection '
                    'timeout.'))
@click.pass_context
def deploy(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str],
           api_gateway_stage: Optional[str], stage: str,
           connection_timeout: Optional[int]) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile
    config = factory.create_config_obj(
        chalice_stage_name=stage, autogen_policy=autogen_policy,
        api_gateway_stage=api_gateway_stage,
    )
    session = factory.create_botocore_session(
        connection_timeout=connection_timeout)
    ui = UI()
    d = factory.create_default_deployer(session=session,
                                        config=config,
                                        ui=ui)
    deployed_values = d.deploy(config, chalice_stage_name=stage)
    reporter = factory.create_deployment_reporter(ui=ui)
    reporter.display_report(deployed_values)


@cli.group()
def dev() -> None:
    """Development and debugging commands for chalice."""
    pass


@dev.command()
@click.option('--autogen-policy/--no-autogen-policy',
              default=None,
              help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage',
              help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help=('Name of the Chalice stage to deploy to. '
                    'Specifying a new chalice stage will create '
                    'an entirely new set of AWS resources.'))
@click.pass_context
def plan(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str],
         api_gateway_stage: Optional[str], stage: str) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile
    config = factory.create_config_obj(
        chalice_stage_name=stage, autogen_policy=autogen_policy,
        api_gateway_stage=api_gateway_stage,
    )
    session = factory.create_botocore_session()
    ui = UI()
    d = factory.create_plan_only_deployer(
        session=session, config=config, ui=ui)
    d.deploy(config, chalice_stage_name=stage)


@dev.command()
@click.option('--autogen-policy/--no-autogen-policy',
              default=None,
              help='Automatically generate IAM policy for app code.')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--api-gateway-stage',
              help='Name of the API gateway stage to deploy to.')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help=('Name of the Chalice stage to deploy to. '
                    'Specifying a new chalice stage will create '
                    'an entirely new set of AWS resources.'))
@click.pass_context
def appgraph(ctx: click.Context, autogen_policy: Optional[bool], profile: Optional[str],
             api_gateway_stage: Optional[str], stage: str) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile
    config = factory.create_config_obj(
        chalice_stage_name=stage, autogen_policy=autogen_policy,
        api_gateway_stage=api_gateway_stage,
    )
    graph_build = ApplicationGraphBuilder()
    graph = graph_build.build(config, stage)
    ui = UI()
    GraphPrettyPrint(ui).display_graph(graph)


@cli.command('invoke')
@click.option('-n', '--name', metavar='NAME', required=True,
              help=('The name of the function to invoke. '
                    'This is the logical name of the function. If the '
                    'function is decorated by app.route use the name '
                    'api_handler instead.'))
@click.option('--profile', metavar='PROFILE',
              help='Override profile at deploy time.')
@click.option('--stage', metavar='STAGE', default=DEFAULT_STAGE_NAME,
              help=('Name of the Chalice stage to deploy to. '
                    'Specifying a new chalice stage will create '
                    'an entirely new set of AWS resources.'))
@click.pass_context
def invoke(ctx: click.Context, name: str, profile: Optional[str],
           stage: str) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile

    try:
        invoke_handler = factory.create_lambda_invoke_handler(name, stage)
        payload = factory.create_stdin_reader().read()
        invoke_handler.invoke(payload)
    except NoSuchFunctionError as e:
        err = click.ClickException(
            "could not find a lambda function named %s." % e.name)
        err.exit_code = 2
        raise err
    except botocore.exceptions.ClientError as e:
        error = e.response['Error']
        err = click.ClickException(
            "got '%s' exception back from Lambda\n%s"
            % (error['Code'], error['Message']))
        err.exit_code = 1
        raise err
    except UnhandledLambdaError:
        err = click.ClickException(
            "Unhandled exception in Lambda function, details above.")
        err.exit_code = 1
        raise err
    except ReadTimeout as e:
        err = click.ClickException(e.message)
        err.exit_code = 1
        raise err


@cli.command('delete')
@click.option('--profile', help='Override profile at deploy time.')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help='Name of the Chalice stage to delete.')
@click.pass_context
def delete(ctx: click.Context, profile: Optional[str], stage: str) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile
    config = factory.create_config_obj(chalice_stage_name=stage)
    session = factory.create_botocore_session()
    d = factory.create_deletion_deployer(session=session, ui=UI())
    d.deploy(config, chalice_stage_name=stage)


@cli.command()
@click.option('--num-entries', default=None, type=int,
              help='Max number of log entries to show.')
@click.option('--include-lambda-messages/--no-include-lambda-messages',
              default=False,
              help='Controls whether or not lambda log messages are included.')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help='Name of the Chalice stage to get logs for.')
@click.option('-n', '--name',
              help='The name of the lambda function to retrieve logs from.',
              default=DEFAULT_HANDLER_NAME)
@click.option('-s', '--since',
              help=('Only display logs since the provided time.  If the '
                    '-f/--follow option is specified, then this value will '
                    'default to 10 minutes from the current time.  Otherwise '
                    'by default all log messages are displayed.  This value '
                    'can either be an ISO8601 formatted timestamp or a '
                    'relative time.  For relative times provide a number '
                    'and a single unit.  Units can be "s" for seconds, '
                    '"m" for minutes, "h" for hours, "d" for days, and "w" '
                    'for weeks.  For example "5m" would indicate to display '
                    'logs starting five minutes in the past.'),
              default=None)
@click.option('-f', '--follow/--no-follow',
              default=False,
              help=('Continuously poll for new log messages.  Note that this '
                    'is a best effort attempt, and in certain cases can '
                    'miss log messages.  This option is intended for '
                    'interactive usage only.'))
@click.option('--profile', help='The profile to use for fetching logs.')
@click.pass_context
def logs(ctx: click.Context, num_entries: Optional[int],
         include_lambda_messages: bool, stage: str, name: str,
         since: Optional[str], follow: bool, profile: Optional[str]) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    factory.profile = profile
    config = factory.create_config_obj(stage, False)
    deployed = config.deployed_resources(stage)
    if name in deployed.resource_names():
        lambda_arn = deployed.resource_values(name)['lambda_arn']
        session = factory.create_botocore_session()
        retriever = factory.create_log_retriever(
            session, lambda_arn, follow)
        options = LogRetrieveOptions.create(
            max_entries=num_entries,
            since=since,
            include_lambda_messages=include_lambda_messages,
        )
        display_logs(retriever, sys.stdout, options)


@cli.command('gen-policy')
@click.option('--filename',
              help='The filename to analyze.  Otherwise app.py is assumed.')
@click.pass_context
def gen_policy(ctx: click.Context, filename: Optional[str]) -> None:
    from chalice import policy
    if filename is None:
        filename = os.path.join(ctx.obj['project_dir'], 'app.py')
    if not os.path.isfile(filename):
        click.echo("App file does not exist: %s" % filename, err=True)
        raise click.Abort()
    with open(filename) as f:
        contents = f.read()
        generated = policy.policy_from_source_code(contents)
        click.echo(serialize_to_json(generated))


@cli.command('new-project')
@click.argument('project_name', required=False)
@click.option('--profile', required=False)
@click.option('-t', '--project-type', required=False, default='legacy')
@click.pass_context
def new_project(ctx: click.Context, project_name: Optional[str],
                profile: Optional[str], project_type: str) -> None:
    if project_name is None:
        prompter = ctx.obj.get('prompter', newproj.getting_started_prompt)
        answers = prompter()
        project_name = answers['project_name']
        project_type = answers['project_type']
    if os.path.isdir(project_name):
        click.echo("Directory already exists: %s" % project_name, err=True)
        raise click.Abort()
    newproj.create_new_project_skeleton(
        project_name, project_type=project_type)
    validate_python_version(Config.create())
    click.echo("Your project has been generated in ./%s" % project_name)


@cli.command('url')
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help='Name of the Chalice stage to get the deployed URL for.')
@click.pass_context
def url(ctx: click.Context, stage: str) -> None:
    factory = ctx.obj['factory']  # type: CLIFactory
    config = factory.create_config_obj(stage)
    deployed = config.deployed_resources(stage)
    if deployed is not None and 'rest_api' in deployed.resource_names():
        click.echo(deployed.resource_values('rest_api')['rest_api_url'])
    else:
        e = click.ClickException(
            "Could not find a record of a Rest API in chalice stage: '%s'"
            % stage)
        e.exit_code = 2
        raise e


@cli.command('generate-sdk')
@click.option('--sdk-type', default='javascript',
              type=click.Choice(['javascript']))
@click.option('--stage', default=DEFAULT_STAGE_NAME,
              help='Name of the Chalice stage to generate an SDK for.')
@click.argument('outdir')
@click.pass_context
def generate_sdk(ctx: click.Context, sdk_type: str, stage: str,
                 outdir: str) -> None:
    factory = ctx.obj['factory']  # type