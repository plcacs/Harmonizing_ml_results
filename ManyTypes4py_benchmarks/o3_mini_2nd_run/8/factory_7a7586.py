from __future__ import annotations
import sys
import os
import json
import importlib
import logging
import functools
import click
from botocore.config import Config as BotocoreConfig
from botocore.session import Session
from typing import Any, Optional, Dict, MutableMapping, cast
from chalice import __version__ as chalice_version
from chalice.awsclient import TypedAWSClient
from chalice.app import Chalice
from chalice.config import Config
from chalice.config import DeployedResources
from chalice.package import create_app_packager
from chalice.package import AppPackager
from chalice.package import PackageOptions
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import DEFAULT_APIGATEWAY_STAGE_NAME
from chalice.constants import DEFAULT_ENDPOINT_TYPE
from chalice.logs import LogRetriever, LogEventGenerator
from chalice.logs import FollowLogEventGenerator
from chalice.logs import BaseLogEventGenerator
from chalice import local
from chalice.utils import UI
from chalice.utils import PipeReader
from chalice.deploy import deployer
from chalice.deploy import validate
from chalice.invoke import LambdaInvokeHandler
from chalice.invoke import LambdaInvoker
from chalice.invoke import LambdaResponseFormatter

OptStr = Optional[str]
OptInt = Optional[int]


def create_botocore_session(profile: Optional[str] = None, debug: bool = False, connection_timeout: Optional[float] = None, read_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> Session:
    s: Session = Session(profile=profile)
    _add_chalice_user_agent(s)
    if debug:
        _inject_large_request_body_filter()
    config_args: Dict[str, Any] = {}
    if connection_timeout is not None:
        config_args['connect_timeout'] = connection_timeout
    if read_timeout is not None:
        config_args['read_timeout'] = read_timeout
    if max_retries is not None:
        config_args['retries'] = {'max_attempts': max_retries}
    if config_args:
        config = BotocoreConfig(**config_args)
        s.set_default_client_config(config)
    return s


def _add_chalice_user_agent(session: Session) -> None:
    suffix = '%s/%s' % (session.user_agent_name, session.user_agent_version)
    session.user_agent_name = 'aws-chalice'
    session.user_agent_version = chalice_version
    session.user_agent_extra = suffix


def _inject_large_request_body_filter() -> None:
    log = logging.getLogger('botocore.endpoint')
    log.addFilter(LargeRequestBodyFilter())


class NoSuchFunctionError(Exception):
    """The specified function could not be found."""

    def __init__(self, name: str) -> None:
        self.name = name
        super(NoSuchFunctionError, self).__init__()


class UnknownConfigFileVersion(Exception):

    def __init__(self, version: str) -> None:
        super(UnknownConfigFileVersion, self).__init__("Unknown version '%s' in config.json" % version)


class LargeRequestBodyFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg.startswith('Making request'):
            if record.args[0].name in ['UpdateFunctionCode', 'CreateFunction']:
                record.args = record.args[:-1] + ('(... omitted from logs due to size ...)',)
        return True


class CLIFactory(object):

    def __init__(self, project_dir: str, debug: bool = False, profile: Optional[str] = None, environ: Optional[Dict[str, str]] = None) -> None:
        self.project_dir: str = project_dir
        self.debug: bool = debug
        self.profile: Optional[str] = profile
        if environ is None:
            environ = dict(os.environ)
        self._environ: Dict[str, str] = environ

    def create_botocore_session(self, connection_timeout: Optional[float] = None, read_timeout: Optional[float] = None, max_retries: Optional[int] = None) -> Session:
        return create_botocore_session(profile=self.profile, debug=self.debug, connection_timeout=connection_timeout, read_timeout=read_timeout, max_retries=max_retries)

    def create_default_deployer(self, session: Session, config: Config, ui: UI) -> Any:
        return deployer.create_default_deployer(session, config, ui)

    def create_plan_only_deployer(self, session: Session, config: Config, ui: UI) -> Any:
        return deployer.create_plan_only_deployer(session, config, ui)

    def create_deletion_deployer(self, session: Session, ui: UI) -> Any:
        return deployer.create_deletion_deployer(TypedAWSClient(session), ui)

    def create_deployment_reporter(self, ui: UI) -> Any:
        return deployer.DeploymentReporter(ui=ui)

    def create_config_obj(self, chalice_stage_name: str = DEFAULT_STAGE_NAME, autogen_policy: Optional[Any] = None, api_gateway_stage: Optional[str] = None, user_provided_params: Optional[Dict[str, Any]] = None) -> Config:
        if user_provided_params is None:
            user_provided_params = {}
        default_params: Dict[str, Any] = {
            'project_dir': self.project_dir,
            'api_gateway_stage': DEFAULT_APIGATEWAY_STAGE_NAME,
            'api_gateway_endpoint_type': DEFAULT_ENDPOINT_TYPE,
            'autogen_policy': True
        }
        try:
            config_from_disk: Dict[str, Any] = self.load_project_config()
        except (OSError, IOError):
            raise RuntimeError('Unable to load the project config file. Are you sure this is a chalice project?')
        except ValueError as err:
            raise RuntimeError('Unable to load the project config file: %s' % err)
        self._validate_config_from_disk(config_from_disk)
        if autogen_policy is not None:
            user_provided_params['autogen_policy'] = autogen_policy
        if self.profile is not None:
            user_provided_params['profile'] = self.profile
        if api_gateway_stage is not None:
            user_provided_params['api_gateway_stage'] = api_gateway_stage
        config: Config = Config(chalice_stage=chalice_stage_name, user_provided_params=user_provided_params, config_from_disk=config_from_disk, default_params=default_params)
        user_provided_params['chalice_app'] = functools.partial(self.load_chalice_app, environment_variables=config.environment_variables)
        return config

    def _validate_config_from_disk(self, config: Dict[str, Any]) -> None:
        string_version: str = config.get('version', '1.0')
        try:
            version: float = float(string_version)
            if version > 2.0:
                raise UnknownConfigFileVersion(string_version)
        except ValueError:
            raise UnknownConfigFileVersion(string_version)

    def create_app_packager(self, config: Config, options: PackageOptions, package_format: str, template_format: str, merge_template: Optional[Any] = None) -> AppPackager:
        return create_app_packager(config, options, package_format, template_format, merge_template=merge_template)

    def create_log_retriever(self, session: Session, lambda_arn: str, follow_logs: bool) -> LogRetriever:
        client = TypedAWSClient(session)
        if follow_logs:
            event_generator: BaseLogEventGenerator = cast(BaseLogEventGenerator, FollowLogEventGenerator(client))
        else:
            event_generator = cast(BaseLogEventGenerator, LogEventGenerator(client))
        retriever: LogRetriever = LogRetriever.create_from_lambda_arn(event_generator, lambda_arn)
        return retriever

    def create_stdin_reader(self) -> PipeReader:
        stream = click.get_binary_stream('stdin')
        reader: PipeReader = PipeReader(stream)
        return reader

    def create_lambda_invoke_handler(self, name: str, stage: str) -> LambdaInvokeHandler:
        config: Config = self.create_config_obj(stage)
        deployed: DeployedResources = config.deployed_resources(stage)
        try:
            resource = deployed.resource_values(name)
            arn: str = resource['lambda_arn']
        except (KeyError, ValueError):
            raise NoSuchFunctionError(name)
        function_scoped_config: Config = config.scope(stage, name)
        session: Session = self.create_botocore_session(read_timeout=function_scoped_config.lambda_timeout, max_retries=0)
        client = TypedAWSClient(session)
        invoker = LambdaInvoker(arn, client)
        handler: LambdaInvokeHandler = LambdaInvokeHandler(invoker, LambdaResponseFormatter(), UI())
        return handler

    def load_chalice_app(self, environment_variables: Optional[Dict[str, str]] = None, validate_feature_flags: bool = True) -> Chalice:
        if self.project_dir not in sys.path:
            sys.path.insert(0, self.project_dir)
        vendor_dir: str = os.path.join(self.project_dir, 'vendor')
        if os.path.isdir(vendor_dir) and vendor_dir not in sys.path:
            sys.path.append(vendor_dir)
        if environment_variables is not None:
            self._environ.update(environment_variables)
        try:
            app = importlib.import_module('app')
            chalice_app: Chalice = getattr(app, 'app')
        except SyntaxError as e:
            message = 'Unable to import your app.py file:\n\nFile "%s", line %s\n  %s\nSyntaxError: %s' % (getattr(e, 'filename'), e.lineno, e.text, e.msg)
            raise RuntimeError(message)
        if validate_feature_flags:
            validate.validate_feature_flags(chalice_app)
        return chalice_app

    def load_project_config(self) -> Dict[str, Any]:
        """Load the chalice config file from the project directory.

        :raise: OSError/IOError if unable to load the config file.

        """
        config_file: str = os.path.join(self.project_dir, '.chalice', 'config.json')
        with open(config_file) as f:
            return json.loads(f.read())

    def create_local_server(self, app_obj: Chalice, config: Config, host: str, port: int) -> Any:
        return local.create_local_server(app_obj, config, host, port)

    def create_package_options(self) -> PackageOptions:
        """Create the package options that are required to target regions."""
        s: Session = Session(profile=self.profile)
        client = TypedAWSClient(session=s)
        return PackageOptions(client)