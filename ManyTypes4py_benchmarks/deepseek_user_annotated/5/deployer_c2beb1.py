"""Chalice deployer module."""
# pylint: disable=too-many-lines
import json
import textwrap
import socket
import logging
from typing import Optional, Dict, List, Any, Type, cast, Union

import botocore.exceptions
from botocore.vendored.requests import ConnectionError as RequestsConnectionError
from botocore.session import Session
from chalice.config import Config
from chalice.compat import is_broken_pipe_error
from chalice.awsclient import DeploymentPackageTooLargeError
from chalice.awsclient import LambdaClientError
from chalice.awsclient import AWSClientError
from chalice.awsclient import TypedAWSClient
from chalice.constants import MAX_LAMBDA_DEPLOYMENT_SIZE
from chalice.constants import VPC_ATTACH_POLICY
from chalice.constants import DEFAULT_LAMBDA_TIMEOUT
from chalice.constants import DEFAULT_LAMBDA_MEMORY_SIZE
from chalice.constants import DEFAULT_TLS_VERSION
from chalice.constants import SQS_EVENT_SOURCE_POLICY
from chalice.constants import KINESIS_EVENT_SOURCE_POLICY
from chalice.constants import DDB_EVENT_SOURCE_POLICY
from chalice.constants import POST_TO_WEBSOCKET_CONNECTION_POLICY
from chalice.deploy import models
from chalice.deploy.appgraph import ApplicationGraphBuilder, DependencyBuilder
from chalice.deploy.executor import BaseExecutor
from chalice.deploy.executor import Executor
from chalice.deploy.executor import DisplayOnlyExecutor
from chalice.deploy.packager import PipRunner
from chalice.deploy.packager import SubprocessPip
from chalice.deploy.packager import DependencyBuilder as PipDependencyBuilder
from chalice.deploy.packager import LambdaDeploymentPackager
from chalice.deploy.packager import AppOnlyDeploymentPackager
from chalice.deploy.packager import LayerDeploymentPackager
from chalice.deploy.packager import BaseLambdaDeploymentPackager
from chalice.deploy.packager import EmptyPackageError
from chalice.deploy.planner import PlanStage
from chalice.deploy.planner import RemoteState
from chalice.deploy.planner import NoopPlanner
from chalice.deploy.swagger import TemplatedSwaggerGenerator
from chalice.deploy.swagger import SwaggerGenerator
from chalice.deploy.sweeper import ResourceSweeper
from chalice.deploy.validate import validate_configuration
from chalice.policy import AppPolicyGenerator
from chalice.utils import OSUtils
from chalice.utils import UI
from chalice.utils import serialize_to_json


OptStr = Optional[str]
LOGGER = logging.getLogger(__name__)

_AWSCLIENT_EXCEPTIONS = (
    botocore.exceptions.ClientError, AWSClientError
)


class ChaliceDeploymentError(Exception):
    def __init__(self, error: Exception) -> None:
        self.original_error = error
        where = self._get_error_location(error)
        msg = self._wrap_text(
            'ERROR - %s, received the following error:' % where
        )
        msg += '\n\n'
        msg += self._wrap_text(self._get_error_message(error), indent=' ')
        msg += '\n\n'
        suggestion = self._get_error_suggestion(error)
        if suggestion is not None:
            msg += self._wrap_text(suggestion)
        super(ChaliceDeploymentError, self).__init__(msg)

    def _get_error_location(self, error: Exception) -> str:
        where = 'While deploying your chalice application'
        if isinstance(error, LambdaClientError):
            where = (
                'While sending your chalice handler code to Lambda to %s '
                'function "%s"' % (
                    self._get_verb_from_client_method(
                        error.context.client_method_name),
                    error.context.function_name
                )
            )
        return where

    def _get_error_message(self, error: Exception) -> str:
        msg = str(error)
        if isinstance(error, LambdaClientError):
            if isinstance(error.original_error, RequestsConnectionError):
                msg = self._get_error_message_for_connection_error(
                    error.original_error)
        return msg

    def _get_error_message_for_connection_error(self, connection_error: RequestsConnectionError) -> str:
        message = connection_error.args[0].args[0]
        underlying_error = connection_error.args[0].args[1]
        if is_broken_pipe_error(underlying_error):
            message += (
                ' Lambda closed the connection before chalice finished '
                'sending all of the data.'
            )
        elif isinstance(underlying_error, socket.timeout):
            message += ' Timed out sending your app to Lambda.'
        return message

    def _get_error_suggestion(self, error: Exception) -> OptStr:
        suggestion = None
        if isinstance(error, DeploymentPackageTooLargeError):
            suggestion = (
                'To avoid this error, decrease the size of your chalice '
                'application by removing code or removing '
                'dependencies from your chalice application.'
            )
            deployment_size = error.context.deployment_size
            if deployment_size > MAX_LAMBDA_DEPLOYMENT_SIZE:
                size_warning = (
                    'This is likely because the deployment package is %s. '
                    'Lambda only allows deployment packages that are %s or '
                    'less in size.' % (
                        self._get_mb(deployment_size),
                        self._get_mb(MAX_LAMBDA_DEPLOYMENT_SIZE)
                    )
                )
                suggestion = size_warning + ' ' + suggestion
        return suggestion

    def _wrap_text(self, text: str, indent: str = '') -> str:
        return '\n'.join(
            textwrap.wrap(
                text, 79, replace_whitespace=False, drop_whitespace=False,
                initial_indent=indent, subsequent_indent=indent
            )
        )

    def _get_verb_from_client_method(self, client_method_name: str) -> str:
        client_method_name_to_verb = {
            'update_function_code': 'update',
            'create_function': 'create'
        }
        return client_method_name_to_verb.get(
            client_method_name, client_method_name)

    def _get_mb(self, value: int) -> str:
        return '%.1f MB' % (float(value) / (1024 ** 2))


def create_plan_only_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, DisplayOnlyExecutor,
                            NoopResultsRecorder)


def create_default_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, Executor, ResultsRecorder)


def _create_deployer(session: Session,
                     config: Config,
                     ui: UI,
                     executor_cls: Type[BaseExecutor],
                     recorder_cls: Type['ResultsRecorder'],
                     ) -> 'Deployer':
    client = TypedAWSClient(session)
    osutils = OSUtils()
    return Deployer(
        application_builder=ApplicationGraphBuilder(),
        deps_builder=DependencyBuilder(),
        build_stage=create_build_stage(
            osutils, UI(), TemplatedSwaggerGenerator(), config
        ),
        plan_stage=PlanStage(
            osutils=osutils, remote_state=RemoteState(
                client, config.deployed_resources(config.chalice_stage)),
        ),
        sweeper=ResourceSweeper(),
        executor=executor_cls(client, ui),
        recorder=recorder_cls(osutils=osutils),
    )


def create_build_stage(osutils: OSUtils, ui: UI, swagger_gen: SwaggerGenerator, config: Config) -> 'BuildStage':
    pip_runner = PipRunner(pip=SubprocessPip(osutils=osutils),
                           osutils=osutils)
    dependency_builder = PipDependencyBuilder(
        osutils=osutils,
        pip_runner=pip_runner
    )
    deployment_packager = cast(BaseDeployStep, None)
    if config.automatic_layer:
        deployment_packager = ManagedLayerDeploymentPackager(
            lambda_packager=AppOnlyDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui,
            ),
            layer_packager=LayerDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui,
            )
        )
    else:
        deployment_packager = DeploymentPackager(
            packager=LambdaDeploymentPackager(
                osutils=osutils,
                dependency_builder=dependency_builder,
                ui=ui,
            )
        )
    build_stage = BuildStage(
        steps=[
            InjectDefaults(),
            deployment_packager,
            PolicyGenerator(
                policy_gen=AppPolicyGenerator(
                    osutils=osutils
                ),
                osutils=osutils,
            ),
            SwaggerBuilder(
                swagger_generator=swagger_gen,
            ),
            LambdaEventSourcePolicyInjector(),
            WebsocketPolicyInjector()
        ],
    )
    return build_stage


def create_deletion_deployer(client: TypedAWSClient, ui: UI) -> 'Deployer':
    return Deployer(
        application_builder=ApplicationGraphBuilder(),
        deps_builder=DependencyBuilder(),
        build_stage=BuildStage(steps=[]),
        plan_stage=NoopPlanner(),
        sweeper=ResourceSweeper(),
        executor=Executor(client, ui),
        recorder=ResultsRecorder(osutils=OSUtils()),
    )


class Deployer(object):
    BACKEND_NAME = 'api'

    def __init__(self,
                 application_builder: ApplicationGraphBuilder,
                 deps_builder: DependencyBuilder,
                 build_stage: 'BuildStage',
                 plan_stage: PlanStage,
                 sweeper: ResourceSweeper,
                 executor: BaseExecutor,
                 recorder: 'ResultsRecorder',
                 ) -> None:
        self._application_builder = application_builder
        self._deps_builder = deps_builder
        self._build_stage = build_stage
        self._plan_stage = plan_stage
        self._sweeper = sweeper
        self._executor = executor
        self._recorder = recorder

    def deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        try:
            return self._deploy(config, chalice_stage_name)
        except _AWSCLIENT_EXCEPTIONS as e:
            raise ChaliceDeploymentError(e)

    def _deploy(self, config: Config, chalice_stage_name: str) -> Dict[str, Any]:
        self._validate_config(config)
        application = self._application_builder.build(
            config, chalice_stage_name)
        resources = self._deps_builder.build_dependencies(application)
        self._build_stage.execute(config, resources)
        resources = self._deps_builder.build_dependencies(application)
        plan = self._plan_stage.execute(resources)
        self._sweeper.execute(plan, config)
        self._executor.execute(plan)
        deployed_values = {
            'resources': self._executor.resource_values,
            'schema_version': '2.0',
            'backend': self.BACKEND_NAME,
        }
        self._recorder.record_results(
            deployed_values,
            chalice_stage_name,
            config.project_dir,
        )
        return deployed_values

    def _validate_config(self, config: Config) -> None:
        try:
            validate_configuration(config)
        except ValueError as e:
            raise ChaliceDeploymentError(e)


class BaseDeployStep(object):
    def handle(self, config: Config, resource: models.Model) -> None:
        name = 'handle_%s' % resource.__class__.__name__.lower()
        handler = getattr(self, name, None)
        if handler is not None:
            handler(config, resource)


class InjectDefaults(BaseDeployStep):
    def __init__(self, lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT,
                 lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE,
                 tls_version: str = DEFAULT_TLS_VERSION) -> None:
        self._lambda_timeout = lambda_timeout
        self._lambda_memory_size = lambda_memory_size
        self._tls_version = DEFAULT_TLS_VERSION

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        if resource.timeout is None:
            resource.timeout = self._lambda_timeout
        if resource.memory_size is None:
            resource.memory_size = self._lambda_memory_size

    def handle_domainname(self, config: Config, resource: models.DomainName) -> None:
        if resource.tls_version is None:
            resource.tls_version = models.TLSVersion.create(
                DEFAULT_TLS_VERSION)


class DeploymentPackager(BaseDeployStep):
    def __init__(self, packager: LambdaDeploymentPackager) -> None:
        self._packager = packager

    def handle_deploymentpackage(self, config: Config, resource: models.DeploymentPackage) -> None:
        if isinstance(resource.filename, models.Placeholder):
            zip_filename = self._packager.create_deployment_package(
                config.project_dir, config.lambda_python_version)
            resource.filename = zip_filename


class ManagedLayerDeploymentPackager(BaseDeployStep):
    def __init__(self,
                 lambda_packager: BaseLambdaDeploymentPackager,
                 layer_packager: BaseLambdaDeploymentPackager,
                 ) -> None:
        self._lambda_packager = lambda_packager
        self._layer_packager = layer_packager

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        if isinstance(resource.deployment_package.filename,
                      models.Placeholder):
            zip_filename = self._lambda_packager.create_deployment_package(
                config.project_dir, config.lambda_python_version
            )
            resource.deployment_package.filename = zip_filename
        if resource.managed_layer is not None and \
                resource.managed_layer.is_empty:
            resource.managed_layer = None

    def handle_lambdalayer(self, config: Config, resource: models.LambdaLayer) -> None:
        if isinstance(resource.deployment_package.filename,
                      models.Placeholder):
            try:
                zip_filename = self._layer_packager.create_deployment_package(
                    config.project_dir, config.lambda_python_version
                )
                resource.deployment_package.filename = zip_filename
            except EmptyPackageError:
                resource.is_empty = True


class SwaggerBuilder(BaseDeployStep):
    def __init__(self, swagger_generator: SwaggerGenerator) -> None:
        self._swagger_generator = swagger_generator

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        swagger_doc = self._swagger_generator.generate_swagger(
            config.chalice_app, resource)
        resource.swagger_doc = swagger_doc


class LambdaEventSourcePolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        self._sqs_policy_injected = False
        self._kinesis_policy_injected = False
        self._ddb_policy_injected = False

    def handle_sqseventsource(self, config: Config, resource: models.SQSEventSource) -> None:
        role = resource.lambda_function.role
        if not self._sqs_policy_injected and \
                self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document,
                                        SQS_EVENT_SOURCE_POLICY.copy())
            self._sqs_policy_injected = True

    def handle_kinesiseventsource(self, config: Config, resource: models.KinesisEventSource) -> None:
        role = resource.lambda_function.role
        if not self._kinesis_policy_injected and \
                self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document,
                                        KINESIS_EVENT_SOURCE_POLICY.copy())
            self._kinesis_policy_injected = True

    def handle_dynamodbeventsource(self, config: Config, resource: models.KinesisEventSource) -> None:
        role = resource.lambda_function.role
        if not self._ddb_policy_injected and \
                self._needs_policy_injected(role):
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document,
                                        DDB_EVENT_SOURCE_POLICY.copy())
            self._ddb_policy_injected = True

    def _needs_policy_injected(self, role: models.IAMRole) -> bool:
        return (
            isinstance(role, models.ManagedIAMRole) and
            isinstance(role.policy, models.AutoGenIAMPolicy) and
            not isinstance(role.policy.document, models.Placeholder)
        )

    def _inject_trigger_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        document['Statement'].append(policy)


class WebsocketPolicyInjector(BaseDeployStep):
    def __init__(self) -> None:
        self._policy_injected = False

    def handle_websocketapi(self, config: Config, resource: models.WebsocketAPI) -> None:
        self._inject_into_function(config, resource.connect_function)
        self._inject_into_function(config, resource.message_function)
        self._inject_into_function(config, resource.disconnect_function)

    def _inject_into_function(self, config: Config, lambda_function: Optional[models.LambdaFunction]) -> None:
        if lambda_function is None:
            return
        role = lambda_function.role
        if role is None:
            return
        if (not self._policy_injected and
            isinstance(role, models.ManagedIAMRole) and
            isinstance(role.policy, models.Auto