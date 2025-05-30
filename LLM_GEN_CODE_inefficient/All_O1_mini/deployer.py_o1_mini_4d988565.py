"""Chalice deployer module.

The deployment system in chalice is broken down into a pipeline of multiple
stages.  Each stage takes the input and transforms it to some other form.  The
reason for this is so that each stage can stay simple and focused on only a
single part of the deployment process.  This makes the code easier to follow
and easier to test.  The biggest downside is that adding support for a new
resource type is split across several objects now, but I imagine as we add
support for more resource types, we'll see common patterns emerge that we can
extract out into higher levels of abstraction.

These are the stages of the deployment process.

Application Graph Builder
=========================

The first stage is the resource graph builder.  This takes the objects in the
``Chalice`` app and structures them into an ``Application`` object which
consists of various ``models.Model`` objects.  These models are just python
objects that describe the attributes of various AWS resources.  These
models don't have any behavior on their own.

Dependency Builder
==================

This process takes the graph of resources created from the previous step and
orders them such that all objects are listed before objects that depend on
them.  The AWS resources in the ``chalice.deploy.models`` module also model
their required dependencies (see the ``dependencies()`` methods of the models).
This is the mechanism that's used to build the correct dependency ordering.

Local Build Stage
=================

This takes the ordered list of resources and allows any local build processes
to occur.  The rule of thumb here is no remote AWS calls.  This stage includes
auto policy generation, pip packaging, injecting default values, etc.  To
clarify which attributes are affected by the build stage, they'll usually have
a value of ``models.Placeholder.BUILD_STAGE``.  Processors in the build stage
will replaced those ``models.Placeholder.BUILD_STAGE`` values with whatever the
"built" value is (e.g the filename of the zipped deployment package).

For example, we know when we create a lambda function that we need to create a
deployment package, but we don't know the name nor contents of the deployment
package until the ``LambdaDeploymentPackager`` runs.  Therefore, the Resource
Builder stage can record the fact that it knows that a
``models.DeploymentPackage`` is needed, but use
``models.Placeholder.BUILD_STAGE`` for the value of the filename.  The enum
values aren't strictly necessary, they just add clarity about when this value
is expected to be filled in.  These could also just be set to ``None`` and be
of type ``Optional[T]``.


Execution Plan Stage
====================

This stage takes the ordered list of resources and figures out what AWS API
calls we have to make.  For example, if a resource doesn't exist at all, we'll
need to make a ``create_*`` call.  If the resource exists, we may need to make
a series of ``update_*`` calls.  If the resource exists and is already up to
date, we might not need to make any calls at all.  The output of this stage is
a list of ``APICall`` objects.  This stage doesn't actually make the mutating
API calls, it only figures out what calls we should make.  This stage will
typically only make ``describe/list`` AWS calls.


The Executor
============

This takes the list of ``APICall`` objects from the previous stage and finally
executes them.  It also manages taking the output of API calls and storing them
in variables so they can be referenced in subsequent ``APICall`` objects (see
the ``Variable`` class to see how this is used).  For example, if a lambda
function needs the ``role_arn`` that's the result of a previous ``create_role``
API call, a ``Variable`` object is used to forward this information.

The executor also records these variables with their associated resources so a
``deployed.json`` file can be written to disk afterwards.  An ``APICall``
takes an optional resource object when it's created whose ``resource_name``
is used as the key in the ``deployed.json`` dictionary.


"""
# pylint: disable=too-many-lines
import json
import textwrap
import socket
import logging

import botocore.exceptions
from botocore.vendored.requests import ConnectionError as RequestsConnectionError
from botocore.session import Session  # noqa
from typing import Optional, Dict, List, Any, Type, cast  # noqa

from chalice.config import Config  # noqa
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
from chalice.deploy.executor import BaseExecutor  # noqa
from chalice.deploy.executor import Executor
from chalice.deploy.executor import DisplayOnlyExecutor
from chalice.deploy.packager import PipRunner
from chalice.deploy.packager import SubprocessPip
from chalice.deploy.packager import DependencyBuilder as PipDependencyBuilder
from chalice.deploy.packager import LambdaDeploymentPackager
from chalice.deploy.packager import AppOnlyDeploymentPackager
from chalice.deploy.packager import LayerDeploymentPackager
from chalice.deploy.packager import BaseLambdaDeploymentPackager  # noqa
from chalice.deploy.packager import EmptyPackageError
from chalice.deploy.planner import PlanStage
from chalice.deploy.planner import RemoteState
from chalice.deploy.planner import NoopPlanner
from chalice.deploy.swagger import TemplatedSwaggerGenerator
from chalice.deploy.swagger import SwaggerGenerator  # noqa
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
            f'ERROR - {where}, received the following error:'
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
                f'While sending your chalice handler code to Lambda to '
                f'function "{error.context.function_name}"'
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
        # To get the underlying error that raised the
        # requests.ConnectionError it is required to go down two levels of
        # arguments to get the underlying exception. The instantiation of
        # one of these exceptions looks like this:
        #
        # requests.ConnectionError(
        #     urllib3.exceptions.ProtocolError(
        #         'Connection aborted.', <SomeException>)
        # )
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
        suggestion: Optional[str] = None
        if isinstance(error, DeploymentPackageTooLargeError):
            suggestion = (
                'To avoid this error, decrease the size of your chalice '
                'application by removing code or removing '
                'dependencies from your chalice application.'
            )
            deployment_size = error.context.deployment_size
            if deployment_size > MAX_LAMBDA_DEPLOYMENT_SIZE:
                size_warning = (
                    f'This is likely because the deployment package is {self._get_mb(deployment_size)}. '
                    f'Lambda only allows deployment packages that are {self._get_mb(MAX_LAMBDA_DEPLOYMENT_SIZE)} or '
                    'less in size.'
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
        return f'{float(value) / (1024 ** 2):.1f} MB'


def create_plan_only_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, DisplayOnlyExecutor, NoopResultsRecorder)


def create_default_deployer(session: Session, config: Config, ui: UI) -> 'Deployer':
    return _create_deployer(session, config, ui, Executor, ResultsRecorder)


def _create_deployer(
    session: Session,
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
    pip_runner = PipRunner(
        pip=SubprocessPip(osutils=osutils),
        osutils=osutils
    )
    dependency_builder = PipDependencyBuilder(
        osutils=osutils,
        pip_runner=pip_runner
    )
    deployment_packager: Optional['BaseDeployStep'] = None
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


class Deployer:
    BACKEND_NAME = 'api'

    def __init__(
        self,
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
        # Rebuild dependencies in case the build stage modified
        # the app graph.
        resources = self._deps_builder.build_dependencies(application)
        plan = self._plan_stage.execute(resources)
        self._sweeper.execute(plan, config)
        self._executor.execute(plan)
        deployed_values: Dict[str, Any] = {
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


class BaseDeployStep:
    def handle(self, config: Config, resource: models.Model) -> None:
        name = f'handle_{resource.__class__.__name__.lower()}'
        handler = getattr(self, name, None)
        if handler is not None:
            handler(config, resource)


class InjectDefaults(BaseDeployStep):
    def __init__(
        self,
        lambda_timeout: int = DEFAULT_LAMBDA_TIMEOUT,
        lambda_memory_size: int = DEFAULT_LAMBDA_MEMORY_SIZE,
        tls_version: str = DEFAULT_TLS_VERSION
    ) -> None:
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
    # If we're creating a layer for non-app code there's two different
    # packagers we need.  One for the Lambda functions (app code) and
    # one for the Lambda layer (requirements.txt + vendor).
    def __init__(
        self,
        lambda_packager: BaseLambdaDeploymentPackager,
        layer_packager: BaseLambdaDeploymentPackager,
    ) -> None:
        self._lambda_packager = lambda_packager
        self._layer_packager = layer_packager

    def handle_lambdafunction(self, config: Config, resource: models.LambdaFunction) -> None:
        if isinstance(resource.deployment_package.filename, models.Placeholder):
            zip_filename = self._lambda_packager.create_deployment_package(
                config.project_dir, config.lambda_python_version
            )
            resource.deployment_package.filename = zip_filename
        if resource.managed_layer is not None and \
                resource.managed_layer.is_empty:
            # Lambda doesn't allow us to create an empty layer so if we've
            # tried to create the deployment package and determined it's empty
            # we should remove the managed layer from the model entirely so
            # downstream consumers don't have to worry about it.
            resource.managed_layer = None

    def handle_lambdalayer(self, config: Config, resource: models.LambdaLayer) -> None:
        if isinstance(resource.deployment_package.filename, models.Placeholder):
            try:
                zip_filename = self._layer_packager.create_deployment_package(
                    config.project_dir, config.lambda_python_version
                )
                resource.deployment_package.filename = zip_filename
            except EmptyPackageError:
                # An empty package is not valid in Lambda.  There's
                # no point in trying to represent it in the model
                # because nothing downstream (planner, SAM/tf) can
                # consume as a useful entity.
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
        self._sqs_policy_injected: bool = False
        self._kinesis_policy_injected: bool = False
        self._ddb_policy_injected: bool = False

    def handle_sqseventsource(self, config: Config, resource: models.SQSEventSource) -> None:
        # The sqs integration works by polling for
        # available records so the lambda function needs
        # permission to call sqs.
        role = resource.lambda_function.role
        if not self._sqs_policy_injected and \
                self._needs_policy_injected(role):
            # mypy can't follow the type narrowing from
            # _needs_policy_injected so we're working around
            # that by explicitly casting the role.
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document,
                                        SQS_EVENT_SOURCE_POLICY.copy())
            self._sqs_policy_injected = True

    def handle_kinesiseventsource(self, config: Config, resource: models.KinesisEventSource) -> None:
        role = resource.lambda_function.role
        if not self._kinesis_policy_injected and \
                self._needs_policy_injected(role):
            # See comment in handle_kinesiseventsource about this cast.
            role = cast(models.ManagedIAMRole, role)
            document = cast(Dict[str, Any], role.policy.document)
            self._inject_trigger_policy(document,
                                        KINESIS_EVENT_SOURCE_POLICY.copy())
            self._kinesis_policy_injected = True

    def handle_dynamodbeventsource(self, config: Config, resource: models.DynamoDBEventSource) -> None:
        role = resource.lambda_function.role
        if not self._ddb_policy_injected and \
                self._needs_policy_injected(role):
            # See comment in handle_kinesiseventsource about this cast.
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
        self._policy_injected: bool = False

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
            isinstance(role.policy, models.AutoGenIAMPolicy) and
            not isinstance(role.policy.document, models.Placeholder)):
            self._inject_policy(
                role.policy.document,
                POST_TO_WEBSOCKET_CONNECTION_POLICY.copy())
        self._policy_injected = True

    def _inject_policy(self, document: Dict[str, Any], policy: Dict[str, Any]) -> None:
        document['Statement'].append(policy)


class PolicyGenerator(BaseDeployStep):
    def __init__(self, policy_gen: AppPolicyGenerator, osutils: OSUtils) -> None:
        self._policy_gen = policy_gen
        self._osutils = osutils

    def _read_document_from_file(self, filename: str) -> Dict[str, Any]:
        try:
            return json.loads(self._osutils.get_file_contents(filename))
        except IOError as e:
            raise RuntimeError(f"Unable to load IAM policy file {filename}: {e}")

    def handle_filebasediampolicy(self, config: Config, resource: models.FileBasedIAMPolicy) -> None:
        resource.document = self._read_document_from_file(resource.filename)

    def handle_restapi(self, config: Config, resource: models.RestAPI) -> None:
        if resource.policy and isinstance(
                resource.policy, models.FileBasedIAMPolicy):
            resource.policy.document = self._read_document_from_file(
                resource.policy.filename)

    def handle_autogeniampolicy(self, config: Config, resource: models.AutoGenIAMPolicy) -> None:
        if isinstance(resource.document, models.Placeholder):
            policy = self._policy_gen.generate_policy(config)
            if models.RoleTraits.VPC_NEEDED in resource.traits:
                policy['Statement'].append(VPC_ATTACH_POLICY)
            resource.document = policy


class BuildStage:
    def __init__(self, steps: List[BaseDeployStep]) -> None:
        self._steps = steps

    def execute(self, config: Config, resources: List[models.Model]) -> None:
        for resource in resources:
            for step in self._steps:
                step.handle(config, resource)


class ResultsRecorder:
    def __init__(self, osutils: OSUtils) -> None:
        self._osutils = osutils

    def record_results(self, results: Any, chalice_stage_name: str, project_dir: str) -> None:
        deployed_dir = self._osutils.joinpath(
            project_dir, '.chalice', 'deployed')
        deployed_filename = self._osutils.joinpath(
            deployed_dir, f'{chalice_stage_name}.json')
        if not self._osutils.directory_exists(deployed_dir):
            self._osutils.makedirs(deployed_dir)
        serialized = serialize_to_json(results)
        self._osutils.set_file_contents(
            filename=deployed_filename,
            contents=serialized,
            binary=False
        )


class NoopResultsRecorder(ResultsRecorder):
    def record_results(self, results: Any, chalice_stage_name: str, project_dir: str) -> None:
        return None


class DeploymentReporter:
    # We want the API URLs to be displayed last.
    _SORT_ORDER = {
        'rest_api': 100,
        'websocket_api': 100,
        'domain_name': 100
    }
    # The default is chosen to sort before the rest_api
    _DEFAULT_ORDERING = 50

    def __init__(self, ui: UI) -> None:
        self._ui = ui

    def generate_report(self, deployed_values: Dict[str, Any]) -> str:
        report: List[str] = [
            'Resources deployed:',
        ]
        ordered = sorted(
            deployed_values['resources'],
            key=lambda x: self._SORT_ORDER.get(x['resource_type'],
                                               self._DEFAULT_ORDERING))
        for resource in ordered:
            getattr(self, f'_report_{resource["resource_type"]}',
                    self._default_report)(resource, report)
        report.append('')
        return '\n'.join(report)

    def _report_rest_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append(f'  - Rest API URL: {resource["rest_api_url"]}')

    def _report_websocket_api(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append(
            f'  - Websocket API URL: {resource["websocket_api_url"]}')

    def _report_domain_name(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append(
            '  - Custom domain name:\n'
            f'      HostedZoneId: {resource["hosted_zone_id"]}\n'
            f'      AliasDomainName: {resource["alias_domain_name"]}'
        )

    def _report_lambda_function(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append(f'  - Lambda ARN: {resource["lambda_arn"]}')

    def _report_lambda_layer(self, resource: Dict[str, Any], report: List[str]) -> None:
        report.append(f'  - Lambda Layer ARN: {resource["layer_version_arn"]}')

    def _default_report(self, resource: Dict[str, Any], report: List[str]) -> None:
        # The default behavior is to not report a resource.  This
        # cuts down on the output verbosity.
        pass

    def display_report(self, deployed_values: Dict[str, Any]) -> None:
        report = self.generate_report(deployed_values)
        self._ui.write(report)
