from unittest import mock
from dataclasses import replace, dataclass
from typing import Tuple, List, Dict, Any, Optional, Union, cast

import pytest

from chalice.awsclient import TypedAWSClient, ResourceDoesNotExistError
from chalice.deploy import models
from chalice.config import DeployedResources
from chalice.utils import OSUtils
from chalice.deploy.planner import PlanStage, Variable, RemoteState, \
    KeyDataVariable
from chalice.deploy.planner import StringFormat
from chalice.deploy.models import APICall
from chalice.deploy.sweeper import ResourceSweeper


def create_function_resource(name: str,
                           function_name: Optional[str] = None,
                           environment_variables: Optional[Dict[str, str]] = None,
                           runtime: str = 'python2.7',
                           handler: str = 'app.app',
                           tags: Optional[Dict[str, str]] = None,
                           timeout: int = 60,
                           memory_size: int = 128,
                           deployment_package: Optional[models.DeploymentPackage] = None,
                           role: Optional[Union[models.PreCreatedIAMRole, models.ManagedIAMRole]] = None,
                           layers: Optional[List[str]] = None,
                           managed_layer: Optional[models.LambdaLayer] = None) -> models.LambdaFunction:
    if function_name is None:
        function_name = 'appname-dev-%s' % name
    if environment_variables is None:
        environment_variables = {}
    if tags is None:
        tags = {}
    if deployment_package is None:
        deployment_package = models.DeploymentPackage(filename='foo')
    if role is None:
        role = models.PreCreatedIAMRole(role_arn='role:arn')
    return models.LambdaFunction(
        resource_name=name,
        function_name=function_name,
        environment_variables=environment_variables,
        runtime=runtime,
        handler=handler,
        tags=tags,
        timeout=timeout,
        memory_size=memory_size,
        xray=None,
        deployment_package=deployment_package,
        role=role,
        security_group_ids=[],
        subnet_ids=[],
        layers=layers if layers is not None else [],
        reserved_concurrency=None,
        managed_layer=managed_layer,
    )


def create_managed_layer() -> models.LambdaLayer:
    layer = models.LambdaLayer(
        resource_name='layer',
        layer_name='bar',
        runtime='python2.7',
        deployment_package=models.DeploymentPackage(
            filename='foo')
    )
    return layer


def create_api_mapping() -> models.APIMapping:
    return models.APIMapping(
        resource_name='api_mapping',
        mount_path='(none)',
        api_gateway_stage='dev'
    )


def create_http_domain_name() -> models.DomainName:
    return models.DomainName(
        protocol=models.APIType.HTTP,
        resource_name='api_gateway_custom_domain',
        domain_name='example.com',
        tls_version=models.TLSVersion.TLS_1_0,
        api_mapping=create_api_mapping(),
        certificate_arn='certificate_arn',
    )


def create_websocket_domain_name() -> models.DomainName:
    return models.DomainName(
        protocol=models.APIType.WEBSOCKET,
        resource_name='websocket_api_custom_domain',
        domain_name='example.com',
        tls_version=models.TLSVersion.TLS_1_0,
        api_mapping=create_api_mapping(),
        certificate_arn='certificate_arn',
    )


@pytest.fixture
def no_deployed_values() -> DeployedResources:
    return DeployedResources({'resources': [], 'schema_version': '2.0'})


class FakeConfig(object):
    def __init__(self, deployed_values: Dict[str, Any]) -> None:
        self._deployed_values = deployed_values
        self.chalice_stage = 'dev'
        self.api_gateway_stage = 'dev'

    def deployed_resources(self, chalice_stage_name: str) -> DeployedResources:
        return DeployedResources(self._deployed_values)


class InMemoryRemoteState(object):
    def __init__(self, known_resources: Optional[Dict[Tuple[str, str], Any]] = None) -> None:
        if known_resources is None:
            known_resources = {}
        self.known_resources = known_resources
        self.deployed_values: Dict[str, Any] = {}

    def resource_exists(self, resource: models.ManagedModel, *args: Any) -> bool:
        if resource.resource_type == 'api_mapping':
            return (
                (resource.resource_type, resource.mount_path)
                in self.known_resources
            )
        return (
            (resource.resource_type, resource.resource_name)
            in self.known_resources
        )

    def get_remote_model(self, resource: models.ManagedModel) -> Any:
        key = (resource.resource_type, resource.resource_name)
        return self.known_resources.get(key)

    def declare_resource_exists(self, resource: models.ManagedModel, **deployed_values: Any) -> None:
        key = (resource.resource_type, resource.resource_name)
        self.known_resources[key] = resource
        if deployed_values:
            deployed_values['name'] = resource.resource_name
            self.deployed_values[resource.resource_name] = deployed_values
            if resource.resource_type == 'domain_name':
                key = (resource.api_mapping.resource_type,
                       resource.api_mapping.mount_path)
                self.known_resources[key] = resource

    def declare_no_resources_exists(self) -> None:
        self.known_resources = {}

    def resource_deployed_values(self, resource: models.ManagedModel) -> Dict[str, Any]:
        return self.deployed_values[resource.resource_name]


class BasePlannerTests(object):
    def setup_method(self) -> None:
        self.osutils = mock.Mock(spec=OSUtils)
        self.remote_state = InMemoryRemoteState()
        self.last_plan: Optional[models.Plan] = None

    def assert_apicall_equals(self, expected: models.APICall, actual_api_call: models.APICall) -> None:
        assert isinstance(expected, models.APICall)
        assert isinstance(actual_api_call, models.APICall)
        assert expected.method_name == actual_api_call.method_name
        assert expected.params == actual_api_call.params

    def determine_plan(self, resource: models.ManagedModel) -> List[models.Instruction]:
        planner = PlanStage(self.remote_state, self.osutils)
        self.last_plan = planner.execute([resource])
        return self.last_plan.instructions

    def filter_api_calls(self, plan: List[models.Instruction]) -> List[models.APICall]:
        api_calls: List[models.APICall] = []
        for instruction in plan:
            if isinstance(instruction, models.APICall):
                api_calls.append(instruction)
        return api_calls

    def assert_recorded_values(self, plan: List[models.Instruction],
                              resource_type: str, resource_name: str,
                              expected_mapping: Dict[str, Any]) -> None:
        actual: Dict[str, Any] = {}
        for step in plan:
            if isinstance(step, models.RecordResourceValue):
                actual[step.name] = step.value
            elif isinstance(step, models.RecordResourceVariable):
                actual[step.name] = Variable(step.variable_name)
        assert actual == expected_mapping


class TestPlanManagedRole(BasePlannerTests):
    def test_can_plan_for_iam_role_creation(self) -> None:
        self.remote_state.declare_no_resources_exists()
        resource = models.ManagedIAMRole(
            resource_name='default-role',
            role_name='myrole',
            trust_policy={'trust': 'policy'},
            policy=models.AutoGenIAMPolicy(document={'iam': 'policy'}),
        )
        plan = self.determine_plan(resource)
        expected = models.APICall(
            method_name='create_role',
            params={'name': 'myrole',
                    'trust_policy': Variable('lambda_trust_policy'),
                    'policy': {'iam': 'policy'}},
        )
        self.assert_apicall_equals(plan[4], expected)
        assert list(self.last_plan.messages.values()) == [
            'Creating IAM role: myrole\n'
        ]

    def test_can_create_plan_for_filebased_role(self) -> None:
        self.remote_state.declare_no_resources_exists()
        resource = models.ManagedIAMRole(
            resource_name='default-role',
            role_name='myrole',
            trust_policy={'trust': 'policy'},
            policy=models.FileBasedIAMPolicy(
                filename='foo.json', document={'iam': 'policy'}),
        )
        plan = self.determine_plan(resource)
        expected = models.APICall(
            method_name='create_role',
            params={'name': 'myrole',
                    'trust_policy': Variable('lambda_trust_policy'),
                    'policy': {'iam': 'policy'}},
        )
        self.assert_apicall_equals(plan[4], expected)
        assert list(self.last_plan.messages.values()) == [
            'Creating IAM role: myrole\n'
        ]

    def test_can_update_managed_role(self) -> None:
        role = models.ManagedIAMRole(
            resource_name='resource_name',
            role_name='myrole',
            trust_policy={},
            policy=models.AutoGenIAMPolicy(document={'role': 'policy'}),
        )
        self.remote_state.declare_resource_exists(
            role, role_arn='myrole:arn')
        plan = self.determine_plan(role)
        assert plan[0] == models.StoreValue(
            name='myrole_role_arn', value='myrole:arn')
        self.assert_apicall_equals(
            plan[1],
            models.APICall(
                method_name='put_role_policy',
                params={'role_name': 'myrole',
                        'policy_name': 'myrole',
                        'policy_document': {'role': 'policy'}},
            )
        )
        assert plan[-2].variable_name == 'myrole_role_arn'
        assert plan[-1].value == 'myrole'
        assert list(self.last_plan.messages.values()) == [
            'Updating policy for IAM role: myrole\n'
        ]

    def test_can_update_file_based_policy(self) -> None:
        role = models.ManagedIAMRole(
            resource_name='resource_name',
            role_name='myrole',
            trust_policy={},
            policy=models.FileBasedIAMPolicy(
                filename='foo.json',
                document={'iam': 'policy'}),
        )
        self.remote_state.declare_resource_exists(role, role_arn='myrole:arn')
        plan = self.determine_plan(role)
        assert plan[0] == models.StoreValue(
            name='myrole_role_arn', value='myrole:arn')
        self.assert_apicall_equals(
            plan[1],
            models.APICall(
                method_name='put_role_policy',
                params={'role_name': 'myrole',
                        'policy_name': 'myrole',
                        'policy_document': {'iam': 'policy'}},
            )
        )

    def test_no_update_for_non_managed_role(self) -> None:
        role = models.PreCreatedIAMRole(role_arn='role:arn')
        plan = self.determine_plan(role)
        assert plan == []


class TestPlanCreateUpdateAPIMapping(BasePlannerTests):
    def test_can_create_api_mapping(self, lambda_function: models.LambdaFunction) -> None:
        rest_api = models.RestAPI(
            resource_name='rest_api',
            swagger_doc={'swagger': '2.0'},
            minimum_compression='',
            api_gateway_stage='api',
            endpoint_type='EDGE',
            lambda_function=lambda_function,
            domain_name=create_http_domain_name()
        )

        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(rest_api)
        params = {
            'domain_name': rest_api.domain_name.domain_name,
            'path_key': '(none)',
            'stage': 'dev',
            'api_id': Variable('rest_api_id')
        }
        expected = [
            models.APICall(
                method_name='create_base_path_mapping',
                params=params,
                output_var='base_path_mapping'
            ),
        ]
        # Create api mapping.
        self.assert_apicall_equals(plan[-3], expected[0])
        msg = 'Creating api mapping: /\n'
        assert list(self.last_plan.messages.values())[-1] == msg

    def test_can_create_websocket_api_mapping_with_path(self) -> None:
        domain_name = create_websocket_domain_name()
        domain_name.api_mapping.mount_path = 'path-key'

        connect_function = create_function_resource(
            'function_name_connect')
        message_function = create_function_resource(
            'function_name_message')
        disconnect_function = create_function_resource(
            'function_name_disconnect')

        websocket_api = models.WebsocketAPI(
            resource_name='websocket_api',
            name='app-dev-websocket-api',
            api_gateway_stage='api',
            routes=['$connect', '$default', '$disconnect'],
            connect_function=connect_function,
            message_function=message_function,
            disconnect_function=disconnect_function,
            domain_name=domain_name
        )

        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(websocket_api)
        params = {
            'domain_name': domain_name.domain_name,
            'path_key': 'path-key',
            'stage': 'dev',
            'api_id': Variable('websocket_api_id')
        }
        expected = [
            models.APICall(
                method_name='create_api_mapping',
                params=params,
                output_var='api_mapping'
            ),
        ]
        # create api mapping
        self.assert_apicall_equals(plan[-3], expected[0])
        msg = 'Creating api mapping: /path-key\n'
        assert list(self.last_plan.messages.values())[-1] == msg

    def test_store_api_mapping_if_already_exists(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        domain_name.api_mapping.mount_path = 'test-path'
        rest_api = models.RestAPI(
            resource_name='rest_api',
            swagger_doc={'swagger': '2.0'},
            minimum_compression='',
            api_gateway_stage='api',
            endpoint_type='EDGE',
            lambda_function=lambda_function,
            domain_name=domain_name
        )

        deployed_value = {
            'name': 'api_gateway_custom_domain',
            'resource_type': 'domain_name',
            'hosted_zone_id': 'hosted_zone_id',
            'certificate_arn': 'certificate_arn',
            'security_policy': 'TLS_1_0',
            'domain_name': 'example.com',
            'api_mapping': [
                {
                    'key': '/test-path'
                },
                {
                    'key': '/test-path-2'
                }
            ]
        }

        self.remote_state.declare_resource_exists(domain_name,
                                                  **deployed_value)
        plan = self.determine_plan(rest_api)
        expected = [
            models.StoreMultipleValue(
                name='rest_api_mapping',
                value=[{
                    'key': '/test-path'
                }]
            )
        ]
        assert plan[-2].name == expected[0].name
        assert plan[-2].value == expected[0].value
        assert isinstance(expected[0], models.StoreMultipleValue)
        assert isinstance(plan[-2], models.StoreMultipleValue)

    def test_store_api_mapping_none_if_already_exists(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        domain_name.api_mapping.mount_path = '(none)'
        rest_api = models.RestAPI(
            resource_name='rest_api',
            swagger_doc={'swagger': '2.0'},
            minimum_compression='',
            api_gateway_stage='api',
            endpoint_type='EDGE',
            lambda_function=lambda_function,
            domain_name=domain_name
        )

        deployed_value = {
            'name': 'api_gateway_custom_domain',
            'resource_type': 'domain_name',
            'hosted_zone_id': 'hosted_zone_id',
            'certificate_arn': 'certificate_arn',
            'security_policy': 'TLS_1_0',
            'domain_name': 'example.com',
            'api_mapping': [
                {
                    'key': '/'
                },
            ]
        }

        self.remote_state.declare_resource_exists(domain_name,
                                                  **deployed_value)
        plan = self.determine_plan(rest_api)
        expected = [
            models.StoreMultipleValue(
                name='rest_api_mapping',
                value=[{
                    'key': '/'
                }]
            )
        ]
        assert plan[-2].name == expected[0].name
        assert plan[-2].value == expected[0].value
        assert isinstance(expected[0], models.StoreMultipleValue)
        assert isinstance(plan[-2], models.StoreMultipleValue)


class TestPlanCreateUpdateDomainName(BasePlannerTests):
    def test_can_create_domain_name(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        rest_api = models.RestAPI(
            resource_name='rest_api',
            swagger_doc={'swagger': '2.0'},
            minimum_compression='',
            api_gateway_stage='api',
            endpoint_type='EDGE',
            lambda_function=lambda_function,
            domain_name=domain_name
        )

        params = {
            'protocol': domain_name.protocol.value,
            'domain_name': domain_name.domain_name,
            'security_policy': domain_name.tls_version.value,
            'certificate_arn': domain_name.certificate_arn,
            'endpoint_type': 'EDGE',
            'tags': None
        }
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(rest_api)
        expected = [
            models.APICall(
                method_name='create_domain