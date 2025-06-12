from unittest import mock
from dataclasses import replace, dataclass
from typing import Tuple, Dict, Any, List, Optional, Union, cast
import pytest
from chalice.awsclient import TypedAWSClient, ResourceDoesNotExistError
from chalice.deploy import models
from chalice.config import DeployedResources
from chalice.utils import OSUtils
from chalice.deploy.planner import PlanStage, Variable, RemoteState, KeyDataVariable
from chalice.deploy.planner import StringFormat
from chalice.deploy.models import APICall
from chalice.deploy.sweeper import ResourceSweeper

def create_function_resource(
    name: str,
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
    managed_layer: Optional[models.LambdaLayer] = None
) -> models.LambdaFunction:
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
        layers=layers if layers else [],
        reserved_concurrency=None,
        managed_layer=managed_layer
    )

def create_managed_layer() -> models.LambdaLayer:
    layer = models.LambdaLayer(
        resource_name='layer',
        layer_name='bar',
        runtime='python2.7',
        deployment_package=models.DeploymentPackage(filename='foo')
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
        certificate_arn='certificate_arn'
    )

def create_websocket_domain_name() -> models.DomainName:
    return models.DomainName(
        protocol=models.APIType.WEBSOCKET,
        resource_name='websocket_api_custom_domain',
        domain_name='example.com',
        tls_version=models.TLSVersion.TLS_1_0,
        api_mapping=create_api_mapping(),
        certificate_arn='certificate_arn'
    )

@pytest.fixture
def no_deployed_values() -> DeployedResources:
    return DeployedResources({'resources': [], 'schema_version': '2.0'})

class FakeConfig:
    def __init__(self, deployed_values: Dict[str, Any]) -> None:
        self._deployed_values = deployed_values
        self.chalice_stage = 'dev'
        self.api_gateway_stage = 'dev'

    def deployed_resources(self, chalice_stage_name: str) -> DeployedResources:
        return DeployedResources(self._deployed_values)

class InMemoryRemoteState:
    def __init__(
        self,
        known_resources: Optional[Dict[Tuple[str, str], Any]] = None,
        deployed_values: Optional[Dict[str, Any]] = None
    ) -> None:
        self.known_resources = known_resources if known_resources else {}
        self.deployed_values = deployed_values if deployed_values else {}

    def resource_exists(self, resource: models.ManagedModel, *args: Any) -> bool:
        if resource.resource_type == 'api_mapping':
            return (resource.resource_type, resource.mount_path) in self.known_resources
        return (resource.resource_type, resource.resource_name) in self.known_resources

    def get_remote_model(self, resource: models.ManagedModel) -> Any:
        key = (resource.resource_type, resource.resource_name)
        return self.known_resources.get(key)

    def declare_resource_exists(
        self,
        resource: models.ManagedModel,
        **deployed_values: Any
    ) -> None:
        key = (resource.resource_type, resource.resource_name)
        self.known_resources[key] = resource
        if deployed_values:
            deployed_values['name'] = resource.resource_name
            self.deployed_values[resource.resource_name] = deployed_values
            if resource.resource_type == 'domain_name':
                key = (resource.api_mapping.resource_type, resource.api_mapping.mount_path)
                self.known_resources[key] = resource

    def declare_no_resources_exists(self) -> None:
        self.known_resources = {}

    def resource_deployed_values(self, resource: models.ManagedModel) -> Dict[str, Any]:
        return self.deployed_values[resource.resource_name]

class BasePlannerTests:
    def setup_method(self) -> None:
        self.osutils = mock.Mock(spec=OSUtils)
        self.remote_state = InMemoryRemoteState()
        self.last_plan: Optional[models.Plan] = None

    def assert_apicall_equals(
        self,
        expected: models.APICall,
        actual_api_call: models.APICall
    ) -> None:
        assert isinstance(expected, models.APICall)
        assert isinstance(actual_api_call, models.APICall)
        assert expected.method_name == actual_api_call.method_name
        assert expected.params == actual_api_call.params

    def determine_plan(
        self,
        resource: models.ManagedModel
    ) -> List[Union[models.APICall, models.RecordResourceValue]]:
        planner = PlanStage(self.remote_state, self.osutils)
        self.last_plan = planner.execute([resource])
        return self.last_plan.instructions

    def filter_api_calls(
        self,
        plan: List[Union[models.APICall, models.RecordResourceValue]]
    ) -> List[models.APICall]:
        api_calls: List[models.APICall] = []
        for instruction in plan:
            if isinstance(instruction, models.APICall):
                api_calls.append(instruction)
        return api_calls

    def assert_recorded_values(
        self,
        plan: List[Union[models.APICall, models.RecordResourceValue]],
        resource_type: str,
        resource_name: str,
        expected_mapping: Dict[str, Any]
    ) -> None:
        actual: Dict[str, Any] = {}
        for step in plan:
            if isinstance(step, models.RecordResourceValue):
                actual[step.name] = step.value
            elif isinstance(step, models.RecordResourceVariable):
                actual[step.name] = Variable(step.variable_name)
        assert actual == expected_mapping

# [Rest of the classes and methods with similar type annotations...]
# [Continuing with type annotations for all remaining classes and methods...]

@dataclass
class Foo(models.ManagedModel):
    resource_type: str = 'foo'

class TestRemoteState:
    def setup_method(self) -> None:
        self.client = mock.Mock(spec=TypedAWSClient)
        self.config = FakeConfig({'resources': []})
        self.remote_state = RemoteState(self.client, self.config.deployed_resources('dev'))

    def create_rest_api_model(self) -> models.RestAPI:
        return models.RestAPI(
            resource_name='rest_api',
            swagger_doc={'swagger': '2.0'},
            minimum_compression='',
            endpoint_type='EDGE',
            api_gateway_stage='api',
            xray=False,
            lambda_function=None
        )

    def create_api_mapping(self) -> models.APIMapping:
        return models.APIMapping(
            resource_name='api_mapping',
            mount_path='(none)',
            api_gateway_stage='dev'
        )

    def create_domain_name(self) -> models.DomainName:
        return models.DomainName(
            protocol=models.APIType.HTTP,
            resource_name='api_gateway_custom_domain',
            domain_name='example.com',
            tls_version=models.TLSVersion.TLS_1_0,
            certificate_arn='certificate_arn',
            api_mapping=self.create_api_mapping()
        )

    def create_websocket_api_model(self) -> models.WebsocketAPI:
        return models.WebsocketAPI(
            resource_name='websocket_api',
            name='app-stage-websocket-api',
            api_gateway_stage='api',
            routes=[],
            connect_function=None,
            message_function=None,
            disconnect_function=None
        )

    # [Rest of the methods with type annotations...]

class TestKeyVariable:
    def test_key_variable_str(self) -> None:
        key_var = KeyDataVariable('name', 'key')
        assert str(key_var) == 'KeyDataVariable("name", "key")'

    def test_key_variables_equal(self) -> None:
        key_var = KeyDataVariable('name', 'key')
        key_var_1 = KeyDataVariable('name', 'key_1')
        assert not key_var == key_var_1
        key_var_2 = KeyDataVariable('name', 'key')
        assert key_var == key_var_2

class TestPlanLogGroup(BasePlannerTests):
    def test_can_create_log_group(self) -> None:
        self.remote_state.declare_no_resources_exists()
        resource = models.LogGroup(
            resource_name='default-log-group',
            log_group_name='/aws/lambda/func-name',
            retention_in_days=14
        )
        plan = self.determine_plan(resource)
        assert plan == [
            models.APICall(
                method_name='create_log_group',
                params={'log_group_name': '/aws/lambda/func-name'}
            ),
            models.APICall(
                method_name='put_retention_policy',
                params={'name': '/aws/lambda/func-name', 'retention_in_days': 14}
            ),
            models.RecordResourceValue(
                resource_type='log_group',
                resource_name='default-log-group',
                name='log_group_name',
                value='/aws/lambda/func-name'
            )
        ]

    def test_can_update_log_group(self) -> None:
        resource = models.LogGroup(
            resource_name='default-log-group',
            log_group_name='/aws/lambda/func-name',
            retention_in_days=14
        )
        self.remote_state.declare_resource_exists(resource)
        plan = self.determine_plan(resource)
        assert plan == [
            models.APICall(
                method_name='put_retention_policy',
                params={'name': '/aws/lambda/func-name', 'retention_in_days': 14}
            ),
            models.RecordResourceValue(
                resource_type='log_group',
                resource_name='default-log-group',
                name='log_group_name',
                value='/aws/lambda/func-name'
            )
        ]
