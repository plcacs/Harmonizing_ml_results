from unittest import mock
from dataclasses import replace, dataclass
from typing import Tuple, Dict, List, Optional, Any, Union
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
    role: Optional[models.IAMRole] = None,
    layers: Optional[List[str]] = None,
    managed_layer: Optional[models.LambdaLayer] = None
) -> models.LambdaFunction:
    if function_name is None:
        function_name = f'appname-dev-{name}'
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
        layers=layers,
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
    def __init__(self, known_resources: Optional[Dict[Tuple[str, str], Any]] = None) -> None:
        if known_resources is None:
            known_resources = {}
        self.known_resources = known_resources
        self.deployed_values: Dict[str, Any] = {}

    def resource_exists(self, resource: models.ManagedModel, *args: Any) -> bool:
        if resource.resource_type == 'api_mapping':
            return (resource.resource_type, resource.mount_path) in self.known_resources
        return (resource.resource_type, resource.resource_name) in self.known_resources

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

    def assert_apicall_equals(self, expected: models.APICall, actual_api_call: models.APICall) -> None:
        assert isinstance(expected, models.APICall)
        assert isinstance(actual_api_call, models.APICall)
        assert expected.method_name == actual_api_call.method_name
        assert expected.params == actual_api_call.params

    def determine_plan(self, resource: models.ManagedModel) -> List[Any]:
        planner = PlanStage(self.remote_state, self.osutils)
        self.last_plan = planner.execute([resource])
        return self.last_plan.instructions

    def filter_api_calls(self, plan: List[Any]) -> List[models.APICall]:
        api_calls: List[models.APICall] = []
        for instruction in plan:
            if isinstance(instruction, models.APICall):
                api_calls.append(instruction)
        return api_calls

    def assert_recorded_values(
        self,
        plan: List[Any],
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

# ... (continue with the rest of the classes and methods in the same way)
