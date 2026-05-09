from unittest import mock
from dataclasses import replace, dataclass
from typing import Dict, List, Optional, Tuple, Union
import pytest
from chalice.awsclient import TypedAWSClient, ResourceDoesNotExistError
from chalice.deploy import models
from chalice.config import DeployedResources
from chalice.utils import OSUtils
from chalice.deploy.planner import PlanStage, Variable, RemoteState, KeyDataVariable
from chalice.deploy.planner import StringFormat
from chalice.deploy.models import APICall
from chalice.deploy.sweeper import ResourceSweeper

def create_function_resource(name: str, function_name: Optional[str] = None, environment_variables: Optional[Dict[str, str]] = None, runtime: str = 'python2.7', handler: str = 'app.app', tags: Optional[Dict[str, str]] = None, timeout: int = 60, memory_size: int = 128, deployment_package: Optional[models.DeploymentPackage] = None, role: Optional[models.IAMRole] = None, layers: Optional[List[str]] = None, managed_layer: Optional[models.LambdaLayer] = None) -> models.LambdaFunction:
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
    return models.LambdaFunction(resource_name=name, function_name=function_name, environment_variables=environment_variables, runtime=runtime, handler=handler, tags=tags, timeout=timeout, memory_size=memory_size, xray=None, deployment_package=deployment_package, role=role, security_group_ids=[], subnet_ids=[], layers=layers, reserved_concurrency=None, managed_layer=managed_layer)

def create_managed_layer() -> models.LambdaLayer:
    layer = models.LambdaLayer(resource_name='layer', layer_name='bar', runtime='python2.7', deployment_package=models.DeploymentPackage(filename='foo'))
    return layer

def create_api_mapping() -> models.APIMapping:
    return models.APIMapping(resource_name='api_mapping', mount_path='(none)', api_gateway_stage='dev')

def create_http_domain_name() -> models.DomainName:
    return models.DomainName(protocol=models.APIType.HTTP, resource_name='api_gateway_custom_domain', domain_name='example.com', tls_version=models.TLSVersion.TLS_1_0, api_mapping=create_api_mapping(), certificate_arn='certificate_arn')

def create_websocket_domain_name() -> models.DomainName:
    return models.DomainName(protocol=models.APIType.WEBSOCKET, resource_name='websocket_api_custom_domain', domain_name='example.com', tls_version=models.TLSVersion.TLS_1_0, api_mapping=create_api_mapping(), certificate_arn='certificate_arn')

@pytest.fixture
def no_deployed_values() -> DeployedResources:
    return DeployedResources({'resources': [], 'schema_version': '2.0'})

class FakeConfig(object):
    def __init__(self, deployed_values: DeployedResources):
        self._deployed_values = deployed_values
        self.chalice_stage = 'dev'
        self.api_gateway_stage = 'dev'

    def deployed_resources(self, chalice_stage_name: str) -> DeployedResources:
        return DeployedResources(self._deployed_values)

class InMemoryRemoteState(object):
    def __init__(self, known_resources: Optional[Dict[Tuple[str, str], models.ManagedModel]] = None):
        if known_resources is None:
            known_resources = {}
        self.known_resources = known_resources
        self.deployed_values: Dict[str, Dict[str, str]] = {}

    def resource_exists(self, resource: models.ManagedModel, *args: str) -> bool:
        if resource.resource_type == 'api_mapping':
            return (resource.resource_type, resource.mount_path) in self.known_resources
        return (resource.resource_type, resource.resource_name) in self.known_resources

    def get_remote_model(self, resource: models.ManagedModel) -> Optional[models.ManagedModel]:
        key = (resource.resource_type, resource.resource_name)
        return self.known_resources.get(key)

    def declare_resource_exists(self, resource: models.ManagedModel, **deployed_values: str) -> None:
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

    def resource_deployed_values(self, resource: models.ManagedModel) -> Dict[str, str]:
        return self.deployed_values[resource.resource_name]

class BasePlannerTests(object):
    def setup_method(self) -> None:
        self.osutils = mock.Mock(spec=OSUtils)
        self.remote_state = InMemoryRemoteState()
        self.last_plan = None

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
        api_calls = []
        for instruction in plan:
            if isinstance(instruction, models.APICall):
                api_calls.append(instruction)
        return api_calls

    def assert_recorded_values(self, plan: List[models.Instruction], resource_type: str, resource_name: str, expected_mapping: Dict[str, Union[str, Variable]]) -> None:
        actual = {}
        for step in plan:
            if isinstance(step, models.RecordResourceValue):
                actual[step.name] = step.value
            elif isinstance(step, models.RecordResourceVariable):
                actual[step.name] = Variable(step.variable_name)
        assert actual == expected_mapping

class TestPlanManagedRole(BasePlannerTests):
    def test_can_plan_for_iam_role_creation(self) -> None:
        self.remote_state.declare_no_resources_exists()
        resource = models.ManagedIAMRole(resource_name='default-role', role_name='myrole', trust_policy={'trust': 'policy'}, policy=models.AutoGenIAMPolicy(document={'iam': 'policy'}))
        plan = self.determine_plan(resource)
        expected = models.APICall(method_name='create_role', params={'name': 'myrole', 'trust_policy': Variable('lambda_trust_policy'), 'policy': {'iam': 'policy'}})
        self.assert_apicall_equals(plan[4], expected)
        assert list(self.last_plan.messages.values()) == ['Creating IAM role: myrole\n']

    def test_can_create_plan_for_filebased_role(self) -> None:
        self.remote_state.declare_no_resources_exists()
        resource = models.ManagedIAMRole(resource_name='default-role', role_name='myrole', trust_policy={'trust': 'policy'}, policy=models.FileBasedIAMPolicy(filename='foo.json', document={'iam': 'policy'}))
        plan = self.determine_plan(resource)
        expected = models.APICall(method_name='create_role', params={'name': 'myrole', 'trust_policy': Variable('lambda_trust_policy'), 'policy': {'iam': 'policy'}})
        self.assert_apicall_equals(plan[4], expected)
        assert list(self.last_plan.messages.values()) == ['Creating IAM role: myrole\n']

    def test_can_update_managed_role(self) -> None:
        role = models.ManagedIAMRole(resource_name='resource_name', role_name='myrole', trust_policy={}, policy=models.AutoGenIAMPolicy(document={'role': 'policy'}))
        self.remote_state.declare_resource_exists(role, role_arn='myrole:arn')
        plan = self.determine_plan(role)
        assert plan[0] == models.StoreValue(name='myrole_role_arn', value='myrole:arn')
        self.assert_apicall_equals(plan[1], models.APICall(method_name='put_role_policy', params={'role_name': 'myrole', 'policy_name': 'myrole', 'policy_document': {'role': 'policy'}}))
        assert plan[-2].variable_name == 'myrole_role_arn'
        assert plan[-1].value == 'myrole'
        assert list(self.last_plan.messages.values()) == ['Updating policy for IAM role: myrole\n']

    def test_can_update_file_based_policy(self) -> None:
        role = models.ManagedIAMRole(resource_name='resource_name', role_name='myrole', trust_policy={}, policy=models.FileBasedIAMPolicy(filename='foo.json', document={'iam': 'policy'}))
        self.remote_state.declare_resource_exists(role, role_arn='myrole:arn')
        plan = self.determine_plan(role)
        assert plan[0] == models.StoreValue(name='myrole_role_arn', value='myrole:arn')
        self.assert_apicall_equals(plan[1], models.APICall(method_name='put_role_policy', params={'role_name': 'myrole', 'policy_name': 'myrole', 'policy_document': {'iam': 'policy'}}))

    def test_no_update_for_non_managed_role(self) -> None:
        role = models.PreCreatedIAMRole(role_arn='role:arn')
        plan = self.determine_plan(role)
        assert plan == []

class TestPlanCreateUpdateAPIMapping(BasePlannerTests):
    def test_can_create_api_mapping(self, lambda_function: models.LambdaFunction) -> None:
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='EDGE', lambda_function=lambda_function, domain_name=create_http_domain_name())
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(rest_api)
        params = {'domain_name': rest_api.domain_name.domain_name, 'path_key': '(none)', 'stage': 'dev', 'api_id': Variable('rest_api_id')}
        expected = [models.APICall(method_name='create_base_path_mapping', params=params, output_var='base_path_mapping')]
        self.assert_apicall_equals(plan[-3], expected[0])
        msg = 'Creating api mapping: /\n'
        assert list(self.last_plan.messages.values())[-1] == msg

    def test_can_create_websocket_api_mapping_with_path(self) -> None:
        domain_name = create_websocket_domain_name()
        domain_name.api_mapping.mount_path = 'path-key'
        connect_function = create_function_resource('function_name_connect')
        message_function = create_function_resource('function_name_message')
        disconnect_function = create_function_resource('function_name_disconnect')
        websocket_api = models.WebsocketAPI(resource_name='websocket_api', name='app-dev-websocket-api', api_gateway_stage='api', routes=['$connect', '$default', '$disconnect'], connect_function=connect_function, message_function=message_function, disconnect_function=disconnect_function, domain_name=domain_name)
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(websocket_api)
        params = {'domain_name': domain_name.domain_name, 'path_key': 'path-key', 'stage': 'dev', 'api_id': Variable('websocket_api_id')}
        expected = [models.APICall(method_name='create_api_mapping', params=params, output_var='api_mapping')]
        self.assert_apicall_equals(plan[-3], expected[0])
        msg = 'Creating api mapping: /path-key\n'
        assert list(self.last_plan.messages.values())[-1] == msg

    def test_store_api_mapping_if_already_exists(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        domain_name.api_mapping.mount_path = 'test-path'
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='EDGE', lambda_function=lambda_function, domain_name=domain_name)
        deployed_value = {'name': 'api_gateway_custom_domain', 'resource_type': 'domain_name', 'hosted_zone_id': 'hosted_zone_id', 'certificate_arn': 'certificate_arn', 'security_policy': 'TLS_1_0', 'domain_name': 'example.com', 'api_mapping': [{'key': '/test-path'}, {'key': '/test-path-2'}]}
        self.remote_state.declare_resource_exists(domain_name, **deployed_value)
        plan = self.determine_plan(rest_api)
        expected = [models.StoreMultipleValue(name='rest_api_mapping', value=[{'key': '/test-path'}])]
        assert plan[-2].name == expected[0].name
        assert plan[-2].value == expected[0].value
        assert isinstance(expected[0], models.StoreMultipleValue)
        assert isinstance(plan[-2], models.StoreMultipleValue)

    def test_store_api_mapping_none_if_already_exists(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        domain_name.api_mapping.mount_path = '(none)'
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='EDGE', lambda_function=lambda_function, domain_name=domain_name)
        deployed_value = {'name': 'api_gateway_custom_domain', 'resource_type': 'domain_name', 'hosted_zone_id': 'hosted_zone_id', 'certificate_arn': 'certificate_arn', 'security_policy': 'TLS_1_0', 'domain_name': 'example.com', 'api_mapping': [{'key': '/'}]}
        self.remote_state.declare_resource_exists(domain_name, **deployed_value)
        plan = self.determine_plan(rest_api)
        expected = [models.StoreMultipleValue(name='rest_api_mapping', value=[{'key': '/'}])]
        assert plan[-2].name == expected[0].name
        assert plan[-2].value == expected[0].value
        assert isinstance(expected[0], models.StoreMultipleValue)
        assert isinstance(plan[-2], models.StoreMultipleValue)

class TestPlanCreateUpdateDomainName(BasePlannerTests):
    def test_can_create_domain_name(self, lambda_function: models.LambdaFunction) -> None:
        domain_name = create_http_domain_name()
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='EDGE', lambda_function=lambda_function, domain_name=domain_name)
        params = {'protocol': domain_name.protocol.value, 'domain_name': domain_name.domain_name, 'security_policy': domain_name.tls_version.value, 'certificate_arn': domain_name.certificate_arn, 'endpoint_type': 'EDGE', 'tags': None}
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(rest_api)
        expected = [models.APICall(method_name='create_domain_name', params=params, output_var=domain_name.resource_name)]
        self.assert_apicall_equals(plan[13], expected[0])
        msg = 'Creating custom domain name: example.com\n'
        assert list(self.last_plan.messages.values())[-2] == msg

    def test_can_update_domain_name(self) -> None:
        deployed_value = {'name': 'rest_api_domain_name', 'resource_type': 'domain_name', 'hosted_zone_id': 'hosted_zone_id', 'certificate_arn': 'certificate_arn', 'security_policy': 'TLS_1_0', 'domain_name': 'example.com'}
        domain_name = create_http_domain_name()
        domain_name.security_policy = 'TLS_1_2'
        domain_name.certificate_arn = 'certificate_arn_1'
        domain_name.hosted_zone_id = ' hosted_zone_1'
        params = {'protocol': domain_name.protocol.value, 'domain_name': domain_name.domain_name, 'security_policy': domain_name.tls_version.value, 'certificate_arn': domain_name.certificate_arn, 'endpoint_type': 'EDGE', 'tags': None}
        self.remote_state.declare_resource_exists(domain_name, **deployed_value)
        planner = PlanStage(self.remote_state, self.osutils)
        plan = planner._add_domainname_plan(domain_name, 'EDGE')
        expected = [models.APICall(method_name='update_domain_name', params=params, output_var=domain_name.resource_name)]
        self.assert_apicall_equals(plan[0][0], expected[0])
        assert plan[0][1] == 'Updating custom domain name: example.com\n'

class TestPlanLambdaFunction(BasePlannerTests):
    def test_can_create_layer(self) -> None:
        layer = models.LambdaLayer(resource_name='layer', layer_name='bar', runtime='python2.7', deployment_package=models.DeploymentPackage(filename='foo'))
        plan = self.determine_plan(layer)
        expected = [models.APICall(method_name='publish_layer', params={'layer_name': 'bar', 'zip_contents': mock.ANY, 'runtime': 'python2.7'})]
        self.assert_apicall_equals(plan[0], expected[0])
        assert list(self.last_plan.messages.values()) == ['Creating lambda layer: bar\n']

    def test_can_update_layer(self) -> None:
        layer = models.LambdaLayer(resource_name='layer', layer_name='bar', runtime='python2.7', deployment_package=models.DeploymentPackage(filename='foo'))
        copy_of_layer = replace(layer)
        self.remote_state.declare_resource_exists(copy_of_layer, layer_version_arn='arn:bar:4')
        plan = self.determine_plan(layer)
        expected = [models.APICall(method_name='delete_layer_version', params={'layer_version_arn': 'arn:bar:4'}), models.APICall(method_name='publish_layer', params={'layer_name': 'bar', 'zip_contents': mock.ANY, 'runtime': 'python2.7'}), models.RecordResourceVariable(resource_type='lambda_layer', resource_name='layer', name='layer_version_arn', variable_name='layer_version_arn')]
        assert len(plan) == 3
        assert plan[0] == expected[0]
        assert plan[2] == expected[2]
        self.assert_apicall_equals(plan[1], expected[1])
        assert list(self.last_plan.messages.values()) == ['Updating lambda layer: bar\n']

    def test_can_create_function(self) -> None:
        function = create_function_resource('function_name')
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(function)
        expected = [models.APICall(method_name='create_function', params={'function_name': 'appname-dev-function_name', 'role_arn': 'role:arn', 'zip_contents': mock.ANY, 'runtime': 'python2.7', 'handler': 'app.app', 'environment_variables': {}, 'tags': {}, 'xray': None, 'timeout': 60, 'memory_size': 128, 'security_group_ids': [], 'subnet_ids': [], 'layers': []}), models.APICall(method_name='delete_function_concurrency', params={'function_name': 'appname-dev-function_name'}, output_var='reserved_concurrency_result')]
        self.assert_apicall_equals(plan[0], expected[0])
        self.assert_apicall_equals(plan[2], expected[1])
        assert list(self.last_plan.messages.values()) == ['Creating lambda function: appname-dev-function_name\n']

    def test_create_function_with_layers(self) -> None:
        layers = ['arn:aws:lambda:us-east-1:111:layer:test_layer:1']
        function = create_function_resource('function_name', layers=layers, managed_layer=create_managed_layer())
        self.remote_state.declare_no_resources_exists()
        plan = self.filter_api_calls(self.determine_plan(function.managed_layer))
        plan.extend(self.filter_api_calls(self.determine_plan(function)))
        expected = [models.APICall(method_name='publish_layer', params={'layer_name': 'bar', 'zip_contents': mock.ANY, 'runtime': 'python2.7'}), models.APICall(method_name='create_function', params={'function_name': 'appname-dev-function_name', 'role_arn': 'role:arn', 'zip_contents': mock.ANY, 'runtime': 'python2.7', 'handler': 'app.app', 'environment_variables': {}, 'tags': {}, 'timeout': 60, 'xray': None, 'memory_size': 128, 'security_group_ids': [], 'subnet_ids': [], 'layers': [Variable('layer_version_arn')] + layers}), models.APICall(method_name='delete_function_concurrency', params={'function_name': 'appname-dev-function_name'}, output_var='reserved_concurrency_result')]
        self.assert_apicall_equals(plan[0], expected[0])
        self.assert_apicall_equals(plan[1], expected[1])

    def test_can_update_lambda_function_code(self) -> None:
        function = create_function_resource('function_name')
        copy_of_function = replace(function)
        self.remote_state.declare_resource_exists(copy_of_function)
        function.memory_size = 256
        plan = self.determine_plan(function)
        existing_params = {'function_name': 'appname-dev-function_name', 'role_arn': 'role:arn', 'zip_contents': mock.ANY, 'runtime': 'python2.7', 'environment_variables': {}, 'xray': None, 'tags': {}, 'timeout': 60, 'security_group_ids': [], 'subnet_ids': [], 'layers': []}
        expected_params = dict(memory_size=256, **existing_params)
        expected = [models.APICall(method_name='update_function', params=expected_params), models.APICall(method_name='delete_function_concurrency', params={'function_name': 'appname-dev-function_name'}, output_var='reserved_concurrency_result')]
        self.assert_apicall_equals(plan[0], expected[0])
        self.assert_apicall_equals(plan[3], expected[1])
        assert list(self.last_plan.messages.values()) == ['Updating lambda function: appname-dev-function_name\n']

    def test_can_update_lambda_function_with_managed_layer(self) -> None:
        function = create_function_resource('function_name', managed_layer=create_managed_layer())
        copy_of_function = replace(function)
        self.remote_state.declare_resource_exists(copy_of_function)
        copy_of_layer = replace(function.managed_layer)
        self.remote_state.declare_resource_exists(copy_of_layer, layer_version_arn='arn:bar:4')
        plan = self.determine_plan(function.managed_layer)
        plan.extend(self.determine_plan(function))
        self.assert_apicall_equals(plan[0], models.APICall(method_name='delete_layer_version', params={'layer_version_arn': 'arn:bar:4'}))
        assert plan[3].method_name == 'update_function'
        assert plan[3].params['layers'] == [Variable('layer_version_arn')]

    def test_can_create_function_with_reserved_concurrency(self) -> None:
        function = create_function_resource('function_name')
        function.reserved_concurrency = 5
        self.remote_state.declare_no_resources_exists()
        plan = self.determine_plan(function)
        expected = [models.APICall(method_name='create_function', params={'function_name': 'appname-dev-function_name', 'role_arn': 'role:arn', 'zip_contents': mock.ANY, 'runtime': 'python2.7', 'handler': 'app.app', 'environment_variables': {}, 'tags': {}, 'xray': None, 'timeout': 60, 'memory_size': 128, 'security_group_ids': [], 'subnet_ids': [], 'layers': []}), models.APICall(method_name='put_function_concurrency', params={'function_name': 'appname-dev-function_name', 'reserved_concurrent_executions': 5}, output_var='reserved_concurrency_result')]
        self.assert_apicall_equals(plan[0], expected[0])
        self.assert_apicall_equals(plan[2], expected[1])
        assert list(self.last_plan.messages.values()) == ['Creating lambda function: appname-dev-function_name\n', 'Updating lambda function concurrency limit: appname-dev-function_name\n']

    def test_can_set_variables_when_needed(self) -> None:
        function = create_function_resource('function_name')
        self.remote_state.declare_no_resources_exists()
        function.role = models.ManagedIAMRole(resource_name='myrole', role_name='myrole-dev', trust_policy={'trust': 'policy'}, policy=models.FileBasedIAMPolicy(filename='foo.json', document={'iam': 'role'}))
        plan = self.determine_plan(function)
        call = plan[0]
        assert call.method_name == 'create_function'
        role_arn = call.params['role_arn']
        assert isinstance(role_arn, Variable)
        assert role_arn.name == 'myrole-dev_role_arn'

class TestPlanS3Events(BasePlannerTests):
    def test_can_plan_s3_event(self) -> None:
        function = create_function_resource('function_name')
        bucket_event = models.S3BucketNotification(resource_name='function_name-s3event', bucket='mybucket', events=['s3:ObjectCreated:*'], prefix=None, suffix=None, lambda_function=function)
        full_plan = self.determine_plan(bucket_event)
        setup_plan, plan = (full_plan[:4], full_plan[4:])
        assert setup_plan[0:4] == [models.BuiltinFunction('parse_arn', [Variable('function_name_lambda_arn')], output_var='parsed_lambda_arn'), models.JPSearch('account_id', input_var='parsed_lambda_arn', output_var='account_id'), models.JPSearch('region', input_var='parsed_lambda_arn', output_var='region_name'), models.JPSearch('partition', input_var='parsed_lambda_arn', output_var='partition')]
        self.assert_apicall_equals(plan[0], models.APICall(method_name='add_permission_for_s3_event', params={'bucket': 'mybucket', 'function_arn': Variable('function_name_lambda_arn'), 'account_id': Variable('account_id')}))
        self.assert_apicall_equals(plan[1], models.APICall(method_name='connect_s3_bucket_to_lambda', params={'bucket': 'mybucket', 'function_arn': Variable('function_name_lambda_arn'), 'events': ['s3:ObjectCreated:*'], 'prefix': None, 'suffix': None}))
        assert plan[2] == models.RecordResourceValue(resource_type='s3_event', resource_name='function_name-s3event', name='bucket', value='mybucket')
        assert plan[3] == models.RecordResourceVariable(resource_type='s3_event', resource_name='function_name-s3event', name='lambda_arn', variable_name='function_name_lambda_arn')

class TestPlanCloudWatchEvent(BasePlannerTests):
    def test_can_plan_cloudwatch_event(self) -> None:
        function = create_function_resource('function_name')
        event = models.CloudWatchEvent(resource_name='bar', rule_name='myrulename', event_pattern='"source": ["aws.ec2"]', lambda_function=function)
        plan = self.determine_plan(event)
        assert len(plan) == 4
        self.assert_apicall_equals(plan[0], models.APICall(method_name='get_or_create_rule_arn', params={'rule_name': 'myrulename', 'event_pattern': '"source": ["aws.ec2"]'}, output_var='rule-arn'))
        self.assert_apicall_equals(plan[1], models.APICall(method_name='connect_rule_to_lambda', params={'rule_name': 'myrulename', 'function_arn': Variable('function_name_lambda_arn')}))
        self.assert_apicall_equals(plan[2], models.APICall(method_name='add_permission_for_cloudwatch_event', params={'rule_arn': Variable('rule-arn'), 'function_arn': Variable('function_name_lambda_arn')}))
        assert plan[3] == models.RecordResourceValue(resource_type='cloudwatch_event', resource_name='bar', name='rule_name', value='myrulename')

class TestPlanScheduledEvent(BasePlannerTests):
    def test_can_plan_scheduled_event(self) -> None:
        function = create_function_resource('function_name')
        event = models.ScheduledEvent(resource_name='bar', rule_name='myrulename', rule_description='my rule description', schedule_expression='rate(5 minutes)', lambda_function=function)
        plan = self.determine_plan(event)
        assert len(plan) == 4
        self.assert_apicall_equals(plan[0], models.APICall(method_name='get_or_create_rule_arn', params={'rule_name': 'myrulename', 'rule_description': 'my rule description', 'schedule_expression': 'rate(5 minutes)'}, output_var='rule-arn'))
        self.assert_apicall_equals(plan[1], models.APICall(method_name='connect_rule_to_lambda', params={'rule_name': 'myrulename', 'function_arn': Variable('function_name_lambda_arn')}))
        self.assert_apicall_equals(plan[2], models.APICall(method_name='add_permission_for_cloudwatch_event', params={'rule_arn': Variable('rule-arn'), 'function_arn': Variable('function_name_lambda_arn')}))
        assert plan[3] == models.RecordResourceValue(resource_type='cloudwatch_event', resource_name='bar', name='rule_name', value='myrulename')

    def test_can_plan_scheduled_event_can_omit_description(self) -> None:
        function = create_function_resource('function_name')
        event = models.ScheduledEvent(resource_name='bar', rule_name='myrulename', schedule_expression='rate(5 minutes)', lambda_function=function)
        plan = self.determine_plan(event)
        self.assert_apicall_equals(plan[0], models.APICall(method_name='get_or_create_rule_arn', params={'rule_name': 'myrulename', 'schedule_expression': 'rate(5 minutes)'}, output_var='rule-arn'))

class TestPlanWebsocketAPI(BasePlannerTests):
    def assert_loads_needed_variables(self, plan: List[models.Instruction]) -> None:
        assert plan[0:5] == [models.BuiltinFunction('parse_arn', [Variable('function_name_connect_lambda_arn')], output_var='parsed_lambda_arn'), models.JPSearch('account_id', input_var='parsed_lambda_arn', output_var='account_id'), models.JPSearch('region', input_var='parsed_lambda_arn', output_var='region_name'), models.JPSearch('partition', input_var='parsed_lambda_arn', output_var='partition'), models.JPSearch('dns_suffix', input_var='parsed_lambda_arn', output_var='dns_suffix')]

    def test_can_plan_websocket_api(self) -> None:
        connect_function = create_function_resource('function_name_connect')
        message_function = create_function_resource('function_name_message')
        disconnect_function = create_function_resource('function_name_disconnect')
        websocket_api = models.WebsocketAPI(resource_name='websocket_api', name='app-dev-websocket-api', api_gateway_stage='api', routes=['$connect', '$default', '$disconnect'], connect_function=connect_function, message_function=message_function, disconnect_function=disconnect_function)
        plan = self.determine_plan(websocket_api)
        self.assert_loads_needed_variables(plan)
        assert plan[5:] == [models.APICall(method_name='create_websocket_api', params={'name': 'app-dev-websocket-api'}, output_var='websocket_api_id'), models.StoreValue(name='routes', value=[]), models.StoreValue(name='websocket-connect-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_connect', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-connect-integration-lambda-path'), 'handler_type': 'connect'}, output_var='connect-integration-id'), models.StoreValue(name='websocket-message-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_message', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-message-integration-lambda-path'), 'handler_type': 'message'}, output_var='message-integration-id'), models.StoreValue(name='websocket-disconnect-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_disconnect', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-disconnect-integration-lambda-path'), 'handler_type': 'disconnect'}, output_var='disconnect-integration-id'), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$connect', 'integration_id': Variable('connect-integration-id')}), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$default', 'integration_id': Variable('message-integration-id')}), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$disconnect', 'integration_id': Variable('disconnect-integration-id')}), models.APICall(method_name='deploy_websocket_api', params={'api_id': Variable('websocket_api_id')}, output_var='deployment-id'), models.APICall(method_name='create_stage', params={'api_id': Variable('websocket_api_id'), 'stage_name': 'api', 'deployment_id': Variable('deployment-id')}), models.StoreValue(name='websocket_api_url', value=StringFormat('wss://{websocket_api_id}.execute-api.{region_name}.{dns_suffix}/%s/' % 'api', ['websocket_api_id', 'region_name', 'dns_suffix'])), models.RecordResourceVariable(resource_type='websocket_api', resource_name='websocket_api', name='websocket_api_url', variable_name='websocket_api_url'), models.RecordResourceVariable(resource_type='websocket_api', resource_name='websocket_api', name='websocket_api_id', variable_name='websocket_api_id'), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_connect', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')}), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_message', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')}), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_disconnect', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')})]

    def test_can_update_websocket_api(self) -> None:
        connect_function = create_function_resource('function_name_connect')
        message_function = create_function_resource('function_name_message')
        disconnect_function = create_function_resource('function_name_disconnect')
        websocket_api = models.WebsocketAPI(resource_name='websocket_api', name='app-dev-websocket-api', api_gateway_stage='api', routes=['$connect', '$default', '$disconnect'], connect_function=connect_function, message_function=message_function, disconnect_function=disconnect_function)
        self.remote_state.declare_resource_exists(websocket_api)
        self.remote_state.deployed_values['websocket_api'] = {'websocket_api_id': 'my_websocket_api_id'}
        plan = self.determine_plan(websocket_api)
        self.assert_loads_needed_variables(plan)
        assert plan[5:] == [models.StoreValue(name='websocket_api_id', value='my_websocket_api_id'), models.APICall(method_name='get_websocket_routes', params={'api_id': Variable('websocket_api_id')}, output_var='routes'), models.APICall(method_name='delete_websocket_routes', params={'api_id': Variable('websocket_api_id'), 'routes': Variable('routes')}), models.APICall(method_name='get_websocket_integrations', params={'api_id': Variable('websocket_api_id')}, output_var='integrations'), models.APICall(method_name='delete_websocket_integrations', params={'api_id': Variable('websocket_api_id'), 'integrations': Variable('integrations')}), models.StoreValue(name='websocket-connect-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_connect', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-connect-integration-lambda-path'), 'handler_type': 'connect'}, output_var='connect-integration-id'), models.StoreValue(name='websocket-message-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_message', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-message-integration-lambda-path'), 'handler_type': 'message'}, output_var='message-integration-id'), models.StoreValue(name='websocket-disconnect-integration-lambda-path', value=StringFormat('arn:{partition}:apigateway:{region_name}:lambda:path/2015-03-31/functions/arn:{partition}:lambda:{region_name}:{account_id}:function:%s/invocations' % 'appname-dev-function_name_disconnect', ['partition', 'region_name', 'account_id'])), models.APICall(method_name='create_websocket_integration', params={'api_id': Variable('websocket_api_id'), 'lambda_function': Variable('websocket-disconnect-integration-lambda-path'), 'handler_type': 'disconnect'}, output_var='disconnect-integration-id'), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$connect', 'integration_id': Variable('connect-integration-id')}), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$default', 'integration_id': Variable('message-integration-id')}), models.APICall(method_name='create_websocket_route', params={'api_id': Variable('websocket_api_id'), 'route_key': '$disconnect', 'integration_id': Variable('disconnect-integration-id')}), models.StoreValue(name='websocket_api_url', value=StringFormat('wss://{websocket_api_id}.execute-api.{region_name}.{dns_suffix}/%s/' % 'api', ['websocket_api_id', 'region_name', 'dns_suffix'])), models.RecordResourceVariable(resource_type='websocket_api', resource_name='websocket_api', name='websocket_api_url', variable_name='websocket_api_url'), models.RecordResourceVariable(resource_type='websocket_api', resource_name='websocket_api', name='websocket_api_id', variable_name='websocket_api_id'), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_connect', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')}), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_message', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')}), models.APICall(method_name='add_permission_for_apigateway_v2', params={'function_name': 'appname-dev-function_name_disconnect', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'api_id': Variable('websocket_api_id')})]

class TestPlanRestAPI(BasePlannerTests):
    def assert_loads_needed_variables(self, plan: List[models.Instruction]) -> None:
        assert plan[0:6] == [models.BuiltinFunction('parse_arn', [Variable('function_name_lambda_arn')], output_var='parsed_lambda_arn'), models.JPSearch('account_id', input_var='parsed_lambda_arn', output_var='account_id'), models.JPSearch('region', input_var='parsed_lambda_arn', output_var='region_name'), models.JPSearch('partition', input_var='parsed_lambda_arn', output_var='partition'), models.JPSearch('dns_suffix', input_var='parsed_lambda_arn', output_var='dns_suffix'), models.CopyVariable(from_var='function_name_lambda_arn', to_var='api_handler_lambda_arn')]

    def test_can_plan_rest_api(self) -> None:
        function = create_function_resource('function_name')
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, endpoint_type='EDGE', minimum_compression='100', api_gateway_stage='api', xray=False, lambda_function=function)
        plan = self.determine_plan(rest_api)
        self.assert_loads_needed_variables(plan)
        assert plan[6:] == [models.APICall(method_name='import_rest_api', params={'swagger_document': {'swagger': '2.0'}, 'endpoint_type': 'EDGE'}, output_var='rest_api_id'), models.RecordResourceVariable(resource_type='rest_api', resource_name='rest_api', name='rest_api_id', variable_name='rest_api_id'), models.APICall(method_name='update_rest_api', params={'rest_api_id': Variable('rest_api_id'), 'patch_operations': [{'op': 'replace', 'path': '/minimumCompressionSize', 'value': '100'}]}), models.APICall(method_name='add_permission_for_apigateway', params={'function_name': 'appname-dev-function_name', 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'rest_api_id': Variable('rest_api_id')}), models.APICall(method_name='deploy_rest_api', params={'rest_api_id': Variable('rest_api_id'), 'xray': False, 'api_gateway_stage': 'api'}), models.StoreValue(name='rest_api_url', value=StringFormat('https://{rest_api_id}.execute-api.{region_name}.{dns_suffix}/api/', ['rest_api_id', 'region_name', 'dns_suffix'])), models.RecordResourceVariable(resource_type='rest_api', resource_name='rest_api', name='rest_api_url', variable_name='rest_api_url')]
        assert list(self.last_plan.messages.values()) == ['Creating Rest API\n']

    def test_can_update_rest_api_with_policy(self) -> None:
        function = create_function_resource('function_name')
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='REGIONAL', xray=False, lambda_function=function)
        self.remote_state.declare_resource_exists(rest_api)
        self.remote_state.deployed_values['rest_api'] = {'rest_api_id': 'my_rest_api_id'}
        plan = self.determine_plan(rest_api)
        assert plan[10].params == {'patch_operations': [{'op': 'replace', 'path': '/minimumCompressionSize', 'value': ''}, {'op': 'replace', 'path': StringFormat('/endpointConfiguration/types/{rest_api[endpointConfiguration][types][0]}', ['rest_api']), 'value': 'REGIONAL'}], 'rest_api_id': Variable('rest_api_id')}

    def test_can_update_rest_api(self) -> None:
        function = create_function_resource('function_name')
        rest_api = models.RestAPI(resource_name='rest_api', swagger_doc={'swagger': '2.0'}, minimum_compression='', api_gateway_stage='api', endpoint_type='REGIONAL', xray=False, lambda_function=function)
        self.remote_state.declare_resource_exists(rest_api)
        self.remote_state.deployed_values['rest_api'] = {'rest_api_id': 'my_rest_api_id'}
        plan = self.determine_plan(rest_api)
        self.assert_loads_needed_variables(plan)
        assert plan[6:] == [models.StoreValue(name='rest_api_id', value='my_rest_api_id'), models.RecordResourceVariable(resource_type='rest_api', resource_name='rest_api', name='rest_api_id', variable_name='rest_api_id'), models.APICall(method_name='update_api_from_swagger', params={'rest_api_id': Variable('rest_api_id'), 'swagger_document': {'swagger': '2.0'}}), models.APICall(method_name='get_rest_api', params={'rest_api_id': Variable('rest_api_id')}, output_var='rest_api'), models.APICall(method_name='update_rest_api', params={'rest_api_id': Variable('rest_api_id'), 'patch_operations': [{'op': 'replace', 'path': '/minimumCompressionSize', 'value': ''}, {'op': 'replace', 'value': 'REGIONAL', 'path': StringFormat('/endpointConfiguration/types/%s' % '{rest_api[endpointConfiguration][types][0]}', ['rest_api'])}]}), models.APICall(method_name='add_permission_for_apigateway', params={'rest_api_id': Variable('rest_api_id'), 'region_name': Variable('region_name'), 'account_id': Variable('account_id'), 'function_name': 'appname-dev-function_name'}, output_var=None), models.APICall(method_name='deploy_rest_api', params={'rest_api_id': Variable('rest_api_id'), 'xray': False, 'api_gateway_stage': 'api'}), models.StoreValue(name='rest_api_url', value=StringFormat('https://{rest_api_id}.execute-api.{region_name}.{dns_suffix}/api/', ['rest_api_id', 'region_name', 'dns_suffix'])), models.RecordResourceVariable(resource_type='rest_api', resource_name='rest_api', name='rest_api_url', variable_name='rest_api_url')]

class TestPlanSNSSubscription(BasePlannerTests):
    def test_can_plan_sns_subscription(self) -> None:
        function = create_function_resource('function_name')
        sns_subscription = models.SNSLambdaSubscription(resource_name='function_name-sns-subscription', topic='mytopic', lambda_function=function)
        plan = self.determine_plan(sns_subscription)
        plan_parse_arn = plan[:5]
        assert plan_parse_arn == [models.BuiltinFunction(function_name='parse_arn', args=[Variable('function_name_lambda_arn')], output_var='parsed_lambda_arn'), models.JPSearch(expression='account_id', input_var='parsed_lambda_arn', output_var='account_id'), models.JPSearch(expression='region', input_var='parsed_lambda_arn', output_var='region_name'), models.JPSearch(expression='partition', input_var='parsed_lambda_arn', output_var='partition'), models.StoreValue(name='function_name-sns-subscription_topic_arn', value=StringFormat('arn:{partition}:sns:{region_name}:{account_id}:mytopic', variables=['partition', 'region_name', 'account_id']))]
        topic_arn_var = Variable('function_name-sns-subscription_topic_arn')
        assert plan[5:7] == [models.APICall(method_name='add_permission_for_sns_topic', params={'function_arn': Variable('function_name_lambda_arn'), 'topic_arn': topic_arn_var}, output_var=None), models.APICall(method_name='subscribe_function_to_topic', params={'function_arn': Variable('function_name_lambda_arn'), 'topic_arn': topic_arn_var}, output_var='function_name-sns-subscription_subscription_arn')]
        self.assert_recorded_values(plan, 'sns_event', 'function_name-sns-subscription', {'topic': 'mytopic', 'lambda_arn': Variable('function_name_lambda_arn'), 'subscription_arn': Variable('function_name-sns-subscription_subscription_arn'), 'topic_arn': Variable('function_name-sns-subscription_topic_arn')})

    def test_can_plan_sns_arn_subscription(self) -> None:
        function = create_function_resource('function_name')
        topic_arn = 'arn:aws:sns:mars-west-2:123456789:mytopic'
        sns_subscription = models.SNSLambdaSubscription(resource_name='function_name-sns-subscription', topic=topic_arn, lambda_function=function)
        plan = self.determine_plan(sns_subscription)
        plan_parse_arn = plan[0]
        assert plan_parse_arn == models.StoreValue(name='function_name-sns-subscription_topic_arn', value=topic_arn)
        topic_arn_var = Variable('function_name-sns-subscription_topic_arn')
        assert plan[1:3] == [models.APICall(method_name='add_permission_for_sns_topic', params={'function_arn': Variable('function_name_lambda_arn'), 'topic_arn': topic_arn_var}, output_var=None), models.APICall(method