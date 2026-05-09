from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic, Type, Union, Callable, Sequence
from dataclasses import dataclass, field
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
    runtime: str = "python2.7",
    handler: str = "app.app",
    tags: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    memory_size: int = 128,
    deployment_package: Optional[models.DeploymentPackage] = None,
    role: Optional[models.PreCreatedIAMRole] = None,
    layers: Optional[List[str]] = None,
    managed_layer: Optional[models.LambdaLayer] = None,
) -> models.LambdaFunction:
    ...

def create_managed_layer() -> models.LambdaLayer:
    ...

def create_api_mapping() -> models.APIMapping:
    ...

def create_http_domain_name() -> models.DomainName:
    ...

def create_websocket_domain_name() -> models.DomainName:
    ...

@pytest.fixture
def no_deployed_values() -> DeployedResources:
    ...

class FakeConfig(object):
    def __init__(
        self,
        deployed_values: Dict[str, Any],
    ):
        self._deployed_values = deployed_values
        self.chalice_stage = "dev"
        self.api_gateway_stage = "dev"

    def deployed_resources(self, chalice_stage_name: str) -> DeployedResources:
        return DeployedResources(self._deployed_values)

class InMemoryRemoteState(object):
    def __init__(
        self,
        known_resources: Optional[Dict[Tuple[str, str], models.ManagedModel]] = None,
    ):
        if known_resources is None:
            known_resources = {}
        self.known_resources = known_resources
        self.deployed_values: Dict[str, Any] = {}

    def resource_exists(self, resource: models.ManagedModel, *args: Any) -> bool:
        ...

    def get_remote_model(self, resource: models.ManagedModel) -> models.ManagedModel:
        ...

    def declare_resource_exists(self, resource: models.ManagedModel, **deployed_values: Any) -> None:
        ...

    def declare_no_resources_exists(self) -> None:
        ...

    def resource_deployed_values(self, resource: models.ManagedModel) -> Dict[str, Any]:
        ...

class BasePlannerTests(object):
    def setup_method(self) -> None:
        self.osutils = mock.Mock(spec=OSUtils)
        self.remote_state = InMemoryRemoteState()
        self.last_plan: Optional[models.Plan] = None

    def assert_apicall_equals(self, expected: APICall, actual_api_call: APICall) -> None:
        ...

    def determine_plan(self, resource: models.ManagedModel) -> List[models.PlanInstruction]:
        ...

    def filter_api_calls(self, plan: List[models.PlanInstruction]) -> List[APICall]:
        ...

    def assert_recorded_values(self, plan: List[models.PlanInstruction], resource_type: str, resource_name: str, expected_mapping: Dict[str, Any]) -> None:
        ...

class TestPlanManagedRole(BasePlannerTests):
    ...

class TestPlanCreateUpdateAPIMapping(BasePlannerTests):
    ...

class TestPlanCreateUpdateDomainName(BasePlannerTests):
    ...

class TestPlanLambdaFunction(BasePlannerTests):
    ...

class TestPlanS3Events(BasePlannerTests):
    ...

class TestPlanCloudWatchEvent(BasePlannerTests):
    ...

class TestPlanScheduledEvent(BasePlannerTests):
    ...

class TestPlanWebsocketAPI(BasePlannerTests):
    ...

class TestPlanRestAPI(BasePlannerTests):
    ...

class TestPlanSNSSubscription(BasePlannerTests):
    ...

class TestPlanSQSSubscription(BasePlannerTests):
    ...

class TestPlanKinesisSubscription(BasePlannerTests):
    ...

class TestRemoteState(object):
    def setup_method(self) -> None:
        self.client = mock.Mock(spec=TypedAWSClient)
        self.config = FakeConfig({"resources": []})
        self.remote_state = RemoteState(self.client, self.config.deployed_resources("dev"))

    def create_rest_api_model(self) -> models.RestAPI:
        ...

    def create_api_mapping(self) -> models.APIMapping:
        ...

    def create_domain_name(self) -> models.DomainName:
        ...

    def create_websocket_api_model(self) -> models.WebsocketAPI:
        ...

    def test_role_exists(self) -> None:
        ...

    def test_role_does_not_exist(self) -> None:
        ...

    def test_lambda_layer_not_exists(self) -> None:
        ...

    def test_lambda_layer_exists(self) -> None:
        ...

    def test_lambda_function_exists(self) -> None:
        ...

    def test_lambda_function_does_not_exist(self) -> None:
        ...

    def test_api_gateway_domain_name_exists(self) -> None:
        ...

    def test_websocket_domain_name_exists(self) -> None:
        ...

    def test_none_api_mapping_exists(self) -> None:
        ...

    def test_path_api_mapping_exists_with_slash(self) -> None:
        ...

    def test_path_api_mapping_exists(self) -> None:
        ...

    def test_domain_name_does_not_exist(self) -> None:
        ...

    def test_exists_check_is_cached(self) -> None:
        ...

    def test_exists_check_is_cached_api_mapping(self) -> None:
        ...

    def test_rest_api_exists_no_deploy(self, no_deployed_values: DeployedResources) -> None:
        ...

    def test_rest_api_exists_with_existing_deploy(self) -> None:
        ...

    def test_rest_api_not_exists_with_preexisting_deploy(self) -> None:
        ...

    def test_websocket_api_exists_no_deploy(self, no_deployed_values: DeployedResources) -> None:
        ...

    def test_websocket_api_exists_with_existing_deploy(self) -> None:
        ...

    def test_websocket_api_not_exists_with_preexisting_deploy(self) -> None:
        ...

    def test_can_get_deployed_values(self) -> None:
        ...

    def test_value_error_raised_on_no_deployed_values(self, no_deployed_values: DeployedResources) -> None:
        ...

    def test_value_error_raised_for_unknown_resource_name(self) -> None:
        ...

    def test_dynamically_lookup_iam_role(self) -> None:
        ...

    def test_unknown_model_type_raises_error(self) -> None:
        ...

    def test_sns_subscription_exists(self) -> None:
        ...

    def test_sns_subscription_not_in_deployed_values(self) -> None:
        ...

    def test_sqs_event_source_exists(self) -> None:
        ...

    def test_kinesis_event_source_not_exists(self) -> None:
        ...

    def test_kinesis_event_source_exists(self) -> None:
        ...

    def test_ddb_event_source_not_exists(self) -> None:
        ...

    def test_ddb_event_source_exists(self) -> None:
        ...

class TestUnreferencedResourcePlanner(BasePlannerTests):
    def setup_method(self) -> None:
        super(TestUnreferencedResourcePlanner, self).setup_method()
        self.sweeper = ResourceSweeper()

    def execute(self, plan: List[models.PlanInstruction], config: FakeConfig) -> None:
        ...

    def one_deployed_lambda_function(self, name: str = "myfunction", arn: str = "arn") -> Dict[str, Any]:
        ...

    def test_noop_when_all_resources_accounted_for(self, function_resource: models.LambdaFunction) -> None:
        ...

    def test_will_delete_unreferenced_resource(self) -> None:
        ...

    def test_will_delete_log_group(self) -> None:
        ...

    def test_supports_multiple_unreferenced_and_unchanged(self) -> None:
        ...

    def test_can_delete_iam_role(self) -> None:
        ...

    def test_correct_deletion_order_for_dependencies(self) -> None:
        ...

    def test_can_delete_lambda_layer(self) -> None:
        ...

    def test_can_delete_scheduled_event(self) -> None:
        ...

    def test_can_delete_s3_event(self) -> None:
        ...

    def test_can_delete_rest_api(self) -> None:
        ...

    def test_can_delete_websocket_api(self) -> None:
        ...

    def test_can_handle_when_resource_changes(self) -> None:
        ...

    def test_no_sweeping_when_resource_value_unchanged(self) -> None:
        ...

    def test_can_delete_sns_subscription(self) -> None:
        ...

    def test_no_deletion_when_no_changes(self) -> None:
        ...

    def test_handles_when_topic_name_change(self) -> None:
        ...

    def test_no_sqs_deletion_when_no_changes(self) -> None:
        ...

    def test_can_delete_sqs_subscription(self) -> None:
        ...

    def test_handles_when_queue_name_change(self) -> None:
        ...

    def test_can_delete_domain_name(self) -> None:
        ...

    def test_can_handle_domain_name_without_api_mapping(self) -> None:
        ...

    def test_can_delete_api_mapping(self) -> None:
        ...

    def test_can_delete_api_mapping_none(self) -> None:
        ...

    def test_raise_error_not_existed_resource_delete(self) -> None:
        ...

    def test_update_plan_with_insert_without_message(self) -> None:
        ...

class TestKeyVariable(object):
    def test_key_variable_str(self) -> None:
        ...

    def test_key_variables_equal(self) -> None:
        ...

class TestPlanLogGroup(BasePlannerTests):
    def test_can_create_log_group(self) -> None:
        ...

    def test_can_update_log_group(self) -> None:
        ...
