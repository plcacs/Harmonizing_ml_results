import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Text, cast
from dataclasses import asdict
from chalice.config import Config
from chalice import app
from chalice.constants import LAMBDA_TRUST_POLICY
from chalice.deploy import models
from chalice.utils import UI

StrMapAny = Dict[str, Any]

class ChaliceBuildError(Exception):
    ...

class ApplicationGraphBuilder:
    def __init__(self) -> None:
        ...

    def build(self, config: Any, stage_name: str) -> models.Application:
        ...

    def _create_log_group(self, config: Any, resource_name: str, log_group_name: str) -> models.LogGroup:
        ...

    def _create_custom_domain_name(self, api_type: models.APIType, domain_name_data: Dict[str, Any], endpoint_configuration: str, api_gateway_stage: str) -> models.DomainName:
        ...

    def _create_api_mapping_model(self, key: str, stage: str) -> models.APIMapping:
        ...

    def _create_lambda_event_resources(self, config: Any, deployment: Any, stage_name: str) -> List[Any]:
        ...

    def _create_rest_api_model(self, config: Any, deployment: Any, stage_name: str) -> models.RestAPI:
        ...

    def _get_default_private_api_policy(self, config: Any) -> Dict[str, Any]:
        ...

    def _create_websocket_api_model(self, config: Any, deployment: Any, stage_name: str) -> models.WebsocketAPI:
        ...

    def _create_cwe_subscription(self, config: Any, deployment: Any, event_source: app.CloudWatchEventConfig, stage_name: str) -> models.CloudWatchEvent:
        ...

    def _create_scheduled_model(self, config: Any, deployment: Any, event_source: app.ScheduledEventConfig, stage_name: str) -> models.ScheduledEvent:
        ...

    def _create_domain_name_model(self, protocol: models.APIType, data: Dict[str, Any], endpoint_type: str, api_mapping: models.APIMapping) -> models.DomainName:
        ...

    def _create_lambda_model(self, config: Any, deployment: Any, name: str, handler_name: str, stage_name: str) -> models.LambdaFunction:
        ...

    def _get_managed_lambda_layer(self, config: Any) -> Optional[models.LambdaLayer]:
        ...

    def _get_role_reference(self, config: Any, stage_name: str, function_name: str) -> Union[models.ManagedIAMRole, models.PreCreatedIAMRole]:
        ...

    def _get_role_identifier(self, role: Union[models.ManagedIAMRole, models.PreCreatedIAMRole]) -> str:
        ...

    def _create_role_reference(self, config: Any, stage_name: str, function_name: str) -> models.ManagedIAMRole:
        ...

    def _get_vpc_params(self, function_name: str, config: Any) -> Tuple[List[str], List[str]]:
        ...

    def _get_lambda_layers(self, config: Any) -> List[str]:
        ...

    def _build_lambda_function(self, config: Any, name: str, handler_name: str, deployment: Any, role: Union[models.ManagedIAMRole, models.PreCreatedIAMRole]) -> models.LambdaFunction:
        ...

    def _inject_role_traits(self, function: models.LambdaFunction, role: Union[models.ManagedIAMRole, models.PreCreatedIAMRole]) -> None:
        ...

    def _create_bucket_notification(self, config: Any, deployment: Any, s3_event: app.S3EventConfig, stage_name: str) -> models.S3BucketNotification:
        ...

    def _create_sns_subscription(self, config: Any, deployment: Any, sns_config: app.SNSEventConfig, stage_name: str) -> models.SNSLambdaSubscription:
        ...

    def _create_sqs_subscription(self, config: Any, deployment: Any, sqs_config: app.SQSEventConfig, stage_name: str) -> models.SQSEventSource:
        ...

    def _create_kinesis_subscription(self, config: Any, deployment: Any, kinesis_config: app.KinesisEventConfig, stage_name: str) -> models.KinesisEventSource:
        ...

    def _create_ddb_subscription(self, config: Any, deployment: Any, ddb_config: app.DynamoDBEventConfig, stage_name: str) -> models.DynamoDBEventSource:
        ...

class DependencyBuilder:
    def __init__(self) -> None:
        ...

    def build_dependencies(self, graph: Any) -> List[Any]:
        ...

    def _traverse(self, resource: Any, ordered: List[Any], seen: Set[int]) -> None:
        ...

class GraphPrettyPrint:
    def __init__(self, ui: UI) -> None:
        ...

    def display_graph(self, graph: Any) -> None:
        ...

    def _traverse(self, graph: Any, level: int) -> None:
        ...

    def _get_model_text(self, model: Any, spaces: str, level: int) -> str:
        ...

    def _add_remaining_lines(self, lines: List[str], remaining: List[Tuple[str, Any]], full: str) -> None:
        ...

    def _get_filtered_params(self, model: Any) -> Dict[str, Any]:
        ...