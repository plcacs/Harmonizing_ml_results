"""Type stubs for awsclient_860c67 module."""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Iterator,
    Iterable,
    IO,
    TypedDict,
    overload,
)
import json
import re
import uuid
from botocore.session import Session
from botocore.exceptions import ClientError
from botocore.vendored.requests.exceptions import ConnectionError as RequestsConnectionError

class CWLogEvent(TypedDict):
    eventId: str
    ingestionTime: datetime
    logStreamName: str
    message: str
    timestamp: datetime
    logShortId: str

class LogEventsResponse(TypedDict, total=False):
    events: List[CWLogEvent]
    nextToken: str

class DomainNameResponse(TypedDict):
    domain_name: str
    security_policy: str
    hosted_zone_id: str
    certificate_arn: str
    alias_domain_name: str

class TypedAWSClient:
    LAMBDA_CREATE_ATTEMPTS: int
    DELAY_TIME: int

    def __init__(self, session: Session, sleep: Callable[[float], None] = time.sleep) -> None: ...

    def resolve_endpoint(self, service: str, region: str) -> Optional[Dict[str, Any]]: ...

    def endpoint_from_arn(self, arn: str) -> Optional[Dict[str, Any]]: ...

    def endpoint_dns_suffix(self, service: str, region: str) -> str: ...

    def endpoint_dns_suffix_from_arn(self, arn: str) -> str: ...

    def service_principal(
        self,
        service: str,
        region: str = 'us-east-1',
        url_suffix: str = 'amazonaws.com',
    ) -> str: ...

    def lambda_function_exists(self, name: str) -> bool: ...

    def api_mapping_exists(self, domain_name: str, api_map_key: str) -> bool: ...

    def get_domain_name(self, domain_name: str) -> Dict[str, Any]: ...

    def domain_name_exists(self, domain_name: str) -> bool: ...

    def domain_name_exists_v2(self, domain_name: str) -> bool: ...

    def get_function_configuration(self, name: str) -> Dict[str, Any]: ...

    def publish_layer(
        self,
        layer_name: str,
        zip_contents: bytes,
        runtime: str,
    ) -> str: ...

    def delete_layer_version(self, layer_version_arn: str) -> None: ...

    def get_layer_version(self, layer_version_arn: str) -> Optional[Dict[str, Any]]: ...

    def create_function(
        self,
        function_name: str,
        role_arn: str,
        zip_contents: bytes,
        runtime: str,
        handler: str,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        xray: bool = False,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        security_group_ids: Optional[List[str]] = None,
        subnet_ids: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
    ) -> str: ...

    def create_api_mapping(
        self,
        domain_name: str,
        path_key: str,
        api_id: str,
        stage: str,
    ) -> Dict[str, str]: ...

    def create_base_path_mapping(
        self,
        domain_name: str,
        path_key: str,
        api_id: str,
        stage: str,
    ) -> Dict[str, str]: ...

    def create_domain_name(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DomainNameResponse: ...

    def update_domain_name(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> DomainNameResponse: ...

    def delete_domain_name(self, domain_name: str) -> None: ...

    def delete_api_mapping(self, domain_name: str, path_key: str) -> None: ...

    def update_function(
        self,
        function_name: str,
        zip_contents: bytes,
        environment_variables: Optional[Dict[str, str]] = None,
        runtime: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        xray: Optional[bool] = None,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        role_arn: Optional[str] = None,
        subnet_ids: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    def invoke_function(
        self,
        name: str,
        payload: Optional[bytes] = None,
    ) -> Dict[str, Any]: ...

    def delete_function(self, function_name: str) -> None: ...

    def get_role_arn_for_name(self, name: str) -> Dict[str, Any]: ...

    def get_role(self, name: str) -> Dict[str, Any]: ...

    def delete_role_policy(self, role_name: str, policy_name: str) -> None: ...

    def put_role_policy(
        self,
        role_name: str,
        policy_name: str,
        policy_document: Dict[str, Any],
    ) -> None: ...

    def create_role(
        self,
        name: str,
        trust_policy: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> str: ...

    def delete_role(self, name: str) -> None: ...

    def log_group_exists(self, name: str) -> bool: ...

    def create_log_group(self, log_group_name: str) -> None: ...

    def delete_retention_policy(self, log_group_name: str) -> None: ...

    def delete_log_group(self, log_group_name: str) -> None: ...

    def put_retention_policy(self, name: str, retention_in_days: int) -> None: ...

    def get_rest_api_id(self, name: str) -> Optional[str]: ...

    def get_rest_api(self, rest_api_id: str) -> Dict[str, Any]: ...

    def import_rest_api(
        self,
        swagger_document: Dict[str, Any],
        endpoint_type: str,
    ) -> str: ...

    def update_api_from_swagger(
        self,
        rest_api_id: str,
        swagger_document: Dict[str, Any],
    ) -> None: ...

    def update_rest_api(
        self,
        rest_api_id: str,
        patch_operations: List[Dict[str, Any]],
    ) -> None: ...

    def delete_rest_api(self, rest_api_id: str) -> None: ...

    def deploy_rest_api(
        self,
        rest_api_id: str,
        api_gateway_stage: str,
        xray: bool,
    ) -> None: ...

    def add_permission_for_apigateway(
        self,
        function_name: str,
        region_name: str,
        account_id: str,
        rest_api_id: str,
        random_id: Optional[str] = None,
    ) -> None: ...

    def get_function_policy(self, function_name: str) -> Dict[str, Any]: ...

    def download_sdk(
        self,
        rest_api_id: str,
        output_dir: str,
        api_gateway_stage: str = 'api',
        sdk_type: str = 'javascript',
    ) -> None: ...

    def get_sdk_download_stream(
        self,
        rest_api_id: str,
        api_gateway_stage: str = 'api',
        sdk_type: str = 'javascript',
    ) -> IO[bytes]: ...

    def subscribe_function_to_topic(
        self,
        topic_arn: str,
        function_arn: str,
    ) -> str: ...

    def unsubscribe_from_topic(self, subscription_arn: str) -> None: ...

    def verify_sns_subscription_current(
        self,
        subscription_arn: str,
        topic_name: str,
        function_arn: str,
    ) -> bool: ...

    def add_permission_for_sns_topic(
        self,
        topic_arn: str,
        function_arn: str,
    ) -> None: ...

    def remove_permission_for_sns_topic(
        self,
        topic_arn: str,
        function_arn: str,
    ) -> None: ...

    def iter_log_events(
        self,
        log_group_name: str,
        start_time: Optional[datetime] = None,
        interleaved: bool = True,
    ) -> Iterator[CWLogEvent]: ...

    def filter_log_events(
        self,
        log_group_name: str,
        start_time: Optional[datetime] = None,
        next_token: Optional[str] = None,
    ) -> LogEventsResponse: ...

    def add_permission_for_authorizer(
        self,
        rest_api_id: str,
        function_arn: str,
        random_id: Optional[str] = None,
    ) -> None: ...

    def get_or_create_rule_arn(
        self,
        rule_name: str,
        schedule_expression: Optional[str] = None,
        event_pattern: Optional[Dict[str, Any]] = None,
        rule_description: Optional[str] = None,
    ) -> str: ...

    def delete_rule(self, rule_name: str) -> None: ...

    def connect_rule_to_lambda(
        self,
        rule_name: str,
        function_arn: str,
    ) -> None: ...

    def add_permission_for_cloudwatch_event(
        self,
        rule_arn: str,
        function_arn: str,
    ) -> None: ...

    def connect_s3_bucket_to_lambda(
        self,
        bucket: str,
        function_arn: str,
        events: List[str],
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None: ...

    def add_permission_for_s3_event(
        self,
        bucket: str,
        function_arn: str,
        account_id: str,
    ) -> None: ...

    def remove_permission_for_s3_event(
        self,
        bucket: str,
        function_arn: str,
        account_id: str,
    ) -> None: ...

    def disconnect_s3_bucket_from_lambda(
        self,
        bucket: str,
        function_arn: str,
    ) -> None: ...

    def create_lambda_event_source(
        self,
        event_source_arn: str,
        function_name: str,
        batch_size: int,
        starting_position: Optional[str] = None,
        maximum_batching_window_in_seconds: int = 0,
        maximum_concurrency: Optional[int] = None,
    ) -> str: ...

    def update_lambda_event_source(
        self,
        event_uuid: str,
        batch_size: int,
        maximum_batching_window_in_seconds: int = 0,
        maximum_concurrency: Optional[int] = None,
    ) -> None: ...

    def remove_lambda_event_source(self, event_uuid: str) -> None: ...

    def verify_event_source_current(
        self,
        event_uuid: str,
        resource_name: str,
        service_name: str,
        function_arn: str,
    ) -> bool: ...

    def verify_event_source_arn_current(
        self,
        event_uuid: str,
        event_source_arn: str,
        function_arn: str,
    ) -> bool: ...

    def create_websocket_api(self, name: str) -> str: ...

    def get_websocket_api_id(self, name: str) -> Optional[str]: ...

    def websocket_api_exists(self, api_id: str) -> bool: ...

    def delete_websocket_api(self, api_id: str) -> None: ...

    def create_websocket_integration(
        self,
        api_id: str,
        lambda_function: str,
        handler_type: str,
    ) -> str: ...

    def create_websocket_route(
        self,
        api_id: str,
        route_key: str,
        integration_id: str,
    ) -> None: ...

    def delete_websocket_routes(self, api_id: str, routes: List[str]) -> None: ...

    def delete_websocket_integrations(self, api_id: str, integrations: List[str]) -> None: ...

    def deploy_websocket_api(self, api_id: str) -> str: ...

    def get_websocket_routes(self, api_id: str) -> List[str]: ...

    def get_websocket_integrations(self, api_id: str) -> List[str]: ...

    def create_stage(
        self,
        api_id: str,
        stage_name: str,
        deployment_id: str,
    ) -> None: ...

    def _call_client_method_with_retries(
        self,
        method: Callable[..., Any],
        kwargs: Dict[str, Any],
        max_attempts: int,
        should_retry: Optional[Callable[[Exception], bool]] = None,
        delay_time: float = DELAY_TIME,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ) -> Any: ...