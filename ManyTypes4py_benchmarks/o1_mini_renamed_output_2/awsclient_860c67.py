"""Simplified AWS client.

This module abstracts the botocore session and clients
to provide a simpler interface.  This interface only
contains the API calls needed to work with AWS services
used by chalice.

The interface provided can range from a direct 1-1 mapping
of a method to a method on a botocore client all the way up
to combining API calls across multiple AWS services.

As a side benefit, I can also add type annotations to
this class to get improved type checking across chalice.

"""
from __future__ import annotations
import os
import time
import tempfile
from datetime import datetime
import zipfile
import shutil
import json
import re
import uuid
from collections import OrderedDict
from typing import Any, Optional, Dict, Callable, List, Iterator, Iterable, Sequence, IO, Tuple, Union
import botocore.session
from botocore.loaders import create_loader
from botocore.exceptions import ClientError
from botocore.utils import datetime2timestamp
from botocore.vendored.requests import ConnectionError as RequestsConnectionError
from botocore.vendored.requests.exceptions import ReadTimeout as RequestsReadTimeout
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import MAX_LAMBDA_DEPLOYMENT_SIZE
from chalice.vendored.botocore.regions import EndpointResolver

StrMap = Optional[Dict[str, str]]
StrAnyMap = Dict[str, Any]
OptStr = Optional[str]
OptInt = Optional[int]
OptStrList = Optional[List[str]]
ClientMethod = Callable[..., Dict[str, Any]]

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

_REMOTE_CALL_ERRORS = (ClientError, RequestsConnectionError)


class AWSClientError(Exception):
    pass


class ReadTimeout(AWSClientError):

    def __init__(self, message: str) -> None:
        self.message = message


class ResourceDoesNotExistError(AWSClientError):
    pass


class LambdaClientError(AWSClientError):

    def __init__(self, original_error: Exception, context: 'LambdaErrorContext') -> None:
        self.original_error = original_error
        self.context = context
        super(LambdaClientError, self).__init__(str(original_error))


class DeploymentPackageTooLargeError(LambdaClientError):
    pass


class LambdaErrorContext:
    function_name: str
    client_method_name: str
    deployment_size: int

    def __init__(self, function_name: str, client_method_name: str, deployment_size: int) -> None:
        self.function_name = function_name
        self.client_method_name = client_method_name
        self.deployment_size = deployment_size


class TypedAWSClient:
    LAMBDA_CREATE_ATTEMPTS: int = 30
    DELAY_TIME: int = 5

    def __init__(self, session: botocore.session.Session, sleep: Callable[[float], None] = time.sleep) -> None:
        self._session = session
        self._sleep = sleep
        self._client_cache: Dict[str, botocore.client.BaseClient] = {}
        loader = create_loader('data_loader')
        endpoints = loader.load_data('endpoints')
        self._endpoint_resolver = EndpointResolver(endpoints)

    def func_cd8zzk5j(self, service: str, region: str) -> Optional[Dict[str, Any]]:
        """Find details of an endpoint based on the service and region.

        This utilizes the botocore EndpointResolver in order to find details on
        the given service and region combination.  If the service and region
        combination is not found the None will be returned.
        """
        return self._endpoint_resolver.construct_endpoint(service, region)

    def func_3p5vlmkv(self, arn: str) -> Optional[Dict[str, Any]]:
        """Find details for the endpoint associated with a resource ARN.

        This allows the an endpoint to be discerned based on an ARN.  This
        is a convenience method due to the need to parse multiple ARNs
        throughout the project. If the service and region combination
        is not found the None will be returned.
        """
        arn_split = arn.split(':')
        return self.resolve_endpoint(arn_split[2], arn_split[3])

    def func_0gm2mffg(self, service: str, region: str) -> str:
        """Discover the dns suffix for a given service and region combination.

        This allows the service DNS suffix to be discoverable throughout the
        framework.  If the ARN's service and region combination is not found
        then amazonaws.com is returned.

        """
        endpoint = self.resolve_endpoint(service, region)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def func_4oyqwkxh(self, arn: str) -> str:
        """Discover the dns suffix for a given ARN.

        This allows the service DNS suffix to be discoverable throughout the
        framework based on the ARN.  If the ARN's service and region
        combination is not found then amazonaws.com is returned.

        """
        endpoint = self.endpoint_from_arn(arn)
        return endpoint['dnsSuffix'] if endpoint else 'amazonaws.com'

    def func_1u4hz44f(
        self,
        service: str,
        region: str = 'us-east-1',
        url_suffix: str = 'amazonaws.com'
    ) -> str:
        """Compute a "standard" AWS Service principal for given arguments.

        Attribution: This code was ported from https://github.com/aws/aws-cdk
        and more specifically, aws-cdk/region-info/lib/default.ts

        Computes a "standard" AWS Service principal for a given service, region
        and suffix. This is useful for example when you need to compute a
        service principal name, but you do not have a synthesize-time region
        literal available (so all you have is `{ "Ref": "AWS::Region" }`). This
        way you get the same defaulting behavior that is normally used for
        built-in data.

        :param service: the name of the service (s3, s3.amazonaws.com, ...)
        :param region: the region in which the service principal is needed.
        :param url_suffix: the URL suffix for the partition in which the region
        is located.
        :return: The service principal for the given combination of arguments
        """
        matches = re.match(
            '^([^.]+)(?:(?:\\.amazonaws\\.com(?:\\.cn)?)|(?:\\.c2s\\.ic\\.gov)|(?:\\.sc2s\\.sgov\\.gov))?$',
            service
        )
        if matches is None:
            return service
        service_name = matches.group(1)
        us_iso_exceptions = {'cloudhsm', 'config', 'states', 'workspaces'}
        us_isob_exceptions = {'dms', 'states'}
        if region.startswith('us-iso-') and service_name in us_iso_exceptions:
            if service_name == 'states':
                return f'{service_name}.amazonaws.com'
            else:
                return f'{service_name}.{url_suffix}'
        if region.startswith('us-isob-') and service_name in us_isob_exceptions:
            if service_name == 'states':
                return f'{service_name}.amazonaws.com'
            else:
                return f'{service_name}.{url_suffix}'
        if service_name in ['codedeploy', 'logs']:
            return f'{service_name}.{region}.{url_suffix}'
        elif service_name == 'states':
            return f'{service_name}.{region}.amazonaws.com'
        elif service_name == 'ec2':
            return f'{service_name}.{url_suffix}'
        else:
            return f'{service_name}.amazonaws.com'

    def func_f08qnv58(self, name: str) -> bool:
        client = self._client('lambda')
        try:
            client.get_function(FunctionName=name)
            return True
        except client.exceptions.ResourceNotFoundException:
            return False

    def func_f5ghht5n(self, domain_name: str, api_map_key: str) -> bool:
        client = self._client('apigatewayv2')
        try:
            result = client.get_api_mappings(DomainName=domain_name)
            api_map = [api_map for api_map in result['Items'] if api_map['ApiMappingKey'] == api_map_key]
            if api_map:
                return True
            return False
        except client.exceptions.NotFoundException:
            return False

    def func_8f207hus(self, domain_name: str) -> DomainNameResponse:
        client = self._client('apigateway')
        try:
            domain = client.get_domain_name(domainName=domain_name)
        except client.exceptions.NotFoundException:
            err_msg = f'No domain name found by {domain_name} name'
            raise ResourceDoesNotExistError(err_msg)
        return domain

    def func_koihjaba(self, domain_name: str) -> bool:
        try:
            self.get_domain_name(domain_name)
            return True
        except ResourceDoesNotExistError:
            return False

    def func_z92mt6nq(self, domain_name: str) -> bool:
        client = self._client('apigatewayv2')
        try:
            client.get_domain_name(DomainName=domain_name)
            return True
        except client.exceptions.NotFoundException:
            return False

    def func_6teiqx1w(self, name: str) -> Dict[str, Any]:
        response = self._client('lambda').get_function_configuration(FunctionName=name)
        return response

    def func_gmm1b4qz(self, security_group_ids: Optional[List[str]], subnet_ids: Optional[List[str]]) -> Dict[str, List[str]]:
        vpc_config: Dict[str, List[str]] = {'SubnetIds': [], 'SecurityGroupIds': []}
        if security_group_ids is not None and subnet_ids is not None:
            vpc_config['SubnetIds'] = subnet_ids
            vpc_config['SecurityGroupIds'] = security_group_ids
        return vpc_config

    def func_11jk1oly(self, layer_name: str, zip_contents: bytes, runtime: str) -> str:
        try:
            return self._client('lambda').publish_layer_version(
                LayerName=layer_name,
                Content={'ZipFile': zip_contents},
                CompatibleRuntimes=[runtime]
            )['LayerVersionArn']
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(layer_name, 'publish_layer_version', len(zip_contents))
            raise self._get_lambda_code_deployment_error(e, context)

    def func_a451pmus(self, layer_version_arn: str) -> None:
        client = self._client('lambda')
        _, layer_name, version_number = layer_version_arn.rsplit(':', 2)
        try:
            client.delete_layer_version(LayerName=layer_name, VersionNumber=int(version_number))
        except client.exceptions.ResourceNotFoundException:
            pass

    def func_w8xvy6m5(self, layer_version_arn: str) -> Dict[str, Any]:
        client = self._client('lambda')
        try:
            return client.get_layer_version_by_arn(Arn=layer_version_arn)
        except client.exceptions.ResourceNotFoundException:
            pass
        return {}

    def func_of66o2kz(
        self,
        function_name: str,
        role_arn: str,
        zip_contents: bytes,
        runtime: str,
        handler: str,
        environment_variables: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        xray: Optional[bool] = None,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        security_group_ids: Optional[List[str]] = None,
        subnet_ids: Optional[List[str]] = None,
        layers: Optional[List[str]] = None
    ) -> str:
        kwargs: Dict[str, Any] = {
            'FunctionName': function_name,
            'Runtime': runtime,
            'Code': {'ZipFile': zip_contents},
            'Handler': handler,
            'Role': role_arn
        }
        if environment_variables is not None:
            kwargs['Environment'] = {'Variables': environment_variables}
        if tags is not None:
            kwargs['Tags'] = tags
        if xray is True:
            kwargs['TracingConfig'] = {'Mode': 'Active'}
        if timeout is not None:
            kwargs['Timeout'] = timeout
        if memory_size is not None:
            kwargs['MemorySize'] = memory_size
        if security_group_ids is not None and subnet_ids is not None:
            kwargs['VpcConfig'] = self._create_vpc_config(
                security_group_ids=security_group_ids,
                subnet_ids=subnet_ids
            )
        if layers is not None:
            kwargs['Layers'] = layers
        arn, state = self._create_lambda_function(kwargs)
        if state != 'Active':
            self._wait_for_active(function_name)
        return arn

    def func_oumay7ly(self, function_name: str) -> None:
        client = self._client('lambda')
        waiter = client.get_waiter('function_active')
        waiter.wait(FunctionName=function_name)

    def func_02yb8ysq(
        self,
        domain_name: str,
        path_key: str,
        api_id: str,
        stage: str
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'DomainName': domain_name,
            'ApiMappingKey': path_key,
            'ApiId': api_id,
            'Stage': stage
        }
        return self._create_api_mapping(kwargs)

    def func_diq6idlx(
        self,
        domain_name: str,
        path_key: str,
        api_id: str,
        stage: str
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'domainName': domain_name,
            'basePath': path_key,
            'restApiId': api_id,
            'stage': stage
        }
        return self._create_base_path_mapping(kwargs)

    def func_t7x76zrt(self, base_path_args: Dict[str, Any]) -> Dict[str, str]:
        result = self._client('apigateway').create_base_path_mapping(**base_path_args)
        if result['basePath'] == '(none)':
            base_path = '/'
        else:
            base_path = f'/{result["basePath"]}'
        base_path_mapping: Dict[str, str] = {'key': base_path}
        return base_path_mapping

    def func_3l1eao9r(self, api_args: Dict[str, Any]) -> Dict[str, str]:
        result = self._client('apigatewayv2').create_api_mapping(**api_args)
        if result['ApiMappingKey'] == '(none)':
            map_key = '/'
        else:
            map_key = f'/{result["ApiMappingKey"]}'
        api_mapping: Dict[str, str] = {'key': map_key}
        return api_mapping

    def func_bme2l3n2(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> DomainNameResponse:
        if protocol == 'HTTP':
            kwargs: Dict[str, Any] = {
                'domainName': domain_name,
                'endpointConfiguration': {'types': [endpoint_type]}
            }
            if security_policy is not None:
                kwargs['securityPolicy'] = security_policy
            if endpoint_type == 'EDGE':
                kwargs['certificateArn'] = certificate_arn
            else:
                kwargs['regionalCertificateArn'] = certificate_arn
            if tags is not None:
                kwargs['tags'] = tags
            created_domain_name = self._create_domain_name(kwargs)
        elif protocol == 'WEBSOCKET':
            kwargs = self.get_custom_domain_params_v2(
                domain_name=domain_name,
                endpoint_type=endpoint_type,
                security_policy=security_policy,
                certificate_arn=certificate_arn,
                tags=tags
            )
            created_domain_name = self._create_domain_name_v2(kwargs)
        else:
            raise ValueError('Unsupported protocol value.')
        return created_domain_name

    def func_9sdvptbw(self, api_args: Dict[str, Any]) -> DomainNameResponse:
        client = self._client('apigateway')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions
        )
        if result.get('regionalHostedZoneId'):
            hosted_zone_id = result['regionalHostedZoneId']
        else:
            hosted_zone_id = result['distributionHostedZoneId']
        if result.get('regionalCertificateArn'):
            certificate_arn = result['regionalCertificateArn']
        else:
            certificate_arn = result['certificateArn']
        if result.get('regionalDomainName') is not None:
            alias_domain_name = result['regionalDomainName']
        else:
            alias_domain_name = result['distributionDomainName']
        domain_name: DomainNameResponse = {
            'domain_name': result['domainName'],
            'security_policy': result['securityPolicy'],
            'hosted_zone_id': hosted_zone_id,
            'certificate_arn': certificate_arn,
            'alias_domain_name': alias_domain_name
        }
        return domain_name

    def func_rsmvxmte(self, api_args: Dict[str, Any]) -> DomainNameResponse:
        client = self._client('apigatewayv2')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = self._call_client_method_with_retries(
            client.create_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions
        )
        result_data = result['DomainNameConfigurations'][0]
        domain_name: DomainNameResponse = {
            'domain_name': result['DomainName'],
            'alias_domain_name': result_data['ApiGatewayDomainName'],
            'security_policy': result_data['SecurityPolicy'],
            'hosted_zone_id': result_data['HostedZoneId'],
            'certificate_arn': result_data['CertificateArn']
        }
        return domain_name

    def func_rlwqubo2(self, api_args: Dict[str, Any]) -> Tuple[str, str]:
        try:
            result = self._call_client_method_with_retries(
                self._client('lambda').create_function,
                api_args,
                max_attempts=self.LAMBDA_CREATE_ATTEMPTS
            )
            return result['FunctionArn'], result['State']
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(api_args['FunctionName'], 'create_function', len(api_args['Code']['ZipFile']))
            raise self._get_lambda_code_deployment_error(e, context)

    def func_cbx7q9z8(self, error: Exception) -> bool:
        message = error.response['Error'].get('Message', '')
        if re.search('event source mapping.*is in use', message):
            return True
        return False

    def func_s9cehh6a(
        self,
        name: str,
        payload: Optional[bytes] = None
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'FunctionName': name,
            'InvocationType': 'RequestResponse'
        }
        if payload is not None:
            kwargs['Payload'] = payload
        try:
            return self._client('lambda').invoke(**kwargs)
        except RequestsReadTimeout as e:
            raise ReadTimeout(str(e))

    def func_c9i90frb(self, error: Exception) -> bool:
        message = error.response['Error'].get('Message', '')
        if re.search('role.*cannot be assumed', message):
            return True
        if re.search('role.*does not have permissions', message):
            return True
        if re.search('InvalidArnException.*valid principal', message):
            return True
        return False

    def func_l4isfo4p(self, error: Exception, context: LambdaErrorContext) -> Exception:
        error_cls: Type[LambdaClientError] = LambdaClientError
        if isinstance(error, RequestsConnectionError) and context.deployment_size > MAX_LAMBDA_DEPLOYMENT_SIZE:
            error_cls = DeploymentPackageTooLargeError
        elif isinstance(error, ClientError):
            code = error.response['Error'].get('Code', '')
            message = error.response['Error'].get('Message', '')
            if code == 'RequestEntityTooLargeException':
                error_cls = DeploymentPackageTooLargeError
            elif code == 'InvalidParameterValueException' and 'Unzipped size must be smaller' in message:
                error_cls = DeploymentPackageTooLargeError
        return error_cls(error, context)

    def func_tk8dbgmd(self, function_name: str) -> None:
        lambda_client = self._client('lambda')
        try:
            lambda_client.delete_function(FunctionName=function_name)
        except lambda_client.exceptions.ResourceNotFoundException:
            raise ResourceDoesNotExistError(function_name)

    def func_2557f71j(
        self,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'DomainName': domain_name,
            'DomainNameConfigurations': [{
                'ApiGatewayDomainName': domain_name,
                'CertificateArn': certificate_arn,
                'EndpointType': endpoint_type,
                'SecurityPolicy': security_policy,
                'DomainNameStatus': 'AVAILABLE'
            }]
        }
        if tags:
            kwargs['Tags'] = tags
        return kwargs

    def func_lp7wfd2s(
        self,
        certificate_arn: str,
        endpoint_type: str,
        security_policy: Optional[str] = None
    ) -> List[Dict[str, str]]:
        patch_operations: List[Dict[str, str]] = []
        if security_policy is not None:
            patch_operations.append({'op': 'replace', 'path': '/securityPolicy', 'value': security_policy})
        if endpoint_type == 'EDGE':
            patch_operations.append({'op': 'replace', 'path': '/certificateArn', 'value': certificate_arn})
        else:
            patch_operations.append({'op': 'replace', 'path': '/regionalCertificateArn', 'value': certificate_arn})
        return patch_operations

    def func_z5lppxkl(
        self,
        protocol: str,
        domain_name: str,
        endpoint_type: str,
        certificate_arn: str,
        security_policy: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> DomainNameResponse:
        if protocol == 'HTTP':
            patch_operations = self.get_custom_domain_patch_operations(certificate_arn, endpoint_type, security_policy)
            updated_domain_name = self._update_domain_name(domain_name, patch_operations)
        elif protocol == 'WEBSOCKET':
            kwargs = self.get_custom_domain_params_v2(
                domain_name=domain_name,
                endpoint_type=endpoint_type,
                security_policy=security_policy,
                certificate_arn=certificate_arn
            )
            updated_domain_name = self._update_domain_name_v2(kwargs)
        else:
            raise ValueError('Unsupported protocol value.')
        resource_arn = f'arn:{self.partition_name}:apigateway:{self.region_name}::/domainnames/{domain_name}'
        self._update_resource_tags(resource_arn, tags)
        return updated_domain_name

    def func_rcqdg7d6(
        self,
        resource_arn: str,
        requested_tags: Optional[Dict[str, str]]
    ) -> None:
        if not requested_tags:
            requested_tags = {}
        remote_tags = self._client('apigatewayv2').get_tags(ResourceArn=resource_arn)['Tags']
        self._remove_unrequested_resource_tags(resource_arn, requested_tags, remote_tags)
        self._add_missing_or_differing_value_resource_tags(resource_arn, requested_tags, remote_tags)

    def func_vkxd01st(
        self,
        resource_arn: str,
        requested_tags: Dict[str, str],
        remote_tags: Dict[str, str]
    ) -> None:
        tag_keys_to_remove = list(set(remote_tags) - set(requested_tags))
        if tag_keys_to_remove:
            self._client('apigatewayv2').untag_resource(ResourceArn=resource_arn, TagKeys=tag_keys_to_remove)

    def func_pmw0u5hk(
        self,
        resource_arn: str,
        requested_tags: Dict[str, str],
        remote_tags: Dict[str, str]
    ) -> None:
        tags_to_add = {k: v for k, v in requested_tags.items() if k not in remote_tags or v != remote_tags[k]}
        if tags_to_add:
            self._client('apigatewayv2').tag_resource(ResourceArn=resource_arn, Tags=tags_to_add)

    def func_qfgqsrvy(
        self,
        custom_domain_name: str,
        patch_operations: List[Dict[str, str]]
    ) -> DomainNameResponse:
        client = self._client('apigateway')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result: Dict[str, Any] = {}
        for patch_operation in patch_operations:
            api_args = {'domainName': custom_domain_name, 'patchOperations': [patch_operation]}
            response = self._call_client_method_with_retries(
                client.update_domain_name,
                api_args,
                max_attempts=6,
                should_retry=lambda x: True,
                retryable_exceptions=exceptions
            )
            result.update(response)
        if result.get('regionalCertificateArn'):
            certificate_arn = result['regionalCertificateArn']
        else:
            certificate_arn = result['certificateArn']
        if result.get('regionalHostedZoneId'):
            hosted_zone_id = result['regionalHostedZoneId']
        else:
            hosted_zone_id = result['distributionHostedZoneId']
        if result.get('regionalDomainName') is not None:
            alias_domain_name = result['regionalDomainName']
        else:
            alias_domain_name = result['distributionDomainName']
        domain_name: DomainNameResponse = {
            'domain_name': result['domainName'],
            'security_policy': result['securityPolicy'],
            'certificate_arn': certificate_arn,
            'hosted_zone_id': hosted_zone_id,
            'alias_domain_name': alias_domain_name
        }
        return domain_name

    def func_mpzp8g1s(self, api_args: Dict[str, Any]) -> DomainNameResponse:
        client = self._client('apigatewayv2')
        exceptions = (client.exceptions.TooManyRequestsException,)
        result = self._call_client_method_with_retries(
            client.update_domain_name,
            api_args,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions
        )
        result_data = result['DomainNameConfigurations'][0]
        domain_name: DomainNameResponse = {
            'domain_name': result['DomainName'],
            'alias_domain_name': result_data['ApiGatewayDomainName'],
            'security_policy': result_data['SecurityPolicy'],
            'hosted_zone_id': result_data['HostedZoneId'],
            'certificate_arn': result_data['CertificateArn']
        }
        return domain_name

    def func_tr1zq3y4(self, domain_name: str) -> None:
        client = self._client('apigatewayv2')
        params: Dict[str, str] = {'DomainName': domain_name}
        exceptions = (client.exceptions.TooManyRequestsException,)
        self._call_client_method_with_retries(
            client.delete_domain_name,
            params,
            max_attempts=6,
            should_retry=lambda x: True,
            retryable_exceptions=exceptions
        )

    def func_6jp45fue(self, domain_name: str, path_key: str) -> None:
        client = self._client('apigateway')
        params: Dict[str, str] = {'domainName': domain_name, 'basePath': path_key}
        client.delete_base_path_mapping(**params)

    def func_23b5nmar(
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
        layers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Update a Lambda function's code and configuration.

        This method only updates the values provided to it. If a parameter
        is not provided, no changes will be made for that that parameter on
        the targeted lambda function.
        """
        return_value = self._update_function_code(function_name=function_name, zip_contents=zip_contents)
        self._update_function_config(
            environment_variables=environment_variables,
            runtime=runtime,
            timeout=timeout,
            memory_size=memory_size,
            role_arn=role_arn,
            xray=xray,
            subnet_ids=subnet_ids,
            security_group_ids=security_group_ids,
            function_name=function_name,
            layers=layers
        )
        if tags is not None:
            self._update_function_tags(return_value['FunctionArn'], tags)
        return return_value

    def func_dxalo1db(self, function_name: str, zip_contents: bytes) -> Dict[str, Any]:
        lambda_client = self._client('lambda')
        try:
            result = lambda_client.update_function_code(FunctionName=function_name, ZipFile=zip_contents)
        except _REMOTE_CALL_ERRORS as e:
            context = LambdaErrorContext(function_name, 'update_function_code', len(zip_contents))
            raise self._get_lambda_code_deployment_error(e, context)
        if result['LastUpdateStatus'] != 'Successful':
            self._wait_for_function_update(function_name)
        return result

    def func_bywp6toh(self, function_name: str) -> None:
        client = self._client('lambda')
        waiter = client.get_waiter('function_updated')
        waiter.wait(FunctionName=function_name)

    def func_b4cij0iy(self, function_name: str, reserved_concurrent_executions: int) -> None:
        lambda_client = self._client('lambda')
        lambda_client.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=reserved_concurrent_executions
        )

    def func_zj38hyq0(self, function_name: str) -> None:
        lambda_client = self._client('lambda')
        lambda_client.delete_function_concurrency(FunctionName=function_name)

    def func_ytlk16uo(
        self,
        environment_variables: Optional[Dict[str, str]],
        runtime: Optional[str],
        timeout: Optional[int],
        memory_size: Optional[int],
        role_arn: Optional[str],
        subnet_ids: Optional[List[str]],
        security_group_ids: Optional[List[str]],
        function_name: str,
        layers: Optional[List[str]],
        xray: Optional[bool]
    ) -> None:
        kwargs: Dict[str, Any] = {}
        if environment_variables is not None:
            kwargs['Environment'] = {'Variables': environment_variables}
        if runtime is not None:
            kwargs['Runtime'] = runtime
        if timeout is not None:
            kwargs['Timeout'] = timeout
        if memory_size is not None:
            kwargs['MemorySize'] = memory_size
        if role_arn is not None:
            kwargs['Role'] = role_arn
        if xray:
            kwargs['TracingConfig'] = {'Mode': 'Active'}
        if security_group_ids is not None and subnet_ids is not None:
            kwargs['VpcConfig'] = self._create_vpc_config(
                subnet_ids=subnet_ids,
                security_group_ids=security_group_ids
            )
        if layers is not None:
            kwargs['Layers'] = layers
        if kwargs:
            self._do_update_function_config(function_name, kwargs)

    def func_spy1q94i(self, function_name: str, kwargs: Dict[str, Any]) -> None:
        kwargs['FunctionName'] = function_name
        lambda_client = self._client('lambda')
        result = self._call_client_method_with_retries(
            lambda_client.update_function_configuration,
            kwargs,
            max_attempts=self.LAMBDA_CREATE_ATTEMPTS
        )
        if result['LastUpdateStatus'] != 'Successful':
            self._wait_for_function_update(function_name)

    def func_k1qh38jc(self, function_arn: str, requested_tags: Dict[str, str]) -> None:
        remote_tags = self._client('lambda').list_tags(Resource=function_arn)['Tags']
        self._remove_unrequested_remote_tags(function_arn, requested_tags, remote_tags)
        self._add_missing_or_differing_value_requested_tags(function_arn, requested_tags, remote_tags)

    def func_n3im3ilf(
        self,
        function_arn: str,
        requested_tags: Dict[str, str],
        remote_tags: Dict[str, str]
    ) -> None:
        tag_keys_to_remove = list(set(remote_tags) - set(requested_tags))
        if tag_keys_to_remove:
            self._client('lambda').untag_resource(Resource=function_arn, TagKeys=tag_keys_to_remove)

    def func_yq6yzob5(
        self,
        function_arn: str,
        requested_tags: Dict[str, str],
        remote_tags: Dict[str, str]
    ) -> None:
        tags_to_add = {k: v for k, v in requested_tags.items() if k not in remote_tags or v != remote_tags[k]}
        if tags_to_add:
            self._client('lambda').tag_resource(Resource=function_arn, Tags=tags_to_add)

    def func_6puezs1n(self, name: str) -> str:
        role = self.get_role(name)
        return role['Arn']

    def func_3w2fdmhu(self, name: str) -> Dict[str, Any]:
        client = self._client('iam')
        try:
            role = client.get_role(RoleName=name)
        except client.exceptions.NoSuchEntityException:
            raise ResourceDoesNotExistError(f'No role ARN found for: {name}')
        return role['Role']

    def func_tk9laisd(self, role_name: str, policy_name: str) -> None:
        self._client('iam').delete_role_policy(RoleName=role_name, PolicyName=policy_name)

    def func_w0zscxdr(
        self,
        role_name: str,
        policy_name: str,
        policy_document: Dict[str, Any]
    ) -> None:
        self._client('iam').put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document, indent=2)
        )

    def func_3vfdu47a(
        self,
        name: str,
        trust_policy: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> str:
        client = self._client('iam')
        response = client.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust_policy)
        )
        role_arn = response['Role']['Arn']
        try:
            self.put_role_policy(role_name=name, policy_name=name, policy_document=policy)
        except client.exceptions.MalformedPolicyDocumentException as e:
            self.delete_role(name=name)
            raise e
        return role_arn

    def func_zes92eei(self, name: str) -> None:
        """Delete a role by first deleting all inline policies."""
        client = self._client('iam')
        inline_policies = client.list_role_policies(RoleName=name)['PolicyNames']
        for policy_name in inline_policies:
            self.delete_role_policy(name, policy_name)
        client.delete_role(RoleName=name)

    def func_ni8m5rdy(self, name: str) -> bool:
        """Check if a CloudWatch LOG GROUP exists."""
        client = self._client('logs')
        result = client.describe_log_groups(logGroupNamePrefix=name)
        if len(result['logGroups']) == 0:
            return False
        return True

    def func_xp6dljiw(self, log_group_name: str) -> None:
        self._client('logs').create_log_group(logGroupName=log_group_name)

    def func_986tvntw(self, log_group_name: str) -> None:
        self._client('logs').delete_retention_policy(logGroupName=log_group_name)

    def func_57ps951j(self, log_group_name: str) -> None:
        self._client('logs').delete_log_group(logGroupName=log_group_name)

    def func_sj2igmse(self, name: str, retention_in_days: int) -> None:
        self._client('logs').put_retention_policy(logGroupName=name, retentionInDays=retention_in_days)

    def func_ez6pgu7u(
        self,
        name: str
    ) -> Optional[str]:
        """Get rest api id associated with an API name.

        :type name: str
        :param name: The name of the rest api.

        :rtype: str
        :return: If the rest api exists, then the restApiId
            is returned, otherwise None.

        """
        rest_apis = self._client('apigateway').get_rest_apis()['items']
        for api in rest_apis:
            if api['name'] == name:
                return api['id']
        return None

    def func_jd051mg3(self, rest_api_id: str) -> Dict[str, Any]:
        """Check if an API Gateway REST API exists."""
        client = self._client('apigateway')
        try:
            result = client.get_rest_api(restApiId=rest_api_id)
            result.pop('ResponseMetadata', None)
            return result
        except client.exceptions.NotFoundException:
            return {}

    def func_bppa19yl(self, swagger_document: Dict[str, Any], endpoint_type: str) -> str:
        """Import a REST API from a Swagger document."""
        client = self._client('apigateway')
        response = client.import_rest_api(
            body=json.dumps(swagger_document, indent=2),
            parameters={'endpointConfigurationTypes': endpoint_type}
        )
        rest_api_id = response['id']
        return rest_api_id

    def func_ns9y8s5y(self, rest_api_id: str, swagger_document: Dict[str, Any]) -> None:
        client = self._client('apigateway')
        client.put_rest_api(restApiId=rest_api_id, mode='overwrite', body=json.dumps(swagger_document, indent=2))

    def func_kify8vx7(self, rest_api_id: str, patch_operations: List[Dict[str, Any]]) -> None:
        client = self._client('apigateway')
        client.update_rest_api(restApiId=rest_api_id, patchOperations=patch_operations)

    def func_hnaxoj0m(self, rest_api_id: str) -> None:
        client = self._client('apigateway')
        try:
            client.delete_rest_api(restApiId=rest_api_id)
        except client.exceptions.NotFoundException:
            raise ResourceDoesNotExistError(rest_api_id)

    def func_eu8e72qf(
        self,
        rest_api_id: str,
        api_gateway_stage: str = DEFAULT_STAGE_NAME,
        xray: bool = False
    ) -> None:
        client = self._client('apigateway')
        client.create_deployment(
            restApiId=rest_api_id,
            stageName=api_gateway_stage,
            tracingEnabled=bool(xray)
        )

    def func_2b2fcg0s(
        self,
        function_name: str,
        region_name: str,
        account_id: str,
        rest_api_id: str,
        random_id: Optional[str] = None
    ) -> None:
        """Authorize API gateway to invoke a lambda function is needed.

        This method will first check if API gateway has permission to call
        the lambda function, and only if necessary will it invoke
        ``self.add_permission_for_apigateway(...).

        """
        source_arn = self._build_source_arn_str(region_name, account_id, rest_api_id)
        self._add_lambda_permission_if_needed(
            source_arn=source_arn,
            function_arn=function_name,
            service_name='apigateway'
        )

    def func_vxngs5kl(
        self,
        function_name: str,
        region_name: str,
        account_id: str,
        api_id: str,
        random_id: Optional[str] = None
    ) -> None:
        """Authorize API gateway v2 to invoke a lambda function."""
        source_arn = self._build_source_arn_str(region_name, account_id, api_id)
        self._add_lambda_permission_if_needed(
            source_arn=source_arn,
            function_arn=function_name,
            service_name='apigateway'
        )

    def func_4jlg1pxe(
        self,
        function_name: str
    ) -> Dict[str, Any]:
        """Return the function policy for a lambda function.

        This function will extract the policy string as a json document
        and return the json.loads(...) version of the policy.

        """
        client = self._client('lambda')
        try:
            policy = client.get_policy(FunctionName=function_name)
            return json.loads(policy['Policy'])
        except client.exceptions.ResourceNotFoundException:
            return {'Statement': []}

    def func_gpl6vd8m(
        self,
        rest_api_id: str,
        output_dir: str,
        api_gateway_stage: str = DEFAULT_STAGE_NAME,
        sdk_type: str = 'javascript'
    ) -> None:
        """Download an SDK to a directory.

        This will generate an SDK and download it to the provided
        ``output_dir``.  If you're using ``get_sdk_download_stream()``,
        you have to handle downloading the stream and unzipping the
        contents yourself.  This method handles that for you.

        """
        zip_stream = self.get_sdk_download_stream(rest_api_id, api_gateway_stage=api_gateway_stage, sdk_type=sdk_type)
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, 'sdk.zip'), 'wb') as f:
            f.write(zip_stream.read())
        tmp_extract = os.path.join(tmpdir, 'extracted')
        with zipfile.ZipFile(os.path.join(tmpdir, 'sdk.zip')) as z:
            z.extractall(tmp_extract)
        dirnames = os.listdir(tmp_extract)
        if len(dirnames) == 1:
            full_dirname = os.path.join(tmp_extract, dirnames[0])
            if os.path.isdir(full_dirname):
                final_dirname = f'chalice-{sdk_type}-sdk'
                full_renamed_name = os.path.join(tmp_extract, final_dirname)
                os.rename(full_dirname, full_renamed_name)
                shutil.move(full_renamed_name, output_dir)
                return
        raise RuntimeError(f'The downloaded SDK had an unexpected directory structure: {", ".join(dirnames)}')

    def func_9tuxudfz(
        self,
        rest_api_id: str,
        api_gateway_stage: str = DEFAULT_STAGE_NAME,
        sdk_type: str = 'javascript'
    ) -> IO[bytes]:
        """Generate an SDK for a given SDK.

        Returns a file like object that streams a zip contents for the
        generated SDK.

        """
        response = self._client('apigateway').get_sdk(
            restApiId=rest_api_id,
            stageName=api_gateway_stage,
            sdkType=sdk_type
        )
        return response['body']

    def func_zhycte73(self, topic_arn: str, function_arn: str) -> str:
        sns_client = self._client('sns')
        response = sns_client.subscribe(TopicArn=topic_arn, Protocol='lambda', Endpoint=function_arn)
        return response['SubscriptionArn']

    def func_bhtd55ch(self, subscription_arn: str) -> None:
        sns_client = self._client('sns')
        sns_client.unsubscribe(SubscriptionArn=subscription_arn)

    def func_x5hodffv(
        self,
        subscription_arn: str,
        topic_name: str,
        function_arn: str
    ) -> bool:
        """Verify a subscription arn matches the topic and function name.

        Given a subscription arn, verify that the associated topic name
        and function arn match up to the parameters passed in.

        """
        sns_client = self._client('sns')
        try:
            attributes = sns_client.get_subscription_attributes(SubscriptionArn=subscription_arn)['Attributes']
            return attributes['TopicArn'].rsplit(':', 1)[1] == topic_name and attributes['Endpoint'] == function_arn
        except sns_client.exceptions.NotFoundException:
            return False

    def func_cg6pb12y(
        self,
        topic_arn: str,
        function_arn: str
    ) -> None:
        self._add_lambda_permission_if_needed(
            source_arn=topic_arn,
            function_arn=function_arn,
            service_name='sns'
        )

    def func_n108f7sg(
        self,
        topic_arn: str,
        function_arn: str
    ) -> None:
        self._remove_lambda_permission_if_needed(
            source_arn=topic_arn,
            function_arn=function_arn,
            service_name='sns'
        )

    def func_0vp5ilaw(
        self,
        region_name: str,
        account_id: str,
        rest_api_id: str
    ) -> str:
        source_arn = (
            f'arn:{self.partition_name}:execute-api:{region_name}:{account_id}:{rest_api_id}/*'
        )
        return source_arn

    @property
    def func_jx1491gx(self) -> str:
        return self._client('apigateway').meta.partition

    @property
    def func_olopu58j(self) -> str:
        return self._client('apigateway').meta.region_name

    def func_qmebem38(
        self,
        log_group_name: str,
        start_time: Optional[datetime] = None,
        interleaved: bool = True
    ) -> Iterator[CWLogEvent]:
        logs = self._client('logs')
        paginator = logs.get_paginator('filter_log_events')
        pages = paginator.paginate(logGroupName=log_group_name, interleaved=interleaved)
        try:
            yield from self._iter_log_messages(pages)
        except logs.exceptions.ResourceNotFoundException:
            pass

    def func_lnupv60f(self, pages: Iterable[Dict[str, Any]]) -> Iterator[CWLogEvent]:
        for page in pages:
            events = page['events']
            for event in events:
                event['ingestionTime'] = self._convert_to_datetime(event['ingestionTime'])
                event['timestamp'] = self._convert_to_datetime(event['timestamp'])
                yield event

    def func_ka5jtkkb(self, integer_timestamp: int) -> datetime:
        return datetime.utcfromtimestamp(integer_timestamp / 1000.0)

    def func_sf8mdqnt(
        self,
        log_group_name: str,
        start_time: Optional[datetime] = None,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        logs = self._client('logs')
        kwargs: Dict[str, Any] = {
            'logGroupName': log_group_name,
            'interleaved': True
        }
        if start_time is not None:
            kwargs['startTime'] = int(datetime2timestamp(start_time) * 1000)
        if next_token is not None:
            kwargs['nextToken'] = next_token
        try:
            response = logs.filter_log_events(**kwargs)
        except logs.exceptions.ResourceNotFoundException:
            return {'events': []}
        self._convert_types_on_response(response)
        return response

    def func_cl6qvt6l(self, response: Dict[str, Any]) -> None:
        response['events'] = list(self._iter_log_messages([response]))

    def func_7g47rsmk(self, service_name: str) -> botocore.client.BaseClient:
        if service_name not in self._client_cache:
            self._client_cache[service_name] = self._session.create_client(service_name)
        return self._client_cache[service_name]

    def func_9xszm1hc(
        self,
        rest_api_id: str,
        function_arn: str,
        random_id: Optional[str] = None
    ) -> None:
        client = self._client('apigateway')
        authorizers = client.get_authorizers(restApiId=rest_api_id)
        for authorizer in authorizers['items']:
            if function_arn in authorizer['authorizerUri']:
                authorizer_id = authorizer['id']
                break
        else:
            raise ResourceDoesNotExistError(
                f'Unable to find authorizer associated with function ARN: {function_arn}'
            )
        parts = function_arn.split(':')
        partition = parts[1]
        region_name = parts[3]
        account_id = parts[4]
        function_name = parts[-1]
        source_arn = f'arn:{partition}:execute-api:{region_name}:{account_id}:{rest_api_id}/authorizers/{authorizer_id}'
        dns_suffix = self.endpoint_dns_suffix('apigateway', region_name)
        if random_id is None:
            random_id = self._random_id()
        self._client('lambda').add_permission(
            Action='lambda:InvokeFunction',
            FunctionName=function_name,
            StatementId=random_id,
            Principal=self.service_principal('apigateway', self.region_name, dns_suffix),
            SourceArn=source_arn
        )

    def func_q7gtpvev(
        self,
        rule_name: str,
        schedule_expression: Optional[str] = None,
        event_pattern: Optional[Dict[str, Any]] = None,
        rule_description: Optional[str] = None
    ) -> str:
        events = self._client('events')
        params: Dict[str, Any] = {'Name': rule_name}
        if schedule_expression:
            params['ScheduleExpression'] = schedule_expression
        elif event_pattern:
            params['EventPattern'] = event_pattern
        else:
            raise ValueError('schedule_expression or event_pattern required')
        if rule_description is not None:
            params['Description'] = rule_description
        rule_arn = events.put_rule(**params)
        return rule_arn['RuleArn']

    def func_vf9rir6s(self, rule_name: str) -> None:
        events = self._client('events')
        events.remove_targets(Rule=rule_name, Ids=['1'])
        events.delete_rule(Name=rule_name)

    def func_kvpn27q8(self, rule_name: str, function_arn: str) -> None:
        events = self._client('events')
        events.put_targets(Rule=rule_name, Targets=[{'Id': '1', 'Arn': function_arn}])

    def func_7u0zl20i(
        self,
        rule_arn: str,
        function_arn: str
    ) -> None:
        self._add_lambda_permission_if_needed(
            source_arn=rule_arn,
            function_arn=function_arn,
            service_name='events'
        )

    def func_uvc0p2gq(
        self,
        bucket: str,
        function_arn: str,
        events: List[str],
        prefix: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> None:
        """Configure S3 bucket to invoke a lambda function.

        The S3 bucket must already have permission to invoke the
        lambda function before you call this function, otherwise
        the service will return an error.  You can add permissions
        by using the ``add_permission_for_s3_event`` below.  The
        ``events`` param matches the event strings supported by the
        service.

        This method also only supports a single prefix/suffix for now,
        which is what's offered in the Lambda console.

        """
        s3 = self._client('s3')
        existing_config = s3.get_bucket_notification_configuration(Bucket=bucket)
        existing_config.pop('ResponseMetadata', None)
        existing_lambda_config = existing_config.get('LambdaFunctionConfigurations', [])
        single_config: Dict[str, Any] = {'LambdaFunctionArn': function_arn, 'Events': events}
        filter_rules: List[Dict[str, str]] = []
        if prefix is not None:
            filter_rules.append({'Name': 'Prefix', 'Value': prefix})
        if suffix is not None:
            filter_rules.append({'Name': 'Suffix', 'Value': suffix})
        if filter_rules:
            single_config['Filter'] = {'Key': {'FilterRules': filter_rules}}
        new_config = self._merge_s3_notification_config(existing_lambda_config, single_config)
        existing_config['LambdaFunctionConfigurations'] = new_config
        s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing_config)

    def func_wun0x0vs(
        self,
        existing_config: List[Dict[str, Any]],
        new_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        final_config: List[Dict[str, Any]] = []
        added_config: bool = False
        for config in existing_config:
            if config['LambdaFunctionArn'] != new_config['LambdaFunctionArn']:
                final_config.append(config)
            else:
                final_config.append(new_config)
                added_config = True
        if not added_config:
            final_config.append(new_config)
        return final_config

    def func_mod0k3wx(self, bucket: str, function_arn: str, account_id: str) -> None:
        bucket_arn = f'arn:{self.partition_name}:s3:::{bucket}'
        self._add_lambda_permission_if_needed(
            source_arn=bucket_arn,
            function_arn=function_arn,
            service_name='s3',
            source_account=account_id
        )

    def func_h4wcxrig(self, bucket: str, function_arn: str, account_id: str) -> None:
        bucket_arn = f'arn:{self.partition_name}:s3:::{bucket}'
        self._remove_lambda_permission_if_needed(
            source_arn=bucket_arn,
            function_arn=function_arn,
            service_name='s3',
            source_account=account_id
        )

    def func_wt0uilhi(self, bucket: str, function_arn: str) -> None:
        s3 = self._client('s3')
        existing_config = s3.get_bucket_notification_configuration(Bucket=bucket)
        existing_config.pop('ResponseMetadata', None)
        existing_lambda_config = existing_config.get('LambdaFunctionConfigurations', [])
        new_lambda_config: List[Dict[str, Any]] = []
        for config in existing_lambda_config:
            if config['LambdaFunctionArn'] == function_arn:
                continue
            new_lambda_config.append(config)
        existing_config['LambdaFunctionConfigurations'] = new_lambda_config
        s3.put_bucket_notification_configuration(Bucket=bucket, NotificationConfiguration=existing_config)

    def func_fiau292f(
        self,
        source_arn: str,
        function_arn: str,
        service_name: str,
        source_account: Optional[str] = None
    ) -> None:
        policy = self.get_function_policy(function_arn)
        if self._policy_gives_access(policy, source_arn, service_name):
            return
        random_id = self._random_id()
        dns_suffix = self.endpoint_dns_suffix_from_arn(source_arn)
        kwargs: Dict[str, Any] = {
            'Action': 'lambda:InvokeFunction',
            'FunctionName': function_arn,
            'StatementId': random_id,
            'Principal': self.service_principal(service_name, self.region_name, dns_suffix),
            'SourceArn': source_arn
        }
        if source_account is not None:
            kwargs['SourceAccount'] = source_account
        self._client('lambda').add_permission(**kwargs)

    def func_uma6gfpg(
        self,
        policy: Dict[str, Any],
        source_arn: str,
        service_name: str
    ) -> bool:
        for statement in policy.get('Statement', []):
            if self._statement_gives_arn_access(statement, source_arn, service_name):
                return True
        return False

    def func_845xupur(
        self,
        statement: Dict[str, Any],
        source_arn: str,
        service_name: str,
        source_account: Optional[str] = None
    ) -> bool:
        dns_suffix = self.endpoint_dns_suffix_from_arn(source_arn)
        principal = self.service_principal(service_name, self.region_name, dns_suffix)
        if not statement['Action'] == 'lambda:InvokeFunction':
            return False
        if statement.get('Condition', {}).get('ArnLike', {}).get('AWS:SourceArn', '') != source_arn:
            return False
        if statement.get('Principal', {}).get('Service', '') != principal:
            return False
        if source_account is not None:
            if statement.get('Condition', {}).get('StringEquals', {}).get('AWS:SourceAccount', '') != source_account:
                return False
        return True

    def func_o46zemlw(
        self,
        source_arn: str,
        function_arn: str,
        service_name: str,
        source_account: Optional[str] = None
    ) -> None:
        client = self._client('lambda')
        policy = self.get_function_policy(function_arn)
        for statement in policy.get('Statement', []):
            kwargs: Dict[str, Any] = {
                'statement': statement,
                'source_arn': source_arn,
                'service_name': service_name
            }
            if source_account is not None:
                kwargs['source_account'] = source_account
            if self._statement_gives_arn_access(**kwargs):
                client.remove_permission(FunctionName=function_arn, StatementId=statement['Sid'])

    def func_dqbhjp6s(
        self,
        event_source_arn: str,
        function_name: str,
        batch_size: int,
        starting_position: Optional[str] = None,
        maximum_batching_window_in_seconds: int = 0,
        maximum_concurrency: Optional[int] = None
    ) -> str:
        lambda_client = self._client('lambda')
        batch_window = maximum_batching_window_in_seconds
        kwargs: Dict[str, Any] = {
            'EventSourceArn': event_source_arn,
            'FunctionName': function_name,
            'BatchSize': batch_size,
            'MaximumBatchingWindowInSeconds': batch_window
        }
        if maximum_concurrency:
            kwargs['ScalingConfig'] = {'MaximumConcurrency': maximum_concurrency}
        if starting_position is not None:
            kwargs['StartingPosition'] = starting_position
        return self._call_client_method_with_retries(
            lambda_client.create_event_source_mapping,
            kwargs,
            max_attempts=self.LAMBDA_CREATE_ATTEMPTS
        )['UUID']

    def func_9x1z0kcb(
        self,
        event_uuid: str,
        batch_size: int,
        maximum_batching_window_in_seconds: int = 0,
        maximum_concurrency: Optional[int] = None
    ) -> None:
        lambda_client = self._client('lambda')
        batch_window = maximum_batching_window_in_seconds
        kwargs: Dict[str, Any] = {
            'UUID': event_uuid,
            'BatchSize': batch_size,
            'MaximumBatchingWindowInSeconds': batch_window
        }
        if maximum_concurrency:
            kwargs['ScalingConfig'] = {'MaximumConcurrency': maximum_concurrency}
        self._call_client_method_with_retries(
            lambda_client.update_event_source_mapping,
            kwargs,
            max_attempts=10,
            should_retry=self._is_settling_error
        )

    def func_7ahn3zng(self, event_uuid: str) -> None:
        lambda_client = self._client('lambda')
        self._call_client_method_with_retries(
            lambda_client.delete_event_source_mapping,
            {'UUID': event_uuid},
            max_attempts=10,
            should_retry=self._is_settling_error
        )

    def func_f49lfi3l(
        self,
        event_uuid: str,
        resource_name: str,
        service_name: str,
        function_arn: str
    ) -> bool:
        """Check if the uuid matches the resource and function arn provided.

        Given a uuid representing an event source mapping for a lambda
        function, verify that the associated source arn
        and function arn match up to the parameters passed in.

        Instead of providing the event source arn, the resource name
        is provided along with the service name.  For example, if we're
        checking an SQS queue event source, the resource name would be
        the queue name (e.g. ``myqueue``) and the service would be ``sqs``.

        """
        client = self._client('lambda')
        try:
            attributes = client.get_event_source_mapping(UUID=event_uuid)
            actual_arn = attributes['EventSourceArn']
            arn_start, actual_name = actual_arn.rsplit(':', 1)
            return bool(
                actual_name == resource_name and
                re.match(f'^arn:aws[a-z\\-]*:{service_name}', arn_start) and
                attributes['FunctionArn'] == function_arn
            )
        except client.exceptions.ResourceNotFoundException:
            return False

    def func_uy28y06n(
        self,
        event_uuid: str,
        event_source_arn: str,
        function_arn: str
    ) -> bool:
        """Check if the uuid matches the event and function ARN.

        This is similar to verify_event_source_current, except that you provide
        an explicit event_source_arn here.  This is useful for cases where you
        know the event source ARN or where you can't construct the event source
        arn solely based on the resource_name and the service_name.

        """
        client = self._client('lambda')
        try:
            attributes = client.get_event_source_mapping(UUID=event_uuid)
        except client.exceptions.ResourceNotFoundException:
            return False
        return bool(event_source_arn == attributes['EventSourceArn'] and function_arn == attributes['FunctionArn'])

    def func_vl5bcllk(self, name: str) -> str:
        client = self._client('apigatewayv2')
        return self._call_client_method_with_retries(
            client.create_api,
            kwargs={
                'Name': name,
                'ProtocolType': 'WEBSOCKET',
                'RouteSelectionExpression': '$request.body.action'
            },
            max_attempts=10,
            should_retry=self._is_settling_error
        )['ApiId']

    def func_2obs9240(self, name: str) -> Optional[str]:
        apis = self._client('apigatewayv2').get_apis()['Items']
        for api in apis:
            if api['Name'] == name:
                return api['ApiId']
        return None

    def func_c0em9128(self, api_id: str) -> bool:
        """Check if an API Gateway WEBSOCKET API exists."""
        client = self._client('apigatewayv2')
        try:
            client.get_api(ApiId=api_id)
            return True
        except client.exceptions.NotFoundException:
            return False

    def func_54o5zxsm(self, api_id: str) -> None:
        client = self._client('apigatewayv2')
        try:
            client.delete_api(ApiId=api_id)
        except client.exceptions.NotFoundException:
            raise ResourceDoesNotExistError(api_id)

    def func_fyxvhvej(
        self,
        api_id: str,
        lambda_function: str,
        handler_type: str
    ) -> str:
        client = self._client('apigatewayv2')
        return client.create_integration(
            ApiId=api_id,
            ConnectionType='INTERNET',
            ContentHandlingStrategy='CONVERT_TO_TEXT',
            Description=handler_type,
            IntegrationType='AWS_PROXY',
            IntegrationUri=lambda_function
        )['IntegrationId']

    def func_9d7rejwq(
        self,
        api_id: str,
        route_key: str,
        integration_id: str
    ) -> None:
        client = self._client('apigatewayv2')
        client.create_route(
            ApiId=api_id,
            RouteKey=route_key,
            RouteResponseSelectionExpression='$default',
            Target=f'integrations/{integration_id}'
        )

    def func_w1w9iesn(self, api_id: str, routes: List[str]) -> None:
        client = self._client('apigatewayv2')
        for route_id in routes:
            client.delete_route(ApiId=api_id, RouteId=route_id)

    def func_nk212wib(self, api_id: str, integrations: List[str]) -> None:
        client = self._client('apigatewayv2')
        for integration_id in integrations:
            client.delete_integration(ApiId=api_id, IntegrationId=integration_id)

    def func_o4blkahn(self, api_id: str) -> str:
        client = self._client('apigatewayv2')
        return client.create_deployment(ApiId=api_id)['DeploymentId']

    def func_79u7oexy(self, api_id: str) -> List[str]:
        client = self._client('apigatewayv2')
        return [i['RouteId'] for i in client.get_routes(ApiId=api_id)['Items']]

    def func_67ngz41x(self, api_id: str) -> List[str]:
        client = self._client('apigatewayv2')
        return [item['IntegrationId'] for item in client.get_integrations(ApiId=api_id)['Items']]

    def func_poqy49sc(
        self,
        api_id: str,
        stage_name: str,
        deployment_id: str
    ) -> None:
        client = self._client('apigatewayv2')
        client.create_stage(ApiId=api_id, StageName=stage_name, DeploymentId=deployment_id)

    def func_ason3noc(
        self,
        method: Callable[..., Any],
        kwargs: Dict[str, Any],
        max_attempts: int,
        should_retry: Optional[Callable[[Exception], bool]] = None,
        delay_time: int = DELAY_TIME,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ) -> Any:
        attempts: int = 0
        if should_retry is None:
            should_retry = self._is_iam_role_related_error
        if not retryable_exceptions:
            client = self._client('lambda')
            retryable_exceptions = (
                client.exceptions.InvalidParameterValueException,
                client.exceptions.ResourceInUseException
            )
        while True:
            try:
                response = method(**kwargs)
            except retryable_exceptions as e:
                self._sleep(delay_time)
                attempts += 1
                if attempts >= max_attempts or not should_retry(e):
                    raise
                continue
            return response

    def func_d6bs0f10(self) -> str:
        return str(uuid.uuid4())
