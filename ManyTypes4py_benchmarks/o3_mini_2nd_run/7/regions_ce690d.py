#!/usr/bin/env python3
"""
Resolves regions and endpoints.

This module implements endpoint resolution, including resolving endpoints for a
given service and region and resolving the available endpoints for a service
in a specific AWS partition.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from botocore.exceptions import NoRegionError

LOG = logging.getLogger(__name__)
DEFAULT_URI_TEMPLATE = '{service}.{region}.{dnsSuffix}'
DEFAULT_SERVICE_DATA: Dict[str, Any] = {'endpoints': {}}


class BaseEndpointResolver(object):
    """Resolves regions and endpoints. Must be subclassed."""

    def construct_endpoint(self, service_name: str, region_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolves an endpoint for a service and region combination.

        :type service_name: string
        :param service_name: Name of the service to resolve an endpoint for
            (e.g., s3)

        :type region_name: string
        :param region_name: Region/endpoint name to resolve (e.g., us-east-1)
            if no region is provided, the first found partition-wide endpoint
            will be used if available.

        :rtype: dict
        :return: Returns a dict containing the following keys:
            - partition: (string, required) Resolved partition name
            - endpointName: (string, required) Resolved endpoint name
            - hostname: (string, required) Hostname to use for this endpoint
            - sslCommonName: (string) sslCommonName to use for this endpoint.
            - credentialScope: (dict) Signature version 4 credential scope
              - region: (string) region name override when signing.
              - service: (string) service name override when signing.
            - signatureVersions: (list<string>) A list of possible signature
              versions, including s3, v4, v2, and s3v4
            - protocols: (list<string>) A list of supported protocols
              (e.g., http, https)
            - ...: Other keys may be included as well based on the metadata
        """
        raise NotImplementedError

    def get_available_partitions(self) -> List[str]:
        """
        Lists the partitions available to the endpoint resolver.

        :return: Returns a list of partition names (e.g., ["aws", "aws-cn"]).
        """
        raise NotImplementedError

    def get_available_endpoints(self, service_name: str, partition_name: str = 'aws', allow_non_regional: bool = False) -> List[str]:
        """
        Lists the endpoint names of a particular partition.

        :type service_name: string
        :param service_name: Name of a service to list endpoint for (e.g., s3)

        :type partition_name: string
        :param partition_name: Name of the partition to limit endpoints to.
            (e.g., aws for the public AWS endpoints, aws-cn for AWS China
            endpoints, aws-us-gov for AWS GovCloud (US) Endpoints, etc.

        :type allow_non_regional: bool
        :param allow_non_regional: Set to True to include endpoints that are
             not regional endpoints (e.g., s3-external-1,
             fips-us-gov-west-1, etc).
        :return: Returns a list of endpoint names (e.g., ["us-east-1"]).
        """
        raise NotImplementedError


class EndpointResolver(BaseEndpointResolver):
    """Resolves endpoints based on partition endpoint metadata"""

    def __init__(self, endpoint_data: Dict[str, Any]) -> None:
        """
        :param endpoint_data: A dict of partition data.
        """
        if 'partitions' not in endpoint_data:
            raise ValueError('Missing "partitions" in endpoint data')
        self._endpoint_data: Dict[str, Any] = endpoint_data

    def get_available_partitions(self) -> List[str]:
        result: List[str] = []
        for partition in self._endpoint_data['partitions']:
            result.append(partition['partition'])
        return result

    def get_available_endpoints(self, service_name: str, partition_name: str = 'aws', allow_non_regional: bool = False) -> List[str]:
        result: List[str] = []
        for partition in self._endpoint_data['partitions']:
            if partition['partition'] != partition_name:
                continue
            services = partition['services']
            if service_name not in services:
                continue
            for endpoint_name in services[service_name]['endpoints']:
                if allow_non_regional or endpoint_name in partition['regions']:
                    result.append(endpoint_name)
        return result

    def construct_endpoint(self, service_name: str, region_name: Optional[str] = None, partition_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if partition_name is not None:
            valid_partition: Optional[Dict[str, Any]] = None
            for partition in self._endpoint_data['partitions']:
                if partition['partition'] == partition_name:
                    valid_partition = partition
            if valid_partition is not None:
                result: Optional[Dict[str, Any]] = self._endpoint_for_partition(valid_partition, service_name, region_name, True)
                return result
            return None
        for partition in self._endpoint_data['partitions']:
            result: Optional[Dict[str, Any]] = self._endpoint_for_partition(partition, service_name, region_name)
            if result:
                return result
        return None

    def _endpoint_for_partition(self, partition: Dict[str, Any], service_name: str, region_name: Optional[str], force_partition: bool = False) -> Optional[Dict[str, Any]]:
        service_data: Dict[str, Any] = partition['services'].get(service_name, DEFAULT_SERVICE_DATA)
        if region_name is None:
            if 'partitionEndpoint' in service_data:
                region_name = service_data['partitionEndpoint']
            else:
                raise NoRegionError()
        if region_name in service_data['endpoints']:
            return self._resolve(partition, service_name, service_data, region_name)
        if self._region_match(partition, region_name) or force_partition:
            partition_endpoint: Optional[str] = service_data.get('partitionEndpoint')
            is_regionalized: bool = service_data.get('isRegionalized', True)
            if partition_endpoint and (not is_regionalized):
                LOG.debug('Using partition endpoint for %s, %s: %s', service_name, region_name, partition_endpoint)
                return self._resolve(partition, service_name, service_data, partition_endpoint)
            LOG.debug('Creating a regex based endpoint for %s, %s', service_name, region_name)
            return self._resolve(partition, service_name, service_data, region_name)
        return None

    def _region_match(self, partition: Dict[str, Any], region_name: str) -> bool:
        if region_name in partition['regions']:
            return True
        if 'regionRegex' in partition:
            return re.compile(partition['regionRegex']).match(region_name) is not None
        return False

    def _resolve(self, partition: Dict[str, Any], service_name: str, service_data: Dict[str, Any], endpoint_name: str) -> Dict[str, Any]:
        result: Dict[str, Any] = service_data['endpoints'].get(endpoint_name, {})
        result['partition'] = partition['partition']
        result['endpointName'] = endpoint_name
        self._merge_keys(service_data.get('defaults', {}), result)
        self._merge_keys(partition.get('defaults', {}), result)
        hostname: str = result.get('hostname', DEFAULT_URI_TEMPLATE)
        result['hostname'] = self._expand_template(partition, hostname, service_name, endpoint_name)
        if 'sslCommonName' in result:
            result['sslCommonName'] = self._expand_template(partition, result['sslCommonName'], service_name, endpoint_name)
        result['dnsSuffix'] = partition['dnsSuffix']
        return result

    def _merge_keys(self, from_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        for key in from_data:
            if key not in result:
                result[key] = from_data[key]

    def _expand_template(self, partition: Dict[str, Any], template: str, service_name: str, endpoint_name: str) -> str:
        return template.format(service=service_name, region=endpoint_name, dnsSuffix=partition['dnsSuffix'])