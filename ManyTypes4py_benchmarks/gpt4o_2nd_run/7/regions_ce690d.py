import logging
import re
from typing import Dict, List, Optional, Union
from botocore.exceptions import NoRegionError

LOG = logging.getLogger(__name__)
DEFAULT_URI_TEMPLATE = '{service}.{region}.{dnsSuffix}'
DEFAULT_SERVICE_DATA = {'endpoints': {}}

class BaseEndpointResolver(object):
    """Resolves regions and endpoints. Must be subclassed."""

    def construct_endpoint(self, service_name: str, region_name: Optional[str] = None) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        """Resolves an endpoint for a service and region combination."""
        raise NotImplementedError

    def get_available_partitions(self) -> List[str]:
        """Lists the partitions available to the endpoint resolver."""
        raise NotImplementedError

    def get_available_endpoints(self, service_name: str, partition_name: str = 'aws', allow_non_regional: bool = False) -> List[str]:
        """Lists the endpoint names of a particular partition."""
        raise NotImplementedError

class EndpointResolver(BaseEndpointResolver):
    """Resolves endpoints based on partition endpoint metadata"""

    def __init__(self, endpoint_data: Dict[str, List[Dict[str, Union[str, Dict[str, Dict[str, Dict[str, Union[str, Dict[str, str]]]]]]]]]) -> None:
        if 'partitions' not in endpoint_data:
            raise ValueError('Missing "partitions" in endpoint data')
        self._endpoint_data = endpoint_data

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

    def construct_endpoint(self, service_name: str, region_name: Optional[str] = None, partition_name: Optional[str] = None) -> Optional[Dict[str, Union[str, List[str], Dict[str, str]]]]:
        if partition_name is not None:
            valid_partition = None
            for partition in self._endpoint_data['partitions']:
                if partition['partition'] == partition_name:
                    valid_partition = partition
            if valid_partition is not None:
                result = self._endpoint_for_partition(valid_partition, service_name, region_name, True)
                return result
            return None
        for partition in self._endpoint_data['partitions']:
            result = self._endpoint_for_partition(partition, service_name, region_name)
            if result:
                return result
        return None

    def _endpoint_for_partition(self, partition: Dict[str, Union[str, Dict[str, Dict[str, Dict[str, Union[str, Dict[str, str]]]]]]], service_name: str, region_name: Optional[str], force_partition: bool = False) -> Optional[Dict[str, Union[str, List[str], Dict[str, str]]]]:
        service_data = partition['services'].get(service_name, DEFAULT_SERVICE_DATA)
        if region_name is None:
            if 'partitionEndpoint' in service_data:
                region_name = service_data['partitionEndpoint']
            else:
                raise NoRegionError()
        if region_name in service_data['endpoints']:
            return self._resolve(partition, service_name, service_data, region_name)
        if self._region_match(partition, region_name) or force_partition:
            partition_endpoint = service_data.get('partitionEndpoint')
            is_regionalized = service_data.get('isRegionalized', True)
            if partition_endpoint and (not is_regionalized):
                LOG.debug('Using partition endpoint for %s, %s: %s', service_name, region_name, partition_endpoint)
                return self._resolve(partition, service_name, service_data, partition_endpoint)
            LOG.debug('Creating a regex based endpoint for %s, %s', service_name, region_name)
            return self._resolve(partition, service_name, service_data, region_name)
        return None

    def _region_match(self, partition: Dict[str, Union[str, Dict[str, Dict[str, Dict[str, Union[str, Dict[str, str]]]]]]], region_name: str) -> bool:
        if region_name in partition['regions']:
            return True
        if 'regionRegex' in partition:
            return re.compile(partition['regionRegex']).match(region_name) is not None
        return False

    def _resolve(self, partition: Dict[str, Union[str, Dict[str, Dict[str, Dict[str, Union[str, Dict[str, str]]]]]]], service_name: str, service_data: Dict[str, Union[str, Dict[str, str]]], endpoint_name: str) -> Dict[str, Union[str, List[str], Dict[str, str]]]:
        result = service_data['endpoints'].get(endpoint_name, {})
        result['partition'] = partition['partition']
        result['endpointName'] = endpoint_name
        self._merge_keys(service_data.get('defaults', {}), result)
        self._merge_keys(partition.get('defaults', {}), result)
        hostname = result.get('hostname', DEFAULT_URI_TEMPLATE)
        result['hostname'] = self._expand_template(partition, result['hostname'], service_name, endpoint_name)
        if 'sslCommonName' in result:
            result['sslCommonName'] = self._expand_template(partition, result['sslCommonName'], service_name, endpoint_name)
        result['dnsSuffix'] = partition['dnsSuffix']
        return result

    def _merge_keys(self, from_data: Dict[str, Union[str, List[str], Dict[str, str]]], result: Dict[str, Union[str, List[str], Dict[str, str]]]) -> None:
        for key in from_data:
            if key not in result:
                result[key] = from_data[key]

    def _expand_template(self, partition: Dict[str, Union[str, Dict[str, Dict[str, Dict[str, Union[str, Dict[str, str]]]]]]], template: str, service_name: str, endpoint_name: str) -> str:
        return template.format(service=service_name, region=endpoint_name, dnsSuffix=partition['dnsSuffix'])
