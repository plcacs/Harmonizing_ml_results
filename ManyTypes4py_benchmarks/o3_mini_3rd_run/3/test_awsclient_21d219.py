from collections import OrderedDict
from typing import Any, Dict, List, Optional
import pytest
from chalice.awsclient import TypedAWSClient

@pytest.mark.parametrize(
    'service,region,endpoint',
    [
        (
            'sns',
            'us-east-1',
            OrderedDict(
                [
                    ('partition', 'aws'),
                    ('endpointName', 'us-east-1'),
                    ('protocols', ['http', 'https']),
                    ('hostname', 'sns.us-east-1.amazonaws.com'),
                    ('signatureVersions', ['v4']),
                    ('dnsSuffix', 'amazonaws.com'),
                ]
            ),
        ),
        (
            'sqs',
            'cn-north-1',
            OrderedDict(
                [
                    ('partition', 'aws-cn'),
                    ('endpointName', 'cn-north-1'),
                    ('protocols', ['http', 'https']),
                    ('sslCommonName', 'cn-north-1.queue.amazonaws.com.cn'),
                    ('hostname', 'sqs.cn-north-1.amazonaws.com.cn'),
                    ('signatureVersions', ['v4']),
                    ('dnsSuffix', 'amazonaws.com.cn'),
                ]
            ),
        ),
        ('dynamodb', 'mars-west-1', None),
    ],
)
def test_resolve_endpoint(
    stubbed_session: Any, service: str, region: str, endpoint: Optional[OrderedDict[str, Any]]
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.resolve_endpoint(service, region) is None
    else:
        assert endpoint.items() <= awsclient.resolve_endpoint(service, region).items()


@pytest.mark.parametrize(
    'arn,endpoint',
    [
        (
            'arn:aws:sns:us-east-1:123456:MyTopic',
            OrderedDict(
                [
                    ('partition', 'aws'),
                    ('endpointName', 'us-east-1'),
                    ('protocols', ['http', 'https']),
                    ('hostname', 'sns.us-east-1.amazonaws.com'),
                    ('signatureVersions', ['v4']),
                    ('dnsSuffix', 'amazonaws.com'),
                ]
            ),
        ),
        (
            'arn:aws-cn:sqs:cn-north-1:444455556666:queue1',
            OrderedDict(
                [
                    ('partition', 'aws-cn'),
                    ('endpointName', 'cn-north-1'),
                    ('protocols', ['http', 'https']),
                    ('sslCommonName', 'cn-north-1.queue.amazonaws.com.cn'),
                    ('hostname', 'sqs.cn-north-1.amazonaws.com.cn'),
                    ('signatureVersions', ['v4']),
                    ('dnsSuffix', 'amazonaws.com.cn'),
                ]
            ),
        ),
        ('arn:aws:dynamodb:mars-west-1:123456:table/MyTable', None),
    ],
)
def test_endpoint_from_arn(
    stubbed_session: Any, arn: str, endpoint: Optional[OrderedDict[str, Any]]
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.endpoint_from_arn(arn) is None
    else:
        assert endpoint.items() <= awsclient.endpoint_from_arn(arn).items()


@pytest.mark.parametrize(
    'service,region,dns_suffix',
    [
        ('sns', 'us-east-1', 'amazonaws.com'),
        ('sns', 'cn-north-1', 'amazonaws.com.cn'),
        ('dynamodb', 'mars-west-1', 'amazonaws.com'),
    ],
)
def test_endpoint_dns_suffix(stubbed_session: Any, service: str, region: str, dns_suffix: str) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix(service, region)


@pytest.mark.parametrize(
    'arn,dns_suffix',
    [
        ('arn:aws:sns:us-east-1:123456:MyTopic', 'amazonaws.com'),
        ('arn:aws-cn:sqs:cn-north-1:444455556666:queue1', 'amazonaws.com.cn'),
        ('arn:aws:dynamodb:mars-west-1:123456:table/MyTable', 'amazonaws.com'),
    ],
)
def test_endpoint_dns_suffix_from_arn(stubbed_session: Any, arn: str, dns_suffix: str) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix_from_arn(arn)


class TestServicePrincipal:
    @pytest.fixture
    def region(self) -> str:
        return 'bermuda-triangle-42'

    @pytest.fixture
    def url_suffix(self) -> str:
        return '.nowhere.null'

    @pytest.fixture
    def non_iso_suffixes(self) -> List[str]:
        return ['', '.amazonaws.com', '.amazonaws.com.cn']

    @pytest.fixture
    def awsclient(self, stubbed_session: Any) -> TypedAWSClient:
        return TypedAWSClient(stubbed_session)

    def test_unmatched_service(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal('taco.magic.food.com', 'us-east-1', 'amazonaws.com')
        assert result == 'taco.magic.food.com'

    def test_defaults(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal('lambda')
        assert result == 'lambda.amazonaws.com'

    def test_states(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['states']
        for suffix in non_iso_suffixes:
            for service in services:
                composed: str = '{}{}'.format(service, suffix)
                expected: str = '{}.{}.amazonaws.com'.format(service, region)
                result: str = awsclient.service_principal(composed, region, url_suffix)
                assert result == expected

    def test_codedeploy_and_logs(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['codedeploy', 'logs']
        for suffix in non_iso_suffixes:
            for service in services:
                composed: str = '{}{}'.format(service, suffix)
                expected: str = '{}.{}.{}'.format(service, region, url_suffix)
                result: str = awsclient.service_principal(composed, region, url_suffix)
                assert result == expected

    def test_ec2(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['ec2']
        for suffix in non_iso_suffixes:
            for service in services:
                composed: str = '{}{}'.format(service, suffix)
                expected: str = '{}.{}'.format(service, url_suffix)
                result: str = awsclient.service_principal(composed, region, url_suffix)
                assert result == expected

    def test_others(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['autoscaling', 'lambda', 'events', 'sns', 'sqs', 'foo-service']
        for suffix in non_iso_suffixes:
            for service in services:
                composed: str = '{}{}'.format(service, suffix)
                expected: str = '{}.amazonaws.com'.format(service)
                result: str = awsclient.service_principal(composed, region, url_suffix)
                assert result == expected

    def test_local_suffix(self, awsclient: TypedAWSClient, region: str, url_suffix: str) -> None:
        result: str = awsclient.service_principal('foo-service.local', region, url_suffix)
        assert result == 'foo-service.local'

    def test_states_iso(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal('states.amazonaws.com', 'us-iso-east-1', 'c2s.ic.gov')
        assert result == 'states.amazonaws.com'

    def test_states_isob(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal('states.amazonaws.com', 'us-isob-east-1', 'sc2s.sgov.gov')
        assert result == 'states.amazonaws.com'

    def test_iso_exceptions(self, awsclient: TypedAWSClient) -> None:
        services: List[str] = ['cloudhsm', 'config', 'workspaces']
        for service in services:
            expected: str = '{}.c2s.ic.gov'.format(service)
            result: str = awsclient.service_principal('{}.amazonaws.com'.format(service), 'us-iso-east-1', 'c2s.ic.gov')
            assert result == expected
