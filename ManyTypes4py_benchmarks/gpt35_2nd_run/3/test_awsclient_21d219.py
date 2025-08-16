from collections import OrderedDict
import pytest
from chalice.awsclient import TypedAWSClient
from botocore.session import Session

def test_resolve_endpoint(stubbed_session: Session, service: str, region: str, endpoint: dict) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.resolve_endpoint(service, region) is None
    else:
        assert endpoint.items() <= awsclient.resolve_endpoint(service, region).items()

def test_endpoint_from_arn(stubbed_session: Session, arn: str, endpoint: dict) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.endpoint_from_arn(arn) is None
    else:
        assert endpoint.items() <= awsclient.endpoint_from_arn(arn).items()

def test_endpoint_dns_suffix(stubbed_session: Session, service: str, region: str, dns_suffix: str) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix(service, region)

def test_endpoint_dns_suffix_from_arn(stubbed_session: Session, arn: str, dns_suffix: str) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix_from_arn(arn)

class TestServicePrincipal(object):

    def region(self) -> str:
        return 'bermuda-triangle-42'

    def url_suffix(self) -> str:
        return '.nowhere.null'

    def non_iso_suffixes(self) -> List[str]:
        return ['', '.amazonaws.com', '.amazonaws.com.cn']

    def awsclient(self, stubbed_session: Session) -> TypedAWSClient:
        return TypedAWSClient(stubbed_session)

    def test_unmatched_service(self, awsclient: TypedAWSClient) -> None:
        assert awsclient.service_principal('taco.magic.food.com', 'us-east-1', 'amazonaws.com') == 'taco.magic.food.com'

    def test_defaults(self, awsclient: TypedAWSClient) -> None:
        assert awsclient.service_principal('lambda') == 'lambda.amazonaws.com'

    def test_states(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['states']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}.amazonaws.com'.format(service, region)

    def test_codedeploy_and_logs(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['codedeploy', 'logs']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}.{}'.format(service, region, url_suffix)

    def test_ec2(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['ec2']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}'.format(service, url_suffix)

    def test_others(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ['autoscaling', 'lambda', 'events', 'sns', 'sqs', 'foo-service']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.amazonaws.com'.format(service)

    def test_local_suffix(self, awsclient: TypedAWSClient, region: str, url_suffix: str) -> None:
        assert awsclient.service_principal('foo-service.local', region, url_suffix) == 'foo-service.local'

    def test_states_iso(self, awsclient: TypedAWSClient) -> None:
        assert awsclient.service_principal('states.amazonaws.com', 'us-iso-east-1', 'c2s.ic.gov') == 'states.amazonaws.com'

    def test_states_isob(self, awsclient: TypedAWSClient) -> None:
        assert awsclient.service_principal('states.amazonaws.com', 'us-isob-east-1', 'sc2s.sgov.gov') == 'states.amazonaws.com'

    def test_iso_exceptions(self, awsclient: TypedAWSClient) -> None:
        services: List[str] = ['cloudhsm', 'config', 'workspaces']
        for service in services:
            assert awsclient.service_principal('{}.amazonaws.com'.format(service), 'us-iso-east-1', 'c2s.ic.gov') == '{}.c2s.ic.gov'.format(service)
