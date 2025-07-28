from collections import OrderedDict
from typing import Any, Optional, List, Dict

import pytest
from chalice.awsclient import TypedAWSClient


@pytest.mark.parametrize(
    "service,region,endpoint",
    [
        (
            "sns",
            "us-east-1",
            OrderedDict(
                [
                    ("partition", "aws"),
                    ("endpointName", "us-east-1"),
                    ("protocols", ["http", "https"]),
                    ("hostname", "sns.us-east-1.amazonaws.com"),
                    ("signatureVersions", ["v4"]),
                    ("dnsSuffix", "amazonaws.com"),
                ]
            ),
        ),
        (
            "sqs",
            "cn-north-1",
            OrderedDict(
                [
                    ("partition", "aws-cn"),
                    ("endpointName", "cn-north-1"),
                    ("protocols", ["http", "https"]),
                    ("sslCommonName", "cn-north-1.queue.amazonaws.com.cn"),
                    ("hostname", "sqs.cn-north-1.amazonaws.com.cn"),
                    ("signatureVersions", ["v4"]),
                    ("dnsSuffix", "amazonaws.com.cn"),
                ]
            ),
        ),
        ("dynamodb", "mars-west-1", None),
    ],
)
def test_resolve_endpoint(
    stubbed_session: Any, service: str, region: str, endpoint: Optional[OrderedDict[str, Any]]
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.resolve_endpoint(service, region) is None
    else:
        # type: ignore
        assert endpoint.items() <= awsclient.resolve_endpoint(service, region).items()


@pytest.mark.parametrize(
    "arn,endpoint",
    [
        (
            "arn:aws:sns:us-east-1:123456:MyTopic",
            OrderedDict(
                [
                    ("partition", "aws"),
                    ("endpointName", "us-east-1"),
                    ("protocols", ["http", "https"]),
                    ("hostname", "sns.us-east-1.amazonaws.com"),
                    ("signatureVersions", ["v4"]),
                    ("dnsSuffix", "amazonaws.com"),
                ]
            ),
        ),
        (
            "arn:aws-cn:sqs:cn-north-1:444455556666:queue1",
            OrderedDict(
                [
                    ("partition", "aws-cn"),
                    ("endpointName", "cn-north-1"),
                    ("protocols", ["http", "https"]),
                    ("sslCommonName", "cn-north-1.queue.amazonaws.com.cn"),
                    ("hostname", "sqs.cn-north-1.amazonaws.com.cn"),
                    ("signatureVersions", ["v4"]),
                    ("dnsSuffix", "amazonaws.com.cn"),
                ]
            ),
        ),
        ("arn:aws:dynamodb:mars-west-1:123456:table/MyTable", None),
    ],
)
def test_endpoint_from_arn(
    stubbed_session: Any, arn: str, endpoint: Optional[OrderedDict[str, Any]]
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.endpoint_from_arn(arn) is None
    else:
        # type: ignore
        assert endpoint.items() <= awsclient.endpoint_from_arn(arn).items()


@pytest.mark.parametrize(
    "service,region,dns_suffix",
    [
        ("sns", "us-east-1", "amazonaws.com"),
        ("sns", "cn-north-1", "amazonaws.com.cn"),
        ("dynamodb", "mars-west-1", "amazonaws.com"),
    ],
)
def test_endpoint_dns_suffix(
    stubbed_session: Any, service: str, region: str, dns_suffix: str
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix(service, region)


@pytest.mark.parametrize(
    "arn,dns_suffix",
    [
        ("arn:aws:sns:us-east-1:123456:MyTopic", "amazonaws.com"),
        ("arn:aws-cn:sqs:cn-north-1:444455556666:queue1", "amazonaws.com.cn"),
        ("arn:aws:dynamodb:mars-west-1:123456:table/MyTable", "amazonaws.com"),
    ],
)
def test_endpoint_dns_suffix_from_arn(
    stubbed_session: Any, arn: str, dns_suffix: str
) -> None:
    awsclient: TypedAWSClient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix_from_arn(arn)


class TestServicePrincipal:
    @pytest.fixture
    def region(self) -> str:
        return "bermuda-triangle-42"

    @pytest.fixture
    def url_suffix(self) -> str:
        return ".nowhere.null"

    @pytest.fixture
    def non_iso_suffixes(self) -> List[str]:
        return ["", ".amazonaws.com", ".amazonaws.com.cn"]

    @pytest.fixture
    def awsclient(self, stubbed_session: Any) -> TypedAWSClient:
        return TypedAWSClient(stubbed_session)

    def test_unmatched_service(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal("taco.magic.food.com", "us-east-1", "amazonaws.com")
        assert result == "taco.magic.food.com"

    def test_defaults(self, awsclient: TypedAWSClient) -> None:
        result: str = awsclient.service_principal("lambda")
        assert result == "lambda.amazonaws.com"

    def test_states(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ["states"]
        for suffix in non_iso_suffixes:
            for service in services:
                sp: str = awsclient.service_principal(f"{service}{suffix}", region, url_suffix)
                assert sp == f"{service}.{region}.amazonaws.com"

    def test_codedeploy_and_logs(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ["codedeploy", "logs"]
        for suffix in non_iso_suffixes:
            for service in services:
                sp: str = awsclient.service_principal(f"{service}{suffix}", region, url_suffix)
                assert sp == f"{service}.{region}.{url_suffix}"

    def test_ec2(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ["ec2"]
        for suffix in non_iso_suffixes:
            for service in services:
                sp: str = awsclient.service_principal(f"{service}{suffix}", region, url_suffix)
                assert sp == f"{service}.{url_suffix}"

    def test_others(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
        services: List[str] = ["autoscaling", "lambda", "events", "sns", "sqs", "foo-service"]
        for suffix in non_iso_suffixes:
            for service in services:
                sp: str = awsclient.service_principal(f"{service}{suffix}", region, url_suffix)
                assert sp == f"{service}.amazonaws.com"

    def test_local_suffix(self, awsclient: TypedAWSClient, region: str, url_suffix: str) -> None:
        sp: str = awsclient.service_principal("foo-service.local", region, url_suffix)
        assert sp == "foo-service.local"

    def test_states_iso(self, awsclient: TypedAWSClient) -> None:
        sp: str = awsclient.service_principal("states.amazonaws.com", "us-iso-east-1", "c2s.ic.gov")
        assert sp == "states.amazonaws.com"

    def test_states_isob(self, awsclient: TypedAWSClient) -> None:
        sp: str = awsclient.service_principal("states.amazonaws.com", "us-isob-east-1", "sc2s.sgov.gov")
        assert sp == "states.amazonaws.com"

    def test_iso_exceptions(self, awsclient: TypedAWSClient) -> None:
        services: List[str] = ["cloudhsm", "config", "workspaces"]
        for service in services:
            sp: str = awsclient.service_principal(f"{service}.amazonaws.com", "us-iso-east-1", "c2s.ic.gov")
            assert sp == f"{service}.c2s.ic.gov"