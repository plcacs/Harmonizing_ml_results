from collections import OrderedDict
import pytest
from chalice.awsclient import TypedAWSClient
from botocore.session import Session

def test_resolve_endpoint(stubbed_session: Session, service: str, region: str, endpoint: dict) -> None:
def test_endpoint_from_arn(stubbed_session: Session, arn: str, endpoint: dict) -> None:
def test_endpoint_dns_suffix(stubbed_session: Session, service: str, region: str, dns_suffix: str) -> None:
def test_endpoint_dns_suffix_from_arn(stubbed_session: Session, arn: str, dns_suffix: str) -> None:

class TestServicePrincipal(object):
    def region(self) -> str:
    def url_suffix(self) -> str:
    def non_iso_suffixes(self) -> List[str]:
    def awsclient(self, stubbed_session: Session) -> TypedAWSClient:

    def test_unmatched_service(self, awsclient: TypedAWSClient) -> None:
    def test_defaults(self, awsclient: TypedAWSClient) -> None:
    def test_states(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
    def test_codedeploy_and_logs(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
    def test_ec2(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
    def test_others(self, awsclient: TypedAWSClient, region: str, url_suffix: str, non_iso_suffixes: List[str]) -> None:
    def test_local_suffix(self, awsclient: TypedAWSClient, region: str, url_suffix: str) -> None:
    def test_states_iso(self, awsclient: TypedAWSClient) -> None:
    def test_states_isob(self, awsclient: TypedAWSClient) -> None:
    def test_iso_exceptions(self, awsclient: TypedAWSClient) -> None:
