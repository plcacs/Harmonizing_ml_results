import pytest
from boto3 import client
from io import BytesIO
from json import dumps
from typing import Any, Optional, Dict, List, Union
from moto import mock_iam, mock_lambda
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction
from pydantic_core import to_json
from prefect import flow

@pytest.fixture
def lambda_mock(aws_credentials: AwsCredentials) -> client:
    ...

@pytest.fixture
def iam_mock(aws_credentials: AwsCredentials) -> client:
    ...

@pytest.fixture
def mock_iam_rule(iam_mock: client) -> Dict[str, Dict[str, str]]:
    ...

@pytest.fixture
def mock_lambda_code() -> bytes:
    ...

@pytest.fixture
def mock_lambda_function(lambda_mock: client, mock_iam_rule: Dict[str, Dict[str, str]], mock_lambda_code: bytes) -> Dict[str, Union[str, Dict]]:
    ...

@pytest.fixture
def mock_lambda_code_v2() -> bytes:
    ...

@pytest.fixture
def add_lambda_version(mock_lambda_function: Dict[str, Union[str, Dict]], lambda_mock: client, mock_lambda_code_v2: bytes) -> Dict[str, Union[str, Dict]]:
    ...

@pytest.fixture
def lambda_function(aws_credentials: AwsCredentials) -> LambdaFunction:
    ...

def make_patched_invocation(client: client, handler: callable) -> callable:
    ...

@pytest.fixture
def mock_invoke_base(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.fixture
def mock_invoke_updated(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

class TestLambdaFunction:
    def test_init(self, aws_credentials: AwsCredentials) -> None:
        ...

    @pytest.mark.parametrize('payload,expected', [({'foo': 'baz'}, {'foo': 'bar'}), (None, {'foo': 'bar'})])
    def test_invoke_lambda_payloads(self, payload: Optional[Dict], expected: Dict, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_invoke_lambda_async_dispatch(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_invoke_lambda_force_sync(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    def test_invoke_lambda_tail(self, lambda_function: LambdaFunction, mock_lambda_function: Dict[str, Union[str, Dict]], mock_invoke_base: None) -> None:
        ...

    def test_invoke_lambda_client_context(self, lambda_function: LambdaFunction, mock_lambda_function: Dict[str, Union[str, Dict]], mock_invoke_base: None) -> None:
        ...

    def test_invoke_lambda_qualifier_base_version(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    def test_invoke_lambda_qualifier_updated_version(self, add_lambda_version: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_updated: None) -> None:
        ...

class TestLambdaFunctionAsync:
    @pytest.mark.asyncio
    async def test_ainvoke_lambda_explicit(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_ainvoke_lambda_payloads(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_ainvoke_lambda_tail(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_ainvoke_lambda_client_context(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_ainvoke_lambda_qualifier_base_version(self, mock_lambda_function: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        ...

    @pytest.mark.asyncio
    async def test_ainvoke_lambda_qualifier_updated_version(self, add_lambda_version: Dict[str, Union[str, Dict]], lambda_function: LambdaFunction, mock_invoke_updated: None) -> None:
        ...