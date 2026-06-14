import io
import zipfile
from typing import Any, Callable, Dict, Optional

import boto3
import pytest
from botocore.response import StreamingBody
from mypy_boto3_iam import IAMClient
from mypy_boto3_lambda import LambdaClient
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction

LAMBDA_TEST_CODE: str
LAMBDA_TEST_CODE_V2: str

@pytest.fixture
def lambda_mock(aws_credentials: AwsCredentials) -> LambdaClient: ...

@pytest.fixture
def iam_mock(aws_credentials: AwsCredentials) -> IAMClient: ...

@pytest.fixture
def mock_iam_rule(iam_mock: IAMClient) -> Dict[str, Any]: ...

def handler_a(event: Any, context: Any) -> Any: ...

@pytest.fixture
def mock_lambda_code() -> bytes: ...

@pytest.fixture
def mock_lambda_function(
    lambda_mock: LambdaClient,
    mock_iam_rule: Dict[str, Any],
    mock_lambda_code: bytes,
) -> Dict[str, Any]: ...

def handler_b(event: Any, context: Any) -> Dict[str, Any]: ...

@pytest.fixture
def mock_lambda_code_v2() -> bytes: ...

@pytest.fixture
def add_lambda_version(
    mock_lambda_function: Dict[str, Any],
    lambda_mock: LambdaClient,
    mock_lambda_code_v2: bytes,
) -> Dict[str, Any]: ...

@pytest.fixture
def lambda_function(aws_credentials: AwsCredentials) -> LambdaFunction: ...

def make_patched_invocation(
    client: Any, handler: Callable[..., Any]
) -> Callable[..., Dict[str, Any]]: ...

@pytest.fixture
def mock_invoke_base(
    lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch
) -> None: ...

@pytest.fixture
def mock_invoke_updated(
    lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch
) -> None: ...

class TestLambdaFunction:
    def test_init(self, aws_credentials: AwsCredentials) -> None: ...
    @pytest.mark.parametrize("payload,expected", [
        ({"foo": "baz"}, {"foo": "bar"}),
        (None, {"foo": "bar"}),
    ])
    def test_invoke_lambda_payloads(
        self,
        payload: Optional[Dict[str, str]],
        expected: Dict[str, str],
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_invoke_lambda_async_dispatch(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_invoke_lambda_force_sync(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_tail(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: Dict[str, Any],
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_client_context(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: Dict[str, Any],
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None: ...

class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_payloads(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_tail(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_client_context(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: Dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None: ...