import io
from typing import Any, Callable, Generator, Optional

import boto3
import pytest
from botocore.response import StreamingBody
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction

def lambda_mock(aws_credentials: AwsCredentials) -> Generator[Any, None, None]: ...
def iam_mock(aws_credentials: AwsCredentials) -> Generator[Any, None, None]: ...
def mock_iam_rule(iam_mock: Any) -> Generator[dict[str, Any], None, None]: ...

def handler_a(event: Any, context: Any) -> Any: ...

LAMBDA_TEST_CODE: str

def mock_lambda_code() -> Generator[bytes, None, None]: ...
def mock_lambda_function(lambda_mock: Any, mock_iam_rule: Any, mock_lambda_code: bytes) -> Generator[dict[str, Any], None, None]: ...

def handler_b(event: Any, context: Any) -> dict[str, list[int]]: ...

LAMBDA_TEST_CODE_V2: str

def mock_lambda_code_v2() -> Generator[bytes, None, None]: ...
def add_lambda_version(mock_lambda_function: dict[str, Any], lambda_mock: Any, mock_lambda_code_v2: bytes) -> Generator[dict[str, Any], None, None]: ...
def lambda_function(aws_credentials: AwsCredentials) -> LambdaFunction: ...
def make_patched_invocation(client: Any, handler: Callable[..., Any]) -> Callable[..., Any]: ...
def mock_invoke_base(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]: ...
def mock_invoke_updated(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]: ...

class TestLambdaFunction:
    def test_init(self, aws_credentials: AwsCredentials) -> None: ...
    @pytest.mark.parametrize("payload,expected", [
        ({"foo": "baz"}, {"foo": "bar"}),
        (None, {"foo": "bar"}),
    ])
    def test_invoke_lambda_payloads(
        self,
        payload: Optional[dict[str, str]],
        expected: dict[str, str],
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_invoke_lambda_async_dispatch(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_invoke_lambda_force_sync(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_tail(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: dict[str, Any],
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_client_context(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: dict[str, Any],
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    def test_invoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None: ...

class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_payloads(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_tail(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_client_context(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: dict[str, Any],
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None: ...