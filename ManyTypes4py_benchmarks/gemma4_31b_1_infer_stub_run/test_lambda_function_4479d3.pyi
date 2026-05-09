import io
import json
import zipfile
from typing import Any, Dict, Optional, Union, Callable, Awaitable
import boto3
from botocore.response import StreamingBody
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction

lambda_mock: Callable[..., Any] = ...
iam_mock: Callable[..., Any] = ...
mock_iam_rule: Callable[..., Any] = ...

def handler_a(event: Any, context: Any) -> Dict[str, Any]: ...

LAMBDA_TEST_CODE: str = ...

mock_lambda_code: Callable[..., Any] = ...
mock_lambda_function: Callable[..., Any] = ...

def handler_b(event: Any, context: Any) -> Dict[str, Any]: ...

LAMBDA_TEST_CODE_V2: str = ...

mock_lambda_code_v2: Callable[..., Any] = ...
add_lambda_version: Callable[..., Any] = ...
lambda_function: Callable[..., Any] = ...

def make_patched_invocation(client: Any, handler: Callable[[Any, Any], Any]) -> Callable[..., Any]: ...

mock_invoke_base: Callable[..., Any] = ...
mock_invoke_updated: Callable[..., Any] = ...

class TestLambdaFunction:
    def test_init(self, aws_credentials: AwsCredentials) -> None: ...
    def test_invoke_lambda_payloads(
        self, 
        payload: Optional[Dict[str, Any]], 
        expected: Dict[str, Any], 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_invoke_lambda_async_dispatch(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_invoke_lambda_force_sync(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    def test_invoke_lambda_tail(
        self, 
        lambda_function: LambdaFunction, 
        mock_lambda_function: Dict[str, Any], 
        mock_invoke_base: Any
    ) -> None: ...
    def test_invoke_lambda_client_context(
        self, 
        lambda_function: LambdaFunction, 
        mock_lambda_function: Dict[str, Any], 
        mock_invoke_base: Any
    ) -> None: ...
    def test_invoke_lambda_qualifier_base_version(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    def test_invoke_lambda_qualifier_updated_version(
        self, 
        add_lambda_version: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_updated: Any
    ) -> None: ...

class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_ainvoke_lambda_payloads(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_ainvoke_lambda_tail(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_ainvoke_lambda_client_context(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_base_version(
        self, 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...
    async def test_ainvoke_lambda_qualifier_updated_version(
        self, 
        add_lambda_version: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_updated: Any
    ) -> None: ...