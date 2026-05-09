import botocore.client
import pytest
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction
from typing import Any, Dict, List, Optional, Union

class TestLambdaFunction:
    def test_init(aws_credentials: AwsCredentials) -> None: ...
    @pytest.mark.parametrize('payload,expected', [({'foo': 'baz'}, {'foo': 'bar'}), (None, {'foo': 'bar'})])
    def test_invoke_lambda_payloads(
        payload: Optional[Dict[str, Any]], 
        expected: Dict[str, str], 
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_invoke_lambda_async_dispatch(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_invoke_lambda_force_sync(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    def test_invoke_lambda_tail(
        lambda_function: LambdaFunction, 
        mock_lambda_function: Dict[str, Any], 
        mock_invoke_base: Any
    ) -> None: ...

    def test_invoke_lambda_client_context(
        lambda_function: LambdaFunction, 
        mock_lambda_function: Dict[str, Any], 
        mock_invoke_base: Any
    ) -> None: ...

    def test_invoke_lambda_qualifier_base_version(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    def test_invoke_lambda_qualifier_updated_version(
        add_lambda_version: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_updated: Any
    ) -> None: ...

class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_ainvoke_lambda_payloads(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_ainvoke_lambda_tail(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_ainvoke_lambda_client_context(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_ainvoke_lambda_qualifier_base_version(
        mock_lambda_function: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_base: Any
    ) -> None: ...

    async def test_ainvoke_lambda_qualifier_updated_version(
        add_lambda_version: Dict[str, Any], 
        lambda_function: LambdaFunction, 
        mock_invoke_updated: Any
    ) -> None: ...

class LambdaFunction:
    def __init__(self, function_name: str, aws_credentials: AwsCredentials, qualifier: Optional[str] = None) -> None: ...
    def invoke(
        self, 
        payload: Optional[Dict[str, Any]] = None, 
        tail: bool = False, 
        client_context: Optional[Dict[str, Any]] = None, 
        _sync: bool = False
    ) -> Dict[str, Any]: ...
    async def ainvoke(
        self, 
        payload: Optional[Dict[str, Any]] = None, 
        tail: bool = False, 
        client_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: ...
    def _get_lambda_client(self) -> botocore.client.Lambda: ...
    @property
    def qualifier(self) -> Optional[str]: ...
    @qualifier.setter
    def qualifier(self, value: Optional[str]) -> None: ...