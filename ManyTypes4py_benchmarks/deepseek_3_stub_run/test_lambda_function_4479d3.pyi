import io
from typing import Any, Callable, Dict, Optional, Union

import boto3
from botocore.response import StreamingBody
from prefect_aws.credentials import AwsCredentials


def handler_a(event: Optional[Dict[str, Any]], context: Any) -> Dict[str, Any]: ...
LAMBDA_TEST_CODE: str = ...


def handler_b(event: Any, context: Any) -> Dict[str, str]: ...
LAMBDA_TEST_CODE_V2: str = ...


class LambdaFunction:
    def __init__(
        self,
        function_name: str,
        aws_credentials: AwsCredentials,
        qualifier: Optional[str] = None,
    ) -> None: ...
    
    @property
    def function_name(self) -> str: ...
    
    @property
    def qualifier(self) -> Optional[str]: ...
    
    @qualifier.setter
    def qualifier(self, value: Optional[str]) -> None: ...
    
    def invoke(
        self,
        payload: Optional[Dict[str, Any]] = None,
        client_context: Optional[Dict[str, Any]] = None,
        tail: bool = False,
        _sync: bool = False,
    ) -> Dict[str, Any]: ...
    
    async def ainvoke(
        self,
        payload: Optional[Dict[str, Any]] = None,
        client_context: Optional[Dict[str, Any]] = None,
        tail: bool = False,
    ) -> Dict[str, Any]: ...
    
    def _get_lambda_client(self) -> boto3.client: ...


def make_patched_invocation(
    client: boto3.client,
    handler: Callable[[Optional[Dict[str, Any]], Any], Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]: ...