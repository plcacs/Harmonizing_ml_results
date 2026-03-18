```python
import io
from typing import Any, Optional, Union, Dict, List, BinaryIO, Iterator, Generator
from typing_extensions import Literal
from botocore.response import StreamingBody
from prefect_aws.credentials import AwsCredentials
from prefect import flow

def handler_a(event: Any, context: Any) -> Dict[str, Any]: ...
def handler_b(event: Any, context: Any) -> Dict[str, Any]: ...

LAMBDA_TEST_CODE: str = ...
LAMBDA_TEST_CODE_V2: str = ...

class LambdaFunction:
    function_name: str
    qualifier: Optional[str]
    aws_credentials: AwsCredentials
    
    def __init__(
        self,
        function_name: str,
        aws_credentials: AwsCredentials,
        qualifier: Optional[str] = None
    ) -> None: ...
    
    def invoke(
        self,
        payload: Optional[Any] = None,
        client_context: Optional[Dict[str, Any]] = None,
        tail: bool = False,
        qualifier: Optional[str] = None,
        _sync: bool = False
    ) -> Dict[str, Any]: ...
    
    async def ainvoke(
        self,
        payload: Optional[Any] = None,
        client_context: Optional[Dict[str, Any]] = None,
        tail: bool = False,
        qualifier: Optional[str] = None
    ) -> Dict[str, Any]: ...
    
    def _get_lambda_client(self) -> Any: ...

@pytest.fixture
def lambda_mock(aws_credentials: Any) -> Generator[Any, None, None]: ...

@pytest.fixture
def iam_mock(aws_credentials: Any) -> Generator[Any, None, None]: ...

@pytest.fixture
def mock_iam_rule(iam_mock: Any) -> Generator[Dict[str, Any], None, None]: ...

@pytest.fixture
def mock_lambda_code() -> Generator[bytes, None, None]: ...

@pytest.fixture
def mock_lambda_function(
    lambda_mock: Any,
    mock_iam_rule: Any,
    mock_lambda_code: Any
) -> Generator[Dict[str, Any], None, None]: ...

@pytest.fixture
def mock_lambda_code_v2() -> Generator[bytes, None, None]: ...

@pytest.fixture
def add_lambda_version(
    mock_lambda_function: Any,
    lambda_mock: Any,
    mock_lambda_code_v2: Any
) -> Generator[Dict[str, Any], None, None]: ...

@pytest.fixture
def lambda_function(aws_credentials: Any) -> LambdaFunction: ...

def make_patched_invocation(
    client: Any,
    handler: Any
) -> Any: ...

@pytest.fixture
def mock_invoke_base(
    lambda_function: LambdaFunction,
    monkeypatch: Any
) -> Generator[None, None, None]: ...

@pytest.fixture
def mock_invoke_updated(
    lambda_function: LambdaFunction,
    monkeypatch: Any
) -> Generator[None, None, None]: ...

class TestLambdaFunction:
    def test_init(self, aws_credentials: Any) -> None: ...
    
    @pytest.mark.parametrize('payload,expected', [({'foo': 'baz'}, {'foo': 'bar'}), (None, {'foo': 'bar'})])
    def test_invoke_lambda_payloads(
        self,
        payload: Optional[Dict[str, Any]],
        expected: Dict[str, Any],
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_invoke_lambda_async_dispatch(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_invoke_lambda_force_sync(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    def test_invoke_lambda_tail(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: Any,
        mock_invoke_base: Any
    ) -> None: ...
    
    def test_invoke_lambda_client_context(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: Any,
        mock_invoke_base: Any
    ) -> None: ...
    
    def test_invoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    def test_invoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: Any,
        lambda_function: LambdaFunction,
        mock_invoke_updated: Any
    ) -> None: ...

class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_ainvoke_lambda_payloads(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_ainvoke_lambda_tail(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_ainvoke_lambda_client_context(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_ainvoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: Any,
        lambda_function: LambdaFunction,
        mock_invoke_base: Any
    ) -> None: ...
    
    async def test_ainvoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: Any,
        lambda_function: LambdaFunction,
        mock_invoke_updated: Any
    ) -> None: ...
```