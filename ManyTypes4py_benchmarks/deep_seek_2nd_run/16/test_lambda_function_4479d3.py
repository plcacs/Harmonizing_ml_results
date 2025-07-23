import inspect
import io
import json
import zipfile
from typing import Optional, Any, Dict, Generator, Union, cast
import boto3
import pytest
from botocore.response import StreamingBody
from moto import mock_iam, mock_lambda
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction
from pydantic_core import from_json, to_json
from prefect import flow
from unittest.mock import patch

@pytest.fixture
def lambda_mock(aws_credentials: AwsCredentials) -> Generator[boto3.client, None, None]:
    with mock_lambda():
        yield boto3.client('lambda', region_name=aws_credentials.region_name)

@pytest.fixture
def iam_mock(aws_credentials: AwsCredentials) -> Generator[boto3.client, None, None]:
    with mock_iam():
        yield boto3.client('iam', region_name=aws_credentials.region_name)

@pytest.fixture
def mock_iam_rule(iam_mock: boto3.client) -> Generator[Dict[str, Any], None, None]:
    yield iam_mock.create_role(RoleName='test-role', AssumeRolePolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}))

def handler_a(event: Optional[Union[Dict[str, Any], Any]], context: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        if 'error' in event:
            raise Exception(event['error'])
        event['foo'] = 'bar'
    else:
        event = {'foo': 'bar'}
    return event

LAMBDA_TEST_CODE: str = inspect.getsource(handler_a)

@pytest.fixture
def mock_lambda_code() -> Generator[bytes, None, None]:
    with io.BytesIO() as f:
        with zipfile.ZipFile(f, mode='w') as z:
            z.writestr('foo.py', LAMBDA_TEST_CODE)
        f.seek(0)
        yield f.read()

@pytest.fixture
def mock_lambda_function(lambda_mock: boto3.client, mock_iam_rule: Dict[str, Any], mock_lambda_code: bytes) -> Generator[Dict[str, Any], None, None]:
    r = lambda_mock.create_function(FunctionName='test-function', Runtime='python3.10', Role=mock_iam_rule['Role']['Arn'], Handler='foo.handler', Code={'ZipFile': mock_lambda_code})
    r2 = lambda_mock.publish_version(FunctionName='test-function')
    r['Version'] = r2['Version']
    yield r

def handler_b(event: Any, context: Any) -> Dict[str, Any]:
    event = {'data': [1, 2, 3]}
    return event

LAMBDA_TEST_CODE_V2: str = inspect.getsource(handler_b)

@pytest.fixture
def mock_lambda_code_v2() -> Generator[bytes, None, None]:
    with io.BytesIO() as f:
        with zipfile.ZipFile(f, mode='w') as z:
            z.writestr('foo.py', LAMBDA_TEST_CODE_V2)
        f.seek(0)
        yield f.read()

@pytest.fixture
def add_lambda_version(mock_lambda_function: Dict[str, Any], lambda_mock: boto3.client, mock_lambda_code_v2: bytes) -> Generator[Dict[str, Any], None, None]:
    r = mock_lambda_function.copy()
    lambda_mock.update_function_code(FunctionName='test-function', ZipFile=mock_lambda_code_v2)
    r2 = lambda_mock.publish_version(FunctionName='test-function')
    r['Version'] = r2['Version']
    yield r

@pytest.fixture
def lambda_function(aws_credentials: AwsCredentials) -> LambdaFunction:
    return LambdaFunction(function_name='test-function', aws_credentials=aws_credentials)

def make_patched_invocation(client: boto3.client, handler: Any) -> Any:
    """Creates a patched invoke method for moto lambda. The method replaces
    the response 'Payload' with the result of the handler function.
    """
    true_invoke = client.invoke

    def invoke(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Calls the true invoke and replaces the Payload with its result."""
        result = true_invoke(*args, **kwargs)
        blob = to_json(handler(event=kwargs.get('Payload'), context=kwargs.get('ClientContext')))
        result['Payload'] = StreamingBody(io.BytesIO(blob), len(blob))
        return result
    return invoke

@pytest.fixture
def mock_invoke_base(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Fixture for base version of lambda function"""
    client = lambda_function._get_lambda_client()
    monkeypatch.setattr(client, 'invoke', make_patched_invocation(client, handler_a))

    def _get_lambda_client() -> boto3.client:
        return client
    monkeypatch.setattr(lambda_function, '_get_lambda_client', _get_lambda_client)
    yield

@pytest.fixture
def mock_invoke_updated(lambda_function: LambdaFunction, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Fixture for updated version of lambda function"""
    client = lambda_function._get_lambda_client()
    monkeypatch.setattr(client, 'invoke', make_patched_invocation(client, handler_b))

    def _get_lambda_client() -> boto3.client:
        return client
    monkeypatch.setattr(lambda_function, '_get_lambda_client', _get_lambda_client)
    yield

class TestLambdaFunction:

    def test_init(self, aws_credentials: AwsCredentials) -> None:
        function = LambdaFunction(function_name='test-function', aws_credentials=aws_credentials)
        assert function.function_name == 'test-function'
        assert function.qualifier is None

    @pytest.mark.parametrize('payload,expected', [({'foo': 'baz'}, {'foo': 'bar'}), (None, {'foo': 'bar'})])
    def test_invoke_lambda_payloads(self, payload: Optional[Dict[str, str]], expected: Dict[str, str], mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        result = lambda_function.invoke(payload)
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == expected

    async def test_invoke_lambda_async_dispatch(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:

        @flow
        async def test_flow() -> Dict[str, Any]:
            result = await lambda_function.invoke()
            return result
        result = await test_flow()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    async def test_invoke_lambda_force_sync(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        result = lambda_function.invoke(_sync=True)
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    def test_invoke_lambda_tail(self, lambda_function: LambdaFunction, mock_lambda_function: Dict[str, Any], mock_invoke_base: None) -> None:
        result = lambda_function.invoke(tail=True)
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}
        assert 'LogResult' in result

    def test_invoke_lambda_client_context(self, lambda_function: LambdaFunction, mock_lambda_function: Dict[str, Any], mock_invoke_base: None) -> None:
        result = lambda_function.invoke(client_context={'bar': 'foo'})
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    def test_invoke_lambda_qualifier_base_version(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        """Test invoking the base version of the lambda function"""
        lambda_function.qualifier = mock_lambda_function['Version']
        result = lambda_function.invoke()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    def test_invoke_lambda_qualifier_updated_version(self, add_lambda_version: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_updated: None) -> None:
        """Test invoking the updated version of the lambda function"""
        lambda_function.qualifier = add_lambda_version['Version']
        result = lambda_function.invoke()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'data': [1, 2, 3]}

class TestLambdaFunctionAsync:

    async def test_ainvoke_lambda_explicit(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:

        @flow
        async def test_flow() -> Dict[str, Any]:
            result = await lambda_function.ainvoke()
            return result
        result = await test_flow()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    async def test_ainvoke_lambda_payloads(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        result = await lambda_function.ainvoke(payload={'foo': 'baz'})
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    async def test_ainvoke_lambda_tail(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        result = await lambda_function.ainvoke(tail=True)
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}
        assert 'LogResult' in result

    async def test_ainvoke_lambda_client_context(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        result = await lambda_function.ainvoke(client_context={'bar': 'foo'})
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    async def test_ainvoke_lambda_qualifier_base_version(self, mock_lambda_function: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_base: None) -> None:
        """Test invoking the base version of the lambda function asynchronously"""
        lambda_function.qualifier = mock_lambda_function['Version']
        result = await lambda_function.ainvoke()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'foo': 'bar'}

    async def test_ainvoke_lambda_qualifier_updated_version(self, add_lambda_version: Dict[str, Any], lambda_function: LambdaFunction, mock_invoke_updated: None) -> None:
        """Test invoking the updated version of the lambda function asynchronously"""
        lambda_function.qualifier = add_lambda_version['Version']
        result = await lambda_function.ainvoke()
        assert result['StatusCode'] == 200
        response_payload = from_json(result['Payload'].read())
        assert response_payload == {'data': [1, 2, 3]}
