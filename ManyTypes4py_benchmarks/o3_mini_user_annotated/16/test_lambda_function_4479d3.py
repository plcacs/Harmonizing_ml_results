import inspect
import io
import json
import zipfile
from typing import Any, Callable, Iterator, Optional

import boto3
import pytest
from botocore.client import BaseClient
from botocore.response import StreamingBody
from moto import mock_iam, mock_lambda
from prefect_aws.credentials import AwsCredentials
from prefect_aws.lambda_function import LambdaFunction
from pydantic_core import from_json, to_json

from prefect import flow


@pytest.fixture
def lambda_mock(aws_credentials: AwsCredentials) -> Iterator[BaseClient]:
    with mock_lambda():
        yield boto3.client(
            "lambda",
            region_name=aws_credentials.region_name,
        )


@pytest.fixture
def iam_mock(aws_credentials: AwsCredentials) -> Iterator[BaseClient]:
    with mock_iam():
        yield boto3.client(
            "iam",
            region_name=aws_credentials.region_name,
        )


@pytest.fixture
def mock_iam_rule(iam_mock: BaseClient) -> Iterator[dict]:
    response = iam_mock.create_role(
        RoleName="test-role",
        AssumeRolePolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
    )
    yield response


def handler_a(event: Any, context: Any) -> dict:
    if isinstance(event, dict):
        if "error" in event:
            raise Exception(event["error"])
        event["foo"] = "bar"
    else:
        event = {"foo": "bar"}
    return event


LAMBDA_TEST_CODE: str = inspect.getsource(handler_a)


@pytest.fixture
def mock_lambda_code() -> Iterator[bytes]:
    with io.BytesIO() as f:
        with zipfile.ZipFile(f, mode="w") as z:
            z.writestr("foo.py", LAMBDA_TEST_CODE)
        f.seek(0)
        yield f.read()


@pytest.fixture
def mock_lambda_function(
    lambda_mock: BaseClient, mock_iam_rule: dict, mock_lambda_code: bytes
) -> Iterator[dict]:
    r = lambda_mock.create_function(
        FunctionName="test-function",
        Runtime="python3.10",
        Role=mock_iam_rule["Role"]["Arn"],
        Handler="foo.handler",
        Code={"ZipFile": mock_lambda_code},
    )
    r2 = lambda_mock.publish_version(
        FunctionName="test-function",
    )
    r["Version"] = r2["Version"]
    yield r


def handler_b(event: Any, context: Any) -> dict:
    event = {"data": [1, 2, 3]}
    return event


LAMBDA_TEST_CODE_V2: str = inspect.getsource(handler_b)


@pytest.fixture
def mock_lambda_code_v2() -> Iterator[bytes]:
    with io.BytesIO() as f:
        with zipfile.ZipFile(f, mode="w") as z:
            z.writestr("foo.py", LAMBDA_TEST_CODE_V2)
        f.seek(0)
        yield f.read()


@pytest.fixture
def add_lambda_version(
    mock_lambda_function: dict,
    lambda_mock: BaseClient,
    mock_lambda_code_v2: bytes,
) -> Iterator[dict]:
    r = mock_lambda_function.copy()
    lambda_mock.update_function_code(
        FunctionName="test-function",
        ZipFile=mock_lambda_code_v2,
    )
    r2 = lambda_mock.publish_version(
        FunctionName="test-function",
    )
    r["Version"] = r2["Version"]
    yield r


@pytest.fixture
def lambda_function(aws_credentials: AwsCredentials) -> LambdaFunction:
    return LambdaFunction(
        function_name="test-function",
        aws_credentials=aws_credentials,
    )


def make_patched_invocation(
    client: BaseClient, handler: Callable[[Any, Any], Any]
) -> Callable[..., dict]:
    """Creates a patched invoke method for moto lambda. The method replaces
    the response 'Payload' with the result of the handler function.
    """
    true_invoke = client.invoke

    def invoke(*args: Any, **kwargs: Any) -> dict:
        """Calls the true invoke and replaces the Payload with its result."""
        result: dict = true_invoke(*args, **kwargs)
        blob: bytes = to_json(
            handler(
                event=kwargs.get("Payload"),
                context=kwargs.get("ClientContext"),
            )
        ).encode("utf-8")
        result["Payload"] = StreamingBody(io.BytesIO(blob), len(blob))
        return result

    return invoke


@pytest.fixture
def mock_invoke_base(
    lambda_function: LambdaFunction,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fixture for base version of lambda function"""
    client: BaseClient = lambda_function._get_lambda_client()

    monkeypatch.setattr(
        client,
        "invoke",
        make_patched_invocation(client, handler_a),
    )

    def _get_lambda_client() -> BaseClient:
        return client

    monkeypatch.setattr(
        lambda_function,
        "_get_lambda_client",
        _get_lambda_client,
    )

    yield


@pytest.fixture
def mock_invoke_updated(
    lambda_function: LambdaFunction,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fixture for updated version of lambda function"""
    client: BaseClient = lambda_function._get_lambda_client()

    monkeypatch.setattr(
        client,
        "invoke",
        make_patched_invocation(client, handler_b),
    )

    def _get_lambda_client() -> BaseClient:
        return client

    monkeypatch.setattr(
        lambda_function,
        "_get_lambda_client",
        _get_lambda_client,
    )

    yield


class TestLambdaFunction:
    def test_init(self, aws_credentials: AwsCredentials) -> None:
        function = LambdaFunction(
            function_name="test-function",
            aws_credentials=aws_credentials,
        )
        assert function.function_name == "test-function"
        assert function.qualifier is None

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"foo": "baz"}, {"foo": "bar"}),
            (None, {"foo": "bar"}),
        ],
    )
    def test_invoke_lambda_payloads(
        self,
        payload: Optional[dict],
        expected: dict,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        result: dict = lambda_function.invoke(payload)
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == expected

    async def test_invoke_lambda_async_dispatch(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        @flow
        async def test_flow() -> dict:
            result_inner: dict = await lambda_function.invoke()
            return result_inner

        result: dict = await test_flow()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    async def test_invoke_lambda_force_sync(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        result: dict = lambda_function.invoke(_sync=True)
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    def test_invoke_lambda_tail(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: dict,
        mock_invoke_base: None,
    ) -> None:
        result: dict = lambda_function.invoke(tail=True)
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}
        assert "LogResult" in result

    def test_invoke_lambda_client_context(
        self,
        lambda_function: LambdaFunction,
        mock_lambda_function: dict,
        mock_invoke_base: None,
    ) -> None:
        result: dict = lambda_function.invoke(client_context={"bar": "foo"})
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    def test_invoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        """Test invoking the base version of the lambda function"""
        lambda_function.qualifier = mock_lambda_function["Version"]
        result: dict = lambda_function.invoke()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    def test_invoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: dict,
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None:
        """Test invoking the updated version of the lambda function"""
        lambda_function.qualifier = add_lambda_version["Version"]
        result: dict = lambda_function.invoke()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"data": [1, 2, 3]}


class TestLambdaFunctionAsync:
    async def test_ainvoke_lambda_explicit(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        @flow
        async def test_flow() -> dict:
            result_inner: dict = await lambda_function.ainvoke()
            return result_inner

        result: dict = await test_flow()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    async def test_ainvoke_lambda_payloads(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        result: dict = await lambda_function.ainvoke(payload={"foo": "baz"})
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    async def test_ainvoke_lambda_tail(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        result: dict = await lambda_function.ainvoke(tail=True)
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}
        assert "LogResult" in result

    async def test_ainvoke_lambda_client_context(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        result: dict = await lambda_function.ainvoke(client_context={"bar": "foo"})
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    async def test_ainvoke_lambda_qualifier_base_version(
        self,
        mock_lambda_function: dict,
        lambda_function: LambdaFunction,
        mock_invoke_base: None,
    ) -> None:
        """Test invoking the base version of the lambda function asynchronously"""
        lambda_function.qualifier = mock_lambda_function["Version"]
        result: dict = await lambda_function.ainvoke()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"foo": "bar"}

    async def test_ainvoke_lambda_qualifier_updated_version(
        self,
        add_lambda_version: dict,
        lambda_function: LambdaFunction,
        mock_invoke_updated: None,
    ) -> None:
        """Test invoking the updated version of the lambda function asynchronously"""
        lambda_function.qualifier = add_lambda_version["Version"]
        result: dict = await lambda_function.ainvoke()
        assert result["StatusCode"] == 200
        response_payload: Any = from_json(result["Payload"].read())
        assert response_payload == {"data": [1, 2, 3]}