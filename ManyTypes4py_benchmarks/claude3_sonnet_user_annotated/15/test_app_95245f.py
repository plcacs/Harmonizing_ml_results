import sys
import base64
import logging
import json
import gzip
import inspect
import collections
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set, Type, TypeVar, cast, Iterator, Mapping

import pytest
from pytest import fixture
import hypothesis.strategies as st
from hypothesis import given, assume
import six

from chalice import app
from chalice import NotFoundError
from chalice.test import Client
from chalice.app import (
    APIGateway,
    Request,
    Response,
    handle_extra_types,
    MultiDict,
    WebsocketEvent,
    BadRequestError,
    WebsocketDisconnectedError,
    WebsocketEventSourceHandler,
    ConvertToMiddleware,
    WebsocketAPI,
    ChaliceUnhandledError,
)
from chalice import __version__ as chalice_version
from chalice.deploy.validate import ExperimentalFeatureError
from chalice.deploy.validate import validate_feature_flags


# These are used to generate sample data for hypothesis tests.
STR_MAP = st.dictionaries(st.text(), st.text())
STR_TO_LIST_MAP = st.dictionaries(
    st.text(),
    st.lists(elements=st.text(), min_size=1, max_size=5)
)
HTTP_METHOD = st.sampled_from(['GET', 'POST', 'PUT', 'PATCH',
                               'OPTIONS', 'HEAD', 'DELETE'])
PATHS = st.sampled_from(['/', '/foo/bar'])
HTTP_BODY = st.none() | st.text()
HTTP_REQUEST = st.fixed_dictionaries({
    'query_params': STR_TO_LIST_MAP,
    'headers': STR_MAP,
    'uri_params': STR_MAP,
    'method': HTTP_METHOD,
    'body': HTTP_BODY,
    'context': STR_MAP,
    'stage_vars': STR_MAP,
    'is_base64_encoded': st.booleans(),
    'path': PATHS,
})
HTTP_REQUEST = st.fixed_dictionaries({
    'multiValueQueryStringParameters': st.fixed_dictionaries({}),
    'headers': STR_MAP,
    'pathParameters': STR_MAP,
    'requestContext': st.fixed_dictionaries({
        'httpMethod': HTTP_METHOD,
        'resourcePath': PATHS,
    }),
    'body': HTTP_BODY,
    'stageVariables': STR_MAP,
    'isBase64Encoded': st.booleans(),
})
BINARY_TYPES = APIGateway().binary_types


class FakeLambdaContextIdentity:
    def __init__(self, cognito_identity_id: str, cognito_identity_pool_id: str) -> None:
        self.cognito_identity_id = cognito_identity_id
        self.cognito_identity_pool_id = cognito_identity_pool_id


class FakeLambdaContext:
    def __init__(self) -> None:
        self.function_name = 'test_name'
        self.function_version = 'version'
        self.invoked_function_arn = 'arn'
        self.memory_limit_in_mb = 256
        self.aws_request_id = 'id'
        self.log_group_name = 'log_group_name'
        self.log_stream_name = 'log_stream_name'
        self.identity = FakeLambdaContextIdentity('id', 'id_pool')
        # client_context is set by the mobile SDK and wont be set for chalice
        self.client_context = None

    def get_remaining_time_in_millis(self) -> int:
        return 500

    def serialize(self) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        serialized.update(vars(self))
        serialized['identity'] = vars(self.identity)
        return serialized


class FakeGoneException(Exception):
    pass


class FakeExceptionFactory:
    def __init__(self) -> None:
        self.GoneException = FakeGoneException


class FakeClient:
    def __init__(self, errors: Optional[List[Exception]] = None, infos: Optional[List[Dict[str, Any]]] = None) -> None:
        if errors is None:
            errors = []
        if infos is None:
            infos = []
        self._errors = errors
        self._infos = infos
        self.calls: Dict[str, List[Tuple[Any, ...]]] = collections.defaultdict(lambda: [])
        self.exceptions = FakeExceptionFactory()

    def post_to_connection(self, ConnectionId: str, Data: str) -> None:
        self._call('post_to_connection', ConnectionId, Data)

    def delete_connection(self, ConnectionId: str) -> None:
        self._call('close', ConnectionId)

    def get_connection(self, ConnectionId: str) -> Optional[Dict[str, Any]]:
        self._call('info', ConnectionId)
        if self._infos is not None:
            info = self._infos.pop()
            return info
        return None

    def _call(self, name: str, *args: Any) -> None:
        self.calls[name].append(tuple(args))
        if self._errors:
            error = self._errors.pop()
            raise error


class FakeSession:
    def __init__(self, client: Optional[FakeClient] = None, region_name: str = 'us-west-2') -> None:
        self.calls: List[Tuple[str, Optional[str]]] = []
        self._client = client
        self.region_name = region_name

    def client(self, name: str, endpoint_url: Optional[str] = None) -> FakeClient:
        self.calls.append((name, endpoint_url))
        return self._client
