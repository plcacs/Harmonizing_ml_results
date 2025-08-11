import datetime
import json
import os
import uuid
from http.client import HTTPConnection
from urllib.parse import urlencode
import requests
from requests.auth import AuthBase, HTTPBasicAuth
from requests_hawk import HawkAuth
from alerta.utils.collections import merge

class Client:
    DEFAULT_ENDPOINT = 'http://localhost:8080'

    def __init__(self, endpoint=None, key=None, secret=None, token=None, username=None, password=None, timeout=5.0, ssl_verify=True, headers=None, debug=False) -> None:
        self.endpoint = endpoint or os.environ.get('ALERTA_ENDPOINT', self.DEFAULT_ENDPOINT)
        if debug:
            HTTPConnection.debuglevel = 1
        key = key or os.environ.get('ALERTA_API_KEY', '')
        self.http = HTTPClient(self.endpoint, key, secret, token, username, password, timeout, ssl_verify, headers, debug)

    def send_alert(self, resource: Union[dict, dict[str, typing.Any]], event: Union[dict, dict[str, typing.Any]], **kwargs) -> Union[str, bool, dict[str, typing.Any]]:
        data = {'id': kwargs.get('id'), 'resource': resource, 'event': event, 'environment': kwargs.get('environment'), 'severity': kwargs.get('severity'), 'correlate': kwargs.get('correlate', None) or list(), 'service': kwargs.get('service', None) or list(), 'group': kwargs.get('group'), 'value': kwargs.get('value'), 'text': kwargs.get('text'), 'tags': kwargs.get('tags', None) or list(), 'attributes': kwargs.get('attributes', None) or dict(), 'origin': kwargs.get('origin'), 'type': kwargs.get('type'), 'createTime': datetime.datetime.utcnow(), 'timeout': kwargs.get('timeout'), 'rawData': kwargs.get('raw_data'), 'customer': kwargs.get('customer')}
        return self.http.post('/alert', data)

    def action(self, id: str, action: Union[str, int, float], text: typing.Text='', timeout: Union[None, str, int, float]=None):
        data = {'action': action, 'text': text, 'timeout': timeout}
        return self.http.put(f'/alert/{id}/action', data)

    def delete_alert(self, id: Union[str, int]) -> Union[bool, str]:
        return self.http.delete(f'/alert/{id}')

class ApiKeyAuth(AuthBase):

    def __init__(self, api_key=None, auth_token: Union[None, str]=None) -> None:
        self.api_key = api_key
        self.auth_token = auth_token

    def __call__(self, r: Union[requests.PreparedRequest, dict[str, typing.Any], dict]) -> Union[requests.PreparedRequest, dict[str, typing.Any], dict]:
        r.headers['Authorization'] = f'Key {self.api_key}'
        return r

class TokenAuth(AuthBase):

    def __init__(self, auth_token: Union[None, str]=None) -> None:
        self.auth_token = auth_token

    def __call__(self, r: Union[requests.PreparedRequest, dict[str, typing.Any], dict]) -> Union[requests.PreparedRequest, dict[str, typing.Any], dict]:
        r.headers['Authorization'] = f'Bearer {self.auth_token}'
        return r

class HTTPClient:

    def __init__(self, endpoint, key=None, secret=None, token=None, username=None, password=None, timeout=30.0, ssl_verify=True, headers=None, debug=False) -> None:
        self.endpoint = endpoint
        self.auth = None
        if username:
            self.auth = HTTPBasicAuth(username, password)
        elif secret:
            self.auth = HawkAuth(id=key, key=secret)
        elif key:
            self.auth = ApiKeyAuth(api_key=key)
        elif token:
            self.auth = TokenAuth(token)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = ssl_verify
        self.headers = headers or dict()
        merge(self.headers, self.default_headers())
        self.debug = debug

    @staticmethod
    def default_headers() -> dict[typing.Text, str]:
        return {'X-Request-ID': str(uuid.uuid4()), 'Content-Type': 'application/json'}

    def get(self, path: str, query: Union[None, str]=None, **kwargs) -> Union[dict[str, str], list]:
        query = query or []
        if 'page' in kwargs:
            query.append(('page', kwargs['page']))
        if 'page_size' in kwargs:
            query.append(('page-size', kwargs['page_size']))
        url = self.endpoint + path + '?' + urlencode(query, doseq=True)
        try:
            response = self.session.get(url, headers=self.headers, auth=self.auth, timeout=self.timeout)
        except requests.exceptions.RequestException:
            raise
        return response

    def post(self, path: str, data: Union[None, dict, str, typing.Mapping]=None) -> Union[requests.models.Response, typing.Generator[typing.Optional[typing.Any]]]:
        url = self.endpoint + path
        try:
            response = self.session.post(url, data=json.dumps(data, cls=CustomJsonEncoder), headers=self.headers, auth=self.auth, timeout=self.timeout)
        except requests.exceptions.RequestException:
            raise
        return response

    def put(self, path: Union[str, typing.Any, None, bytes], data: Union[None, dict, typing.Mapping]=None) -> Union[requests.models.Response, typing.Generator[typing.Optional[typing.Any]], PyLibs.objection.utils.frida_transporRunnerMessage]:
        url = self.endpoint + path
        try:
            response = self.session.put(url, data=json.dumps(data, cls=CustomJsonEncoder), headers=self.headers, auth=self.auth, timeout=self.timeout)
        except requests.exceptions.RequestException:
            raise
        return response

    def delete(self, path: str) -> Union[requests.models.Response, list, dict[str, str]]:
        url = self.endpoint + path
        try:
            response = self.session.delete(url, headers=self.headers, auth=self.auth, timeout=self.timeout)
        except requests.exceptions.RequestException:
            raise
        return response

class CustomJsonEncoder(json.JSONEncoder):

    def default(self, o: Union[float, None, datetime.datetime, datetime.time]) -> Union[str, int]:
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%S') + f'.{int(o.microsecond // 1000):03}Z'
        elif isinstance(o, datetime.timedelta):
            return int(o.total_seconds())
        else:
            return json.JSONEncoder.default(self, o)