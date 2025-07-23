import datetime
import json
import os
import uuid
from http.client import HTTPConnection
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

import requests
from requests import Response, Session
from requests.auth import AuthBase, HTTPBasicAuth
from requests_hawk import HawkAuth
from alerta.utils.collections import merge


class Client:
    DEFAULT_ENDPOINT: str = 'http://localhost:8080'

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 5.0,
        ssl_verify: Union[bool, str] = True,
        headers: Optional[Dict[str, str]] = None,
        debug: bool = False
    ) -> None:
        self.endpoint: str = endpoint or os.environ.get('ALERTA_ENDPOINT', self.DEFAULT_ENDPOINT)
        if debug:
            HTTPConnection.debuglevel = 1
        api_key: str = key or os.environ.get('ALERTA_API_KEY', '')
        self.http: 'HTTPClient' = HTTPClient(
            self.endpoint,
            key=key,
            secret=secret,
            token=token,
            username=username,
            password=password,
            timeout=timeout,
            ssl_verify=ssl_verify,
            headers=headers,
            debug=debug
        )

    def send_alert(self, resource: str, event: str, **kwargs: Any) -> Response:
        data: Dict[str, Any] = {
            'id': kwargs.get('id'),
            'resource': resource,
            'event': event,
            'environment': kwargs.get('environment'),
            'severity': kwargs.get('severity'),
            'correlate': kwargs.get('correlate', []) or [],
            'service': kwargs.get('service', []) or [],
            'group': kwargs.get('group'),
            'value': kwargs.get('value'),
            'text': kwargs.get('text'),
            'tags': kwargs.get('tags', []) or [],
            'attributes': kwargs.get('attributes', {}) or {},
            'origin': kwargs.get('origin'),
            'type': kwargs.get('type'),
            'createTime': datetime.datetime.utcnow(),
            'timeout': kwargs.get('timeout'),
            'rawData': kwargs.get('raw_data'),
            'customer': kwargs.get('customer')
        }
        return self.http.post('/alert', data)

    def action(self, id: str, action: str, text: str = '', timeout: Optional[float] = None) -> Response:
        data: Dict[str, Any] = {'action': action, 'text': text, 'timeout': timeout}
        return self.http.put(f'/alert/{id}/action', data)

    def delete_alert(self, id: str) -> Response:
        return self.http.delete(f'/alert/{id}')


class ApiKeyAuth(AuthBase):

    def __init__(self, api_key: Optional[str] = None, auth_token: Optional[str] = None) -> None:
        self.api_key: Optional[str] = api_key
        self.auth_token: Optional[str] = auth_token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        if self.api_key:
            r.headers['Authorization'] = f'Key {self.api_key}'
        return r


class TokenAuth(AuthBase):

    def __init__(self, auth_token: Optional[str] = None) -> None:
        self.auth_token: Optional[str] = auth_token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        if self.auth_token:
            r.headers['Authorization'] = f'Bearer {self.auth_token}'
        return r


class HTTPClient:

    def __init__(
        self,
        endpoint: str,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
        ssl_verify: Union[bool, str] = True,
        headers: Optional[Dict[str, str]] = None,
        debug: bool = False
    ) -> None:
        self.endpoint: str = endpoint
        self.auth: Optional[AuthBase] = None
        if username:
            self.auth = HTTPBasicAuth(username, password or '')
        elif secret:
            self.auth = HawkAuth(id=key or '', key=secret)
        elif key:
            self.auth = ApiKeyAuth(api_key=key)
        elif token:
            self.auth = TokenAuth(auth_token=token)
        self.timeout: float = timeout
        self.session: Session = requests.Session()
        self.session.verify = ssl_verify
        self.headers: Dict[str, str] = headers.copy() if headers else {}
        merge(self.headers, self.default_headers())
        self.debug: bool = debug

    @staticmethod
    def default_headers() -> Dict[str, str]:
        return {
            'X-Request-ID': str(uuid.uuid4()),
            'Content-Type': 'application/json'
        }

    def get(self, path: str, query: Optional[List[Tuple[str, Any]]] = None, **kwargs: Any) -> Response:
        query = query or []
        if 'page' in kwargs:
            query.append(('page', kwargs['page']))
        if 'page_size' in kwargs:
            query.append(('page-size', kwargs['page_size']))
        query_string: str = urlencode(query, doseq=True)
        url: str = f"{self.endpoint}{path}?{query_string}"
        try:
            response: Response = self.session.get(
                url,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException:
            raise
        return response

    def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Response:
        url: str = f"{self.endpoint}{path}"
        try:
            response: Response = self.session.post(
                url,
                data=json.dumps(data, cls=CustomJsonEncoder) if data else None,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException:
            raise
        return response

    def put(self, path: str, data: Optional[Dict[str, Any]] = None) -> Response:
        url: str = f"{self.endpoint}{path}"
        try:
            response: Response = self.session.put(
                url,
                data=json.dumps(data, cls=CustomJsonEncoder) if data else None,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException:
            raise
        return response

    def delete(self, path: str) -> Response:
        url: str = f"{self.endpoint}{path}"
        try:
            response: Response = self.session.delete(
                url,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException:
            raise
        return response


class CustomJsonEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%S') + f'.{int(o.microsecond // 1000):03}Z'
        elif isinstance(o, datetime.timedelta):
            return int(o.total_seconds())
        else:
            return super().default(o)
