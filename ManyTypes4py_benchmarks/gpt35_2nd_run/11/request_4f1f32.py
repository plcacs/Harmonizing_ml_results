from __future__ import absolute_import
from .filepost import encode_multipart_formdata
from .packages import six
from .packages.six.moves.urllib.parse import urlencode
from typing import Dict, Any, Optional, Union

__all__ = ['RequestMethods']

class RequestMethods(object):
    _encode_url_methods: set = set(['DELETE', 'GET', 'HEAD', 'OPTIONS'])

    def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
        self.headers: Dict[str, str] = headers or {}

    def urlopen(self, method: str, url: str, body: Optional[Union[str, bytes]] = None, headers: Optional[Dict[str, str]] = None, encode_multipart: bool = True, multipart_boundary: Optional[str] = None, **kw: Any) -> None:
        raise NotImplementedError('Classes extending RequestMethods must implement their own ``urlopen`` method.')

    def request(self, method: str, url: str, fields: Optional[Dict[str, Union[str, Tuple[str, str, Optional[str]]]]] = None, headers: Optional[Dict[str, str]] = None, **urlopen_kw: Any) -> None:
        method = method.upper()
        if method in self._encode_url_methods:
            return self.request_encode_url(method, url, fields=fields, headers=headers, **urlopen_kw)
        else:
            return self.request_encode_body(method, url, fields=fields, headers=headers, **urlopen_kw)

    def request_encode_url(self, method: str, url: str, fields: Optional[Dict[str, str]] = None, headers: Optional[Dict[str, str]] = None, **urlopen_kw: Any) -> None:
        if headers is None:
            headers = self.headers
        extra_kw: Dict[str, Any] = {'headers': headers}
        extra_kw.update(urlopen_kw)
        if fields:
            url += '?' + urlencode(fields)
        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(self, method: str, url: str, fields: Optional[Dict[str, Union[str, Tuple[str, str, Optional[str]]]]] = None, headers: Optional[Dict[str, str]] = None, encode_multipart: bool = True, multipart_boundary: Optional[str] = None, **urlopen_kw: Any) -> None:
        if headers is None:
            headers = self.headers
        extra_kw: Dict[str, Any] = {'headers': {}}
        if fields:
            if 'body' in urlopen_kw:
                raise TypeError("request got values for both 'fields' and 'body', can only specify one.")
            if encode_multipart:
                body, content_type = encode_multipart_formdata(fields, boundary=multipart_boundary)
            else:
                body, content_type = (urlencode(fields), 'application/x-www-form-urlencoded')
            if isinstance(body, six.text_type):
                body = body.encode('utf-8')
            extra_kw['body'] = body
            extra_kw['headers'] = {'Content-Type': content_type}
        extra_kw['headers'].update(headers)
        extra_kw.update(urlopen_kw)
        return self.urlopen(method, url, **extra_kw)
