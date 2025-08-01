from __future__ import absolute_import
from .filepost import encode_multipart_formdata
from .packages import six
from .packages.six.moves.urllib.parse import urlencode
from typing import Any, Dict, Optional
__all__ = ['RequestMethods']

class RequestMethods(object):
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`~urllib3.connectionpool.HTTPConnectionPool` and
    :class:`~urllib3.poolmanager.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.

    Specifically,

    :meth:`.request_encode_url` is for sending requests whose fields are
    encoded in the URL (such as GET, HEAD, DELETE).

    :meth:`.request_encode_body` is for sending requests whose fields are
    encoded in the *body* of the request using multipart or www-form-urlencoded
    (such as for POST, PUT, PATCH).

    :meth:`.request` is for making any kind of request, it will look up the
    appropriate encoding format and use one of the above two methods to make
    the request.

    Initializer parameters:

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    """
    _encode_url_methods = set(['DELETE', 'GET', 'HEAD', 'OPTIONS'])

    def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
        self.headers: Dict[str, str] = headers or {}

    def urlopen(self, method: str, url: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None, encode_multipart: bool = True, multipart_boundary: Optional[str] = None, **kw: Any) -> Any:
        raise NotImplementedError('Classes extending RequestMethods must implement their own ``urlopen`` method.')

    def request(self, method: str, url: str, fields: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **urlopen_kw: Any) -> Any:
        """
        Make a request using :meth:`urlopen` with the appropriate encoding of
        ``fields`` based on the ``method`` used.

        This is a convenience method that requires the least amount of manual
        effort. It can be used in most situations, while still having the
        option to drop down to more specific methods when necessary, such as
        :meth:`request_encode_url`, :meth:`request_encode_body`,
        or even the lowest level :meth:`urlopen`.
        """
        method = method.upper()
        if method in self._encode_url_methods:
            return self.request_encode_url(method, url, fields=fields, headers=headers, **urlopen_kw)
        else:
            return self.request_encode_body(method, url, fields=fields, headers=headers, **urlopen_kw)

    def request_encode_url(self, method: str, url: str, fields: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **urlopen_kw: Any) -> Any:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.
        """
        if headers is None:
            headers = self.headers
        extra_kw: Dict[str, Any] = {'headers': headers}
        extra_kw.update(urlopen_kw)
        if fields:
            url += '?' + urlencode(fields)
        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(self, method: str, url: str, fields: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, encode_multipart: bool = True, multipart_boundary: Optional[str] = None, **urlopen_kw: Any) -> Any:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.

        When ``encode_multipart=True`` (default), then
        :meth:`urllib3.filepost.encode_multipart_formdata` is used to encode
        the payload with the appropriate content type. Otherwise
        :meth:`urllib.urlencode` is used with the
        'application/x-www-form-urlencoded' content type.

        Multipart encoding must be used when posting files, and it's reasonably
        safe to use it in other times too. However, it may break request
        signing, such as with OAuth.

        Supports an optional ``fields`` parameter of key/value strings AND
        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where
        the MIME type is optional. For example::

            fields = {
                'foo': 'bar',
                'fakefile': ('foofile.txt', 'contents of foofile'),
                'realfile': ('barfile.txt', open('realfile').read()),
                'typedfile': ('bazfile.bin', open('bazfile').read(),
                              'image/jpeg'),
                'nonamefile': 'contents of nonamefile field',
            }

        When uploading a file, providing a filename (the first parameter of the
        tuple) is optional but recommended to best mimick behavior of browsers.

        Note that if ``headers`` are supplied, the 'Content-Type' header will
        be overwritten because it depends on the dynamic random boundary string
        which is used to compose the body of the request. The random boundary
        string can be explicitly set with the ``multipart_boundary`` parameter.
        """
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