from __future__ import absolute_import
from typing import Optional, Dict, Any, Union, Tuple, List, Set, cast
from .filepost import encode_multipart_formdata
from .packages import six
from .packages.six.moves.urllib.parse import urlencode

__all__ = ["RequestMethods"]

class RequestMethods(object):
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`~urllib3.connectionpool.HTTPConnectionPool` and
    :class:`~urllib3.poolmanager.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.
    """
    _encode_url_methods: Set[str] = {"DELETE", "GET", "HEAD", "OPTIONS"}

    def __init__(self, headers: Optional[Dict[str, str]] = None) -> None:
        self.headers = headers or {}

    def urlopen(
        self,
        method: str,
        url: str,
        body: Optional[Union[bytes, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        encode_multipart: bool = True,
        multipart_boundary: Optional[str] = None,
        **kw: Any
    ) -> Any:  # Abstract
        raise NotImplementedError(
            "Classes extending RequestMethods must implement "
            "their own ``urlopen`` method."
        )

    def request(
        self,
        method: str,
        url: str,
        fields: Optional[Union[Dict[str, str], Dict[str, Union[str, Tuple[str, ...]]]] = None,
        headers: Optional[Dict[str, str]] = None,
        **urlopen_kw: Any
    ) -> Any:
        """
        Make a request using :meth:`urlopen` with the appropriate encoding of
        ``fields`` based on the ``method`` used.
        """
        method = method.upper()
        if method in self._encode_url_methods:
            return self.request_encode_url(
                method, url, fields=fields, headers=headers, **urlopen_kw
            )
        else:
            return self.request_encode_body(
                method, url, fields=fields, headers=headers, **urlopen_kw
            )

    def request_encode_url(
        self,
        method: str,
        url: str,
        fields: Optional[Union[Dict[str, str], Dict[str, Union[str, Tuple[str, ...]]]] = None,
        headers: Optional[Dict[str, str]] = None,
        **urlopen_kw: Any
    ) -> Any:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.
        """
        if headers is None:
            headers = self.headers
        extra_kw: Dict[str, Any] = {"headers": headers}
        extra_kw.update(urlopen_kw)
        if fields:
            url += "?" + urlencode(fields)
        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(
        self,
        method: str,
        url: str,
        fields: Optional[Union[Dict[str, str], Dict[str, Union[str, Tuple[str, ...]]]]] = None,
        headers: Optional[Dict[str, str]] = None,
        encode_multipart: bool = True,
        multipart_boundary: Optional[str] = None,
        **urlopen_kw: Any
    ) -> Any:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.
        """
        if headers is None:
            headers = self.headers
        extra_kw: Dict[str, Any] = {"headers": {}}
        if fields:
            if "body" in urlopen_kw:
                raise TypeError(
                    "request got values for both 'fields' and 'body', can only specify one."
                )

            if encode_multipart:
                body, content_type = encode_multipart_formdata(
                    fields, boundary=multipart_boundary
                )
            else:
                body, content_type = (
                    urlencode(fields),
                    "application/x-www-form-urlencoded",
                )
            if isinstance(body, six.text_type):
                body = body.encode("utf-8")
            extra_kw["body"] = body
            extra_kw["headers"] = {"Content-Type": content_type}
        extra_kw["headers"].update(headers)
        extra_kw.update(urlopen_kw)
        return self.urlopen(method, url, **extra_kw)
