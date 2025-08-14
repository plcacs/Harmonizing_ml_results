# -*- coding: utf-8 -*-
"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime
import codecs
from typing import (
    Mapping, 
    Callable, 
    Any, 
    Optional, 
    Union, 
    List, 
    Tuple, 
    Dict, 
    Iterable, 
    Iterator, 
    AsyncIterator,
    cast
)
from types import TracebackType
from io import IOBase

# Import encoding now, to avoid implicit import later.
# Implicit import within threads may cause LookupError when standard library is in a ZIP,
# such as in Embedded Python. See https://github.com/requests/requests/issues/3578.
import rfc3986
import encodings.idna

from .core._http.fields import RequestField
from .core._http.filepost import encode_multipart_formdata
from .core._http.exceptions import (
    DecodeError,
    ReadTimeoutError,
    ProtocolError,
    LocationParseError,
)

from io import UnsupportedOperation
from ._hooks import default_hooks
from ._structures import CaseInsensitiveDict

import requests3 as requests
from .http_auth import HTTPBasicAuth
from .http_cookies import (
    cookiejar_from_dict,
    get_cookie_header,
    _copy_cookie_jar,
)
from .exceptions import (
    HTTPError,
    MissingScheme,
    InvalidURL,
    ChunkedEncodingError,
    ContentDecodingError,
    ConnectionError,
    StreamConsumedError,
    InvalidHeader,
    InvalidBodyError,
    ReadTimeout,
)
from ._internal_utils import to_native_string, unicode_is_ascii
from .http_utils import (
    guess_filename,
    get_auth_from_url,
    requote_uri,
    stream_decode_response_unicode,
    to_key_val_list,
    parse_header_links,
    iter_slices,
    guess_json_utf,
    super_len,
    check_header_validity,
    is_stream,
)
from ._basics import (
    cookielib,
    urlunparse,
    urlsplit,
    urlencode,
    str,
    bytes,
    chardet,
    builtin_str,
    basestring,
)
import json as complexjson
from .http_stati import codes

# : The set of HTTP status codes that indicate an automatically
#: processable redirect.
REDIRECT_STATI = (
    codes["moved"],  # 301
    codes["found"],  # 302
    codes["other"],  # 303
    codes["temporary_redirect"],  # 307
    codes["permanent_redirect"],  # 308
)
DEFAULT_REDIRECT_LIMIT = 30
CONTENT_CHUNK_SIZE = 10 * 1024
ITER_CHUNK_SIZE = 512


class RequestEncodingMixin(object):
    @property
    def path_url(self) -> str:
        """Build the path URL to use."""
        url = []
        p = urlsplit(self.url)
        path = p.path
        if not path:
            path = "/"
        url.append(path)
        query = p.query
        if query:
            url.append("?")
            url.append(query)
        return "".join(url)

    @staticmethod
    def _encode_params(data: Union[str, bytes, Mapping, Iterable]) -> Union[str, bytes]:
        """Encode parameters in a piece of data.

        Will successfully encode parameters when passed as a dict or a list of
        2-tuples. Order is retained if data is a list of 2-tuples but arbitrary
        if parameters are supplied as a dict.
        """
        if isinstance(data, (str, bytes)):
            return data

        elif hasattr(data, "read"):
            return cast(Union[str, bytes], data)

        elif hasattr(data, "__iter__"):
            result = []
            for k, vs in to_key_val_list(data):
                if isinstance(vs, basestring) or not hasattr(vs, "__iter__"):
                    vs = [vs]
                for v in vs:
                    if v is not None:
                        result.append(
                            (
                                k.encode("utf-8") if isinstance(k, str) else k,
                                v.encode("utf-8") if isinstance(v, str) else v,
                            )
                        )
            return urlencode(result, doseq=True)

        else:
            return cast(Union[str, bytes], data)

    @staticmethod
    def _encode_files(
        files: Mapping, data: Optional[Union[str, bytes, Mapping, Iterable]]
    ) -> Tuple[bytes, str]:
        """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
        if not files:
            raise ValueError("Files must be provided.")

        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})
        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
                    # Don't call str() on bytestrings: in Py3 it all goes wrong.
                    if not isinstance(v, bytes):
                        v = str(v)
                    new_fields.append(
                        (
                            field.decode("utf-8")
                            if isinstance(field, bytes)
                            else field,
                            v.encode("utf-8") if isinstance(v, str) else v,
                        )
                    )
        for (k, v) in files:
            # support for explicit filename
            ft = None
            fh = None
            if isinstance(v, (tuple, list)):
                if len(v) == 2:
                    fn, fp = v
                elif len(v) == 3:
                    fn, fp, ft = v
                else:
                    fn, fp, ft, fh = v
            else:
                fn = guess_filename(v) or k
                fp = v
            if isinstance(fp, (str, bytes, bytearray)):
                fdata = fp
            elif hasattr(fp, "read"):
                fdata = fp.read()
            elif fp is None:
                continue
            else:
                fdata = fp

            rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
            rf.make_multipart(content_type=ft)
            new_fields.append(rf)
        body, content_type = encode_multipart_formdata(new_fields)
        return body, content_type


class RequestHooksMixin(object):
    def register_hook(self, event: str, hook: Union[Callable, Iterable[Callable]]) -> None:
        """Properly register a hook."""
        if event not in self.hooks:
            raise ValueError(
                'Unsupported event specified, with event name "%s"' % (event)
            )

        if isinstance(hook, Callable):
            self.hooks[event].append(hook)
        elif hasattr(hook, "__iter__"):
            self.hooks[event].extend(
                h for h in hook if isinstance(h, Callable)
            )

    def deregister_hook(self, event: str, hook: Callable) -> bool:
        """Deregister a previously registered hook.
        Returns True if the hook existed, False if not.
        """
        try:
            self.hooks[event].remove(hook)
            return True

        except ValueError:
            return False


class Request(RequestHooksMixin):
    """A user-created :class:`Request <Request>` object.

    Used to prepare a :class:`PreparedRequest <PreparedRequest>`, which is sent to the server.
    """

    __slots__ = (
        "method",
        "url",
        "headers",
        "files",
        "data",
        "params",
        "auth",
        "cookies",
        "hooks",
        "json",
    )

    def __init__(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Mapping] = None,
        files: Optional[Mapping] = None,
        data: Optional[Union[str, bytes, Mapping, Iterable]] = None,
        params: Optional[Mapping] = None,
        auth: Optional[Union[Tuple[str, str], HTTPBasicAuth]] = None,
        cookies: Optional[Union[Mapping, cookielib.CookieJar]] = None,
        hooks: Optional[Mapping] = None,
        json: Optional[Any] = None,
    ):
        # Default empty dicts for dict params.
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks
        self.hooks = default_hooks()
        for (k, v) in list(hooks.items()):
            self.register_hook(event=k, hook=v)
        self.method = method
        self.url = url
        self.headers = headers
        self.files = files
        self.data = data
        self.json = json
        self.params = params
        self.auth = auth
        self.cookies = cookies

    def __repr__(self) -> str:
        return "<Request [%s]>" % (self.method)

    def prepare(self) -> 'PreparedRequest':
        """Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission and returns it."""
        p = PreparedRequest()
        p.prepare(
            method=self.method,
            url=self.url,
            headers=self.headers,
            files=self.files,
            data=self.data,
            json=self.json,
            params=self.params,
            auth=self.auth,
            cookies=self.cookies,
            hooks=self.hooks,
        )
        return p


class PreparedRequest(RequestEncodingMixin, RequestHooksMixin):
    """The fully mutable :class:`PreparedRequest <PreparedRequest>` object,
    containing the exact bytes that will be sent to the server.
    """

    __slots__ = (
        "method",
        "url",
        "headers",
        "_cookies",
        "body",
        "hooks",
        "_body_position",
    )

    def __init__(self) -> None:
        # : HTTP verb to send to the server.
        self.method: Optional[str] = None
        # : HTTP URL to send the request to.
        self.url: Optional[str] = None
        # : dictionary of HTTP headers.
        self.headers: Optional[CaseInsensitiveDict] = None
        # The `CookieJar` used to create the Cookie header will be stored here
        # after prepare_cookies is called
        self._cookies: Optional[cookielib.CookieJar] = None
        # : request body to send to the server.
        self.body: Optional[Union[bytes, str, IOBase]] = None
        # : dictionary of callback hooks, for internal usage.
        self.hooks: Dict[str, List[Callable]] = default_hooks()
        # : integer denoting starting position of a readable file-like body.
        self._body_position: Optional[int] = None

    def prepare(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Mapping] = None,
        files: Optional[Mapping] = None,
        data: Optional[Union[str, bytes, Mapping, Iterable]] = None,
        params: Optional[Mapping] = None,
        auth: Optional[Union[Tuple[str, str], HTTPBasicAuth]] = None,
        cookies: Optional[Union[Mapping, cookielib.CookieJar]] = None,
        hooks: Optional[Mapping] = None,
        json: Optional[Any] = None,
    ) -> None:
        """Prepares the entire request with the given parameters."""
        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)
        # Note that prepare_auth must be last to enable authentication schemes
        # such as OAuth to work on a fully prepared request.
        # This MUST go after prepare_auth. Authenticators could add a hook
        self.prepare_hooks(hooks)

    def __repr__(self) -> str:
        return f"<PreparedRequest [{self.method}]>"

    def copy(self) -> 'PreparedRequest':
        p = PreparedRequest()
        p.method = self.method
        p.url = self.url
        p.headers = self.headers.copy() if self.headers is not None else None
        p._cookies = _copy_cookie_jar(self._cookies)
        p.body = self.body
        p.hooks = self.hooks
        p._body_position = self._body_position
        return p

    def prepare_method(self, method: Optional[str]) -> None:
        """Prepares the given HTTP method."""
        self.method = method
        if self.method is None:
            raise ValueError('Request method cannot be "None"')

        self.method = to_native_string(self.method.upper())

    @staticmethod
    def _get_idna_encoded_host(host: str) -> str:
        import idna

        try:
            host = idna.encode(host, uts46=True).decode("utf-8")
        except idna.IDNAError:
            raise UnicodeError

        return host

    def prepare_url(self, url: Optional[str], params: Optional[Mapping], validate: bool = False) -> None:
        """Prepares the given HTTP URL."""
        # : Accept objects that have string representations.
        #: We're unable to blindly call unicode/str functions
        #: as this will include the bytestring indicator (b'')
        #: on python 3.x.
        #: https://github.com/requests/requests/pull/2238
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)
        # Ignore any leading and trailing whitespace characters.
        url = url.strip()
        # Don't do any URL preparation for non-HTTP schemes like `mailto`,
        # `data` etc to work around exceptions from `url_parse`, which
        # handles RFC 3986 only.
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return

        # Support for unicode domain names and paths.
        try:
            uri = rfc3986.urlparse(url)
            if validate:
                rfc3986.normalize_uri(url)
        except rfc3986.exceptions.RFC3986Exception:
            raise InvalidURL(f"Invalid URL {url!r}: URL is imporoper.")

        if not uri.scheme:
            error = "Invalid URL {0!r}: No scheme supplied. Perhaps you meant http://{0}?"
            error = error.format(to_native_string(url, "utf8"))
            raise MissingScheme(error)

        if not uri.host:
            raise InvalidURL(f"Invalid URL {url!r}: No host supplied")

        # In general, we want to try IDNA encoding the hostname if the string contains
        # non-ASCII characters. This allows users to automatically get the correct IDNA
        # behaviour. For strings containing only ASCII characters, we need to also verify
        # it doesn't start with a wildcard (*), before allowing the unencoded hostname.
        if not unicode_is_ascii(uri.host):
            try:
                uri = uri.copy_with(host=self._get_idna_encoded_host(uri.host))
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")

        elif uri.host.startswith("*"):
            raise InvalidURL("URL has an invalid label.")

        # Bare domains aren't valid URLs.
        if not uri.path:
            uri = uri.copy_with(path="/")
        if isinstance(params, (str, bytes)):
            params = to_native_string(params)
        enc_params = self._encode_params(params)
        if enc_params:
            if uri.query:
                uri = uri.copy_with(query=f"{uri.query}&{enc_params}")
            else:
                uri = uri.copy_with(query=enc_params)
        # url = requote_uri(
        #     urlunparse([uri.scheme, uri.authority, uri.path, None, uri.query, uri.fragment])
        # )
        # Normalize the URI.
        self.url = rfc3986.normalize_uri(uri.unsplit())

    def prepare_headers(self, headers: Optional[Mapping]) -> None:
        """Prepares the given HTTP headers."""
        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                # Raise exception on invalid header value.
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(
        self,
        data: Optional[Union[str, bytes, Mapping, Iterable]],
        files: Optional[Mapping],
        json: Optional[Any] = None,
    ) -> None:
        """Prepares the given HTTP body data."""
        # Check if file, fo, generator, iterator.
        # If not, run through normal process.
        # Nottin' on you.
        body = None
        content_type = None
        if not data and json is not None:
            # urllib3 requires a bytes-like body. Python 2's json.dumps
            # provides this natively, but Python 3 gives a Unicode string.
            content_type = "application/json"
            body = complexjson.dumps(json)
            if not isinstance(body, bytes):
                body = body.encode("utf-8")

        is_stream = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )

        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None

        if is_stream:
            body = data
            if getattr(body, "tell", None) is not None:
                # Record the current file position before reading.
                # This will allow us to rewind a