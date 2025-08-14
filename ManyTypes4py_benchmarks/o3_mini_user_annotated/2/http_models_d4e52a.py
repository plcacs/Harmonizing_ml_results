#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
requests.models
~~~~~~~~~~~~~~~

This module contains the primary objects that power Requests.
"""

import datetime
import codecs
from typing import Any, Mapping, Callable, Dict, List, Tuple, Optional, Union, Iterable

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
REDIRECT_STATI: Tuple[int, ...] = (
    codes["moved"],  # 301
    codes["found"],  # 302
    codes["other"],  # 303
    codes["temporary_redirect"],  # 307
    codes["permanent_redirect"],  # 308
)
DEFAULT_REDIRECT_LIMIT: int = 30
CONTENT_CHUNK_SIZE: int = 10 * 1024
ITER_CHUNK_SIZE: int = 512


class RequestEncodingMixin(object):
    @property
    def path_url(self) -> str:
        """Build the path URL to use."""
        url_parts: List[str] = []
        p = urlsplit(self.url)
        path: str = p.path
        if not path:
            path = "/"
        url_parts.append(path)
        query: str = p.query
        if query:
            url_parts.append("?")
            url_parts.append(query)
        return "".join(url_parts)

    @staticmethod
    def _encode_params(data: Any) -> Union[str, bytes, Any]:
        """Encode parameters in a piece of data."""
        if isinstance(data, (str, bytes)):
            return data
        elif hasattr(data, "read"):
            return data
        elif hasattr(data, "__iter__"):
            result: List[Tuple[Union[bytes, Any], Union[bytes, Any]]] = []
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
            return data

    @staticmethod
    def _encode_files(files: Any, data: Any) -> Tuple[Any, str]:
        """Build the body for a multipart/form-data request."""
        if not files:
            raise ValueError("Files must be provided.")
        elif isinstance(data, basestring):
            raise ValueError("Data must not be a string.")

        new_fields: List[Any] = []
        fields = to_key_val_list(data or {})
        files = to_key_val_list(files or {})
        for field, val in fields:
            if isinstance(val, basestring) or not hasattr(val, "__iter__"):
                val = [val]
            for v in val:
                if v is not None:
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
            ft: Optional[Any] = None
            fh: Optional[Any] = None
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
    """A user-created Request object."""

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
        headers: Optional[Dict[str, Any]] = None,
        files: Optional[Union[Dict[str, Any], List[Tuple[Any, Any]]]] = None,
        data: Optional[Any] = None,
        params: Optional[Union[Dict[str, Any], List[Tuple[Any, Any]]]] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
    ) -> None:
        data = [] if data is None else data
        files = [] if files is None else files
        headers = {} if headers is None else headers
        params = {} if params is None else params
        hooks = {} if hooks is None else hooks
        self.hooks: Dict[str, List[Callable]] = default_hooks()
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

    def prepare(self) -> "PreparedRequest":
        """Constructs a PreparedRequest for transmission and returns it."""
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
    """The fully mutable PreparedRequest object."""
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
        self.method: Optional[str] = None
        self.url: Optional[str] = None
        self.headers: Optional[CaseInsensitiveDict] = None
        self._cookies: Optional[Any] = None
        self.body: Optional[Any] = None
        self.hooks: Dict[str, List[Callable]] = default_hooks()
        self._body_position: Optional[Any] = None

    def prepare(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        files: Optional[Any] = None,
        data: Optional[Any] = None,
        params: Optional[Any] = None,
        auth: Optional[Any] = None,
        cookies: Optional[Any] = None,
        hooks: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
    ) -> None:
        """Prepares the entire request with the given parameters."""
        self.prepare_method(method)
        self.prepare_url(url, params)
        self.prepare_headers(headers)
        self.prepare_cookies(cookies)
        self.prepare_body(data, files, json)
        self.prepare_auth(auth, url)
        self.prepare_hooks(hooks)

    def __repr__(self) -> str:
        return f"<PreparedRequest [{self.method}]>"

    def copy(self) -> "PreparedRequest":
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

    def prepare_url(self, url: Union[str, bytes, None], params: Any, validate: bool = False) -> None:
        """Prepares the given HTTP URL."""
        if isinstance(url, bytes):
            url = url.decode("utf8")
        else:
            url = str(url)
        url = url.strip()
        if ":" in url and not url.lower().startswith("http"):
            self.url = url
            return
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
        if not unicode_is_ascii(uri.host):
            try:
                uri = uri.copy_with(host=self._get_idna_encoded_host(uri.host))
            except UnicodeError:
                raise InvalidURL("URL has an invalid label.")
        elif uri.host.startswith("*"):
            raise InvalidURL("URL has an invalid label.")
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
        self.url = rfc3986.normalize_uri(uri.unsplit())

    def prepare_headers(self, headers: Optional[Dict[str, Any]]) -> None:
        """Prepares the given HTTP headers."""
        self.headers = CaseInsensitiveDict()
        if headers:
            for header in headers.items():
                check_header_validity(header)
                name, value = header
                self.headers[to_native_string(name)] = value

    def prepare_body(self, data: Any, files: Any, json: Optional[Any] = None) -> None:
        """Prepares the given HTTP body data."""
        body: Optional[Any] = None
        content_type: Optional[str] = None
        if not data and json is not None:
            content_type = "application/json"
            body = complexjson.dumps(json)
            if not isinstance(body, bytes):
                body = body.encode("utf-8")
        is_stream_flag: bool = all(
            [
                hasattr(data, "__iter__"),
                not isinstance(data, (basestring, list, tuple, Mapping)),
            ]
        )
        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None
        if is_stream_flag:
            body = data
            if getattr(body, "tell", None) is not None:
                try:
                    self._body_position = body.tell()
                except (IOError, OSError):
                    self._body_position = object()
            if files:
                raise NotImplementedError("Streamed bodies and files are mutually exclusive.")
        else:
            if files:
                (body, content_type) = self._encode_files(files, data)
            else:
                if data:
                    body = self._encode_params(data)
                    if isinstance(data, basestring) or hasattr(data, "read"):
                        content_type = None
                    else:
                        content_type = "application/x-www-form-urlencoded"
            if content_type and ("content-type" not in self.headers):
                self.headers["Content-Type"] = content_type
        self.prepare_content_length(body)
        self.body = body

    def prepare_content_length(self, body: Any) -> None:
        """Prepares Content-Length header."""
        if body is not None:
            length = super_len(body)
            if length:
                self.headers["Content-Length"] = builtin_str(length)
            elif is_stream(body):
                self.headers["Transfer-Encoding"] = "chunked"
            else:
                raise InvalidBodyError("Non-null body must have length or be streamable.")
        elif (
            self.method not in ("GET", "HEAD")
            and self.headers.get("Content-Length") is None
        ):
            self.headers["Content-Length"] = "0"
        if (
            "Transfer-Encoding" in self.headers
            and "Content-Length" in self.headers
        ):
            raise InvalidHeader(
                "Conflicting Headers: Both Transfer-Encoding and Content-Length are set."
            )

    def prepare_auth(self, auth: Optional[Any], url: str = "") -> None:
        """Prepares the given HTTP auth data."""
        if auth is None:
            url_auth = get_auth_from_url(self.url)
            auth = url_auth if any(url_auth) else None
        if auth:
            if isinstance(auth, tuple) and len(auth) == 2:
                auth = HTTPBasicAuth(*auth)
            r = auth(self)
            self.__dict__.update(r.__dict__)
            self.prepare_content_length(self.body)

    def prepare_cookies(self, cookies: Any) -> None:
        """Prepares the given HTTP cookie data."""
        if isinstance(cookies, cookielib.CookieJar):
            self._cookies = cookies
        else:
            self._cookies = cookiejar_from_dict(cookies)
        cookie_header: Optional[str] = get_cookie_header(self._cookies, self)
        if cookie_header is not None:
            self.headers["Cookie"] = cookie_header

    def prepare_hooks(self, hooks: Optional[Dict[str, Any]]) -> None:
        """Prepares the given hooks."""
        hooks = hooks or []
        for event in hooks:
            self.register_hook(event, hooks[event])

    def send(self, session: Optional[Any] = None, **send_kwargs: Any) -> Any:
        """Sends the PreparedRequest to the given Session."""
        session = requests.Session() if session is None else session
        with session:
            return session.send(self, **send_kwargs)


class Response(object):
    """The Response object, which contains a server's response to an HTTP request."""

    __attrs__ = [
        "_content",
        "status_code",
        "headers",
        "url",
        "history",
        "encoding",
        "reason",
        "cookies",
        "elapsed",
        "request",
        "protocol",
    ]
    __slots__ = __attrs__ + ["_content_consumed", "raw", "_next", "connection"]

    def __init__(self) -> None:
        self._content: Union[bool, bytes] = False
        self._content_consumed: bool = False
        self._next: Optional[Any] = None
        self.status_code: Optional[int] = None
        self.headers: CaseInsensitiveDict = CaseInsensitiveDict()
        self.raw: Optional[Any] = None
        self.url: Optional[str] = None
        self.encoding: Optional[str] = None
        self.history: List[Any] = []
        self.reason: Optional[Union[str, bytes]] = None
        self.cookies: Any = cookiejar_from_dict({})
        self.elapsed: datetime.timedelta = datetime.timedelta(0)
        self.request: Optional[Any] = None

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __getstate__(self) -> Dict[str, Any]:
        if not self._content_consumed:
            _ = self.content
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        setattr(self, "_content_consumed", True)
        setattr(self, "raw", None)

    def __repr__(self) -> str:
        return f"<Response status=({self.status_code}) authority={self.uri.authority!r} protocol={self.protocol!r} elapsed={self.elapsed.microseconds:_}ms>"

    def __iter__(self) -> Iterable[bytes]:
        return self.iter_content(decode_unicode=False)

    @property
    def uri(self) -> Any:
        return rfc3986.urlparse(self.url)  # type: ignore

    @property
    def ok(self) -> bool:
        try:
            self.raise_for_status()
        except HTTPError:
            return False
        return True

    @property
    def is_redirect(self) -> bool:
        return (
            "location" in self.headers and self.status_code in REDIRECT_STATI  # type: ignore
        )

    @property
    def is_permanent_redirect(self) -> bool:
        return "location" in self.headers and self.status_code in (
            codes.moved_permanently,
            codes.permanent_redirect,
        )

    @property
    def next(self) -> Any:
        return self._next

    @property
    def apparent_encoding(self) -> Optional[str]:
        return chardet.detect(self.content)["encoding"]  # type: ignore

    def iter_content(self, decode_unicode: bool = False) -> Iterable[Any]:
        DEFAULT_CHUNK_SIZE: int = 1

        def generate() -> Iterable[bytes]:
            if hasattr(self.raw, "stream"):
                try:
                    for chunk in self.raw.stream():
                        yield chunk
                except ProtocolError as e:
                    if self.headers.get("Transfer-Encoding") == "chunked":
                        raise ChunkedEncodingError(e)
                    else:
                        raise ConnectionError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ReadTimeout(e)
            else:
                while True:
                    chunk = self.raw.read(DEFAULT_CHUNK_SIZE)  # type: ignore
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()

        reused_chunks = iter_slices(self._content, DEFAULT_CHUNK_SIZE) if isinstance(self._content, bytes) else []
        stream_chunks = generate()
        chunks: Iterable[Any] = reused_chunks if self._content_consumed else stream_chunks
        if decode_unicode:
            if self.encoding is None:
                raise TypeError(
                    "encoding must be set before consuming streaming responses"
                )
            codecs.lookup(self.encoding)
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks

    def iter_lines(
        self, chunk_size: int = ITER_CHUNK_SIZE, decode_unicode: bool = False, delimiter: Optional[Any] = None
    ) -> Iterable[Any]:
        carriage_return: Union[str, bytes] = "\r" if decode_unicode else b"\r"
        line_feed: Union[str, bytes] = "\n" if decode_unicode else b"\n"
        pending: Optional[Any] = None
        last_chunk_ends_with_cr: bool = False
        for chunk in self.iter_content(decode_unicode=decode_unicode):
            if not chunk:
                continue
            if pending is not None:
                chunk = pending + chunk
                pending = None
            if delimiter:
                lines = chunk.split(delimiter)
            else:
                skip_first_char: bool = last_chunk_ends_with_cr and chunk.startswith(line_feed)
                last_chunk_ends_with_cr = chunk.endswith(carriage_return)
                if skip_first_char:
                    chunk = chunk[1:]
                    if not chunk:
                        continue
                lines = chunk.splitlines()
            incomplete_line: bool = bool(lines[-1]) and lines[-1][-1:] == chunk[-1:]
            if delimiter or incomplete_line:
                pending = lines.pop()
            for line in lines:
                yield line
        if pending is not None:
            yield pending

    @property
    def content(self) -> bytes:
        if self._content is False:
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")
            if self.status_code == 0 or self.raw is None:
                self._content = None  # type: ignore
            else:
                self._content = bytes().join(self.iter_content()) or bytes()
        self._content_consumed = True
        return self._content  # type: ignore

    @property
    def text(self) -> str:
        content: Optional[str] = None
        encoding: Optional[str] = self.encoding
        if not self.content:
            return str("")
        if self.encoding is None:
            encoding = self.apparent_encoding
        try:
            content = str(self.content, encoding, errors="replace")  # type: ignore
        except (LookupError, TypeError):
            content = str(self.content, errors="replace")  # type: ignore
        return content

    def json(self, **kwargs: Any) -> Any:
        if not self.encoding and self.content and len(self.content) > 3:  # type: ignore
            encoding = guess_json_utf(self.content)  # type: ignore
            if encoding is not None:
                try:
                    content = self.content  # type: ignore
                    return complexjson.loads(content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    pass
        return complexjson.loads(self.text, **kwargs)

    @property
    def links(self) -> Dict[Any, Any]:
        header: Optional[str] = self.headers.get("link")
        l: Dict[Any, Any] = {}
        if header:
            links = parse_header_links(header)
            for link in links:
                key = link.get("rel") or link.get("url")
                l[key] = link
        return l

    def raise_for_status(self) -> "Response":
        http_error_msg: str = ""
        if isinstance(self.reason, bytes):
            try:
                reason = self.reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = self.reason.decode("iso-8859-1")
        else:
            reason = self.reason  # type: ignore
        if 400 <= self.status_code < 500:  # type: ignore
            http_error_msg = "%s Client Error: %s for url: %s" % (
                self.status_code,  # type: ignore
                reason,
                self.url,
            )
        elif 500 <= self.status_code < 600:  # type: ignore
            http_error_msg = "%s Server Error: %s for url: %s" % (
                self.status_code,  # type: ignore
                reason,
                self.url,
            )
        if http_error_msg:
            raise HTTPError(http_error_msg, response=self)
        return self

    def close(self) -> None:
        if not self._content_consumed:
            self.raw.close()  # type: ignore
        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()


class AsyncResponse(Response):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(AsyncResponse, self).__init__(*args, **kwargs)

    async def json(self, **kwargs: Any) -> Any:
        if (not self.encoding and await self.content and len(await self.content) > 3):
            encoding = guess_json_utf(await self.content)
            if encoding is not None:
                try:
                    content = await self.content
                    return complexjson.loads(content.decode(encoding), **kwargs)
                except UnicodeDecodeError:
                    pass
        return complexjson.loads(await self.text, **kwargs)

    @property
    async def text(self) -> str:
        content: Optional[str] = None
        encoding: Optional[str] = self.encoding
        if not await self.content:
            return str("")
        if self.encoding is None:
            encoding = await self.apparent_encoding
        try:
            content = str(await self.content, encoding, errors="replace")
        except (LookupError, TypeError):
            content = str(await self.content, errors="replace")
        return content

    @property
    async def content(self) -> bytes:
        if self._content is False:
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")
            if self.status_code == 0 or self.raw is None:
                self._content = None  # type: ignore
            else:
                self._content = (bytes().join([await self.iter_content()]) or bytes())
        self._content_consumed = True
        return self._content  # type: ignore

    @property
    async def apparent_encoding(self) -> Optional[str]:
        return chardet.detect(await self.content)["encoding"]

    async def iter_content(self, decode_unicode: bool = False) -> Any:
        DEFAULT_CHUNK_SIZE: int = 1

        async def generate() -> AsyncIterable[bytes]:
            if hasattr(self.raw, "stream"):
                try:
                    async for chunk in self.raw.stream():
                        yield chunk
                except ProtocolError as e:
                    if self.headers.get("Transfer-Encoding") == "chunked":
                        raise ChunkedEncodingError(e)
                    else:
                        raise ConnectionError(e)
                except DecodeError as e:
                    raise ContentDecodingError(e)
                except ReadTimeoutError as e:
                    raise ReadTimeout(e)
            else:
                while True:
                    chunk = await self.raw.read(DEFAULT_CHUNK_SIZE)  # type: ignore
                    if not chunk:
                        break
                    yield chunk
            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        try:
            stream_chunks = await generate().__anext__()
        except StopAsyncIteration:
            stream_chunks = None
        reused_chunks = iter_slices(self._content, DEFAULT_CHUNK_SIZE) if isinstance(self._content, bytes) else []
        chunks: Any = reused_chunks if self._content_consumed else stream_chunks
        if decode_unicode:
            if self.encoding is None:
                raise TypeError(
                    "encoding must be set before consuming streaming responses"
                )
            codecs.lookup(self.encoding)
            chunks = stream_decode_response_unicode(chunks, self)
        return chunks
