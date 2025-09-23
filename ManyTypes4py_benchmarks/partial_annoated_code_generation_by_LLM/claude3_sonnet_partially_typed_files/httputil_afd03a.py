def _parse_request_range(range_header: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """Parses a Range header.

    Returns either ``None`` or tuple ``(start, end)``.
    Note that while the HTTP headers use inclusive byte positions,
    this method returns indexes suitable for use in slices.

    >>> start, end = _parse_request_range("bytes=1-2")
    >>> start, end
    (1, 3)
    >>> [0, 1, 2, 3, 4][start:end]
    [1, 2]
    >>> _parse_request_range("bytes=6-")
    (6, None)
    >>> _parse_request_range("bytes=-6")
    (-6, None)
    >>> _parse_request_range("bytes=-0")
    (None, 0)
    >>> _parse_request_range("bytes=")
    (None, None)
    >>> _parse_request_range("foo=42")
    >>> _parse_request_range("bytes=1-2,6-10")

    Note: only supports one range (ex, ``bytes=1-2,6-10`` is not allowed).

    See [0] for the details of the range header.

    [0]: http://greenbytes.de/tech/webdav/draft-ietf-httpbis-p5-range-latest.html#byte.ranges
    """
    (unit, _, value) = range_header.partition('=')
    (unit, value) = (unit.strip(), value.strip())
    if unit != 'bytes':
        return None
    (start_b, _, end_b) = value.partition('-')
    try:
        start = _int_or_none(start_b)
        end = _int_or_none(end_b)
    except ValueError:
        return None
    if end is not None:
        if start is None:
            if end != 0:
                start = -end
                end = None
        else:
            end += 1
    return (start, end)

def _get_content_range(start: Optional[int], end: Optional[int], total: int) -> str:
    """Returns a suitable Content-Range header:

    >>> print(_get_content_range(None, 1, 4))
    bytes 0-0/4
    >>> print(_get_content_range(1, 3, 4))
    bytes 1-2/4
    >>> print(_get_content_range(None, None, 4))
    bytes 0-3/4
    """
    start = start or 0
    end = (end or total) - 1
    return f'bytes {start}-{end}/{total}'

def _int_or_none(val: str) -> Optional[int]:
    val = val.strip()
    if val == '':
        return None
    return int(val)

def parse_body_arguments(content_type: str, body: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]], headers: Optional[HTTPHeaders]=None) -> None:
    """Parses a form request body.

    Supports ``application/x-www-form-urlencoded`` and
    ``multipart/form-data``.  The ``content_type`` parameter should be
    a string and ``body`` should be a byte string.  The ``arguments``
    and ``files`` parameters are dictionaries that will be updated
    with the parsed contents.
    """
    if content_type.startswith('application/x-www-form-urlencoded'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            uri_arguments = parse_qs_bytes(body, keep_blank_values=True)
        except Exception as e:
            gen_log.warning('Invalid x-www-form-urlencoded body: %s', e)
            uri_arguments = {}
        for (name, values) in uri_arguments.items():
            if values:
                arguments.setdefault(name, []).extend(values)
    elif content_type.startswith('multipart/form-data'):
        if headers and 'Content-Encoding' in headers:
            gen_log.warning('Unsupported Content-Encoding: %s', headers['Content-Encoding'])
            return
        try:
            fields = content_type.split(';')
            for field in fields:
                (k, sep, v) = field.strip().partition('=')
                if k == 'boundary' and v:
                    parse_multipart_form_data(utf8(v), body, arguments, files)
                    break
            else:
                raise ValueError('multipart boundary not found')
        except Exception as e:
            gen_log.warning('Invalid multipart/form-data: %s', e)

def parse_multipart_form_data(boundary: bytes, data: bytes, arguments: Dict[str, List[bytes]], files: Dict[str, List[HTTPFile]]) -> None:
    """Parses a ``multipart/form-data`` body.

    The ``boundary`` and ``data`` parameters are both byte strings.
    The dictionaries given in the arguments and files parameters
    will be updated with the contents of the body.

    .. versionchanged:: 5.1

       Now recognizes non-ASCII filenames in RFC 2231/5987
       (``filename*=``) format.
    """
    if boundary.startswith(b'"') and boundary.endswith(b'"'):
        boundary = boundary[1:-1]
    final_boundary_index = data.rfind(b'--' + boundary + b'--')
    if final_boundary_index == -1:
        gen_log.warning('Invalid multipart/form-data: no final boundary')
        return
    parts = data[:final_boundary_index].split(b'--' + boundary + b'\r\n')
    for part in parts:
        if not part:
            continue
        eoh = part.find(b'\r\n\r\n')
        if eoh == -1:
            gen_log.warning('multipart/form-data missing headers')
            continue
        headers = HTTPHeaders.parse(part[:eoh].decode('utf-8'))
        disp_header = headers.get('Content-Disposition', '')
        (disposition, disp_params) = _parse_header(disp_header)
        if disposition != 'form-data' or not part.endswith(b'\r\n'):
            gen_log.warning('Invalid multipart/form-data')
            continue
        value = part[eoh + 4:-2]
        if not disp_params.get('name'):
            gen_log.warning('multipart/form-data value missing name')
            continue
        name = disp_params['name']
        if disp_params.get('filename'):
            ctype = headers.get('Content-Type', 'application/unknown')
            files.setdefault(name, []).append(HTTPFile(filename=disp_params['filename'], body=value, content_type=ctype))
        else:
            arguments.setdefault(name, []).append(value)

def format_timestamp(ts: Union[int, float, tuple, time.struct_time, datetime.datetime]) -> str:
    """Formats a timestamp in the format used by HTTP.

    The argument may be a numeric timestamp as returned by `time.time`,
    a time tuple as returned by `time.gmtime`, or a `datetime.datetime`
    object. Naive `datetime.datetime` objects are assumed to represent
    UTC; aware objects are converted to UTC before formatting.

    >>> format_timestamp(1359312200)
    'Sun, 27 Jan 2013 18:43:20 GMT'
    """
    if isinstance(ts, (int, float)):
        time_num = ts
    elif isinstance(ts, (tuple, time.struct_time)):
        time_num = calendar.timegm(ts)
    elif isinstance(ts, datetime.datetime):
        time_num = calendar.timegm(ts.utctimetuple())
    else:
        raise TypeError('unknown timestamp type: %r' % ts)
    return email.utils.formatdate(time_num, usegmt=True)

def parse_request_start_line(line: str) -> RequestStartLine:
    """Returns a (method, path, version) tuple for an HTTP 1.x request line.

    The response is a `typing.NamedTuple`.

    >>> parse_request_start_line("GET /foo HTTP/1.1")
    RequestStartLine(method='GET', path='/foo', version='HTTP/1.1')
    """
    try:
        (method, path, version) = line.split(' ')
    except ValueError:
        raise HTTPInputError('Malformed HTTP request line')
    if not _http_version_re.match(version):
        raise HTTPInputError('Malformed HTTP version in HTTP Request-Line: %r' % version)
    return RequestStartLine(method, path, version)

def parse_response_start_line(line: Union[str, bytes]) -> ResponseStartLine:
    """Returns a (version, code, reason) tuple for an HTTP 1.x response line.

    The response is a `typing.NamedTuple`.

    >>> parse_response_start_line("HTTP/1.1 200 OK")
    ResponseStartLine(version='HTTP/1.1', code=200, reason='OK')
    """
    line = native_str(line)
    match = _http_response_line_re.match(line)
    if not match:
        raise HTTPInputError('Error parsing response start line')
    return ResponseStartLine(match.group(1), int(match.group(2)), match.group(3))

def url_concat(url: str, args: Optional[Union[Dict[str, str], List[Tuple[str, str]], Tuple[Tuple[str, str], ...]]]) -> str:
    """Concatenate url and arguments regardless of whether
    url has existing query parameters.

    ``args`` may be either a dictionary or a list of key-value pairs
    (the latter allows for multiple values with the same key.

    >>> url_concat("http://example.com/foo", dict(c="d"))
    'http://example.com/foo?c=d'
    >>> url_concat("http://example.com/foo?a=b", dict(c="d"))
    'http://example.com/foo?a=b&c=d'
    >>> url_concat("http://example.com/foo?a=b", [("c", "d"), ("c", "d2")])
    'http://example.com/foo?a=b&c=d&c=d2'
    """
    if args is None:
        return url
    parsed_url = urlparse(url)
    if isinstance(args, dict):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args.items())
    elif isinstance(args, list) or isinstance(args, tuple):
        parsed_query = parse_qsl(parsed_url.query, keep_blank_values=True)
        parsed_query.extend(args)
    else:
        err = "'args' parameter should be dict, list or tuple. Not {0}".format(type(args))
        raise TypeError(err)
    final_query = urlencode(parsed_query)
    url = urlunparse((parsed_url[0], parsed_url[1], parsed_url[2], parsed_url[3], final_query, parsed_url[5]))
    return url

def doctests() -> unittest.TestSuite:
    import doctest
    return doctest.DocTestSuite()

def split_host_and_port(netloc: str) -> Tuple[str, Optional[int]]:
    """Returns ``(host, port)`` tuple from ``netloc``.

    Returned ``port`` will be ``None`` if not present.

    .. versionadded:: 4.1
    """
    match = _netloc_re.match(netloc)
    if match:
        host = match.group(1)
        port = int(match.group(2))
    else:
        host = netloc
        port = None
    return (host, port)

def qs_to_qsl(qs: Dict[str, List[AnyStr]]) -> Iterable[Tuple[str, AnyStr]]:
    """Generator converting a result of ``parse_qs`` back to name-value pairs.

    .. versionadded:: 5.0
    """
    for (k, vs) in qs.items():
        for v in vs:
            yield (k, v)

def _unquote_replace(m: re.Match) -> str:
    if m[1]:
        return chr(int(m[1], 8))
    else:
        return m[2]

def _unquote_cookie(s: Optional[str]) -> Optional[str]:
    """Handle double quotes and escaping in cookie values.

    This method is copied verbatim from the Python 3.13 standard
    library (http.cookies._unquote) so we don't have to depend on
    non-public interfaces.
    """
    if s is None or len(s) < 2:
        return s
    if s[0] != '"' or s[-1] != '"':
        return s
    s = s[1:-1]
    return _unquote_sub(_unquote_replace, s)

def parse_cookie(cookie: str) -> Dict[str, str]:
    """Parse a ``Cookie`` HTTP header into a dict of name/value pairs.

    This function attempts to mimic browser cookie parsing behavior;
    it specifically does not follow any of the cookie-related RFCs
    (because browsers don't either).

    The algorithm used is identical to that used by Django version 1.9.10.

    .. versionadded:: 4.4.2
    """
    cookiedict = {}
    for chunk in cookie.split(';'):
        if '=' in chunk:
            (key, val) = chunk.split('=', 1)
        else:
            (key, val) = ('', chunk)
        (key, val) = (key.strip(), val.strip())
        if key or val:
            cookiedict[key] = _unquote_cookie(val)
    return cookiedict
