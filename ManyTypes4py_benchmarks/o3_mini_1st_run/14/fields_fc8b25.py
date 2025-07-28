from __future__ import absolute_import
import email.utils
import mimetypes
from .packages import six
from typing import Optional, Dict, Any, Union, Tuple, Iterable, Type

def guess_content_type(filename: Optional[str], default: str = 'application/octet-stream') -> str:
    """
    Guess the "Content-Type" of a file.

    :param filename:
        The filename to guess the "Content-Type" of using :mod:`mimetypes`.
    :param default:
        If no "Content-Type" can be guessed, default to `default`.
    """
    if filename:
        return mimetypes.guess_type(filename)[0] or default
    return default

def format_header_param(name: str, value: str) -> str:
    """
    Helper function to format and quote a single header parameter.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows RFC 2231, as
    suggested by RFC 2388 Section 4.4.

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as a unicode string.
    """
    if not any((ch in value for ch in '"\\\r\n')):
        result = '%s="%s"' % (name, value)
        try:
            result.encode('ascii')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            return result
    if not six.PY3 and isinstance(value, six.text_type):
        value = value.encode('utf-8')  # type: ignore
    value = email.utils.encode_rfc2231(value, 'utf-8')
    value = '%s*=%s' % (name, value)
    return value

class RequestField(object):
    """
    A data container for request body parameters.

    :param name:
        The name of this request field.
    :param data:
        The data/value body.
    :param filename:
        An optional filename of the request field.
    :param headers:
        An optional dict-like object of headers to initially use for the field.
    """

    def __init__(self, name: str, data: Any, filename: Optional[str] = None,
                 headers: Optional[Dict[str, Optional[str]]] = None) -> None:
        self._name: str = name
        self._filename: Optional[str] = filename
        self.data: Any = data
        self.headers: Dict[str, Optional[str]] = {}
        if headers:
            self.headers = dict(headers)

    @classmethod
    def from_tuples(cls: Type["RequestField"], fieldname: str, 
                    value: Union[str, Tuple[str, str], Tuple[str, str, str]]) -> "RequestField":
        """
        A :class:`~urllib3.fields.RequestField` factory from old-style tuple parameters.

        Supports constructing :class:`~urllib3.fields.RequestField` from
        parameter of key/value strings AND key/filetuple. A filetuple is a
        (filename, data, MIME type) tuple where the MIME type is optional.
        For example::

            'foo': 'bar',
            'fakefile': ('foofile.txt', 'contents of foofile'),
            'realfile': ('barfile.txt', open('realfile').read()),
            'typedfile': ('bazfile.bin', open('bazfile').read(), 'image/jpeg'),
            'nonamefile': 'contents of nonamefile field',

        Field names and filenames must be unicode.
        """
        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            else:
                filename, data = value
                content_type = guess_content_type(filename)
        else:
            filename = None
            content_type = None
            data = value
        request_param = cls(fieldname, data, filename=filename)
        request_param.make_multipart(content_disposition=None, content_type=content_type)
        return request_param

    def _render_part(self, name: str, value: str) -> str:
        """
        Overridable helper function to format a single header parameter.

        :param name:
            The name of the parameter, a string expected to be ASCII only.
        :param value:
            The value of the parameter, provided as a unicode string.
        """
        return format_header_param(name, value)

    def _render_parts(self, header_parts: Union[Dict[str, Optional[str]], 
                                                 Iterable[Tuple[str, Optional[str]]]]
                       ) -> str:
        """
        Helper function to format and quote a single header.

        Useful for single headers that are composed of multiple items. E.g.,
        'Content-Disposition' fields.

        :param header_parts:
            A sequence of (k, v) tuples or a :class:`dict` of (k, v) to format
            as `k1="v1"; k2="v2"; ...`.
        """
        parts = []
        iterable: Iterable[Tuple[str, Optional[str]]] = header_parts  # type: ignore
        if isinstance(header_parts, dict):
            iterable = header_parts.items()
        for name, value in iterable:
            if value is not None:
                parts.append(self._render_part(name, value))
        return '; '.join(parts)

    def render_headers(self) -> str:
        """
        Renders the headers for this request field.
        """
        lines = []
        sort_keys = ['Content-Disposition', 'Content-Type', 'Content-Location']
        for sort_key in sort_keys:
            if self.headers.get(sort_key, False):
                lines.append('%s: %s' % (sort_key, self.headers[sort_key]))
        for header_name, header_value in self.headers.items():
            if header_name not in sort_keys:
                if header_value:
                    lines.append('%s: %s' % (header_name, header_value))
        lines.append('\r\n')
        return '\r\n'.join(lines)

    def make_multipart(self, content_disposition: Optional[str] = None, 
                       content_type: Optional[str] = None, 
                       content_location: Optional[str] = None) -> None:
        """
        Makes this request field into a multipart request field.

        This method overrides "Content-Disposition", "Content-Type" and
        "Content-Location" headers to the request parameter.

        :param content_type:
            The 'Content-Type' of the request body.
        :param content_location:
            The 'Content-Location' of the request body.

        """
        self.headers['Content-Disposition'] = content_disposition or 'form-data'
        self.headers['Content-Disposition'] += '; '.join(['', self._render_parts((('name', self._name), ('filename', self._filename)))])
        self.headers['Content-Type'] = content_type
        self.headers['Content-Location'] = content_location