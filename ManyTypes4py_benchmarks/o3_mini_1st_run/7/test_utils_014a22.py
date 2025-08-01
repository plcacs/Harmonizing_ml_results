#!/usr/bin/env python3
import os
import copy
import filecmp
from io import BytesIO
import zipfile
from collections import deque
from typing import Any, Callable, Optional, List, Tuple, Iterator, Dict, Type
import pytest
from requests3 import _basics as basics
from requests3.http_cookies import RequestsCookieJar
from requests3._structures import CaseInsensitiveDict
from requests3.http_utils import (
    address_in_network,
    dotted_netmask,
    get_auth_from_url,
    get_encoding_from_headers,
    get_encodings_from_content,
    get_environ_proxies,
    guess_filename,
    guess_json_utf,
    is_ipv4_address,
    is_valid_cidr,
    iter_slices,
    parse_dict_header,
    parse_header_links,
    prepend_scheme_if_needed,
    requote_uri,
    select_proxy,
    should_bypass_proxies,
    super_len,
    to_key_val_list,
    unquote_header_value,
    unquote_unreserved,
    urldefragauth,
    add_dict_to_cookiejar,
    set_environ,
    extract_zipped_paths,
)
from requests3._internal_utils import unicode_is_ascii
from .compat import StringIO

class TestSuperLen:
    @pytest.mark.parametrize('stream, value', ((StringIO.StringIO, 'Test'), (BytesIO, b'Test')))
    def test_io_streams(self, stream: Callable[..., Any], value: Any) -> None:
        """Ensures that we properly deal with different kinds of IO streams."""
        assert super_len(stream()) == 0
        assert super_len(stream(value)) == 4

    def test_super_len_correctly_calculates_len_of_partially_read_file(self) -> None:
        """Ensure that we handle partially consumed file like objects."""
        s: Any = StringIO.StringIO()
        s.write('foobarbogus')
        assert super_len(s) == 0

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error: Type[Exception]) -> None:
        """If tell() raises errors, assume the cursor is at position zero."""
        class BoomFile(object):
            def __len__(self) -> int:
                return 5
            def tell(self) -> int:
                raise error()
        assert super_len(BoomFile()) == 0

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_tell_ioerror(self, error: Type[Exception]) -> None:
        """Ensure that if tell gives an IOError super_len doesn't fail"""
        class NoLenBoomFile(object):
            def tell(self) -> int:
                raise error()
            def seek(self, offset: int, whence: int) -> None:
                pass
        assert super_len(NoLenBoomFile()) == 0

    def test_string(self) -> None:
        assert super_len('Test') == 4

    @pytest.mark.parametrize('mode, warnings_num', (('r', 1), ('rb', 0)))
    def test_file(self, tmpdir: Any, mode: str, warnings_num: int, recwarn: Any) -> None:
        file_obj = tmpdir.join('test.txt')
        file_obj.write('Test')
        with file_obj.open(mode) as fd:
            assert super_len(fd) == 4
        assert len(recwarn) == warnings_num

    def test_super_len_with__len__(self) -> None:
        foo: List[int] = [1, 2, 3, 4]
        len_foo: int = super_len(foo)
        assert len_foo == 4

    def test_super_len_with_no__len__(self) -> None:
        class LenFile(object):
            def __init__(self) -> None:
                self.len: int = 5
        assert super_len(LenFile()) == 5

    def test_super_len_with_tell(self) -> None:
        foo: Any = StringIO.StringIO('12345')
        assert super_len(foo) == 5
        foo.read(2)
        assert super_len(foo) == 3

    def test_super_len_with_fileno(self) -> None:
        with open(__file__, 'rb') as f:
            length: int = super_len(f)
            file_data: bytes = f.read()
        assert length == len(file_data)

    def test_super_len_with_no_matches(self) -> None:
        """Ensure that objects without any length methods default to 0"""
        assert super_len(object()) == 0

class TestToKeyValList:
    @pytest.mark.parametrize(
        'value, expected',
        (
            ([('key', 'val')], [('key', 'val')]),
            ((('key', 'val'),), [('key', 'val')]),
            ({'key': 'val'}, [('key', 'val')]),
            (None, None),
        )
    )
    def test_valid(self, value: Any, expected: Optional[List[Tuple[str, str]]]) -> None:
        assert to_key_val_list(value) == expected

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            to_key_val_list('string')

class TestUnquoteHeaderValue:
    @pytest.mark.parametrize(
        'value, expected',
        (
            (None, None),
            ('Test', 'Test'),
            ('"Test"', 'Test'),
            ('"Test\\\\"', 'Test\\'),
            ('"\\\\Comp\\Res"', '\\Comp\\Res'),
        )
    )
    def test_valid(self, value: Optional[str], expected: Optional[str]) -> None:
        assert unquote_header_value(value) == expected

    def test_is_filename(self) -> None:
        assert unquote_header_value('"\\\\Comp\\Res"', True) == '\\\\Comp\\Res'

class TestGetEnvironProxies:
    """Ensures that IP addresses are correctly matches with ranges in no_proxy variable."""
    @pytest.fixture(autouse=True, params=['no_proxy', 'NO_PROXY'])
    def no_proxy(self, request: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv(request.param, '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')

    @pytest.mark.parametrize(
        'url',
        (
            'http://192.168.0.1:5000/', 
            'http://192.168.0.1/', 
            'http://172.16.1.1/', 
            'http://172.16.1.1:5000/', 
            'http://localhost.localdomain:5000/v1.0/'
        )
    )
    def test_bypass(self, url: str) -> None:
        assert get_environ_proxies(url, no_proxy=None) == {}

    @pytest.mark.parametrize(
        'url',
        (
            'http://192.168.1.1:5000/', 
            'http://192.168.1.1/', 
            'http://www.requests.com/'
        )
    )
    def test_not_bypass(self, url: str) -> None:
        assert get_environ_proxies(url, no_proxy=None) != {}

    @pytest.mark.parametrize(
        'url',
        (
            'http://192.168.1.1:5000/', 
            'http://192.168.1.1/', 
            'http://www.requests.com/'
        )
    )
    def test_bypass_no_proxy_keyword(self, url: str) -> None:
        no_proxy: str = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) == {}

    @pytest.mark.parametrize(
        'url',
        (
            'http://192.168.0.1:5000/', 
            'http://192.168.0.1/', 
            'http://172.16.1.1/', 
            'http://172.16.1.1:5000/', 
            'http://localhost.localdomain:5000/v1.0/'
        )
    )
    def test_not_bypass_no_proxy_keyword(self, url: str, monkeypatch: Any) -> None:
        monkeypatch.setenv('http_proxy', 'http://proxy.example.com:3128/')
        no_proxy: str = '192.168.1.1,requests.com'
        assert get_environ_proxies(url, no_proxy=no_proxy) != {}

class TestIsIPv4Address:
    def test_valid(self) -> None:
        assert is_ipv4_address('8.8.8.8')

    @pytest.mark.parametrize('value', ('8.8.8.8.8', 'localhost.localdomain'))
    def test_invalid(self, value: str) -> None:
        assert not is_ipv4_address(value)

class TestIsValidCIDR:
    def test_valid(self) -> None:
        assert is_valid_cidr('192.168.1.0/24')

    @pytest.mark.parametrize('value', ('8.8.8.8', '192.168.1.0/a', '192.168.1.0/128', '192.168.1.0/-1', '192.168.1.999/24'))
    def test_invalid(self, value: str) -> None:
        assert not is_valid_cidr(value)

class TestAddressInNetwork:
    def test_valid(self) -> None:
        assert address_in_network('192.168.1.1', '192.168.1.0/24')

    def test_invalid(self) -> None:
        assert not address_in_network('172.16.0.1', '192.168.1.0/24')

class TestGuessFilename:
    @pytest.mark.parametrize('value', (1, type('Fake', (object,), {'name': 1})()))
    def test_guess_filename_invalid(self, value: Any) -> None:
        assert guess_filename(value) is None

    @pytest.mark.parametrize('value, expected_type', ((b'value', basics.bytes), (b'value'.decode('utf-8'), basics.str)))
    def test_guess_filename_valid(self, value: Any, expected_type: Any) -> None:
        obj: Any = type('Fake', (object,), {'name': value})()
        result: Any = guess_filename(obj)
        assert result == value
        assert isinstance(result, expected_type)

class TestExtractZippedPaths:
    @pytest.mark.parametrize('path', ('/', __file__, pytest.__file__, '/etc/invalid/location'))
    def test_unzipped_paths_unchanged(self, path: str) -> None:
        assert path == extract_zipped_paths(path)

    def test_zipped_paths_extracted(self, tmpdir: Any) -> None:
        zipped_py = tmpdir.join('test.zip')
        with zipfile.ZipFile(zipped_py.strpath, 'w') as f:
            f.write(__file__)
        _, name = os.path.splitdrive(__file__)
        zipped_path: str = os.path.join(zipped_py.strpath, name.lstrip('\\/'))
        extracted_path: str = extract_zipped_paths(zipped_path)
        assert extracted_path != zipped_path
        assert os.path.exists(extracted_path)
        assert filecmp.cmp(extracted_path, __file__)

class TestContentEncodingDetection:
    def test_none(self) -> None:
        encodings: List[str] = get_encodings_from_content('')
        assert not len(encodings)

    @pytest.mark.parametrize(
        'content',
        (
            '<meta charset="UTF-8">',
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">',
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />',
            '<?xml version="1.0" encoding="UTF-8"?>'
        )
    )
    def test_pragmas(self, content: str) -> None:
        encodings: List[str] = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == 'UTF-8'

    def test_precedence(self) -> None:
        content: str = (
            '\n        <?xml version="1.0" encoding="XML"?>\n'
            '        <meta charset="HTML5">\n'
            '        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />\n'
            '        '
        ).strip()
        assert get_encodings_from_content(content) == ['HTML5', 'HTML4', 'XML']

class TestGuessJSONUTF:
    @pytest.mark.parametrize('encoding', ('utf-32', 'utf-8-sig', 'utf-16', 'utf-8', 'utf-16-be', 'utf-16-le', 'utf-32-be', 'utf-32-le'))
    def test_encoded(self, encoding: str) -> None:
        data: bytes = '{}'.encode(encoding)
        assert guess_json_utf(data) == encoding

    def test_bad_utf_like_encoding(self) -> None:
        assert guess_json_utf(b'\x00\x00\x00\x00') is None

    @pytest.mark.parametrize(('encoding', 'expected'), (('utf-16-be', 'utf-16'), ('utf-16-le', 'utf-16'), ('utf-32-be', 'utf-32'), ('utf-32-le', 'utf-32')))
    def test_guess_by_bom(self, encoding: str, expected: str) -> None:
        data: bytes = u'\ufeff{}'.encode(encoding)
        assert guess_json_utf(data) == expected

USER: str = PASSWORD: str = "%!*'();:@&=+$,/?#[] "
ENCODED_USER: str = basics.quote(USER, '')
ENCODED_PASSWORD: str = basics.quote(PASSWORD, '')

@pytest.mark.parametrize(
    'url, auth',
    (
        ('http://' + ENCODED_USER + ':' + ENCODED_PASSWORD + '@' + 'request.com/url.html#test', (USER, PASSWORD)),
        ('http://user:pass@complex.url.com/path?query=yes', ('user', 'pass')),
        ('http://user:pass%20pass@complex.url.com/path?query=yes', ('user', 'pass pass')),
        ('http://user:pass pass@complex.url.com/path?query=yes', ('user', 'pass pass')),
        ('http://user%25user:pass@complex.url.com/path?query=yes', ('user%user', 'pass')),
        ('http://user:pass%23pass@complex.url.com/path?query=yes', ('user', 'pass#pass')),
        ('http://complex.url.com/path?query=yes', ('', ''))
    )
)
def test_get_auth_from_url(url: str, auth: Tuple[str, str]) -> None:
    assert get_auth_from_url(url) == auth

@pytest.mark.parametrize(
    'uri, expected',
    (
        ('http://example.com/fiz?buz=%25ppicture', 'http://example.com/fiz?buz=%25ppicture'),
        ('http://example.com/fiz?buz=%ppicture', 'http://example.com/fiz?buz=%25ppicture')
    )
)
def test_requote_uri_with_unquoted_percents(uri: str, expected: str) -> None:
    """See: https://github.com/requests/requests/issues/2356"""
    assert requote_uri(uri) == expected

@pytest.mark.parametrize(
    'uri, expected',
    (
        ('http://example.com/?a=%--', 'http://example.com/?a=%--'),
        ('http://example.com/?a=%300', 'http://example.com/?a=00')
    )
)
def test_unquote_unreserved(uri: str, expected: str) -> None:
    assert unquote_unreserved(uri) == expected

@pytest.mark.parametrize('mask, expected', ((8, '255.0.0.0'), (24, '255.255.255.0'), (25, '255.255.255.128')))
def test_dotted_netmask(mask: int, expected: str) -> None:
    assert dotted_netmask(mask) == expected

http_proxies: Dict[str, str] = {'http': 'http://http.proxy', 'http://some.host': 'http://some.host.proxy'}
all_proxies: Dict[str, str] = {'all': 'socks5://http.proxy', 'all://some.host': 'socks5://some.host.proxy'}
mixed_proxies: Dict[str, str] = {'http': 'http://http.proxy', 'http://some.host': 'http://some.host.proxy', 'all': 'socks5://http.proxy'}

@pytest.mark.parametrize(
    'url, expected, proxies',
    (
        ('hTTp://u:p@Some.Host/path', 'http://some.host.proxy', http_proxies),
        ('hTTp://u:p@Other.Host/path', 'http://http.proxy', http_proxies),
        ('hTTp:///path', 'http://http.proxy', http_proxies),
        ('hTTps://Other.Host', None, http_proxies),
        ('file:///etc/motd', None, http_proxies),
        ('hTTp://u:p@Some.Host/path', 'socks5://some.host.proxy', all_proxies),
        ('hTTp://u:p@Other.Host/path', 'socks5://http.proxy', all_proxies),
        ('hTTp:///path', 'socks5://http.proxy', all_proxies),
        ('hTTps://Other.Host', 'socks5://http.proxy', all_proxies),
        ('http://u:p@other.host/path', 'http://http.proxy', mixed_proxies),
        ('http://u:p@some.host/path', 'http://some.host.proxy', mixed_proxies),
        ('https://u:p@other.host/path', 'socks5://http.proxy', mixed_proxies),
        ('https://u:p@some.host/path', 'socks5://http.proxy', mixed_proxies),
        ('https://', 'socks5://http.proxy', mixed_proxies),
        ('file:///etc/motd', 'socks5://http.proxy', all_proxies)
    )
)
def test_select_proxies(url: str, expected: Optional[str], proxies: Dict[str, str]) -> None:
    """Make sure we can select per-host proxies correctly."""
    assert select_proxy(url, proxies) == expected

@pytest.mark.parametrize(
    'value, expected',
    (
        ('foo="is a fish", bar="as well"', {'foo': 'is a fish', 'bar': 'as well'}),
        ('key_without_value', {'key_without_value': None})
    )
)
def test_parse_dict_header(value: str, expected: Dict[str, Optional[str]]) -> None:
    assert parse_dict_header(value) == expected

@pytest.mark.parametrize(
    'value, expected',
    (
        (CaseInsensitiveDict(), None),
        (CaseInsensitiveDict({'content-type': 'application/json; charset=utf-8'}), 'utf-8'),
        (CaseInsensitiveDict({'content-type': 'text/plain'}), 'ISO-8859-1')
    )
)
def test_get_encoding_from_headers(value: CaseInsensitiveDict, expected: Optional[str]) -> None:
    assert get_encoding_from_headers(value) == expected

@pytest.mark.parametrize(
    'value, length',
    (
        ('', 0),
        ('T', 1),
        ('Test', 4),
        ('Cont', 0),
        ('Other', -5),
        ('Content', None)
    )
)
def test_iter_slices(value: str, length: Optional[int]) -> None:
    if length is None or (length <= 0 and len(value) > 0):
        assert len(list(iter_slices(value, length))) == 1
    else:
        assert len(list(iter_slices(value, 1))) == length

@pytest.mark.parametrize(
    'value, expected',
    (
        ('<http:/.../front.jpeg>; rel=front; type="image/jpeg"', [{'url': 'http:/.../front.jpeg', 'rel': 'front', 'type': 'image/jpeg'}]),
        ('<http:/.../front.jpeg>', [{'url': 'http:/.../front.jpeg'}]),
        ('<http:/.../front.jpeg>;', [{'url': 'http:/.../front.jpeg'}]),
        ('<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;', [{'url': 'http:/.../front.jpeg', 'type': 'image/jpeg'}, {'url': 'http://.../back.jpeg'}]),
        ('', [])
    )
)
def test_parse_header_links(value: str, expected: List[Dict[str, Any]]) -> None:
    assert parse_header_links(value) == expected

@pytest.mark.parametrize(
    'value, expected',
    (
        ('example.com/path', 'http://example.com/path'),
        ('//example.com/path', 'http://example.com/path')
    )
)
def test_prepend_scheme_if_needed(value: str, expected: str) -> None:
    assert prepend_scheme_if_needed(value, 'http') == expected

@pytest.mark.parametrize(
    'url, expected',
    (
        ('http://u:p@example.com/path?a=1#test', 'http://example.com/path?a=1'),
        ('http://example.com/path', 'http://example.com/path'),
        ('//u:p@example.com/path', '//example.com/path'),
        ('//example.com/path', '//example.com/path'),
        ('example.com/path', '//example.com/path'),
        ('scheme:u:p@example.com/path', 'scheme://example.com/path')
    )
)
def test_urldefragauth(url: str, expected: str) -> None:
    assert urldefragauth(url) == expected

@pytest.mark.parametrize(
    'url, expected',
    (
        ('http://192.168.0.1:5000/', True),
        ('http://192.168.0.1/', True),
        ('http://172.16.1.1/', True),
        ('http://172.16.1.1:5000/', True),
        ('http://localhost.localdomain:5000/v1.0/', True),
        ('http://172.16.1.12/', False),
        ('http://172.16.1.12:5000/', False),
        ('http://google.com:5000/v1.0/', False)
    )
)
def test_should_bypass_proxies(url: str, expected: bool, monkeypatch: Any) -> None:
    """Tests for function should_bypass_proxies to check if proxy can be bypassed or not"""
    monkeypatch.setenv('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')
    monkeypatch.setenv('NO_PROXY', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1')
    assert should_bypass_proxies(url, no_proxy=None) == expected

@pytest.mark.parametrize('cookiejar', (basics.cookielib.CookieJar(), RequestsCookieJar()))
def test_add_dict_to_cookiejar(cookiejar: Any) -> None:
    """Ensure add_dict_to_cookiejar works for non-RequestsCookieJar CookieJars"""
    cookiedict: Dict[str, str] = {'test': 'cookies', 'good': 'cookies'}
    cj: Any = add_dict_to_cookiejar(cookiejar, cookiedict)
    cookies: Dict[str, str] = {cookie.name: cookie.value for cookie in cj}
    assert cookiedict == cookies

@pytest.mark.parametrize('value, expected', ((u'test', True), (u'æíöû', False), (u'ジェーピーニック', False)))
def test_unicode_is_ascii(value: str, expected: bool) -> None:
    assert unicode_is_ascii(value) is expected

@pytest.mark.parametrize(
    'url, expected',
    (
        ('http://192.168.0.1:5000/', True),
        ('http://192.168.0.1/', True),
        ('http://172.16.1.1/', True),
        ('http://172.16.1.1:5000/', True),
        ('http://localhost.localdomain:5000/v1.0/', True),
        ('http://172.16.1.12/', False),
        ('http://172.16.1.12:5000/', False),
        ('http://google.com:5000/v1.0/', False)
    )
)
def test_should_bypass_proxies_no_proxy(url: str, expected: bool) -> None:
    """Tests for function should_bypass_proxies to check if proxy can be bypassed or not using the 'no_proxy' argument"""
    no_proxy: str = '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1'
    assert should_bypass_proxies(url, no_proxy=no_proxy) == expected

@pytest.mark.skipif(os.name != 'nt', reason='Test only on Windows')
@pytest.mark.parametrize(
    'url, expected, override',
    (
        ('http://192.168.0.1:5000/', True, None),
        ('http://192.168.0.1/', True, None),
        ('http://172.16.1.1/', True, None),
        ('http://172.16.1.1:5000/', True, None),
        ('http://localhost.localdomain:5000/v1.0/', True, None),
        ('http://172.16.1.22/', False, None),
        ('http://172.16.1.22:5000/', False, None),
        ('http://google.com:5000/v1.0/', False, None),
        ('http://mylocalhostname:5000/v1.0/', True, '<local>'),
        ('http://192.168.0.1/', False, '')
    )
)
def test_should_bypass_proxies_win_registry(url: str, expected: bool, override: Optional[str], monkeypatch: Any) -> None:
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows registry settings
    """
    if override is None:
        override = '192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1'
    import winreg
    class RegHandle:
        def Close(self) -> None:
            pass
    ie_settings: RegHandle = RegHandle()
    proxyEnableValues: Any = deque([1, '1'])
    def OpenKey(key: Any, subkey: Any) -> RegHandle:
        return ie_settings
    def QueryValueEx(key: Any, value_name: str) -> List[Any]:
        if key is ie_settings:
            if value_name == 'ProxyEnable':
                return [1]
            elif value_name == 'ProxyOverride':
                return [override]
        return [None]
    monkeypatch.setenv('http_proxy', '')
    monkeypatch.setenv('https_proxy', '')
    monkeypatch.setenv('ftp_proxy', '')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setattr(winreg, 'OpenKey', OpenKey)
    monkeypatch.setattr(winreg, 'QueryValueEx', QueryValueEx)
    assert should_bypass_proxies(url, None) == expected

@pytest.mark.parametrize(
    'env_name, value',
    (
        ('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain'),
        ('no_proxy', None),
        ('a_new_key', '192.168.0.0/24,127.0.0.1,localhost.localdomain'),
        ('a_new_key', None)
    )
)
def test_set_environ(env_name: str, value: Optional[str]) -> None:
    """Tests set_environ will set environ values and will restore the environ."""
    environ_copy: Dict[str, str] = copy.deepcopy(os.environ)
    with set_environ(env_name, value):
        assert os.environ.get(env_name) == value
    assert os.environ == environ_copy

def test_set_environ_raises_exception() -> None:
    """Tests set_environ will raise exceptions in context when the value parameter is None."""
    with pytest.raises(Exception) as exception:
        with set_environ('test1', None):
            raise Exception('Expected exception')
    assert 'Expected exception' in str(exception.value)