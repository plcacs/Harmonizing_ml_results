from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Deque,
    IO,
    Iterator,
    Iterable,
    AnyStr,
    Callable,
    Type,
    TypeVar,
    overload,
)
import io
import os
import pytest
import zipfile
import http.cookiejar
from requests3._structures import CaseInsensitiveDict
from requests3.http_cookies import RequestsCookieJar
from _io import StringIO as PyStringIO

T = TypeVar('T')

class TestSuperLen:
    @pytest.mark.parametrize('stream, value', ((PyStringIO, 'Test'), (BytesIO, b'Test')))
    def test_io_streams(self, stream: type, value: Union[str, bytes]) -> None: ...

    def test_super_len_correctly_calculates_len_of_partially_read_file(self) -> None: ...

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error: Union[IOError, OSError]) -> None: ...

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_tell_ioerror(self, error: Union[IOError, OSError], recwarn: pytest.WarningsRecorder) -> None: ...

    def test_string(self) -> None: ...

    @pytest.mark.parametrize('mode, warnings_num', (('r', 1), ('rb', 0)))
    def test_file(self, tmpdir: Any, mode: str, warnings_num: int, recwarn: pytest.WarningsRecorder) -> None: ...

    def test_super_len_with__len__(self) -> None: ...

    def test_super_len_with_no__len__(self) -> None: ...

    def test_super_len_with_tell(self) -> None: ...

    def test_super_len_with_fileno(self) -> None: ...

    def test_super_len_with_no_matches(self) -> None: ...

class TestToKeyValList:
    @pytest.mark.parametrize('value, expected', (([('key', 'val')], [('key', 'val')]), ((('key', 'val'),), [('key', 'val')]), ({'key': 'val'}, [('key', 'val')]), (None, None)))
    def test_valid(self, value: Union[List[Tuple[str, str]], Tuple[Tuple[str, str]], Dict[str, str], None], expected: List[Tuple[str, str]]) -> None: ...

    def test_invalid(self) -> None: ...

class TestUnquoteHeaderValue:
    @pytest.mark.parametrize('value, expected', ((None, None), ('Test', 'Test'), ('"Test"', 'Test'), ('"Test\\\\"', 'Test\\'), ('"\\\\Comp\\Res"', '\\Comp\\Res')))
    def test_valid(self, value: str, expected: str) -> None: ...

    def test_is_filename(self) -> None: ...

class TestGetEnvironProxies:
    @pytest.fixture(autouse=True, params=['no_proxy', 'NO_PROXY'])
    def no_proxy(self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None: ...

    @pytest.mark.parametrize('url', ('http://192.168.0.1:5000/', 'http://192.168.0.1/', 'http://172.16.1.1/', 'http://172.16.1.1:5000/', 'http://localhost.localdomain:5000/v1.0/'))
    def test_bypass(self, url: str) -> None: ...

    @pytest.mark.parametrize('url', ('http://192.168.1.1:5000/', 'http://192.168.1.1/', 'http://www.requests.com/'))
    def test_not_bypass(self, url: str) -> None: ...

    @pytest.mark.parametrize('url', ('http://192.168.1.1:5000/', 'http://192.168.1.1/', 'http://www.requests.com/'))
    def test_bypass_no_proxy_keyword(self, url: str) -> None: ...

    @pytest.mark.parametrize('url', ('http://192.168.0.1:5000/', 'http://192.168.0.1/', 'http://172.16.1.1/', 'http://172.16.1.1:5000/', 'http://localhost.localdomain:5000/v1.0/'))
    def test_not_bypass_no_proxy_keyword(self, url: str, monkeypatch: pytest.MonkeyPatch) -> None: ...

class TestIsIPv4Address:
    def test_valid(self) -> None: ...

    @pytest.mark.parametrize('value', ('8.8.8.8.8', 'localhost.localdomain'))
    def test_invalid(self, value: str) -> None: ...

class TestIsValidCIDR:
    def test_valid(self) -> None: ...

    @pytest.mark.parametrize('value', ('8.8.8.8', '192.168.1.0/a', '192.168.1.0/128', '192.168.1.0/-1', '192.168.1.999/24'))
    def test_invalid(self, value: str) -> None: ...

class TestAddressInNetwork:
    def test_valid(self) -> None: ...

    def test_invalid(self) -> None: ...

class TestGuessFilename:
    @pytest.mark.parametrize('value', (1, type('Fake', (object,), {'name': 1})()))
    def test_guess_filename_invalid(self, value: Union[int, object]) -> None: ...

    @pytest.mark.parametrize('value, expected_type', ((b'value', bytes), ('value'.decode('utf-8'), str)))
    def test_guess_filename_valid(self, value: Union[bytes, str], expected_type: Union[bytes, str]) -> None: ...

class TestExtractZippedPaths:
    @pytest.mark.parametrize('path', ('/', __file__, pytest.__file__, '/etc/invalid/location'))
    def test_unzipped_paths_unchanged(self, path: str) -> None: ...

    def test_zipped_paths_extracted(self, tmpdir: Any) -> None: ...

class TestContentEncodingDetection:
    def test_none(self) -> None: ...

    @pytest.mark.parametrize('content', ('<meta charset="UTF-8">', '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">', '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />', '<?xml version="1.0" encoding="UTF-8"?>'))
    def test_pragmas(self, content: str) -> None: ...

    def test_precedence(self) -> None: ...

class TestGuessJSONUTF:
    @pytest.mark.parametrize('encoding', ('utf-32', 'utf-8-sig', 'utf-16', 'utf-8', 'utf-16-be', 'utf-16-le', 'utf-32-be', 'utf-32-le'))
    def test_encoded(self, encoding: str) -> None: ...

    def test_bad_utf_like_encoding(self) -> None: ...

    @pytest.mark.parametrize(('encoding', 'expected'), (('utf-16-be', 'utf-16'), ('utf-16-le', 'utf-16'), ('utf-32-be', 'utf-32'), ('utf-32-le', 'utf-32')))
    def test_guess_by_bom(self, encoding: str, expected: str) -> None: ...

@pytest.mark.parametrize('url, auth', (('http://' + ENCODED_USER + ':' + ENCODED_PASSWORD + '@' + 'request.com/url.html#test', (USER, PASSWORD)), ('http://user:pass@complex.url.com/path?query=yes', ('user', 'pass')), ('http://user:pass%20pass@complex.url.com/path?query=yes', ('user', 'pass pass')), ('http://user:pass pass@complex.url.com/path?query=yes', ('user', 'pass pass')), ('http://user%25user:pass@complex.url.com/path?query=yes', ('user%user', 'pass')), ('http://user:pass%23pass@complex.url.com/path?query=yes', ('user', 'pass#pass')), ('http://complex.url.com/path?query=yes', ('', ''))))
def test_get_auth_from_url(url: str, auth: Tuple[str, str]) -> None: ...

@pytest.mark.parametrize('uri, expected', (('http://example.com/fiz?buz=%25ppicture', 'http://example.com/fiz?buz=%25ppicture'), ('http://example.com/fiz?buz=%ppicture', 'http://example.com/fiz?buz=%25ppicture'))
def test_requote_uri_with_unquoted_percents(uri: str, expected: str) -> None: ...

@pytest.mark.parametrize('uri, expected', (('http://example.com/?a=%--', 'http://example.com/?a=%--'), ('http://example.com/?a=%300', 'http://example.com/?a=00'))
def test_unquote_unreserved(uri: str, expected: str) -> None: ...

@pytest.mark.parametrize('mask, expected', ((8, '255.0.0.0'), (24, '255.255.255.0'), (25, '255.255.255.128'))
def test_dotted_netmask(mask: int, expected: str) -> None: ...

@pytest.mark.parametrize('url, expected, proxies', (('hTTp://u:p@Some.Host/path', 'http://some.host.proxy', http_proxies), ('hTTp://u:p@Other.Host/path', 'http://http.proxy', http_proxies), ('hTTp:///path', 'http://http.proxy', http_proxies), ('hTTps://Other.Host', None, http_proxies), ('file:///etc/motd', None, http_proxies), ('hTTp://u:p@Some.Host/path', 'socks5://some.host.proxy', all_proxies), ('hTTp://u:p@Other.Host/path', 'socks5://http.proxy', all_proxies), ('hTTp:///path', 'socks5://http.proxy', all_proxies), ('hTTps://Other.Host', 'socks5://http.proxy', all_proxies), ('http://u:p@other.host/path', 'http://http.proxy', mixed_proxies), ('http://u:p@some.host/path', 'http://some.host.proxy', mixed_proxies), ('https://u:p@other.host/path', 'socks5://http.proxy', mixed_proxies), ('https://u:p@some.host/path', 'socks5://http.proxy', mixed_proxies), ('https://', 'socks5://http.proxy', mixed_proxies), ('file:///etc/motd', 'socks5://http.proxy', all_proxies))
def test_select_proxies(url: str, expected: Optional[str], proxies: Dict[str, str]) -> None: ...

@pytest.mark.parametrize('value, expected', (('foo="is a fish", bar="as well"', {'foo': 'is a fish', 'bar': 'as well'}), ('key_without_value', {'key_without_value': None}))
def test_parse_dict_header(value: str, expected: Dict[str, Optional[str]]) -> None: ...

@pytest.mark.parametrize('value, expected', (('', None), ('T', 'utf-8'), ('Test', 'utf-8'), ('Cont', 'ISO-8859-1'))
def test_get_encoding_from_headers(value: CaseInsensitiveDict, expected: Optional[str]) -> None: ...

@pytest.mark.parametrize('value, length', (('', 0), ('T', 1), ('Test', 4), ('Cont', 0), ('Other', -5), ('Content', None))
def test_iter_slices(value: str, length: Optional[int]) -> None: ...

@pytest.mark.parametrize('value, expected', (('<http:/.../front.jpeg>; rel=front; type="image/jpeg"', [{'url': 'http:/.../front.jpeg', 'rel': 'front', 'type': 'image/jpeg'}]), ('<http:/.../front.jpeg>', [{'url': 'http:/.../front.jpeg'}]), ('<http:/.../front.jpeg>;', [{'url': 'http:/.../front.jpeg'}]), ('<http:/.../front.jpeg>; type="image/jpeg",<http://.../back.jpeg>;', [{'url': 'http:/.../front.jpeg', 'type': 'image/jpeg'}, {'url': 'http://.../back.jpeg'}]), ('', []))
def test_parse_header_links(value: str, expected: List[Dict[str, str]]) -> None: ...

@pytest.mark.parametrize('value, expected', (('example.com/path', 'http://example.com/path'), ('//example.com/path', 'http://example.com/path'))
def test_prepend_scheme_if_needed(value: str, expected: str) -> None: ...

@pytest.mark.parametrize('url, expected', (('http://u:p@example.com/path?a=1#test', 'http://example.com/path?a=1'), ('http://example.com/path', 'http://example.com/path'), ('//u:p@example.com/path', '//example.com/path'), ('//example.com/path', '//example.com/path'), ('example.com/path', '//example.com/path'), ('scheme:u:p@example.com/path', 'scheme://example.com/path'))
def test_urldefragauth(url: str, expected: str) -> None: ...

@pytest.mark.parametrize('url, expected', (('http://192.168.0.1:5000/', True), ('http://192.168.0.1/', True), ('http://172.16.1.1/', True), ('http://172.16.1.1:5000/', True), ('http://localhost.localdomain:5000/v1.0/', True), ('http://172.16.1.12/', False), ('http://172.16.1.12:5000/', False), ('http://google.com:5000/v1.0/', False))
def test_should_bypass_proxies(url: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None: ...

@pytest.mark.parametrize('cookiejar', (http.cookiejar.CookieJar(), RequestsCookieJar()))
def test_add_dict_to_cookiejar(cookiejar: Union[http.cookiejar.CookieJar, RequestsCookieJar]) -> None: ...

@pytest.mark.parametrize('value, expected', ((u'test', True), (u'æíöû', False), (u'ジェーピーニック', False))
def test_unicode_is_ascii(value: str, expected: bool) -> None: ...

@pytest.mark.parametrize('url, expected', (('http://192.168.0.1:5000/', True), ('http://192.168.0.1/', True), ('http://172.16.1.1/', True), ('http://172.16.1.1:5000/', True), ('http://localhost.localdomain:5000/v1.0/', True), ('http://172.16.1.12/', False), ('http://172.16.1.12:5000/', False), ('http://google.com:5000/v1.0/', False))
def test_should_bypass_proxies_no_proxy(url: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None: ...

@pytest.mark.skipif(os.name != 'nt', reason='Test only on Windows')
@pytest.mark.parametrize('url, expected, override', (('http://192.168.0.1:5000/', True, None), ('http://192.168.0.1/', True, None), ('http://172.16.1.1/', True, None), ('http://172.16.1.1:5000/', True, None), ('http://localhost.localdomain:5000/v1.0/', True, None), ('http://172.16.1.22/', False, None), ('http://172.16.1.22:5000/', False, None), ('http://google.com:5000/v1.0/', False, None), ('http://mylocalhostname:5000/v1.0/', True, '<local>'), ('http://192.168.0.1/', False, ''))
def test_should_bypass_proxies_win_registry(url: str, expected: bool, override: Optional[str], monkeypatch: pytest.MonkeyPatch) -> None: ...

@pytest.mark.parametrize('env_name, value', (('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain'), ('no_proxy', None), ('a_new_key', '192.168.0.0/24,127.0.0.1,localhost.localdomain'), ('a_new_key', None))
def test_set_environ(env_name: str, value: Optional[str]) -> None: ...

def test_set_environ_raises_exception() -> None: ...