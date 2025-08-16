# -*- coding: utf-8 -*-
import os
import copy
import filecmp
from io import BytesIO
import zipfile
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union, Deque

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
    extract_zipped_paths
)
from requests3._internal_utils import unicode_is_ascii

from .compat import StringIO


class TestSuperLen:
    @pytest.mark.parametrize(
        "stream, value", ((StringIO.StringIO, "Test"), (BytesIO, b"Test"))
    )
    def test_io_streams(self, stream: Any, value: Union[str, bytes]) -> None:
        """Ensures that we properly deal with different kinds of IO streams."""
        assert super_len(stream()) == 0
        assert super_len(stream(value)) == 4

    def test_super_len_correctly_calculates_len_of_partially_read_file(self) -> None:
        """Ensure that we handle partially consumed file like objects."""
        s = StringIO.StringIO()
        s.write("foobarbogus")
        assert super_len(s) == 0

    @pytest.mark.parametrize("error", [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error: Any) -> None:
        """If tell() raises errors, assume the cursor is at position zero."""

        class BoomFile(object):
            def __len__(self) -> int:
                return 5

            def tell(self) -> None:
                raise error()

        assert super_len(BoomFile()) == 0

    @pytest.mark.parametrize("error", [IOError, OSError])
    def test_super_len_tell_ioerror(self, error: Any) -> None:
        """Ensure that if tell gives an IOError super_len doesn't fail"""

        class NoLenBoomFile(object):
            def tell(self) -> None:
                raise error()

            def seek(self, offset: int, whence: int) -> None:
                pass

        assert super_len(NoLenBoomFile()) == 0

    def test_string(self) -> None:
        assert super_len("Test") == 4

    @pytest.mark.parametrize("mode, warnings_num", (("r", 1), ("rb", 0)))
    def test_file(self, tmpdir: Any, mode: str, warnings_num: int, recwarn: Any) -> None:
        file_obj = tmpdir.join("test.txt")
        file_obj.write("Test")
        with file_obj.open(mode) as fd:
            assert super_len(fd) == 4
        assert len(recwarn) == warnings_num

    def test_super_len_with__len__(self) -> None:
        foo = [1, 2, 3, 4]
        len_foo = super_len(foo)
        assert len_foo == 4

    def test_super_len_with_no__len__(self) -> None:
        class LenFile(object):
            def __init__(self) -> None:
                self.len = 5

        assert super_len(LenFile()) == 5

    def test_super_len_with_tell(self) -> None:
        foo = StringIO.StringIO("12345")
        assert super_len(foo) == 5
        foo.read(2)
        assert super_len(foo) == 3

    def test_super_len_with_fileno(self) -> None:
        with open(__file__, "rb") as f:
            length = super_len(f)
            file_data = f.read()
        assert length == len(file_data)

    def test_super_len_with_no_matches(self) -> None:
        """Ensure that objects without any length methods default to 0"""
        assert super_len(object()) == 0


class TestToKeyValList:
    @pytest.mark.parametrize(
        "value, expected",
        (
            ([("key", "val")], [("key", "val")]),
            ((("key", "val"),), [("key", "val")]),
            ({"key": "val"}, [("key", "val")]),
            (None, None),
        ),
    )
    def test_valid(self, value: Any, expected: Any) -> None:
        assert to_key_val_list(value) == expected

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            to_key_val_list("string")


class TestUnquoteHeaderValue:
    @pytest.mark.parametrize(
        "value, expected",
        (
            (None, None),
            ("Test", "Test"),
            ('"Test"', "Test"),
            ('"Test\\\\"', "Test\\"),
            ('"\\\\Comp\\Res"', "\\Comp\\Res"),
        ),
    )
    def test_valid(self, value: Optional[str], expected: Optional[str]) -> None:
        assert unquote_header_value(value) == expected

    def test_is_filename(self) -> None:
        assert unquote_header_value('"\\\\Comp\\Res"', True) == "\\\\Comp\\Res"


class TestGetEnvironProxies:
    """Ensures that IP addresses are correctly matches with ranges
    in no_proxy variable.
    """

    @pytest.fixture(autouse=True, params=["no_proxy", "NO_PROXY"])
    def no_proxy(self, request: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv(
            request.param, "192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1"
        )

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.0.1:5000/",
            "http://192.168.0.1/",
            "http://172.16.1.1/",
            "http://172.16.1.1:5000/",
            "http://localhost.localdomain:5000/v1.0/",
        ),
    )
    def test_bypass(self, url: str) -> None:
        assert get_environ_proxies(url, no_proxy=None) == {}

    @pytest.mark.parametrize(
        "url",
        ("http://192.168.1.1:5000/", "http://192.168.1.1/", "http://www.requests.com/"),
    )
    def test_not_bypass(self, url: str) -> None:
        assert get_environ_proxies(url, no_proxy=None) != {}

    @pytest.mark.parametrize(
        "url",
        ("http://192.168.1.1:5000/", "http://192.168.1.1/", "http://www.requests.com/"),
    )
    def test_bypass_no_proxy_keyword(self, url: str) -> None:
        no_proxy = "192.168.1.1,requests.com"
        assert get_environ_proxies(url, no_proxy=no_proxy) == {}

    @pytest.mark.parametrize(
        "url",
        (
            "http://192.168.0.1:5000/",
            "http://192.168.0.1/",
            "http://172.16.1.1/",
            "http://172.16.1.1:5000/",
            "http://localhost.localdomain:5000/v1.0/",
        ),
    )
    def test_not_bypass_no_proxy_keyword(self, url: str, monkeypatch: Any) -> None:
        monkeypatch.setenv("http_proxy", "http://proxy.example.com:3128/")
        no_proxy = "192.168.1.1,requests.com"
        assert get_environ_proxies(url, no_proxy=no_proxy) != {}


class TestIsIPv4Address:
    def test_valid(self) -> None:
        assert is_ipv4_address("8.8.8.8")

    @pytest.mark.parametrize("value", ("8.8.8.8.8", "localhost.localdomain"))
    def test_invalid(self, value: str) -> None:
        assert not is_ipv4_address(value)


class TestIsValidCIDR:
    def test_valid(self) -> None:
        assert is_valid_cidr("192.168.1.0/24")

    @pytest.mark.parametrize(
        "value",
        (
            "8.8.8.8",
            "192.168.1.0/a",
            "192.168.1.0/128",
            "192.168.1.0/-1",
            "192.168.1.999/24",
        ),
    )
    def test_invalid(self, value: str) -> None:
        assert not is_valid_cidr(value)


class TestAddressInNetwork:
    def test_valid(self) -> None:
        assert address_in_network("192.168.1.1", "192.168.1.0/24")

    def test_invalid(self) -> None:
        assert not address_in_network("172.16.0.1", "192.168.1.0/24")


class TestGuessFilename:
    @pytest.mark.parametrize("value", (1, type("Fake", (object,), {"name": 1})()))
    def test_guess_filename_invalid(self, value: Any) -> None:
        assert guess_filename(value) is None

    @pytest.mark.parametrize(
        "value, expected_type",
        ((b"value", basics.bytes), (b"value".decode("utf-8"), basics.str)),
    )
    def test_guess_filename_valid(self, value: Union[bytes, str], expected_type: type) -> None:
        obj = type("Fake", (object,), {"name": value})()
        result = guess_filename(obj)
        assert result == value
        assert isinstance(result, expected_type)


class TestExtractZippedPaths:
    @pytest.mark.parametrize(
        "path", ("/", __file__, pytest.__file__, "/etc/invalid/location")
    )
    def test_unzipped_paths_unchanged(self, path: str) -> None:
        assert path == extract_zipped_paths(path)

    def test_zipped_paths_extracted(self, tmpdir: Any) -> None:
        zipped_py = tmpdir.join("test.zip")
        with zipfile.ZipFile(zipped_py.strpath, "w") as f:
            f.write(__file__)

        _, name = os.path.splitdrive(__file__)
        zipped_path = os.path.join(zipped_py.strpath, name.lstrip(r"\/"))
        extracted_path = extract_zipped_paths(zipped_path)

        assert extracted_path != zipped_path
        assert os.path.exists(extracted_path)
        assert filecmp.cmp(extracted_path, __file__)


class TestContentEncodingDetection:
    def test_none(self) -> None:
        encodings = get_encodings_from_content("")
        assert not len(encodings)

    @pytest.mark.parametrize(
        "content",
        (
            '<meta charset="UTF-8">',
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">',
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />',
            '<?xml version="1.0" encoding="UTF-8"?>',
        ),
    )
    def test_pragmas(self, content: str) -> None:
        encodings = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == "UTF-8"

    def test_precedence(self) -> None:
        content = """
        <?xml version="1.0" encoding="XML"?>
        <meta charset="HTML5">
        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />
        """.strip()
        assert get_encodings_from_content(content) == ["HTML5", "HTML4", "XML"]


class TestGuessJSONUTF:
    @pytest.mark.parametrize(
        "encoding",
        (
            "utf-32",
            "utf-8-sig",
            "utf-16",
            "utf-8",
            "utf-16-be",
            "utf-16-le",
            "utf-32-be",
            "utf-32-le",
        ),
    )
    def test_encoded(self, encoding: str) -> None:
        data = "{}".encode(encoding)
        assert guess_json_utf(data) == encoding

    def test_bad_utf_like_encoding(self) -> None:
        assert guess_json_utf(b"\x00\x00\x00\x00") is None

    @pytest.mark.parametrize(
        ("encoding", "expected"),
        (
            ("utf-16-be", "utf-16"),
            ("utf-16-le", "utf-16"),
            ("utf-32-be", "utf-32"),
            ("utf-32-le", "utf-32"),
        ),
    )
    def test_guess_by_bom(self, encoding: str, expected: str) -> None:
        data = u"\ufeff{}".encode(encoding)
        assert guess_json_utf(data) == expected


USER = PASSWORD = "%!*'();:@&=+$,/?#[] "
ENCODED_USER = basics.quote(USER, "")
ENCODED_PASSWORD = basics.quote(PASSWORD, "")


@pytest.mark.parametrize(
    "url, auth",
    (
        (
            "http://"
            + ENCODED_USER
            + ":"
            + ENCODED_PASSWORD
            + "@"
            + "request.com/url.html#test",
            (USER, PASSWORD),
        ),
        ("http://user:pass@complex.url.com/path?query=yes", ("user", "pass")),
        (
            "http://user:pass%20pass@complex.url.com/path?query=yes",
            ("user", "pass pass"),
        ),
        ("http://user:pass pass@complex.url.com/path?query=yes", ("user", "pass pass")),
        (
            "http://user%25user:pass@complex.url.com/path?query=yes",
            ("user%user", "pass"),
        ),
        (
            "http://user:pass%23pass@complex.url.com/path?query=yes",
            ("user", "pass#pass"),
        ),
        ("http://complex.url.com/path?query=yes", ("", "")),
    ),
)
def test_get_auth_from_url(url: str, auth: Tuple[str, str]) -> None:
    assert get_auth_from_url(url) == auth


@pytest.mark.parametrize(
    "uri, expected",
    (
        (
            "http://example.com/fiz?buz=%25ppicture",
            "http://example.com/fiz?buz=%25ppicture",
        ),
        (
            "http://example.com/fiz?buz=%ppicture",
            "http://example.com/fiz?buz=%25ppicture",
        ),
    ),
)
def test_requote_uri_with_unquoted_percents(uri: str, expected: str) -> None:
    assert requote_uri(uri) == expected


@pytest.mark.parametrize(
    "uri, expected",
    (
        ("http://example.com/?a=%--", "http://example.com/?a=%--"),
        ("http://example.com/?a=%300", "http://example.com/?a=00"),
    ),
)
def test_unquote_unreserved(uri: str, expected: str) -> None:
    assert unquote_unreserved(uri) == expected


@pytest.mark.parametrize(
    "mask, expected", ((8, "255.0.0.0"), (24, "255.255.255.0"), (25, "255.255.255.128"))
)
def test_dotted_netmask(mask: int, expected: str) -> None:
    assert dotted_netmask(mask) == expected


http_proxies = {
    "http": "http://http.proxy",
    "http://some.host": "http://some.host.proxy",
}
all_proxies = {
    "all": "socks5://http.proxy",
    "all://some.host": "socks5://some.host.proxy",
}
mixed_proxies = {
    "http": "http://http.proxy",
    "http://some.host": "http://some.host.proxy",
    "all": "socks5://http.proxy",
}


@pytest.mark.parametrize(
    "url, expected, proxies",
    (
        ("hTTp://u:p@Some.Host/path", "http://some.host.proxy", http_proxies),
        ("hTTp://u:p@Other.Host/path", "http://http.proxy", http_proxies),
        ("hTTp:///path", "http://http.proxy", http_proxies),
        ("hTTps://Other.Host", None, http_proxies),
        ("file:///etc/motd", None, http_proxies),
        ("hTTp://u:p@Some.Host/path", "socks5://some.host.proxy", all_proxies),
        ("hTTp://u:p@Other.Host/path", "socks5://http.proxy