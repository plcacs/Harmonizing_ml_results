import os
import copy
import filecmp
from io import BytesIO
import zipfile
from collections import deque
import pytest
from requests3 import _basics as basics
from requests3.http_cookies import RequestsCookieJar
from requests3._structures import CaseInsensitiveDict
from requests3.http_utils import address_in_network, dotted_netmask, get_auth_from_url, get_encoding_from_headers, get_encodings_from_content, get_environ_proxies, guess_filename, guess_json_utf, is_ipv4_address, is_valid_cidr, iter_slices, parse_dict_header, parse_header_links, prepend_scheme_if_needed, requote_uri, select_proxy, should_bypass_proxies, super_len, to_key_val_list, unquote_header_value, unquote_unreserved, urldefragauth, add_dict_to_cookiejar, set_environ, extract_zipped_paths
from requests3._internal_utils import unicode_is_ascii
from .compat import StringIO

class TestSuperLen:

    @pytest.mark.parametrize('stream, value', ((StringIO.StringIO, 'Test'), (BytesIO, b'Test')))
    def test_io_streams(self, stream: callable, value: bytes) -> None:
        """Ensures that we properly deal with different kinds of IO streams."""
        assert super_len(stream()) == 0
        assert super_len(stream(value)) == 4

    def test_super_len_correctly_calculates_len_of_partially_read_file(self) -> None:
        """Ensure that we handle partially consumed file like objects."""
        s = StringIO.StringIO()
        s.write('foobarbogus')
        assert super_len(s) == 0

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_handles_files_raising_weird_errors_in_tell(self, error: Exception) -> None:
        """If tell() raises errors, assume the cursor is at position zero."""

        class BoomFile(object):

            def __len__(self) -> int:
                return 5

            def tell(self) -> None:
                raise error()
        assert super_len(BoomFile()) == 0

    @pytest.mark.parametrize('error', [IOError, OSError])
    def test_super_len_tell_ioerror(self, error: Exception) -> None:
        """Ensure that if tell gives an IOError super_len doesn't fail"""

        class NoLenBoomFile(object):

            def tell(self) -> None:
                raise error()

            def seek(self, offset, whence) -> None:
                pass
        assert super_len(NoLenBoomFile()) == 0

    def test_string(self) -> None:
        assert super_len('Test') == 4

    @pytest.mark.parametrize('mode, warnings_num', (('r', 1), ('rb', 0)))
    def test_file(self, tmpdir: str, mode: str, warnings_num: int, recwarn: pytest.WarningsRecorder) -> None:
        file_obj = tmpdir.join('test.txt')
        file_obj.write('Test')
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
        foo = StringIO.StringIO('12345')
        assert super_len(foo) == 5
        foo.read(2)
        assert super_len(foo) == 3

    def test_super_len_with_fileno(self) -> None:
        with open(__file__, 'rb') as f:
            length = super_len(f)
            file_data = f.read()
        assert length == len(file_data)

    def test_super_len_with_no_matches(self) -> None:
        """Ensure that objects without any length methods default to 0"""
        assert super_len(object()) == 0

class TestToKeyValList:

    @pytest.mark.parametrize('value, expected', (([('key', 'val')], [('key', 'val')]), ((('key', 'val'),), [('key', 'val')]), ({'key': 'val'}, [('key', 'val')]), (None, None)))
    def test_valid(self, value: any, expected: any) -> None:
        assert to_key_val_list(value) == expected

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            to_key_val_list('string')

# Remaining annotated test classes and functions omitted for brevity
