import pytest
import httpx
from typing import Any, Dict, List, Optional, Tuple, Union


def test_basic_url() -> None:
    url = httpx.URL("https://www.example.com/")

    assert url.scheme == "https"
    assert url.userinfo == b""
    assert url.netloc == b"www.example.com"
    assert url.host == "www.example.com"
    assert url.port is None
    assert url.path == "/"
    assert url.query == b""
    assert url.fragment == ""

    assert str(url) == "https://www.example.com/"
    assert repr(url) == "URL('https://www.example.com/')"


def test_complete_url() -> None:
    url = httpx.URL("https://example.org:123/path/to/somewhere?abc=123#anchor")
    assert url.scheme == "https"
    assert url.host == "example.org"
    assert url.port == 123
    assert url.path == "/path/to/somewhere"
    assert url.query == b"abc=123"
    assert url.raw_path == b"/path/to/somewhere?abc=123"
    assert url.fragment == "anchor"

    assert str(url) == "https://example.org:123/path/to/somewhere?abc=123#anchor"
    assert (
        repr(url) == "URL('https://example.org:123/path/to/somewhere?abc=123#anchor')"
    )


def test_url_with_empty_query() -> None:
    """
    URLs with and without a trailing `?` but an empty query component
    should preserve the information on the raw path.
    """
    url = httpx.URL("https://www.example.com/path")
    assert url.path == "/path"
    assert url.query == b""
    assert url.raw_path == b"/path"

    url = httpx.URL("https://www.example.com/path?")
    assert url.path == "/path"
    assert url.query == b""
    assert url.raw_path == b"/path?"


def test_url_no_scheme() -> None:
    url = httpx.URL("://example.com")
    assert url.scheme == ""
    assert url.host == "example.com"
    assert url.path == "/"


def test_url_no_authority() -> None:
    url = httpx.URL("http://")
    assert url.scheme == "http"
    assert url.host == ""
    assert url.path == "/"


@pytest.mark.parametrize(
    "url,raw_path,path,query,fragment",
    [
        # URL with unescaped chars in path.
        (
            "https://example.com/!$&'()*+,;= abc ABC 123 :/[]@",
            b"/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@",
            "/!$&'()*+,;= abc ABC 123 :/[]@",
            b"",
            "",
        ),
        # URL with escaped chars in path.
        (
            "https://example.com/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@",
            b"/!$&'()*+,;=%20abc%20ABC%20123%20:/[]@",
            "/!$&'()*+,;= abc ABC 123 :/[]@",
            b"",
            "",
        ),
        # URL with mix of unescaped and escaped chars in path.
        # WARNING: This has the incorrect behaviour, adding the test as an interim step.
        (
            "https://example.com/ %61%62%63",
            b"/%20%61%62%63",
            "/ abc",
            b"",
            "",
        ),
        # URL with unescaped chars in query.
        (
            "https://example.com/?!$&'()*+,;= abc ABC 123 :/[]@?",
            b"/?!$&'()*+,;=%20abc%20ABC%20123%20:/[]@?",
            "/",
            b"!$&'()*+,;=%20abc%20ABC%20123%20:/[]@?",
            "",
        ),
        # URL with escaped chars in query.
        (
            "https://example.com/?!$&%27()*+,;=%20abc%20ABC%20123%20:%2F[]@?",
            b"/?!$&%27()*+,;=%20abc%20ABC%20123%20:%2F[]@?",
            "/",
            b"!$&%27()*+,;=%20abc%20ABC%20123%20:%2F[]@?",
            "",
        ),
        # URL with mix of unescaped and escaped chars in query.
        (
            "https://example.com/?%20%97%98%99",
            b"/?%20%97%98%99",
            "/",
            b"%20%97%98%99",
            "",
        ),
        # URL encoding characters in fragment.
        (
            "https://example.com/#!$&'()*+,;= abc ABC 123 :/[]@?#",
            b"/",
            "/",
            b"",
            "!$&'()*+,;= abc ABC 123 :/[]@?#",
        ),
    ],
)
def test_path_query_fragment(
    url: str, raw_path: bytes, path: str, query: bytes, fragment: str
) -> None:
    url = httpx.URL(url)
    assert url.raw_path == raw_path
    assert url.path == path
    assert url.query == query
    assert url.fragment == fragment


def test_url_query_encoding() -> None:
    url = httpx.URL("https://www.example.com/?a=b c&d=e/f")
    assert url.raw_path == b"/?a=b%20c&d=e/f"

    url = httpx.URL("https://www.example.com/?a=b+c&d=e/f")
    assert url.raw_path == b"/?a=b+c&d=e/f"

    url = httpx.URL("https://www.example.com/", params={"a": "b c", "d": "e/f"})
    assert url.raw_path == b"/?a=b+c&d=e%2Ff"


def test_url_params() -> None:
    url = httpx.URL("https://example.org:123/path/to/somewhere", params={"a": "123"})
    assert str(url) == "https://example.org:123/path/to/somewhere?a=123"
    assert url.params == httpx.QueryParams({"a": "123"})

    url = httpx.URL(
        "https://example.org:123/path/to/somewhere?b=456", params={"a": "123"}
    )
    assert str(url) == "https://example.org:123/path/to/somewhere?a=123"
    assert url.params == httpx.QueryParams({"a": "123"})


@pytest.mark.parametrize(
    "url,userinfo,username,password",
    [
        # username and password in URL.
        (
            "https://username:password@example.com",
            b"username:password",
            "username",
            "password",
        ),
        # username and password in URL with percent escape sequences.
        (
            "https://username%40gmail.com:pa%20ssword@example.com",
            b"username%40gmail.com:pa%20ssword",
            "username@gmail.com",
            "pa ssword",
        ),
        (
            "https://user%20name:p%40ssword@example.com",
            b"user%20name:p%40ssword",
            "user name",
            "p@ssword",
        ),
        # username and password in URL without percent escape sequences.
        (
            "https://username@gmail.com:pa ssword@example.com",
            b"username%40gmail.com:pa%20ssword",
            "username@gmail.com",
            "pa ssword",
        ),
        (
            "https://user name:p@ssword@example.com",
            b"user%20name:p%40ssword",
            "user name",
            "p@ssword",
        ),
    ],
)
def test_url_username_and_password(
    url: str, userinfo: bytes, username: str, password: str
) -> None:
    url = httpx.URL(url)
    assert url.userinfo == userinfo
    assert url.username == username
    assert url.password == password


def test_url_valid_host() -> None:
    url = httpx.URL("https://example.com/")
    assert url.host == "example.com"


def test_url_normalized_host() -> None:
    url = httpx.URL("https://EXAMPLE.com/")
    assert url.host == "example.com"


def test_url_percent_escape_host() -> None:
    url = httpx.URL("https://exam le.com/")
    assert url.host == "exam%20le.com"


def test_url_ipv4_like_host() -> None:
    """rare host names used to quality as IPv4"""
    url = httpx.URL("https://023b76x43144/")
    assert url.host == "023b76x43144"


def test_url_valid_port() -> None:
    url = httpx.URL("https://example.com:123/")
    assert url.port == 123


def test_url_normalized_port() -> None:
    # If the port matches the scheme default it is normalized to None.
    url = httpx.URL("https://example.com:443/")
    assert url.port is None


def test_url_invalid_port() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL("https://example.com:abc/")
    assert str(exc.value) == "Invalid port: 'abc'"


def test_url_normalized_path() -> None:
    url = httpx.URL("https://example.com/abc/def/../ghi/./jkl")
    assert url.path == "/abc/ghi/jkl"


def test_url_escaped_path() -> None:
    url = httpx.URL("https://example.com/ /🌟/")
    assert url.raw_path == b"/%20/%F0%9F%8C%9F/"


def test_url_leading_dot_prefix_on_absolute_url() -> None:
    url = httpx.URL("https://example.com/../abc")
    assert url.path == "/abc"


def test_url_leading_dot_prefix_on_relative_url() -> None:
    url = httpx.URL("../abc")
    assert url.path == "../abc"


def test_param_with_space() -> None:
    # Params passed as form key-value pairs should be form escaped,
    # Including the special case of "+" for space seperators.
    url = httpx.URL("http://webservice", params={"u": "with spaces"})
    assert str(url) == "http://webservice?u=with+spaces"


def test_param_requires_encoding() -> None:
    # Params passed as form key-value pairs should be escaped.
    url = httpx.URL("http://webservice", params={"u": "%"})
    assert str(url) == "http://webservice?u=%25"


def test_param_with_percent_encoded() -> None:
    # Params passed as form key-value pairs should always be escaped,
    # even if they include a valid escape sequence.
    # We want to match browser form behaviour here.
    url = httpx.URL("http://webservice", params={"u": "with%20spaces"})
    assert str(url) == "http://webservice?u=with%2520spaces"


def test_param_with_existing_escape_requires_encoding() -> None:
    # Params passed as form key-value pairs should always be escaped,
    # even if they include a valid escape sequence.
    # We want to match browser form behaviour here.
    url = httpx.URL("http://webservice", params={"u": "http://example.com?q=foo%2Fa"})
    assert str(url) == "http://webservice?u=http%3A%2F%2Fexample.com%3Fq%3Dfoo%252Fa"


def test_query_with_existing_percent_encoding() -> None:
    # Valid percent encoded sequences should not be double encoded.
    url = httpx.URL("http://webservice?u=phrase%20with%20spaces")
    assert str(url) == "http://webservice?u=phrase%20with%20spaces"


def test_query_requiring_percent_encoding() -> None:
    # Characters that require percent encoding should be encoded.
    url = httpx.URL("http://webservice?u=phrase with spaces")
    assert str(url) == "http://webservice?u=phrase%20with%20spaces"


def test_query_with_mixed_percent_encoding() -> None:
    # When a mix of encoded and unencoded characters are present,
    # characters that require percent encoding should be encoded,
    # while existing sequences should not be double encoded.
    url = httpx.URL("http://webservice?u=phrase%20with spaces")
    assert str(url) == "http://webservice?u=phrase%20with%20spaces"


def test_url_invalid_hostname() -> None:
    """
    Ensure that invalid URLs raise an `httpx.InvalidURL` exception.
    """
    with pytest.raises(httpx.InvalidURL):
        httpx.URL("https://😇/")


def test_url_excessively_long_url() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL("https://www.example.com/" + "x" * 100_000)
    assert str(exc.value) == "URL too long"


def test_url_excessively_long_component() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL("https://www.example.com", path="/" + "x" * 100_000)
    assert str(exc.value) == "URL component 'path' too long"


def test_url_non_printing_character_in_url() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL("https://www.example.com/\n")
    assert str(exc.value) == (
        "Invalid non-printable ASCII character in URL, '\\n' at position 24."
    )


def test_url_non_printing_character_in_component() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL("https://www.example.com", path="/\n")
    assert str(exc.value) == (
        "Invalid non-printable ASCII character in URL path component, "
        "'\\n' at position 1."
    )


def test_url_with_components() -> None:
    url = httpx.URL(scheme="https", host="www.example.com", path="/")

    assert url.scheme == "https"
    assert url.userinfo == b""
    assert url.host == "www.example.com"
    assert url.port is None
    assert url.path == "/"
    assert url.query == b""
    assert url.fragment == ""

    assert str(url) == "https://www.example.com/"


def test_urlparse_with_invalid_component() -> None:
    with pytest.raises(TypeError) as exc:
        httpx.URL(scheme="https", host="www.example.com", incorrect="/")
    assert str(exc.value) == "'incorrect' is an invalid keyword argument for URL()"


def test_urlparse_with_invalid_scheme() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL(scheme="~", host="www.example.com", path="/")
    assert str(exc.value) == "Invalid URL component 'scheme'"


def test_urlparse_with_invalid_path() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL(scheme="https", host="www.example.com", path="abc")
    assert str(exc.value) == "For absolute URLs, path must be empty or begin with '/'"

    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL(path="//abc")
    assert str(exc.value) == "Relative URLs cannot have a path starting with '//'"

    with pytest.raises(httpx.InvalidURL) as exc:
        httpx.URL(path=":abc")
    assert str(exc.value) == "Relative URLs cannot have a path starting with ':'"


def test_url_with_relative_path() -> None:
    # This path would be invalid for an absolute URL, but is valid as a relative URL.
    url = httpx.URL(path="abc")
    assert url.path == "abc"


def test_url_eq_str() -> None:
    """
    Ensure that `httpx.URL` supports the equality operator.
    """
    url = httpx.URL("https://example.org:123/path/to/somewhere?abc=123#anchor")
    assert url == "https://example.org:123/path/to/somewhere?abc=123#anchor"
    assert str(url) == url


def test_url_set() -> None:
    """
    Ensure that `httpx.URL` instances can be used in sets.
    """
    urls = (
        httpx.URL("http://example.org:123/path/to/somewhere"),
        httpx.URL("http://example.org:123/path/to/somewhere/else"),
    )

    url_set = set(urls)

    assert all(url in urls for url in url_set)


def test_url_invalid_type() -> None:
    """
    Ensure that invalid types on `httpx.URL()` raise a `TypeError`.
    """

    class ExternalURLClass:  # representing external URL class
        pass

    with pytest.raises(TypeError):
        httpx.URL(ExternalURLClass())  # type: ignore


def test_url_with_invalid_component() -> None:
    with pytest.raises(TypeError) as exc:
        httpx.URL(scheme="https", host="www.example.com", incorrect="/")
    assert str(exc.value) == "'incorrect' is an invalid keyword argument for URL()"


def test_url_join() -> None:
    """
   