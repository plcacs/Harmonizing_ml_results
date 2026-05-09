from httpx import URL
from typing import Dict, Any

def test_url_valid_ipv4() -> None:
    url = URL('https://1.2.3.4/')
    assert url.host == '1.2.3.4'

def test_url_invalid_ipv4() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        URL('https://999.999.999.999/')
    assert str(exc.value) == "Invalid IPv4 address: '999.999.999.999'"

def test_ipv6_url() -> None:
    url = URL('http://[::ffff:192.168.0.1]:5678/')
    assert url.host == '::ffff:192.168.0.1'
    assert url.netloc == b'[::ffff:192.168.0.1]:5678'

def test_url_valid_ipv6() -> None:
    url = URL('https://[2001:db8::ff00:42:8329]/')
    assert url.host == '2001:db8::ff00:42:8329'

def test_url_invalid_ipv6() -> None:
    with pytest.raises(httpx.InvalidURL) as exc:
        URL('https://[2001]/')
    assert str(exc.value) == "Invalid IPv6 address: '[2001]'"

def test_ipv6_url_from_raw_url() -> None:
    url = URL(scheme='https', host='[::ffff:192.168.0.1]', port=443, path='/')
    assert url.host == '::ffff:192.168.0.1'
    assert url.netloc == b'[::ffff:192.168.0.1]'
    assert str(url) == 'https://[::ffff:192.168.0.1]/'

def test_ipv6_url_copy_with_host() -> None:
    url_str = 'http://127.0.0.1:1234'
    new_host = '[::ffff:192.168.0.1]'
    url = URL(url_str).copy_with(host=new_host)
    assert url.host == '::ffff:192.168.0.1'
    assert url.netloc == b'[::ffff:192.168.0.1]:1234'
    assert str(url) == 'http://[::ffff:192.168.0.1]:1234'
