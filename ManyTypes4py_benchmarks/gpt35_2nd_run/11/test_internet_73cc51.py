from ipaddress import IPv4Address, IPv6Address
from mimesis.enums import DSNType, IPv4Purpose, MimeType, PortRange, TLDType, URLScheme
from mimesis.exceptions import NonEnumerableError
from mimesis import Internet, datasets
import pytest
import re

class TestInternet:

    def net(self) -> Internet:
        return Internet()

    def test_str(self, net: Internet) -> None:
        assert re.match(patterns.PROVIDER_STR_REGEX, str(net))

    def test_dsn(self, net: Internet, dsn_type: DSNType) -> None:
        scheme, port = dsn_type.value
        assert net.dsn(dsn_type=dsn_type).endswith(f':{port}')
        assert net.dsn(dsn_type=dsn_type).startswith(f'{scheme}://')

    # Add type annotations for other test methods as well

class TestSeededInternet:

    def i1(self, seed) -> Internet:
        return Internet(seed=seed)

    def i2(self, seed) -> Internet:
        return Internet(seed=seed)

    # Add type annotations for test methods in TestSeededInternet class
