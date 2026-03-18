```python
from typing import Any
from _pytest.fixtures import FixtureRequest
from eth_utils import Address
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.tests.utils.client import RPCClient
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.pfs import PFSServerMock
from raiden.utils.typing import TokenAmount, TokenNetworkRegistryAddress

def test_api_channel_status_channel_nonexistant(
    api_server_test_instance: APIServer,
    token_addresses: list[Address]
) -> None: ...

def test_api_channel_open_and_deposit(
    api_server_test_instance: APIServer,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_open_and_deposit_race(
    api_server_test_instance: APIServer,
    raiden_network: list[RaidenService],
    token_addresses: list[Address],
    reveal_timeout: int,
    token_network_registry_address: TokenNetworkRegistryAddress,
    retry_timeout: float
) -> None: ...

def test_api_channel_open_close_and_settle(
    api_server_test_instance: APIServer,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_close_insufficient_eth(
    api_server_test_instance: APIServer,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_open_channel_invalid_input(
    api_server_test_instance: APIServer,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_state_change_errors(
    api_server_test_instance: APIServer,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_withdraw(
    api_server_test_instance: APIServer,
    raiden_network: list[RaidenService],
    token_addresses: list[Address],
    pfs_mock: PFSServerMock
) -> None: ...

def test_api_channel_withdraw_with_offline_partner(
    api_server_test_instance: APIServer,
    raiden_network: list[RaidenService],
    token_addresses: list[Address]
) -> None: ...

def test_api_channel_set_reveal_timeout(
    api_server_test_instance: APIServer,
    raiden_network: list[RaidenService],
    token_addresses: list[Address],
    settle_timeout: int
) -> None: ...

def test_api_channel_deposit_limit(
    api_server_test_instance: APIServer,
    proxy_manager: Any,
    token_network_registry_address: TokenNetworkRegistryAddress,
    token_addresses: list[Address],
    reveal_timeout: int
) -> None: ...
```