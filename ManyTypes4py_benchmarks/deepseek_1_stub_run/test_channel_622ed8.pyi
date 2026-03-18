```python
from typing import Any, List
from http import HTTPStatus
import gevent
import grequests
import pytest
from eth_utils import to_canonical_address, to_checksum_address
from raiden.api.rest import APIServer
from raiden.constants import BLOCK_ID_LATEST, NULL_ADDRESS_HEX
from raiden.raiden_service import RaidenService
from raiden.tests.integration.api.rest.test_rest import DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT
from raiden.tests.integration.api.rest.utils import api_url_for, assert_proper_response, assert_response_with_code, assert_response_with_error, get_json_response
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import check_dict_nested_attrs
from raiden.transfer import views
from raiden.transfer.state import ChannelState
from raiden.utils.typing import TokenAmount
from raiden.waiting import wait_for_participant_deposit
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN

def test_api_channel_status_channel_nonexistant(
    api_server_test_instance: Any,
    token_addresses: Any
) -> None: ...

def test_api_channel_open_and_deposit(
    api_server_test_instance: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...

def test_api_channel_open_and_deposit_race(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    reveal_timeout: Any,
    token_network_registry_address: Any,
    retry_timeout: Any
) -> None: ...

def test_api_channel_open_close_and_settle(
    api_server_test_instance: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...

def test_api_channel_close_insufficient_eth(
    api_server_test_instance: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...

def test_api_channel_open_channel_invalid_input(
    api_server_test_instance: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...

def test_api_channel_state_change_errors(
    api_server_test_instance: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...

def test_api_channel_withdraw(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any
) -> None: ...

def test_api_channel_withdraw_with_offline_partner(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any
) -> None: ...

def test_api_channel_set_reveal_timeout(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    settle_timeout: Any
) -> None: ...

def test_api_channel_deposit_limit(
    api_server_test_instance: Any,
    proxy_manager: Any,
    token_network_registry_address: Any,
    token_addresses: Any,
    reveal_timeout: Any
) -> None: ...
```