```python
import pytest
from gevent import Timeout
from raiden.raiden_service import RaidenService
from raiden.tests.utils.detect_failure import raise_on_failure, expect_failure
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    BlockTimeout as BlockOffset,
    List,
    MessageID,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretRegistryAddress,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    WithdrawAmount,
)
from typing import Any

MSG_BLOCKCHAIN_EVENTS: str = ...

def wait_for_batch_unlock(
    app: Any,
    token_network_address: Any,
    receiver: Any,
    sender: Any,
) -> None: ...

def is_channel_registered(
    node_app: Any,
    partner_app: Any,
    canonical_identifier: Any,
) -> bool: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_settle_is_automatically_called(
    raiden_network: Any,
    token_addresses: Any,
) -> None: ...

@pytest.mark.flaky
@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_coop_settle_is_automatically_called(
    raiden_network: Any,
    token_addresses: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_coop_settle_fails_with_pending_lock(
    raiden_network: Any,
    token_addresses: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_lock_expiry(
    raiden_network: Any,
    token_addresses: Any,
    deposit: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_batch_unlock(
    raiden_network: Any,
    token_addresses: Any,
    secret_registry_address: Any,
    deposit: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("reveal_timeout", [8])
def test_register_secret(
    raiden_network: Any,
    token_addresses: Any,
    secret_registry_address: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_channel_withdraw(
    raiden_network: Any,
    token_addresses: Any,
    deposit: Any,
    retry_timeout: Any,
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_channel_withdraw_expired(
    raiden_network: Any,
    network_wait: Any,
    number_of_nodes: Any,
    token_addresses: Any,
    deposit: Any,
    retry_timeout: Any,
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("channels_per_node", [CHAIN])
def test_settled_lock(
    token_addresses: Any,
    raiden_network: Any,
    deposit: Any,
    retry_timeout: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("channels_per_node", [1])
def test_automatic_secret_registration(
    raiden_chain: Any,
    token_addresses: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.xfail(reason="test incomplete")
@pytest.mark.parametrize("number_of_nodes", [3])
def test_start_end_attack(
    token_addresses: Any,
    raiden_chain: Any,
    deposit: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_automatic_dispute(
    raiden_network: Any,
    deposit: Any,
    token_addresses: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_batch_unlock_after_restart(
    raiden_network: Any,
    restart_node: Any,
    token_addresses: Any,
    deposit: Any,
) -> None: ...

@expect_failure
@pytest.mark.parametrize("number_of_nodes", (2,))
@pytest.mark.parametrize("channels_per_node", (1,))
def test_handle_insufficient_eth(
    raiden_network: Any,
    restart_node: Any,
    token_addresses: Any,
    caplog: Any,
) -> None: ...
```