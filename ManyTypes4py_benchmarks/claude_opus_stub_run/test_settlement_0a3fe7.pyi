from typing import Any

from raiden.raiden_service import RaidenService
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.utils.typing import TokenNetworkAddress, Address

MSG_BLOCKCHAIN_EVENTS: str

def wait_for_batch_unlock(
    app: RaidenService,
    token_network_address: TokenNetworkAddress,
    receiver: Address,
    sender: Address,
) -> None: ...

def is_channel_registered(
    node_app: RaidenService,
    partner_app: RaidenService,
    canonical_identifier: CanonicalIdentifier,
) -> bool: ...

def test_settle_is_automatically_called(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
) -> None: ...

def test_coop_settle_is_automatically_called(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
) -> None: ...

def test_coop_settle_fails_with_pending_lock(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
) -> None: ...

def test_lock_expiry(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
    deposit: int,
) -> None: ...

def test_batch_unlock(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
    secret_registry_address: Any,
    deposit: int,
) -> None: ...

def test_register_secret(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
    secret_registry_address: Any,
) -> None: ...

def test_channel_withdraw(
    raiden_network: list[RaidenService],
    token_addresses: list[Any],
    deposit: int,
    retry_timeout: float,
    pfs_mock: Any,
) -> None: ...

def test_channel_withdraw_expired(
    raiden_network: list[RaidenService],
    network_wait: float,
    number_of_nodes: int,
    token_addresses: list[Any],
    deposit: int,
    retry_timeout: float,
    pfs_mock: Any,
) -> None: ...

def test_settled_lock(
    token_addresses: list[Any],
    raiden_network: list[RaidenService],
    deposit: int,
    retry_timeout: float,
) -> None: ...

def test_automatic_secret_registration(
    raiden_chain: list[RaidenService],
    token_addresses: list[Any],
) -> None: ...

def test_start_end_attack(
    token_addresses: list[Any],
    raiden_chain: list[RaidenService],
    deposit: int,
) -> None: ...

def test_automatic_dispute(
    raiden_network: list[RaidenService],
    deposit: int,
    token_addresses: list[Any],
) -> None: ...

def test_batch_unlock_after_restart(
    raiden_network: list[RaidenService],
    restart_node: Any,
    token_addresses: list[Any],
    deposit: int,
) -> None: ...

def test_handle_insufficient_eth(
    raiden_network: list[RaidenService],
    restart_node: Any,
    token_addresses: list[Any],
    caplog: Any,
) -> None: ...