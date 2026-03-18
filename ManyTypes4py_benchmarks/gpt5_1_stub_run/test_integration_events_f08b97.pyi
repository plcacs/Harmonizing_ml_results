from typing import Any

def get_netting_channel_closed_events(
    proxy_manager: Any,
    token_network_address: Any,
    netting_channel_identifier: Any,
    contract_manager: Any,
    from_block: Any = ...,
    to_block: Any = ...,
) -> Any: ...

def get_netting_channel_deposit_events(
    proxy_manager: Any,
    token_network_address: Any,
    netting_channel_identifier: Any,
    contract_manager: Any,
    from_block: Any = ...,
    to_block: Any = ...,
) -> Any: ...

def get_netting_channel_settled_events(
    proxy_manager: Any,
    token_network_address: Any,
    netting_channel_identifier: Any,
    contract_manager: Any,
    from_block: Any = ...,
    to_block: Any = ...,
) -> Any: ...

def wait_both_channel_open(
    app0: Any,
    app1: Any,
    registry_address: Any,
    token_address: Any,
    retry_timeout: Any,
) -> None: ...

def test_channel_new(
    raiden_chain: Any,
    retry_timeout: Any,
    token_addresses: Any,
) -> None: ...

def test_channel_deposit(
    raiden_chain: Any,
    deposit: Any,
    retry_timeout: Any,
    token_addresses: Any,
) -> None: ...

def test_query_events(
    raiden_chain: Any,
    token_addresses: Any,
    deposit: Any,
    settle_timeout: Any,
    retry_timeout: Any,
    contract_manager: Any,
    blockchain_type: Any,
) -> None: ...

def test_secret_revealed_on_chain(
    raiden_chain: Any,
    deposit: Any,
    settle_timeout: Any,
    token_addresses: Any,
    retry_interval_initial: Any,
) -> None: ...

def test_clear_closed_queue(
    raiden_network: Any,
    token_addresses: Any,
    network_wait: Any,
) -> None: ...