```pyi
from typing import Any, Set, Tuple
from gevent.greenlet import Greenlet
from gevent.queue import Queue
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.utils.typing import T_ChannelID

SIGNATURE_SIZE_IN_BITS: int

def test_token_network_deposit_race(
    token_network_proxy: TokenNetwork,
    private_keys: Any,
    token_proxy: Any,
    web3: Any,
    contract_manager: Any,
) -> None: ...

def test_token_network_proxy(
    token_network_proxy: TokenNetwork,
    private_keys: Any,
    token_proxy: Any,
    chain_id: Any,
    web3: Any,
    contract_manager: Any,
) -> None: ...

def test_token_network_proxy_update_transfer(
    token_network_proxy: TokenNetwork,
    private_keys: Any,
    token_proxy: Any,
    chain_id: Any,
    web3: Any,
    contract_manager: Any,
) -> None: ...

def test_query_pruned_state(
    token_network_proxy: TokenNetwork,
    private_keys: Any,
    web3: Any,
    contract_manager: Any,
) -> None: ...

def test_token_network_actions_at_pruned_blocks(
    token_network_proxy: TokenNetwork,
    private_keys: Any,
    token_proxy: Any,
    web3: Any,
    chain_id: Any,
    contract_manager: Any,
) -> None: ...

def test_concurrent_set_total_deposit(token_network_proxy: TokenNetwork) -> None: ...
```