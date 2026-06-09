from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: decode_hex, encode_hex, to_canonical_address

# === Third-party dependency: gevent ===
# Used symbols: joinall, spawn

# === Extension dependency: gevent.greenlet ===
# Used symbols: Greenlet

# === Third-party dependency: gevent.queue ===
class Queue(SimpleQueue):
    def __init__(self, maxsize = ..., items = ..., unfinished_tasks = ...) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fail, raises

# === Internal dependency: raiden.constants ===
GENESIS_BLOCK_NUMBER = BlockNumber(...)
STATE_PRUNING_AFTER_BLOCKS = 64
EMPTY_HASH = BlockHash(...)
EMPTY_BALANCE_HASH = BalanceHash(...)
EMPTY_SIGNATURE = Signature(...)
LOCKSROOT_OF_NO_LOCKS = Locksroot(...)
BLOCK_ID_LATEST = 'latest'

# === Internal dependency: raiden.exceptions ===
class RaidenRecoverableError(RaidenError): ...
class RaidenUnrecoverableError(RaidenError): ...
class InvalidChannelID(RaidenError): ...
class InvalidSettleTimeout(RaidenError): ...
class SamePeerAddress(RaidenError): ...
class BrokenPreconditionError(RaidenError): ...

# === Internal dependency: raiden.network.proxies.proxy_manager ===
class ProxyManagerMetadata: ...
class ProxyManager:
    def __init__(self, rpc_client, contract_manager, metadata): ...
    def token_network(self, address, block_identifier): ...

# === Internal dependency: raiden.network.rpc.client ===
class JSONRPCClient:
    def __init__(self, web3, privkey, gas_price_strategy=..., block_num_confirmations=...): ...
    def block_number(self): ...
    def can_query_state_for_block(self, block_identifier): ...
    def wait_until_block(self, target_block_number, retry_timeout=...): ...

# === Internal dependency: raiden.tests.integration.network.proxies ===
class BalanceProof:
    def __init__(self, channel_identifier, token_network_address, balance_hash=..., nonce=..., additional_hash=..., chain_id=..., signature=..., transferred_amount=..., locked_amount=..., locksroot=...): ...
    def serialize_bin(self, msg_type=...): ...
    def balance_hash(self): ...
    def balance_hash(self, value): ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address(): ...

# === Internal dependency: raiden.tests.utils.smartcontracts ===
def is_tx_hash_bytes(bytes_): ...

# === Internal dependency: raiden.utils.formatting ===
def to_hex_address(address): ...

# === Internal dependency: raiden.utils.signer ===
class LocalSigner(Signer):
    def __init__(self, private_key): ...
    def sign(self, data, v=...): ...

# === Internal dependency: raiden.utils.typing ===
from eth_typing import BlockNumber
from eth_typing import Hash32
from raiden_contracts.utils.type_aliases import BalanceHash
from raiden_contracts.utils.type_aliases import Locksroot
from raiden_contracts.utils.type_aliases import Signature
from raiden_contracts.utils.type_aliases import T_ChannelID
BlockHash = Hash32

# === Third-party dependency: raiden_contracts.constants ===
class MessageTypeId(IntEnum): ...
TEST_SETTLE_TIMEOUT_MIN: int
TEST_SETTLE_TIMEOUT_MAX: int