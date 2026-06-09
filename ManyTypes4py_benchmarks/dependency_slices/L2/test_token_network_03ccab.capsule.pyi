from typing import Any

# === Third-party dependency: eth_typing ===
# Used symbols: Hash32

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
BLOCK_ID_LATEST: Literal['latest']
GENESIS_BLOCK_NUMBER: BlockNumber
STATE_PRUNING_AFTER_BLOCKS: int
EMPTY_HASH: BlockHash
EMPTY_BALANCE_HASH: BalanceHash
EMPTY_SIGNATURE: Signature
LOCKSROOT_OF_NO_LOCKS: Locksroot

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
    def __init__(self, rpc_client: JSONRPCClient, contract_manager: ContractManager, metadata: ProxyManagerMetadata) -> None: ...
    def token_network(self, address: TokenNetworkAddress, block_identifier: BlockIdentifier) -> TokenNetwork: ...

# === Internal dependency: raiden.network.rpc.client ===
class JSONRPCClient:
    def __init__(self, web3: Web3, privkey: PrivateKey, gas_price_strategy: Callable = ..., block_num_confirmations: int = ...) -> None: ...
    def block_number(self) -> BlockNumber: ...
    def can_query_state_for_block(self, block_identifier: BlockIdentifier) -> bool: ...
    def wait_until_block(self, target_block_number: BlockNumber, retry_timeout: float = ...) -> BlockNumber: ...

# === Internal dependency: raiden.tests.integration.network.proxies ===
class BalanceProof:
    def __init__(self, channel_identifier: ChannelID, token_network_address: TokenNetworkAddress, balance_hash: BalanceHash = ..., nonce: int = ..., additional_hash: str = ..., chain_id: int = ..., signature: str = ..., transferred_amount: TokenAmount = ..., locked_amount: LockedAmount = ..., locksroot: Locksroot = ...) -> Any: ...
    def serialize_bin(self, msg_type: MessageTypeId = ...) -> Any: ...
    def balance_hash(self) -> BalanceHash: ...
    def balance_hash(self, value) -> None: ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address() -> Address: ...

# === Internal dependency: raiden.tests.utils.smartcontracts ===
def is_tx_hash_bytes(bytes_: Any) -> bool: ...

# === Internal dependency: raiden.utils.formatting ===
def to_hex_address(address: AddressTypes) -> AddressHex: ...

# === Internal dependency: raiden.utils.signer ===
class LocalSigner(Signer):
    def __init__(self, private_key: bytes) -> None: ...
    def sign(self, data: bytes, v: int = ...) -> Signature: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from eth_typing import BlockNumber
# re-export: from eth_typing import Hash32
# re-export: from raiden_contracts.utils.type_aliases import BalanceHash
# re-export: from raiden_contracts.utils.type_aliases import Locksroot
# re-export: from raiden_contracts.utils.type_aliases import Signature
# re-export: from raiden_contracts.utils.type_aliases import T_ChannelID
BlockHash = Hash32

# === Third-party dependency: raiden_contracts.constants ===
class MessageTypeId(IntEnum): ...
TEST_SETTLE_TIMEOUT_MIN: int
TEST_SETTLE_TIMEOUT_MAX: int