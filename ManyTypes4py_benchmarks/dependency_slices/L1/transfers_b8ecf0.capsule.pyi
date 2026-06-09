# === Third-party dependency: eth_hash.auto ===
keccak: Keccak256

# === Third-party dependency: eth_hash.main ===
class Keccak256: ...

# === Third-party dependency: eth_utils ===
# Used symbols: keccak

# === Internal dependency: raiden.constants ===
UINT256_MAX = 2 ** 256 - 1
UINT64_MAX = 2 ** 64 - 1
EMPTY_SIGNATURE = Signature(...)

# === Internal dependency: raiden.messages.abstract ===
class SignedRetrieableMessage(SignedMessage, RetrieableMessage):
    ...

# === Internal dependency: raiden.messages.cmdid ===
class CmdId(enum.Enum): ...

# === Internal dependency: raiden.messages.metadata ===
class Metadata: ...

# === Internal dependency: raiden.transfer.identifiers ===
class CanonicalIdentifier: ...

# === Internal dependency: raiden.transfer.utils ===
def hash_balance_data(transferred_amount, locked_amount, locksroot): ...

# === Internal dependency: raiden.utils.packing ===
def pack_balance_proof(nonce, balance_hash, additional_hash, canonical_identifier): ...

# === Internal dependency: raiden.utils.predicates ===
def ishash(data): ...

# === Internal dependency: raiden.utils.typing ===
from eth_typing import Address
from web3.types import Nonce
from raiden_contracts.utils.type_aliases import AdditionalHash
from raiden_contracts.utils.type_aliases import BlockExpiration
from raiden_contracts.utils.type_aliases import ChainID
from raiden_contracts.utils.type_aliases import ChannelID
from raiden_contracts.utils.type_aliases import Locksroot
from raiden_contracts.utils.type_aliases import Signature
from raiden_contracts.utils.type_aliases import TokenAmount
T_InitiatorAddress = bytes
InitiatorAddress = NewType(...)
T_PaymentID = int
PaymentID = NewType(...)
T_PaymentAmount = int
PaymentAmount = NewType(...)
T_FeeAmount = int
T_LockedAmount = int
LockedAmount = NewType(...)
PaymentWithFeeAmount = NewType(...)
T_TargetAddress = bytes
TargetAddress = NewType(...)
T_TokenAddress = bytes
TokenAddress = NewType(...)
T_TokenNetworkAddress = bytes
TokenNetworkAddress = NewType(...)
T_Secret = bytes
Secret = NewType(...)
T_SecretHash = bytes
SecretHash = NewType(...)