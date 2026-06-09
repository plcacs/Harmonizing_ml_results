from typing import Any

# === Third-party dependency: eth_hash.auto ===
keccak: Keccak256

# === Third-party dependency: eth_hash.main ===
class Keccak256: ...

# === Third-party dependency: eth_utils ===
# Used symbols: keccak

# === Internal dependency: raiden.constants ===
UINT256_MAX: Any
UINT64_MAX: Any
EMPTY_SIGNATURE: Signature

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
def hash_balance_data(transferred_amount: TokenAmount, locked_amount: LockedAmount, locksroot: Locksroot) -> BalanceHash: ...

# === Internal dependency: raiden.utils.packing ===
def pack_balance_proof(nonce: Nonce, balance_hash: BalanceHash, additional_hash: AdditionalHash, canonical_identifier: CanonicalIdentifier) -> bytes: ...

# === Internal dependency: raiden.utils.predicates ===
def ishash(data: bytes) -> bool: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from eth_typing import Address
# re-export: from web3.types import Nonce
# re-export: from raiden_contracts.utils.type_aliases import AdditionalHash
# re-export: from raiden_contracts.utils.type_aliases import BlockExpiration
# re-export: from raiden_contracts.utils.type_aliases import ChainID
# re-export: from raiden_contracts.utils.type_aliases import ChannelID
# re-export: from raiden_contracts.utils.type_aliases import Locksroot
# re-export: from raiden_contracts.utils.type_aliases import Signature
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
InitiatorAddress: NewType
PaymentID: NewType
PaymentAmount: NewType
LockedAmount: NewType
PaymentWithFeeAmount: NewType
TargetAddress: NewType
TokenAddress: NewType
TokenNetworkAddress: NewType
Secret: NewType
SecretHash: NewType