import random
import string
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import singledispatch
from hashlib import sha256
from operator import itemgetter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union, ClassVar, Type
from eth_utils import keccak
from raiden.constants import EMPTY_SIGNATURE, LOCKSROOT_OF_NO_LOCKS, UINT64_MAX, UINT256_MAX
from raiden.messages.decode import balanceproof_from_envelope
from raiden.messages.metadata import Metadata, RouteMetadata
from raiden.messages.transfers import Lock, LockedTransfer, LockExpired, RefundTransfer, Unlock
from raiden.transfer import channel, token_network, views
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state import HashTimeLockState, LockedTransferSignedState, LockedTransferUnsignedState, MediationPairState, TransferDescriptionWithSecretState
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitMediator
from raiden.transfer.state import BalanceProofSignedState, BalanceProofUnsignedState, ChainState, HopState, NettingChannelEndState, NettingChannelState, NetworkState, PendingLocksState, RouteState, SuccessfulTransactionState, TokenNetworkRegistryState, TokenNetworkState, TransactionExecutionStatus, message_identifier_from_prng
from raiden.transfer.state_change import ContractReceiveChannelNew, ContractReceiveRouteNew
from raiden.transfer.utils import hash_balance_data
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.packing import pack_balance_proof
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.signer import LocalSigner, Signer
from raiden.utils.transfers import random_secret
from raiden.utils.typing import AdditionalHash, Address, AddressHex, AddressMetadata, Balance, BlockExpiration, BlockHash, BlockNumber, BlockTimeout, ChainID, ChannelID, FeeAmount, InitiatorAddress, Locksroot, MessageID, MonitoringServiceAddress, Nonce, PaymentAmount, PaymentID, PrivateKey, Secret, SecretHash, Signature, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, TransactionHash, WithdrawAmount

EMPTY = 'empty'
GENERATE = 'generate'
K = TypeVar('K')
V = TypeVar('V')

def _partial_dict(full_dict: Dict[K, V], *args: K) -> Dict[K, V]:
    return {key: full_dict[key] for key in args}

class Properties:
    """
    Base class for all properties classes.
    """
    DEFAULTS: ClassVar[Any] = None
    TARGET_TYPE: ClassVar[Type[Any]] = None

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not EMPTY}

    def extract(self, subset_type: Type[Any]) -> Any:
        field_names = [field.name for field in fields(subset_type)]
        return subset_type(**_partial_dict(self.__dict__, *field_names))

    def partial_dict(self, *args: str) -> Dict[str, Any]:
        return _partial_dict(self.__dict__, *args)

def if_empty(value: Any, default: Any) -> Any:
    return value if value is not EMPTY else default

def _replace_properties(properties: Properties, defaults: Properties) -> Properties:
    replacements = {k: create_properties(v, defaults.__dict__[k]) if isinstance(v, Properties) else v for k, v in properties.kwargs.items()}
    return replace(defaults, **replacements)

def create_properties(properties: Properties, defaults: Optional[Properties] = None) -> Properties:
    full_defaults = deepcopy(type(properties).DEFAULTS)
    if defaults is not None:
        full_defaults = _replace_properties(defaults, full_defaults)
    return _replace_properties(properties, full_defaults)

def make_uint256() -> int:
    return random.randint(0, UINT256_MAX)

def make_channel_identifier() -> ChannelID:
    return ChannelID(make_uint256())

def make_uint64() -> int:
    return random.randint(0, UINT64_MAX)

def make_payment_id() -> PaymentID:
    return PaymentID(make_uint64())

def make_nonce() -> Nonce:
    return Nonce(make_uint64())

def make_token_amount() -> TokenAmount:
    return TokenAmount(random.randint(0, UINT256_MAX))

def make_withdraw_amount() -> WithdrawAmount:
    return WithdrawAmount(random.randint(0, UINT256_MAX))

def make_payment_amount() -> PaymentAmount:
    return PaymentAmount(random.randint(0, UINT256_MAX))

def make_balance() -> Balance:
    return Balance(random.randint(0, UINT256_MAX))

def make_block_number() -> BlockNumber:
    return BlockNumber(random.randint(0, UINT256_MAX))

def make_block_timeout() -> BlockTimeout:
    return BlockTimeout(random.randint(0, UINT256_MAX))

def make_block_expiration_number() -> BlockExpiration:
    return BlockExpiration(random.randint(0, UINT256_MAX))

def make_chain_id() -> ChainID:
    return ChainID(random.randint(0, UINT64_MAX))

def make_message_identifier() -> MessageID:
    return MessageID(random.randint(0, UINT64_MAX))

def make_bytes(length: int) -> bytes:
    return bytes(''.join((random.choice(string.printable) for _ in range(length))), encoding='utf-8')

def make_32bytes() -> bytes:
    return make_bytes(32)

def make_locksroot() -> Locksroot:
    return Locksroot(make_bytes(32))

def make_address() -> Address:
    return Address(make_bytes(20))

def make_monitoring_service_address() -> MonitoringServiceAddress:
    return MonitoringServiceAddress(make_bytes(20))

def make_initiator_address() -> InitiatorAddress:
    return InitiatorAddress(make_bytes(20))

def make_target_address() -> TargetAddress:
    return TargetAddress(make_bytes(20))

def make_checksum_address() -> str:
    return to_checksum_address(make_address())

def make_token_address() -> bytes:
    return make_bytes(20)

def make_token_network_address() -> TokenNetworkAddress:
    return TokenNetworkAddress(make_address())

def make_token_network_registry_address() -> TokenNetworkRegistryAddress:
    return TokenNetworkRegistryAddress(make_address())

def make_additional_hash() -> AdditionalHash:
    return AdditionalHash(make_bytes(32))

def make_transaction_hash() -> TransactionHash:
    return TransactionHash(make_bytes(32))

def make_block_hash() -> BlockHash:
    return BlockHash(make_bytes(32))

def make_privatekey_bin() -> bytes:
    return make_bytes(32)

def make_secret(i: Union[int, str] = EMPTY) -> Secret:
    if i is not EMPTY:
        return format(i, '>032').encode()
    else:
        return make_bytes(32)

def make_secret_hash(i: Union[int, str] = EMPTY) -> bytes:
    if i is not EMPTY:
        return sha256(format(i, '>032').encode()).digest()
    else:
        return make_bytes(32)

def make_secret_with_hash(i: Union[int, str] = EMPTY) -> Tuple[Secret, bytes]:
    secret = make_secret(i)
    secrethash = sha256_secrethash(secret)
    return (secret, secrethash)

def make_signature() -> Signature:
    return make_bytes(65)

def make_lock() -> HashTimeLockState:
    return HashTimeLockState(amount=random.randint(0, UINT256_MAX), expiration=random.randint(0, UINT64_MAX), secrethash=random_secret())

def make_privkey_address(privatekey: Union[bytes, str] = EMPTY) -> Tuple[bytes, Address]:
    privatekey = if_empty(privatekey, make_privatekey_bin())
    address = privatekey_to_address(privatekey)
    return (privatekey, address)

def make_privkeys_ordered(count: int, reverse: bool = False) -> List[bytes]:
    """Return ``count`` private keys ordered by their respective address"""
    key_address_pairs = [make_privkey_address() for _ in range(count)]
    return [key for key, _ in sorted(key_address_pairs, key=itemgetter(1), reverse=reverse)]

def make_signer() -> LocalSigner:
    privatekey = make_privatekey_bin()
    return LocalSigner(privatekey)

def make_hop_from_channel(channel_state: Union[NettingChannelState, str] = EMPTY) -> HopState:
    channel_state = if_empty(channel_state, create(NettingChannelStateProperties()))
    return HopState(channel_state.partner_state.address, channel_state.identifier)

def make_hop_to_channel(channel_state: Union[NettingChannelState, str] = EMPTY) -> HopState:
    channel_state = if_empty(channel_state, create(NettingChannelStateProperties()))
    return HopState(channel_state.our_state.address, channel_state.identifier)

UNIT_SETTLE_TIMEOUT = BlockTimeout(50)
UNIT_REVEAL_TIMEOUT = BlockTimeout(5)
UNIT_TRANSFER_AMOUNT = TokenAmount(50)
UNIT_TRANSFER_FEE = 2
UNIT_SECRET = Secret(b'secretsecretsecretsecretsecretse')
UNIT_SECRETHASH = sha256_secrethash(UNIT_SECRET)
UNIT_TOKEN_ADDRESS = TokenAddress(b'tokentokentokentoken')
UNIT_TOKEN_NETWORK_ADDRESS = TokenNetworkAddress(b'networknetworknetwor')
UNIT_CHANNEL_ID = ChannelID(1338)
UNIT_CHAIN_ID = ChainID(337)
UNIT_CANONICAL_ID = CanonicalIdentifier(chain_identifier=UNIT_CHAIN_ID, token_network_address=UNIT_TOKEN_NETWORK_ADDRESS, channel_identifier=UNIT_CHANNEL_ID)
UNIT_OUR_KEY = b'ourourourourourourourourourourou'
UNIT_OUR_ADDRESS = privatekey_to_address(UNIT_OUR_KEY)
UNIT_TOKEN_NETWORK_REGISTRY_ADDRESS = TokenNetworkRegistryAddress(b'tokennetworkregistryidentifier')
UNIT_TRANSFER_IDENTIFIER = 37
UNIT_TRANSFER_INITIATOR = Address(b'initiatorinitiatorin')
UNIT_TRANSFER_TARGET = Address(b'targettargettargetta')
UNIT_TRANSFER_PKEY_BIN = keccak(b'transfer pkey')
UNIT_TRANSFER_PKEY = UNIT_TRANSFER_PKEY_BIN
UNIT_TRANSFER_SENDER = Address(privatekey_to_address(UNIT_TRANSFER_PKEY))
HOP1_KEY = b'11111111111111111111111111111111'
HOP2_KEY = b'22222222222222222222222222222222'
HOP3_KEY = b'33333333333333333333333333333333'
HOP4_KEY = b'44444444444444444444444444444444'
HOP5_KEY = b'55555555555555555555555555555555'
HOP1 = InitiatorAddress(privatekey_to_address(HOP1_KEY))
HOP2 = Address(privatekey_to_address(HOP2_KEY))
HOP3 = Address(privatekey_to_address(HOP3_KEY))
ADDR = TargetAddress(b'addraddraddraddraddr')

def make_pending_locks(locks: List[Lock]) -> PendingLocksState:
    ret = PendingLocksState([])
    for lock in locks:
        ret.locks.append(bytes(lock.encoded))
    return ret

@singledispatch
def create(properties: Any, defaults: Optional[Any] = None) -> Any:
    """Create objects from their associated property class."""
    if isinstance(properties, Properties):
        return properties.TARGET_TYPE(**_properties_to_kwargs(properties, defaults))
    return properties

def _properties_to_kwargs(properties: Properties, defaults: Optional[Properties]) -> Dict[str, Any]:
    if not defaults:
        defaults = deepcopy(properties.DEFAULTS)
    properties = create_properties(properties, defaults)
    return {key: create(value) for key, value in properties.__dict__.items()}

@dataclass(frozen=True)
class CanonicalIdentifierProperties(Properties):
    chain_identifier: Union[ChainID, str] = EMPTY
    token_network_address: Union[TokenNetworkAddress, str] = EMPTY
    channel_identifier: Union[ChannelID, str] = EMPTY
    TARGET_TYPE = CanonicalIdentifier

CanonicalIdentifierProperties.DEFAULTS = CanonicalIdentifierProperties(chain_identifier=UNIT_CHAIN_ID, token_network_address=UNIT_TOKEN_NETWORK_ADDRESS, channel_identifier=GENERATE)

@create.register(CanonicalIdentifierProperties)
def _(properties: CanonicalIdentifierProperties, defaults: Optional[CanonicalIdentifierProperties] = None) -> CanonicalIdentifier:
    kwargs = _properties_to_kwargs(properties, defaults)
    if kwargs['channel_identifier'] == GENERATE:
        kwargs['channel_identifier'] = make_channel_identifier()
    return CanonicalIdentifier(**kwargs)

def make_canonical_identifier(chain_identifier: Union[ChainID, str] = EMPTY, token_network_address: Union[TokenNetworkAddress, str] = EMPTY, channel_identifier: Union[ChannelID, str] = EMPTY) -> CanonicalIdentifier:
    """Alias of the CanonicalIdentifier create function"""
    return create(CanonicalIdentifierProperties(chain_identifier=chain_identifier, token_network_address=token_network_address, channel_identifier=channel_identifier or make_channel_identifier()))

@dataclass(frozen=True)
class SuccessfulTransactionStateProperties(Properties):
    started_block_number: Union[BlockNumber, str] = EMPTY
    finished_block_number: Union[BlockNumber, str] = EMPTY

SuccessfulTransactionStateProperties.DEFAULTS = SuccessfulTransactionStateProperties(started_block_number=1, finished_block_number=1)

@create.register(SuccessfulTransactionStateProperties)
def _(properties: SuccessfulTransactionStateProperties, defaults: Optional[SuccessfulTransactionStateProperties] = None) -> SuccessfulTransactionState:
    kwargs = _properties_to_kwargs(properties, defaults)
    return SuccessfulTransactionState(**kwargs)

@dataclass(frozen=True)
class TransactionExecutionStatusProperties(Properties):
    started_block_number: Union[BlockNumber, str] = EMPTY
    finished_block_number: Union[BlockNumber, str] = EMPTY
    result: Union[str, str] = EMPTY
    TARGET_TYPE = TransactionExecutionStatus

TransactionExecutionStatusProperties.DEFAULTS = TransactionExecutionStatusProperties(started_block_number=None, finished_block_number=None, result=TransactionExecutionStatus.SUCCESS)

@dataclass(frozen=True)
class NettingChannelEndStateProperties(Properties):
    address: Union[Address, str] = EMPTY
    privatekey: Union[bytes, str] = EMPTY
    balance: Union[Balance, str] = EMPTY
    onchain_total_withdraw: Union[WithdrawAmount, str] = EMPTY
    pending_locks: Union[PendingLocksState, str] = EMPTY
    TARGET_TYPE = NettingChannelEndState

NettingChannelEndStateProperties.DEFAULTS = NettingChannelEndStateProperties(address=None, privatekey=None, balance=100, onchain_total_withdraw=0, pending_locks=None)
NettingChannelEndStateProperties.OUR_STATE = NettingChannelEndStateProperties(address=UNIT_OUR_ADDRESS, privatekey=UNIT_OUR_KEY, balance=100, onchain_total_withdraw=0, pending_locks=None)

@create.register(NettingChannelEndStateProperties)
def _(properties: NettingChannelEndStateProperties, defaults: Optional[NettingChannelEndStateProperties] = None) -> NettingChannelEndState:
    args = _properties_to_kwargs(properties, defaults or NettingChannelEndStateProperties.DEFAULTS)
    state = NettingChannelEndState(args['address'] or make_address(), args['balance'])
    pending_locks = args['pending_locks'] or None
    if pending_locks:
        state.pending_locks = pending_locks
    return state

@dataclass(frozen=True)
class RouteMetadataProperties(Properties):
    route: Union[List[Address], str] = EMPTY
    TARGET_TYPE = RouteMetadata

RouteMetadataProperties.DEFAULTS = RouteMetadataProperties(route=[HOP1, HOP2])

@dataclass(frozen=True)
class MetadataProperties(Properties):
    routes: Union[List[RouteMetadata], str] = EMPTY
    _original_data: Union[Dict, str] = EMPTY
    TARGET_TYPE = Metadata

MetadataProperties.DEFAULTS = MetadataProperties(routes=[RouteMetadata(route=[HOP1, HOP2])], _original_data=None)

@dataclass(frozen=True)
class FeeScheduleStateProperties(Properties):
    flat: Union[FeeAmount, str] = EMPTY
    proportional: Union[FeeAmount, str] = EMPTY
    TARGET_TYPE = FeeScheduleState

FeeScheduleStateProperties.DEFAULTS = FeeScheduleStateProperties(flat=0, proportional=0)

@dataclass(frozen=True)
class NettingChannelStateProperties(Properties):
    canonical_identifier: Union[CanonicalIdentifier, str] = EMPTY
    token_address: Union[TokenAddress, str] = EMPTY
    token_network_registry_address: Union[TokenNetworkRegistryAddress, str] = EMPTY
    reveal_timeout: Union[BlockTimeout, str] = EMPTY
    settle_timeout: Union[BlockTimeout, str] = EMPTY
    fee_schedule: Union[FeeScheduleState, str] = EMPTY
    our_state: Union[NettingChannelEndState, str] = EMPTY
    partner_state: Union[NettingChannelEndState, str] = EMPTY
    open_transaction: Union[SuccessfulTransactionState, str] = EMPTY
    close_transaction: Union[SuccessfulTransactionState, str] = EMPTY
    settle_transaction: Union[SuccessfulTransactionState, str] = EMPTY
    TARGET_TYPE = NettingChannelState

NettingChannelStateProperties.DEFAULTS = NettingChannelStateProperties(canonical_identifier=CanonicalIdentifierProperties.DEFAULTS