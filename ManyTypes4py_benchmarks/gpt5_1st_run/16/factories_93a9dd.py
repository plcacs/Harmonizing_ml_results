import random
import string
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import singledispatch
from hashlib import sha256
from operator import itemgetter
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
from raiden.utils.typing import AdditionalHash, Address, AddressHex, AddressMetadata, Any, Balance, BlockExpiration, BlockHash, BlockNumber, BlockTimeout, ChainID, ChannelID, ClassVar, Dict, FeeAmount, InitiatorAddress, List, Locksroot, MessageID, MonitoringServiceAddress, NamedTuple, NodeNetworkStateMap, Nonce, Optional, PaymentAmount, PaymentID, PrivateKey, Secret, SecretHash, Signature, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, TransactionHash, Tuple, Type, TypeVar, WithdrawAmount
EMPTY: str = 'empty'
GENERATE: str = 'generate'
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')

def _partial_dict(full_dict: Dict[str, V], *args: str) -> Dict[str, V]:
    return {key: full_dict[key] for key in args}

class Properties:
    """
    Base class for all properties classes.

    Each properties class is a frozen dataclass used for creating a
    specific type of object. It is called `TProperties`, where `T`
    is the type of the object to be created, which is also specified
    in the class variable `TARGET_TYPE`. An object of type `T` is
    created by passing a `TProperties` instance to `create`.

    When subclassing `Properties`, all fields should be given `EMPTY`
    as a default value. The class variable `DEFAULTS` should be set to
    a fully initialized instance of `TProperties`.

    The advantage of this is that we can change defaults later: If
    some test module needs many slightly varied instances of the same
    object, it can define its own defaults instance and use it like
    this:
    ```
    my_defaults = create_properties(custom_defaults)
    # my_defaults is now a fully initialized version of
    # custom_defaults, no EMPTY fields
    ...
    object1 = create(properties1, my_defaults)
    object2 = create(properties2, my_defaults)
    ...
    ```

    Fields in the default instance can be set to `GENERATE` to indicate
    they should be generated (either randomly, like a secret, or from
    the other fields, like a signature).
    """
    DEFAULTS: ClassVar[Any] = None
    TARGET_TYPE: ClassVar[Optional[Type[Any]]] = None

    @property
    def kwargs(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not EMPTY}

    def extract(self, subset_type: Type[T]) -> T:
        field_names = [field.name for field in fields(subset_type)]
        return subset_type(**_partial_dict(self.__dict__, *field_names))  # type: ignore

    def partial_dict(self, *args: str) -> Dict[str, Any]:
        return _partial_dict(self.__dict__, *args)

def if_empty(value: T, default: T) -> T:
    return value if value is not EMPTY else default

def _replace_properties(properties: Properties, defaults: Properties) -> Any:
    replacements = {k: create_properties(v, defaults.__dict__[k]) if isinstance(v, Properties) else v for k, v in properties.kwargs.items()}
    return replace(defaults, **replacements)

def create_properties(properties: Properties, defaults: Optional[Properties] = None) -> Any:
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

def make_checksum_address() -> AddressHex:
    return to_checksum_address(make_address())

def make_token_address() -> TokenAddress:
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

def make_privatekey_bin() -> PrivateKey:
    return make_bytes(32)

def make_secret(i: Any = EMPTY) -> Secret:
    if i is not EMPTY:
        return format(i, '>032').encode()
    else:
        return make_bytes(32)

def make_secret_hash(i: Any = EMPTY) -> SecretHash:
    if i is not EMPTY:
        return sha256(format(i, '>032').encode()).digest()
    else:
        return make_bytes(32)

def make_secret_with_hash(i: Any = EMPTY) -> Tuple[Secret, SecretHash]:
    secret = make_secret(i)
    secrethash = sha256_secrethash(secret)
    return (secret, secrethash)

def make_signature() -> Signature:
    return make_bytes(65)

def make_lock() -> HashTimeLockState:
    return HashTimeLockState(amount=random.randint(0, UINT256_MAX), expiration=random.randint(0, UINT64_MAX), secrethash=random_secret())

def make_privkey_address(privatekey: Any = EMPTY) -> Tuple[PrivateKey, Address]:
    privatekey = if_empty(privatekey, make_privatekey_bin())
    address = privatekey_to_address(privatekey)
    return (privatekey, address)

def make_privkeys_ordered(count: int, reverse: bool = False) -> List[PrivateKey]:
    """Return ``count`` private keys ordered by their respective address"""
    key_address_pairs = [make_privkey_address() for _ in range(count)]
    return [key for key, _ in sorted(key_address_pairs, key=itemgetter(1), reverse=reverse)]

def make_signer() -> Signer:
    privatekey = make_privatekey_bin()
    return LocalSigner(privatekey)

def make_hop_from_channel(channel_state: Any = EMPTY) -> HopState:
    channel_state = if_empty(channel_state, create(NettingChannelStateProperties()))
    return HopState(channel_state.partner_state.address, channel_state.identifier)

def make_hop_to_channel(channel_state: Any = EMPTY) -> HopState:
    channel_state = if_empty(channel_state, create(NettingChannelStateProperties()))
    return HopState(channel_state.our_state.address, channel_state.identifier)
UNIT_SETTLE_TIMEOUT: BlockTimeout = BlockTimeout(50)
UNIT_REVEAL_TIMEOUT: BlockTimeout = BlockTimeout(5)
UNIT_TRANSFER_AMOUNT: TokenAmount = TokenAmount(50)
UNIT_TRANSFER_FEE: int = 2
UNIT_SECRET: Secret = Secret(b'secretsecretsecretsecretsecretse')
UNIT_SECRETHASH: SecretHash = sha256_secrethash(UNIT_SECRET)
UNIT_TOKEN_ADDRESS: TokenAddress = TokenAddress(b'tokentokentokentoken')
UNIT_TOKEN_NETWORK_ADDRESS: TokenNetworkAddress = TokenNetworkAddress(b'networknetworknetwor')
UNIT_CHANNEL_ID: ChannelID = ChannelID(1338)
UNIT_CHAIN_ID: ChainID = ChainID(337)
UNIT_CANONICAL_ID: CanonicalIdentifier = CanonicalIdentifier(chain_identifier=UNIT_CHAIN_ID, token_network_address=UNIT_TOKEN_NETWORK_ADDRESS, channel_identifier=UNIT_CHANNEL_ID)
UNIT_OUR_KEY: PrivateKey = b'ourourourourourourourourourourou'
UNIT_OUR_ADDRESS: Address = privatekey_to_address(UNIT_OUR_KEY)
UNIT_TOKEN_NETWORK_REGISTRY_ADDRESS: TokenNetworkRegistryAddress = TokenNetworkRegistryAddress(b'tokennetworkregistryidentifier')
UNIT_TRANSFER_IDENTIFIER: int = 37
UNIT_TRANSFER_INITIATOR: Address = Address(b'initiatorinitiatorin')
UNIT_TRANSFER_TARGET: Address = Address(b'targettargettargetta')
UNIT_TRANSFER_PKEY_BIN: PrivateKey = keccak(b'transfer pkey')
UNIT_TRANSFER_PKEY: PrivateKey = UNIT_TRANSFER_PKEY_BIN
UNIT_TRANSFER_SENDER: Address = Address(privatekey_to_address(UNIT_TRANSFER_PKEY))
HOP1_KEY: PrivateKey = b'11111111111111111111111111111111'
HOP2_KEY: PrivateKey = b'22222222222222222222222222222222'
HOP3_KEY: PrivateKey = b'33333333333333333333333333333333'
HOP4_KEY: PrivateKey = b'44444444444444444444444444444444'
HOP5_KEY: PrivateKey = b'55555555555555555555555555555555'
HOP1: InitiatorAddress = InitiatorAddress(privatekey_to_address(HOP1_KEY))
HOP2: Address = Address(privatekey_to_address(HOP2_KEY))
HOP3: Address = Address(privatekey_to_address(HOP3_KEY))
ADDR: TargetAddress = TargetAddress(b'addraddraddraddraddr')

def make_pending_locks(locks: List[HashTimeLockState]) -> PendingLocksState:
    ret = PendingLocksState([])
    for lock in locks:
        ret.locks.append(bytes(lock.encoded))
    return ret

@singledispatch
def create(properties: Any, defaults: Any = None) -> Any:
    """Create objects from their associated property class.

    E. g. a NettingChannelState from NettingChannelStateProperties. For any field in
    properties set to EMPTY a default will be used. The default values can be changed
    by giving another object of the same property type as the defaults argument.
    """
    if isinstance(properties, Properties):
        return properties.TARGET_TYPE(**_properties_to_kwargs(properties, defaults))  # type: ignore
    return properties

def _properties_to_kwargs(properties: Properties, defaults: Optional[Properties]) -> Dict[str, Any]:
    if not defaults:
        defaults = deepcopy(properties.DEFAULTS)
    properties = create_properties(properties, defaults)
    return {key: create(value) for key, value in properties.__dict__.items()}

@dataclass(frozen=True)
class CanonicalIdentifierProperties(Properties):
    chain_identifier: Any = EMPTY
    token_network_address: Any = EMPTY
    channel_identifier: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[CanonicalIdentifier]] = CanonicalIdentifier
CanonicalIdentifierProperties.DEFAULTS = CanonicalIdentifierProperties(chain_identifier=UNIT_CHAIN_ID, token_network_address=UNIT_TOKEN_NETWORK_ADDRESS, channel_identifier=GENERATE)

@create.register(CanonicalIdentifierProperties)
def _(properties: CanonicalIdentifierProperties, defaults: Optional[CanonicalIdentifierProperties] = None) -> CanonicalIdentifier:
    kwargs = _properties_to_kwargs(properties, defaults)
    if kwargs['channel_identifier'] == GENERATE:
        kwargs['channel_identifier'] = make_channel_identifier()
    return CanonicalIdentifier(**kwargs)

def make_canonical_identifier(chain_identifier: Any = EMPTY, token_network_address: Any = EMPTY, channel_identifier: Any = EMPTY) -> CanonicalIdentifier:
    """Alias of the CanonicalIdentifier create function"""
    return create(CanonicalIdentifierProperties(chain_identifier=chain_identifier, token_network_address=token_network_address, channel_identifier=channel_identifier or make_channel_identifier()))

@dataclass(frozen=True)
class SuccessfulTransactionStateProperties(Properties):
    started_block_number: Any = EMPTY
    finished_block_number: Any = EMPTY
SuccessfulTransactionStateProperties.DEFAULTS = SuccessfulTransactionStateProperties(started_block_number=1, finished_block_number=1)

@create.register(SuccessfulTransactionStateProperties)
def _(properties: SuccessfulTransactionStateProperties, defaults: Optional[SuccessfulTransactionStateProperties] = None) -> SuccessfulTransactionState:
    kwargs = _properties_to_kwargs(properties, defaults)
    return SuccessfulTransactionState(**kwargs)

@dataclass(frozen=True)
class TransactionExecutionStatusProperties(Properties):
    started_block_number: Any = EMPTY
    finished_block_number: Any = EMPTY
    result: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[TransactionExecutionStatus]] = TransactionExecutionStatus
TransactionExecutionStatusProperties.DEFAULTS = TransactionExecutionStatusProperties(started_block_number=None, finished_block_number=None, result=TransactionExecutionStatus.SUCCESS)

@dataclass(frozen=True)
class NettingChannelEndStateProperties(Properties):
    address: Any = EMPTY
    privatekey: Any = EMPTY
    balance: Any = EMPTY
    onchain_total_withdraw: Any = EMPTY
    pending_locks: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[NettingChannelEndState]] = NettingChannelEndState
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
    route: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[RouteMetadata]] = RouteMetadata
RouteMetadataProperties.DEFAULTS = RouteMetadataProperties(route=[HOP1, HOP2])

@dataclass(frozen=True)
class MetadataProperties(Properties):
    routes: Any = EMPTY
    _original_data: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[Metadata]] = Metadata
MetadataProperties.DEFAULTS = MetadataProperties(routes=[RouteMetadata(route=[HOP1, HOP2])], _original_data=None)

@dataclass(frozen=True)
class FeeScheduleStateProperties(Properties):
    flat: Any = EMPTY
    proportional: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[FeeScheduleState]] = FeeScheduleState
FeeScheduleStateProperties.DEFAULTS = FeeScheduleStateProperties(flat=0, proportional=0)

@dataclass(frozen=True)
class NettingChannelStateProperties(Properties):
    canonical_identifier: Any = EMPTY
    token_address: Any = EMPTY
    token_network_registry_address: Any = EMPTY
    reveal_timeout: Any = EMPTY
    settle_timeout: Any = EMPTY
    fee_schedule: Any = EMPTY
    our_state: Any = EMPTY
    partner_state: Any = EMPTY
    open_transaction: Any = EMPTY
    close_transaction: Any = EMPTY
    settle_transaction: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[NettingChannelState]] = NettingChannelState
NettingChannelStateProperties.DEFAULTS = NettingChannelStateProperties(canonical_identifier=CanonicalIdentifierProperties.DEFAULTS, token_address=UNIT_TOKEN_ADDRESS, token_network_registry_address=UNIT_TOKEN_NETWORK_REGISTRY_ADDRESS, reveal_timeout=UNIT_REVEAL_TIMEOUT, settle_timeout=UNIT_SETTLE_TIMEOUT, fee_schedule=FeeScheduleStateProperties.DEFAULTS, our_state=NettingChannelEndStateProperties.OUR_STATE, partner_state=NettingChannelEndStateProperties.DEFAULTS, open_transaction=SuccessfulTransactionStateProperties.DEFAULTS, close_transaction=None, settle_transaction=None)

@dataclass(frozen=True)
class TransferDescriptionProperties(Properties):
    token_network_registry_address: Any = EMPTY
    payment_identifier: Any = EMPTY
    amount: Any = EMPTY
    token_network_address: Any = EMPTY
    initiator: Any = EMPTY
    target: Any = EMPTY
    secret: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[TransferDescriptionWithSecretState]] = TransferDescriptionWithSecretState
TransferDescriptionProperties.DEFAULTS = TransferDescriptionProperties(token_network_registry_address=UNIT_TOKEN_NETWORK_REGISTRY_ADDRESS, payment_identifier=UNIT_TRANSFER_IDENTIFIER, amount=UNIT_TRANSFER_AMOUNT, token_network_address=UNIT_TOKEN_NETWORK_ADDRESS, initiator=UNIT_TRANSFER_INITIATOR, target=UNIT_TRANSFER_TARGET, secret=GENERATE)

@create.register(TransferDescriptionProperties)
def _(properties: TransferDescriptionProperties, defaults: Optional[TransferDescriptionProperties] = None) -> TransferDescriptionWithSecretState:
    properties = create_properties(properties, defaults)
    params = {key: value for key, value in properties.__dict__.items()}
    if params['secret'] == GENERATE:
        params['secret'] = random_secret()
    return TransferDescriptionWithSecretState(**params)
UNIT_TRANSFER_DESCRIPTION: TransferDescriptionWithSecretState = create(TransferDescriptionProperties(secret=UNIT_SECRET))

@dataclass(frozen=True)
class BalanceProofProperties(Properties):
    nonce: Any = EMPTY
    transferred_amount: Any = EMPTY
    locked_amount: Any = EMPTY
    locksroot: Any = EMPTY
    canonical_identifier: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[BalanceProofUnsignedState]] = BalanceProofUnsignedState

    @property
    def balance_proof(self) -> 'BalanceProofProperties':
        """Convenience method to extract balance proof properties from the child classes."""
        return self.extract(BalanceProofProperties)
BalanceProofProperties.DEFAULTS = BalanceProofProperties(nonce=1, transferred_amount=UNIT_TRANSFER_AMOUNT, locked_amount=0, locksroot=LOCKSROOT_OF_NO_LOCKS, canonical_identifier=UNIT_CANONICAL_ID)

@dataclass(frozen=True)
class UnlockProperties(BalanceProofProperties):
    message_identifier: Any = EMPTY
    payment_identifier: Any = EMPTY
    secret: Any = EMPTY
    signature: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[Unlock]] = Unlock
UnlockProperties.DEFAULTS = UnlockProperties(**BalanceProofProperties.DEFAULTS.__dict__, message_identifier=1, payment_identifier=1, secret=UNIT_SECRET, signature=EMPTY_SIGNATURE)

def unwrap_canonical_identifier(params: Dict[str, Any]) -> Dict[str, Any]:
    params_copy = dict(params)
    canonical_identifier = params_copy.pop('canonical_identifier')
    params_copy['chain_id'] = canonical_identifier.chain_identifier
    params_copy['token_network_address'] = canonical_identifier.token_network_address
    params_copy['channel_identifier'] = canonical_identifier.channel_identifier
    return params_copy

@create.register(UnlockProperties)
def _(properties: UnlockProperties, defaults: Optional[UnlockProperties] = None) -> Unlock:
    properties = create_properties(properties, defaults)
    return Unlock(**unwrap_canonical_identifier(properties.__dict__))

@dataclass(frozen=True)
class LockExpiredProperties(BalanceProofProperties):
    recipient: Any = EMPTY
    secrethash: Any = EMPTY
    message_identifier: Any = EMPTY
    signature: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[LockExpired]] = LockExpired
LockExpiredProperties.DEFAULTS = LockExpiredProperties(**BalanceProofProperties.DEFAULTS.__dict__, recipient=UNIT_TRANSFER_TARGET, secrethash=UNIT_SECRETHASH, message_identifier=1, signature=EMPTY_SIGNATURE)

@create.register(LockExpiredProperties)
def _(properties: LockExpiredProperties, defaults: Optional[LockExpiredProperties] = None) -> LockExpired:
    properties = create_properties(properties, defaults)
    return LockExpired(**unwrap_canonical_identifier(properties.__dict__))

@dataclass(frozen=True)
class BalanceProofSignedStateProperties(BalanceProofProperties):
    message_hash: Any = EMPTY
    signature: Any = GENERATE
    sender: Any = EMPTY
    pkey: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[BalanceProofSignedState]] = BalanceProofSignedState
BalanceProofSignedStateProperties.DEFAULTS = BalanceProofSignedStateProperties(**BalanceProofProperties.DEFAULTS.__dict__, message_hash=UNIT_SECRETHASH, sender=UNIT_TRANSFER_SENDER, pkey=UNIT_TRANSFER_PKEY)

def make_signed_balance_proof_from_unsigned(unsigned: BalanceProofUnsignedState, signer: Signer, additional_hash: Optional[AdditionalHash] = None) -> BalanceProofSignedState:
    balance_hash = hash_balance_data(transferred_amount=unsigned.transferred_amount, locked_amount=unsigned.locked_amount, locksroot=unsigned.locksroot)
    if additional_hash is None:
        additional_hash = make_additional_hash()
    data_to_sign = pack_balance_proof(balance_hash=balance_hash, additional_hash=additional_hash, canonical_identifier=unsigned.canonical_identifier, nonce=unsigned.nonce)
    signature = signer.sign(data=data_to_sign)
    sender = signer.address
    return BalanceProofSignedState(nonce=unsigned.nonce, transferred_amount=unsigned.transferred_amount, locked_amount=unsigned.locked_amount, locksroot=unsigned.locksroot, message_hash=additional_hash, signature=signature, sender=sender, canonical_identifier=unsigned.canonical_identifier)

@create.register(BalanceProofSignedStateProperties)
def _(properties: BalanceProofSignedStateProperties, defaults: Optional[BalanceProofSignedStateProperties] = None) -> BalanceProofSignedState:
    defaults = defaults or BalanceProofSignedStateProperties.DEFAULTS
    params = create_properties(properties, defaults).__dict__
    signer = LocalSigner(params.pop('pkey'))
    if params['signature'] is GENERATE:
        keys = ('transferred_amount', 'locked_amount', 'locksroot')
        balance_hash = hash_balance_data(**_partial_dict(params, *keys))
        data_to_sign = pack_balance_proof(balance_hash=balance_hash, additional_hash=params['message_hash'], canonical_identifier=params['canonical_identifier'], nonce=params.get('nonce'))
        params['signature'] = signer.sign(data=data_to_sign)
    return BalanceProofSignedState(**params)

@dataclass(frozen=True)
class LockedTransferUnsignedStateProperties(BalanceProofProperties):
    amount: Any = EMPTY
    expiration: Any = EMPTY
    initiator: Any = EMPTY
    target: Any = EMPTY
    payment_identifier: Any = EMPTY
    token: Any = EMPTY
    secret: Any = EMPTY
    route_states: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[LockedTransferUnsignedState]] = LockedTransferUnsignedState
LockedTransferUnsignedStateProperties.DEFAULTS = LockedTransferUnsignedStateProperties(**create_properties(BalanceProofProperties(locked_amount=UNIT_TRANSFER_AMOUNT, transferred_amount=0)).__dict__, amount=UNIT_TRANSFER_AMOUNT, expiration=UNIT_REVEAL_TIMEOUT, initiator=UNIT_TRANSFER_INITIATOR, target=UNIT_TRANSFER_TARGET, payment_identifier=1, token=UNIT_TOKEN_ADDRESS, secret=UNIT_SECRET)

@create.register(LockedTransferUnsignedStateProperties)
def _(properties: LockedTransferUnsignedStateProperties, defaults: Optional[LockedTransferUnsignedStateProperties] = None) -> LockedTransferUnsignedState:
    transfer = create_properties(properties, defaults)
    lock = HashTimeLockState(amount=transfer.amount, expiration=transfer.expiration, secrethash=sha256(transfer.secret).digest())
    if transfer.locksroot == LOCKSROOT_OF_NO_LOCKS:
        transfer = replace(transfer, locksroot=keccak(lock.encoded))
    balance_proof_properties = transfer.extract(BalanceProofProperties)
    if properties.transferred_amount == EMPTY:
        balance_proof_properties = replace(balance_proof_properties, transferred_amount=0)
    if properties.locked_amount == EMPTY:
        balance_proof_properties = replace(balance_proof_properties, locked_amount=transfer.amount)
    balance_proof = create(balance_proof_properties)
    netting_channel_state = create(NettingChannelStateProperties(canonical_identifier=balance_proof.canonical_identifier))
    route_states = create_route_states_from_routes(routes=[[netting_channel_state.partner_state.address, transfer.target]])
    return LockedTransferUnsignedState(balance_proof=balance_proof, lock=lock, route_states=route_states, **transfer.partial_dict('initiator', 'target', 'payment_identifier', 'token'))

@dataclass(frozen=True)
class LockedTransferSignedStateProperties(BalanceProofProperties):
    amount: Any = EMPTY
    expiration: Any = EMPTY
    initiator: Any = EMPTY
    target: Any = EMPTY
    payment_identifier: Any = EMPTY
    token: Any = EMPTY
    secret: Any = EMPTY
    sender: Any = EMPTY
    recipient: Any = EMPTY
    pkey: Any = EMPTY
    message_identifier: Any = EMPTY
    route_states: Any = GENERATE
    TARGET_TYPE: ClassVar[Type[LockedTransferSignedState]] = LockedTransferSignedState
LOCKED_TRANSFER_BASE_DEFAULTS = LockedTransferUnsignedStateProperties.DEFAULTS.__dict__.copy()
LockedTransferSignedStateProperties.DEFAULTS = LockedTransferSignedStateProperties(**LOCKED_TRANSFER_BASE_DEFAULTS, sender=UNIT_TRANSFER_SENDER, recipient=UNIT_TRANSFER_TARGET, pkey=UNIT_TRANSFER_PKEY, message_identifier=1)

@create.register(LockedTransferSignedStateProperties)
def _(properties: LockedTransferSignedStateProperties, defaults: Optional[LockedTransferSignedStateProperties] = None) -> LockedTransferSignedState:
    transfer = create_properties(properties, defaults)
    params = unwrap_canonical_identifier(transfer.__dict__)
    lock = Lock(amount=params.pop('amount'), expiration=params.pop('expiration'), secrethash=sha256(params.pop('secret')).digest())
    pkey = params.pop('pkey')
    signer = LocalSigner(pkey)
    sender = params.pop('sender')
    if params['locksroot'] == LOCKSROOT_OF_NO_LOCKS:
        params['locksroot'] = keccak(lock.as_bytes)
    route_states = params.pop('route_states')
    if route_states == GENERATE:
        route_states = create_route_states_from_routes([list(dict.fromkeys([Address(transfer.initiator), Address(transfer.sender), Address(transfer.recipient), Address(transfer.target)]))])
    routes = [RouteMetadata(route=route_state.route, address_metadata=route_state.address_to_metadata) for route_state in route_states]
    original_metadata = Metadata(routes=routes).to_dict()
    metadata = Metadata(routes=routes, _original_data=original_metadata)
    params['metadata'] = metadata
    locked_transfer = LockedTransfer(lock=lock, **params, signature=EMPTY_SIGNATURE)
    if properties.locked_amount == EMPTY:
        locked_transfer.locked_amount = transfer.amount
    if properties.transferred_amount == EMPTY:
        locked_transfer.transferred_amount = 0
    locked_transfer.sign(signer)
    assert locked_transfer.metadata
    assert locked_transfer.sender == sender
    balance_proof = balanceproof_from_envelope(locked_transfer)
    lock = HashTimeLockState(locked_transfer.lock.amount, locked_transfer.lock.expiration, locked_transfer.lock.secrethash)
    return LockedTransferSignedState(message_identifier=locked_transfer.message_identifier, payment_identifier=locked_transfer.payment_identifier, token=locked_transfer.token, balance_proof=balance_proof, lock=lock, initiator=locked_transfer.initiator, target=locked_transfer.target, route_states=route_states, metadata=original_metadata)

@dataclass(frozen=True)
class LockedTransferProperties(LockedTransferSignedStateProperties):
    metadata: Any = EMPTY
    TARGET_TYPE: ClassVar[Type[LockedTransfer]] = LockedTransfer
LockedTransferProperties.DEFAULTS = LockedTransferProperties(**replace(LockedTransferSignedStateProperties.DEFAULTS, locksroot=GENERATE).__dict__, metadata=GENERATE)

def prepare_locked_transfer(properties: LockedTransferProperties, defaults: Optional[LockedTransferProperties]) -> Tuple[Dict[str, Any], LocalSigner, Address]:
    properties = create_properties(properties, defaults)
    params = unwrap_canonical_identifier(properties.__dict__)
    secrethash = sha256(params.pop('secret')).digest()
    params['lock'] = Lock(amount=params.pop('amount'), expiration=params.pop('expiration'), secrethash=secrethash)
    if params['locksroot'] == GENERATE:
        params['locksroot'] = keccak(params['lock'].as_bytes)
    params['signature'] = EMPTY_SIGNATURE
    params.pop('route_states')
    if params['metadata'] == GENERATE:
        params['metadata'] = create(MetadataProperties())
    return (params, LocalSigner(params.pop('pkey')), params.pop('sender'))

@create.register(LockedTransferProperties)
def _(properties: LockedTransferProperties, defaults: Optional[LockedTransferProperties] = None) -> LockedTransfer:
    params, signer, expected_sender = prepare_locked_transfer(properties, defaults)
    transfer = LockedTransfer(**params)
    transfer.sign(signer)
    assert transfer.sender == expected_sender
    return transfer

@dataclass(frozen=True)
class RefundTransferProperties(LockedTransferProperties):
    TARGET_TYPE: ClassVar[Type[RefundTransfer]] = RefundTransfer
RefundTransferProperties.DEFAULTS = RefundTransferProperties(**LockedTransferProperties.DEFAULTS.__dict__)

@create.register(RefundTransferProperties)
def _(properties: RefundTransferProperties, defaults: Optional[RefundTransferProperties] = None) -> RefundTransfer:
    params, signer, expected_sender = prepare_locked_transfer(properties, defaults)
    transfer = RefundTransfer(**params)
    transfer.sign(signer)
    assert transfer.sender == expected_sender
    return transfer
SIGNED_TRANSFER_FOR_CHANNEL_DEFAULTS: LockedTransferSignedStateProperties = create_properties(LockedTransferSignedStateProperties(expiration=UNIT_SETTLE_TIMEOUT - UNIT_REVEAL_TIMEOUT))

def create_route_states_from_routes(routes: List[List[Address]], address_to_address_metadata: Optional[Dict[Address, AddressMetadata]] = None, mock_missing_metadata: bool = False) -> List[RouteState]:
    route_states: List[RouteState] = []
    for route in routes:
        address_metadata: Dict[Address, AddressMetadata] = {}
        if address_to_address_metadata:
            for address in route:
                metadata = address_to_address_metadata.get(address)
                if metadata is not None:
                    address_metadata[address] = metadata
        route_states.append(RouteState(route=route, address_to_metadata=address_metadata))
    return route_states

def make_signed_transfer_for(channel_state: Any = EMPTY, properties: Optional[LockedTransferSignedStateProperties] = None, defaults: Optional[LockedTransferSignedStateProperties] = None, calculate_locksroot: bool = False, allow_invalid: bool = False, only_transfer: bool = True) -> LockedTransferSignedState:
    properties = create_properties(properties or LockedTransferSignedStateProperties(), defaults or SIGNED_TRANSFER_FOR_CHANNEL_DEFAULTS)
    channel_state = if_empty(channel_state, create(NettingChannelStateProperties()))
    if not allow_invalid:
        ok = channel_state.reveal_timeout < properties.expiration < channel_state.settle_timeout
        assert ok, 'Expiration must be between reveal_timeout and settle_timeout.'
    assert privatekey_to_address(properties.pkey) == properties.sender
    if properties.sender == channel_state.our_state.address:
        recipient = channel_state.partner_state.address
    elif properties.sender == channel_state.partner_state.address:
        recipient = channel_state.our_state.address
    else:
        raise RuntimeError('Given sender does not participate in given channel.')
    if calculate_locksroot:
        lock = HashTimeLockState(amount=properties.amount, expiration=properties.expiration, secrethash=sha256(properties.secret).digest())
        locksroot = compute_locksroot(channel.compute_locks_with(locks=channel_state.partner_state.pending_locks, lock=lock))
    else:
        locksroot = properties.locksroot
    if only_transfer:
        transfer_properties = LockedTransferUnsignedStateProperties(locksroot=locksroot, canonical_identifier=channel_state.canonical_identifier, locked_amount=properties.amount, transferred_amount=0)
    else:
        transfer_properties = LockedTransferUnsignedStateProperties(locksroot=locksroot, canonical_identifier=channel_state.canonical_identifier, locked_amount=properties.locked_amount, transferred_amount=properties.transferred_amount)
    transfer = create(LockedTransferSignedStateProperties(recipient=recipient, **transfer_properties.__dict__), defaults=properties)
    if not allow_invalid:
        is_valid, msg, _ = channel.is_valid_lockedtransfer(transfer_state=transfer, channel_state=channel_state, sender_state=channel_state.partner_state, receiver_state=channel_state.our_state)
        assert is_valid, msg
    return transfer

def pkeys_from_channel_state(properties: NettingChannelStateProperties, defaults: NettingChannelStateProperties = NettingChannelStateProperties.DEFAULTS) -> Tuple[Optional[PrivateKey], Optional[PrivateKey]]:
    our_key: Optional[PrivateKey] = None
    if properties.our_state is not EMPTY:
        our_key = properties.our_state.privatekey
    elif defaults is not None:
        our_key = defaults.our_state.privatekey
    partner_key: Optional[PrivateKey] = None
    if properties.partner_state is not EMPTY:
        partner_key = properties.partner_state.privatekey
    elif defaults is not None:
        partner_key = defaults.partner_state.privatekey
    return (our_key, partner_key)

class ChannelSet:
    """Manage a list of channels. The channels can be accessed by subscript."""
    HOP3_KEY: ClassVar[PrivateKey]
    HOP3: ClassVar[Address]
    HOP4_KEY: ClassVar[PrivateKey]
    HOP4: ClassVar[Address]
    HOP5_KEY: ClassVar[PrivateKey]
    HOP5: ClassVar[Address]
    HOP3_KEY, HOP3 = make_privkey_address()
    HOP4_KEY, HOP4 = make_privkey_address()
    HOP5_KEY, HOP5 = make_privkey_address()
    PKEYS: ClassVar[Tuple[PrivateKey, ...]] = (HOP1_KEY, HOP2_KEY, HOP3_KEY, HOP4_KEY, HOP5_KEY)
    ADDRESSES: ClassVar[Tuple[Address, ...]] = (HOP1, HOP2, HOP3, HOP4, HOP5)

    def __init__(self, channels: List[NettingChannelState], our_privatekeys: List[Optional[PrivateKey]], partner_privatekeys: List[Optional[PrivateKey]]) -> None:
        self.channels: List[NettingChannelState] = channels
        self.our_privatekeys: List[Optional[PrivateKey]] = our_privatekeys
        self.partner_privatekeys: List[Optional[PrivateKey]] = partner_privatekeys

    @property
    def channel_map(self) -> Dict[ChannelID, NettingChannelState]:
        return {channel.identifier: channel for channel in self.channels}

    @property
    def nodeaddresses_to_networkstates(self) -> NodeNetworkStateMap:
        return {channel.partner_state.address: NetworkState.REACHABLE for channel in self.channels}

    def addresses_to_channel(self, token_network_address: TokenNetworkAddress = UNIT_TOKEN_NETWORK_ADDRESS) -> Dict[Tuple[TokenNetworkAddress, Address], NettingChannelState]:
        return {(token_network_address, channel.partner_state.address): channel for channel in self.channels}

    def our_address(self, index: int) -> Address:
        return self.channels[index].our_state.address

    def partner_address(self, index: int) -> Address:
        return self.channels[index].partner_state.address

    def get_hop(self, channel_index: int) -> HopState:
        return make_hop_from_channel(self.channels[channel_index])

    def get_hops(self, *args: int) -> List[HopState]:
        return [self.get_hop(index) for index in args or range(len(self.channels))]

    def get_route(self, channel_index: int, estimated_fee: FeeAmount = FeeAmount(0)) -> RouteState:
        """Creates an *outbound* (possibly incomplete) RouteState,
        based on channel our/partner addresses.

        Only use this in order to create routes to be processed by the initiator/mediator the
        channel belongs to. This route is not suited for constructing the route-metadata of
        a multihop payment (e.g. as the initiator).

        The route create is the minimally required route for processing a transfer *locally*.
        The requirement for this is that our address and the next hop address has to be included
        in order for us to find the recipient of the mediated locked-transfer that will be sent
        out.
        A valid route for a multihop payment should include the full path from initiator to
        target.
        """
        channel_state = self.channels[channel_index]
        route = [channel_state.our_state.address, channel_state.partner_state.address]
        return RouteState(route=route, estimated_fee=estimated_fee)

    def get_routes(self, *args: int, estimated_fee: FeeAmount = FeeAmount(0)) -> List[RouteState]:
        return [self.get_route(index, estimated_fee) for index in args or range(len(self.channels))]

    def __getitem__(self, item: int) -> NettingChannelState:
        return self.channels[item]

def make_channel_set(properties: Optional[List[NettingChannelStateProperties]] = None, defaults: NettingChannelStateProperties = NettingChannelStateProperties.DEFAULTS, number_of_channels: Optional[int] = None) -> ChannelSet:
    if number_of_channels is None:
        number_of_channels = len(properties) if properties is not None else 0
    channels: List[NettingChannelState] = []
    our_pkeys: List[Optional[PrivateKey]] = [None] * number_of_channels
    partner_pkeys: List[Optional[PrivateKey]] = [None] * number_of_channels
    if properties is None:
        properties = []
    while len(properties) < number_of_channels:
        properties.append(NettingChannelStateProperties())
    for i in range(number_of_channels):
        our_pkeys[i], partner_pkeys[i] = pkeys_from_channel_state(properties[i], defaults)
        channels.append(create(properties[i], defaults))
    return ChannelSet(channels, our_pkeys, partner_pkeys)

def make_channel_set_from_amounts(amounts: List[Balance]) -> ChannelSet:
    properties = [NettingChannelStateProperties(our_state=replace(NettingChannelEndStateProperties.OUR_STATE, balance=amount)) for amount in amounts]
    return make_channel_set(properties)

def mediator_make_channel_pair(defaults: Optional[NettingChannelStateProperties] = None, amount: TokenAmount = UNIT_TRANSFER_AMOUNT, token_network_address: TokenNetworkAddress = UNIT_TOKEN_NETWORK_ADDRESS) -> ChannelSet:
    properties_list = [NettingChannelStateProperties(canonical_identifier=make_canonical_identifier(channel_identifier=1, token_network_address=token_network_address), our_state=NettingChannelEndStateProperties.OUR_STATE, partner_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_SENDER, balance=amount)), NettingChannelStateProperties(canonical_identifier=make_canonical_identifier(channel_identifier=2, token_network_address=token_network_address), our_state=replace(NettingChannelEndStateProperties.OUR_STATE, balance=amount), partner_state=NettingChannelEndStateProperties(address=UNIT_TRANSFER_TARGET))]
    return make_channel_set(properties_list, defaults)

def mediator_make_init_action(channels: 'ChannelSet', transfer: LockedTransferSignedState) -> ActionInitMediator:

    def get_forward_channel(route_state: RouteState) -> Optional[ChannelID]:
        for channel_state in channels.channels:
            next_hop = route_state.hop_after(channel_state.our_state.address)
            if next_hop:
                if next_hop == channel_state.partner_state.address:
                    return channel_state.identifier
        return None
    forwards = [get_forward_channel(route_state) for route_state in transfer.route_states]
    assert len(forwards) == len(transfer.route_states)
    return ActionInitMediator(from_hop=channels.get_hop(0), candidate_route_states=transfer.route_states, from_transfer=transfer, balance_proof=transfer.balance_proof, sender=transfer.balance_proof.sender)

def initiator_make_init_action(channels: 'ChannelSet', routes: List[List[Address]], transfer: TransferDescriptionWithSecretState, estimated_fee: FeeAmount) -> ActionInitInitiator:

    def get_forward_channel(route_state: RouteState) -> Optional[ChannelID]:
        for channel_state in channels.channels:
            next_hop = route_state.hop_after(channel_state.our_state.address)
            if next_hop:
                if next_hop == channel_state.partner_state.address:
                    return channel_state.identifier
        return None
    route_states = [RouteState(route=route, estimated_fee=estimated_fee) for route in routes]
    forwards = [get_forward_channel(route_state) for route_state in route_states]
    assert len(forwards) == len(routes)
    return ActionInitInitiator(transfer=transfer, routes=route_states)

class MediatorTransfersPair(NamedTuple):
    channels: 'ChannelSet'
    transfers_pair: List[MediationPairState]
    amount: TokenAmount
    block_number: BlockNumber
    block_hash: BlockHash

    @property
    def channel_map(self) -> Dict[ChannelID, NettingChannelState]:
        return self.channels.channel_map

def make_transfers_pair(number_of_channels: int, amount: TokenAmount = UNIT_TRANSFER_AMOUNT, block_number: BlockNumber = BlockNumber(5)) -> MediatorTransfersPair:
    """
    Creates a set of channels and transfers_pair:


    For example:

    We are HOP1 (set as "our_state" for all channels)
    HOP1 = US

    E.g. for 5 number of channels we will create transfer pairs
    for the following payments, always as seen from our state (US)

    INITIATOR -> HOP2 -> US -> HOP3 -> HOP4 -> HOP5 -> TARGET
      received_transfer from HOP2
      sent transfer to HOP3
    INITIATOR -> HOP2 -> HOP3 -> US -> HOP4 -> HOP5 -> TARGET
      received_transfer from HOP3
      sent transfer to HOP4
    INITIATOR -> HOP2 -> US -> HOP3 -> HOP4 -> US -> HOP5 -> TARGET
      received_transfer from HOP4
      sent transfer to HOP5
    """
    deposit = 5 * amount
    defaults = create_properties(NettingChannelStateProperties(our_state=NettingChannelEndStateProperties(balance=deposit), partner_state=NettingChannelEndStateProperties(balance=deposit), open_transaction=SuccessfulTransactionStateProperties(finished_block_number=10)))
    properties_list = [NettingChannelStateProperties(canonical_identifier=make_canonical_identifier(channel_identifier=i), our_state=NettingChannelEndStateProperties(address=ChannelSet.ADDRESSES[0], privatekey=ChannelSet.PKEYS[0]), partner_state=NettingChannelEndStateProperties(address=ChannelSet.ADDRESSES[i + 1], privatekey=ChannelSet.PKEYS[i + 1])) for i in range(number_of_channels)]
    channels = make_channel_set(properties_list, defaults)
    lock_expiration = block_number + UNIT_REVEAL_TIMEOUT * 2
    pseudo_random_generator = random.Random()
    transfers_pairs: List[MediationPairState] = []
    our_address = channels[0].our_state.address
    for payer_index in range(number_of_channels - 1):
        payee_index = payer_index + 1
        receiver_channel = channels[payer_index]
        received_transfer = create(LockedTransferSignedStateProperties(amount=amount, expiration=lock_expiration, payment_identifier=UNIT_TRANSFER_IDENTIFIER, canonical_identifier=receiver_channel.canonical_identifier, sender=channels.partner_address(payer_index), recipient=our_address, pkey=channels.partner_privatekeys[payer_index]))
        is_valid, _, msg = channel.handle_receive_lockedtransfer(receiver_channel, received_transfer)
        assert is_valid, msg
        message_identifier = message_identifier_from_prng(pseudo_random_generator)
        secret: Optional[Secret] = None
        lockedtransfer_event = channel.send_lockedtransfer(channel_state=channels[payee_index], initiator=UNIT_TRANSFER_INITIATOR, target=UNIT_TRANSFER_TARGET, secret=secret, amount=amount, message_identifier=message_identifier, payment_identifier=UNIT_TRANSFER_IDENTIFIER, expiration=lock_expiration, secrethash=UNIT_SECRETHASH, route_states=received_transfer.route_states, previous_metadata=received_transfer.metadata)
        assert lockedtransfer_event
        lock_timeout = lock_expiration - block_number
        assert channel.is_channel_usable_for_mediation(channel_state=channels[payee_index], transfer_amount=amount, lock_timeout=lock_timeout)
        sent_transfer = lockedtransfer_event.transfer
        pair = MediationPairState(payer_transfer=received_transfer, payee_address=lockedtransfer_event.recipient, payee_transfer=sent_transfer)
        transfers_pairs.append(pair)
    return MediatorTransfersPair(channels=channels, transfers_pair=transfers_pairs, amount=amount, block_number=block_number, block_hash=make_block_hash())

@dataclass
class ContainerForChainStateTests:
    chain_state: ChainState
    our_address: Address
    token_network_registry_address: TokenNetworkRegistryAddress
    token_address: TokenAddress
    token_network_address: TokenNetworkAddress
    channel_set: ChannelSet

    @property
    def channels(self) -> List[NettingChannelState]:
        return self.channel_set.channels

    @property
    def token_network(self) -> Optional[TokenNetworkState]:
        return views.get_token_network_by_address(chain_state=self.chain_state, token_network_address=self.token_network_address)

def make_chain_state(number_of_channels: int, properties: Optional[List[NettingChannelStateProperties]] = None, defaults: NettingChannelStateProperties = NettingChannelStateProperties.DEFAULTS) -> ContainerForChainStateTests:
    """Factory for populating a complete `ChainState`.

    Sets up a `ChainState` instance with `number_of_channels` `NettingChannelState`s inside one
    `TokenNetworkState` inside one `TokenNetworkRegistryState`.

    The returned container, `ContainerForChainStateTests`, provides direct access to the most used
    function parameters when traversing a `ChainState` (i.e. the `token_network_address` of the
    populated `TokenNetworkState`), as well as the `ChannelSet` that created the
    `NettingChannelState`s.
    """
    channel_set = make_channel_set(number_of_channels=number_of_channels, properties=properties, defaults=defaults)
    assert len(set((c.canonical_identifier.token_network_address for c in channel_set.channels))) == 1
    assert len(set((c.our_state.address for c in channel_set.channels))) == 1
    token_network_address = channel_set.channels[0].canonical_identifier.token_network_address
    token_address = make_address()
    token_network_state = TokenNetworkState(address=token_network_address, token_address=token_address)
    for netting_channel in channel_set.channels:
        token_network_state.channelidentifiers_to_channels[netting_channel.canonical_identifier.channel_identifier] = netting_channel
        token_network_state.partneraddresses_to_channelidentifiers[netting_channel.partner_state.address].append(netting_channel.canonical_identifier.channel_identifier)
    token_network_registry_address = make_address()
    our_address = channel_set.channels[0].our_state.address
    chain_state = ChainState(pseudo_random_generator=random.Random(), block_number=BlockNumber(1), block_hash=make_block_hash(), our_address=our_address, chain_id=UNIT_CHAIN_ID)
    chain_state.identifiers_to_tokennetworkregistries[token_network_registry_address] = TokenNetworkRegistryState(address=token_network_registry_address, token_network_list=[token_network_state])
    chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses[token_network_address] = token_network_registry_address
    chain_state.nodeaddresses_to_networkstates = make_node_availability_map([channel.partner_state.address for channel in channel_set.channels])
    return ContainerForChainStateTests(chain_state=chain_state, our_address=our_address, token_network_registry_address=token_network_registry_address, token_address=token_address, token_network_address=token_network_address, channel_set=channel_set)

def make_node_availability_map(nodes: List[Address]) -> NodeNetworkStateMap:
    return {node: NetworkState.REACHABLE for node in nodes}

def make_route_from_channel(channel: NettingChannelState) -> RouteState:
    return RouteState(route=[channel.our_state.address, channel.partner_state.address])

@dataclass(frozen=True)
class RouteProperties(Properties):
    capacity2to1: int = 0

def route_properties_to_channel(route: Any) -> NettingChannelState:
    channel_state = create(NettingChannelStateProperties(canonical_identifier=make_canonical_identifier(), our_state=NettingChannelEndStateProperties(address=route.address1, balance=route.capacity1to2), partner_state=NettingChannelEndStateProperties(address=route.address2, balance=route.capacity2to1), open_transaction=SuccessfulTransactionState(1, 0)))
    return channel_state

def create_network(token_network_state: TokenNetworkState, our_address: Address, routes: List[Any], block_number: BlockNumber, block_hash: Optional[BlockHash] = None) -> Tuple[TokenNetworkState, List[NettingChannelState]]:
    """Creates a network from route properties.

    If the address in the route is our_address, create a channel also.
    Returns a list of created channels and the new state.
    """
    block_hash = block_hash or make_block_hash()
    state = token_network_state
    channels: List[NettingChannelState] = []
    for route in routes:
        if route.address1 == our_address:
            channel_state = route_properties_to_channel(route)
            state_change = ContractReceiveChannelNew(transaction_hash=make_transaction_hash(), channel_state=channel_state, block_number=block_number, block_hash=block_hash)
            channels.append(channel_state)
        else:
            state_change = ContractReceiveRouteNew(transaction_hash=make_transaction_hash(), canonical_identifier=make_canonical_identifier(), participant1=route.address1, participant2=route.address2, block_number=block_number, block_hash=block_hash)
        iteration = token_network.state_transition(token_network_state=state, state_change=state_change, block_number=block_number, block_hash=block_hash, pseudo_random_generator=random.Random())
        state = iteration.new_state
    return (state, channels)