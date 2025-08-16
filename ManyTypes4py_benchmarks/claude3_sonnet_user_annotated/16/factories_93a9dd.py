import random
import string
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import singledispatch
from hashlib import sha256
from operator import itemgetter
from typing import (
    Any, Dict, List, Optional, Tuple, Type, TypeVar, ClassVar, NamedTuple, Union, cast
)

from eth_utils import keccak

from raiden.constants import EMPTY_SIGNATURE, LOCKSROOT_OF_NO_LOCKS, UINT64_MAX, UINT256_MAX
from raiden.messages.decode import balanceproof_from_envelope
from raiden.messages.metadata import Metadata, RouteMetadata
from raiden.messages.transfers import Lock, LockedTransfer, LockExpired, RefundTransfer, Unlock
from raiden.transfer import channel, token_network, views
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state import (
    HashTimeLockState,
    LockedTransferSignedState,
    LockedTransferUnsignedState,
    MediationPairState,
    TransferDescriptionWithSecretState,
)
from raiden.transfer.mediated_transfer.state_change import ActionInitInitiator, ActionInitMediator
from raiden.transfer.state import (
    BalanceProofSignedState,
    BalanceProofUnsignedState,
    ChainState,
    HopState,
    NettingChannelEndState,
    NettingChannelState,
    NetworkState,
    PendingLocksState,
    RouteState,
    SuccessfulTransactionState,
    TokenNetworkRegistryState,
    TokenNetworkState,
    TransactionExecutionStatus,
    message_identifier_from_prng,
)
from raiden.transfer.state_change import ContractReceiveChannelNew, ContractReceiveRouteNew
from raiden.transfer.utils import hash_balance_data
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.packing import pack_balance_proof
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.signer import LocalSigner, Signer
from raiden.utils.transfers import random_secret
from raiden.utils.typing import (
    AdditionalHash,
    Address,
    AddressHex,
    AddressMetadata,
    Balance,
    BlockExpiration,
    BlockHash,
    BlockNumber,
    BlockTimeout,
    ChainID,
    ChannelID,
    FeeAmount,
    InitiatorAddress,
    Locksroot,
    MessageID,
    MonitoringServiceAddress,
    Nonce,
    NodeNetworkStateMap,
    PaymentAmount,
    PaymentID,
    PrivateKey,
    Secret,
    SecretHash,
    Signature,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    TransactionHash,
    WithdrawAmount,
)

EMPTY = "empty"
GENERATE = "generate"

K = TypeVar("K")
V = TypeVar("V")


def _partial_dict(full_dict: Dict[K, V], *args: K) -> Dict[K, V]:
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
    