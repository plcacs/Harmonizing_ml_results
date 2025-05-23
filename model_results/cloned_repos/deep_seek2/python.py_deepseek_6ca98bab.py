import gevent
import opentracing
import structlog
from eth_utils import is_binary_address

from raiden import waiting
from raiden.api.exceptions import ChannelNotFound, NonexistingChannel
from raiden.constants import BLOCK_ID_PENDING, NULL_ADDRESS_BYTES, UINT64_MAX, UINT256_MAX
from raiden.exceptions import (
    AlreadyRegisteredTokenAddress,
    DepositMismatch,
    DepositOverLimit,
    DuplicatedChannelError,
    InsufficientFunds,
    InsufficientGasReserve,
    InvalidAmount,
    InvalidBinaryAddress,
    InvalidPaymentIdentifier,
    InvalidRevealTimeout,
    InvalidSecret,
    InvalidSecretHash,
    InvalidSettleTimeout,
    InvalidTokenAddress,
    RaidenRecoverableError,
    SamePeerAddress,
    ServiceRequestFailed,
    TokenNetworkDeprecated,
    TokenNotRegistered,
    UnexpectedChannelState,
    UnknownTokenAddress,
    UserDepositNotConfigured,
    WithdrawMismatch,
)
from raiden.settings import DEFAULT_RETRY_TIMEOUT, PythonApiConfig
from raiden.storage.utils import TimestampedEvent
from raiden.transfer import channel, views
from raiden.transfer.architecture import Event, StateChange, TransferTask
from raiden.transfer.events import (
    EventPaymentReceivedSuccess,
    EventPaymentSentFailed,
    EventPaymentSentSuccess,
)
from raiden.transfer.mediated_transfer.tasks import InitiatorTask, MediatorTask, TargetTask
from raiden.transfer.state import (
    ChainState,
    ChannelState,
    NettingChannelState,
    NetworkState,
    RouteState,
)
from raiden.transfer.state_change import ActionChannelClose, ActionChannelCoopSettle
from raiden.transfer.views import TransferRole, get_token_network_by_address
from raiden.utils.formatting import to_checksum_address
from raiden.utils.gas_reserve import has_enough_gas_reserve
from raiden.utils.transfers import create_default_identifier
from raiden.utils.typing import (
    TYPE_CHECKING,
    Address,
    Any,
    BlockIdentifier,
    BlockNumber,
    BlockTimeout,
    ChannelID,
    Dict,
    InitiatorAddress,
    List,
    LockedTransferType,
    NetworkTimeout,
    Optional,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretHash,
    T_Secret,
    T_SecretHash,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    TransactionHash,
    Tuple,
    WithdrawAmount,
)

if TYPE_CHECKING:
    from raiden.raiden_service import PaymentStatus, RaidenService

log = structlog.get_logger(__name__)


def event_filter_for_payments(
    event: Event,
    chain_state: Optional[ChainState] = None,
    partner_address: Optional[Address] = None,
    token_address: Optional[TokenAddress] = None,
) -> bool:
    """Filters payment history related events depending on given arguments

    - If no other args are given, all payment related events match.
    - If a token network identifier is given then only payment events for that match.
    - If a partner is also given then if the event is a payment sent event and the
      target matches it's returned. If it's a payment received and the initiator matches.
      then it's returned.
    - If a token address is given then all events are filtered to be about that token.
    """
    sent_and_target_matches = isinstance(
        event, (EventPaymentSentFailed, EventPaymentSentSuccess)
    ) and (partner_address is None or event.target == TargetAddress(partner_address))
    received_and_initiator_matches = isinstance(event, EventPaymentReceivedSuccess) and (
        partner_address is None or event.initiator == InitiatorAddress(partner_address)
    )

    token_address_matches = True
    if token_address:
        assert chain_state, "Filtering for token_address without a chain state is an error"
        token_network = get_token_network_by_address(
            chain_state=chain_state,
            token_network_address=event.token_network_address,  # type: ignore
        )
        if not token_network:
            token_address_matches = False
        else:
            token_address_matches = token_address == token_network.token_address

    return token_address_matches and (sent_and_target_matches or received_and_initiator_matches)


def flatten_transfer(transfer: LockedTransferType, role: TransferRole) -> Dict[str, Any]:
    return {
        "payment_identifier": str(transfer.payment_identifier),
        "token_address": to_checksum_address(transfer.token),
        "token_network_address": to_checksum_address(transfer.balance_proof.token_network_address),
        "channel_identifier": str(transfer.balance_proof.channel_identifier),
        "initiator": to_checksum_address(transfer.initiator),
        "target": to_checksum_address(transfer.target),
        "transferred_amount": str(transfer.balance_proof.transferred_amount),
        "locked_amount": str(transfer.balance_proof.locked_amount),
        "role": role.value,
    }


def get_transfer_from_task(
    secrethash: SecretHash, transfer_task: TransferTask
) -> Optional[LockedTransferType]:
    if isinstance(transfer_task, InitiatorTask):
        # Work around for https://github.com/raiden-network/raiden/issues/5480,
        # can be removed when
        # https://github.com/raiden-network/raiden/issues/5515 is done.
        if secrethash not in transfer_task.manager_state.initiator_transfers:
            return None

        return transfer_task.manager_state.initiator_transfers[secrethash].transfer
    elif isinstance(transfer_task, MediatorTask):
        pairs = transfer_task.mediator_state.transfers_pair
        if pairs:
            return pairs[-1].payer_transfer

        assert transfer_task.mediator_state.waiting_transfer, "Invalid mediator_state"
        return transfer_task.mediator_state.waiting_transfer.transfer
    elif isinstance(transfer_task, TargetTask):
        return transfer_task.target_state.transfer

    raise ValueError("get_transfer_from_task for a non TransferTask argument")


def transfer_tasks_view(
    transfer_tasks: Dict[SecretHash, TransferTask],
    token_address: Optional[TokenAddress] = None,
    channel_id: Optional[ChannelID] = None,
) -> List[Dict[str, Any]]:
    view = []

    for secrethash, transfer_task in transfer_tasks.items():
        transfer = get_transfer_from_task(secrethash, transfer_task)

        if transfer is None:
            continue
        if token_address is not None:
            if transfer.token != token_address:
                continue
            elif channel_id is not None:
                if transfer.balance_proof.channel_identifier != channel_id:
                    continue

        view.append(flatten_transfer(transfer, transfer_task.role))

    return view


class RaidenAPI:  # pragma: no unittest
    # pylint: disable=too-many-public-methods

    def __init__(self, raiden: "RaidenService") -> None:
        self.raiden = raiden

    @property
    def address(self) -> Address:
        return self.raiden.address

    @property
    def notifications(self) -> Dict:
        return self.raiden.notifications

    @property
    def config(self) -> PythonApiConfig:
        return self.raiden.config.python_api

    def _raise_for_invalid_channel_timeouts(
        self, settle_timeout: BlockTimeout, reveal_timeout: BlockTimeout
    ) -> None:
        min_reveal_timeout = self.config.minimum_reveal_timeout
        if reveal_timeout < min_reveal_timeout:
            if reveal_timeout <= 0:
                raise InvalidRevealTimeout("reveal_timeout should be larger than zero.")
            else:
                raise InvalidRevealTimeout(
                    "reveal_timeout is lower than the required minimum value of"
                    f" { min_reveal_timeout }"
                )

        if settle_timeout < reveal_timeout * 2:
            raise InvalidSettleTimeout(
                "`settle_timeout` can not be smaller than double the "
                "`reveal_timeout`.\n "
                "\n "
                "The setting `reveal_timeout` determines the maximum number of "
                "blocks it should take a transaction to be mined when the "
                "blockchain is under congestion. This setting determines the "
                "when a node must go on-chain to register a secret, and it is "
                "therefore the lower bound of the lock expiration. The "
                "`settle_timeout` determines when a channel can be settled "
                "on-chain, for this operation to be safe all locks must have "
                "been resolved, for this reason the `settle_timeout` has to be "
                "larger than `reveal_timeout`."
            )

    def get_channel(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
    ) -> NettingChannelState:
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress("Expected binary address format for token in get_channel")

        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress("Expected binary address format for partner in get_channel")

        with opentracing.tracer.start_span("get_channel_list"):
            channel_list = self.get_channel_list(registry_address, token_address, partner_address)
        msg = f"Found {len(channel_list)} channels, but expected 0 or 1."
        assert len(channel_list) <= 1, msg

        if not channel_list:
            msg = (
                f"Channel with partner '{to_checksum_address(partner_address)}' "
                f"for token '{to_checksum_address(token_address)}' could not be "
                f"found."
            )
            raise ChannelNotFound(msg)

        return channel_list[0]

    def token_network_register(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        channel_participant_deposit_limit: TokenAmount,
        token_network_deposit_limit: TokenAmount,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> TokenNetworkAddress:
        """Register the `token_address` in the blockchain. If the address is already
           registered but the event has not been processed this function will block
           until the next block to make sure the event is processed.

        Raises:
            InvalidBinaryAddress: If the registry_address or token_address is not a valid address.
            AlreadyRegisteredTokenAddress: If the token is already registered.
            RaidenRecoverableError: If the register transaction failed, this may
                happen because the account has not enough balance to pay for the
                gas or this register call raced with another transaction and lost.
            InvalidTokenAddress: If token_address is the null address (0x000000....00).
        """

        if not is_binary_address(registry_address):
            raise InvalidBinaryAddress("registry_address must be a valid address in binary")

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress("token_address must be a valid address in binary")

        if token_address == NULL_ADDRESS_BYTES:
            raise InvalidTokenAddress("token_address must be non-zero")

        # The following check is on the same chain state as the
        # `chainstate` variable defined below because the chain state does
        # not change between this line and seven lines below.
        # views.state_from_raiden() returns the same state again and again
        # as far as this gevent context is running.
        if token_address in self.get_tokens_list(registry_address):
            raise AlreadyRegisteredTokenAddress("Token already registered")

        chainstate = views.state_from_raiden(self.raiden)

        registry = self.raiden.proxy_manager.token_network_registry(
            registry_address, block_identifier=chainstate.block_hash
        )

        _, token_network_address = registry.add_token(
            token_address=token_address,
            channel_participant_deposit_limit=channel_participant_deposit_limit,
            token_network_deposit_limit=token_network_deposit_limit,
            given_block_identifier=chainstate.block_hash,
        )

        waiting.wait_for_token_network(
            raiden=self.raiden,
            token_network_registry_address=registry_address,
            token_address=token_address,
            retry_timeout=retry_timeout,
        )

        return token_network_address

    def token_network_leave(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> List[NettingChannelState]:
        """Close all channels and wait for settlement."""
        if not is_binary_address(registry_address):
            raise InvalidBinaryAddress("registry_address must be a valid address in binary")
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress("token_address must be a valid address in binary")

        token_network_address = views.get_token_network_address_by_token_address(
            chain_state=views.state_from_raiden(self.raiden),
            token_network_registry_address=registry_address,
            token_address=token_address,
        )

        if token_network_address is None:
            raise UnknownTokenAddress(
                f"Token {to_checksum_address(token_address)} is not registered "
                f"with the network {to_checksum_address(registry_address)}."
            )

        channels = self.get_channel_list(registry_address, token_address)
        self.channel_batch_close(
            registry_address=registry_address,
            token_address=token_address,
            partner_addresses=[c.partner_state.address for c in channels],
            retry_timeout=retry_timeout,
        )
        return channels

    def is_already_existing_channel(
        self,
        token_network_address: TokenNetworkAddress,
        partner_address: Address,
        block_identifier: BlockIdentifier = BLOCK_ID_PENDING,
    ) -> bool:
        proxy_manager = self.raiden.proxy_manager
        proxy = proxy_manager.address_to_token_network[token_network_address]
        channel_identifier = proxy.get_channel_identifier_or_none(
            participant1=self.raiden.address,
            participant2=partner_address,
            block_identifier=block_identifier,
        )

        return channel_identifier is not None

    def channel_open(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
        settle_timeout: Optional[BlockTimeout] = None,
        reveal_timeout: Optional[BlockTimeout] = None,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> ChannelID:
        """Open a channel with the peer at `partner_address`
        with the given `token_address`.
        """
        if settle_timeout is None:
            settle_timeout = self.raiden.config.settle_timeout

        if reveal_timeout is None:
            reveal_timeout = self.raiden.config.reveal_timeout

        self._raise_for_invalid_channel_timeouts(settle_timeout, reveal_timeout)

        if not is_binary_address(registry_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for registry in channel open"
            )

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress("Expected binary address format for token in channel open")

        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for partner in channel open"
            )

        confirmed_block_identifier = views.get_confirmed_blockhash(self.raiden)
        registry = self.raiden.proxy_manager.token_network_registry(
            registry_address, block_identifier=confirmed_block_identifier
        )

        settlement_timeout_min = registry.settlement_timeout_min(
            block_identifier=confirmed_block_identifier
        )
        settlement_timeout_max = registry.settlement_timeout_max(
            block_identifier=confirmed_block_identifier
        )

        if settle_timeout < settlement_timeout_min:
            raise InvalidSettleTimeout(
                f"Settlement timeout should be at least {settlement_timeout_min}"
            )

        if settle_timeout > settlement_timeout_max:
            raise InvalidSettleTimeout(
                f"Settlement timeout exceeds max of {settlement_timeout_max}"
            )

        token_network_address = registry.get_token_network(
            token_address=token_address, block_identifier=confirmed_block_identifier
        )
        if token_network_address is None:
            raise TokenNotRegistered(
                "Token network for token %s does not exist" % to_checksum_address(token_address)
            )

        token_network = self.raiden.proxy_manager.token_network(
            address=token_network_address, block_identifier=confirmed_block_identifier
        )

        safety_deprecation_switch = token_network.safety_deprecation_switch(
            block_identifier=confirmed_block_identifier
        )

        if safety_deprecation_switch:
            msg = (
                "This token_network has been deprecated. New channels cannot be "
                "open for this network, usage of the newly deployed token "
                "network contract is highly encouraged."
            )
            raise TokenNetworkDeprecated(msg)

        duplicated_channel = self.is_already_existing_channel(
            token_network_address=token_network_address,
            partner_address=partner_address,
            block_identifier=confirmed_block_identifier,
        )
        if duplicated_channel:
            raise DuplicatedChannelError(
                f"A channel with {to_checksum_address(partner_address)} for token "
                f"{to_checksum_address(token_address)} already exists. "
                f"(At blockhash: {confirmed_block_identifier.hex()})"
            )

        has_enough_reserve, estimated_required_reserve = has_enough_gas_reserve(
            self.raiden, channels_to_open=1
        )

        if not has_enough_reserve:
            raise InsufficientGasReserve(
                "The account balance is below the estimated amount necessary to "
                "finish the lifecycles of all active channels. A balance of at "
                f"least {estimated_required_reserve} wei is required."
            )

        try:
            token_network.new_netting_channel(
                partner=partner_address,
                settle_timeout=settle_timeout,
                given_block_identifier=confirmed_block_identifier,
            )
        except DuplicatedChannelError:
            log.info("partner opened channel first")
        except RaidenRecoverableError:
            # The channel may have been created in the pending block.
            duplicated_channel = self.is_already_existing_channel(
                token_network_address=token_network_address, partner_address=partner_address
            )
            if duplicated_channel:
                log.info("Channel has already been opened")
            else:
                raise

        waiting.wait_for_newchannel(
            raiden=self.raiden,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
            retry_timeout=retry_timeout,
        )

        chain_state = views.state_from_raiden(self.raiden)
        channel_state = views.get_channelstate_for(
            chain_state=chain_state,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
        )

        assert channel_state, f"channel {channel_state} is gone"

        self.raiden.set_channel_reveal_timeout(
            canonical_identifier=channel_state.canonical_identifier, reveal_timeout=reveal_timeout
        )

        return channel_state.identifier

    def mint_token_for(
        self, token_address: TokenAddress, to: Address, value: TokenAmount
    ) -> TransactionHash:
        """Try to mint `value` units of the token at `token_address` and
        assign them to `to`, using `mintFor`.

        Raises:
            MintFailed if the minting fails for any reason.

        Returns:
            TransactionHash of the successfully mined Ethereum transaction
            associated with the token mint.
        """
        confirmed_block_identifier = self.raiden.get_block_number()
        token_proxy = self.raiden.proxy_manager.custom_token(
            token_address, block_identifier=confirmed_block_identifier
        )
        return token_proxy.mint_for(value, to)

    def set_total_channel_withdraw(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
        total_withdraw: WithdrawAmount,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> None:
        """Set the `total_withdraw` in the channel with the peer at `partner_address` and the
        given `token_address`.

        Raises:
            InvalidBinaryAddress: If either token_address or partner_address is not
                20 bytes long.
            RaidenUnrecoverableError: May happen for multiple reasons:
                - During preconditions checks, if the channel was not open
                  at the time of the approve_and_set_total_deposit call.
                - If the transaction fails during gas estimation or
                  if a previous withdraw transaction with the same value
                   was already mined.
            DepositMismatch: The total withdraw amount did not increase.
        """
        chain_state = views.state_from_raiden(self.raiden)

        token_addresses = views.get_token_identifiers(chain_state, registry_address)
        channel_state = views.get_channelstate_for(
            chain_state=chain_state,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
        )

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for token in channel deposit"
            )

        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for partner in channel deposit"
            )

        if token_address not in token_addresses:
            raise UnknownTokenAddress("Unknown token address")

        if channel_state is None:
            raise NonexistingChannel("No channel with partner_address for the given token")

        if total_withdraw <= channel_state.our_total_withdraw:
            raise WithdrawMismatch(f"Total withdraw {total_withdraw} did not increase")

        current_balance = channel.get_balance(
            sender=channel_state.our_state, receiver=channel_state.partner_state
        )
        amount_to_withdraw = total_withdraw - channel_state.our_total_withdraw
        if amount_to_withdraw > current_balance:
            raise InsufficientFunds(
                "The withdraw of {} is bigger than the current balance of {}".format(
                    amount_to_withdraw, current_balance
                )
            )

        pfs_proxy = self.raiden.pfs_proxy
        recipient_address = channel_state.partner_state.address
        recipient_metadata = pfs_proxy.query_address_metadata(recipient_address)
        self.raiden.withdraw(
            canonical_identifier=channel_state.canonical_identifier,
            total_withdraw=total_withdraw,
            recipient_metadata=recipient_metadata,
        )

        waiting.wait_for_withdraw_complete(
            raiden=self.raiden,
            canonical_identifier=channel_state.canonical_identifier,
            total_withdraw=total_withdraw,
            retry_timeout=retry_timeout,
        )

    def set_total_channel_deposit(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
        total_deposit: TokenAmount,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> None:
        """Set the `total_deposit` in the channel with the peer at `partner_address` and the
        given `token_address` in order to be able to do transfers.

        Raises:
            InvalidBinaryAddress: If either token_address or partner_address is not
                20 bytes long.
            RaidenRecoverableError: May happen for multiple reasons:
                - If the token approval fails, e.g. the token may validate if
                account has enough balance for the allowance.
                - The deposit failed, e.g. the allowance did not set the token
                aside for use and the user spent it before deposit was called.
                - The channel was closed/settled between the allowance call and
                the deposit call.
            AddressWithoutCode: The channel was settled during the deposit
                execution.
            DepositOverLimit: The total deposit amount is higher than the limit.
            UnexpectedChannelState: The channel is no longer in an open state.
        """
        chain_state = views.state_from_raiden(self.raiden)

        token_addresses = views.get_token_identifiers(chain_state, registry_address)
        channel_state = views.get_channelstate_for(
            chain_state=chain_state,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
        )

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for token in channel deposit"
            )

        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for partner in channel deposit"
            )

        if token_address not in token_addresses:
            raise UnknownTokenAddress("Unknown token address")

        if channel_state is None:
            raise NonexistingChannel("No channel with partner_address for the given token")

        confirmed_block_identifier = chain_state.block_hash
        token = self.raiden.proxy_manager.token(
            token_address, block_identifier=confirmed_block_identifier
        )
        token_network_registry = self.raiden.proxy_manager.token_network_registry(
            registry_address, block_identifier=confirmed_block_identifier
        )
        token_network_address = token_network_registry.get_token_network(
            token_address=token_address, block_identifier=confirmed_block_identifier
        )

        if token_network_address is None:
            raise UnknownTokenAddress(
                f"Token {to_checksum_address(token_address)} is not registered "
                f"with the network {to_checksum_address(registry_address)}."
            )

        token_network_proxy = self.raiden.proxy_manager.token_network(
            address=token_network_address, block_identifier=confirmed_block_identifier
        )
        channel_proxy = self.raiden.proxy_manager.payment_channel(
            channel_state=channel_state, block_identifier=confirmed_block_identifier
        )

        blockhash = chain_state.block_hash
        token_network_proxy = channel_proxy.token_network

        safety_deprecation_switch = token_network_proxy.safety_deprecation_switch(
            block_identifier=blockhash
        )

        balance = token.balance_of(self.raiden.address, block_identifier=blockhash)

        network_balance = token.balance_of(
            address=Address(token_network_address), block_identifier=blockhash
        )
        token_network_deposit_limit = token_network_proxy.token_network_deposit_limit(
            block_identifier=blockhash
        )

        deposit_increase = total_deposit - channel_state.our_state.contract_balance

        channel_participant_deposit_limit = token_network_proxy.channel_participant_deposit_limit(
            block_identifier=blockhash
        )
        total_channel_deposit = total_deposit + channel_state.partner_state.contract_balance

        is_channel_open = channel.get_status(channel_state) == ChannelState.STATE_OPENED

        if not is_channel_open:
            raise UnexpectedChannelState("Channel is not in an open state.")

        if safety_deprecation_switch:
            msg = (
                "This token_network has been deprecated. "
                "All channels in this network should be closed and "
                "the usage of the newly deployed token network contract "
                "is highly encouraged."
            )
            raise TokenNetworkDeprecated(msg)

        if total_deposit <= channel_state.our_state.contract_balance:
            raise DepositMismatch("Total deposit did not increase.")

        # If this check succeeds it does not imply the `deposit` will
        # succeed, since the `deposit` transaction may race with another
        # transaction.
        if not (balance >= deposit_increase):
            msg = "Not enough balance to deposit. {} Available={} Needed={}".format(
                to_checksum_address(token_address), balance, deposit_increase
            )
            raise InsufficientFunds(msg)

        if network_balance + deposit_increase > token_network_deposit_limit:
            msg = (
                f"Deposit of {deposit_increase} would have exceeded "
                "the token network deposit limit."
            )
            raise DepositOverLimit(msg)

        if total_deposit > channel_participant_deposit_limit:
            msg = (
                f"Deposit of {total_deposit} is larger than the "
                f"channel participant deposit limit"
            )
            raise DepositOverLimit(msg)

        if total_channel_deposit >= UINT256_MAX:
            raise DepositOverLimit("Deposit overflow")

        try:
            channel_proxy.approve_and_set_total_deposit(
                total_deposit=total_deposit, block_identifier=blockhash
            )
        except RaidenRecoverableError as e:
            log.info(f"Deposit failed. {str(e)}")

        target_address = self.raiden.address
        waiting.wait_for_participant_deposit(
            raiden=self.raiden,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
            target_address=target_address,
            target_balance=total_deposit,
            retry_timeout=retry_timeout,
        )

    def set_reveal_timeout(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
        reveal_timeout: BlockTimeout,
    ) -> None:
        """Set the `reveal_timeout` in the channel with the peer at `partner_address` and the
        given `token_address`.

        Raises:
            InvalidBinaryAddress: If either token_address or partner_address is not
                20 bytes long.
            InvalidRevealTimeout: If reveal_timeout has an invalid value.
        """
        chain_state = views.state_from_raiden(self.raiden)

        token_addresses = views.get_token_identifiers(chain_state, registry_address)
        channel_state = views.get_channelstate_for(
            chain_state=chain_state,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_address=partner_address,
        )

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for token in channel deposit"
            )

        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                "Expected binary address format for partner in channel deposit"
            )

        if token_address not in token_addresses:
            raise UnknownTokenAddress("Unknown token address")

        if channel_state is None:
            raise NonexistingChannel("No channel with partner_address for the given token")

        try:
            self._raise_for_invalid_channel_timeouts(channel_state.settle_timeout, reveal_timeout)
        except InvalidSettleTimeout as ex:
            # convert the invalid settle timeout, since from the perspective of set_reveal_timeout
            # the new RevealTimeout is invalid
            raise InvalidRevealTimeout(str(ex)) from ex

        self.raiden.set_channel_reveal_timeout(
            canonical_identifier=channel_state.canonical_identifier, reveal_timeout=reveal_timeout
        )

    def channel_close(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_address: Address,
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
        coop_settle: bool = True,
    ) -> None:
        """Close a channel opened with `partner_address` for the given
        `token_address`.

        Race condition, this can fail if channel was closed externally.
        """
        self.channel_batch_close(
            registry_address=registry_address,
            token_address=token_address,
            partner_addresses=[partner_address],
            retry_timeout=retry_timeout,
            coop_settle=coop_settle,
        )

    def channel_batch_close(
        self,
        registry_address: TokenNetworkRegistryAddress,
        token_address: TokenAddress,
        partner_addresses: List[Address],
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
        coop_settle: bool = True,
    ) -> None:
        """Close a channel opened with `partner_address` for the given
        `token_address`.

        Race condition, this can fail if channel was closed externally.
        """

        if not is_binary_address(token_address):
            raise InvalidBinaryAddress("Expected binary address format for token in channel close")

        if not all(map(is_binary_address, partner_addresses)):
            raise InvalidBinaryAddress(
                "Expected binary address format for partner in channel close"
            )

        valid_tokens = views.get_token_identifiers(
            chain_state=views.state_from_raiden(self.raiden),
            token_network_registry_address=registry_address,
        )
        if token_address not in valid_tokens:
            raise UnknownTokenAddress("Token address is not known.")

        chain_state = views.state_from_raiden(self.raiden)
        channels_to_close = views.filter_channels_by_partneraddress(
            chain_state=chain_state,
            token_network_registry_address=registry_address,
            token_address=token_address,
            partner_addresses=partner_addresses,
        )

        if coop_settle:
            non_settled_channels = self._batch_coop_settle(channels_to_close, retry_timeout)
            if not non_settled_channels:
                return

        close_state_changes: List[StateChange] = [
            ActionChannelClose(canonical_identifier=channel_state.canonical_identifier)
            for channel_state in channels_to_close
        ]

        greenlets = set(self.raiden.handle_state_changes(close_state_changes))
        gevent.joinall(greenlets, raise_error=True)

        channel_ids = [channel_state.identifier for channel_state in channels_to_close]

        waiting.wait_for_close(
            raiden=self.raiden,
            token_network_registry_address=registry_address,
            token_address=token_address,
            channel_ids=channel_ids,
            retry_timeout=retry_timeout,
        )

    def _batch_coop_settle(
        self,
        channels_to_settle: List[NettingChannelState],
        retry_timeout: NetworkTimeout = DEFAULT_RETRY_TIMEOUT,
    ) -> List[NettingChannelState]:
        pfs_proxy = self.raiden.pfs_proxy
        coop_settle_state_changes: List[StateChange] = []
        to_coop_settle = set()
        for channel_state in channels_to_settle:
            recipient_address = channel_state.partner_state.address
            try:
                metadata = pfs_proxy.query_address_metadata(recipient_address)
            except ServiceRequestFailed:
                log.error(
                    "Partner is offline, coop settle not possible", address=recipient_address
                )
                continue
            to_coop_settle.add(channel_state.canonical_identifier)
            coop_settle_state_changes.append(
                ActionChannelCoopSettle(
                    canonical_identifier=channel_state.canonical_identifier,
                    recipient_metadata=metadata,
                )
            )

        greenlets = set(self.raiden.handle_state_changes(coop_settle_state_changes))
        gevent.joinall(greenlets, raise_error=True)

        # we need to get a new list of channel state objects since the ones we
        # have may have become stale
        ids = frozenset(ch.canonical_identifier for ch in channels_to_settle)
        chain_state = views.state_from_raiden(self.raiden)
        channels_to_settle = [
            ch for ch in views.list_all_channelstate(chain_state) if ch.canonical_identifier in ids
        ]

        channels_to_conditions = {}
        for channel_state in channels_to_settle:
            if channel_state.canonical_identifier not in to_coop_settle:
                continue
            # FIXME is there a race condition when we "get" the total-withdraw-values after they
            # have been determined by the state machine?
