import gevent
import opentracing
import structlog
from eth_utils import is_binary_address
from raiden import waiting
from raiden.api.exceptions import ChannelNotFound, NonexistingChannel
from raiden.constants import BLOCK_ID_PENDING, NULL_ADDRESS_BYTES, UINT64_MAX, UINT256_MAX
from raiden.exceptions import AlreadyRegisteredTokenAddress, DepositMismatch, DepositOverLimit, DuplicatedChannelError, InsufficientFunds, InsufficientGasReserve, InvalidAmount, InvalidBinaryAddress, InvalidPaymentIdentifier, InvalidRevealTimeout, InvalidSecret, InvalidSecretHash, InvalidSettleTimeout, InvalidTokenAddress, RaidenRecoverableError, SamePeerAddress, ServiceRequestFailed, TokenNetworkDeprecated, TokenNotRegistered, UnexpectedChannelState, UnknownTokenAddress, UserDepositNotConfigured, WithdrawMismatch
from raiden.settings import DEFAULT_RETRY_TIMEOUT, PythonApiConfig
from raiden.storage.utils import TimestampedEvent
from raiden.transfer import channel, views
from raiden.transfer.architecture import Event, StateChange, TransferTask
from raiden.transfer.events import EventPaymentReceivedSuccess, EventPaymentSentFailed, EventPaymentSentSuccess
from raiden.transfer.mediated_transfer.tasks import InitiatorTask, MediatorTask, TargetTask
from raiden.transfer.state import ChainState, ChannelState, NettingChannelState, NetworkState, RouteState
from raiden.transfer.state_change import ActionChannelClose, ActionChannelCoopSettle
from raiden.transfer.views import TransferRole, get_token_network_by_address
from raiden.utils.formatting import to_checksum_address
from raiden.utils.gas_reserve import has_enough_gas_reserve
from raiden.utils.transfers import create_default_identifier
from raiden.utils.typing import TYPE_CHECKING, Address, Any, BlockIdentifier, BlockNumber, BlockTimeout, ChannelID, Dict, InitiatorAddress, List, LockedTransferType, NetworkTimeout, Optional, PaymentAmount, PaymentID, Secret, SecretHash, T_Secret, T_SecretHash, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, TransactionHash, Tuple, WithdrawAmount
if TYPE_CHECKING:
    from raiden.raiden_service import PaymentStatus, RaidenService
log = structlog.get_logger(__name__)


def event_filter_for_payments(event, chain_state=None, partner_address=None,
    token_address=None):
    """Filters payment history related events depending on given arguments

    - If no other args are given, all payment related events match.
    - If a token network identifier is given then only payment events for that match.
    - If a partner is also given then if the event is a payment sent event and the
      target matches it's returned. If it's a payment received and the initiator matches.
      then it's returned.
    - If a token address is given then all events are filtered to be about that token.
    """
    sent_and_target_matches: bool = isinstance(event, (
        EventPaymentSentFailed, EventPaymentSentSuccess)) and (
        partner_address is None or event.target == TargetAddress(
        partner_address))
    received_and_initiator_matches: bool = isinstance(event,
        EventPaymentReceivedSuccess) and (partner_address is None or event.
        initiator == InitiatorAddress(partner_address))
    token_address_matches: bool = True
    if token_address:
        assert chain_state, 'Filtering for token_address without a chain state is an error'
        token_network = get_token_network_by_address(chain_state=
            chain_state, token_network_address=event.token_network_address)
        if not token_network:
            token_address_matches = False
        else:
            token_address_matches = (token_address == token_network.
                token_address)
    return token_address_matches and (sent_and_target_matches or
        received_and_initiator_matches)


def flatten_transfer(transfer, role):
    return {'payment_identifier': str(transfer.payment_identifier),
        'token_address': to_checksum_address(transfer.token),
        'token_network_address': to_checksum_address(transfer.balance_proof
        .token_network_address), 'channel_identifier': str(transfer.
        balance_proof.channel_identifier), 'initiator': to_checksum_address
        (transfer.initiator), 'target': to_checksum_address(transfer.target
        ), 'transferred_amount': str(transfer.balance_proof.
        transferred_amount), 'locked_amount': str(transfer.balance_proof.
        locked_amount), 'role': role.value}


def get_transfer_from_task(secrethash, transfer_task):
    if isinstance(transfer_task, InitiatorTask):
        if secrethash not in transfer_task.manager_state.initiator_transfers:
            return None
        return transfer_task.manager_state.initiator_transfers[secrethash
            ].transfer
    elif isinstance(transfer_task, MediatorTask):
        pairs: Optional[List[Any]
            ] = transfer_task.mediator_state.transfers_pair
        if pairs:
            return pairs[-1].payer_transfer
        assert transfer_task.mediator_state.waiting_transfer, 'Invalid mediator_state'
        return transfer_task.mediator_state.waiting_transfer.transfer
    elif isinstance(transfer_task, TargetTask):
        return transfer_task.target_state.transfer
    raise ValueError('get_transfer_from_task for a non TransferTask argument')


def transfer_tasks_view(transfer_tasks, token_address=None, channel_id=None):
    view: List[Dict[str, Any]] = []
    for secrethash, transfer_task in transfer_tasks.items():
        transfer: Optional[LockedTransferType] = get_transfer_from_task(
            secrethash, transfer_task)
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


class RaidenAPI:
    raiden: 'RaidenService'

    def __init__(self, raiden):
        self.raiden = raiden

    @property
    def address(self):
        return self.raiden.address

    @property
    def notifications(self):
        return self.raiden.notifications

    @property
    def config(self):
        return self.raiden.config.python_api

    def _raise_for_invalid_channel_timeouts(self, settle_timeout,
        reveal_timeout):
        min_reveal_timeout: BlockTimeout = self.config.minimum_reveal_timeout
        if reveal_timeout < min_reveal_timeout:
            if reveal_timeout <= 0:
                raise InvalidRevealTimeout(
                    'reveal_timeout should be larger than zero.')
            else:
                raise InvalidRevealTimeout(
                    f'reveal_timeout is lower than the required minimum value of {min_reveal_timeout}'
                    )
        if settle_timeout < reveal_timeout * 2:
            raise InvalidSettleTimeout(
                """`settle_timeout` can not be smaller than double the `reveal_timeout`.
 
 The setting `reveal_timeout` determines the maximum number of blocks it should take a transaction to be mined when the blockchain is under congestion. This setting determines the when a node must go on-chain to register a secret, and it is therefore the lower bound of the lock expiration. The `settle_timeout` determines when a channel can be settled on-chain, for this operation to be safe all locks must have been resolved, for this reason the `settle_timeout` has to be larger than `reveal_timeout`."""
                )

    def get_channel(self, registry_address, token_address, partner_address):
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in get_channel')
        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in get_channel')
        with opentracing.tracer.start_span('get_channel_list'):
            channel_list: List[NettingChannelState] = self.get_channel_list(
                registry_address, token_address, partner_address)
        msg: str = f'Found {len(channel_list)} channels, but expected 0 or 1.'
        assert len(channel_list) <= 1, msg
        if not channel_list:
            msg = (
                f"Channel with partner '{to_checksum_address(partner_address)}' for token '{to_checksum_address(token_address)}' could not be found."
                )
            raise ChannelNotFound(msg)
        return channel_list[0]

    def token_network_register(self, registry_address, token_address,
        channel_participant_deposit_limit, token_network_deposit_limit,
        retry_timeout=DEFAULT_RETRY_TIMEOUT):
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
            raise InvalidBinaryAddress(
                'registry_address must be a valid address in binary')
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'token_address must be a valid address in binary')
        if token_address == NULL_ADDRESS_BYTES:
            raise InvalidTokenAddress('token_address must be non-zero')
        if token_address in self.get_tokens_list(registry_address):
            raise AlreadyRegisteredTokenAddress('Token already registered')
        chainstate: ChainState = views.state_from_raiden(self.raiden)
        registry = self.raiden.proxy_manager.token_network_registry(
            registry_address, block_identifier=chainstate.block_hash)
        _, token_network_address = registry.add_token(token_address=
            token_address, channel_participant_deposit_limit=
            channel_participant_deposit_limit, token_network_deposit_limit=
            token_network_deposit_limit, given_block_identifier=chainstate.
            block_hash)
        waiting.wait_for_token_network(raiden=self.raiden,
            token_network_registry_address=registry_address, token_address=
            token_address, retry_timeout=retry_timeout)
        return token_network_address

    def token_network_leave(self, registry_address, token_address,
        retry_timeout=DEFAULT_RETRY_TIMEOUT):
        """Close all channels and wait for settlement."""
        if not is_binary_address(registry_address):
            raise InvalidBinaryAddress(
                'registry_address must be a valid address in binary')
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'token_address must be a valid address in binary')
        token_network_address: Optional[TokenNetworkAddress
            ] = views.get_token_network_address_by_token_address(chain_state
            =views.state_from_raiden(self.raiden),
            token_network_registry_address=registry_address, token_address=
            token_address)
        if token_network_address is None:
            raise UnknownTokenAddress(
                f'Token {to_checksum_address(token_address)} is not registered with the network {to_checksum_address(registry_address)}.'
                )
        channels: List[NettingChannelState] = self.get_channel_list(
            registry_address, token_address)
        self.channel_batch_close(registry_address=registry_address,
            token_address=token_address, partner_addresses=[c.partner_state
            .address for c in channels], retry_timeout=retry_timeout)
        return channels

    def is_already_existing_channel(self, token_network_address,
        partner_address, block_identifier=BLOCK_ID_PENDING):
        proxy_manager = self.raiden.proxy_manager
        proxy = proxy_manager.address_to_token_network[token_network_address]
        channel_identifier: Optional[ChannelID
            ] = proxy.get_channel_identifier_or_none(participant1=self.
            raiden.address, participant2=partner_address, block_identifier=
            block_identifier)
        return channel_identifier is not None

    def channel_open(self, registry_address, token_address, partner_address,
        settle_timeout=None, reveal_timeout=None, retry_timeout=
        DEFAULT_RETRY_TIMEOUT):
        """Open a channel with the peer at `partner_address`
        with the given `token_address`.
        """
        if settle_timeout is None:
            settle_timeout = self.raiden.config.settle_timeout
        if reveal_timeout is None:
            reveal_timeout = self.raiden.config.reveal_timeout
        self._raise_for_invalid_channel_timeouts(settle_timeout, reveal_timeout
            )
        if not is_binary_address(registry_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for registry in channel open')
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in channel open')
        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in channel open')
        confirmed_block_identifier: BlockIdentifier = (views.
            get_confirmed_blockhash(self.raiden))
        registry = self.raiden.proxy_manager.token_network_registry(
            registry_address, block_identifier=confirmed_block_identifier)
        settlement_timeout_min: int = registry.settlement_timeout_min(
            block_identifier=confirmed_block_identifier)
        settlement_timeout_max: int = registry.settlement_timeout_max(
            block_identifier=confirmed_block_identifier)
        if settle_timeout < settlement_timeout_min:
            raise InvalidSettleTimeout(
                f'Settlement timeout should be at least {settlement_timeout_min}'
                )
        if settle_timeout > settlement_timeout_max:
            raise InvalidSettleTimeout(
                f'Settlement timeout exceeds max of {settlement_timeout_max}')
        token_network_address: Optional[TokenNetworkAddress
            ] = registry.get_token_network(token_address=token_address,
            block_identifier=confirmed_block_identifier)
        if token_network_address is None:
            raise TokenNotRegistered(
                'Token network for token %s does not exist' %
                to_checksum_address(token_address))
        token_network = self.raiden.proxy_manager.token_network(address=
            token_network_address, block_identifier=confirmed_block_identifier)
        safety_deprecation_switch: bool = (token_network.
            safety_deprecation_switch(block_identifier=
            confirmed_block_identifier))
        if safety_deprecation_switch:
            msg: str = (
                'This token_network has been deprecated. New channels cannot be open for this network, usage of the newly deployed token network contract is highly encouraged.'
                )
            raise TokenNetworkDeprecated(msg)
        duplicated_channel: bool = self.is_already_existing_channel(
            token_network_address=token_network_address, partner_address=
            partner_address, block_identifier=confirmed_block_identifier)
        if duplicated_channel:
            raise DuplicatedChannelError(
                f'A channel with {to_checksum_address(partner_address)} for token {to_checksum_address(token_address)} already exists. (At blockhash: {confirmed_block_identifier.hex()})'
                )
        has_enough_reserve: bool
        estimated_required_reserve: int
        has_enough_reserve, estimated_required_reserve = (
            has_enough_gas_reserve(self.raiden, channels_to_open=1))
        if not has_enough_reserve:
            raise InsufficientGasReserve(
                f'The account balance is below the estimated amount necessary to finish the lifecycles of all active channels. A balance of at least {estimated_required_reserve} wei is required.'
                )
        try:
            token_network.new_netting_channel(partner=partner_address,
                settle_timeout=settle_timeout, given_block_identifier=
                confirmed_block_identifier)
        except DuplicatedChannelError:
            log.info('partner opened channel first')
        except RaidenRecoverableError:
            duplicated_channel = self.is_already_existing_channel(
                token_network_address=token_network_address,
                partner_address=partner_address)
            if duplicated_channel:
                log.info('Channel has already been opened')
            else:
                raise
        waiting.wait_for_newchannel(raiden=self.raiden,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address, retry_timeout=
            retry_timeout)
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        channel_state: Optional[NettingChannelState
            ] = views.get_channelstate_for(chain_state=chain_state,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address)
        assert channel_state, f'channel {channel_state} is gone'
        self.raiden.set_channel_reveal_timeout(canonical_identifier=
            channel_state.canonical_identifier, reveal_timeout=reveal_timeout)
        return channel_state.identifier

    def mint_token_for(self, token_address, to, value):
        """Try to mint `value` units of the token at `token_address` and
        assign them to `to`, using `mintFor`.

        Raises:
            MintFailed if the minting fails for any reason.

        Returns:
            TransactionHash of the successfully mined Ethereum transaction
            associated with the token mint.
        """
        confirmed_block_identifier: BlockIdentifier = (self.raiden.
            get_block_number())
        token_proxy = self.raiden.proxy_manager.custom_token(token_address,
            block_identifier=confirmed_block_identifier)
        return token_proxy.mint_for(value, to)

    def set_total_channel_withdraw(self, registry_address, token_address,
        partner_address, total_withdraw, retry_timeout=DEFAULT_RETRY_TIMEOUT):
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
                - AddressWithoutCode: The channel was settled during the deposit
                    execution.
                - DepositOverLimit: The total deposit amount is higher than the limit.
                - UnexpectedChannelState: The channel is no longer in an open state.
        """
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        token_addresses: List[TokenAddress] = views.get_token_identifiers(
            chain_state, registry_address)
        channel_state: Optional[NettingChannelState
            ] = views.get_channelstate_for(chain_state=chain_state,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address)
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in channel deposit')
        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in channel deposit'
                )
        if token_address not in token_addresses:
            raise UnknownTokenAddress('Unknown token address')
        if channel_state is None:
            raise NonexistingChannel(
                'No channel with partner_address for the given token')
        if total_withdraw <= channel_state.our_total_withdraw:
            raise WithdrawMismatch(
                f'Total withdraw {total_withdraw} did not increase')
        current_balance: TokenAmount = channel.get_balance(sender=
            channel_state.our_state, receiver=channel_state.partner_state)
        amount_to_withdraw: WithdrawAmount = (total_withdraw -
            channel_state.our_total_withdraw)
        if amount_to_withdraw > current_balance:
            raise InsufficientFunds(
                'The withdraw of {} is bigger than the current balance of {}'
                .format(amount_to_withdraw, current_balance))
        pfs_proxy = self.raiden.pfs_proxy
        recipient_address: Address = channel_state.partner_state.address
        recipient_metadata: Any = pfs_proxy.query_address_metadata(
            recipient_address)
        self.raiden.withdraw(canonical_identifier=channel_state.
            canonical_identifier, total_withdraw=total_withdraw,
            recipient_metadata=recipient_metadata)
        waiting.wait_for_withdraw_complete(raiden=self.raiden,
            canonical_identifier=channel_state.canonical_identifier,
            total_withdraw=total_withdraw, retry_timeout=retry_timeout)

    def set_total_channel_deposit(self, registry_address, token_address,
        partner_address, total_deposit, retry_timeout=DEFAULT_RETRY_TIMEOUT):
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
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        token_addresses: List[TokenAddress] = views.get_token_identifiers(
            chain_state, registry_address)
        channel_state: Optional[NettingChannelState
            ] = views.get_channelstate_for(chain_state=chain_state,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address)
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in channel deposit')
        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in channel deposit'
                )
        if token_address not in token_addresses:
            raise UnknownTokenAddress('Unknown token address')
        if channel_state is None:
            raise NonexistingChannel(
                'No channel with partner_address for the given token')
        confirmed_block_identifier: BlockIdentifier = chain_state.block_hash
        token: Any = self.raiden.proxy_manager.token(token_address,
            block_identifier=confirmed_block_identifier)
        token_network_registry = (self.raiden.proxy_manager.
            token_network_registry(registry_address, block_identifier=
            confirmed_block_identifier))
        token_network_address: Optional[TokenNetworkAddress
            ] = token_network_registry.get_token_network(token_address=
            token_address, block_identifier=confirmed_block_identifier)
        if token_network_address is None:
            raise UnknownTokenAddress(
                f'Token {to_checksum_address(token_address)} is not registered with the network {to_checksum_address(registry_address)}.'
                )
        token_network_proxy = self.raiden.proxy_manager.token_network(address
            =token_network_address, block_identifier=confirmed_block_identifier
            )
        channel_proxy = self.raiden.proxy_manager.payment_channel(channel_state
            =channel_state, block_identifier=confirmed_block_identifier)
        blockhash: BlockIdentifier = chain_state.block_hash
        token_network_proxy = channel_proxy.token_network
        safety_deprecation_switch: bool = (token_network_proxy.
            safety_deprecation_switch(block_identifier=blockhash))
        balance: TokenAmount = token.balance_of(self.raiden.address,
            block_identifier=blockhash)
        network_balance: TokenAmount = token.balance_of(address=Address(
            token_network_address), block_identifier=blockhash)
        token_network_deposit_limit: TokenAmount = (token_network_proxy.
            token_network_deposit_limit(block_identifier=blockhash))
        deposit_increase: TokenAmount = (total_deposit - channel_state.
            our_state.contract_balance)
        channel_participant_deposit_limit: TokenAmount = (token_network_proxy
            .channel_participant_deposit_limit(block_identifier=blockhash))
        total_channel_deposit: TokenAmount = (total_deposit + channel_state
            .partner_state.contract_balance)
        is_channel_open: bool = channel.get_status(channel_state
            ) == ChannelState.STATE_OPENED
        if not is_channel_open:
            raise UnexpectedChannelState('Channel is not in an open state.')
        if safety_deprecation_switch:
            msg: str = (
                'This token_network has been deprecated. All channels in this network should be closed and the usage of the newly deployed token network contract is highly encouraged.'
                )
            raise TokenNetworkDeprecated(msg)
        if total_deposit <= channel_state.our_state.contract_balance:
            raise DepositMismatch('Total deposit did not increase.')
        if not balance >= deposit_increase:
            msg: str = (
                'Not enough balance to deposit. {} Available={} Needed={}'.
                format(to_checksum_address(token_address), balance,
                deposit_increase))
            raise InsufficientFunds(msg)
        if network_balance + deposit_increase > token_network_deposit_limit:
            msg: str = (
                f'Deposit of {deposit_increase} would have exceeded the token network deposit limit.'
                )
            raise DepositOverLimit(msg)
        if total_deposit > channel_participant_deposit_limit:
            msg: str = (
                f'Deposit of {total_deposit} is larger than the channel participant deposit limit'
                )
            raise DepositOverLimit(msg)
        if total_channel_deposit >= UINT256_MAX:
            raise DepositOverLimit('Deposit overflow')
        try:
            channel_proxy.approve_and_set_total_deposit(total_deposit=
                total_deposit, block_identifier=blockhash)
        except RaidenRecoverableError as e:
            log.info(f'Deposit failed. {str(e)}')
        target_address: Address = self.raiden.address
        waiting.wait_for_participant_deposit(raiden=self.raiden,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address, target_address=
            target_address, target_balance=total_deposit, retry_timeout=
            retry_timeout)

    def set_reveal_timeout(self, registry_address, token_address,
        partner_address, reveal_timeout):
        """Set the `reveal_timeout` in the channel with the peer at `partner_address` and the
        given `token_address`.

        Raises:
            InvalidBinaryAddress: If either token_address or partner_address is not
                20 bytes long.
            InvalidRevealTimeout: If reveal_timeout has an invalid value.
        """
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        token_addresses: List[TokenAddress] = views.get_token_identifiers(
            chain_state, registry_address)
        channel_state: Optional[NettingChannelState
            ] = views.get_channelstate_for(chain_state=chain_state,
            token_network_registry_address=registry_address, token_address=
            token_address, partner_address=partner_address)
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in channel deposit')
        if not is_binary_address(partner_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in channel deposit'
                )
        if token_address not in token_addresses:
            raise UnknownTokenAddress('Unknown token address')
        if channel_state is None:
            raise NonexistingChannel(
                'No channel with partner_address for the given token')
        try:
            self._raise_for_invalid_channel_timeouts(channel_state.
                settle_timeout, reveal_timeout)
        except InvalidSettleTimeout as ex:
            raise InvalidRevealTimeout(str(ex)) from ex
        self.raiden.set_channel_reveal_timeout(canonical_identifier=
            channel_state.canonical_identifier, reveal_timeout=reveal_timeout)

    def channel_close(self, registry_address, token_address,
        partner_address, retry_timeout=DEFAULT_RETRY_TIMEOUT, coop_settle=True
        ):
        """Close a channel opened with `partner_address` for the given
        `token_address`.

        Race condition, this can fail if channel was closed externally.
        """
        self.channel_batch_close(registry_address=registry_address,
            token_address=token_address, partner_addresses=[partner_address
            ], retry_timeout=retry_timeout, coop_settle=coop_settle)

    def channel_batch_close(self, registry_address, token_address,
        partner_addresses, retry_timeout=DEFAULT_RETRY_TIMEOUT, coop_settle
        =True):
        """Close a channel opened with `partner_address` for the given
        `token_address`.

        Race condition, this can fail if channel was closed externally.
        """
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in channel close')
        if not all(map(is_binary_address, partner_addresses)):
            raise InvalidBinaryAddress(
                'Expected binary address format for partner in channel close')
        valid_tokens: List[TokenAddress] = views.get_token_identifiers(
            chain_state=views.state_from_raiden(self.raiden),
            token_network_registry_address=registry_address)
        if token_address not in valid_tokens:
            raise UnknownTokenAddress('Token address is not known.')
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        channels_to_close: List[NettingChannelState
            ] = views.filter_channels_by_partneraddress(chain_state=
            chain_state, token_network_registry_address=registry_address,
            token_address=token_address, partner_addresses=partner_addresses)
        if coop_settle:
            non_settled_channels: List[NettingChannelState
                ] = self._batch_coop_settle(channels_to_close, retry_timeout)
            if not non_settled_channels:
                return
        close_state_changes: List[StateChange] = [ActionChannelClose(
            canonical_identifier=channel_state.canonical_identifier) for
            channel_state in channels_to_close]
        greenlets = set(self.raiden.handle_state_changes(close_state_changes))
        gevent.joinall(greenlets, raise_error=True)
        channel_ids: List[ChannelID] = [channel_state.identifier for
            channel_state in channels_to_close]
        waiting.wait_for_close(raiden=self.raiden,
            token_network_registry_address=registry_address, token_address=
            token_address, channel_ids=channel_ids, retry_timeout=retry_timeout
            )

    def _batch_coop_settle(self, channels_to_settle, retry_timeout=
        DEFAULT_RETRY_TIMEOUT):
        pfs_proxy = self.raiden.pfs_proxy
        coop_settle_state_changes: List[StateChange] = []
        to_coop_settle: set = set()
        for channel_state in channels_to_settle:
            recipient_address: Address = channel_state.partner_state.address
            try:
                metadata: Any = pfs_proxy.query_address_metadata(
                    recipient_address)
            except ServiceRequestFailed:
                log.error('Partner is offline, coop settle not possible',
                    address=recipient_address)
                continue
            to_coop_settle.add(channel_state.canonical_identifier)
            coop_settle_state_changes.append(ActionChannelCoopSettle(
                canonical_identifier=channel_state.canonical_identifier,
                recipient_metadata=metadata))
        greenlets = set(self.raiden.handle_state_changes(
            coop_settle_state_changes))
        gevent.joinall(greenlets, raise_error=True)
        ids: frozenset = frozenset(ch.canonical_identifier for ch in
            channels_to_settle)
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        channels_to_settle = [ch for ch in views.list_all_channelstate(
            chain_state) if ch.canonical_identifier in ids]
        channels_to_conditions: Dict[Tuple[Address, TokenNetworkAddress,
            ChannelID], Any] = {}
        for channel_state in channels_to_settle:
            if channel_state.canonical_identifier not in to_coop_settle:
                continue
            if channel_state.our_state.initiated_coop_settle is None:
                continue
            expired = waiting.ChannelExpiredCoopSettle(self.raiden,
                channel_state.canonical_identifier)
            success = waiting.ChannelCoopSettleSuccess(self.raiden,
                channel_state.canonical_identifier)
            condition = waiting.Or((expired, success))
            channels_to_conditions[channel_state.canonical_identifier
                ] = condition
        waiting.wait_for_channels(self.raiden, channels_to_conditions,
            retry_timeout=retry_timeout)
        unsuccessful_channels: List[NettingChannelState] = []
        for channel_state in channels_to_settle:
            if channel.get_status(channel_state
                ) is not ChannelState.STATE_SETTLED:
                unsuccessful_channels.append(channel_state)
        return unsuccessful_channels

    def get_channel_list(self, registry_address, token_address=None,
        partner_address=None):
        """Returns a list of channels associated with the optionally given
           `token_address` and/or `partner_address`.

        Args:
            token_address: an optionally provided token address
            partner_address: an optionally provided partner address

        Return:
            A list containing all channels the node participates. Optionally
            filtered by a token address and/or partner address.

        Raises:
            KeyError: An error occurred when the token address is unknown to the node.
        """
        if registry_address and not is_binary_address(registry_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for registry in get_channel_list'
                )
        if token_address and not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in get_channel_list')
        if partner_address:
            if not is_binary_address(partner_address):
                raise InvalidBinaryAddress(
                    'Expected binary address format for partner in get_channel_list'
                    )
            if not token_address:
                raise UnknownTokenAddress(
                    'Provided a partner address but no token address')
        if token_address and partner_address:
            with opentracing.tracer.start_span('get_channelstate_for') as span:
                span.set_tag('token_address', to_checksum_address(
                    token_address))
                span.set_tag('partner_address', to_checksum_address(
                    partner_address))
                channel_state: Optional[NettingChannelState
                    ] = views.get_channelstate_for(chain_state=views.
                    state_from_raiden(self.raiden),
                    token_network_registry_address=registry_address,
                    token_address=token_address, partner_address=
                    partner_address)
            if channel_state:
                result: List[NettingChannelState] = [channel_state]
            else:
                result = []
        elif token_address:
            with opentracing.tracer.start_span(
                'list_channelstate_for_tokennetwork') as span:
                span.set_tag('token_address', to_checksum_address(
                    token_address))
                result: List[NettingChannelState
                    ] = views.list_channelstate_for_tokennetwork(chain_state
                    =views.state_from_raiden(self.raiden),
                    token_network_registry_address=registry_address,
                    token_address=token_address)
        else:
            result: List[NettingChannelState] = views.list_all_channelstate(
                chain_state=views.state_from_raiden(self.raiden))
        return result

    def get_node_network_state(self, node_address):
        """Returns the currently network status of `node_address`."""
        return views.get_node_network_status(chain_state=views.
            state_from_raiden(self.raiden), node_address=node_address)

    def get_tokens_list(self, registry_address):
        """Returns a list of tokens the node knows about"""
        return views.get_token_identifiers(chain_state=views.
            state_from_raiden(self.raiden), token_network_registry_address=
            registry_address)

    def get_token_network_address_for_token_address(self, registry_address,
        token_address):
        return views.get_token_network_address_by_token_address(chain_state
            =views.state_from_raiden(self.raiden),
            token_network_registry_address=registry_address, token_address=
            token_address)

    def transfer_and_wait(self, registry_address, token_address, amount,
        target, identifier=None, transfer_timeout=None, secret=None,
        secrethash=None, lock_timeout=None, route_states=None):
        """Do a transfer with `target` with the given `amount` of `token_address`."""
        with opentracing.tracer.start_span('transfer_and_wait') as span:
            span.set_tag('token_address', to_checksum_address(token_address))
            if isinstance(target, bytes):
                span.set_tag('target', to_checksum_address(target))
            span.set_tag('payment_identifier', identifier)
            payment_status: 'PaymentStatus' = self.transfer_async(
                registry_address=registry_address, token_address=
                token_address, amount=amount, target=target, identifier=
                identifier, secret=secret, secrethash=secrethash,
                lock_timeout=lock_timeout, route_states=route_states)
            payment_status.payment_done.wait(timeout=transfer_timeout)
        return payment_status

    def transfer_async(self, registry_address, token_address, amount,
        target, identifier=None, secret=None, secrethash=None, lock_timeout
        =None, route_states=None):
        current_state: ChainState = views.state_from_raiden(self.raiden)
        token_network_registry_address: TokenNetworkRegistryAddress = (self
            .raiden.default_registry.address)
        if not isinstance(amount, int):
            raise InvalidAmount('Amount not a number')
        if Address(target) == self.address:
            raise SamePeerAddress('Address must be different than ours')
        if amount <= 0:
            raise InvalidAmount('Amount negative')
        if amount > UINT256_MAX:
            raise InvalidAmount('Amount too large')
        if not is_binary_address(token_address):
            raise InvalidBinaryAddress('token address is not valid.')
        if token_address not in views.get_token_identifiers(current_state,
            registry_address):
            raise UnknownTokenAddress('Token address is not known.')
        if not is_binary_address(target):
            raise InvalidBinaryAddress('target address is not valid.')
        valid_tokens: List[TokenAddress] = views.get_token_identifiers(views
            .state_from_raiden(self.raiden), registry_address)
        if token_address not in valid_tokens:
            raise UnknownTokenAddress('Token address is not known.')
        if secret is not None and not isinstance(secret, T_Secret):
            raise InvalidSecret('secret is not valid.')
        if secrethash is not None and not isinstance(secrethash, T_SecretHash):
            raise InvalidSecretHash('secrethash is not valid.')
        if identifier is None:
            identifier = create_default_identifier()
        if identifier <= 0:
            raise InvalidPaymentIdentifier(
                'Payment identifier cannot be 0 or negative')
        if identifier > UINT64_MAX:
            raise InvalidPaymentIdentifier('Payment identifier is too large')
        log.debug('Initiating transfer', initiator=to_checksum_address(self
            .raiden.address), target=to_checksum_address(target), token=
            to_checksum_address(token_address), amount=amount, identifier=
            identifier)
        token_network_address: Optional[TokenNetworkAddress
            ] = views.get_token_network_address_by_token_address(chain_state
            =current_state, token_network_registry_address=
            token_network_registry_address, token_address=token_address)
        if token_network_address is None:
            raise UnknownTokenAddress(
                f'Token {to_checksum_address(token_address)} is not registered with the network {to_checksum_address(registry_address)}.'
                )
        with opentracing.tracer.start_span('mediated_transfer_async'):
            payment_status: 'PaymentStatus' = (self.raiden.
                mediated_transfer_async(token_network_address=
                token_network_address, amount=amount, target=target,
                identifier=identifier, secret=secret, secrethash=secrethash,
                lock_timeout=lock_timeout, route_states=route_states))
        return payment_status

    def get_raiden_events_payment_history_with_timestamps(self,
        registry_address, token_address=None, target_address=None, limit=
        None, offset=None):
        if token_address and not is_binary_address(token_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for token in get_raiden_events_payment_history'
                )
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        token_network_address: Optional[TokenNetworkAddress] = None
        if token_address:
            token_network_address = (views.
                get_token_network_address_by_token_address(chain_state=
                chain_state, token_network_registry_address=
                registry_address, token_address=token_address))
            if not token_network_address:
                raise InvalidTokenAddress(
                    'Token address does not match a Raiden token network')
        if target_address and not is_binary_address(target_address):
            raise InvalidBinaryAddress(
                'Expected binary address format for target_address in get_raiden_events_payment_history'
                )
        assert self.raiden.wal, 'Raiden service has to be started for the API to be usable.'
        event_types: List[str] = [
            'raiden.transfer.events.EventPaymentReceivedSuccess',
            'raiden.transfer.events.EventPaymentSentFailed',
            'raiden.transfer.events.EventPaymentSentSuccess']
        events: List[TimestampedEvent] = (self.raiden.wal.storage.
            get_raiden_events_payment_history_with_timestamps(event_types=
            event_types, limit=limit, offset=offset, token_network_address=
            token_network_address, partner_address=target_address))
        events = [e for e in events if event_filter_for_payments(event=e.
            event, chain_state=chain_state, partner_address=target_address,
            token_address=token_address)]
        return events

    def get_raiden_internal_events_with_timestamps(self, limit=None, offset
        =None):
        assert self.raiden.wal, 'Raiden service has to be started for the API to be usable.'
        return self.raiden.wal.storage.get_events_with_timestamps(limit=
            limit, offset=offset)

    def get_pending_transfers(self, token_address=None, partner_address=None):
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        transfer_tasks: Dict[SecretHash, TransferTask
            ] = views.get_all_transfer_tasks(chain_state)
        channel_id: Optional[ChannelID] = None
        confirmed_block_identifier: BlockIdentifier = chain_state.block_hash
        if token_address is not None:
            token_network: Optional[Any
                ] = self.raiden.default_registry.get_token_network(
                token_address=token_address, block_identifier=
                confirmed_block_identifier)
            if token_network is None:
                raise UnknownTokenAddress(
                    f'Token {to_checksum_address(token_address)} not found.')
            if partner_address is not None:
                partner_channel: Optional[NettingChannelState
                    ] = views.get_channelstate_for(chain_state=chain_state,
                    token_network_registry_address=self.raiden.
                    default_registry.address, token_address=token_address,
                    partner_address=partner_address)
                if not partner_channel:
                    raise ChannelNotFound(
                        'Channel with partner `partner_address not found.`')
                channel_id = partner_channel.identifier
        return transfer_tasks_view(transfer_tasks, token_address, channel_id)

    def set_total_udc_deposit(self, new_total_deposit):
        """Set the `total_deposit` in the UserDeposit contract by sending an on-chain transaction.

        Raises:
            UserDepositNotConfigured: No UserDeposit is configured for the
                Raiden node.
            DepositMismatch: The new `total_deposit` is not higher than the
                previous one.
            DepositOverLimit: Either an overflow happened or the
                `whole_balance_limit` of the UserDeposit contract is exceeded.
            InsufficientFunds: Not enough tokens for the deposit.
            RaidenRecoverableError: The transaction failed for any reason.

        Returns: TransactionHash of the successfully mined transaction.
        """
        user_deposit = self.raiden.default_user_deposit
        if user_deposit is None:
            raise UserDepositNotConfigured(
                'No UserDeposit contract is configured.')
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        confirmed_block_identifier: BlockIdentifier = chain_state.block_hash
        current_total_deposit: TokenAmount = user_deposit.get_total_deposit(
            address=self.address, block_identifier=confirmed_block_identifier)
        deposit_increase: TokenAmount = (new_total_deposit -
            current_total_deposit)
        whole_balance: TokenAmount = user_deposit.whole_balance(
            block_identifier=confirmed_block_identifier)
        whole_balance_limit: TokenAmount = user_deposit.whole_balance_limit(
            block_identifier=confirmed_block_identifier)
        token_address: TokenAddress = user_deposit.token_address(
            block_identifier=confirmed_block_identifier)
        token: Any = self.raiden.proxy_manager.token(token_address,
            block_identifier=confirmed_block_identifier)
        balance: TokenAmount = token.balance_of(address=self.address,
            block_identifier=confirmed_block_identifier)
        if new_total_deposit <= current_total_deposit:
            raise DepositMismatch('Total deposit did not increase.')
        if whole_balance + deposit_increase > UINT256_MAX:
            raise DepositOverLimit('Deposit overflow.')
        if whole_balance + deposit_increase > whole_balance_limit:
            msg: str = (
                f'Deposit of {deposit_increase} would have exceeded the UDC balance limit.'
                )
            raise DepositOverLimit(msg)
        if balance < deposit_increase:
            msg: str = (
                f'Not enough balance to deposit. Available={balance} Needed={deposit_increase}'
                )
            raise InsufficientFunds(msg)
        return user_deposit.approve_and_deposit(beneficiary=self.address,
            total_deposit=new_total_deposit, given_block_identifier=
            confirmed_block_identifier)

    def plan_udc_withdraw(self, amount):
        """Plan a withdraw of `amount` from the UserDeposit contract by sending an on-chain
        transaction.

        Raises:
            UserDepositNotConfigured: No UserDeposit is configured for the
                Raiden node.
            WithdrawMismatch: Withdrawing more than the available balance or
                a zero or negative amount.
            RaidenRecoverableError: The transaction failed for any reason.

        Returns:
            Tuple of the TransactionHash and the BlockNumber at which the
            withdraw is ready.
        """
        user_deposit = self.raiden.default_user_deposit
        if user_deposit is None:
            raise UserDepositNotConfigured(
                'No UserDeposit contract is configured.')
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        confirmed_block_identifier: BlockIdentifier = chain_state.block_hash
        balance: TokenAmount = user_deposit.get_balance(address=self.
            address, block_identifier=confirmed_block_identifier)
        if amount <= 0:
            raise WithdrawMismatch('Withdraw amount must be greater than zero.'
                )
        if amount > balance:
            msg: str = (
                f'The withdraw of {amount} is bigger than the current balance of {balance}.'
                )
            raise WithdrawMismatch(msg)
        return user_deposit.plan_withdraw(amount=amount,
            given_block_identifier=confirmed_block_identifier)

    def withdraw_from_udc(self, amount):
        """Withdraw an `amount` from the UserDeposit contract by sending an on-chain
        transaction. The withdraw has to be planned first with `plan_udc_withdraw`.

        Raises:
            UserDepositNotConfigured: No UserDeposit is configured for the
                Raiden node.
            WithdrawMismatch: Withdrawing more than the planned withdraw amount
                or at an earlier block than the planned withdraw is ready.
            RaidenRecoverableError: The transaction failed for any reason.

        Returns: TransactionHash of the successfully mined transaction.
        """
        user_deposit = self.raiden.default_user_deposit
        if user_deposit is None:
            raise UserDepositNotConfigured(
                'No UserDeposit contract is configured.')
        chain_state: ChainState = views.state_from_raiden(self.raiden)
        confirmed_block_identifier: BlockIdentifier = chain_state.block_hash
        block_number: BlockNumber = chain_state.block_number
        withdraw_plan = user_deposit.get_withdraw_plan(withdrawer_address=
            self.address, block_identifier=confirmed_block_identifier)
        whole_balance: TokenAmount = user_deposit.whole_balance(
            block_identifier=confirmed_block_identifier)
        if amount <= 0:
            raise WithdrawMismatch('Withdraw amount must be greater than zero.'
                )
        if amount > withdraw_plan.withdraw_amount:
            raise WithdrawMismatch('Withdrawing more than planned.')
        if block_number < withdraw_plan.withdraw_block:
            raise WithdrawMismatch(
                f'Withdrawing too early. Planned withdraw at block {withdraw_plan.withdraw_block}. Current block: {block_number}.'
                )
        if whole_balance - amount < 0:
            raise WithdrawMismatch('Whole balance underflow.')
        return user_deposit.withdraw(amount=amount, given_block_identifier=
            confirmed_block_identifier)

    def get_new_notifications(self):
        notifications: List[Any] = list(self.raiden.notifications.values())
        self.raiden.notifications = {}
        return notifications

    def shutdown(self):
        self.raiden.stop()
