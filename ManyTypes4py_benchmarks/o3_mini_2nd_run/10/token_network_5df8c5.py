from collections import defaultdict
from dataclasses import dataclass
import structlog
from eth_utils import encode_hex, is_binary_address, to_canonical_address, to_hex
from gevent.lock import RLock
from web3.exceptions import BadFunctionCallOutput
from raiden.constants import (
    BLOCK_ID_LATEST,
    EMPTY_BALANCE_HASH,
    EMPTY_SIGNATURE,
    LOCKSROOT_OF_NO_LOCKS,
    NULL_ADDRESS_BYTES,
    UINT256_MAX,
    UNLOCK_TX_GAS_LIMIT,
)
from raiden.exceptions import (
    BrokenPreconditionError,
    DepositOverLimit,
    DuplicatedChannelError,
    InvalidChannelID,
    InvalidSettleTimeout,
    RaidenRecoverableError,
    RaidenUnrecoverableError,
    SamePeerAddress,
    WithdrawMismatch,
)
from raiden.network.proxies.metadata import SmartContractMetadata
from raiden.network.proxies.utils import raise_on_call_returned_empty
from raiden.network.rpc.client import JSONRPCClient, check_address_has_code_handle_pruned_block, check_transaction_failure, was_transaction_successfully_mined
from raiden.transfer.channel import compute_locksroot
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.state import PendingLocksState
from raiden.transfer.utils import hash_balance_data
from raiden.utils.formatting import format_block_id, to_checksum_address
from raiden.utils.packing import pack_balance_proof, pack_signed_balance_proof, pack_withdraw
from raiden.utils.signer import recover
from raiden.utils.smart_contracts import safe_gas_limit
from raiden.utils.typing import (
    TYPE_CHECKING,
    AdditionalHash,
    Address,
    Any,
    BalanceHash,
    BlockExpiration,
    BlockHash,
    BlockIdentifier,
    BlockNumber,
    ChainID,
    ChannelID,
    Dict,
    LockedAmount,
    Locksroot,
    NamedTuple,
    Nonce,
    Optional,
    Signature,
    T_ChannelID,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    TransactionHash,
    WithdrawAmount,
    typecheck,
)
from raiden_contracts.constants import (
    CONTRACT_TOKEN_NETWORK,
    ChannelInfoIndex,
    ChannelState,
    MessageTypeId,
    ParticipantInfoIndex,
)
from raiden_contracts.contract_manager import ContractManager

if TYPE_CHECKING:
    from raiden.network.proxies.proxy_manager import ProxyManager

log = structlog.get_logger(__name__)


def raise_if_invalid_address_pair(address1: bytes, address2: bytes) -> None:
    msg = 'The null address is not allowed as a channel participant.'
    assert NULL_ADDRESS_BYTES not in (address1, address2), msg
    msg = 'Addresses must be in binary'
    assert is_binary_address(address1) and is_binary_address(address2), msg
    if address1 == address2:
        raise SamePeerAddress('Using the same address for both participants is forbiden.')


class WithdrawInput(NamedTuple):
    pass


class ChannelData(NamedTuple):
    pass


class ParticipantDetails(NamedTuple):
    pass


class ParticipantsDetails(NamedTuple):
    pass


class ChannelDetails(NamedTuple):
    pass


class NewNettingChannelDetails(NamedTuple):
    pass


@dataclass
class TokenNetworkMetadata(SmartContractMetadata):
    pass


class TokenNetwork:
    def __init__(
        self,
        jsonrpc_client: JSONRPCClient,
        contract_manager: ContractManager,
        proxy_manager: "ProxyManager",
        metadata: TokenNetworkMetadata,
        block_identifier: BlockIdentifier,
    ) -> None:
        if not is_binary_address(metadata.address):
            raise ValueError('Expected binary address format for token nework')
        check_address_has_code_handle_pruned_block(
            client=jsonrpc_client,
            address=Address(metadata.address),
            contract_name=CONTRACT_TOKEN_NETWORK,
            given_block_identifier=block_identifier,
        )
        self.contract_manager = contract_manager
        proxy = jsonrpc_client.new_contract_proxy(abi=metadata.abi, contract_address=Address(metadata.address))
        self._chain_id: ChainID = proxy.functions.chain_id().call()
        self._token_address: TokenAddress = TokenAddress(to_canonical_address(proxy.functions.token().call()))
        self.address: TokenNetworkAddress = TokenNetworkAddress(metadata.address)
        self.proxy = proxy
        self.client: JSONRPCClient = jsonrpc_client
        self.node_address: Address = self.client.address
        self.metadata: TokenNetworkMetadata = metadata
        self.token = proxy_manager.token(token_address=self.token_address(), block_identifier=block_identifier)
        self.channel_operations_lock: Dict[Address, RLock] = defaultdict(RLock)
        self.opening_channels_count: int = 0

    def chain_id(self) -> ChainID:
        """Return the token of this manager."""
        return self._chain_id

    def token_address(self) -> TokenAddress:
        """Return the token of this manager."""
        return self._token_address

    def channel_participant_deposit_limit(self, block_identifier: BlockIdentifier) -> TokenAmount:
        """Return the deposit limit of a channel participant."""
        return TokenAmount(self.proxy.functions.channel_participant_deposit_limit().call(block_identifier=block_identifier))

    def token_network_deposit_limit(self, block_identifier: BlockIdentifier) -> TokenAmount:
        """Return the token of this manager."""
        return TokenAmount(self.proxy.functions.token_network_deposit_limit().call(block_identifier=block_identifier))

    def safety_deprecation_switch(self, block_identifier: BlockIdentifier) -> Any:
        return self.proxy.functions.safety_deprecation_switch().call(block_identifier=block_identifier)

    def new_netting_channel(
        self, partner: Address, settle_timeout: int, given_block_identifier: BlockIdentifier
    ) -> NewNettingChannelDetails:
        """Creates a new channel in the TokenNetwork contract.

        Args:
            partner: The peer to open the channel with.
            settle_timeout: The settle timeout to use for this channel.
            given_block_identifier: The block identifier of the state change that
                                    prompted this proxy action

        Returns:
            The ChannelID of the new netting channel.
        """
        raise_if_invalid_address_pair(self.node_address, partner)
        timeout_min: int = self.settlement_timeout_min()
        timeout_max: int = self.settlement_timeout_max()
        invalid_timeout: bool = settle_timeout < timeout_min or settle_timeout > timeout_max
        if invalid_timeout:
            msg = f'settle_timeout must be in range [{timeout_min}, {timeout_max}], is {settle_timeout}'
            raise InvalidSettleTimeout(msg)
        with self.channel_operations_lock[partner]:
            try:
                existing_channel_identifier: Optional[T_ChannelID] = self.get_channel_identifier_or_none(
                    participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier
                )
                network_total_deposit: TokenAmount = self.token.balance_of(address=Address(self.address), block_identifier=given_block_identifier)
                limit: TokenAmount = self.token_network_deposit_limit(block_identifier=given_block_identifier)
                _ = self.safety_deprecation_switch(given_block_identifier)
            except ValueError:
                pass
            except BadFunctionCallOutput:
                raise_on_call_returned_empty(given_block_identifier)
            else:
                if existing_channel_identifier is not None:
                    raise BrokenPreconditionError('A channel with the given partner address already exists.')
                if network_total_deposit >= limit:
                    raise BrokenPreconditionError('Cannot open another channel, token network deposit limit reached.')
                if self.safety_deprecation_switch(given_block_identifier):
                    raise BrokenPreconditionError('This token network is deprecated.')
            self.opening_channels_count += 1
            try:
                netting_channel_details: NewNettingChannelDetails = self._new_netting_channel(partner, settle_timeout)
            finally:
                self.opening_channels_count -= 1
            return netting_channel_details

    def _new_netting_channel(self, partner: Address, settle_timeout: int) -> NewNettingChannelDetails:
        estimated_transaction = self.client.estimate_gas(
            self.proxy,
            'openChannel',
            extra_log_details={},
            participant1=self.node_address,
            participant2=partner,
            settle_timeout=settle_timeout,
        )
        if estimated_transaction is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: BlockNumber = failed_at['number']
            self.client.check_for_insufficient_eth(
                transaction_name='openChannel',
                transaction_executed=False,
                required_gas=self.metadata.gas_measurements['TokenNetwork.openChannel'],
                block_identifier=failed_at_blocknumber,
            )
            existing_channel_identifier = self.get_channel_identifier_or_none(
                participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash
            )
            if existing_channel_identifier is not None:
                raise DuplicatedChannelError('Channel with given partner address already exists')
            network_total_deposit = self.token.balance_of(address=Address(self.address), block_identifier=failed_at_blockhash)
            limit = self.token_network_deposit_limit(block_identifier=failed_at_blockhash)
            if network_total_deposit >= limit:
                raise DepositOverLimit('Could open another channel, token network deposit limit has been reached.')
            if self.safety_deprecation_switch(block_identifier=failed_at_blockhash):
                raise RaidenRecoverableError('This token network is deprecated.')
            raise RaidenRecoverableError(
                f'Creating a new channel will fail - Gas estimation failed for unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )
        else:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.openChannel']
            )
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            receipt = transaction_mined.receipt
            if not was_transaction_successfully_mined(transaction_mined):
                failed_at_blockhash = encode_hex(receipt['blockHash'])
                existing_channel_identifier = self.get_channel_identifier_or_none(
                    participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash
                )
                if existing_channel_identifier is not None:
                    raise DuplicatedChannelError('Channel with given partner address already exists')
                network_total_deposit = self.token.balance_of(address=Address(self.address), block_identifier=failed_at_blockhash)
                limit = self.token_network_deposit_limit(block_identifier=failed_at_blockhash)
                if network_total_deposit >= limit:
                    raise DepositOverLimit('Could open another channel, token network deposit limit has been reached.')
                if self.safety_deprecation_switch(block_identifier=failed_at_blockhash):
                    raise RaidenRecoverableError('This token network is deprecated.')
                raise RaidenRecoverableError('Creating new channel failed.')
            channel_identifier = self._detail_channel(
                participant1=self.node_address,
                participant2=partner,
                block_identifier=encode_hex(receipt['blockHash']),
            ).channel_identifier
            return NewNettingChannelDetails(
                channel_identifier=channel_identifier,
                block_hash=BlockHash(receipt['blockHash']),
                block_number=BlockNumber(receipt['blockNumber']),
                transaction_hash=TransactionHash(transaction_mined.transaction_hash),
            )

    def get_channel_identifier(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier) -> T_ChannelID:
        """Return the channel identifier for the opened channel among
        `(participant1, participant2)`.

        Raises:
            RaidenRecoverableError: If there is not open channel among
                `(participant1, participant2)`. Note this is the case even if
                there is a channel in a settled state.
            BadFunctionCallOutput: If the `block_identifier` points to a block
                prior to the deployment of the TokenNetwork.
            SamePeerAddress: If an both addresses are equal.
        """
        raise_if_invalid_address_pair(participant1, participant2)
        channel_identifier: T_ChannelID = self.proxy.functions.getChannelIdentifier(
            participant=participant1, partner=participant2
        ).call(block_identifier=block_identifier)
        if channel_identifier == 0:
            msg = (
                f'getChannelIdentifier returned 0, meaning no channel currently exists between '
                f'{to_checksum_address(participant1)} and {to_checksum_address(participant2)}'
            )
            raise RaidenRecoverableError(msg)
        return channel_identifier

    def get_channel_identifier_or_none(
        self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier
    ) -> Optional[T_ChannelID]:
        """Returns the channel identifier if an open channel exists, else None."""
        try:
            return self.get_channel_identifier(participant1=participant1, participant2=participant2, block_identifier=block_identifier)
        except RaidenRecoverableError:
            return None

    def _detail_participant(
        self, channel_identifier: T_ChannelID, detail_for: Address, partner: Address, block_identifier: BlockIdentifier
    ) -> ParticipantDetails:
        """Returns a dictionary with the channel participant information."""
        raise_if_invalid_address_pair(detail_for, partner)
        data = self.proxy.functions.getChannelParticipantInfo(
            channel_identifier=channel_identifier, participant=detail_for, partner=partner
        ).call(block_identifier=block_identifier)
        return ParticipantDetails(
            address=detail_for,
            deposit=data[ParticipantInfoIndex.DEPOSIT],
            withdrawn=data[ParticipantInfoIndex.WITHDRAWN],
            is_closer=data[ParticipantInfoIndex.IS_CLOSER],
            balance_hash=data[ParticipantInfoIndex.BALANCE_HASH],
            nonce=data[ParticipantInfoIndex.NONCE],
            locksroot=data[ParticipantInfoIndex.LOCKSROOT],
            locked_amount=data[ParticipantInfoIndex.LOCKED_AMOUNT],
        )

    def _detail_channel(
        self,
        participant1: Address,
        participant2: Address,
        block_identifier: BlockIdentifier,
        channel_identifier: Optional[T_ChannelID] = None,
    ) -> ChannelData:
        """Returns a ChannelData instance with the channel specific information.

        If no specific channel_identifier is given then it tries to see if there
        is a currently open channel and uses that identifier.
        """
        raise_if_invalid_address_pair(participant1, participant2)
        if channel_identifier is None:
            channel_identifier = self.get_channel_identifier(
                participant1=participant1, participant2=participant2, block_identifier=block_identifier
            )
        elif not isinstance(channel_identifier, T_ChannelID):
            raise InvalidChannelID('channel_identifier must be of type T_ChannelID')
        elif channel_identifier <= 0 or channel_identifier > UINT256_MAX:
            raise InvalidChannelID('channel_identifier must be larger then 0 and smaller then uint256')
        channel_data = self.proxy.functions.getChannelInfo(
            channel_identifier=channel_identifier, participant1=participant1, participant2=participant2
        ).call(block_identifier=block_identifier)
        return ChannelData(
            channel_identifier=channel_identifier,
            settle_block_number=channel_data[ChannelInfoIndex.SETTLE_BLOCK],
            state=channel_data[ChannelInfoIndex.STATE],
        )

    def detail_participants(
        self,
        participant1: Address,
        participant2: Address,
        block_identifier: BlockIdentifier,
        channel_identifier: Optional[T_ChannelID] = None,
    ) -> ParticipantsDetails:
        """Returns a ParticipantsDetails instance with the participants'
            channel information.

        Note:
            For now one of the participants has to be the node_address
        """
        if self.node_address not in (participant1, participant2):
            raise ValueError('One participant must be the node address')
        if self.node_address == participant2:
            participant1, participant2 = (participant2, participant1)
        if channel_identifier is None:
            channel_identifier = self.get_channel_identifier(
                participant1=participant1, participant2=participant2, block_identifier=block_identifier
            )
        elif not isinstance(channel_identifier, T_ChannelID):
            raise InvalidChannelID('channel_identifier must be of type T_ChannelID')
        elif channel_identifier <= 0 or channel_identifier > UINT256_MAX:
            raise InvalidChannelID('channel_identifier must be larger then 0 and smaller then uint256')
        our_data = self._detail_participant(
            channel_identifier=channel_identifier, detail_for=participant1, partner=participant2, block_identifier=block_identifier
        )
        partner_data = self._detail_participant(
            channel_identifier=channel_identifier, detail_for=participant2, partner=participant1, block_identifier=block_identifier
        )
        return ParticipantsDetails(our_details=our_data, partner_details=partner_data)

    def detail(
        self,
        participant1: Address,
        participant2: Address,
        block_identifier: BlockIdentifier,
        channel_identifier: Optional[T_ChannelID] = None,
    ) -> ChannelDetails:
        """Returns a ChannelDetails instance with all the details of the
            channel and the channel participants.

        Note:
            For now one of the participants has to be the node_address
        """
        if self.node_address not in (participant1, participant2):
            raise ValueError('One participant must be the node address')
        if self.node_address == participant2:
            participant1, participant2 = (participant2, participant1)
        channel_data = self._detail_channel(
            participant1=participant1,
            participant2=participant2,
            block_identifier=block_identifier,
            channel_identifier=channel_identifier,
        )
        participants_data = self.detail_participants(
            participant1=participant1, participant2=participant2, block_identifier=block_identifier, channel_identifier=channel_data.channel_identifier
        )
        chain_id: ChainID = self.chain_id()
        return ChannelDetails(chain_id=chain_id, token_address=self.token_address(), channel_data=channel_data, participants_data=participants_data)

    def settlement_timeout_min(self) -> int:
        """Returns the minimal settlement timeout for the token network."""
        return self.proxy.functions.settlement_timeout_min().call()

    def settlement_timeout_max(self) -> int:
        """Returns the maximal settlement timeout for the token network."""
        return self.proxy.functions.settlement_timeout_max().call()

    def channel_is_opened(
        self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: T_ChannelID
    ) -> bool:
        """Returns true if the channel is in an open state, false otherwise."""
        try:
            channel_data = self._detail_channel(
                participant1=participant1, participant2=participant2, block_identifier=block_identifier, channel_identifier=channel_identifier
            )
        except RaidenRecoverableError:
            return False
        return channel_data.state == ChannelState.OPENED

    def channel_is_closed(
        self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: T_ChannelID
    ) -> bool:
        """Returns true if the channel is in a closed state, false otherwise."""
        try:
            channel_data = self._detail_channel(
                participant1=participant1, participant2=participant2, block_identifier=block_identifier, channel_identifier=channel_identifier
            )
        except RaidenRecoverableError:
            return False
        return channel_data.state == ChannelState.CLOSED

    def channel_is_settled(
        self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: T_ChannelID
    ) -> bool:
        """Returns true if the channel is in a settled state, false otherwise."""
        try:
            channel_data = self._detail_channel(
                participant1=participant1, participant2=participant2, block_identifier=block_identifier, channel_identifier=channel_identifier
            )
        except RaidenRecoverableError:
            return False
        return channel_data.state >= ChannelState.SETTLED

    def can_transfer(
        self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: T_ChannelID
    ) -> bool:
        """Returns True if the channel is opened and the node has deposit in
        it.

        Note: Having a deposit does not imply having a balance for off-chain
        transfers.
        """
        opened: bool = self.channel_is_opened(
            participant1=participant1, participant2=participant2, block_identifier=block_identifier, channel_identifier=channel_identifier
        )
        if not opened:
            return False
        deposit = self._detail_participant(
            channel_identifier=channel_identifier, detail_for=participant1, partner=participant2, block_identifier=block_identifier
        ).deposit
        return deposit > 0

    def approve_and_set_total_deposit(
        self, given_block_identifier: BlockIdentifier, channel_identifier: T_ChannelID, total_deposit: int, partner: Address
    ) -> TransactionHash:
        """Set channel's total deposit.
        Raises:
            DepositMismatch: If the new request total deposit is lower than the
                existing total deposit on-chain for the `given_block_identifier`.
            RaidenRecoverableError: If the channel was closed meanwhile the
                deposit was in transit.
            RaidenUnrecoverableError: If the transaction was successful and the
                deposit_amount is not as large as the requested value.
            RuntimeError: If the token address is empty.
            ValueError: If an argument is of the invalid type.
        """
        typecheck(total_deposit, int)
        if total_deposit <= 0 and total_deposit > UINT256_MAX:
            msg = f'Total deposit {total_deposit} is not in range [1, {UINT256_MAX}]'
            raise BrokenPreconditionError(msg)
        with self.channel_operations_lock[partner]:
            try:
                queried_channel_identifier: Optional[T_ChannelID] = self.get_channel_identifier_or_none(
                    participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier
                )
                channel_onchain_detail = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier, channel_identifier=channel_identifier
                )
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=given_block_identifier
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=given_block_identifier
                )
                current_balance = self.token.balance_of(address=self.node_address, block_identifier=given_block_identifier)
                safety_deprecation_switch = self.safety_deprecation_switch(block_identifier=given_block_identifier)
                token_network_deposit_limit = self.token_network_deposit_limit(block_identifier=given_block_identifier)
                channel_participant_deposit_limit = self.channel_participant_deposit_limit(block_identifier=given_block_identifier)
                network_total_deposit = self.token.balance_of(address=Address(self.address), block_identifier=given_block_identifier)
            except ValueError:
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=BLOCK_ID_LATEST
                )
            except BadFunctionCallOutput:
                raise_on_call_returned_empty(given_block_identifier)
            else:
                if queried_channel_identifier != channel_identifier:
                    msg = (
                        f'There is a channel open between {to_checksum_address(self.node_address)} and {to_checksum_address(partner)}. '
                        f'However the channel id on-chain {queried_channel_identifier} and the provided id {channel_identifier} do not match.'
                    )
                    raise BrokenPreconditionError(msg)
                if safety_deprecation_switch:
                    msg = 'This token_network has been deprecated.'
                    raise BrokenPreconditionError(msg)
                if channel_onchain_detail.state != ChannelState.OPENED:
                    msg = (
                        f'The channel was not opened at the provided block ({format_block_id(given_block_identifier)}). '
                        f'This call should never have been attempted.'
                    )
                    raise BrokenPreconditionError(msg)
                amount_to_deposit = total_deposit - our_details.deposit
                if total_deposit <= our_details.deposit:
                    msg = (
                        f'Current total deposit ({our_details.deposit}) is already larger than the requested total deposit amount ({total_deposit})'
                    )
                    raise BrokenPreconditionError(msg)
                total_channel_deposit = total_deposit + partner_details.deposit
                if total_channel_deposit > UINT256_MAX:
                    raise BrokenPreconditionError('Deposit overflow')
                if total_deposit > channel_participant_deposit_limit:
                    msg = f'Deposit of {total_deposit} is larger than the channel participant deposit limit'
                    raise BrokenPreconditionError(msg)
                if network_total_deposit + amount_to_deposit > token_network_deposit_limit:
                    msg = f'Deposit of {amount_to_deposit} will have exceeded the token network deposit limit.'
                    raise BrokenPreconditionError(msg)
                if current_balance < amount_to_deposit:
                    msg = (
                        f'new_total_deposit - previous_total_deposit =  {amount_to_deposit} can not be larger than the available balance {current_balance}, '
                        f'for token at address {to_checksum_address(self.token.address)}'
                    )
                    raise BrokenPreconditionError(msg)
            log_details = {'previous_total_deposit': our_details.deposit, 'given_block_identifier': format_block_id(given_block_identifier)}
            return self._approve_and_set_total_deposit(
                channel_identifier=channel_identifier,
                total_deposit=total_deposit,
                previous_total_deposit=our_details.deposit,
                partner=partner,
                log_details=log_details,
            )

    def _approve_and_set_total_deposit(
        self, channel_identifier: T_ChannelID, total_deposit: int, partner: Address, previous_total_deposit: int, log_details: Any
    ) -> TransactionHash:
        amount_to_deposit: TokenAmount = TokenAmount(total_deposit - previous_total_deposit)
        estimated_transaction = None
        with self.token.token_lock:
            allowance: TokenAmount = TokenAmount(amount_to_deposit + 1)
            self.token.approve(allowed_address=Address(self.address), allowance=allowance)
            estimated_transaction = self.client.estimate_gas(
                self.proxy,
                'setTotalDeposit',
                extra_log_details=log_details,
                channel_identifier=channel_identifier,
                participant=self.node_address,
                total_deposit=total_deposit,
                partner=partner,
            )
        if estimated_transaction is not None:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.setTotalDeposit']
            )
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                receipt = transaction_mined.receipt
                failed_at_blockhash = encode_hex(receipt['blockHash'])
                failed_at_blocknumber: BlockNumber = receipt['blockNumber']
                check_transaction_failure(transaction_mined, self.client)
                safety_deprecation_switch = self.safety_deprecation_switch(block_identifier=failed_at_blockhash)
                if safety_deprecation_switch:
                    msg = 'This token_network has been deprecated.'
                    raise RaidenRecoverableError(msg)
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
                )
                channel_data = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
                )
                if channel_data.state == ChannelState.CLOSED:
                    msg = 'Deposit failed because the channel was closed meanwhile'
                    raise RaidenRecoverableError(msg)
                if channel_data.state == ChannelState.SETTLED:
                    msg = 'Deposit failed because the channel was settled meanwhile'
                    raise RaidenRecoverableError(msg)
                if channel_data.state == ChannelState.REMOVED:
                    msg = 'Deposit failed because the channel was settled and unlocked meanwhile'
                    raise RaidenRecoverableError(msg)
                deposit_amount = total_deposit - our_details.deposit
                total_channel_deposit = total_deposit + partner_details.deposit
                if total_channel_deposit > UINT256_MAX:
                    raise RaidenRecoverableError('Deposit overflow')
                total_deposit_done = our_details.deposit >= total_deposit
                if total_deposit_done:
                    raise RaidenRecoverableError('Requested total deposit was already performed')
                token_network_deposit_limit = self.token_network_deposit_limit(block_identifier=receipt['blockHash'])
                network_total_deposit = self.token.balance_of(address=Address(self.address), block_identifier=receipt['blockHash'])
                if network_total_deposit + deposit_amount > token_network_deposit_limit:
                    msg = f'Deposit of {deposit_amount} would have exceeded the token network deposit limit.'
                    raise RaidenRecoverableError(msg)
                channel_participant_deposit_limit = self.channel_participant_deposit_limit(block_identifier=receipt['blockHash'])
                if total_deposit > channel_participant_deposit_limit:
                    msg = f'Deposit of {total_deposit} is larger than the channel participant deposit limit'
                    raise RaidenRecoverableError(msg)
                has_sufficient_balance = self.token.balance_of(self.node_address, failed_at_blocknumber) < amount_to_deposit
                if not has_sufficient_balance:
                    raise RaidenRecoverableError('The account does not have enough balance to complete the deposit')
                allowance = self.token.allowance(owner=self.node_address, spender=Address(self.address), block_identifier=failed_at_blockhash)
                if allowance < amount_to_deposit:
                    msg = f'The allowance of the {amount_to_deposit} deposit changed. Check concurrent deposits for the same token network but different proxies.'
                    raise RaidenRecoverableError(msg)
                latest_deposit = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
                ).deposit
                if latest_deposit < total_deposit:
                    raise RaidenRecoverableError('The tokens were not transferred')
                raise RaidenRecoverableError('Unlocked failed for an unknown reason')
            else:
                return transaction_mined.transaction_hash
        else:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: BlockNumber = failed_at['number']
            self.client.check_for_insufficient_eth(
                transaction_name='setTotalDeposit',
                transaction_executed=False,
                required_gas=self.metadata.gas_measurements['TokenNetwork.setTotalDeposit'],
                block_identifier=failed_at_blocknumber,
            )
            safety_deprecation_switch = self.safety_deprecation_switch(block_identifier=failed_at_blockhash)
            if safety_deprecation_switch:
                msg = 'This token_network has been deprecated.'
                raise RaidenRecoverableError(msg)
            allowance = self.token.allowance(owner=self.node_address, spender=Address(self.address), block_identifier=failed_at_blockhash)
            has_sufficient_balance = self.token.balance_of(self.node_address, failed_at_blocknumber) < amount_to_deposit
            if allowance < amount_to_deposit:
                msg = 'The allowance is insufficient. Check concurrent deposits for the same token network but different proxies.'
                raise RaidenRecoverableError(msg)
            if has_sufficient_balance:
                msg = 'The address doesnt have enough tokens'
                raise RaidenRecoverableError(msg)
            queried_channel_identifier = self.get_channel_identifier_or_none(
                participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash
            )
            our_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
            )
            partner_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=failed_at_blockhash
            )
            channel_data = self._detail_channel(
                participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
            )
            token_network_deposit_limit = self.token_network_deposit_limit(block_identifier=failed_at_blockhash)
            channel_participant_deposit_limit = self.channel_participant_deposit_limit(block_identifier=failed_at_blockhash)
            total_channel_deposit = total_deposit + partner_details.deposit
            network_total_deposit = self.token.balance_of(Address(self.address), failed_at_blocknumber)
            is_invalid_channel_id = channel_data.state in (ChannelState.OPENED, ChannelState.CLOSED) and queried_channel_identifier != channel_identifier
            if is_invalid_channel_id:
                msg = (
                    f'There is an open channel with the id {channel_identifier}. However addresses '
                    f'{to_checksum_address(self.node_address)} and {to_checksum_address(partner)} are not participants of that channel. '
                    f'The correct id is {queried_channel_identifier}.'
                )
                raise RaidenUnrecoverableError(msg)
            if channel_data.state == ChannelState.CLOSED:
                msg = 'Deposit was prohibited because the channel is closed'
                raise RaidenRecoverableError(msg)
            if channel_data.state == ChannelState.SETTLED:
                msg = 'Deposit was prohibited because the channel is settled'
                raise RaidenRecoverableError(msg)
            if channel_data.state == ChannelState.REMOVED:
                msg = 'Deposit was prohibited because the channel is settled and unlocked'
                raise RaidenRecoverableError(msg)
            if our_details.deposit >= total_deposit:
                msg = 'Attempted deposit has already been done'
                raise RaidenRecoverableError(msg)
            if channel_data.state == ChannelState.NONEXISTENT:
                msg = f'Channel between participant {to_checksum_address(self.node_address)} and {to_checksum_address(partner)} does not exist'
                raise RaidenUnrecoverableError(msg)
            if total_channel_deposit >= UINT256_MAX:
                raise RaidenRecoverableError('Deposit overflow')
            if total_deposit > channel_participant_deposit_limit:
                msg = f'Deposit of {total_deposit} exceeded the channel participant deposit limit'
                raise RaidenRecoverableError(msg)
            if network_total_deposit + amount_to_deposit > token_network_deposit_limit:
                msg = f'Deposit of {amount_to_deposit} exceeded the token network deposit limit.'
                raise RaidenRecoverableError(msg)
            raise RaidenRecoverableError(
                f'Deposit gas estimatation failed for unknown reasons. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )

    def set_total_withdraw(
        self, given_block_identifier: BlockIdentifier, channel_identifier: T_ChannelID, partner: Address, withdraw_input: WithdrawInput
    ) -> TransactionHash:
        """Set total token withdraw in the channel to total_withdraw.
        Raises:
            ValueError: If provided total_withdraw is not an integer value.
        """
        total_withdraw: int = withdraw_input.total_withdraw  # type: ignore
        initiator: Address = withdraw_input.initiator  # type: ignore
        partner_signature: Signature = withdraw_input.partner_signature  # type: ignore
        initiator_signature: Signature = withdraw_input.initiator_signature  # type: ignore
        expiration_block: int = withdraw_input.expiration_block  # type: ignore
        if not isinstance(total_withdraw, int):
            raise ValueError('total_withdraw needs to be an integer number.')
        if total_withdraw <= 0:
            raise ValueError('total_withdraw should be larger than zero.')
        with self.channel_operations_lock[partner]:
            try:
                channel_onchain_detail = self._detail_channel(
                    participant1=initiator, participant2=partner, block_identifier=given_block_identifier, channel_identifier=channel_identifier
                )
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=initiator, partner=partner, block_identifier=given_block_identifier
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=initiator, block_identifier=given_block_identifier
                )
                given_block_number: int = self.client.get_block(given_block_identifier)['number']
            except ValueError:
                pass
            except BadFunctionCallOutput:
                raise_on_call_returned_empty(given_block_identifier)
            else:
                if channel_onchain_detail.state != ChannelState.OPENED:
                    msg = (
                        f'The channel was not opened at the provided block ({format_block_id(given_block_identifier)}). '
                        f'This call should never have been attempted.'
                    )
                    raise RaidenUnrecoverableError(msg)
                if our_details.withdrawn >= total_withdraw:
                    msg = f'The provided total_withdraw amount on-chain is {our_details.withdrawn}. Requested total withdraw {total_withdraw} did not increase.'
                    raise WithdrawMismatch(msg)
                total_channel_deposit = our_details.deposit + partner_details.deposit
                total_channel_withdraw = total_withdraw + partner_details.withdrawn
                if total_channel_withdraw > total_channel_deposit:
                    msg = f'The total channel withdraw amount {total_channel_withdraw} is larger than the total channel deposit of {total_channel_deposit}.'
                    raise WithdrawMismatch(msg)
                if expiration_block <= given_block_number:
                    msg = f'The current block number {given_block_number} is already at expiration block {expiration_block} or later.'
                    raise BrokenPreconditionError(msg)
                if initiator_signature == EMPTY_SIGNATURE:
                    msg = 'set_total_withdraw requires a valid participant signature'
                    raise RaidenUnrecoverableError(msg)
                if partner_signature == EMPTY_SIGNATURE:
                    msg = 'set_total_withdraw requires a valid partner signature'
                    raise RaidenUnrecoverableError(msg)
                canonical_identifier = CanonicalIdentifier(
                    chain_identifier=self.proxy.functions.chain_id().call(),
                    token_network_address=self.address,
                    channel_identifier=channel_identifier,
                )
                participant_signed_data = pack_withdraw(
                    participant=initiator, total_withdraw=total_withdraw, canonical_identifier=canonical_identifier, expiration_block=expiration_block
                )
                try:
                    participant_recovered_address = recover(data=participant_signed_data, signature=initiator_signature)
                except Exception:
                    raise RaidenUnrecoverableError("Couldn't verify the initiator withdraw signature")
                else:
                    if participant_recovered_address != initiator:
                        raise RaidenUnrecoverableError('Invalid withdraw initiator signature')
                partner_signed_data = pack_withdraw(
                    participant=initiator, total_withdraw=total_withdraw, canonical_identifier=canonical_identifier, expiration_block=expiration_block
                )
                try:
                    partner_recovered_address = recover(data=partner_signed_data, signature=partner_signature)
                except Exception:
                    raise RaidenUnrecoverableError("Couldn't verify the partner withdraw signature")
                else:
                    if partner_recovered_address != partner:
                        raise RaidenUnrecoverableError('Invalid withdraw partner signature')
            log_details = {'given_block_identifier': format_block_id(given_block_identifier)}
            return self._set_total_withdraw(
                channel_identifier=channel_identifier,
                total_withdraw=total_withdraw,
                expiration_block=expiration_block,
                participant=initiator,
                partner=partner,
                partner_signature=partner_signature,
                participant_signature=initiator_signature,
                log_details=log_details,
            )

    def _set_total_withdraw(
        self,
        channel_identifier: T_ChannelID,
        total_withdraw: int,
        expiration_block: int,
        participant: Address,
        partner: Address,
        partner_signature: Signature,
        participant_signature: Signature,
        log_details: Any,
    ) -> TransactionHash:
        estimated_transaction = self.client.estimate_gas(
            self.proxy,
            'setTotalWithdraw',
            extra_log_details=log_details,
            channel_identifier=channel_identifier,
            participant=participant,
            total_withdraw=total_withdraw,
            expiration_block=expiration_block,
            partner_signature=partner_signature,
            participant_signature=participant_signature,
        )
        if estimated_transaction is not None:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.setTotalWithdraw']
            )
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                receipt = transaction_mined.receipt
                failed_at_blockhash = encode_hex(receipt['blockHash'])
                failed_at_blocknumber: int = receipt['blockNumber']
                check_transaction_failure(transaction_mined, self.client)
                detail = self._detail_channel(
                    participant1=participant, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
                )
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=participant, partner=partner, block_identifier=failed_at_blockhash
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=participant, block_identifier=failed_at_blockhash
                )
                total_withdraw_done = our_details.withdrawn >= total_withdraw
                if total_withdraw_done:
                    raise RaidenRecoverableError('Requested total withdraw was already performed')
                if detail.state > ChannelState.OPENED:
                    msg = f'setTotalWithdraw failed because the channel closed before the transaction was mined. current_state={detail.state}'
                    raise RaidenRecoverableError(msg)
                if detail.state < ChannelState.OPENED:
                    msg = f'setTotalWithdraw failed because the channel never existed. current_state={detail.state}'
                    raise RaidenUnrecoverableError(msg)
                if expiration_block <= failed_at_blocknumber:
                    msg = f'setTotalWithdraw failed because the transaction was mined after the withdraw expired expiration_block={expiration_block} transation_mined_at={failed_at_blocknumber}'
                    raise RaidenRecoverableError(msg)
                total_channel_deposit = our_details.deposit + partner_details.deposit
                total_channel_withdraw = total_withdraw + partner_details.withdrawn
                if total_channel_withdraw > total_channel_deposit:
                    msg = f'The total channel withdraw amount {total_channel_withdraw} became larger than the total channel deposit of {total_channel_deposit}.'
                    raise WithdrawMismatch(msg)
                raise RaidenUnrecoverableError('SetTotalwithdraw failed for an unknown reason')
            else:
                return transaction_mined.transaction_hash
        else:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: int = failed_at['number']
            self.client.check_for_insufficient_eth(
                transaction_name='total_withdraw',
                transaction_executed=False,
                required_gas=self.metadata.gas_measurements['TokenNetwork.setTotalWithdraw'],
                block_identifier=failed_at_blocknumber,
            )
            detail = self._detail_channel(
                participant1=participant, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
            )
            our_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=participant, partner=partner, block_identifier=failed_at_blockhash
            )
            if detail.state > ChannelState.OPENED:
                msg = f'cannot call setTotalWithdraw on a channel that is not open. current_state={detail.state}'
                raise RaidenRecoverableError(msg)
            if detail.state < ChannelState.OPENED:
                msg = f'cannot call setTotalWithdraw on a channel that does not exist. current_state={detail.state}'
                raise RaidenUnrecoverableError(msg)
            if expiration_block <= failed_at_blocknumber:
                msg = (
                    f'setTotalWithdraw would have failed because current block has already reached the withdraw expiration '
                    f'expiration_block={expiration_block} transation_checked_at={failed_at_blocknumber}'
                )
                raise RaidenRecoverableError(msg)
            total_withdraw_done = our_details.withdrawn >= total_withdraw
            if total_withdraw_done:
                raise RaidenRecoverableError('Requested total withdraw was already performed')
            raise RaidenUnrecoverableError(
                f'setTotalWithdraw gas estimation failed for an unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )

    def coop_settle(
        self, channel_identifier: T_ChannelID, withdraw_initiator: Any, withdraw_partner: Any, given_block_identifier: BlockIdentifier
    ) -> TransactionHash:
        log_details = {'given_block_identifier': format_block_id(given_block_identifier)}
        estimated_transaction = self.client.estimate_gas(
            self.proxy, 'cooperativeSettle', extra_log_details=log_details, channel_identifier=channel_identifier, data1=tuple(withdraw_initiator), data2=tuple(withdraw_partner)
        )
        if estimated_transaction is None:
            raise RaidenRecoverableError('CoopSettle: gas-estimation of transaction failed.')
        estimated_transaction.estimated_gas = safe_gas_limit(
            estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.cooperativeSettle']
        )
        transaction_sent = self.client.transact(estimated_transaction)
        transaction_mined = self.client.poll_transaction(transaction_sent)
        if was_transaction_successfully_mined(transaction_mined):
            return transaction_mined.transaction_hash
        receipt = transaction_mined.receipt
        failed_at_blockhash = encode_hex(receipt['blockHash'])
        failed_at_blocknumber: int = receipt['blockNumber']
        check_transaction_failure(transaction_mined, self.client)
        raise RaidenRecoverableError(
            f'CoopSettle: mining of transaction failed (block_hash={failed_at_blockhash}, block_number={failed_at_blocknumber})'
        )

    def close(
        self,
        channel_identifier: T_ChannelID,
        partner: Address,
        balance_hash: BalanceHash,
        nonce: Nonce,
        additional_hash: AdditionalHash,
        non_closing_signature: Signature,
        closing_signature: Signature,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        """Close the channel using the provided balance proof.
        Note:
            This method must *not* be called without updating the application
            state, otherwise the node may accept new transfers which cannot be
            used, because the closer is not allowed to update the balance proof
            submitted on chain after closing
        Raises:
            RaidenRecoverableError: If the close call failed but it is not
                critical.
            RaidenUnrecoverableError: If the operation was illegal at the
                `given_block_identifier` or if the channel changes in a way that
                cannot be recovered.
        """
        canonical_identifier = CanonicalIdentifier(
            chain_identifier=self.proxy.functions.chain_id().call(),
            token_network_address=self.address,
            channel_identifier=channel_identifier,
        )
        our_signed_data = pack_signed_balance_proof(
            msg_type=MessageTypeId.BALANCE_PROOF,
            nonce=nonce,
            balance_hash=balance_hash,
            additional_hash=additional_hash,
            canonical_identifier=canonical_identifier,
            partner_signature=non_closing_signature,
        )
        try:
            our_recovered_address = recover(data=our_signed_data, signature=closing_signature)
        except Exception:
            raise RaidenUnrecoverableError("Couldn't verify the closing signature")
        else:
            if our_recovered_address != self.node_address:
                raise RaidenUnrecoverableError('Invalid closing signature')
        if non_closing_signature != EMPTY_SIGNATURE:
            partner_signed_data = pack_balance_proof(
                nonce=nonce, balance_hash=balance_hash, additional_hash=additional_hash, canonical_identifier=canonical_identifier
            )
            try:
                partner_recovered_address = recover(data=partner_signed_data, signature=non_closing_signature)
            except Exception:
                raise RaidenUnrecoverableError("Couldn't verify the non-closing signature")
            else:
                if partner_recovered_address != partner:
                    raise RaidenUnrecoverableError('Invalid non-closing signature')
        try:
            channel_onchain_detail = self._detail_channel(
                participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier, channel_identifier=channel_identifier
            )
        except ValueError:
            pass
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        else:
            onchain_channel_identifier = channel_onchain_detail.channel_identifier
            if onchain_channel_identifier != channel_identifier:
                msg = (
                    f'The provided channel identifier does not match the value on-chain at the provided block '
                    f'({format_block_id(given_block_identifier)}). This call should never have been attempted. '
                    f'provided_channel_identifier={channel_identifier}, onchain_channel_identifier={channel_onchain_detail.channel_identifier}'
                )
                raise RaidenUnrecoverableError(msg)
            if channel_onchain_detail.state != ChannelState.OPENED:
                msg = (
                    f'The channel was not open at the provided block ({format_block_id(given_block_identifier)}). '
                    f'This call should never have been attempted.'
                )
                raise RaidenUnrecoverableError(msg)
        log_details = {'given_block_identifier': format_block_id(given_block_identifier)}
        return self._close(
            channel_identifier=channel_identifier,
            partner=partner,
            balance_hash=balance_hash,
            nonce=nonce,
            additional_hash=additional_hash,
            non_closing_signature=non_closing_signature,
            closing_signature=closing_signature,
            log_details=log_details,
        )

    def _close(
        self,
        channel_identifier: T_ChannelID,
        partner: Address,
        balance_hash: BalanceHash,
        nonce: Nonce,
        additional_hash: AdditionalHash,
        non_closing_signature: Signature,
        closing_signature: Signature,
        log_details: Any,
    ) -> TransactionHash:
        with self.channel_operations_lock[partner]:
            estimated_transaction = self.client.estimate_gas(
                self.proxy,
                'closeChannel',
                extra_log_details=log_details,
                channel_identifier=channel_identifier,
                non_closing_participant=partner,
                closing_participant=self.node_address,
                balance_hash=balance_hash,
                nonce=nonce,
                additional_hash=additional_hash,
                non_closing_signature=non_closing_signature,
                closing_signature=closing_signature,
            )
            if estimated_transaction is not None:
                estimated_transaction.estimated_gas = safe_gas_limit(
                    estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.closeChannel']
                )
                transaction_sent = self.client.transact(estimated_transaction)
                transaction_mined = self.client.poll_transaction(transaction_sent)
                if not was_transaction_successfully_mined(transaction_mined):
                    receipt = transaction_mined.receipt
                    mining_block: BlockNumber = BlockNumber(receipt['blockNumber'])
                    check_transaction_failure(transaction_mined, self.client)
                    partner_details = self._detail_participant(
                        channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=mining_block
                    )
                    if partner_details.is_closer:
                        msg = 'Channel was already closed by channel partner first.'
                        raise RaidenRecoverableError(msg)
                    raise RaidenUnrecoverableError('closeChannel call failed')
                else:
                    return transaction_mined.transaction_hash
            else:
                failed_at = self.client.get_block(BLOCK_ID_LATEST)
                failed_at_blockhash: str = encode_hex(failed_at['hash'])
                failed_at_blocknumber: int = failed_at['number']
                self.client.check_for_insufficient_eth(
                    transaction_name='closeChannel',
                    transaction_executed=True,
                    required_gas=self.metadata.gas_measurements['TokenNetwork.closeChannel'],
                    block_identifier=failed_at_blocknumber,
                )
                detail = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
                )
                if detail.state < ChannelState.OPENED:
                    msg = f'cannot call close channel has not been opened yet. current_state={detail.state}'
                    raise RaidenUnrecoverableError(msg)
                if detail.state >= ChannelState.CLOSED:
                    msg = f'cannot call close on a channel that has been closed already. current_state={detail.state}'
                    raise RaidenRecoverableError(msg)
                raise RaidenUnrecoverableError(
                    f'close channel gas estimation failed for an unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
                )

    def update_transfer(
        self,
        channel_identifier: T_ChannelID,
        partner: Address,
        balance_hash: BalanceHash,
        nonce: Nonce,
        additional_hash: AdditionalHash,
        closing_signature: Signature,
        non_closing_signature: Signature,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        if balance_hash is EMPTY_BALANCE_HASH:
            raise RaidenUnrecoverableError('update_transfer called with an empty balance_hash')
        if nonce <= 0 or nonce > UINT256_MAX:
            raise RaidenUnrecoverableError('update_transfer called with an invalid nonce')
        canonical_identifier = CanonicalIdentifier(
            chain_identifier=self.proxy.functions.chain_id().call(),
            token_network_address=self.address,
            channel_identifier=channel_identifier,
        )
        partner_signed_data = pack_balance_proof(
            nonce=nonce, balance_hash=balance_hash, additional_hash=additional_hash, canonical_identifier=canonical_identifier
        )
        our_signed_data = pack_signed_balance_proof(
            msg_type=MessageTypeId.BALANCE_PROOF_UPDATE,
            nonce=nonce,
            balance_hash=balance_hash,
            additional_hash=additional_hash,
            canonical_identifier=canonical_identifier,
            partner_signature=closing_signature,
        )
        try:
            partner_recovered_address = recover(data=partner_signed_data, signature=closing_signature)
            our_recovered_address = recover(data=our_signed_data, signature=non_closing_signature)
        except Exception:
            raise RaidenUnrecoverableError("Couldn't verify the balance proof signature")
        else:
            if our_recovered_address != self.node_address:
                raise RaidenUnrecoverableError('Invalid balance proof signature')
            if partner_recovered_address != partner:
                raise RaidenUnrecoverableError('Invalid update transfer signature')
        try:
            channel_onchain_detail = self._detail_channel(
                participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier, channel_identifier=channel_identifier
            )
            closer_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=given_block_identifier
            )
            given_block_number: int = self.client.get_block(given_block_identifier)['number']
        except ValueError:
            pass
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        else:
            if channel_onchain_detail.state != ChannelState.CLOSED:
                msg = f'The channel was not closed at the provided block ({format_block_id(given_block_identifier)}). This call should never have been attempted.'
                raise RaidenUnrecoverableError(msg)
            if channel_onchain_detail.settle_block_number < given_block_number:
                msg = 'update transfer cannot be called after the settlement period, this call should never have been attempted.'
                raise RaidenUnrecoverableError(msg)
            if closer_details.nonce == nonce:
                msg = 'update transfer was already done, this call should never have been attempted.'
                raise RaidenRecoverableError(msg)
        log_details = {'given_block_identifier': format_block_id(given_block_identifier)}
        return self._update_transfer(
            channel_identifier=channel_identifier,
            partner=partner,
            balance_hash=balance_hash,
            nonce=nonce,
            additional_hash=additional_hash,
            closing_signature=closing_signature,
            non_closing_signature=non_closing_signature,
            log_details=log_details,
        )

    def _update_transfer(
        self,
        channel_identifier: T_ChannelID,
        partner: Address,
        balance_hash: BalanceHash,
        nonce: Nonce,
        additional_hash: AdditionalHash,
        closing_signature: Signature,
        non_closing_signature: Signature,
        log_details: Any,
    ) -> TransactionHash:
        estimated_transaction = self.client.estimate_gas(
            self.proxy,
            'updateNonClosingBalanceProof',
            extra_log_details=log_details,
            channel_identifier=channel_identifier,
            closing_participant=partner,
            non_closing_participant=self.node_address,
            balance_hash=balance_hash,
            nonce=nonce,
            additional_hash=additional_hash,
            closing_signature=closing_signature,
            non_closing_signature=non_closing_signature,
        )
        if estimated_transaction is not None:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas,
                self.metadata.gas_measurements['TokenNetwork.updateNonClosingBalanceProof'],
            )
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                receipt = transaction_mined.receipt
                mining_block: BlockNumber = BlockNumber(receipt['blockNumber'])
                check_transaction_failure(transaction_mined, self.client)
                channel_data = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=mining_block, channel_identifier=channel_identifier
                )
                was_channel_gone = channel_data.channel_identifier == 0 or channel_data.channel_identifier > channel_identifier
                if was_channel_gone:
                    msg = (
                        f'The provided channel identifier does not match the value on-chain at the block the update transfer was mined '
                        f'({mining_block}). provided_channel_identifier={channel_identifier}, onchain_channel_identifier={channel_data.channel_identifier}'
                    )
                    raise RaidenRecoverableError(msg)
                if channel_data.state >= ChannelState.SETTLED:
                    msg = 'Channel was already settled when update transfer was mined.'
                    raise RaidenRecoverableError(msg)
                if channel_data.settle_block_number < mining_block:
                    msg = 'update transfer was mined after the settlement window.'
                    raise RaidenRecoverableError(msg)
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=mining_block
                )
                if partner_details.nonce != nonce:
                    msg = (
                        f'update transfer failed, the on-chain nonce is higher then our expected value expected={nonce} actual={partner_details.nonce}'
                    )
                    raise RaidenUnrecoverableError(msg)
                if channel_data.state < ChannelState.CLOSED:
                    msg = f'The channel state changed unexpectedly. block=({mining_block}) onchain_state={channel_data.state}'
                    raise RaidenUnrecoverableError(msg)
                raise RaidenUnrecoverableError('update transfer failed for an unknown reason')
            else:
                return transaction_mined.transaction_hash
        else:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: int = failed_at['number']
            self.client.check_for_insufficient_eth(
                transaction_name='updateNonClosingBalanceProof',
                transaction_executed=False,
                required_gas=self.metadata.gas_measurements['TokenNetwork.updateNonClosingBalanceProof'],
                block_identifier=failed_at_blocknumber,
            )
            detail = self._detail_channel(
                participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
            )
            if detail.state < ChannelState.CLOSED:
                msg = f'cannot call update_transfer channel has not been closed yet. current_state={detail.state}'
                raise RaidenUnrecoverableError(msg)
            if detail.state >= ChannelState.SETTLED:
                msg = f'cannot call update_transfer channel has been settled already. current_state={detail.state}'
                raise RaidenRecoverableError(msg)
            if detail.settle_block_number < failed_at_blocknumber:
                raise RaidenRecoverableError('update_transfer transation sent after settlement window')
            partner_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=failed_at_blockhash
            )
            if not partner_details.is_closer:
                raise RaidenUnrecoverableError('update_transfer cannot be sent if the partner did not close the channel')
            raise RaidenUnrecoverableError(
                f'update_transfer gas estimation failed for an unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )

    def unlock(
        self, channel_identifier: T_ChannelID, sender: Address, receiver: Address, pending_locks: PendingLocksState, given_block_identifier: BlockIdentifier
    ) -> TransactionHash:
        if not pending_locks:
            raise ValueError('unlock cannot be done without pending locks')
        try:
            channel_onchain_detail = self._detail_channel(
                participant1=sender, participant2=receiver, block_identifier=given_block_identifier, channel_identifier=channel_identifier
            )
            sender_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=sender, partner=receiver, block_identifier=given_block_identifier
            )
        except ValueError:
            pass
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        else:
            if channel_onchain_detail.state != ChannelState.SETTLED:
                msg = (
                    f'The channel was not settled at the provided block ({format_block_id(given_block_identifier)}). '
                    f'This call should never have been attempted.'
                )
                raise RaidenUnrecoverableError(msg)
            local_locksroot = compute_locksroot(pending_locks)
            if sender_details.locksroot != local_locksroot:
                msg = (
                    f'The provided locksroot ({to_hex(local_locksroot)}) does correspond to the on-chain locksroot '
                    f'{to_hex(sender_details.locksroot)} for sender {to_checksum_address(sender)}.'
                )
                raise RaidenUnrecoverableError(msg)
            if sender_details.locked_amount == 0:
                msg = (
                    f'The provided locked amount on-chain is 0. This should never happen because a lock with an amount 0 is forbidden'
                    f'{to_hex(sender_details.locksroot)} for sender {to_checksum_address(sender)}.'
                )
                raise RaidenUnrecoverableError(msg)
        log_details = {'pending_locks': pending_locks, 'given_block_identifier': format_block_id(given_block_identifier)}
        return self._unlock(
            channel_identifier=channel_identifier,
            sender=sender,
            receiver=receiver,
            pending_locks=pending_locks,
            given_block_identifier=given_block_identifier,
            log_details=log_details,
        )

    def _unlock(
        self, channel_identifier: T_ChannelID, sender: Address, receiver: Address, pending_locks: PendingLocksState, given_block_identifier: BlockIdentifier, log_details: Any
    ) -> TransactionHash:
        leaves_packed: bytes = b''.join(pending_locks.locks)
        estimated_transaction = self.client.estimate_gas(
            self.proxy,
            'unlock',
            extra_log_details=log_details,
            channel_identifier=channel_identifier,
            receiver=receiver,
            sender=sender,
            locks=encode_hex(leaves_packed),
        )
        if estimated_transaction is not None:
            estimated_transaction.estimated_gas = safe_gas_limit(estimated_transaction.estimated_gas, UNLOCK_TX_GAS_LIMIT)
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                check_transaction_failure(transaction_mined, self.client)
                sender_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=sender, partner=receiver, block_identifier=given_block_identifier
                )
                is_unlock_done: bool = sender_details.locksroot == LOCKSROOT_OF_NO_LOCKS
                if is_unlock_done:
                    raise RaidenRecoverableError('The locks are already unlocked')
                raise RaidenRecoverableError('Unlocked failed for an unknown reason')
            else:
                return transaction_mined.transaction_hash
        else:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: int = failed_at['number']
            self.client.check_for_insufficient_eth(
                transaction_name='unlock',
                transaction_executed=False,
                required_gas=UNLOCK_TX_GAS_LIMIT,
                block_identifier=failed_at_blocknumber,
            )
            detail = self._detail_channel(
                participant1=sender, participant2=receiver, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
            )
            sender_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=sender, partner=receiver, block_identifier=failed_at_blockhash
            )
            if detail.state < ChannelState.SETTLED:
                msg = f'cannot call unlock on a channel that has not been settled yet. current_state={detail.state}'
                raise RaidenUnrecoverableError(msg)
            is_unlock_done = sender_details.locksroot == LOCKSROOT_OF_NO_LOCKS
            if is_unlock_done:
                raise RaidenRecoverableError('The locks are already unlocked ')
            raise RaidenUnrecoverableError(
                f'unlock estimation failed for an unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )

    def settle(
        self,
        channel_identifier: T_ChannelID,
        transferred_amount: int,
        locked_amount: int,
        locksroot: Locksroot,
        partner: Address,
        partner_transferred_amount: int,
        partner_locked_amount: int,
        partner_locksroot: Locksroot,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        with self.channel_operations_lock[partner]:
            try:
                channel_onchain_detail = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=given_block_identifier, channel_identifier=channel_identifier
                )
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=given_block_identifier
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=given_block_identifier
                )
                given_block_number: int = self.client.get_block(given_block_identifier)['number']
            except ValueError:
                pass
            except BadFunctionCallOutput:
                raise_on_call_returned_empty(given_block_identifier)
            else:
                if channel_identifier != channel_onchain_detail.channel_identifier:
                    msg = (
                        f'The provided channel identifier {channel_identifier} does not match onchain channel_identifier '
                        f'{channel_onchain_detail.channel_identifier}.'
                    )
                    raise BrokenPreconditionError(msg)
                if given_block_number < channel_onchain_detail.settle_block_number:
                    msg = 'settle cannot be called before the settlement period ends, this call should never have been attempted.'
                    raise BrokenPreconditionError(msg)
                if channel_onchain_detail.state != ChannelState.CLOSED:
                    msg = (
                        f'The channel was not closed at the provided block ({format_block_id(given_block_identifier)}). '
                        f'This call should never have been attempted.'
                    )
                    raise BrokenPreconditionError(msg)
                our_balance_hash = hash_balance_data(
                    transferred_amount=transferred_amount, locked_amount=locked_amount, locksroot=locksroot
                )
                partner_balance_hash = hash_balance_data(
                    transferred_amount=partner_transferred_amount, locked_amount=partner_locked_amount, locksroot=partner_locksroot
                )
                if our_details.balance_hash != our_balance_hash:
                    msg = 'Our balance hash does not match the on-chain value'
                    raise BrokenPreconditionError(msg)
                if partner_details.balance_hash != partner_balance_hash:
                    msg = 'Partner balance hash does not match the on-chain value'
                    raise BrokenPreconditionError(msg)
            log_details = {'given_block_identifier': format_block_id(given_block_identifier)}
            return self._settle(
                channel_identifier=channel_identifier,
                transferred_amount=transferred_amount,
                locked_amount=locked_amount,
                locksroot=locksroot,
                partner=partner,
                partner_transferred_amount=partner_transferred_amount,
                partner_locked_amount=partner_locked_amount,
                partner_locksroot=partner_locksroot,
                log_details=log_details,
            )

    def _settle(
        self,
        channel_identifier: T_ChannelID,
        transferred_amount: int,
        locked_amount: int,
        locksroot: Locksroot,
        partner: Address,
        partner_transferred_amount: int,
        partner_locked_amount: int,
        partner_locksroot: Locksroot,
        log_details: Any,
    ) -> TransactionHash:
        our_maximum: int = transferred_amount + locked_amount
        partner_maximum: int = partner_transferred_amount + partner_locked_amount
        our_bp_is_larger: bool = our_maximum > partner_maximum
        if our_bp_is_larger:
            kwargs = {
                'participant1': partner,
                'participant1_transferred_amount': partner_transferred_amount,
                'participant1_locked_amount': partner_locked_amount,
                'participant1_locksroot': partner_locksroot,
                'participant2': self.node_address,
                'participant2_transferred_amount': transferred_amount,
                'participant2_locked_amount': locked_amount,
                'participant2_locksroot': locksroot,
            }
        else:
            kwargs = {
                'participant1': self.node_address,
                'participant1_transferred_amount': transferred_amount,
                'participant1_locked_amount': locked_amount,
                'participant1_locksroot': locksroot,
                'participant2': partner,
                'participant2_transferred_amount': partner_transferred_amount,
                'participant2_locked_amount': partner_locked_amount,
                'participant2_locksroot': partner_locksroot,
            }
        estimated_transaction = self.client.estimate_gas(
            self.proxy, 'settleChannel', extra_log_details=log_details, channel_identifier=channel_identifier, **kwargs
        )
        if estimated_transaction is not None:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas, self.metadata.gas_measurements['TokenNetwork.settleChannel']
            )
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                receipt = transaction_mined.receipt
                failed_at_blockhash = encode_hex(receipt['blockHash'])
                failed_at_blocknumber: BlockNumber = BlockNumber(receipt['blockNumber'])
                self.client.check_for_insufficient_eth(
                    transaction_name='settleChannel',
                    transaction_executed=True,
                    required_gas=self.metadata.gas_measurements['TokenNetwork.settleChannel'],
                    block_identifier=failed_at_blocknumber,
                )
                channel_onchain_detail = self._detail_channel(
                    participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
                )
                our_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
                )
                partner_details = self._detail_participant(
                    channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=failed_at_blockhash
                )
                our_balance_hash = hash_balance_data(
                    transferred_amount=transferred_amount, locked_amount=locked_amount, locksroot=locksroot
                )
                partner_balance_hash = hash_balance_data(
                    transferred_amount=partner_transferred_amount, locked_amount=partner_locked_amount, locksroot=partner_locksroot
                )
                if channel_onchain_detail.state in (ChannelState.SETTLED, ChannelState.REMOVED):
                    raise RaidenRecoverableError('Channel is already settled')
                if channel_onchain_detail.state == ChannelState.OPENED:
                    raise RaidenUnrecoverableError('Channel is still open. It cannot be settled')
                is_settle_window_over: bool = channel_onchain_detail.state == ChannelState.CLOSED and failed_at_blocknumber > channel_onchain_detail.settle_block_number
                if not is_settle_window_over:
                    raise RaidenUnrecoverableError('Channel cannot be settled before settlement window is over')
                if our_details.balance_hash != our_balance_hash:
                    msg = 'Our balance hash does not match the on-chain value'
                    raise RaidenUnrecoverableError(msg)
                if partner_details.balance_hash != partner_balance_hash:
                    msg = 'Partner balance hash does not match the on-chain value'
                    raise RaidenUnrecoverableError(msg)
                raise RaidenRecoverableError('Settle failed for an unknown reason')
            else:
                return transaction_mined.transaction_hash
        else:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash: str = encode_hex(failed_at['hash'])
            failed_at_blocknumber: int = failed_at['number']
            channel_onchain_detail = self._detail_channel(
                participant1=self.node_address, participant2=partner, block_identifier=failed_at_blockhash, channel_identifier=channel_identifier
            )
            our_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=self.node_address, partner=partner, block_identifier=failed_at_blockhash
            )
            partner_details = self._detail_participant(
                channel_identifier=channel_identifier, detail_for=partner, partner=self.node_address, block_identifier=failed_at_blockhash
            )
            our_balance_hash = hash_balance_data(
                transferred_amount=transferred_amount, locked_amount=locked_amount, locksroot=locksroot
            )
            partner_balance_hash = hash_balance_data(
                transferred_amount=partner_transferred_amount, locked_amount=partner_locked_amount, locksroot=partner_locksroot
            )
            if channel_onchain_detail.state in (ChannelState.SETTLED, ChannelState.REMOVED):
                raise RaidenRecoverableError('Channel is already settled')
            if channel_onchain_detail.state == ChannelState.OPENED:
                raise RaidenUnrecoverableError('Channel is still open. It cannot be settled')
            is_settle_window_over = channel_onchain_detail.state == ChannelState.CLOSED and failed_at_blocknumber > channel_onchain_detail.settle_block_number
            if not is_settle_window_over:
                raise RaidenUnrecoverableError('Channel cannot be settled before settlement window is over')
            if our_details.balance_hash != our_balance_hash:
                msg = 'Our balance hash does not match the on-chain value'
                raise RaidenUnrecoverableError(msg)
            if partner_details.balance_hash != partner_balance_hash:
                msg = 'Partner balance hash does not match the on-chain value'
                raise RaidenUnrecoverableError(msg)
            raise RaidenRecoverableError(
                f'Settle gas estimation failed for an unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
            )