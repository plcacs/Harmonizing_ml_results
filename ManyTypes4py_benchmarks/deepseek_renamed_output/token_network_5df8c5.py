from collections import defaultdict
from dataclasses import dataclass
import structlog
from eth_utils import encode_hex, is_binary_address, to_canonical_address, to_hex
from gevent.lock import RLock
from web3.exceptions import BadFunctionCallOutput
from raiden.constants import BLOCK_ID_LATEST, EMPTY_BALANCE_HASH, EMPTY_SIGNATURE, LOCKSROOT_OF_NO_LOCKS, NULL_ADDRESS_BYTES, UINT256_MAX, UNLOCK_TX_GAS_LIMIT
from raiden.exceptions import BrokenPreconditionError, DepositOverLimit, DuplicatedChannelError, InvalidChannelID, InvalidSettleTimeout, RaidenRecoverableError, RaidenUnrecoverableError, SamePeerAddress, WithdrawMismatch
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
from raiden.utils.typing import TYPE_CHECKING, AdditionalHash, Address, Any, BalanceHash, BlockExpiration, BlockHash, BlockIdentifier, BlockNumber, ChainID, ChannelID, Dict, LockedAmount, Locksroot, NamedTuple, Nonce, Optional, Signature, T_ChannelID, TokenAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, TransactionHash, WithdrawAmount, typecheck
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK, ChannelInfoIndex, ChannelState, MessageTypeId, ParticipantInfoIndex
from raiden_contracts.contract_manager import ContractManager
if TYPE_CHECKING:
    from raiden.network.proxies.proxy_manager import ProxyManager
log = structlog.get_logger(__name__)


def func_kd8ckgsv(address1: bytes, address2: bytes) -> None:
    msg = 'The null address is not allowed as a channel participant.'
    assert NULL_ADDRESS_BYTES not in (address1, address2), msg
    msg = 'Addresses must be in binary'
    assert is_binary_address(address1) and is_binary_address(address2), msg
    if address1 == address2:
        raise SamePeerAddress(
            'Using the same address for both participants is forbiden.')


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

    def __init__(self, jsonrpc_client: JSONRPCClient, contract_manager: ContractManager, proxy_manager: 'ProxyManager',
                 metadata: TokenNetworkMetadata, block_identifier: BlockIdentifier) -> None:
        if not is_binary_address(metadata.address):
            raise ValueError('Expected binary address format for token nework')
        check_address_has_code_handle_pruned_block(client=jsonrpc_client,
            address=Address(metadata.address), contract_name=
            CONTRACT_TOKEN_NETWORK, given_block_identifier=block_identifier)
        self.contract_manager = contract_manager
        proxy = jsonrpc_client.new_contract_proxy(abi=metadata.abi,
            contract_address=Address(metadata.address))
        self._chain_id = proxy.functions.chain_id().call()
        self._token_address = TokenAddress(to_canonical_address(proxy.
            functions.token().call()))
        self.address = TokenNetworkAddress(metadata.address)
        self.proxy = proxy
        self.client = jsonrpc_client
        self.node_address = self.client.address
        self.metadata = metadata
        self.token = proxy_manager.token(token_address=self.token_address(),
            block_identifier=block_identifier)
        self.channel_operations_lock = defaultdict(RLock)
        self.opening_channels_count = 0

    def func_y4x5pv0i(self) -> ChainID:
        """Return the token of this manager."""
        return self._chain_id

    def func_g1ndrdkt(self) -> TokenAddress:
        """Return the token of this manager."""
        return self._token_address

    def func_p4kve3rn(self, block_identifier: BlockIdentifier) -> TokenAmount:
        """Return the deposit limit of a channel participant."""
        return TokenAmount(self.proxy.functions.
            channel_participant_deposit_limit().call(block_identifier=
            block_identifier))

    def func_p9yhlqe9(self, block_identifier: BlockIdentifier) -> TokenAmount:
        """Return the token of this manager."""
        return TokenAmount(self.proxy.functions.token_network_deposit_limit
            ().call(block_identifier=block_identifier))

    def func_6snc4kbl(self, block_identifier: BlockIdentifier) -> bool:
        return self.proxy.functions.safety_deprecation_switch().call(
            block_identifier=block_identifier)

    def func_p98y6522(self, partner: Address, settle_timeout: int, given_block_identifier: BlockIdentifier) -> NewNettingChannelDetails:
        """Creates a new channel in the TokenNetwork contract.

        Args:
            partner: The peer to open the channel with.
            settle_timeout: The settle timeout to use for this channel.
            given_block_identifier: The block identifier of the state change that
                                    prompted this proxy action

        Returns:
            The ChannelID of the new netting channel.
        """
        func_kd8ckgsv(self.node_address, partner)
        timeout_min = self.settlement_timeout_min()
        timeout_max = self.settlement_timeout_max()
        invalid_timeout = (settle_timeout < timeout_min or settle_timeout >
            timeout_max)
        if invalid_timeout:
            msg = (
                f'settle_timeout must be in range [{timeout_min}, {timeout_max}], is {settle_timeout}'
                )
            raise InvalidSettleTimeout(msg)
        with self.channel_operations_lock[partner]:
            try:
                existing_channel_identifier = (self.
                    get_channel_identifier_or_none(participant1=self.
                    node_address, participant2=partner, block_identifier=
                    given_block_identifier))
                network_total_deposit = self.token.balance_of(address=
                    Address(self.address), block_identifier=
                    given_block_identifier)
                limit = self.token_network_deposit_limit(block_identifier=
                    given_block_identifier)
                safety_deprecation_switch = self.safety_deprecation_switch(
                    given_block_identifier)
            except ValueError:
                pass
            except BadFunctionCallOutput:
                raise_on_call_returned_empty(given_block_identifier)
            else:
                if existing_channel_identifier is not None:
                    raise BrokenPreconditionError(
                        'A channel with the given partner address already exists.'
                        )
                if network_total_deposit >= limit:
                    raise BrokenPreconditionError(
                        'Cannot open another channel, token network deposit limit reached.'
                        )
                if safety_deprecation_switch:
                    raise BrokenPreconditionError(
                        'This token network is deprecated.')
            self.opening_channels_count += 1
            try:
                netting_channel_details = self._new_netting_channel(partner,
                    settle_timeout)
            finally:
                self.opening_channels_count -= 1
            return netting_channel_details

    def func_1x0yqqhk(self, partner: Address, settle_timeout: int) -> NewNettingChannelDetails:
        estimated_transaction = self.client.estimate_gas(self.proxy,
            'openChannel', extra_log_details={}, participant1=self.
            node_address, participant2=partner, settle_timeout=settle_timeout)
        if estimated_transaction is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blockhash = encode_hex(failed_at['hash'])
            failed_at_blocknumber = failed_at['number']
            self.client.check_for_insufficient_eth(transaction_name=
                'openChannel', transaction_executed=False, required_gas=
                self.metadata.gas_measurements['TokenNetwork.openChannel'],
                block_identifier=failed_at_blocknumber)
            existing_channel_identifier = self.get_channel_identifier_or_none(
                participant1=self.node_address, participant2=partner,
                block_identifier=failed_at_blockhash)
            if existing_channel_identifier is not None:
                raise DuplicatedChannelError(
                    'Channel with given partner address already exists')
            network_total_deposit = self.token.balance_of(address=Address(
                self.address), block_identifier=failed_at_blockhash)
            limit = self.token_network_deposit_limit(block_identifier=
                failed_at_blockhash)
            if network_total_deposit >= limit:
                raise DepositOverLimit(
                    'Could open another channel, token network deposit limit has been reached.'
                    )
            if self.safety_deprecation_switch(block_identifier=
                failed_at_blockhash):
                raise RaidenRecoverableError(
                    'This token network is deprecated.')
            raise RaidenRecoverableError(
                f'Creating a new channel will fail - Gas estimation failed for unknown reason. Reference block {failed_at_blockhash} {failed_at_blocknumber}.'
                )
        else:
            estimated_transaction.estimated_gas = safe_gas_limit(
                estimated_transaction.estimated_gas, self.metadata.
                gas_measurements['TokenNetwork.openChannel'])
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            receipt = transaction_mined.receipt
            if not was_transaction_successfully_mined(transaction_mined):
                failed_at_blockhash = encode_hex(receipt['blockHash'])
                existing_channel_identifier = (self.
                    get_channel_identifier_or_none(participant1=self.
                    node_address, participant2=partner, block_identifier=
                    failed_at_blockhash))
                if existing_channel_identifier is not None:
                    raise DuplicatedChannelError(
                        'Channel with given partner address already exists')
                network_total_deposit = self.token.balance_of(address=
                    Address(self.address), block_identifier=failed_at_blockhash
                    )
                limit = self.token_network_deposit_limit(block_identifier=
                    failed_at_blockhash)
                if network_total_deposit >= limit:
                    raise DepositOverLimit(
                        'Could open another channel, token network deposit limit has been reached.'
                        )
                if self.safety_deprecation_switch(block_identifier=
                    failed_at_blockhash):
                    raise RaidenRecoverableError(
                        'This token network is deprecated.')
                raise RaidenRecoverableError('Creating new channel failed.')
        channel_identifier = self._detail_channel(participant1=self.
            node_address, participant2=partner, block_identifier=encode_hex
            (receipt['blockHash'])).channel_identifier
        return NewNettingChannelDetails(channel_identifier=
            channel_identifier, block_hash=BlockHash(receipt['blockHash']),
            block_number=BlockNumber(receipt['blockNumber']),
            transaction_hash=TransactionHash(transaction_mined.
            transaction_hash))

    def func_qaa6tz0q(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier) -> ChannelID:
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
        func_kd8ckgsv(participant1, participant2)
        channel_identifier = self.proxy.functions.getChannelIdentifier(
            participant=participant1, partner=participant2).call(
            block_identifier=block_identifier)
        if channel_identifier == 0:
            msg = (
                f'getChannelIdentifier returned 0, meaning no channel currently exists between {to_checksum_address(participant1)} and {to_checksum_address(participant2)}'
                )
            raise RaidenRecoverableError(msg)
        return channel_identifier

    def func_jmgjqjuu(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier) -> Optional[ChannelID]:
        """Returns the channel identifier if an open channel exists, else None."""
        try:
            return self.get_channel_identifier(participant1=participant1,
                participant2=participant2, block_identifier=block_identifier)
        except RaidenRecoverableError:
            return None

    def func_3qr83x09(self, channel_identifier: ChannelID, detail_for: Address, partner: Address,
        block_identifier: BlockIdentifier) -> ParticipantDetails:
        """Returns a dictionary with the channel participant information."""
        func_kd8ckgsv(detail_for, partner)
        data = self.proxy.functions.getChannelParticipantInfo(
            channel_identifier=channel_identifier, participant=detail_for,
            partner=partner).call(block_identifier=block_identifier)
        return ParticipantDetails(address=detail_for, deposit=data[
            ParticipantInfoIndex.DEPOSIT], withdrawn=data[
            ParticipantInfoIndex.WITHDRAWN], is_closer=data[
            ParticipantInfoIndex.IS_CLOSER], balance_hash=data[
            ParticipantInfoIndex.BALANCE_HASH], nonce=data[
            ParticipantInfoIndex.NONCE], locksroot=data[
            ParticipantInfoIndex.LOCKSROOT], locked_amount=data[
            ParticipantInfoIndex.LOCKED_AMOUNT])

    def func_m6ukqz9d(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier,
        channel_identifier: Optional[ChannelID] = None) -> ChannelData:
        """Returns a ChannelData instance with the channel specific information.

        If no specific channel_identifier is given then it tries to see if there
        is a currently open channel and uses that identifier.

        """
        func_kd8ckgsv(participant1, participant2)
        if channel_identifier is None:
            channel_identifier = self.get_channel_identifier(participant1=
                participant1, participant2=participant2, block_identifier=
                block_identifier)
        elif not isinstance(channel_identifier, T_ChannelID):
            raise InvalidChannelID(
                'channel_identifier must be of type T_ChannelID')
        elif channel_identifier <= 0 or channel_identifier > UINT256_MAX:
            raise InvalidChannelID(
                'channel_identifier must be larger then 0 and smaller then uint256'
                )
        channel_data = self.proxy.functions.getChannelInfo(channel_identifier
            =channel_identifier, participant1=participant1, participant2=
            participant2).call(block_identifier=block_identifier)
        return ChannelData(channel_identifier=channel_identifier,
            settle_block_number=channel_data[ChannelInfoIndex.SETTLE_BLOCK],
            state=channel_data[ChannelInfoIndex.STATE])

    def func_8h91ulid(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier,
        channel_identifier: Optional[ChannelID] = None) -> ParticipantsDetails:
        """Returns a ParticipantsDetails instance with the participants'
            channel information.

        Note:
            For now one of the participants has to be the node_address
        """
        if self.node_address not in (participant1, participant2):
            raise ValueError('One participant must be the node address')
        if self.node_address == participant2:
            participant1, participant2 = participant2, participant1
        if channel_identifier is None:
            channel_identifier = self.get_channel_identifier(participant1=
                participant1, participant2=participant2, block_identifier=
                block_identifier)
        elif not isinstance(channel_identifier, T_ChannelID):
            raise InvalidChannelID(
                'channel_identifier must be of type T_ChannelID')
        elif channel_identifier <= 0 or channel_identifier > UINT256_MAX:
            raise InvalidChannelID(
                'channel_identifier must be larger then 0 and smaller then uint256'
                )
        our_data = self._detail_participant(channel_identifier=
            channel_identifier, detail_for=participant1, partner=
            participant2, block_identifier=block_identifier)
        partner_data = self._detail_participant(channel_identifier=
            channel_identifier, detail_for=participant2, partner=
            participant1, block_identifier=block_identifier)
        return ParticipantsDetails(our_details=our_data, partner_details=
            partner_data)

    def func_kg8n8g0b(self, participant1: Address, participant2: Address, block_identifier: BlockIdentifier,
        channel_identifier: Optional[ChannelID] = None) -> ChannelDetails:
        """Returns a ChannelDetails instance with all the details of the
            channel and the channel participants.

        Note:
            For now one of the participants has to be the node_address
        """
        if self.node_address not in (participant1, participant2):
            raise ValueError('One participant must be the node address')
        if self.node_address == participant2:
            participant1, participant2 = participant2, participant1
        channel_data = self._detail_channel(participant1=participant1,
            participant2=participant2, block_identifier=block_identifier,
            channel_identifier=channel_identifier)
        participants_data = self.detail_participants(participant1=
            participant1, participant2=participant2, block_identifier=
            block_identifier, channel_identifier=channel_data.
            channel