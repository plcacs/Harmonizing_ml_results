from typing import Any, Dict
from raiden.blockchain.filters import decode_event, get_filter_args_for_specific_event_from_channel
from raiden.network.proxies.token_network import ChannelDetails, TokenNetwork, WithdrawInput
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import (
    AdditionalHash,
    BalanceHash,
    BlockExpiration,
    BlockIdentifier,
    BlockTimeout,
    Nonce,
    Signature,
    TokenAddress,
    TokenAmount,
    TransactionHash,
    WithdrawAmount,
)
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK, ChannelEvent
from raiden_contracts.contract_manager import ContractManager


class PaymentChannel:
    def __init__(
        self,
        token_network: TokenNetwork,
        channel_state: NettingChannelState,
        contract_manager: ContractManager,
    ) -> None:
        self.channel_identifier: int = channel_state.canonical_identifier.channel_identifier
        self.participant1: TokenAddress = channel_state.our_state.address
        self.participant2: TokenAddress = channel_state.partner_state.address
        self.token_network: TokenNetwork = token_network
        self.client: Any = token_network.client
        self.contract_manager: ContractManager = contract_manager

    def token_address(self) -> TokenAddress:
        """Returns the address of the token for the channel."""
        return self.token_network.token_address()

    def detail(self, block_identifier: BlockIdentifier) -> ChannelDetails:
        """Returns the channel details."""
        return self.token_network.detail(
            participant1=self.participant1,
            participant2=self.participant2,
            block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
        )

    def settle_timeout(self) -> BlockTimeout:
        """Returns the channels settle_timeout."""
        filter_args: Dict[str, Any] = get_filter_args_for_specific_event_from_channel(
            token_network_address=self.token_network.address,
            channel_identifier=self.channel_identifier,
            event_name=ChannelEvent.OPENED,
            contract_manager=self.contract_manager,
        )
        events = self.client.web3.eth.get_logs(filter_args)
        assert len(events) > 0, 'No matching ChannelOpen event found.'
        event: Dict[str, Any] = decode_event(
            self.contract_manager.get_contract_abi(CONTRACT_TOKEN_NETWORK), events[-1]
        )
        return event["args"]["settle_timeout"]

    def opened(self, block_identifier: BlockIdentifier) -> bool:
        """Returns if the channel is opened."""
        return self.token_network.channel_is_opened(
            participant1=self.participant1,
            participant2=self.participant2,
            block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
        )

    def closed(self, block_identifier: BlockIdentifier) -> bool:
        """Returns if the channel is closed."""
        return self.token_network.channel_is_closed(
            participant1=self.participant1,
            participant2=self.participant2,
            block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
        )

    def settled(self, block_identifier: BlockIdentifier) -> bool:
        """Returns if the channel is settled."""
        return self.token_network.channel_is_settled(
            participant1=self.participant1,
            participant2=self.participant2,
            block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
        )

    def can_transfer(self, block_identifier: BlockIdentifier) -> bool:
        """Returns True if the channel is opened and the node has deposit in it."""
        return self.token_network.can_transfer(
            participant1=self.participant1,
            participant2=self.participant2,
            block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
        )

    def approve_and_set_total_deposit(
        self, total_deposit: TokenAmount, block_identifier: BlockIdentifier
    ) -> None:
        self.token_network.approve_and_set_total_deposit(
            given_block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
            total_deposit=total_deposit,
            partner=self.participant2,
        )

    def set_total_withdraw(
        self,
        total_withdraw: WithdrawAmount,
        participant_signature: Signature,
        partner_signature: Signature,
        expiration_block: BlockExpiration,
        block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        withdraw_input: WithdrawInput = WithdrawInput(
            total_withdraw=total_withdraw,
            initiator=self.participant1,
            initiator_signature=participant_signature,
            partner_signature=partner_signature,
            expiration_block=expiration_block,
        )
        return self.token_network.set_total_withdraw(
            given_block_identifier=block_identifier,
            channel_identifier=self.channel_identifier,
            partner=self.participant2,
            withdraw_input=withdraw_input,
        )

    def close(
        self,
        nonce: Nonce,
        balance_hash: BalanceHash,
        additional_hash: AdditionalHash,
        non_closing_signature: Signature,
        closing_signature: Signature,
        block_identifier: BlockIdentifier,
    ) -> None:
        """Closes the channel using the provided balance proof, and our closing signature."""
        self.token_network.close(
            channel_identifier=self.channel_identifier,
            partner=self.participant2,
            balance_hash=balance_hash,
            nonce=nonce,
            additional_hash=additional_hash,
            non_closing_signature=non_closing_signature,
            closing_signature=closing_signature,
            given_block_identifier=block_identifier,
        )

    def update_transfer(
        self,
        nonce: Nonce,
        balance_hash: BalanceHash,
        additional_hash: AdditionalHash,
        partner_signature: Signature,
        signature: Signature,
        block_identifier: BlockIdentifier,
    ) -> None:
        """Updates the channel using the provided balance proof."""
        self.token_network.update_transfer(
            channel_identifier=self.channel_identifier,
            partner=self.participant2,
            balance_hash=balance_hash,
            nonce=nonce,
            additional_hash=additional_hash,
            closing_signature=partner_signature,
            non_closing_signature=signature,
            given_block_identifier=block_identifier,
        )