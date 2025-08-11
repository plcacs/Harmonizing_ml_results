from raiden.blockchain.filters import decode_event, get_filter_args_for_specific_event_from_channel
from raiden.network.proxies.token_network import ChannelDetails, TokenNetwork, WithdrawInput
from raiden.transfer.state import NettingChannelState
from raiden.utils.typing import AdditionalHash, BalanceHash, BlockExpiration, BlockIdentifier, BlockTimeout, Nonce, Signature, TokenAddress, TokenAmount, TransactionHash, WithdrawAmount
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK, ChannelEvent
from raiden_contracts.contract_manager import ContractManager

class PaymentChannel:

    def __init__(self, token_network: Union[raiden.network.proxies.proxy_manager.ProxyManager, nucypher.blockchain.eth.registry.BaseContractRegistry], channel_state: Union[raiden.utils.Address, raiden_contracts.contract_manager.ContractManager, raiden.utils.TokenNetworkAddress], contract_manager: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.proxy_manager.ProxyManager, nucypher.blockchain.eth.registry.BaseContractRegistry]) -> None:
        self.channel_identifier = channel_state.canonical_identifier.channel_identifier
        self.participant1 = channel_state.our_state.address
        self.participant2 = channel_state.partner_state.address
        self.token_network = token_network
        self.client = token_network.client
        self.contract_manager = contract_manager

    def token_address(self) -> Union[str, bytes]:
        """Returns the address of the token for the channel."""
        return self.token_network.token_address()

    def detail(self, block_identifier: Union[raiden.utils.BlockSpecification, list[raiden.transfer.architecture.Event], raiden.utils.TokenNetworkID]) -> Union[str, int, nucypher.blockchain.eth.registry.BaseContractRegistry]:
        """Returns the channel details."""
        return self.token_network.detail(participant1=self.participant1, participant2=self.participant2, block_identifier=block_identifier, channel_identifier=self.channel_identifier)

    def settle_timeout(self) -> Union[bool, str]:
        """Returns the channels settle_timeout."""
        filter_args = get_filter_args_for_specific_event_from_channel(token_network_address=self.token_network.address, channel_identifier=self.channel_identifier, event_name=ChannelEvent.OPENED, contract_manager=self.contract_manager)
        events = self.client.web3.eth.get_logs(filter_args)
        assert len(events) > 0, 'No matching ChannelOpen event found.'
        event = decode_event(self.contract_manager.get_contract_abi(CONTRACT_TOKEN_NETWORK), events[-1])
        return event['args']['settle_timeout']

    def opened(self, block_identifier: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier, raiden.transfer.identifiers.CanonicalIdentifier]) -> bool:
        """Returns if the channel is opened."""
        return self.token_network.channel_is_opened(participant1=self.participant1, participant2=self.participant2, block_identifier=block_identifier, channel_identifier=self.channel_identifier)

    def closed(self, block_identifier: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier]) -> Union[bool, str]:
        """Returns if the channel is closed."""
        return self.token_network.channel_is_closed(participant1=self.participant1, participant2=self.participant2, block_identifier=block_identifier, channel_identifier=self.channel_identifier)

    def settled(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.Address]) -> Union[bool, str]:
        """Returns if the channel is settled."""
        return self.token_network.channel_is_settled(participant1=self.participant1, participant2=self.participant2, block_identifier=block_identifier, channel_identifier=self.channel_identifier)

    def can_transfer(self, block_identifier: Union[raiden.utils.BlockSpecification, raiden.utils.BlockIdentifier, raiden.utils.Address]) -> Union[str, bool, raiden.transfer.state.NettingChannelState]:
        """Returns True if the channel is opened and the node has deposit in it."""
        return self.token_network.can_transfer(participant1=self.participant1, participant2=self.participant2, block_identifier=block_identifier, channel_identifier=self.channel_identifier)

    def approve_and_set_total_deposit(self, total_deposit: Union[raiden.utils.BlockSpecification, raiden.utils.TokenAmount, int], block_identifier: Union[raiden.utils.BlockSpecification, raiden.utils.TokenAmount, int]) -> None:
        self.token_network.approve_and_set_total_deposit(given_block_identifier=block_identifier, channel_identifier=self.channel_identifier, total_deposit=total_deposit, partner=self.participant2)

    def set_total_withdraw(self, total_withdraw: Union[raiden.utils.TokenAmount, raiden.utils.LockedAmount, raiden.utils.ChannelID], participant_signature: Union[raiden.utils.TokenAmount, raiden.utils.LockedAmount, raiden.utils.ChannelID], partner_signature: Union[raiden.utils.TokenAmount, raiden.utils.LockedAmount, raiden.utils.ChannelID], expiration_block: Union[raiden.utils.TokenAmount, raiden.utils.LockedAmount, raiden.utils.ChannelID], block_identifier: Union[raiden.utils.BlockNumber, raiden.utils.TokenAmount, raiden.utils.Address]) -> Union[bool, tuple[str]]:
        withdraw_input = WithdrawInput(total_withdraw=total_withdraw, initiator=self.participant1, initiator_signature=participant_signature, partner_signature=partner_signature, expiration_block=expiration_block)
        return self.token_network.set_total_withdraw(given_block_identifier=block_identifier, channel_identifier=self.channel_identifier, partner=self.participant2, withdraw_input=withdraw_input)

    def close(self, nonce: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID], balance_hash: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID], additional_hash: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID], non_closing_signature: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID], closing_signature: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID], block_identifier: Union[bytes, raiden.utils.Signature, raiden.utils.ChannelID]) -> None:
        """Closes the channel using the provided balance proof, and our closing signature."""
        self.token_network.close(channel_identifier=self.channel_identifier, partner=self.participant2, balance_hash=balance_hash, nonce=nonce, additional_hash=additional_hash, non_closing_signature=non_closing_signature, closing_signature=closing_signature, given_block_identifier=block_identifier)

    def update_transfer(self, nonce: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification], balance_hash: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification], additional_hash: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification], partner_signature: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification], signature: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification], block_identifier: Union[raiden.utils.Signature, int, raiden.utils.BlockSpecification]) -> None:
        """Updates the channel using the provided balance proof."""
        self.token_network.update_transfer(channel_identifier=self.channel_identifier, partner=self.participant2, balance_hash=balance_hash, nonce=nonce, additional_hash=additional_hash, closing_signature=partner_signature, non_closing_signature=signature, given_block_identifier=block_identifier)