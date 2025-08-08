from typing import TYPE_CHECKING, List, Mapping, Optional, Callable, Iterable
from raiden.transfer.state import ChainState, ChannelState, NettingChannelEndState, NettingChannelState
from raiden.transfer.state_change import ContractReceiveChannelSettled, ContractReceiveChannelWithdraw, ContractReceiveSecretReveal
from raiden.utils.typing import Address, Any, BlockNumber, ChannelID, PaymentAmount, PaymentID, SecretHash, Sequence, TokenAddress, TokenAmount, TokenNetworkRegistryAddress, WithdrawAmount
if TYPE_CHECKING:
    from raiden.raiden_service import RaidenService

def _get_channel_state_by_partner_address(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address) -> Optional[ChannelState]:

class ChannelStateCondition(ABC):

    @abstractmethod
    def evaluate(self, chain_state: ChainState, channel_state: ChannelState) -> bool:

    def __call__(self, chain_state: ChainState, channel_state: ChannelState) -> bool:

@dataclass
class ChannelHasDeposit(ChannelStateCondition):

@dataclass
class ChannelExists(ChannelStateCondition):

@dataclass
class ChannelHasPaymentBalance(ChannelStateCondition):

@dataclass
class ChannelInTargetStates(ChannelStateCondition):

@dataclass
class ChannelExpiredCoopSettle(ChannelStateCondition):

@dataclass
class ChannelCoopSettleSuccess(ChannelStateCondition):

@dataclass
class And(ChannelStateCondition):

@dataclass
class Or(ChannelStateCondition):

@dataclass
class ChannelStateWaiter:

def _get_canonical_identifier_by_channel_id(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, channel_id: ChannelID) -> CanonicalIdentifier:

def wait_until(func: Callable[[], Any], wait_for: Optional[float] = None, sleep_for: float = 0.5) -> Any:

def wait_for_block(raiden: RaidenService, block_number: BlockNumber, retry_timeout: float):

def wait_for_newchannel(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, retry_timeout: float):

def wait_for_participant_deposit(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, target_address: Address, target_balance: TokenAmount, retry_timeout: float):

def wait_single_channel_deposit(app_deposit: RaidenService, app_partner: RaidenService, registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, total_deposit: TokenAmount, retry_timeout: float):

def wait_both_channel_deposit(app_deposit: RaidenService, app_partner: RaidenService, registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, total_deposit: TokenAmount, retry_timeout: float):

def wait_for_payment_balance(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, target_address: Address, target_balance: TokenAmount, retry_timeout: float):

def wait_for_channels(raiden: RaidenService, canonical_id_to_condition: Mapping[CanonicalIdentifier, ChannelStateCondition], retry_timeout: float, timeout: Optional[float] = None):

def wait_for_channel_in_states(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, channel_ids: List[ChannelID], retry_timeout: float, target_states: Iterable[str]):

def wait_for_close(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, channel_ids: List[ChannelID], retry_timeout: float):

def wait_for_token_network(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, retry_timeout: float):

def wait_for_settle(raiden: RaidenService, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, channel_ids: List[ChannelID], retry_timeout: float):

class TransferWaitResult(Enum):

def wait_for_received_transfer_result(raiden: RaidenService, payment_identifier: PaymentID, amount: PaymentAmount, retry_timeout: float, secrethash: SecretHash):

def wait_for_withdraw_complete(raiden: RaidenService, canonical_identifier: CanonicalIdentifier, total_withdraw: WithdrawAmount, retry_timeout: float):
