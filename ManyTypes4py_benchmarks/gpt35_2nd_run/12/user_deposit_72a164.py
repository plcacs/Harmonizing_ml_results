from contextlib import contextmanager
from dataclasses import dataclass, field
import structlog
from eth_utils import is_binary_address, to_canonical_address
from gevent.event import AsyncResult
from gevent.threading import Lock
from web3.exceptions import BadFunctionCallOutput
from raiden.constants import BLOCK_ID_LATEST, BLOCK_ID_PENDING, EMPTY_ADDRESS, UINT256_MAX
from raiden.exceptions import BrokenPreconditionError, RaidenRecoverableError
from raiden.network.proxies.token import Token
from raiden.network.proxies.utils import raise_on_call_returned_empty
from raiden.network.rpc.client import JSONRPCClient, TransactionMined, TransactionSent, check_address_has_code_handle_pruned_block, was_transaction_successfully_mined
from raiden.utils.formatting import format_block_id, to_checksum_address
from raiden.utils.typing import TYPE_CHECKING, Address, Any, Balance, BlockIdentifier, BlockNumber, Dict, Iterator, MonitoringServiceAddress, OneToNAddress, Optional, TokenAddress, TokenAmount, TransactionHash, Tuple, UserDepositAddress
from raiden_contracts.constants import CONTRACT_MONITORING_SERVICE, CONTRACT_ONE_TO_N, CONTRACT_USER_DEPOSIT
from raiden_contracts.contract_manager import ContractManager, gas_measurements
if TYPE_CHECKING:
    from raiden.network.proxies.proxy_manager import ProxyManager
log = structlog.get_logger(__name__)

@dataclass
class InflightDeposit:
    total_deposit: TokenAmount = field(default_factory=lambda: TokenAmount(0))
    async_result: AsyncResult = field(default_factory=AsyncResult)

@dataclass(frozen=True)
class WithdrawPlan:
    withdraw_amount: TokenAmount
    withdraw_block: BlockNumber

class UserDeposit:

    def __init__(self, jsonrpc_client: JSONRPCClient, user_deposit_address: UserDepositAddress, contract_manager: ContractManager, proxy_manager: 'ProxyManager', block_identifier: BlockIdentifier):
        ...

    def token_address(self, block_identifier: BlockIdentifier) -> TokenAddress:
        ...

    def monitoring_service_address(self, block_identifier: BlockIdentifier) -> MonitoringServiceAddress:
        ...

    def one_to_n_address(self, block_identifier: BlockIdentifier) -> OneToNAddress:
        ...

    def get_total_deposit(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def get_balance(self, address: Address, block_identifier: BlockIdentifier) -> Balance:
        ...

    def whole_balance(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def whole_balance_limit(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def get_withdraw_delay(self) -> BlockNumber:
        ...

    def get_withdraw_plan(self, withdrawer_address: Address, block_identifier: BlockIdentifier) -> WithdrawPlan:
        ...

    def init(self, monitoring_service_address: MonitoringServiceAddress, one_to_n_address: OneToNAddress, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def _init(self, monitoring_service_address: MonitoringServiceAddress, one_to_n_address: OneToNAddress) -> TransactionHash:
        ...

    def effective_balance(self, address: Address, block_identifier: BlockIdentifier) -> Balance:
        ...

    def deposit(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def approve_and_deposit(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def plan_withdraw(self, amount: TokenAmount, given_block_identifier: BlockIdentifier) -> Tuple[TransactionHash, BlockNumber]:
        ...

    def withdraw(self, amount: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def _deposit_preconditions(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier, token: Token) -> Tuple[TokenAmount, TokenAmount]:
        ...

    @contextmanager
    def _deposit_inflight(self, beneficiary: Address, total_deposit: TokenAmount) -> Iterator[InflightDeposit]:
        ...

    def _deposit_check_result(self, transaction_sent: Optional[TransactionSent], token: Token, beneficiary: Address, total_deposit: TokenAmount, amount_to_deposit: TokenAmount) -> TransactionHash:
        ...

    def _plan_withdraw_preconditions(self, amount_to_plan_withdraw: TokenAmount, given_block_identifier: BlockIdentifier) -> None:
        ...

    def _withdraw_preconditions(self, amount_to_withdraw: TokenAmount, given_block_identifier: BlockIdentifier) -> None:
        ...

    def _plan_withdraw_check_result(self, transaction_sent: Optional[TransactionSent], amount_to_plan_withdraw: TokenAmount) -> TransactionMined:
        ...

    def _withdraw_check_result(self, transaction_sent: Optional[TransactionSent], amount_to_withdraw: TokenAmount, token: Token, previous_token_balance: TokenAmount) -> TransactionHash:
        ...
