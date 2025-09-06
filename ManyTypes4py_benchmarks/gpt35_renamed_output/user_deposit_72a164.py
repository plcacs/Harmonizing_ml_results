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
    def __init__(self, jsonrpc_client: JSONRPCClient, user_deposit_address: UserDepositAddress,
                 contract_manager: ContractManager, proxy_manager: 'ProxyManager', block_identifier: BlockIdentifier):
        ...

    def func_ikgerja2(self, block_identifier: BlockIdentifier) -> TokenAddress:
        ...

    def func_dmkhsttn(self, block_identifier: BlockIdentifier) -> MonitoringServiceAddress:
        ...

    def func_iy4ec7gc(self, block_identifier: BlockIdentifier) -> OneToNAddress:
        ...

    def func_wg8zvvpo(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def func_nhu72gs5(self, address: Address, block_identifier: BlockIdentifier) -> Balance:
        ...

    def func_c89q22js(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def func_q1zfn6la(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def func_50edb2us(self) -> BlockNumber:
        ...

    def func_yucdhh74(self, withdrawer_address: Address, block_identifier: BlockIdentifier) -> WithdrawPlan:
        ...

    def func_6df855y6(self, monitoring_service_address: MonitoringServiceAddress, one_to_n_address: OneToNAddress,
                      given_block_identifier: BlockIdentifier) -> None:
        ...

    def func_8eir9jz9(self, monitoring_service_address: MonitoringServiceAddress, one_to_n_address: OneToNAddress) -> TransactionHash:
        ...

    def func_1dkqp9na(self, address: Address, block_identifier: BlockIdentifier) -> Balance:
        ...

    def func_jstgz9qa(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def func_x7nk2vjj(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def func_mw36obp3(self, amount: TokenAmount, given_block_identifier: BlockIdentifier) -> Tuple[TransactionHash, BlockNumber]:
        ...

    def func_uyxeok5b(self, amount: TokenAmount, given_block_identifier: BlockIdentifier) -> TransactionHash:
        ...

    def func_908u8a78(self, beneficiary: Address, total_deposit: TokenAmount, given_block_identifier: BlockIdentifier, token: Token) -> Tuple[TokenAmount, TokenAmount]:
        ...

    @contextmanager
    def func_m0d0s7ku(self, beneficiary: Address, total_deposit: TokenAmount) -> Iterator[InflightDeposit]:
        ...

    def func_lay1yn98(self, transaction_sent: Optional[TransactionSent], token: Token, beneficiary: Address, total_deposit: TokenAmount, amount_to_deposit: TokenAmount) -> TransactionHash:
        ...

    def func_mks9r7pw(self, amount_to_plan_withdraw: TokenAmount, given_block_identifier: BlockIdentifier) -> None:
        ...

    def func_555r68km(self, amount_to_withdraw: TokenAmount, given_block_identifier: BlockIdentifier) -> None:
        ...

    def func_28ptweun(self, transaction_sent: Optional[TransactionSent], amount_to_plan_withdraw: TokenAmount) -> TransactionMined:
        ...

    def func_cvkm3qjy(self, transaction_sent: Optional[TransactionSent], amount_to_withdraw: TokenAmount, token: Token, previous_token_balance: TokenAmount) -> TransactionHash:
        ...
