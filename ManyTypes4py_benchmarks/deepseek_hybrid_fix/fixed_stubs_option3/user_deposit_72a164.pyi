from contextlib import contextmanager
from dataclasses import dataclass, field
from gevent.event import AsyncResult
from gevent.threading import Lock
from typing import Any, Dict, Iterator, Optional, Tuple

from eth_typing import ChecksumAddress
from web3.types import BlockIdentifier

from raiden.network.proxies.token import Token
from raiden.network.rpc.client import JSONRPCClient, TransactionMined, TransactionSent
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    MonitoringServiceAddress,
    OneToNAddress,
    TokenAddress,
    TokenAmount,
    TransactionHash,
    UserDepositAddress,
)
from raiden_contracts.contract_manager import ContractManager


@dataclass
class InflightDeposit:
    total_deposit: TokenAmount = field(default_factory=lambda: TokenAmount(0))
    async_result: AsyncResult = field(default_factory=AsyncResult)

    def __init__(self, total_deposit: TokenAmount = TokenAmount(0), async_result: AsyncResult = AsyncResult()) -> None: ...


@dataclass(frozen=True)
class WithdrawPlan:
    withdraw_amount: TokenAmount
    withdraw_block: BlockNumber


class UserDeposit:
    _inflight_deposits: Dict[Address, InflightDeposit]

    def __init__(
        self,
        jsonrpc_client: JSONRPCClient,
        user_deposit_address: UserDepositAddress,
        contract_manager: ContractManager,
        proxy_manager: "ProxyManager",
        block_identifier: BlockIdentifier,
    ) -> None: ...

    def token_address(self, block_identifier: BlockIdentifier) -> TokenAddress: ...
    def monitoring_service_address(self, block_identifier: BlockIdentifier) -> MonitoringServiceAddress: ...
    def one_to_n_address(self, block_identifier: BlockIdentifier) -> OneToNAddress: ...
    def get_total_deposit(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount: ...
    def get_balance(self, address: Address, block_identifier: BlockIdentifier) -> Balance: ...
    def whole_balance(self, block_identifier: BlockIdentifier) -> TokenAmount: ...
    def whole_balance_limit(self, block_identifier: BlockIdentifier) -> TokenAmount: ...
    def get_withdraw_delay(self) -> BlockNumber: ...
    def get_withdraw_plan(self, withdrawer_address: Address, block_identifier: BlockIdentifier) -> WithdrawPlan: ...
    def init(
        self,
        monitoring_service_address: MonitoringServiceAddress,
        one_to_n_address: OneToNAddress,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash: ...
    def effective_balance(self, address: Address, block_identifier: BlockIdentifier) -> Balance: ...
    def deposit(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash: ...
    def approve_and_deposit(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash: ...
    def plan_withdraw(
        self, amount: TokenAmount, given_block_identifier: BlockIdentifier
    ) -> Tuple[TransactionHash, BlockNumber]: ...
    def withdraw(
        self, amount: TokenAmount, given_block_identifier: BlockIdentifier
    ) -> TransactionHash: ...

    def _init(
        self,
        monitoring_service_address: MonitoringServiceAddress,
        one_to_n_address: OneToNAddress,
    ) -> TransactionHash: ...
    def _deposit_preconditions(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
        token: Token,
    ) -> Tuple[TokenAmount, TokenAmount]: ...
    @contextmanager
    def _deposit_inflight(
        self, beneficiary: Address, total_deposit: TokenAmount
    ) -> Iterator[InflightDeposit]: ...
    def _deposit_check_result(
        self,
        transaction_sent: Optional[TransactionSent],
        token: Token,
        beneficiary: Address,
        total_deposit: TokenAmount,
        amount_to_deposit: TokenAmount,
    ) -> TransactionHash: ...
    def _plan_withdraw_preconditions(
        self, amount_to_plan_withdraw: TokenAmount, given_block_identifier: BlockIdentifier
    ) -> None: ...
    def _withdraw_preconditions(
        self, amount_to_withdraw: TokenAmount, given_block_identifier: BlockIdentifier
    ) -> None: ...
    def _plan_withdraw_check_result(
        self, transaction_sent: Optional[TransactionSent], amount_to_plan_withdraw: TokenAmount
    ) -> TransactionMined: ...
    def _withdraw_check_result(
        self,
        transaction_sent: Optional[TransactionSent],
        amount_to_withdraw: TokenAmount,
        token: Token,
        previous_token_balance: TokenAmount,
    ) -> TransactionHash: ...