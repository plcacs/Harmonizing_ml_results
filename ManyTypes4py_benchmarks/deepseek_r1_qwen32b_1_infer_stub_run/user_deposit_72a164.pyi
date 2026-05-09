from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from eth_utils import Address
from gevent.event import AsyncResult
from gevent.threading import Lock
from raiden.utils.typing import (
    BlockIdentifier,
    BlockNumber,
    MonitoringServiceAddress,
    OneToNAddress,
    TokenAddress,
    TokenAmount,
    TransactionHash,
)
from raiden_contracts.contract_manager import ContractManager
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.exceptions import RaidenRecoverableError
from web3.types import BlockData

@dataclass
class InflightDeposit:
    total_deposit: TokenAmount
    async_result: AsyncResult

@dataclass(frozen=True)
class WithdrawPlan:
    withdraw_amount: TokenAmount
    withdraw_block: BlockNumber

class UserDeposit:
    def __init__(
        self,
        jsonrpc_client: Any,
        user_deposit_address: Address,
        contract_manager: ContractManager,
        proxy_manager: ProxyManager,
        block_identifier: BlockIdentifier,
    ) -> None:
        ...

    def token_address(self, block_identifier: BlockIdentifier) -> TokenAddress:
        ...

    def monitoring_service_address(self, block_identifier: BlockIdentifier) -> MonitoringServiceAddress:
        ...

    def one_to_n_address(self, block_identifier: BlockIdentifier) -> OneToNAddress:
        ...

    def get_total_deposit(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def get_balance(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def whole_balance(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def whole_balance_limit(self, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def get_withdraw_delay(self) -> BlockNumber:
        ...

    def get_withdraw_plan(self, withdrawer_address: Address, block_identifier: BlockIdentifier) -> WithdrawPlan:
        ...

    def init(
        self,
        monitoring_service_address: MonitoringServiceAddress,
        one_to_n_address: OneToNAddress,
        given_block_identifier: BlockIdentifier,
    ) -> None:
        ...

    def _init(
        self,
        monitoring_service_address: MonitoringServiceAddress,
        one_to_n_address: OneToNAddress,
    ) -> TransactionHash:
        ...

    def effective_balance(self, address: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        ...

    def deposit(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        ...

    def approve_and_deposit(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        ...

    def plan_withdraw(
        self,
        amount: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> Tuple[TransactionHash, BlockNumber]:
        ...

    def withdraw(
        self,
        amount: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash:
        ...

    def _deposit_preconditions(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
        token: Any,
    ) -> Tuple[TokenAmount, TokenAmount]:
        ...

    @contextmanager
    def _deposit_inflight(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
    ) -> Iterator[InflightDeposit]:
        ...

    def _deposit_check_result(
        self,
        transaction_sent: Optional[Any],
        token: Any,
        beneficiary: Address,
        total_deposit: TokenAmount,
        amount_to_deposit: TokenAmount,
    ) -> TransactionHash:
        ...

    def _plan_withdraw_preconditions(
        self,
        amount_to_plan_withdraw: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> None:
        ...

    def _withdraw_preconditions(
        self,
        amount_to_withdraw: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> None:
        ...

    def _plan_withdraw_check_result(
        self,
        transaction_sent: Optional[Any],
        amount_to_plan_withdraw: TokenAmount,
    ) -> Any:
        ...

    def _withdraw_check_result(
        self,
        transaction_sent: Optional[Any],
        amount_to_withdraw: TokenAmount,
        token: Any,
        previous_token_balance: TokenAmount,
    ) -> TransactionHash:
        ...