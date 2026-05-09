from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, Iterator, Optional, Tuple, Union

from eth_utils import Address, MonitoringServiceAddress, OneToNAddress, TokenAddress, UserDepositAddress
from gevent.event import AsyncResult
from gevent.threading import Lock
from raiden.exceptions import BrokenPreconditionError, RaidenRecoverableError
from raiden.network.rpc.client import JSONRPCClient, TransactionHash
from raiden.utils.typing import BlockIdentifier, BlockNumber, TokenAmount
from structlog import Logger

log: Logger = ...

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
        jsonrpc_client: JSONRPCClient,
        user_deposit_address: UserDepositAddress,
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

    @contextmanager
    def _deposit_inflight(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
    ) -> ContextManager[InflightDeposit]:
        ...