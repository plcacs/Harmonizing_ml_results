```python
from contextlib import contextmanager
from typing import Dict, Tuple, Iterator, Optional, Any
from dataclasses import dataclass
from gevent.event import AsyncResult
from gevent.threading import Lock
from raiden.utils.typing import (
    Address,
    Balance,
    BlockIdentifier,
    BlockNumber,
    MonitoringServiceAddress,
    OneToNAddress,
    TokenAddress,
    TokenAmount,
    TransactionHash,
    UserDepositAddress,
)
from raiden.network.rpc.client import JSONRPCClient, TransactionMined, TransactionSent
from raiden_contracts.contract_manager import ContractManager

log: Any = ...

@dataclass
class InflightDeposit:
    total_deposit: TokenAmount = ...
    async_result: AsyncResult = ...

@dataclass(frozen=True)
class WithdrawPlan:
    withdraw_amount: TokenAmount = ...
    withdraw_block: BlockNumber = ...

class UserDeposit:
    def __init__(
        self,
        jsonrpc_client: JSONRPCClient,
        user_deposit_address: UserDepositAddress,
        contract_manager: ContractManager,
        proxy_manager: Any,
        block_identifier: BlockIdentifier,
    ) -> None: ...
    
    @property
    def client(self) -> JSONRPCClient: ...
    @property
    def address(self) -> UserDepositAddress: ...
    @property
    def node_address(self) -> Address: ...
    @property
    def contract_manager(self) -> ContractManager: ...
    @property
    def gas_measurements(self) -> Dict[str, int]: ...
    @property
    def proxy_manager(self) -> Any: ...
    @property
    def proxy(self) -> Any: ...
    @property
    def _inflight_deposits(self) -> Dict[Address, InflightDeposit]: ...
    @property
    def _withdraw_lock(self) -> Lock: ...
    
    def token_address(self, block_identifier: BlockIdentifier) -> TokenAddress: ...
    def monitoring_service_address(self, block_identifier: BlockIdentifier) -> MonitoringServiceAddress: ...
    def one_to_n_address(self, block_identifier: BlockIdentifier) -> OneToNAddress: ...
    def get_total_deposit(self, address: Address, block_identifier: BlockIdentifier) -> Balance: ...
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
    def _init(
        self,
        monitoring_service_address: MonitoringServiceAddress,
        one_to_n_address: OneToNAddress,
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
        self,
        amount: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> Tuple[TransactionHash, BlockNumber]: ...
    def withdraw(
        self,
        amount: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> TransactionHash: ...
    def _deposit_preconditions(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
        given_block_identifier: BlockIdentifier,
        token: Any,
    ) -> Tuple[Balance, TokenAmount]: ...
    @contextmanager
    def _deposit_inflight(
        self,
        beneficiary: Address,
        total_deposit: TokenAmount,
    ) -> Iterator[InflightDeposit]: ...
    def _deposit_check_result(
        self,
        transaction_sent: Optional[TransactionSent],
        token: Any,
        beneficiary: Address,
        total_deposit: TokenAmount,
        amount_to_deposit: TokenAmount,
    ) -> TransactionHash: ...
    def _plan_withdraw_preconditions(
        self,
        amount_to_plan_withdraw: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> None: ...
    def _withdraw_preconditions(
        self,
        amount_to_withdraw: TokenAmount,
        given_block_identifier: BlockIdentifier,
    ) -> None: ...
    def _plan_withdraw_check_result(
        self,
        transaction_sent: Optional[TransactionSent],
        amount_to_plan_withdraw: TokenAmount,
    ) -> TransactionMined: ...
    def _withdraw_check_result(
        self,
        transaction_sent: Optional[TransactionSent],
        amount_to_withdraw: TokenAmount,
        token: Any,
        previous_token_balance: TokenAmount,
    ) -> TransactionHash: ...
```