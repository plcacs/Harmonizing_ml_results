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
    total_deposit = field(default_factory=lambda: TokenAmount(0))
    async_result = field(default_factory=AsyncResult)

@dataclass(frozen=True)
class WithdrawPlan:
    pass

class UserDeposit:

    def __init__(self, jsonrpc_client: Union[raiden.utils.BlockIdentifier, raiden.network.rpc.clienJSONRPCClient, raiden_contracts.contract_manager.ContractManager], user_deposit_address: Union[raiden.utils.UserDepositAddress, raiden.utils.BlockSpecification, raiden.utils.MonitoringServiceAddress], contract_manager: Union[raiden.network.rpc.clienJSONRPCClient, raiden_contracts.contract_manager.ContractManager, raiden.utils.BlockIdentifier], proxy_manager: Union[ProxyManager, raiden_contracts.contract_manager.ContractManager], block_identifier: Union[raiden.utils.BlockIdentifier, raiden.network.rpc.clienJSONRPCClient, raiden_contracts.contract_manager.ContractManager]) -> None:
        if not is_binary_address(user_deposit_address):
            raise ValueError('Expected binary address format for token nework')
        check_address_has_code_handle_pruned_block(client=jsonrpc_client, address=Address(user_deposit_address), contract_name=CONTRACT_USER_DEPOSIT, given_block_identifier=block_identifier)
        self.client = jsonrpc_client
        self.address = user_deposit_address
        self.node_address = self.client.address
        self.contract_manager = contract_manager
        self.gas_measurements = gas_measurements(self.contract_manager.contracts_version)
        self.proxy_manager = proxy_manager
        self.proxy = jsonrpc_client.new_contract_proxy(abi=self.contract_manager.get_contract_abi(CONTRACT_USER_DEPOSIT), contract_address=Address(user_deposit_address))
        self._inflight_deposits = {}
        self._withdraw_lock = Lock()

    def token_address(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> TokenAddress:
        return TokenAddress(to_canonical_address(self.proxy.functions.token().call(block_identifier=block_identifier)))

    def monitoring_service_address(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> MonitoringServiceAddress:
        return MonitoringServiceAddress(to_canonical_address(self.proxy.functions.msc_address().call(block_identifier=block_identifier)))

    def one_to_n_address(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> OneToNAddress:
        return OneToNAddress(to_canonical_address(self.proxy.functions.one_to_n_address().call(block_identifier=block_identifier)))

    def get_total_deposit(self, address: Union[raiden.utils.BlockSpecification, raiden.utils.Address, raiden.utils.TokenAmount], block_identifier: Union[raiden.utils.BlockSpecification, raiden.utils.Address, raiden.utils.TokenAmount]) -> Union[int, str]:
        return self.proxy.functions.total_deposit(address).call(block_identifier=block_identifier)

    def get_balance(self, address: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification], block_identifier: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> Union[int, str, list[int]]:
        return self.proxy.functions.balances(address).call(block_identifier=block_identifier)

    def whole_balance(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.Address]) -> TokenAmount:
        return TokenAmount(self.proxy.functions.whole_balance().call(block_identifier=block_identifier))

    def whole_balance_limit(self, block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.Address]) -> TokenAmount:
        return TokenAmount(self.proxy.functions.whole_balance_limit().call(block_identifier=block_identifier))

    def get_withdraw_delay(self) -> BlockNumber:
        return BlockNumber(self.proxy.functions.withdraw_delay().call())

    def get_withdraw_plan(self, withdrawer_address: Union[raiden.utils.BlockIdentifier, raiden.utils.Address], block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.Address]) -> WithdrawPlan:
        withdraw_amount, withdraw_block = self.proxy.functions.withdraw_plans(withdrawer_address).call(block_identifier=block_identifier)
        return WithdrawPlan(withdraw_amount=TokenAmount(withdraw_amount), withdraw_block=BlockNumber(withdraw_block))

    def init(self, monitoring_service_address: Union[raiden.utils.BlockIdentifier, raiden.utils.MonitoringServiceAddress, raiden.utils.OneToNAddress], one_to_n_address: Union[raiden.utils.OneToNAddress, raiden.utils.MonitoringServiceAddress, raiden.utils.BlockIdentifier], given_block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.MonitoringServiceAddress, raiden.utils.OneToNAddress]) -> Union[str, cmk.utils.type_defs.HostAddress, None, int]:
        """Initialize the UserDeposit contract with MS and OneToN addresses"""
        check_address_has_code_handle_pruned_block(client=self.client, address=Address(monitoring_service_address), contract_name=CONTRACT_MONITORING_SERVICE, given_block_identifier=given_block_identifier)
        check_address_has_code_handle_pruned_block(client=self.client, address=Address(one_to_n_address), contract_name=CONTRACT_ONE_TO_N, given_block_identifier=given_block_identifier)
        try:
            existing_monitoring_service_address = self.monitoring_service_address(block_identifier=given_block_identifier)
            existing_one_to_n_address = self.one_to_n_address(block_identifier=given_block_identifier)
        except ValueError:
            pass
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        else:
            if existing_monitoring_service_address != EMPTY_ADDRESS:
                msg = f'MonitoringService contract address is already set to {to_checksum_address(existing_monitoring_service_address)}'
                raise BrokenPreconditionError(msg)
            if existing_one_to_n_address != EMPTY_ADDRESS:
                msg = f'OneToN contract address is already set to {to_checksum_address(existing_one_to_n_address)}'
                raise BrokenPreconditionError(msg)
        return self._init(monitoring_service_address=monitoring_service_address, one_to_n_address=one_to_n_address)

    def _init(self, monitoring_service_address: Union[raiden.utils.MonitoringServiceAddress, raiden.utils.OneToNAddress, bytes], one_to_n_address: Union[raiden.utils.MonitoringServiceAddress, raiden.utils.OneToNAddress, bytes]):
        log_details = {}
        estimated_transaction = self.client.estimate_gas(self.proxy, 'init', log_details, monitoring_service_address, one_to_n_address)
        if estimated_transaction is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blocknumber = failed_at['number']
            self.client.check_for_insufficient_eth(transaction_name='init', transaction_executed=False, required_gas=self.gas_measurements['UserDeposit.init'], block_identifier=failed_at_blocknumber)
            existing_monitoring_service_address = self.monitoring_service_address(block_identifier=failed_at_blocknumber)
            existing_one_to_n_address = self.one_to_n_address(block_identifier=failed_at_blocknumber)
            if existing_monitoring_service_address != EMPTY_ADDRESS:
                msg = f'MonitoringService contract address was set to {to_checksum_address(existing_monitoring_service_address)}'
                raise RaidenRecoverableError(msg)
            if existing_one_to_n_address != EMPTY_ADDRESS:
                msg = f'OneToN contract address was set to {to_checksum_address(existing_one_to_n_address)}'
                raise RaidenRecoverableError(msg)
            raise RaidenRecoverableError('Deposit failed of unknown reason')
        else:
            transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                failed_at_blocknumber = BlockNumber(transaction_mined.receipt['blockNumber'])
                existing_monitoring_service_address = self.monitoring_service_address(block_identifier=failed_at_blocknumber)
                existing_one_to_n_address = self.one_to_n_address(block_identifier=failed_at_blocknumber)
                if existing_monitoring_service_address != EMPTY_ADDRESS:
                    msg = f'MonitoringService contract address was set to {to_checksum_address(existing_monitoring_service_address)}'
                    raise RaidenRecoverableError(msg)
                if existing_one_to_n_address != EMPTY_ADDRESS:
                    msg = f'OneToN contract address was set to {to_checksum_address(existing_one_to_n_address)}'
                    raise RaidenRecoverableError(msg)
                raise RaidenRecoverableError('Deposit failed of unknown reason')
            return transaction_mined.transaction_hash

    def effective_balance(self, address: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification], block_identifier: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> Union[int, raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]:
        """The user's balance with planned withdrawals deducted."""
        balance = self.proxy.functions.effectiveBalance(address).call(block_identifier=block_identifier)
        if balance == b'':
            raise RuntimeError("Call to 'effectiveBalance' returned nothing")
        return balance

    def deposit(self, beneficiary: Union[raiden.utils.TokenAmount, raiden.utils.Address, raiden.utils.BlockSpecification], total_deposit: Union[raiden.utils.TokenAmount, raiden.utils.Address, raiden.utils.FeeAmount], given_block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification, raiden.utils.TokenAddress]) -> Union[TransactionHash, bytes, dict, raiden.utils.TransactionHash]:
        """Increase the total deposit of the beneficiary's account to `total_deposit`."""
        token_address = self.token_address(given_block_identifier)
        token = self.proxy_manager.token(token_address=token_address, block_identifier=given_block_identifier)
        previous_total_deposit, amount_to_deposit = self._deposit_preconditions(beneficiary, total_deposit, given_block_identifier, token)
        current_inflight = self._inflight_deposits.get(beneficiary)
        if current_inflight is not None and current_inflight.total_deposit >= total_deposit:
            return TransactionHash(current_inflight.async_result.get())
        with self._deposit_inflight(beneficiary, total_deposit) as inflight_deposit:
            log_details = {'given_block_identifier': format_block_id(given_block_identifier), 'previous_total_deposit': previous_total_deposit}
            estimated_transaction = self.client.estimate_gas(self.proxy, 'deposit', log_details, beneficiary, total_deposit)
            transaction_hash = None
            if estimated_transaction is not None:
                transaction_hash = self.client.transact(estimated_transaction)
            transaction_mined_hash = self._deposit_check_result(transaction_hash, token, beneficiary, total_deposit, amount_to_deposit)
            inflight_deposit.async_result.set(transaction_mined_hash)
            return transaction_mined_hash

    def approve_and_deposit(self, beneficiary: Union[raiden.utils.TokenAmount, raiden.utils.Address, raiden.utils.BlockSpecification], total_deposit: Union[raiden.utils.TokenAmount, raiden.utils.Address, raiden.utils.BlockSpecification], given_block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.TokenAddress, raiden.utils.BlockSpecification]) -> Union[TransactionHash, bytes, raiden.utils.TransactionHash, dict]:
        """Deposit provided amount into the user-deposit contract
        to the beneficiary's account.

        This function will also call approve with the *same* amount of tokens
        for the deposit. Note that this will overwrite the existing value, so
        large allowances are not useful when this method is used.
        """
        token_address = self.token_address(given_block_identifier)
        token = self.proxy_manager.token(token_address=token_address, block_identifier=given_block_identifier)
        previous_total_deposit, amount_to_deposit = self._deposit_preconditions(beneficiary, total_deposit, given_block_identifier, token)
        log_details = {'given_block_identifier': format_block_id(given_block_identifier), 'previous_total_deposit': previous_total_deposit}
        current_inflight = self._inflight_deposits.get(beneficiary)
        if current_inflight is not None and current_inflight.total_deposit >= total_deposit:
            return TransactionHash(current_inflight.async_result.get())
        with self._deposit_inflight(beneficiary, total_deposit) as inflight_deposit:
            transaction_sent = None
            with token.token_lock:
                allowance = TokenAmount(amount_to_deposit + 1)
                token.approve(allowed_address=Address(self.address), allowance=allowance)
                estimated_transaction = self.client.estimate_gas(self.proxy, 'deposit', log_details, beneficiary, total_deposit)
                if estimated_transaction is not None:
                    transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined_hash = self._deposit_check_result(transaction_sent, token, beneficiary, total_deposit, amount_to_deposit)
            inflight_deposit.async_result.set(transaction_mined_hash)
            return transaction_mined_hash

    def plan_withdraw(self, amount: Union[raiden.utils.TokenAmount, raiden.utils.BlockIdentifier, int], given_block_identifier: Union[raiden.utils.BlockIdentifier, int, raiden.utils.TokenAmount]) -> tuple[BlockNumber]:
        """Announce that you plan to withdraw tokens from the UserDeposit contract

        Returns the block number at which the withdraw is ready.
        """
        self._plan_withdraw_preconditions(amount, given_block_identifier)
        with self._withdraw_lock:
            estimated_transaction = self.client.estimate_gas(self.proxy, 'planWithdraw', {}, amount)
            transaction_sent = None
            if estimated_transaction is not None:
                transaction_sent = self.client.transact(estimated_transaction)
            transaction_mined = self._plan_withdraw_check_result(transaction_sent=transaction_sent, amount_to_plan_withdraw=amount)
        return (transaction_mined.transaction_hash, BlockNumber(transaction_mined.receipt['blockNumber'] + self.get_withdraw_delay()))

    def withdraw(self, amount: Union[raiden.utils.TokenAmount, raiden.utils.Address, int], given_block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.BlockSpecification]) -> Union[dict[int, typing.Any], typing.Callable, str]:
        """Withdraw tokens from UDC, requires a mature withdraw plan"""
        token_address = self.token_address(given_block_identifier)
        token = self.proxy_manager.token(token_address=token_address, block_identifier=given_block_identifier)
        self._withdraw_preconditions(amount_to_withdraw=amount, given_block_identifier=given_block_identifier)
        previous_token_balance = TokenAmount(token.balance_of(self.node_address, given_block_identifier))
        with self._withdraw_lock:
            estimated_transaction = self.client.estimate_gas(self.proxy, 'withdraw', {}, amount)
            transaction_sent = None
            if estimated_transaction is not None:
                transaction_sent = self.client.transact(estimated_transaction)
            return self._withdraw_check_result(transaction_sent=transaction_sent, amount_to_withdraw=amount, token=token, previous_token_balance=previous_token_balance)

    def _deposit_preconditions(self, beneficiary: Union[raiden.utils.Address, raiden.network.proxies.token.Token, raiden.utils.BlockIdentifier], total_deposit: Union[raiden.utils.TokenAmount, raiden.utils.PaymentAmount, raiden.utils.BlockSpecification], given_block_identifier: Union[raiden.utils.TokenAmount, raiden.utils.BlockIdentifier, raiden.utils.Address], token: Union[raiden.network.proxies.token.Token, raiden.utils.Address, raiden.utils.BlockSpecification]) -> tuple[typing.Union[web3.types.Wei,TokenAmount]]:
        try:
            previous_total_deposit = self.get_total_deposit(address=beneficiary, block_identifier=given_block_identifier)
            current_balance = token.balance_of(address=self.node_address, block_identifier=given_block_identifier)
            whole_balance = self.whole_balance(block_identifier=given_block_identifier)
            whole_balance_limit = self.whole_balance_limit(block_identifier=given_block_identifier)
        except ValueError:
            previous_total_deposit = self.get_total_deposit(address=beneficiary, block_identifier=BLOCK_ID_PENDING)
            amount_to_deposit = TokenAmount(total_deposit - previous_total_deposit)
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        else:
            amount_to_deposit = TokenAmount(total_deposit - previous_total_deposit)
            if whole_balance + amount_to_deposit > UINT256_MAX:
                msg = f'Current whole balance is {whole_balance}. The new deposit of {amount_to_deposit} would lead to an overflow.'
                raise BrokenPreconditionError(msg)
            if whole_balance + amount_to_deposit > whole_balance_limit:
                msg = f'Current whole balance is {whole_balance}. With the new deposit of {amount_to_deposit}, the deposit limit of {whole_balance_limit} would be exceeded.'
                raise BrokenPreconditionError(msg)
            if total_deposit < previous_total_deposit:
                msg = f'Current total deposit {previous_total_deposit} is already larger than the requested total deposit amount {total_deposit}'
                raise BrokenPreconditionError(msg)
            if total_deposit == previous_total_deposit:
                msg = f'Current total deposit {previous_total_deposit} is the same as the requested total deposit amount.'
                raise BrokenPreconditionError(msg)
            if current_balance < amount_to_deposit:
                msg = f'new_total_deposit - previous_total_deposit = {amount_to_deposit} can not be larger than the available balance {current_balance}, for token at address {to_checksum_address(token.address)}'
                raise BrokenPreconditionError(msg)
        return (previous_total_deposit, amount_to_deposit)

    @contextmanager
    def _deposit_inflight(self, beneficiary: Union[raiden.utils.TokenAmount, typing.Callable, web3.contracContract], total_deposit: Union[raiden.utils.TokenAmount, int, raiden.utils.Dict]) -> typing.Generator[InflightDeposit]:
        """Updates the `_inflight_deposits` dictionary to handle concurrent deposits.

        Note:
          This must be called after `_deposit_preconditions`.
          The returned InflightDeposit.async_result has to be set within the 'with' block,
          when no Exception is raised
        """
        async_result = AsyncResult()
        current_inflight = InflightDeposit(total_deposit, async_result)
        self._inflight_deposits[beneficiary] = current_inflight
        try:
            yield current_inflight
        except Exception as e:
            async_result.set_exception(e)
            raise
        else:
            assert async_result.ready(), "The AsyncResult created by this contextmanager must beset within the 'with' block!"
        finally:
            stored_inflight = self._inflight_deposits.get(beneficiary)
            if stored_inflight == current_inflight:
                del self._inflight_deposits[beneficiary]

    def _deposit_check_result(self, transaction_sent: Union[raiden.network.rpc.clienTransactionSent, None, raiden.utils.BlockIdentifier, str], token: Union[raiden.utils.Address, raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.token.Token], beneficiary: Union[raiden.utils.Address, raiden.utils.BlockSpecification, raiden.network.blockchain_service.BlockChainService], total_deposit: Union[raiden.utils.TokenAmount, int], amount_to_deposit: Union[raiden.utils.TokenAmount, raiden.utils.Address, raiden.network.proxies.token.Token]):
        if transaction_sent is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blocknumber = failed_at['number']
            self.client.check_for_insufficient_eth(transaction_name='deposit', transaction_executed=False, required_gas=self.gas_measurements['UserDeposit.deposit'], block_identifier=failed_at_blocknumber)
            latest_deposit = self.get_total_deposit(address=beneficiary, block_identifier=failed_at_blocknumber)
            amount_to_deposit = TokenAmount(total_deposit - latest_deposit)
            allowance = token.allowance(owner=self.node_address, spender=Address(self.address), block_identifier=failed_at_blocknumber)
            whole_balance = self.whole_balance(block_identifier=failed_at_blocknumber)
            whole_balance_limit = self.whole_balance_limit(block_identifier=failed_at_blocknumber)
            if allowance < amount_to_deposit:
                msg = 'The allowance is insufficient. Check concurrent deposits for the same user deposit but different proxies.'
                raise RaidenRecoverableError(msg)
            if token.balance_of(self.node_address, failed_at_blocknumber) < amount_to_deposit:
                msg = 'The address doesnt have enough tokens'
                raise RaidenRecoverableError(msg)
            if latest_deposit < total_deposit:
                msg = 'Deposit amount did not increase after deposit transaction'
                raise RaidenRecoverableError(msg)
            if whole_balance + amount_to_deposit > UINT256_MAX:
                msg = f'Current whole balance is {whole_balance}. The new deposit of {amount_to_deposit} would lead to an overflow.'
                raise RaidenRecoverableError(msg)
            if whole_balance + amount_to_deposit > whole_balance_limit:
                msg = f'Current whole balance is {whole_balance}. With the new deposit of {amount_to_deposit}, the deposit limit of {whole_balance_limit} would be exceeded.'
                raise RaidenRecoverableError(msg)
            raise RaidenRecoverableError('Deposit failed of unknown reason')
        else:
            transaction_mined = self.client.poll_transaction(transaction_sent)
            if not was_transaction_successfully_mined(transaction_mined):
                failed_at_blocknumber = BlockNumber(transaction_mined.receipt['blockNumber'])
                latest_deposit = self.get_total_deposit(address=beneficiary, block_identifier=failed_at_blocknumber)
                amount_to_deposit = TokenAmount(total_deposit - latest_deposit)
                allowance = token.allowance(owner=self.node_address, spender=Address(self.address), block_identifier=failed_at_blocknumber)
                whole_balance = self.whole_balance(block_identifier=failed_at_blocknumber)
                whole_balance_limit = self.whole_balance_limit(block_identifier=failed_at_blocknumber)
                if latest_deposit >= total_deposit:
                    msg = 'Deposit amount already increased after another transaction'
                    raise RaidenRecoverableError(msg)
                if allowance < amount_to_deposit:
                    msg = 'The allowance is insufficient. Check concurrent deposits for the same token network but different proxies.'
                    raise RaidenRecoverableError(msg)
                if token.balance_of(self.node_address, failed_at_blocknumber) < amount_to_deposit:
                    msg = 'Transaction failed and balance decreased unexpectedly. This could be a bug in Raiden or a mallicious ERC20 Token.'
                    raise RaidenRecoverableError(msg)
                if whole_balance + amount_to_deposit > UINT256_MAX:
                    msg = f'Current whole balance is {whole_balance}. The new deposit of {amount_to_deposit} caused an overflow.'
                    raise RaidenRecoverableError(msg)
                if whole_balance + amount_to_deposit > whole_balance_limit:
                    msg = f'Current whole balance is {whole_balance}. With the new deposit of {amount_to_deposit}, the deposit limit of {whole_balance_limit} was exceeded.'
                    raise RaidenRecoverableError(msg)
                if latest_deposit < total_deposit:
                    msg = 'Deposit amount did not increase after deposit transaction'
                    raise RaidenRecoverableError(msg)
                raise RaidenRecoverableError('Deposit failed of unknown reason')
            else:
                return transaction_mined.transaction_hash

    def _plan_withdraw_preconditions(self, amount_to_plan_withdraw: Union[raiden.utils.TokenAmount, raiden.utils.BlockIdentifier, raiden.utils.Address], given_block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.BlockSpecification]) -> None:
        """Check if a WithdrawPlan for the given amount can be created."""
        if amount_to_plan_withdraw <= 0:
            raise BrokenPreconditionError('Planned withdraw amount must be greater than zero.')
        try:
            current_balance = self.get_balance(address=self.node_address, block_identifier=given_block_identifier)
        except ValueError:
            return
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        if current_balance < amount_to_plan_withdraw:
            raise BrokenPreconditionError(f"Can't create WithdrawPlan for amount {amount_to_plan_withdraw}, it exceeds the current balance of {current_balance}.")

    def _withdraw_preconditions(self, amount_to_withdraw: Union[raiden.utils.TokenAmount, raiden.utils.BlockIdentifier], given_block_identifier: Union[raiden.utils.Address, raiden.utils.BlockIdentifier, raiden.utils.TokenAddress]) -> None:
        try:
            withdraw_plan = self.get_withdraw_plan(withdrawer_address=self.node_address, block_identifier=given_block_identifier)
            whole_balance = self.whole_balance(block_identifier=given_block_identifier)
        except ValueError:
            return
        except BadFunctionCallOutput:
            raise_on_call_returned_empty(given_block_identifier)
        if amount_to_withdraw > withdraw_plan.withdraw_amount:
            raise BrokenPreconditionError(f"Can't withdraw {amount_to_withdraw}, current withdraw plan only allows for {withdraw_plan.withdraw_amount}.")
        given_block_number = self.client.get_block(given_block_identifier)['number']
        if withdraw_plan.withdraw_block > given_block_number:
            raise BrokenPreconditionError(f"Can't withdraw at block {given_block_number}. The current withdraw plan requires block number {withdraw_plan.withdraw_block}.")
        if whole_balance - amount_to_withdraw < 0:
            raise BrokenPreconditionError(f'The current whole balance is {whole_balance}. The withdraw of {amount_to_withdraw} would lead to an underflow.')

    def _plan_withdraw_check_result(self, transaction_sent: Union[raiden.network.rpc.clienTransactionSent, None, int, list], amount_to_plan_withdraw: Union[raiden.utils.TokenAmount, raiden.utils.BlockTimeout, raiden.network.proxies.token.Token]) -> Union[int, str]:
        if transaction_sent is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blocknumber = failed_at['number']
            self.client.check_for_insufficient_eth(transaction_name='planWithdraw', transaction_executed=False, required_gas=self.gas_measurements['UserDeposit.planWithdraw'], block_identifier=failed_at_blocknumber)
            raise RaidenRecoverableError('Plan withdraw transaction failed to be sent for an unknown reason.')
        transaction_mined = self.client.poll_transaction(transaction_sent)
        if not was_transaction_successfully_mined(transaction_mined):
            if amount_to_plan_withdraw <= 0:
                raise RaidenRecoverableError(f'Planned withdraw amount was <= 0: {amount_to_plan_withdraw}.')
            failed_at_blocknumber = BlockNumber(transaction_mined.receipt['blockNumber'])
            current_balance = self.get_balance(address=self.node_address, block_identifier=failed_at_blocknumber)
            if current_balance < amount_to_plan_withdraw:
                raise RaidenRecoverableError(f"Couldn't plan withdraw because planned amount {amount_to_plan_withdraw} exceeded current balance of {current_balance}.")
            raise RaidenRecoverableError('Plan withdraw failed for an unknown reason.')
        return transaction_mined

    def _withdraw_check_result(self, transaction_sent: Union[raiden.network.rpc.clienTransactionSent, None, typing.Callable[..., typing.Any], raiden.transfer.state.ChainState], amount_to_withdraw: Union[raiden.utils.TokenAmount, raiden.network.proxies.token.Token, web3.contracContract], token: Union[raiden.network.proxies.token.Token, raiden.utils.TokenAmount, typing.Callable], previous_token_balance: Union[raiden.utils.TokenAmount, raiden.network.proxies.token.Token, raiden.utils.Address]):
        if transaction_sent is None:
            failed_at = self.client.get_block(BLOCK_ID_LATEST)
            failed_at_blocknumber = failed_at['number']
            self.client.check_for_insufficient_eth(transaction_name='withdraw', transaction_executed=False, required_gas=self.gas_measurements['UserDeposit.withdraw'], block_identifier=failed_at_blocknumber)
            raise RaidenRecoverableError('Withdraw transaction failed to be sent for an unknown reason.')
        transaction_mined = self.client.poll_transaction(transaction_sent)
        if not was_transaction_successfully_mined(transaction_mined):
            failed_at_blocknumber = BlockNumber(transaction_mined.receipt['blockNumber'])
            withdraw_plan = self.get_withdraw_plan(withdrawer_address=self.node_address, block_identifier=failed_at_blocknumber)
            whole_balance = self.whole_balance(block_identifier=failed_at_blocknumber)
            if amount_to_withdraw > withdraw_plan.withdraw_amount:
                raise RaidenRecoverableError(f"Couldn't withdraw {amount_to_withdraw}, current withdraw plan only allows for {withdraw_plan.withdraw_amount}.")
            if withdraw_plan.withdraw_block > failed_at_blocknumber:
                raise RaidenRecoverableError(f"Couldn't withdraw at block {failed_at_blocknumber}. The current withdraw plan requires block number {withdraw_plan.withdraw_block}.")
            if whole_balance - amount_to_withdraw < 0:
                raise RaidenRecoverableError(f'The current whole balance is {whole_balance}. The withdraw of {amount_to_withdraw} would have lead to an underflow.')
            current_token_balance = TokenAmount(token.balance_of(self.node_address, block_identifier=failed_at_blocknumber))
            if current_token_balance != previous_token_balance + amount_to_withdraw:
                raise RaidenRecoverableError('Token transfer during withdraw failed.')
            raise RaidenRecoverableError('Withdraw failed for an unknown reason.')
        else:
            return transaction_mined.transaction_hash