from typing import Any, List
import gevent
import structlog
from eth_utils import encode_hex, is_binary_address
from gevent.event import AsyncResult
from gevent.lock import Semaphore
from raiden.constants import BLOCK_ID_PENDING, GAS_REQUIRED_PER_SECRET_IN_BATCH, GAS_REQUIRED_REGISTER_SECRET_BATCH_BASE
from raiden.exceptions import NoStateForBlockIdentifier, RaidenRecoverableError, RaidenUnrecoverableError
from raiden.network.rpc.client import JSONRPCClient, check_address_has_code_handle_pruned_block, was_transaction_successfully_mined
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.smart_contracts import safe_gas_limit
from raiden.utils.typing import MYPY_ANNOTATION, Address, BlockIdentifier, BlockNumber, Dict, Optional, Secret, SecretHash, SecretRegistryAddress, TransactionHash, Union
from raiden_contracts.constants import CONTRACT_SECRET_REGISTRY
from raiden_contracts.contract_manager import ContractManager
log = structlog.get_logger(__name__)

class SecretRegistry:

    def __init__(self, jsonrpc_client: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.rpc.clienJSONRPCClient, raiden.utils.BlockIdentifier], secret_registry_address: Union[raiden.utils.OneToNAddress, raiden.utils.BlockIdentifier, raiden.network.rpc.clienJSONRPCClient], contract_manager: Union[raiden.utils.BlockNumber, raiden_contracts.contract_manager.ContractManager], block_identifier: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.rpc.clienJSONRPCClient, raiden.utils.BlockIdentifier]) -> None:
        if not is_binary_address(secret_registry_address):
            raise ValueError('Expected binary address format for secret registry')
        self.contract_manager = contract_manager
        check_address_has_code_handle_pruned_block(client=jsonrpc_client, address=Address(secret_registry_address), contract_name=CONTRACT_SECRET_REGISTRY, given_block_identifier=block_identifier)
        proxy = jsonrpc_client.new_contract_proxy(abi=self.contract_manager.get_contract_abi(CONTRACT_SECRET_REGISTRY), contract_address=Address(secret_registry_address))
        self.address = secret_registry_address
        self.proxy = proxy
        self.client = jsonrpc_client
        self.node_address = self.client.address
        self.open_secret_transactions = {}
        self._open_secret_transactions_lock = Semaphore()

    def register_secret(self, secret: Union[raiden.utils.TransactionHash, str]) -> None:
        self.register_secret_batch([secret])

    def register_secret_batch(self, secrets: Union[bytes, str]) -> list:
        """Register a batch of secrets. Check if they are already registered at
        the given block identifier."""
        secrets_to_register = []
        secrethashes_to_register = []
        secrethashes_not_sent = []
        secrets_results = []
        transaction_result = AsyncResult()
        wait_for = set()
        with self._open_secret_transactions_lock:
            verification_block_hash = self.client.get_confirmed_blockhash()
            for secret in secrets:
                secrethash = sha256_secrethash(secret)
                secrethash_hex = encode_hex(secrethash)
                other_result = self.open_secret_transactions.get(secret)
                if other_result is not None:
                    wait_for.add(other_result)
                    secrethashes_not_sent.append(secrethash_hex)
                    secrets_results.append(other_result)
                elif not self.is_secret_registered(secrethash, verification_block_hash):
                    secrets_to_register.append(secret)
                    secrethashes_to_register.append(secrethash_hex)
                    self.open_secret_transactions[secret] = transaction_result
                    secrets_results.append(transaction_result)
        if secrets_to_register:
            log_details = {'secrethashes_not_sent': secrethashes_not_sent}
            self._register_secret_batch(secrets_to_register, transaction_result, log_details)
        gevent.joinall(wait_for, raise_error=True)
        return [result.get() for result in secrets_results]

    def _register_secret_batch(self, secrets_to_register: Union[raiden.utils.BlockNumber, raiden.utils.ChannelMap, raiden.utils.TokenAmount], transaction_result: typing.Callable, log_details: Union[raiden.utils.BlockNumber, raiden.utils.BlockIdentifier, raiden.utils.Address]) -> None:
        estimated_transaction = self.client.estimate_gas(self.proxy, 'registerSecretBatch', log_details, secrets_to_register)
        msg = None
        transaction_mined = None
        if estimated_transaction is not None:
            gas_limit = safe_gas_limit(GAS_REQUIRED_REGISTER_SECRET_BATCH_BASE + len(secrets_to_register) * GAS_REQUIRED_PER_SECRET_IN_BATCH)
            assert estimated_transaction.estimated_gas <= gas_limit, f'Our safe gas calculation must be larger than the gas cost estimated by the ethereum node, but {estimated_transaction.estimated_gas} > {gas_limit}.'
            estimated_transaction.estimated_gas = gas_limit
            try:
                transaction_sent = self.client.transact(estimated_transaction)
                transaction_mined = self.client.poll_transaction(transaction_sent)
            except Exception as e:
                msg = f'Unexpected exception {e} at sending registerSecretBatch transaction.'
        with self._open_secret_transactions_lock:
            for secret in secrets_to_register:
                self.open_secret_transactions.pop(secret)
        unrecoverable_error = transaction_mined is None or not was_transaction_successfully_mined(transaction_mined)
        if unrecoverable_error:
            if transaction_mined is not None:
                receipt = transaction_mined.receipt
                if receipt['gasUsed'] == transaction_mined.startgas:
                    error = 'Secret registration failed because of a bug in either the solidity compiler, the running ethereum client, or a configuration error in Raiden.'
                else:
                    error = 'Secret registration failed because of a configuration bug or compiler bug. Please double check the secret smart contract is at version 0.4.0, if it is then a compiler bug was hit.'
                exception = RaidenUnrecoverableError(error)
                transaction_result.set_exception(exception)
                raise exception
            if estimated_transaction is not None:
                assert msg, 'Unexpected control flow, an exception should have been raised.'
                error = f'Sending the transaction for registerSecretBatch failed with: `{msg}`.  This happens if the same ethereum account is being used by more than one program which is not supported.'
                exception = RaidenUnrecoverableError(error)
                transaction_result.set_exception(exception)
                raise exception
            self.client.check_for_insufficient_eth(transaction_name='registerSecretBatch', transaction_executed=True, required_gas=GAS_REQUIRED_PER_SECRET_IN_BATCH * len(secrets_to_register), block_identifier=BLOCK_ID_PENDING)
            error = "Call to registerSecretBatch couldn't be done"
            exception = RaidenRecoverableError(error)
            transaction_result.set_exception(exception)
            raise exception
        assert transaction_mined is not None, MYPY_ANNOTATION
        transaction_result.set(transaction_mined.transaction_hash)

    def get_secret_registration_block_by_secrethash(self, secrethash: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.BlockSpecification], block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.BlockSpecification]) -> Union[None, bool, str, list]:
        """Return the block number at which the secret for `secrethash` was
        registered, None if the secret was never registered.
        """
        result = self.proxy.functions.getSecretRevealBlockHeight(secrethash).call(block_identifier=block_identifier)
        if result == 0:
            return None
        return result

    def is_secret_registered(self, secrethash: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.BlockSpecification], block_identifier: Union[raiden.utils.BlockIdentifier, raiden.utils.Address, raiden.utils.BlockSpecification]) -> bool:
        """True if the secret for `secrethash` is registered at `block_identifier`.

        Throws NoStateForBlockIdentifier if the given block_identifier
        is older than the pruning limit
        """
        if not self.client.can_query_state_for_block(block_identifier):
            raise NoStateForBlockIdentifier()
        block = self.get_secret_registration_block_by_secrethash(secrethash=secrethash, block_identifier=block_identifier)
        return block is not None