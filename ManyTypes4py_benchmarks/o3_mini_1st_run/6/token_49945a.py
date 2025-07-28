import structlog
from eth_utils import encode_hex, is_binary_address
from gevent.lock import RLock
from raiden.constants import BLOCK_ID_LATEST, GAS_LIMIT_FOR_TOKEN_CONTRACT_CALL
from raiden.exceptions import RaidenRecoverableError
from raiden.network.rpc.client import JSONRPCClient, check_address_has_code_handle_pruned_block, check_transaction_failure, was_transaction_successfully_mined
from raiden.utils.typing import ABI, Address, Any, Balance, BlockIdentifier, BlockNumber, Dict, Optional, TokenAddress, TokenAmount, TransactionHash
from raiden_contracts.constants import CONTRACT_HUMAN_STANDARD_TOKEN
from raiden_contracts.contract_manager import ContractManager

log = structlog.get_logger(__name__)
GAS_REQUIRED_FOR_APPROVE: int = 58792

class Token:
    def __init__(self, jsonrpc_client: JSONRPCClient, token_address: Address, contract_manager: ContractManager, block_identifier: BlockIdentifier) -> None:
        proxy = jsonrpc_client.new_contract_proxy(self.abi(contract_manager), Address(token_address))
        if not is_binary_address(token_address):
            raise ValueError('token_address must be a valid address')
        check_address_has_code_handle_pruned_block(jsonrpc_client, Address(token_address), 'Token', given_block_identifier=block_identifier)
        self.address: Address = token_address
        self.client: JSONRPCClient = jsonrpc_client
        self.node_address: Address = jsonrpc_client.address
        self.proxy: Any = proxy
        self.token_lock: RLock = RLock()

    @staticmethod
    def abi(contract_manager: ContractManager) -> ABI:
        """Overwrittable by subclasses to change the proxies ABI."""
        return contract_manager.get_contract_abi(CONTRACT_HUMAN_STANDARD_TOKEN)

    def allowance(self, owner: Address, spender: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        result = self.proxy.functions.allowance(owner, spender).call(block_identifier=block_identifier)
        return TokenAmount(result)

    def approve(self, allowed_address: Address, allowance: TokenAmount) -> TransactionHash:
        """Approve `allowed_address` to transfer up to `deposit` amount of token.

        Note:
            For channel deposit please use the channel proxy, since it does
            additional validations.
            We assume there to be sufficient balance as a precondition if this
            is called, so it is not checked as a precondition here.
        """
        with self.token_lock:
            log_details: Dict[str, Any] = {}
            error_prefix: str = 'Call to approve will fail'
            estimated_transaction: Optional[Any] = self.client.estimate_gas(self.proxy, 'approve', log_details, allowed_address, allowance)
            if estimated_transaction is not None:
                error_prefix = 'Call to approve failed'
                transaction_sent: Any = self.client.transact(estimated_transaction)
                transaction_mined: Any = self.client.poll_transaction(transaction_sent)
                if not was_transaction_successfully_mined(transaction_mined):
                    failed_receipt: Dict[str, Any] = transaction_mined.receipt
                    failed_at_blockhash: str = encode_hex(failed_receipt['blockHash'])
                    check_transaction_failure(transaction_mined, self.client)
                    balance: TokenAmount = self.balance_of(self.client.address, failed_at_blockhash)
                    if balance < allowance:
                        msg: str = f'{error_prefix} Your balance of {balance} is below the required amount of {{allowance}}.'
                        if balance == 0:
                            msg += ' Note: The balance was 0, which may also happen if the contract is not a valid ERC20 token (balanceOf method missing).'
                        raise RaidenRecoverableError(msg)
                    raise RaidenRecoverableError(f'{error_prefix}. The reason is unknown, you have enough tokens for the requested allowance and enough eth to pay the gas. There may be a problem with the token contract.')
                else:
                    return transaction_mined.transaction_hash
            else:
                failed_at: Dict[str, Any] = self.client.get_block(BLOCK_ID_LATEST)
                failed_at_blockhash: str = encode_hex(failed_at['hash'])
                failed_at_blocknumber: BlockNumber = failed_at['number']
                self.client.check_for_insufficient_eth(transaction_name='approve', transaction_executed=False, required_gas=GAS_REQUIRED_FOR_APPROVE, block_identifier=failed_at_blocknumber)
                balance: TokenAmount = self.balance_of(self.client.address, failed_at_blockhash)
                if balance < allowance:
                    msg: str = f'{error_prefix} Your balance of {balance} is below the required amount of {{allowance}}.'
                    if balance == 0:
                        msg += ' Note: The balance was 0, which may also happen if the contract is not a valid ERC20 token (balanceOf method missing).'
                    raise RaidenRecoverableError(msg)
                raise RaidenRecoverableError(f'{error_prefix} Gas estimation failed for unknown reason. Please make sure the contract is a valid ERC20 token.')

    def balance_of(self, address: Address, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> TokenAmount:
        """Return the balance of `address`."""
        result = self.proxy.functions.balanceOf(address).call(block_identifier=block_identifier)
        return TokenAmount(result)

    def total_supply(self, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> Optional[TokenAmount]:
        """Return the total supply of the token at the given block identifier.

        Because Token is just an interface, it is not possible to check the
        bytecode during the proxy instantiation. This means it is possible for
        the proxy to be instantiated with a a smart contrat address of the
        wrong type (a non ERC20 contract), or a partial implementation of the
        ERC20 standard (the function totalSupply is missing). If that happens
        this method will return `None`.
        """
        total_supply_result = self.proxy.functions.totalSupply().call(block_identifier=block_identifier)
        if isinstance(total_supply_result, int):
            return TokenAmount(total_supply_result)
        return None

    def transfer(self, to_address: Address, amount: TokenAmount) -> TransactionHash:
        """Transfer `amount` tokens to `to_address`.

        Note:
            We assume there to be sufficient balance as a precondition if
            this is called, so that is not checked as a precondition here.
        """
        def check_for_insufficient_token_balance(block_number: BlockNumber) -> None:
            failed_at_hash: str = encode_hex(self.client.blockhash_from_blocknumber(block_number))
            self.client.check_for_insufficient_eth(transaction_name='transfer', transaction_executed=False, required_gas=GAS_LIMIT_FOR_TOKEN_CONTRACT_CALL, block_identifier=block_number)
            balance: TokenAmount = self.balance_of(self.client.address, failed_at_hash)
            if balance < amount:
                msg: str = f'Call to transfer will fail. Your balance of {balance} is below the required amount of {amount}.'
                if balance == 0:
                    msg += ' Note: The balance was 0, which may also happen if the contract is not a valid ERC20 token (balanceOf method missing).'
                raise RaidenRecoverableError(msg)

        with self.token_lock:
            log_details: Dict[str, Any] = {}
            estimated_transaction: Optional[Any] = self.client.estimate_gas(self.proxy, 'transfer', log_details, to_address, amount)
            if estimated_transaction is not None:
                transaction_sent: Any = self.client.transact(estimated_transaction)
                transaction_mined: Any = self.client.poll_transaction(transaction_sent)
                if was_transaction_successfully_mined(transaction_mined):
                    return transaction_mined.transaction_hash
                failed_at_block_number: BlockNumber = BlockNumber(transaction_mined.receipt['blockNumber'])
                check_for_insufficient_token_balance(failed_at_block_number)
                raise RaidenRecoverableError('Call to transfer failed for unknown reason. Please make sure the contract is a valid ERC20 token.')
            else:
                failed_at_block_number: BlockNumber = self.client.get_block(BLOCK_ID_LATEST)['number']
                check_for_insufficient_token_balance(failed_at_block_number)
                raise RaidenRecoverableError('Call to transfer will fail. Gas estimation failed for unknown reason. Please make sure the contract is a valid ERC20 token.')