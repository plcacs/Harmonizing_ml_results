import structlog
from eth_utils import encode_hex, is_binary_address
from gevent.lock import RLock
from raiden.constants import BLOCK_ID_LATEST, GAS_LIMIT_FOR_TOKEN_CONTRACT_CALL
from raiden.exceptions import RaidenRecoverableError
from raiden.network.rpc.client import JSONRPCClient, check_address_has_code_handle_pruned_block, check_transaction_failure, was_transaction_successfully_mined
from raiden.utils.typing import ABI, Address, Any, Balance, BlockIdentifier, BlockNumber, Dict, Optional, TokenAddress, TokenAmount, TransactionHash
from raiden_contracts.constants import CONTRACT_HUMAN_STANDARD_TOKEN
from raiden_contracts.contract_manager import ContractManager
log: structlog.BoundLogger = structlog.get_logger(__name__)
GAS_REQUIRED_FOR_APPROVE: int = 58792

class Token:

    def __init__(self, jsonrpc_client: JSONRPCClient, token_address: TokenAddress, contract_manager: ContractManager, block_identifier: BlockIdentifier):
        proxy = jsonrpc_client.new_contract_proxy(self.abi(contract_manager), Address(token_address))
        if not is_binary_address(token_address):
            raise ValueError('token_address must be a valid address')
        check_address_has_code_handle_pruned_block(jsonrpc_client, Address(token_address), 'Token', given_block_identifier=block_identifier)
        self.address: TokenAddress = token_address
        self.client: JSONRPCClient = jsonrpc_client
        self.node_address: Address = jsonrpc_client.address
        self.proxy = proxy
        self.token_lock: RLock = RLock()

    @staticmethod
    def abi(contract_manager: ContractManager) -> ABI:
        """Overwrittable by subclasses to change the proxies ABI."""
        return contract_manager.get_contract_abi(CONTRACT_HUMAN_STANDARD_TOKEN)

    def allowance(self, owner: Address, spender: Address, block_identifier: BlockIdentifier) -> TokenAmount:
        return TokenAmount(self.proxy.functions.allowance(owner, spender).call(block_identifier=block_identifier))

    def approve(self, allowed_address: Address, allowance: TokenAmount) -> TransactionHash:
        ...

    def balance_of(self, address: Address, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> TokenAmount:
        ...

    def total_supply(self, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> Optional[TokenAmount]:
        ...

    def transfer(self, to_address: Address, amount: TokenAmount) -> TransactionHash:
        ...
