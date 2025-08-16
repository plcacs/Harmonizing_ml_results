from web3 import Web3
from raiden.network.rpc.client import JSONRPCClient
from raiden.utils.typing import T_TransactionHash, TokenAmount

def is_tx_hash_bytes(bytes_: T_TransactionHash) -> bool:
    ...

def deploy_token(deploy_client: JSONRPCClient, contract_manager: ContractManager, initial_amount: int, decimals: int, token_name: str, token_symbol: str, token_contract_name: str) -> Contract:
    ...

def deploy_service_registry_and_set_urls(private_keys: List[str], web3: Web3, contract_manager: ContractManager, service_registry_address: str) -> Tuple[ServiceRegistry, List[str]]:
    ...

def compile_files_cwd(*args: List[str], **kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ...

def compile_test_smart_contract(name: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
    ...

def deploy_rpc_test_contract(deploy_client: JSONRPCClient, name: str) -> Tuple[Contract, TxReceipt]:
    ...

def get_list_of_block_numbers(item: Any) -> List[int]:
    ...
