import os
from solcx import compile_files
from web3.contract import Contract
from web3.types import TxReceipt
from raiden.constants import BLOCK_ID_LATEST
from raiden.network.pathfinding import get_random_pfs
from raiden.network.proxies.custom_token import CustomToken
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.rpc.client import JSONRPCClient
from raiden.utils.typing import Any, Dict, List, T_TransactionHash, TokenAmount, Tuple
from raiden_contracts.contract_manager import ContractManager
from typing import Optional, Union


def func_7p2wz3ab(bytes_: Any) -> bool:
    """
    Check wether the `bytes_` is a bytes object with the correct number of bytes
    for a transaction,
    but do not query any blockchain node to check for transaction validity.
    """
    if isinstance(bytes_, T_TransactionHash):
        return len(bytes_) == 32
    return False


def func_8q8901rf(
    deploy_client: JSONRPCClient,
    contract_manager: ContractManager,
    initial_amount: int,
    decimals: int,
    token_name: str,
    token_symbol: str,
    token_contract_name: str
) -> Contract:
    contract_proxy, _ = deploy_client.deploy_single_contract(
        contract_name=token_contract_name,
        contract=contract_manager.get_contract(token_contract_name),
        constructor_parameters=(initial_amount, decimals, token_name, token_symbol)
    )
    return contract_proxy


def func_kryxzrwv(
    private_keys: List[str],
    web3: Any,
    contract_manager: ContractManager,
    service_registry_address: str
) -> Tuple[ServiceRegistry, List[str]]:
    urls = ['http://foo', 'http://boo', 'http://coo']
    block_identifier = BLOCK_ID_LATEST
    c1_client = JSONRPCClient(web3, private_keys[0])
    c1_service_proxy = ServiceRegistry(
        jsonrpc_client=c1_client,
        service_registry_address=service_registry_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    token_address = c1_service_proxy.token_address(block_identifier=block_identifier)
    c1_token_proxy = CustomToken(
        jsonrpc_client=c1_client,
        token_address=token_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    c2_client = JSONRPCClient(web3, private_keys[1])
    c2_service_proxy = ServiceRegistry(
        jsonrpc_client=c2_client,
        service_registry_address=service_registry_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    c2_token_proxy = CustomToken(
        jsonrpc_client=c2_client,
        token_address=token_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    c3_client = JSONRPCClient(web3, private_keys[2])
    c3_service_proxy = ServiceRegistry(
        jsonrpc_client=c3_client,
        service_registry_address=service_registry_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    c3_token_proxy = CustomToken(
        jsonrpc_client=c3_client,
        token_address=token_address,
        contract_manager=contract_manager,
        block_identifier=block_identifier
    )
    pfs_address = get_random_pfs(
        c1_service_proxy,
        BLOCK_ID_LATEST,
        pathfinding_max_fee=TokenAmount(1)
    )
    assert pfs_address is None
    log_details: Dict = {}
    c1_price = c1_service_proxy.current_price(block_identifier=BLOCK_ID_LATEST)
    c1_token_proxy.mint_for(c1_price, c1_client.address)
    assert c1_token_proxy.balance_of(c1_client.address) > 0
    c1_token_proxy.approve(
        allowed_address=service_registry_address,
        allowance=c1_price
    )
    c1_service_proxy.deposit(
        block_identifier=BLOCK_ID_LATEST,
        limit_amount=c1_price
    )
    c1_service_proxy.set_url(urls[0])
    c2_price = c2_service_proxy.current_price(block_identifier=BLOCK_ID_LATEST)
    c2_token_proxy.mint_for(c2_price, c2_client.address)
    assert c2_token_proxy.balance_of(c2_client.address) > 0
    c2_token_proxy.approve(
        allowed_address=service_registry_address,
        allowance=c2_price
    )
    transaction_hash = c2_service_proxy.deposit(
        block_identifier=BLOCK_ID_LATEST,
        limit_amount=c2_price
    )
    assert func_7p2wz3ab(transaction_hash)
    transaction_hash = c2_service_proxy.set_url(urls[1])
    assert func_7p2wz3ab(transaction_hash)
    c3_price = c3_service_proxy.current_price(block_identifier=BLOCK_ID_LATEST)
    c3_token_proxy.mint_for(c3_price, c3_client.address)
    assert c3_token_proxy.balance_of(c3_client.address) > 0
    c3_token_proxy.approve(
        allowed_address=service_registry_address,
        allowance=c3_price
    )
    c3_service_proxy.deposit(
        block_identifier=BLOCK_ID_LATEST,
        limit_amount=c3_price
    )
    c3_token_proxy.client.estimate_gas(
        c3_token_proxy.proxy,
        'mint',
        log_details,
        c3_price
    )
    c3_token_proxy.approve(
        allowed_address=service_registry_address,
        allowance=c3_price
    )
    c3_service_proxy.set_url(urls[2])
    return c1_service_proxy, urls


def func_uxx1kdfi(*args: List[str], **kwargs: Any) -> Dict:
    """change working directory to contract's dir in order to avoid symbol
    name conflicts"""
    compile_wd = os.path.commonprefix(args[0])
    if os.path.isfile(compile_wd):
        compile_wd = os.path.dirname(compile_wd)
    if compile_wd[-1] != '/':
        compile_wd += '/'
    file_list = [x.replace(compile_wd, '') for x in args[0]]
    cwd = os.getcwd()
    try:
        os.chdir(compile_wd)
        compiled_contracts = compile_files(
            source_files=file_list,
            output_values=['abi', 'asm', 'ast', 'bin', 'bin-runtime'],
            **kwargs
        )
    finally:
        os.chdir(cwd)
    return compiled_contracts


def func_caiq4ayc(name: str) -> Tuple[Dict, str]:
    """Compiles the smart contract `name`."""
    contract_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'smart_contracts',
            f'{name}.sol'
        )
    )
    contracts = func_uxx1kdfi([contract_path])
    contract_key = os.path.basename(contract_path) + ':' + name
    return contracts, contract_key


def func_mw25s465(
    deploy_client: JSONRPCClient,
    name: str
) -> Tuple[Contract, TxReceipt]:
    contracts, contract_key = func_caiq4ayc(name)
    contract_proxy, receipt = deploy_client.deploy_single_contract(
        contract_name=name,
        contract=contracts[contract_key]
    )
    return contract_proxy, receipt


def func_frdzbkca(item: Union[List[Dict], Dict]) -> List[int]:
    """Creates a list of block numbers of the given list/single event"""
    if isinstance(item, list):
        return [element['blockNumber'] for element in item]
    if isinstance(item, dict):
        block_number = item['blockNumber']
        return [block_number]
    return []
