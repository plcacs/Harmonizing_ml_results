import os
import pytest
from eth_typing import URI
from eth_utils import to_canonical_address
from web3 import HTTPProvider, Web3
from raiden.constants import GENESIS_BLOCK_NUMBER
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.utils.eth_node import AccountDescription, GenesisDescription, run_private_blockchain
from raiden.tests.utils.network import jsonrpc_services
from raiden.tests.utils.tests import cleanup_tasks
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import TokenAddress
from raiden_contracts.constants import CONTRACT_CUSTOM_TOKEN, CONTRACT_HUMAN_STANDARD_TOKEN

@pytest.fixture
def web3(deploy_key: Union[typing.Iterator, list[raiden.utils.upgrades.UpgradeRecord]], eth_nodes_configuration: Union[raiden.utils.BlockNumber, raiden.transfer.mediated_transfer.state.LockedTransferSignedState, raiden.constants.EthClient], private_keys: list[raiden.utils.upgrades.UpgradeRecord], account_genesis_eth_balance: Union[raiden.utils.Address, raiden.transfer.state.NettingChannelState, str], random_marker: Union[raiden.utils.Address, int, str], tmpdir: Union[str, bool, dict], chain_id: Union[raiden.utils.Address, int, str], logs_storage: Union[str, web3.Web3, raiden.utils.Address], blockchain_type: Union[str, web3.Web3, raiden.utils.Address]) -> typing.Generator[Web3]:
    """Starts a private chain with accounts funded."""
    keys_to_fund = sorted(set(private_keys + [deploy_key]))
    host = '127.0.0.1'
    rpc_port = eth_nodes_configuration[0].rpc_port
    endpoint = f'http://{host}:{rpc_port}'
    web3 = Web3(HTTPProvider(URI(endpoint)))
    accounts_to_fund = [AccountDescription(privatekey_to_address(key), account_genesis_eth_balance) for key in keys_to_fund]
    base_datadir = str(tmpdir)
    base_logdir = os.path.join(logs_storage, blockchain_type)
    genesis_description = GenesisDescription(prefunded_accounts=accounts_to_fund, chain_id=chain_id, random_marker=random_marker)
    eth_node_runner = run_private_blockchain(web3=web3, eth_nodes=eth_nodes_configuration, base_datadir=base_datadir, log_dir=base_logdir, verbosity='info', genesis_description=genesis_description)
    with eth_node_runner:
        yield web3
    cleanup_tasks()

@pytest.fixture
def deploy_client(deploy_key: Union[str, raiden.utils.Address], web3: Union[str, raiden.utils.Address], blockchain_type: Union[str, typing.Mapping, None, bool]) -> JSONRPCClient:
    return JSONRPCClient(web3=web3, privkey=deploy_key)

@pytest.fixture
def proxy_manager(deploy_key: Union[list[str], bool], deploy_client: Union[raiden.utils.Address, raiden.network.rpc.clienJSONRPCClient], contract_manager: Union[raiden.utils.Address, raiden.network.rpc.clienJSONRPCClient]) -> ProxyManager:
    return ProxyManager(rpc_client=deploy_client, contract_manager=contract_manager, metadata=ProxyManagerMetadata(token_network_registry_deployed_at=GENESIS_BLOCK_NUMBER, filters_start_at=GENESIS_BLOCK_NUMBER))

@pytest.fixture
def blockchain_services(proxy_manager: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], private_keys: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], secret_registry_address: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], service_registry_address: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], token_network_registry_address: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], web3: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3], contract_manager: Union[raiden_contracts.contract_manager.ContractManager, raiden.network.proxies.secret_registry.SecretRegistry, web3.Web3]) -> Union[str, None, typing.Callable, GenesisDescription]:
    return jsonrpc_services(proxy_manager=proxy_manager, private_keys=private_keys, secret_registry_address=secret_registry_address, service_registry_address=service_registry_address, token_network_registry_address=token_network_registry_address, web3=web3, contract_manager=contract_manager)

@pytest.fixture
def unregistered_token(token_amount: Union[raiden_contracts.contract_manager.ContractManager, str, raiden.utils.Balance], deploy_client: Union[raiden_contracts.contract_manager.ContractManager, str, raiden.utils.Balance], contract_manager: Union[raiden_contracts.contract_manager.ContractManager, str, raiden.utils.Balance]) -> TokenAddress:
    contract_proxy, _ = deploy_client.deploy_single_contract(contract_name=CONTRACT_HUMAN_STANDARD_TOKEN, contract=contract_manager.get_contract(CONTRACT_HUMAN_STANDARD_TOKEN), constructor_parameters=(token_amount, 2, 'raiden', 'Rd'))
    return TokenAddress(to_canonical_address(contract_proxy.address))

@pytest.fixture
def unregistered_custom_token(token_amount: Union[raiden_contracts.contract_manager.ContractManager, web3.contracContract, int], deploy_client: Union[raiden_contracts.contract_manager.ContractManager, web3.contracContract, int], contract_manager: Union[raiden_contracts.contract_manager.ContractManager, web3.contracContract, int]) -> TokenAddress:
    contract_proxy, _ = deploy_client.deploy_single_contract(contract_name=CONTRACT_CUSTOM_TOKEN, contract=contract_manager.get_contract(CONTRACT_CUSTOM_TOKEN), constructor_parameters=(token_amount, 2, 'raiden', 'Rd'))
    return TokenAddress(to_canonical_address(contract_proxy.address))