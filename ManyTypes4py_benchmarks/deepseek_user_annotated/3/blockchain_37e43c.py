import os
from pathlib import Path
from typing import Generator, List, Sequence, Set

import pytest
from eth_typing import URI
from eth_utils import to_canonical_address
from web3 import HTTPProvider, Web3
from web3.types import Wei

from raiden.constants import GENESIS_BLOCK_NUMBER
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.utils.eth_node import (
    AccountDescription,
    EthNodeDescription,
    GenesisDescription,
    run_private_blockchain,
)
from raiden.tests.utils.network import jsonrpc_services
from raiden.tests.utils.tests import cleanup_tasks
from raiden.utils.keys import privatekey_to_address
from raiden.utils.typing import (
    BlockNumber,
    ChainID,
    PrivateKey,
    TokenAddress,
    TokenAmount,
)
from raiden_contracts.constants import CONTRACT_CUSTOM_TOKEN, CONTRACT_HUMAN_STANDARD_TOKEN
from raiden_contracts.contract_manager import ContractManager

# pylint: disable=redefined-outer-name,too-many-arguments,unused-argument,too-many-locals


@pytest.fixture
def web3(
    deploy_key: PrivateKey,
    eth_nodes_configuration: Sequence[EthNodeDescription],
    private_keys: List[PrivateKey],
    account_genesis_eth_balance: TokenAmount,
    random_marker: str,
    tmpdir: Path,
    chain_id: ChainID,
    logs_storage: Path,
    blockchain_type: str,
) -> Generator[Web3, None, None]:
    """Starts a private chain with accounts funded."""
    # include the deploy key in the list of funded accounts
    keys_to_fund: Set[PrivateKey] = set(private_keys + [deploy_key])

    host = "127.0.0.1"
    rpc_port = eth_nodes_configuration[0].rpc_port
    endpoint = f"http://{host}:{rpc_port}"
    web3 = Web3(HTTPProvider(URI(endpoint)))

    accounts_to_fund = [
        AccountDescription(privatekey_to_address(key), Wei(account_genesis_eth_balance))
        for key in sorted(keys_to_fund)
    ]

    # The private chain data is always discarded on the CI
    base_datadir = str(tmpdir)

    # Save the Ethereum node's log for debugging
    base_logdir = os.path.join(logs_storage, blockchain_type)

    genesis_description = GenesisDescription(
        prefunded_accounts=accounts_to_fund, chain_id=chain_id, random_marker=random_marker
    )
    eth_node_runner = run_private_blockchain(
        web3=web3,
        eth_nodes=eth_nodes_configuration,
        base_datadir=base_datadir,
        log_dir=base_logdir,
        verbosity="info",
        genesis_description=genesis_description,
    )
    with eth_node_runner:
        yield web3

    cleanup_tasks()


@pytest.fixture
def deploy_client(deploy_key: PrivateKey, web3: Web3, blockchain_type: str) -> JSONRPCClient:
    return JSONRPCClient(web3=web3, privkey=deploy_key)


@pytest.fixture
def proxy_manager(
    deploy_key: PrivateKey, deploy_client: JSONRPCClient, contract_manager: ContractManager
) -> ProxyManager:
    return ProxyManager(
        rpc_client=deploy_client,
        contract_manager=contract_manager,
        metadata=ProxyManagerMetadata(
            token_network_registry_deployed_at=GENESIS_BLOCK_NUMBER,
            filters_start_at=GENESIS_BLOCK_NUMBER,
        ),
    )


@pytest.fixture
def blockchain_services(
    proxy_manager: ProxyManager,
    private_keys: List[PrivateKey],
    secret_registry_address: TokenAddress,
    service_registry_address: TokenAddress,
    token_network_registry_address: TokenAddress,
    web3: Web3,
    contract_manager: ContractManager,
):
    return jsonrpc_services(
        proxy_manager=proxy_manager,
        private_keys=private_keys,
        secret_registry_address=secret_registry_address,
        service_registry_address=service_registry_address,
        token_network_registry_address=token_network_registry_address,
        web3=web3,
        contract_manager=contract_manager,
    )


@pytest.fixture
def unregistered_token(
    token_amount: TokenAmount, deploy_client: JSONRPCClient, contract_manager: ContractManager
) -> TokenAddress:
    contract_proxy, _ = deploy_client.deploy_single_contract(
        contract_name=CONTRACT_HUMAN_STANDARD_TOKEN,
        contract=contract_manager.get_contract(CONTRACT_HUMAN_STANDARD_TOKEN),
        constructor_parameters=(token_amount, 2, "raiden", "Rd"),
    )
    return TokenAddress(to_canonical_address(contract_proxy.address))


@pytest.fixture
def unregistered_custom_token(
    token_amount: TokenAmount, deploy_client: JSONRPCClient, contract_manager: ContractManager
) -> TokenAddress:
    contract_proxy, _ = deploy_client.deploy_single_contract(
        contract_name=CONTRACT_CUSTOM_TOKEN,
        contract=contract_manager.get_contract(CONTRACT_CUSTOM_TOKEN),
        constructor_parameters=(token_amount, 2, "raiden", "Rd"),
    )
    return TokenAddress(to_canonical_address(contract_proxy.address))
