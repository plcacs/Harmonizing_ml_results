from typing import List, NamedTuple

class EthNodeDescription(NamedTuple):
    blockchain_type: str

class AccountDescription(NamedTuple):
    pass

class GenesisDescription(NamedTuple):
    prefunded_accounts: List[str]
    random_marker: str
    chain_id: int

def geth_clique_extradata(extra_vanity: str, extra_seal: str) -> str:
    ...

def parity_extradata(random_marker: str) -> str:
    ...

def geth_to_cmd(node: dict, datadir: str, chain_id: int, verbosity: str) -> List[str]:
    ...

def parity_to_cmd(node: dict, datadir: str, chain_id: int, chain_spec: str, verbosity: str) -> List[str]:
    ...

def geth_keystore(datadir: str) -> str:
    ...

def geth_keyfile(datadir: str) -> str:
    ...

def eth_create_account_file(keyfile_path: str, privkey: str) -> None:
    ...

def parity_generate_chain_spec(genesis_path: str, genesis_description: GenesisDescription, seal_account: str) -> None:
    ...

def geth_generate_poa_genesis(genesis_path: str, genesis_description: GenesisDescription, seal_account: str) -> None:
    ...

def geth_init_datadir(datadir: str, genesis_path: str) -> None:
    ...

def parity_keystore(datadir: str) -> str:
    ...

def parity_keyfile(datadir: str) -> str:
    ...

def eth_check_balance(web3, accounts_addresses: List[str], retries: int = 10) -> None:
    ...

def eth_node_config(node_pkey: str, p2p_port: int, rpc_port: int, **extra_config) -> dict:
    ...

def eth_node_config_set_bootnodes(nodes_configuration: List[dict]) -> None:
    ...

def eth_node_to_datadir(node_address: str, base_datadir: str) -> str:
    ...

def eth_node_to_logpath(node_config: dict, base_logdir: str) -> str:
    ...

def geth_prepare_datadir(datadir: str, genesis_file: str) -> None:
    ...

def eth_nodes_to_cmds(nodes_configuration: List[dict], eth_node_descs: List[EthNodeDescription], base_datadir: str, genesis_file: str, chain_id: int, verbosity: str) -> List[List[str]]:
    ...

def eth_run_nodes(eth_node_descs: List[EthNodeDescription], nodes_configuration: List[dict], base_datadir: str, genesis_file: str, chain_id: int, random_marker: str, verbosity: str, logdir: str) -> None:
    ...

def run_private_blockchain(web3, eth_nodes: List[dict], base_datadir: str, log_dir: str, verbosity: str, genesis_description: GenesisDescription) -> None:
    ...
