from web3.types import BlockNumber, BlockIdentifier
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    ChannelID,
    Dict,
    FeeAmount,
    List,
    PaymentAmount,
    PaymentID,
    Secret,
    TargetAddress,
    TokenNetworkAddress,
)
from raiden_contracts.contract_manager import ContractManager
from eth_utils import keccak

def get_netting_channel_closed_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict]:
    ...

def get_netting_channel_deposit_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict]:
    ...

def get_netting_channel_settled_events(
    proxy_manager: ProxyManager,
    token_network_address: TokenNetworkAddress,
    netting_channel_identifier: ChannelID,
    contract_manager: ContractManager,
    from_block: BlockNumber = GENESIS_BLOCK_NUMBER,
    to_block: BlockIdentifier = BLOCK_ID_LATEST,
) -> List[Dict]:
    ...

def wait_both_channel_open(
    app0: RaidenService,
    app1: RaidenService,
    registry_address: Address,
    token_address: Address,
    retry_timeout: float,
) -> None:
    ...

def test_channel_new(
    raiden_chain: List[RaidenService],
    retry_timeout: float,
    token_addresses: List[Address],
) -> None:
    ...

def test_channel_deposit(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    retry_timeout: float,
    token_addresses: List[Address],
) -> None:
    ...

def test_query_events(
    raiden_chain: List[RaidenService],
    token_addresses: List[Address],
    deposit: Balance,
    settle_timeout: BlockNumber,
    retry_timeout: float,
    contract_manager: ContractManager,
    blockchain_type: str,
) -> None:
    ...

def test_secret_revealed_on_chain(
    raiden_chain: List[RaidenService],
    deposit: Balance,
    settle_timeout: BlockNumber,
    token_addresses: List[Address],
    retry_interval_initial: float,
) -> None:
    ...

def test_clear_closed_queue(
    raiden_network: List[RaidenService],
    token_addresses: List[Address],
    network_wait: float,
) -> None:
    ...