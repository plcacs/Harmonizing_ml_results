from dataclasses import dataclass
from datetime import datetime
from typing import Union
from urllib.parse import urlparse
from uuid import UUID
import click
import gevent
import structlog
from eth_utils import decode_hex, to_canonical_address, to_hex
from requests.exceptions import RequestException
from requests.sessions import Session
from web3 import Web3
from raiden.constants import DEFAULT_HTTP_REQUEST_TIMEOUT, MATRIX_AUTO_SELECT_SERVER, PFS_PATHS_REQUEST_TIMEOUT, ZERO_TOKENS, RoutingMode
from raiden.exceptions import PFSError, PFSReturnedError, RaidenError, ServiceRequestFailed, ServiceRequestIOURejected
from raiden.messages.metadata import RouteMetadata
from raiden.network.proxies.service_registry import ServiceRegistry
from raiden.network.utils import get_response_json
from raiden.utils.formatting import to_checksum_address
from raiden.utils.http import TimeoutHTTPAdapter
from raiden.utils.signer import LocalSigner
from raiden.utils.system import get_system_spec
from raiden.utils.transfers import to_rdn
from raiden.utils.typing import Address, AddressMetadata, Any, BlockExpiration, BlockIdentifier, BlockNumber, BlockTimeout, ChainID, Dict, InitiatorAddress, List, OneToNAddress, Optional, PaymentAmount, PrivateKey, Signature, TargetAddress, TokenAmount, TokenNetworkAddress, TokenNetworkRegistryAddress, Tuple
from raiden_contracts.utils.proofs import sign_one_to_n_iou

log: structlog.BoundLogger = structlog.get_logger(__name__)
iou_semaphore: gevent.lock.BoundedSemaphore = gevent.lock.BoundedSemaphore()

@dataclass(frozen=True)
class PFSInfo:
    url: str
    price: int
    chain_id: ChainID
    token_network_registry_address: TokenNetworkRegistryAddress
    user_deposit_address: Address
    payment_address: Address
    message: str
    operator: str
    version: str
    confirmed_block_number: BlockNumber
    matrix_server: str

@dataclass
class PFSConfig:
    info: PFSInfo
    iou_timeout: int
    max_paths: int
    maximum_fee: int

@dataclass
class IOU:
    sender: Address
    receiver: Address
    one_to_n_address: OneToNAddress
    amount: TokenAmount
    expiration_block: BlockExpiration
    chain_id: ChainID
    signature: Signature = None

    def sign(self, privkey: PrivateKey) -> None:
        self.signature = Signature(sign_one_to_n_iou(privatekey=privkey, sender=to_checksum_address(self.sender), receiver=to_checksum_address(self.receiver), amount=self.amount, expiration_block=self.expiration_block, one_to_n_address=to_checksum_address(self.one_to_n_address), chain_id=self.chain_id))

    def as_json(self) -> Dict[str, Union[str, int]]:
        data = dict(sender=to_checksum_address(self.sender), receiver=to_checksum_address(self.receiver), one_to_n_address=to_checksum_address(self.one_to_n_address), amount=self.amount, expiration_block=self.expiration_block, chain_id=self.chain_id)
        if self.signature is not None:
            data['signature'] = to_hex(self.signature)
        return data

def get_pfs_info(url: str) -> PFSInfo:
    ...

def get_valid_pfs_url(service_registry: ServiceRegistry, index_in_service_registry: int, block_identifier: BlockIdentifier, pathfinding_max_fee: int) -> Union[str, None]:
    ...

def get_random_pfs(service_registry: ServiceRegistry, block_identifier: BlockIdentifier, pathfinding_max_fee: int) -> Union[str, None]:
    ...

def configure_pfs_or_exit(pfs_url: str, routing_mode: RoutingMode, service_registry: ServiceRegistry, node_chain_id: ChainID, token_network_registry_address: TokenNetworkRegistryAddress, pathfinding_max_fee: int) -> PFSInfo:
    ...

def check_pfs_for_production(service_registry: ServiceRegistry, pfs_info: PFSInfo) -> None:
    ...

def get_last_iou(url: str, token_network_address: TokenNetworkAddress, sender: Address, receiver: Address, privkey: PrivateKey) -> Union[IOU, None]:
    ...

def make_iou(pfs_config: PFSConfig, our_address: Address, one_to_n_address: OneToNAddress, privkey: PrivateKey, block_number: BlockNumber, chain_id: ChainID, offered_fee: TokenAmount) -> IOU:
    ...

def update_iou(iou: IOU, privkey: PrivateKey, added_amount: TokenAmount = ZERO_TOKENS, expiration_block: BlockExpiration = None) -> IOU:
    ...

def create_current_iou(pfs_config: PFSConfig, token_network_address: TokenNetworkAddress, one_to_n_address: OneToNAddress, our_address: Address, privkey: PrivateKey, block_number: BlockNumber, chain_id: ChainID, offered_fee: TokenAmount, scrap_existing_iou: bool = False) -> IOU:
    ...

def post_pfs_paths(url: str, token_network_address: TokenNetworkAddress, payload: Dict[str, Union[str, int]]) -> Tuple[List[Any], UUID]:
    ...

def _query_address_metadata(pfs_config: PFSConfig, user_address: Address) -> AddressMetadata:
    ...

def _query_paths(pfs_config: PFSConfig, our_address: Address, privkey: PrivateKey, current_block_number: BlockNumber, token_network_address: TokenNetworkAddress, one_to_n_address: OneToNAddress, chain_id: ChainID, route_from: Address, route_to: Address, value: PaymentAmount, pfs_wait_for_block: BlockNumber) -> Tuple[List[Any], Union[None, UUID]]:
    ...

def post_pfs_feedback(routing_mode: RoutingMode, pfs_config: PFSConfig, token_network_address: TokenNetworkAddress, route: List[Address], token: UUID, successful: bool) -> None:
    ...

class PFSProxy:
    def __init__(self, pfs_config: PFSConfig):
        ...

    def query_address_metadata(self, address: Address) -> AddressMetadata:
        ...

    def query_paths(self, our_address: Address, privkey: PrivateKey, current_block_number: BlockNumber, token_network_address: TokenNetworkAddress, one_to_n_address: OneToNAddress, chain_id: ChainID, route_from: Address, route_to: Address, value: PaymentAmount, pfs_wait_for_block: BlockNumber) -> Tuple[List[Any], Union[None, UUID]]:
        ...
