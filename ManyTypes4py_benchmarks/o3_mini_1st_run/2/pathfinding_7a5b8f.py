import json
import random
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, ParseResult
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
from raiden.utils.typing import (
    Address,
    AddressMetadata,
    Any as RaidenAny,
    BlockExpiration,
    BlockIdentifier,
    BlockNumber,
    BlockTimeout,
    ChainID,
    Dict as RaidenDict,
    InitiatorAddress,
    List as RaidenList,
    OneToNAddress,
    Optional as RaidenOptional,
    PaymentAmount,
    PrivateKey,
    Signature,
    TargetAddress,
    TokenAmount,
    TokenNetworkAddress,
    TokenNetworkRegistryAddress,
    Tuple as RaidenTuple,
)

from raiden_contracts.utils.proofs import sign_one_to_n_iou

log = structlog.get_logger(__name__)
iou_semaphore = gevent.lock.BoundedSemaphore()


@dataclass(frozen=True)
class PFSInfo:
    url: str
    price: TokenAmount
    chain_id: ChainID
    token_network_registry_address: TokenNetworkRegistryAddress
    user_deposit_address: Address
    payment_address: str
    message: str
    operator: str
    version: str
    confirmed_block_number: int
    matrix_server: str


@dataclass
class PFSConfig:
    info: PFSInfo
    iou_timeout: int
    max_paths: int
    maximum_fee: TokenAmount


@dataclass
class IOU:
    sender: Address
    receiver: Address
    one_to_n_address: OneToNAddress
    amount: TokenAmount
    expiration_block: BlockExpiration
    chain_id: ChainID
    signature: Optional[Signature] = None

    def sign(self, privkey: PrivateKey) -> None:
        self.signature = Signature(
            sign_one_to_n_iou(
                privatekey=privkey,
                sender=to_checksum_address(self.sender),
                receiver=to_checksum_address(self.receiver),
                amount=self.amount,
                expiration_block=self.expiration_block,
                one_to_n_address=to_checksum_address(self.one_to_n_address),
                chain_id=self.chain_id,
            )
        )

    def as_json(self) -> Dict[str, Any]:
        data = dict(
            sender=to_checksum_address(self.sender),
            receiver=to_checksum_address(self.receiver),
            one_to_n_address=to_checksum_address(self.one_to_n_address),
            amount=self.amount,
            expiration_block=self.expiration_block,
            chain_id=self.chain_id,
        )
        if self.signature is not None:
            data['signature'] = to_hex(self.signature)
        return data


USER_AGENT_STR = 'Raiden/{raiden}/DB:{raiden_db_version}/{python_implementation}/{python_version}/{system}/{architecture}/{distribution}'.format(**get_system_spec()).replace(' ', '-')
session: Session = Session()
session.headers['User-Agent'] = USER_AGENT_STR
timeout_adapter = TimeoutHTTPAdapter(timeout=DEFAULT_HTTP_REQUEST_TIMEOUT)
session.mount('http://', timeout_adapter)
session.mount('https://', timeout_adapter)
MAX_PATHS_QUERY_ATTEMPTS: int = 2


def get_pfs_info(url: str) -> PFSInfo:
    try:
        response = session.get(f'{url}/api/v1/info')
        infos = get_response_json(response)
        matrix_server_info: ParseResult = urlparse(infos['matrix_server'])
        return PFSInfo(
            url=url,
            price=infos['price_info'],
            chain_id=infos['network_info']['chain_id'],
            token_network_registry_address=TokenNetworkRegistryAddress(to_canonical_address(infos['network_info']['token_network_registry_address'])),
            user_deposit_address=Address(to_canonical_address(infos['network_info']['user_deposit_address'])),
            payment_address=to_canonical_address(infos['payment_address']),
            message=infos['message'],
            operator=infos['operator'],
            version=infos['version'],
            confirmed_block_number=infos['network_info']['confirmed_block']['number'],
            matrix_server=matrix_server_info.netloc,
        )
    except RequestException as e:
        msg = 'Selected Pathfinding Service did not respond'
        raise ServiceRequestFailed(msg) from e
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        msg = 'Selected Pathfinding Service returned unexpected reply'
        raise ServiceRequestFailed(msg) from e


def get_valid_pfs_url(
    service_registry: ServiceRegistry,
    index_in_service_registry: int,
    block_identifier: BlockIdentifier,
    pathfinding_max_fee: TokenAmount,
) -> Optional[str]:
    address = service_registry.ever_made_deposits(block_identifier=block_identifier, index=index_in_service_registry)
    if not address:
        return None
    if not service_registry.has_valid_registration(service_address=address, block_identifier=block_identifier):
        return None
    url: Optional[str] = service_registry.get_service_url(block_identifier=block_identifier, service_address=address)
    if not url:
        return None
    try:
        pfs_info: PFSInfo = get_pfs_info(url)
    except ServiceRequestFailed:
        return None
    if pfs_info.price > pathfinding_max_fee:
        return None
    return url


def get_random_pfs(
    service_registry: ServiceRegistry,
    block_identifier: BlockIdentifier,
    pathfinding_max_fee: TokenAmount,
) -> Optional[str]:
    number_of_addresses: int = service_registry.ever_made_deposits_len(block_identifier=block_identifier)
    indices_to_try: List[int] = list(range(number_of_addresses))
    random.shuffle(indices_to_try)
    while indices_to_try:
        index: int = indices_to_try.pop()
        url: Optional[str] = get_valid_pfs_url(
            service_registry=service_registry,
            index_in_service_registry=index,
            block_identifier=block_identifier,
            pathfinding_max_fee=pathfinding_max_fee,
        )
        if url:
            return url
    return None


def configure_pfs_or_exit(
    pfs_url: str,
    routing_mode: RoutingMode,
    service_registry: Optional[ServiceRegistry],
    node_chain_id: ChainID,
    token_network_registry_address: TokenNetworkRegistryAddress,
    pathfinding_max_fee: TokenAmount,
) -> PFSInfo:
    msg = 'Invalid code path; configure_pfs needs routing mode PFS'
    assert routing_mode is RoutingMode.PFS, msg
    msg = "With PFS routing mode we shouldn't get to configure_pfs with pfs_address being None"
    assert pfs_url, msg
    if pfs_url == MATRIX_AUTO_SELECT_SERVER:
        if service_registry is None:
            raise RaidenError("Raiden was started with routing mode set to PFS, the pathfinding service address set to 'auto' but no service registry address was given. Either specifically provide a PFS address or provide a service registry address.")
        block_hash: BlockIdentifier = service_registry.client.get_confirmed_blockhash()
        maybe_pfs_url: Optional[str] = get_random_pfs(
            service_registry=service_registry,
            block_identifier=block_hash,
            pathfinding_max_fee=pathfinding_max_fee,
        )
        if maybe_pfs_url is None:
            raise RaidenError('Can not find any registered pathfinding service and basic routing is not used.')
        else:
            pfs_url = maybe_pfs_url
    try:
        pathfinding_service_info: PFSInfo = get_pfs_info(pfs_url)
    except ServiceRequestFailed as e:
        raise RaidenError(f'There was an error with the Pathfinding Service with address {pfs_url}. Raiden will shut down. Please try a different Pathfinding Service. \nError Message: {str(e)}')
    if pathfinding_service_info.price > 0 and (not pathfinding_service_info.payment_address):
        raise RaidenError(f'The Pathfinding Service at {pfs_url} did not provide a payment address. Raiden will shut down. Please try a different Pathfinding Service.')
    if node_chain_id != pathfinding_service_info.chain_id:
        raise RaidenError(f'Invalid reply from Pathfinding Service {pfs_url}\nPathfinding Service is not operating on the same network ({pathfinding_service_info.chain_id}) as your node is ({node_chain_id}).\nRaiden will shut down. Please choose a different Pathfinding Service.')
    if pathfinding_service_info.token_network_registry_address != token_network_registry_address:
        raise RaidenError(f'Invalid reply from Pathfinding Service {pfs_url}Pathfinding Service is not operating on the same Token Network Registry ({to_checksum_address(pathfinding_service_info.token_network_registry_address)}) as your node ({to_checksum_address(token_network_registry_address)}).\nRaiden will shut down. Please choose a different Pathfinding Service.')
    click.secho(
        f'You have chosen the Pathfinding Service at {pfs_url}.\nOperator: {pathfinding_service_info.operator}, running version: {pathfinding_service_info.version}, chain_id: {pathfinding_service_info.chain_id}.\nFees will be paid to {to_checksum_address(pathfinding_service_info.payment_address)}. Each request costs {to_rdn(pathfinding_service_info.price)} RDN.\nMessage from the Pathfinding Service:\n{pathfinding_service_info.message}'
    )
    log.info('Using Pathfinding Service', pfs_info=pathfinding_service_info)
    return pathfinding_service_info


def check_pfs_for_production(service_registry: Optional[ServiceRegistry], pfs_info: PFSInfo) -> None:
    if service_registry is None:
        raise RaidenError('Cannot verify registration of Pathfinding Service because no Service Registry is set. Raiden will shut down. Please select a Service Registry.')
    pfs_registered: bool = service_registry.has_valid_registration(
        block_identifier=service_registry.client.get_confirmed_blockhash(), service_address=pfs_info.payment_address
    )
    registered_pfs_url: Optional[str] = service_registry.get_service_url(
        block_identifier=service_registry.client.get_confirmed_blockhash(), service_address=pfs_info.payment_address
    )
    pfs_url_matches: bool = registered_pfs_url == pfs_info.url
    if not (pfs_registered and pfs_url_matches):
        raise RaidenError(
            f"The Pathfinding Service at {pfs_info.url} is not registered with the Service Registry at {to_checksum_address(service_registry.address)} or the registered URL ({registered_pfs_url}) doesn't match the given URL {pfs_info.url}. Raiden will shut down. Please select a registered Pathfinding Service."
        )


def get_last_iou(
    url: str,
    token_network_address: TokenNetworkAddress,
    sender: Address,
    receiver: Address,
    privkey: PrivateKey,
) -> Optional[IOU]:
    timestamp: str = datetime.utcnow().isoformat(timespec='seconds')
    signature_data: bytes = sender + receiver + Web3.toBytes(text=timestamp)
    signature: str = to_hex(LocalSigner(privkey).sign(signature_data))
    try:
        response = session.get(
            f'{url}/api/v1/{to_checksum_address(token_network_address)}/payment/iou',
            params=dict(sender=to_checksum_address(sender), receiver=to_checksum_address(receiver), timestamp=timestamp, signature=signature),
        )
        data: Any = json.loads(response.content).get('last_iou')
        if data is None:
            return None
        return IOU(
            sender=to_canonical_address(data['sender']),
            receiver=to_canonical_address(data['receiver']),
            one_to_n_address=OneToNAddress(to_canonical_address(data['one_to_n_address'])),
            amount=data['amount'],
            expiration_block=data['expiration_block'],
            chain_id=data['chain_id'],
            signature=Signature(decode_hex(data['signature'])),
        )
    except (RequestException, ValueError, KeyError) as e:
        raise ServiceRequestFailed(str(e))


def make_iou(
    pfs_config: PFSConfig,
    our_address: Address,
    one_to_n_address: OneToNAddress,
    privkey: PrivateKey,
    block_number: BlockNumber,
    chain_id: ChainID,
    offered_fee: TokenAmount,
) -> IOU:
    expiration: BlockExpiration = block_number + pfs_config.iou_timeout  # type: ignore
    iou: IOU = IOU(
        sender=our_address,
        receiver=pfs_config.info.payment_address,
        one_to_n_address=one_to_n_address,
        amount=offered_fee,
        expiration_block=expiration,
        chain_id=chain_id,
    )
    iou.sign(privkey)
    return iou


def update_iou(iou: IOU, privkey: PrivateKey, added_amount: TokenAmount = ZERO_TOKENS, expiration_block: Optional[BlockExpiration] = None) -> IOU:
    expected_signature: Signature = Signature(
        sign_one_to_n_iou(
            privatekey=privkey,
            sender=to_checksum_address(iou.sender),
            receiver=to_checksum_address(iou.receiver),
            amount=iou.amount,
            expiration_block=iou.expiration_block,
            one_to_n_address=to_checksum_address(iou.one_to_n_address),
            chain_id=iou.chain_id,
        )
    )
    if iou.signature != expected_signature:
        raise ServiceRequestFailed('Last IOU as given by the Pathfinding Service is invalid (signature does not match)')
    iou.amount = iou.amount + added_amount  # type: ignore
    if expiration_block:
        iou.expiration_block = expiration_block  # type: ignore
    iou.sign(privkey)
    return iou


def create_current_iou(
    pfs_config: PFSConfig,
    token_network_address: TokenNetworkAddress,
    one_to_n_address: OneToNAddress,
    our_address: Address,
    privkey: PrivateKey,
    block_number: BlockNumber,
    chain_id: ChainID,
    offered_fee: TokenAmount,
    scrap_existing_iou: bool = False,
) -> IOU:
    latest_iou: Optional[IOU] = None
    if not scrap_existing_iou:
        latest_iou = get_last_iou(
            url=pfs_config.info.url,
            token_network_address=token_network_address,
            sender=our_address,
            receiver=pfs_config.info.payment_address,
            privkey=privkey,
        )
    if latest_iou is None:
        return make_iou(
            pfs_config=pfs_config,
            our_address=our_address,
            one_to_n_address=one_to_n_address,
            privkey=privkey,
            block_number=block_number,
            chain_id=chain_id,
            offered_fee=offered_fee,
        )
    else:
        added_amount: TokenAmount = offered_fee
        return update_iou(iou=latest_iou, privkey=privkey, added_amount=added_amount)


def post_pfs_paths(url: str, token_network_address: TokenNetworkAddress, payload: Dict[str, Any]) -> Tuple[Any, UUID]:
    try:
        response = session.post(
            f'{url}/api/v1/{to_checksum_address(token_network_address)}/paths',
            json=payload,
            timeout=PFS_PATHS_REQUEST_TIMEOUT,
        )
    except RequestException as e:
        raise ServiceRequestFailed(f'Could not connect to Pathfinding Service ({str(e)})', dict(parameters=payload, exc_info=True))
    if response.status_code != 200:
        try:
            response_json: Any = get_response_json(response)
        except ValueError:
            log.error('Pathfinding Service returned error code (malformed json in response)', response=response)
            raise ServiceRequestFailed('Pathfinding Service returned error code (malformed json in response)', {'http_error': response.status_code})
        raise PFSReturnedError.from_response(response_json)
    try:
        response_json = get_response_json(response)
        return (response_json['result'], UUID(response_json['feedback_token']))
    except KeyError:
        raise ServiceRequestFailed("Answer from Pathfinding Service not understood ('result' field missing)", dict(response=get_response_json(response)))
    except ValueError:
        raise ServiceRequestFailed('Pathfinding Service returned invalid json', dict(response_text=response.text, exc_info=True))


def _query_address_metadata(pfs_config: PFSConfig, user_address: Address) -> Dict[str, Any]:
    try:
        response = session.get(f'{pfs_config.info.url}/api/v1/address/{to_checksum_address(user_address)}/metadata')
    except RequestException as e:
        raise ServiceRequestFailed(f'Could not connect to Pathfinding Service ({str(e)})', dict(exc_info=True))
    try:
        response_json: Any = get_response_json(response)
    except (ValueError, JSONDecodeError):
        raise ServiceRequestFailed('Pathfinding Service returned malformed json in response', {'http_code': response.status_code})
    if response.status_code != 200:
        raise PFSReturnedError.from_response(response_json)
    route_metadata = RouteMetadata([Address(user_address)], {Address(user_address): response_json})
    if route_metadata.address_metadata is not None and Address(user_address) not in route_metadata.address_metadata:
        raise ServiceRequestFailed(f'Pathfinding Service returned invalid metadata for {to_checksum_address(user_address)}', {'http_code': response.status_code})
    return response_json


def _query_paths(
    pfs_config: PFSConfig,
    our_address: Address,
    privkey: PrivateKey,
    current_block_number: BlockNumber,
    token_network_address: TokenNetworkAddress,
    one_to_n_address: OneToNAddress,
    chain_id: ChainID,
    route_from: Address,
    route_to: Address,
    value: TokenAmount,
    pfs_wait_for_block: BlockNumber,
) -> Tuple[List[Any], Optional[UUID]]:
    payload: Dict[str, Any] = {
        'from': to_checksum_address(route_from),
        'to': to_checksum_address(route_to),
        'value': value,
        'max_paths': pfs_config.max_paths,
    }
    offered_fee: TokenAmount = pfs_config.info.price
    scrap_existing_iou: bool = False
    current_info: PFSInfo = get_pfs_info(pfs_config.info.url)
    while current_info.confirmed_block_number < pfs_wait_for_block:
        log.info('Waiting for PFS to reach target confirmed block number', pfs_wait_for_block=pfs_wait_for_block, pfs_confirmed_block_number=current_info.confirmed_block_number)
        gevent.sleep(0.5)
        current_info = get_pfs_info(pfs_config.info.url)
    for retries in reversed(range(MAX_PATHS_QUERY_ATTEMPTS)):
        with iou_semaphore:
            if offered_fee > 0:
                new_iou: IOU = create_current_iou(
                    pfs_config=pfs_config,
                    token_network_address=token_network_address,
                    one_to_n_address=one_to_n_address,
                    our_address=our_address,
                    privkey=privkey,
                    chain_id=chain_id,
                    block_number=current_block_number,
                    offered_fee=offered_fee,
                    scrap_existing_iou=scrap_existing_iou,
                )
                payload['iou'] = new_iou.as_json()
            log.info('Requesting paths from Pathfinding Service', url=pfs_config.info.url, token_network_address=to_checksum_address(token_network_address), payload=payload)
            try:
                return post_pfs_paths(url=pfs_config.info.url, token_network_address=token_network_address, payload=payload)
            except ServiceRequestIOURejected as error:
                code = error.error_code
                log.debug('Pathfinding Service rejected IOU', error=error, details=error.error_details)
                if retries == 0 or code in (PFSError.WRONG_IOU_RECIPIENT, PFSError.DEPOSIT_TOO_LOW):
                    raise
                elif code in (PFSError.IOU_ALREADY_CLAIMED, PFSError.IOU_EXPIRED_TOO_EARLY):
                    scrap_existing_iou = True
                elif code == PFSError.INSUFFICIENT_SERVICE_PAYMENT:
                    try:
                        new_info: PFSInfo = get_pfs_info(pfs_config.info.url)
                    except ServiceRequestFailed:
                        raise ServiceRequestFailed('Could not get updated fee information from Pathfinding Service.')
                    if new_info.price > pfs_config.maximum_fee:
                        raise ServiceRequestFailed('PFS fees too high.')
                    if new_info.price > pfs_config.info.price:
                        log.info('Pathfinding Service increased fees', new_price=new_info.price)
                        pfs_config.info = new_info
                    else:
                        log.error('Concurrent Pathfinding Service requests by same node', value=str(iou_semaphore))
                elif code == PFSError.NO_ROUTE_FOUND:
                    log.info(f'Pathfinding Service can not find a route: {error}.')
                    return ([], None)
                log.info(f'Pathfinding Service rejected our payment. Reason: {error}. Attempting again.')
    return ([], None)


def post_pfs_feedback(
    routing_mode: RoutingMode,
    pfs_config: Optional[PFSConfig],
    token_network_address: TokenNetworkAddress,
    route: List[Address],
    token: bytes,
    successful: bool,
) -> None:
    feedback_disabled: bool = routing_mode == RoutingMode.PRIVATE or pfs_config is None
    if feedback_disabled:
        return
    hex_route: List[str] = [to_checksum_address(address) for address in route]
    payload: Dict[str, Any] = dict(token=token.hex(), path=hex_route, success=successful)
    log.info('Sending routing feedback to Pathfinding Service', url=pfs_config.info.url, token_network_address=to_checksum_address(token_network_address), payload=payload)
    try:
        session.post(f'{pfs_config.info.url}/api/v1/{to_checksum_address(token_network_address)}/feedback', json=payload)
    except RequestException as e:
        log.warning('Could not send feedback to Pathfinding Service', exception_=str(e), payload=payload)


class PFSProxy:
    def __init__(self, pfs_config: PFSConfig) -> None:
        self._pfs_config: PFSConfig = pfs_config

    def query_address_metadata(self, address: Address) -> Dict[str, Any]:
        return _query_address_metadata(self._pfs_config, address)

    def query_paths(
        self,
        our_address: Address,
        privkey: PrivateKey,
        current_block_number: BlockNumber,
        token_network_address: TokenNetworkAddress,
        one_to_n_address: OneToNAddress,
        chain_id: ChainID,
        route_from: Address,
        route_to: Address,
        value: TokenAmount,
        pfs_wait_for_block: BlockNumber,
    ) -> Tuple[List[Any], Optional[UUID]]:
        return _query_paths(
            self._pfs_config,
            our_address=our_address,
            privkey=privkey,
            current_block_number=current_block_number,
            token_network_address=token_network_address,
            one_to_n_address=one_to_n_address,
            chain_id=chain_id,
            route_from=route_from,
            route_to=route_to,
            value=value,
            pfs_wait_for_block=pfs_wait_for_block,
        )