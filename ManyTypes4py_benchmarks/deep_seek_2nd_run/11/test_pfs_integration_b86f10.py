import time
from copy import copy
from dataclasses import replace
from unittest.mock import Mock, call, patch
from uuid import UUID, uuid4
import gevent
import pytest
import requests
from eth_utils import encode_hex, is_checksum_address, is_hex, is_hex_address
from raiden.api.v1.encoding import CapabilitiesSchema
from raiden.constants import RoutingMode
from raiden.exceptions import ServiceRequestFailed, ServiceRequestIOURejected
from raiden.network import pathfinding
from raiden.network.pathfinding import IOU, MAX_PATHS_QUERY_ATTEMPTS, PFSConfig, PFSError, PFSInfo, PFSProxy, get_last_iou, make_iou, post_pfs_feedback, session, update_iou
from raiden.network.transport.matrix.utils import make_user_id
from raiden.routing import get_best_routes, make_route_state
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils import factories
from raiden.tests.utils.mocks import mocked_failed_response, mocked_json_response
from raiden.transfer.state import ChannelState, NettingChannelState, NetworkState, TokenNetworkState
from raiden.utils import typing
from raiden.utils.capabilities import capconfig_to_dict
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.signer import Signer
from raiden.utils.typing import Address, AddressMetadata, Any, BlockNumber, BlockTimeout, ChainID, Dict, PaymentAmount, TokenAmount, TokenNetworkAddress, List, Optional, Tuple, Union

DEFAULT_FEEDBACK_TOKEN: UUID = UUID('381e4a005a4d4687ac200fa1acd15c6f')

def assert_checksum_address_in_url(url: str) -> None:
    message: str = 'URL does not contain properly encoded address.'
    assert any((is_checksum_address(token) for token in url.split('/'))), message

def make_address_metadata(signer: Signer) -> Dict[str, Any]:
    user_id: str = make_user_id(signer.address, 'homeserver')
    cap_dict: Dict[str, Any] = capconfig_to_dict(CapabilitiesConfig())
    caps: Dict[str, Any] = CapabilitiesSchema().dump({'capabilities': cap_dict})['capabilities']
    signature_bytes: bytes = signer.sign(str(user_id).encode())
    signature_hex: str = encode_hex(signature_bytes)
    return dict(user_id=user_id, capabilities=caps, displayname=signature_hex)

def create_square_network_topology(token_network_state: TokenNetworkState, our_address: Address) -> Tuple[TokenNetworkState, List[Address], List[NettingChannelState]]:
    address1: Address = factories.make_address()
    address2: Address = factories.make_address()
    address3: Address = factories.make_address()
    address4: Address = factories.make_address()
    routes: List[factories.RouteProperties] = [
        factories.RouteProperties(address1=our_address, address2=address1, capacity1to2=TokenAmount(50)),
        factories.RouteProperties(address1=our_address, address2=address2, capacity1to2=TokenAmount(100)),
        factories.RouteProperties(address1=address4, address2=address1, capacity1to2=TokenAmount(50)),
        factories.RouteProperties(address1=address2, address2=address3, capacity1to2=TokenAmount(100)),
        factories.RouteProperties(address1=address3, address2=address4, capacity1to2=TokenAmount(100), capacity2to1=TokenAmount(100))
    ]
    new_state: TokenNetworkState
    channels: List[NettingChannelState]
    new_state, channels = factories.create_network(token_network_state=token_network_state, our_address=our_address, routes=routes, block_number=factories.make_block_number())
    return (new_state, [address1, address2, address3, address4], channels)

PFS_CONFIG: PFSConfig = PFSConfig(
    info=PFSInfo(
        url='abc',
        price=TokenAmount(12),
        chain_id=ChainID(42),
        token_network_registry_address=factories.make_token_network_registry_address(),
        user_deposit_address=factories.make_address(),
        payment_address=factories.make_address(),
        message='',
        operator='',
        version='',
        confirmed_block_number=BlockNumber(10),
        matrix_server='http://matrix.example.com'
    ),
    maximum_fee=TokenAmount(100),
    iou_timeout=BlockTimeout(100),
    max_paths=5
)
CONFIG: Dict[str, PFSConfig] = {'pfs_config': PFS_CONFIG}
PRIVKEY: bytes = b'privkeyprivkeyprivkeyprivkeypriv'

def get_best_routes_with_iou_request_mocked(
    chain_state: typing.ChainState,
    token_network_state: TokenNetworkState,
    one_to_n_address: Address,
    from_address: Address,
    to_address: Address,
    amount: PaymentAmount,
    our_address_metadata: AddressMetadata,
    iou_json_data: Optional[Dict[str, Any]] = None
) -> Tuple[List[typing.RouteState], Optional[UUID]]:

    def iou_side_effect(*args: Any, **kwargs: Any) -> Mock:
        if args[0].endswith('/info'):
            return mocked_json_response({
                'price_info': 5,
                'network_info': {
                    'chain_id': 42,
                    'token_network_registry_address': to_checksum_address(factories.make_token_network_registry_address()),
                    'user_deposit_address': to_checksum_address(factories.make_address()),
                    'confirmed_block': {'number': 11}
                },
                'version': '0.0.3',
                'operator': 'John Doe',
                'message': 'This is your favorite pathfinding service',
                'payment_address': to_checksum_address(factories.make_address()),
                'matrix_server': 'http://matrix.example.com'
            })
        else:
            assert 'params' in kwargs
            body: Dict[str, Any] = kwargs['params']
            assert is_hex_address(body['sender'])
            assert is_hex_address(body['receiver'])
            assert 'timestamp' in body
            assert is_hex(body['signature'])
            assert len(body['signature']) == 65 * 2 + 2
            return mocked_json_response(response_data=iou_json_data)
    
    with patch.object(session, 'get', side_effect=iou_side_effect) as patched:
        _, best_routes, feedback_token = get_best_routes(
            chain_state=chain_state,
            token_network_address=token_network_state.address,
            one_to_n_address=one_to_n_address,
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            previous_address=None,
            pfs_proxy=PFSProxy(PFS_CONFIG),
            privkey=PRIVKEY,
            our_address_metadata=our_address_metadata
        )
        assert_checksum_address_in_url(patched.call_args[0][0])
        return (best_routes, feedback_token)

@pytest.fixture
def happy_path_fixture(chain_state: typing.ChainState, token_network_state: TokenNetworkState, our_address: Address) -> Tuple[List[Address], typing.ChainState, List[NettingChannelState], Mock, TokenNetworkState]:
    token_network_state, addresses, channel_states = create_square_network_topology(token_network_state=token_network_state, our_address=our_address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {
        address1: NetworkState.REACHABLE,
        address2: NetworkState.REACHABLE,
        address3: NetworkState.REACHABLE,
        address4: NetworkState.REACHABLE
    }
    json_data: Dict[str, Any] = {
        'result': [{
            'path': [
                to_checksum_address(our_address),
                to_checksum_address(address2),
                to_checksum_address(address3),
                to_checksum_address(address4)
            ],
            'estimated_fee': 0
        }],
        'feedback_token': DEFAULT_FEEDBACK_TOKEN.hex
    }
    response: Mock = mocked_json_response(response_data=json_data)
    return (addresses, chain_state, channel_states, response, token_network_state)

def test_routing_mocked_pfs_happy_path(happy_path_fixture: Tuple[List[Address], typing.ChainState, List[NettingChannelState], Mock, TokenNetworkState], one_to_n_address: Address, our_signer: Signer) -> None:
    addresses, chain_state, _, response, token_network_state = happy_path_fixture
    _, address2, _, address4 = addresses
    with patch.object(session, 'post', return_value=response) as patched:
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer)
        )
    assert_checksum_address_in_url(patched.call_args[0][0])
    assert routes[0].hop_after(our_signer.address) == address2
    assert feedback_token == DEFAULT_FEEDBACK_TOKEN
    iou: Dict[str, Any] = patched.call_args[1]['json']['iou']
    pfs_config: PFSConfig = CONFIG['pfs_config']
    for key in ('amount', 'expiration_block', 'signature', 'sender', 'receiver'):
        assert key in iou
    assert iou['amount'] <= pfs_config.maximum_fee
    latest_expected_expiration: BlockNumber = pfs_config.iou_timeout + chain_state.block_number
    assert iou['expiration_block'] <= latest_expected_expiration

def test_routing_mocked_pfs_happy_path_with_updated_iou(happy_path_fixture: Tuple[List[Address], typing.ChainState, List[NettingChannelState], Mock, TokenNetworkState], one_to_n_address: Address, our_signer: Signer) -> None:
    addresses, chain_state, _, response, token_network_state = happy_path_fixture
    _, address2, _, address4 = addresses
    iou: IOU = make_iou(
        pfs_config=PFS_CONFIG,
        our_address=factories.UNIT_TRANSFER_SENDER,
        one_to_n_address=one_to_n_address,
        privkey=PRIVKEY,
        block_number=BlockNumber(10),
        chain_id=ChainID(5),
        offered_fee=TokenAmount(1)
    )
    last_iou: IOU = copy(iou)
    with patch.object(session, 'post', return_value=response) as patched:
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer),
            iou_json_data=dict(last_iou=last_iou.as_json())
        )
    assert_checksum_address_in_url(patched.call_args[0][0])
    assert routes[0].hop_after(our_signer.address) == address2
    assert feedback_token == DEFAULT_FEEDBACK_TOKEN
    payload: Dict[str, Any] = patched.call_args[1]['json']
    pfs_config: PFSConfig = CONFIG['pfs_config']
    old_amount: TokenAmount = last_iou.amount
    assert old_amount < payload['iou']['amount'] <= pfs_config.maximum_fee + old_amount
    assert payload['iou']['expiration_block'] == last_iou.expiration_block
    assert payload['iou']['sender'] == to_checksum_address(last_iou.sender)
    assert payload['iou']['receiver'] == to_checksum_address(last_iou.receiver)
    assert 'signature' in payload['iou']

def test_routing_mocked_pfs_request_error(chain_state: typing.ChainState, token_network_state: TokenNetworkState, one_to_n_address: Address, our_signer: Signer) -> None:
    token_network_state, addresses, _ = create_square_network_topology(token_network_state=token_network_state, our_address=our_signer.address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {
        address1: NetworkState.REACHABLE,
        address2: NetworkState.REACHABLE,
        address3: NetworkState.REACHABLE
    }
    with patch.object(session, 'post', side_effect=requests.RequestException()):
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer)
        )
        assert len(routes) == 0
        assert feedback_token is None

def test_routing_mocked_pfs_bad_http_code(chain_state: typing.ChainState, token_network_state: TokenNetworkState, one_to_n_address: Address, our_signer: Signer) -> None:
    token_network_state, addresses, _ = create_square_network_topology(token_network_state=token_network_state, our_address=our_signer.address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {
        address1: NetworkState.REACHABLE,
        address2: NetworkState.REACHABLE,
        address3: NetworkState.REACHABLE
    }
    json_data: Dict[str, Any] = {
        'result': [{
            'path': [
                to_checksum_address(our_signer.address),
                to_checksum_address(address2),
                to_checksum_address(address3),
                to_checksum_address(address4)
            ],
            'fees': 0
        }]
    }
    response: Mock = mocked_json_response(response_data=json_data, status_code=400)
    with patch.object(session, 'post', return_value=response):
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer)
        )
        assert len(routes) == 0
        assert feedback_token is None

def test_routing_mocked_pfs_invalid_json(chain_state: typing.ChainState, token_network_state: TokenNetworkState, one_to_n_address: Address, our_signer: Signer) -> None:
    token_network_state, addresses, _ = create_square_network_topology(token_network_state=token_network_state, our_address=our_signer.address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {
        address1: NetworkState.REACHABLE,
        address2: NetworkState.REACHABLE,
        address3: NetworkState.REACHABLE
    }
    response: Mock = mocked_failed_response(error=ValueError(), status_code=200)
    with patch.object(requests, 'post', return_value=response):
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer)
        )
        assert len(routes) == 0
        assert feedback_token is None

def test_routing_mocked_pfs_invalid_json_structure(chain_state: typing.ChainState, one_to_n_address: Address, token_network_state: TokenNetworkState, our_signer: Signer) -> None:
    token_network_state, addresses, _ = create_square_network_topology(token_network_state=token_network_state, our_address=our_signer.address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {
        address1: NetworkState.REACHABLE,
        address2: NetworkState.REACHABLE,
        address3: NetworkState.REACHABLE
    }
    response: Mock = mocked_json_response(response_data={}, status_code=400)
    with patch.object(session, 'post', return_value=response):
        routes, feedback_token = get_best_routes_with_iou_request_mocked(
            chain_state=chain_state,
            token_network_state=token_network_state,
            one_to_n_address=one_to_n_address,
            from_address=our_signer.address,
            to_address=address4,
            amount=50,
            our_address_metadata=make_address_metadata(our_signer)
        )
        assert len(routes) == 0
        assert feedback_token is None

def test_routing_mocked_pfs_unavailable_peer(chain_state: typing.ChainState, token_network_state: TokenNetworkState, one_to_n_address: Address, our_signer: Signer) -> None:
    our_address: Address = our_signer.address
    token_network_state, addresses, _ = create_square_network_topology(token_network_state=token_network_state, our_address=our_address)
    address1, address2, address3, address4 = addresses
    json_data: Dict[str, Any] = {
        'result': [{
            'path': [
                to_checksum_address(our_address),
                to_checksum_address(address2),
                to_checksum_address(address3),
                to_checksum_address