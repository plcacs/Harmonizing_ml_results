from typing import Dict, Any, List, Union
import time
from dataclasses import replace
from unittest.mock import Mock, call, patch
from uuid import UUID, uuid4
import gevent
import pytest
from eth_utils import encode_hex, is_checksum_address, is_hex, is_hex_address
from raiden.api.v1.encoding import CapabilitiesSchema
from raiden.constants import RoutingMode
from raiden.exceptions import ServiceRequestFailed, ServiceRequestIOURejected
from raiden.network import pathfinding
from raiden.network.pathfinding import IOU, MAX_PATHS_QUERY_ATTEMPTS, PFSConfig, PFSError, PFSInfo, PFSProxy, get_last_iou, make_iou, post_pfs_feedback, session, update_iou
from raiden.settings import CapabilitiesConfig
from raiden.tests.utils import factories
from raiden.transfer.state import ChannelState, NettingChannelState, NetworkState, TokenNetworkState
from raiden.utils import typing
from raiden.utils.capabilities import capconfig_to_dict
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.signer import Signer
from raiden.utils.typing import Address, AddressMetadata, Any, BlockNumber, BlockTimeout, ChainID, Dict, PaymentAmount, TokenAmount, TokenNetworkAddress

def assert_checksum_address_in_url(url: str) -> None:
    message: str = 'URL does not contain properly encoded address.'
    assert any((is_checksum_address(token) for token in url.split('/'))), message

def make_address_metadata(signer: Signer) -> AddressMetadata:
    user_id: str = make_user_id(signer.address, 'homeserver')
    caps: Dict[str, Any] = capconfig_to_dict(CapabilitiesConfig())
    caps_schema: Dict[str, Any] = CapabilitiesSchema().dump({'capabilities': caps})['capabilities']
    signature_bytes: bytes = signer.sign(str(user_id).encode())
    signature_hex: str = encode_hex(signature_bytes)
    return dict(user_id=user_id, capabilities=caps_schema, displayname=signature_hex)

def create_square_network_topology(token_network_state: TokenNetworkState, our_address: Address) -> Tuple[TokenNetworkState, List[Address], List[ChannelState]]:
    address1: Address = factories.make_address()
    address2: Address = factories.make_address()
    address3: Address = factories.make_address()
    address4: Address = factories.make_address()
    routes: List[RouteProperties] = [RouteProperties(address1=our_address, address2=address1, capacity1to2=TokenAmount(50)), 
                                    RouteProperties(address1=our_address, address2=address2, capacity1to2=TokenAmount(100)), 
                                    RouteProperties(address1=address4, address2=address1, capacity1to2=TokenAmount(50)), 
                                    RouteProperties(address1=address2, address2=address3, capacity1to2=TokenAmount(100)), 
                                    RouteProperties(address1=address3, address2=address4, capacity1to2=TokenAmount(100), capacity2to1=TokenAmount(100))]
    new_state, channels: List[ChannelState] = factories.create_network(token_network_state=token_network_state, our_address=our_address, routes=routes, block_number=factories.make_block_number())
    return new_state, [address1, address2, address3, address4], channels

@pytest.fixture
def happy_path_fixture(chain_state: ChainState, token_network_state: TokenNetworkState, our_address: Address) -> Tuple[TokenNetworkState, List[Address], List[ChannelState], Any, TokenNetworkState]:
    token_network_state, addresses, channel_states = create_square_network_topology(token_network_state=token_network_state, our_address=our_address)
    address1, address2, address3, address4 = addresses
    chain_state.nodeaddresses_to_networkstates = {address1: NetworkState.REACHABLE, address2: NetworkState.REACHABLE, address3: NetworkState.REACHABLE, address4: NetworkState.REACHABLE}
    json_data: Dict[str, Any] = {'result': [{'path': [to_checksum_address(our_address), to_checksum_address(address2), to_checksum_address(address3), to_checksum_address(address4)], 'estimated_fee': 0}], 'feedback_token': DEFAULT_FEEDBACK_TOKEN.hex}
    response: Any = mocked_json_response(response_data=json_data)
    return (addresses, chain_state, channel_states, response, token_network_state)

def test_routing_mocked_pfs_happy_path