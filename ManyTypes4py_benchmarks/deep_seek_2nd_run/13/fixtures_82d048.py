import random
import typing
import pytest
import responses
from raiden.tests.utils import factories
from raiden.tests.utils.factories import UNIT_CHAIN_ID
from raiden.transfer.state import ChainState, TokenNetworkRegistryState, TokenNetworkState
from raiden.utils.typing import BlockNumber, TokenAmount
from typing import Generator, Optional, Dict, List, Any

@pytest.fixture
def our_signer() -> factories.Signer:
    return factories.make_signer()

@pytest.fixture
def our_address(our_signer: factories.Signer) -> str:
    return our_signer.address

@pytest.fixture
def token_id() -> str:
    return factories.make_address()

@pytest.fixture
def token_network_address() -> str:
    return factories.make_address()

@pytest.fixture
def one_to_n_address() -> str:
    return factories.make_address()

@pytest.fixture
def token_network_registry_address() -> str:
    return factories.make_address()

@pytest.fixture
def chain_state(our_address: str) -> ChainState:
    block_number = BlockNumber(1)
    return ChainState(
        pseudo_random_generator=random.Random(),
        block_number=block_number,
        block_hash=factories.make_block_hash(),
        our_address=our_address,
        chain_id=UNIT_CHAIN_ID,
    )

@pytest.fixture
def token_network_registry_state(
    chain_state: ChainState,
    token_network_registry_address: str,
) -> TokenNetworkRegistryState:
    token_network_registry = TokenNetworkRegistryState(token_network_registry_address, [])
    chain_state.identifiers_to_tokennetworkregistries[token_network_registry_address] = token_network_registry
    return token_network_registry

@pytest.fixture
def token_network_state(
    chain_state: ChainState,
    token_network_registry_state: TokenNetworkRegistryState,
    token_network_registry_address: str,
    token_network_address: str,
    token_id: str,
) -> TokenNetworkState:
    token_network = TokenNetworkState(address=token_network_address, token_address=token_id)
    token_network_registry_state.add_token_network(token_network)
    mapping = chain_state.tokennetworkaddresses_to_tokennetworkregistryaddresses
    mapping[token_network_address] = token_network_registry_address
    return token_network

@pytest.fixture
def partner() -> Optional[str]:
    return None

@pytest.fixture
def netting_channel_state(
    chain_state: ChainState,
    token_network_state: TokenNetworkState,
    token_network_registry_state: TokenNetworkRegistryState,
    partner: Optional[str],
) -> Any:  # Replace Any with actual return type if known
    if partner is None:
        partner = factories.make_address()
    canonical_identifier = factories.make_canonical_identifier(token_network_address=token_network_state.address)
    channel_state = factories.create(
        factories.NettingChannelStateProperties(
            our_state=factories.NettingChannelEndStateProperties(
                balance=TokenAmount(10),
                address=chain_state.our_address,
            ),
            partner_state=factories.NettingChannelEndStateProperties(
                balance=TokenAmount(10),
                address=partner,
            ),
            token_address=token_network_state.token_address,
            token_network_registry_address=token_network_registry_state.address,
            canonical_identifier=canonical_identifier,
        )
    )
    channel_id = canonical_identifier.channel_identifier
    token_network_state.partneraddresses_to_channelidentifiers[partner].append(channel_id)
    token_network_state.channelidentifiers_to_channels[channel_id] = channel_state
    return channel_state

@pytest.fixture
def requests_responses() -> Generator[responses.RequestsMock, None, None]:
    """Uses ``responses`` to provide easy requests tests."""
    with responses.RequestsMock() as mock_responses:
        yield mock_responses
