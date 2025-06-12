import random
import gevent
import pytest
from eth_utils import decode_hex, encode_hex, to_canonical_address
from gevent.greenlet import Greenlet
from gevent.queue import Queue
from raiden.constants import BLOCK_ID_LATEST, EMPTY_BALANCE_HASH, EMPTY_HASH, EMPTY_SIGNATURE, GENESIS_BLOCK_NUMBER, LOCKSROOT_OF_NO_LOCKS, STATE_PRUNING_AFTER_BLOCKS
from raiden.exceptions import BrokenPreconditionError, InvalidChannelID, InvalidSettleTimeout, RaidenRecoverableError, RaidenUnrecoverableError, SamePeerAddress
from raiden.network.proxies.proxy_manager import ProxyManager, ProxyManagerMetadata
from raiden.network.proxies.token_network import TokenNetwork
from raiden.network.rpc.client import JSONRPCClient
from raiden.tests.integration.network.proxies import BalanceProof
from raiden.tests.utils import factories
from raiden.tests.utils.factories import make_address
from raiden.tests.utils.smartcontracts import is_tx_hash_bytes
from raiden.utils.formatting import to_hex_address
from raiden.utils.signer import LocalSigner
from raiden.utils.typing import Set, T_ChannelID, BlockHash, BlockNumber, Address, TokenAmount, Signature, PrivateKey, ChainID
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN, MessageTypeId
from typing import Dict, Tuple, Optional, Any, List
SIGNATURE_SIZE_IN_BITS: int = 520

def test_token_network_deposit_race(
    token_network_proxy: TokenNetwork,
    private_keys: List[PrivateKey],
    token_proxy: Any,
    web3: Any,
    contract_manager: Any
) -> None:
    assert token_network_proxy.settlement_timeout_min() == TEST_SETTLE_TIMEOUT_MIN
    assert token_network_proxy.settlement_timeout_max() == TEST_SETTLE_TIMEOUT_MAX
    token_network_address: Address = to_canonical_address(token_network_proxy.proxy.address)
    c1_client: JSONRPCClient = JSONRPCClient(web3, private_keys[1])
    c2_client: JSONRPCClient = JSONRPCClient(web3, private_keys[2])
    proxy_manager: ProxyManager = ProxyManager(rpc_client=c1_client, contract_manager=contract_manager, metadata=ProxyManagerMetadata(token_network_registry_deployed_at=GENESIS_BLOCK_NUMBER, filters_start_at=GENESIS_BLOCK_NUMBER))
    c1_token_network_proxy: TokenNetwork = proxy_manager.token_network(address=token_network_address, block_identifier=BLOCK_ID_LATEST)
    token_proxy.transfer(c1_client.address, 10)
    channel_details: Any = c1_token_network_proxy.new_netting_channel(partner=c2_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MIN, given_block_identifier=BLOCK_ID_LATEST)
    assert is_tx_hash_bytes(channel_details.transaction_hash)
    channel_identifier: T_ChannelID = channel_details.channel_identifier
    assert channel_identifier is not None
    c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=2, partner=c2_client.address)
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=1, partner=c2_client.address)

def test_token_network_proxy(
    token_network_proxy: TokenNetwork,
    private_keys: List[PrivateKey],
    token_proxy: Any,
    chain_id: ChainID,
    web3: Any,
    contract_manager: Any
) -> None:
    assert token_network_proxy.settlement_timeout_min() == TEST_SETTLE_TIMEOUT_MIN
    assert token_network_proxy.settlement_timeout_max() == TEST_SETTLE_TIMEOUT_MAX
    token_network_address: Address = to_canonical_address(token_network_proxy.proxy.address)
    c1_signer: LocalSigner = LocalSigner(private_keys[1])
    c1_client: JSONRPCClient = JSONRPCClient(web3, private_keys[1])
    c1_proxy_manager: ProxyManager = ProxyManager(rpc_client=c1_client, contract_manager=contract_manager, metadata=ProxyManagerMetadata(token_network_registry_deployed_at=GENESIS_BLOCK_NUMBER, filters_start_at=GENESIS_BLOCK_NUMBER))
    c2_client: JSONRPCClient = JSONRPCClient(web3, private_keys[2])
    c2_proxy_manager: ProxyManager = ProxyManager(rpc_client=c2_client, contract_manager=contract_manager, metadata=ProxyManagerMetadata(token_network_registry_deployed_at=GENESIS_BLOCK_NUMBER, filters_start_at=GENESIS_BLOCK_NUMBER))
    c2_signer: LocalSigner = LocalSigner(private_keys[2])
    c1_token_network_proxy: TokenNetwork = c1_proxy_manager.token_network(address=token_network_address, block_identifier=BLOCK_ID_LATEST)
    c2_token_network_proxy: TokenNetwork = c2_proxy_manager.token_network(address=token_network_address, block_identifier=BLOCK_ID_LATEST)
    initial_token_balance: TokenAmount = 100
    token_proxy.transfer(c1_client.address, initial_token_balance)
    token_proxy.transfer(c2_client.address, initial_token_balance)
    initial_balance_c1: TokenAmount = token_proxy.balance_of(c1_client.address)
    assert initial_balance_c1 == initial_token_balance
    initial_balance_c2: TokenAmount = token_proxy.balance_of(c2_client.address)
    assert initial_balance_c2 == initial_token_balance
    assert c1_token_network_proxy.get_channel_identifier_or_none(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST) is None
    msg: str = 'Hex encoded addresses are not supported, an assertion must be raised'
    with pytest.raises(AssertionError):
        c1_token_network_proxy.get_channel_identifier(participant1=to_hex_address(c1_client.address), participant2=to_hex_address(c2_client.address), block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    msg = 'Zero is not a valid channel_identifier identifier, an exception must be raised.'
    with pytest.raises(InvalidChannelID):
        assert c1_token_network_proxy.channel_is_opened(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST, channel_identifier=0)
        pytest.fail(msg)
    msg = 'Zero is not a valid channel_identifier identifier. an exception must be raised.'
    with pytest.raises(InvalidChannelID):
        assert c1_token_network_proxy.channel_is_closed(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST, channel_identifier=0)
        pytest.fail(msg)
    msg = "Opening a channel with a settle_timeout lower then token network's minimum will fail. This must be validated and the transaction must not be sent."
    with pytest.raises(InvalidSettleTimeout):
        c1_token_network_proxy.new_netting_channel(partner=c2_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MIN - 1, given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    channel_details: Any = c1_token_network_proxy.new_netting_channel(partner=make_address(), settle_timeout=TEST_SETTLE_TIMEOUT_MIN, given_block_identifier=BLOCK_ID_LATEST)
    assert is_tx_hash_bytes(channel_details.transaction_hash)
    msg = "Opening a channel with a settle_timeout larger then token network's maximum will fail. This must be validated and the transaction must not be sent."
    with pytest.raises(InvalidSettleTimeout):
        c1_token_network_proxy.new_netting_channel(partner=c2_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MAX + 1, given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    c1_token_network_proxy.new_netting_channel(partner=make_address(), settle_timeout=TEST_SETTLE_TIMEOUT_MAX, given_block_identifier=BLOCK_ID_LATEST)
    msg = 'Opening a channel with itself is not allow. This must be validated and the transaction must not be sent.'
    with pytest.raises(SamePeerAddress):
        c1_token_network_proxy.new_netting_channel(partner=c1_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MIN, given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    msg = 'Trying a deposit to an inexisting channel must fail.'
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=100, total_deposit=1, partner=c2_client.address)
        pytest.fail(msg)
    empty_balance_proof: BalanceProof = BalanceProof(channel_identifier=100, token_network_address=c1_token_network_proxy.address, balance_hash=EMPTY_BALANCE_HASH, nonce=0, chain_id=chain_id, transferred_amount=0)
    closing_data: bytes = empty_balance_proof.serialize_bin(msg_type=MessageTypeId.BALANCE_PROOF) + EMPTY_SIGNATURE
    msg = 'Trying to close an inexisting channel must fail.'
    match: str = 'The channel was not open at the provided block'
    with pytest.raises(RaidenUnrecoverableError, match=match):
        c1_token_network_proxy.close(channel_identifier=100, partner=c2_client.address, balance_hash=EMPTY_HASH, nonce=0, additional_hash=EMPTY_HASH, non_closing_signature=EMPTY_SIGNATURE, closing_signature=c1_signer.sign(data=closing_data), given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    channel_details: Any = c1_token_network_proxy.new_netting_channel(partner=c2_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MIN, given_block_identifier=BLOCK_ID_LATEST)
    channel_identifier: T_ChannelID = channel_details.channel_identifier
    msg = 'new_netting_channel did not return a valid channel id'
    assert isinstance(channel_identifier, T_ChannelID), msg
    msg = 'multiple channels with the same peer are not allowed'
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.new_netting_channel(partner=c2_client.address, settle_timeout=TEST_SETTLE_TIMEOUT_MIN, given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    assert c1_token_network_proxy.get_channel_identifier_or_none(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST) is not None
    assert c1_token_network_proxy.channel_is_opened(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier) is True
    msg = "approve_and_set_total_deposit must fail if the amount exceed the account's balance"
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=initial_token_balance + 1, partner=c2_client.address)
        pytest.fail(msg)
    msg = 'approve_and_set_total_deposit must fail with a negative amount'
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=-1, partner=c2_client.address)
        pytest.fail(msg)
    msg = 'approve_and_set_total_deposit must fail with a zero amount'
    with pytest.raises(BrokenPreconditionError):
        c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=0, partner=c2_client.address)
        pytest.fail(msg)
    c1_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier, total_deposit=10, partner=c2_client.address)
    transferred_amount: TokenAmount = 3
    balance_proof: BalanceProof = BalanceProof(channel_identifier=channel_identifier, token_network_address=token_network_address, nonce=1, chain_id=chain_id, transferred_amount=transferred_amount)
    signature: Signature = c1_signer.sign(data=balance_proof.serialize_bin())
    balance_proof.signature = encode_hex(signature)
    signature_number: int = int.from_bytes(signature, 'big')
    bit_to_change: int = random.randint(0, SIGNATURE_SIZE_IN_BITS - 1)
    signature_number_bit_flipped: int = signature_number ^ 2 ** bit_to_change
    invalid_signatures: List[bytes] = [EMPTY_SIGNATURE, b'\x11' * 65, signature_number_bit_flipped.to_bytes(len(signature), 'big')]
    msg = 'close must fail if the closing_signature is invalid'
    for invalid_signature in invalid_signatures:
        closing_data = balance_proof.serialize_bin(msg_type=MessageTypeId.BALANCE_PROOF) + invalid_signature
        with pytest.raises(RaidenUnrecoverableError):
            c2_token_network_proxy.close(channel_identifier=channel_identifier, partner=c1_client.address, balance_hash=balance_proof.balance_hash, nonce=balance_proof.nonce, additional_hash=decode_hex(balance_proof.additional_hash), non_closing_signature=invalid_signature, closing_signature=c2_signer.sign(data=closing_data), given_block_identifier=BLOCK_ID_LATEST)
            pytest.fail(msg)
    blocknumber_prior_to_close: BlockNumber = c2_client.block_number()
    closing_data = balance_proof.serialize_bin(msg_type=MessageTypeId.BALANCE_PROOF) + decode_hex(balance_proof.signature)
    transaction_hash: bytes = c2_token_network_proxy.close(channel_identifier=channel_identifier, partner=c1_client.address, balance_hash=balance_proof.balance_hash, nonce=balance_proof.nonce, additional_hash=decode_hex(balance_proof.additional_hash), non_closing_signature=decode_hex(balance_proof.signature), closing_signature=c2_signer.sign(data=closing_data), given_block_identifier=BLOCK_ID_LATEST)
    assert is_tx_hash_bytes(transaction_hash)
    assert c1_token_network_proxy.channel_is_closed(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST, channel_identifier=channel_identifier) is True
    assert c1_token_network_proxy.get_channel_identifier_or_none(participant1=c1_client.address, participant2=c2_client.address, block_identifier=BLOCK_ID_LATEST) is not None
    msg = 'given_block_identifier is the block at which the transaction is being  sent. If the channel is already closed at that block the client code  has a programming error. An exception is raised for that.'
    with pytest.raises(RaidenUnrecoverableError):
        c2_token_network_proxy.close(channel_identifier=channel_identifier, partner=c1_client.address, balance_hash=balance_proof.balance_hash, nonce=balance_proof.nonce, additional_hash=decode_hex(balance_proof.additional_hash), non_closing_signature=decode_hex(balance_proof.signature), closing_signature=c2_signer.sign(data=closing_data), given_block_identifier=BLOCK_ID_LATEST)
        pytest.fail(msg)
    msg = 'The channel cannot be closed two times. If it was not closed at given_block_identifier but it is closed at the time the proxy is called an exception must be raised.'
    with pytest.raises(RaidenRecoverableError):
        c2_token_network_proxy.close(channel_identifier=channel_identifier, partner=c1_client.address, balance_hash=balance_proof.balance_hash, nonce=balance_proof.nonce, additional_hash=decode_hex(balance_proof.additional_hash), non_closing_signature=decode_hex(balance_proof.signature), closing_signature=c2_signer.sign(data=closing_data), given_block_identifier=blocknumber_prior_to_close)
        pytest.fail(msg)
    msg = 'depositing to a closed channel must fail'
    match = 'closed'
    with pytest.raises(RaidenRecoverableError, match=match):
        c2_token_network_proxy.approve_and_set_total_deposit(given_block_identifier=blocknumber_prior_to_close, channel_identifier=channel_identifier, total_deposit=20, partner=c1_client.address)
        pytest.fail(msg)
    c1_client.wait_until_block(target_block_number=c1_client.block_number() + TEST_SETTLE_TIMEOUT_MIN)
    invalid_transferred_amount: TokenAmount = 1
    msg = 'settle with invalid transferred_amount data must fail'
    with pytest.raises(BrokenPreconditionError):
        c2_token_network_proxy.settle(channel_identifier=