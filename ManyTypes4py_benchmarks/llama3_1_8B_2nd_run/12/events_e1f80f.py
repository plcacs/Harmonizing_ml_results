import time
from dataclasses import dataclass
from typing import Callable, Tuple
import structlog
from eth_utils import to_canonical_address, to_checksum_address
from gevent.lock import Semaphore
from requests.exceptions import ReadTimeout
from web3 import Web3
from web3.types import LogReceipt, RPCEndpoint
from raiden.blockchain.exceptions import EthGetLogsTimeout, UnknownRaidenEventType
from raiden.blockchain.filters import RaidenContractFilter, decode_event, get_filter_args_for_all_events_from_channel
from raiden.blockchain.utils import BlockBatchSizeAdjuster
from raiden.constants import BLOCK_ID_LATEST, ETH_GET_LOGS_THRESHOLD_FAST, ETH_GET_LOGS_THRESHOLD_SLOW, GENESIS_BLOCK_NUMBER, UINT64_MAX
from raiden.exceptions import InvalidBlockNumberInput
from raiden.network.proxies.proxy_manager import ProxyManager
from raiden.settings import BlockBatchSizeConfig
from raiden.utils.typing import ABI, Address, Any, BlockGasLimit, BlockHash, BlockIdentifier, BlockNumber, ChainID, ChannelID, Dict, List, Optional, TokenNetworkAddress, TransactionHash
from raiden_contracts.constants import CONTRACT_TOKEN_NETWORK, EVENT_REGISTERED_SERVICE, EVENT_TOKEN_NETWORK_CREATED, ChannelEvent
from raiden_contracts.contract_manager import ContractManager

log: structlog.Logger = structlog.get_logger(__name__)

ALL_EVENTS: Optional[List[str]] = None

@dataclass(frozen=True)
class DecodedEvent:
    """A confirmed event with the data decoded to conform with Raiden's internals.

    Raiden prefers bytes for addresses and hashes, and it uses snake_case as a
    naming convention. Instances of this class are created at the edges of the
    code base to conform with the internal data types, i.e. this type describes
    is used at the IO boundaries to conform with the sandwich encoding.
    """
    chain_id: ChainID
    originating_contract: Address
    event_data: Dict[str, Any]
    block_number: BlockNumber
    block_hash: BlockHash
    transaction_hash: TransactionHash

@dataclass(frozen=True)
class PollResult:
    """Result of a poll request. The block number is provided so that the
    caller can confirm it in its storage.
    """

def verify_block_number(number: BlockNumber, argname: str) -> None:
    if isinstance(number, int) and (number < 0 or number > UINT64_MAX):
        raise InvalidBlockNumberInput('Provided block number {} for {} is invalid. Has to be in the range of [0, UINT64_MAX]'.format(number, argname))

def get_contract_events(proxy_manager: ProxyManager, abi: ABI, contract_address: Address, topics: Optional[List[str]] = ALL_EVENTS, from_block: BlockNumber = GENESIS_BLOCK_NUMBER, to_block: BlockIdentifier = BLOCK_ID_LATEST) -> List[Dict[str, Any]]:
    """Query the blockchain for all events of the smart contract at
    `contract_address` that match the filters `topics`, `from_block`, and
    `to_block`.
    """
    verify_block_number(from_block, 'from_block')
    verify_block_number(to_block, 'to_block')
    events = proxy_manager.client.get_filter_events(contract_address, topics=topics, from_block=from_block, to_block=to_block)
    result = []
    for event in events:
        decoded_event = dict(decode_event(abi, event))
        if event.get('blockNumber'):
            decoded_event['block_number'] = event['blockNumber']
            del decoded_event['blockNumber']
        result.append(decoded_event)
    return result

def get_all_netting_channel_events(proxy_manager: ProxyManager, token_network_address: TokenNetworkAddress, netting_channel_identifier: ChannelID, contract_manager: ContractManager, from_block: BlockNumber = GENESIS_BLOCK_NUMBER, to_block: BlockIdentifier = BLOCK_ID_LATEST) -> List[Dict[str, Any]]:
    """Helper to get all events of a NettingChannelContract."""
    filter_args = get_filter_args_for_all_events_from_channel(token_network_address=token_network_address, channel_identifier=netting_channel_identifier, contract_manager=contract_manager, from_block=from_block, to_block=to_block)
    return get_contract_events(proxy_manager, contract_manager.get_contract_abi(CONTRACT_TOKEN_NETWORK), Address(token_network_address), filter_args['topics'], from_block, to_block)

def decode_raiden_event_to_internal(abi: ABI, chain_id: ChainID, log_event: LogReceipt) -> DecodedEvent:
    """Enforce the sandwich encoding. Converts the JSON RPC/web3 data types
    to the internal representation.

    Note::

        This function must only on confirmed data.
    """
    decoded_event = decode_event(abi, log_event)
    if not decoded_event:
        raise UnknownRaidenEventType()
    data = dict(decoded_event)
    args = dict(decoded_event['args'])
    data['args'] = args
    data['block_number'] = log_event['blockNumber']
    data['transaction_hash'] = log_event['transactionHash']
    data['block_hash'] = bytes(log_event['blockHash'])
    del data['blockNumber']
    del data['transactionHash']
    del data['blockHash']
    assert data['block_number'], 'The event must have the block_number'
    assert data['transaction_hash'], 'The event must have the transaction hash field'
    assert data['block_hash'], 'The event must have the block_hash'
    event = data['event']
    if event == EVENT_TOKEN_NETWORK_CREATED:
        args['token_network_address'] = to_canonical_address(args['token_network_address'])
        args['token_address'] = to_canonical_address(args['token_address'])
    elif event == ChannelEvent.OPENED:
        args['participant1'] = to_canonical_address(args['participant1'])
        args['participant2'] = to_canonical_address(args['participant2'])
    elif event == ChannelEvent.DEPOSIT:
        args['participant'] = to_canonical_address(args['participant'])
    elif event == ChannelEvent.WITHDRAW:
        args['participant'] = to_canonical_address(args['participant'])
    elif event == ChannelEvent.BALANCE_PROOF_UPDATED:
        args['closing_participant'] = to_canonical_address(args['closing_participant'])
    elif event == ChannelEvent.CLOSED:
        args['closing_participant'] = to_canonical_address(args['closing_participant'])
    elif event == ChannelEvent.UNLOCKED:
        args['receiver'] = to_canonical_address(args['receiver'])
        args['sender'] = to_canonical_address(args['sender'])
    elif event == EVENT_REGISTERED_SERVICE:
        args['service_address'] = to_canonical_address(args.pop('service'))
        assert 'valid_till' in args, f"{EVENT_REGISTERED_SERVICE} without 'valid_till'"
    return DecodedEvent(chain_id=chain_id, originating_contract=to_canonical_address(log_event['address']), event_data=data, block_number=log_event['blockNumber'], block_hash=BlockHash(log_event['blockHash']), transaction_hash=TransactionHash(log_event['transactionHash']))

def new_filters_from_events(events: List[DecodedEvent]) -> RaidenContractFilter:
    new_filter = RaidenContractFilter(token_network_addresses={entry.event_data['args']['token_network_address'] for entry in events if entry.event_data['event'] == EVENT_TOKEN_NETWORK_CREATED}, ignore_secret_registry_until_channel_found=True)
    for entry in events:
        if entry.event_data['event'] == ChannelEvent.OPENED:
            new_filter.channels_of_token_network.setdefault(TokenNetworkAddress(entry.originating_contract), set()).add(entry.event_data['args']['channel_identifier'])
            new_filter.ignore_secret_registry_until_channel_found = False
    return new_filter

def sort_events(events: List[DecodedEvent]) -> None:
    events.sort(key=lambda e: e.block_number)

class BlockchainEvents:

    def __init__(self, web3: Web3, chain_id: ChainID, contract_manager: ContractManager, last_fetched_block: BlockNumber, event_filter: RaidenContractFilter, block_batch_size_config: BlockBatchSizeConfig, node_address: Address) -> None:
        self.web3: Web3 = web3
        self.chain_id: ChainID = chain_id
        self.last_fetched_block: BlockNumber = last_fetched_block
        self.contract_manager: ContractManager = contract_manager
        self.event_filter: RaidenContractFilter = event_filter
        self.block_batch_size_adjuster: BlockBatchSizeAdjuster = BlockBatchSizeAdjuster(block_batch_size_config)
        self.node_address: Address = node_address
        self._filters_lock: Semaphore = Semaphore()
        self._address_to_abi: Dict[Address, ABI] = event_filter.abi_of_contract_address(contract_manager)
        self._listeners: List[Callable[[List[DecodedEvent]], None]] = []

    def fetch_logs_in_batch(self, target_block_number: BlockNumber) -> Optional[PollResult]:
        """Poll the smart contract events for a limited number of blocks to
        avoid read timeouts (issue #3558).

        The block ``target_block_number`` will not be reached if it is more than
        ``self.block_batch_size_adjuster.batch_size`` blocks away. To ensure the
        target is reached keep calling ``fetch_logs_in_batch`` until
        ``PollResult.polled_block_number`` is the same as ``target_block_number``.

        This function will make sure that the block range for the queries is
        not too big, this is necessary because it may take a long time for an
        Ethereum node to process the request, which will result in read
        timeouts (issue #3558).

        The block batch size is adjusted dynamically based on the request
        processing duration (see ``_query_and_track()``, issue #5538).
        If the request times out the batch size is decreased and ``None``
        is returned.
        If the batch size falls below the lower threshold an exception is raised
        by the ``BlockBatchSizeAdjuster``.

        This will also group the queries as an optimization for a healthy node
        (issue #4872). This is enforced by the design of the datastructures,
        this will always fetch all the events for all the registered addresses.
        """
        if target_block_number <= self.last_fetched_block:
            raise ValueError(f'target {target_block_number} is in the past, the block has been fetched already. Current {self.last_fetched_block}')
        with self._filters_lock:
            from_block = BlockNumber(self.last_fetched_block + 1)
            to_block = BlockNumber(min(from_block + self.block_batch_size_adjuster.batch_size, target_block_number))
            try:
                decoded_result, max_request_duration = self._query_and_track(from_block, to_block)
            except EthGetLogsTimeout:
                log.debug('Timeout while fetching blocks, decreasing batch size')
                self.block_batch_size_adjuster.decrease()
                return None
            can_use_bigger_batches = target_block_number - from_block > self.block_batch_size_adjuster.batch_size
            if max_request_duration < ETH_GET_LOGS_THRESHOLD_FAST:
                if can_use_bigger_batches:
                    self.block_batch_size_adjuster.increase()
            elif max_request_duration > ETH_GET_LOGS_THRESHOLD_SLOW:
                self.block_batch_size_adjuster.decrease()
            latest_confirmed_block = self.web3.eth.get_block(to_block)
            self.last_fetched_block = to_block
            return PollResult(polled_block_number=to_block, polled_block_hash=BlockHash(bytes(latest_confirmed_block['hash'])), polled_block_gas_limit=BlockGasLimit(latest_confirmed_block['gasLimit']), events=decoded_result)

    def _query_and_track(self, from_block: BlockNumber, to_block: BlockIdentifier) -> Tuple[List[DecodedEvent], float]:
        """Query the blockchain up to `to_block` and create the filters for the
        smart contracts deployed during the current batch.

        Because of how polling is optimized, filters for smart contracts
        deployed in the current batch must be created, queried, and be merged
        into the same batch. This is necessary to avoid race conditions on
        restarts that could lead to loss of events. Example:

                   last confirmed block
                   |
                   v    v end of current batch / new confirmed block
                   4    9
        Batch  ####------
        TNR    ####--*---
        TN           --*-
                     ^ ^
                     | new channel opened
                     |
                     new token network registered

        For this example, the current batch is fetching the range `[4, 9]`. In
        this range a new token is registered at block 6, at block 8 a new
        channel is opened in the new network.

        If the events of the new TN are *not* queried, block 9 will be
        confirmed after processing the batch which adds the TN, and iff the
        node crashes right after processing this batch, on the next restart
        *all* filters will start from 9, thus missing the event for the new
        channel on block 8.
        """
        max_request_duration = 0
        result = []
        event_filter = self.event_filter
        i = 0
        while event_filter:
            i += 1
            blockchain_events = []
            for filter_params in event_filter.to_web3_filters(self.contract_manager, from_block, to_block, self.node_address):
                log.debug('Querying new blockchain events', from_block=from_block, to_block=to_block, event_filter=event_filter, filter_params=filter_params, i=i, node=to_checksum_address(self.node_address))
                filter_name = filter_params.pop('_name')
                try:
                    start = time.monotonic()
                    new_events = self.web3.manager.request_blocking(RPCEndpoint('eth_getLogs'), [filter_params])
                    request_duration = time.monotonic() - start
                    max_request_duration = max(max_request_duration, request_duration)
                except ReadTimeout as ex:
                    raise EthGetLogsTimeout() from ex
                log.debug('Fetched new blockchain events', from_block=filter_params['fromBlock'], to_block=filter_params['toBlock'], addresses=filter_params['address'], filter_name=filter_name, new_events=new_events, request_duration=request_duration, i=i, node=to_checksum_address(self.node_address))
                blockchain_events.extend(new_events)
            if blockchain_events:
                decoded_events = [decode_raiden_event_to_internal(self._address_to_abi[to_canonical_address(event['address'])], self.chain_id, event) for event in blockchain_events]
                sort_events(decoded_events)
                from dataclasses import asdict
                log.debug('Decoded new blockchain events', decoded_events=[asdict(e) for e in decoded_events], node=to_checksum_address(self.node_address))
                result.extend(decoded_events)
                for listener in self._listeners:
                    listener(decoded_events)
                event_filter = new_filters_from_events(decoded_events)
                self.event_filter = self.event_filter.union(event_filter)
                self._address_to_abi.update(event_filter.abi_of_contract_address(self.contract_manager))
            else:
                event_filter = None
        return (result, max_request_duration)

    def stop(self) -> None:
        with self._filters_lock:
            self._address_to_abi = {}
        del self._listeners[:]

    def register_listener(self, listener: Callable[[List[DecodedEvent]], None]) -> None:
        self._listeners.append(listener)

    def unregister_listener(self, listener: Callable[[List[DecodedEvent]], None]) -> None:
        self._listeners.remove(listener)
