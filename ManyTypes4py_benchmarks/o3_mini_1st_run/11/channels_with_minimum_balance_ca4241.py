#!/usr/bin/env python3
from gevent import monkey
monkey.patch_all()
import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, NewType, Optional, Tuple
from urllib.parse import urlsplit
import gevent
import requests
import structlog
from raiden.network.transport.matrix.rtc.utils import setup_asyncio_event_loop

setup_asyncio_event_loop()

NODE_SECTION_RE = re.compile('^node[0-9]+')
API_VERSION = 'v1'
Address = NewType('Address', str)
log = structlog.get_logger(__name__)

@dataclass
class ChannelNew:
    token_address: str
    participant: str
    partner: str
    endpoint: str
    initial_deposit: int

@dataclass
class ChannelDeposit:
    token_address: str
    partner: str
    endpoint: str
    minimum_capacity: int

OpenQueue = Dict[str, List[ChannelNew]]
DepositQueue = Dict[Tuple[str, str], List[ChannelDeposit]]

def http_response_is_okay(response: Optional[requests.Response]) -> bool:
    return (
        response is not None and
        response.headers.get('Content-Type') == 'application/json' and
        (response.status_code == HTTPStatus.OK)
    )

def http_response_is_created(response: Optional[requests.Response]) -> bool:
    return (
        response is not None and
        response.headers.get('Content-Type') == 'application/json' and
        (response.status_code == HTTPStatus.CREATED)
    )

def is_json_reponse(response: Optional[requests.Response]) -> bool:
    return response is not None and response.headers.get('Content-Type') == 'application/json'

def channel_details(endpoint: str, token_address: str, partner: str) -> Optional[Dict[str, Any]]:
    url_channel = f'{endpoint}/api/{API_VERSION}/channels/{token_address}/{partner}'
    channel_response: requests.Response = requests.get(url_channel)
    if not is_json_reponse(channel_response):
        raise RuntimeError(f'Unexpected response from server, {channel_response}')
    if channel_response.status_code == HTTPStatus.OK:
        return channel_response.json()
    return None

def necessary_deposit(channel_details: Dict[str, Any], minimum_capacity: int) -> int:
    balance_current: int = int(channel_details['balance'])
    return minimum_capacity - balance_current

def channel_open(open_queue: List[ChannelNew]) -> None:
    """
    As of 0.100.5 channels cannot be opened in parallel, starting multiple
    opens at the same time can lead to the HTTP request timing out.  Therefore
    channels are opened one after the other.
    """
    for chan_open in open_queue:
        chan_details = channel_details(chan_open.endpoint, chan_open.token_address, chan_open.partner)
        assert chan_details is None, 'Channel already exists, the operation should not have been scheduled.'
        channel_open_request: Dict[str, Any] = {
            'token_address': chan_open.token_address,
            'partner_address': chan_open.partner,
            'total_deposit': chan_open.initial_deposit
        }
        log.info(f'Opening {chan_open}')
        url_channel_open: str = f'{chan_open.endpoint}/api/{API_VERSION}/channels'
        response: requests.Response = requests.put(url_channel_open, json=channel_open_request)
        assert http_response_is_created(response), (response, response.text)

def channel_deposit_with_the_same_token_network(deposit_queue: List[ChannelDeposit]) -> None:
    """
    Because of how the ERC20 standard is defined, two concurrent approve
    calls overwrite each other.
    
    Additionally, to prevent a node from trying to deposit more tokens than it
    has, and by consequence sending an unnecessary transaction, a lock is used.
    This has the side effect of forbidding concurrent deposits on the same token network.
    """
    while deposit_queue:
        to_delete: List[int] = []
        for pos, channel_deposit in enumerate(deposit_queue):
            channel = channel_details(channel_deposit.endpoint, channel_deposit.token_address, channel_deposit.partner)
            if channel is None:
                continue
            deposit = necessary_deposit(channel, channel_deposit.minimum_capacity)
            to_delete.append(pos)
            if deposit:
                current_total_deposit: int = int(channel['total_deposit'])
                new_total_deposit: int = current_total_deposit + deposit
                deposit_json: Dict[str, Any] = {'total_deposit': new_total_deposit}
                log.info(f'Depositing to channel {channel_deposit}')
                url_channel: str = f'{channel_deposit.endpoint}/api/{API_VERSION}/channels/{channel_deposit.token_address}/{channel_deposit.partner}'
                response: requests.Response = requests.patch(url_channel, json=deposit_json)
                assert http_response_is_okay(response), (response, response.text)
            else:
                log.info(f'Channel exists and has enough capacity {channel_deposit}')
        for pos in reversed(to_delete):
            deposit_queue.pop(pos)

def queue_channel_open(
    nodeaddress_to_channelopenqueue: Dict[str, List[ChannelNew]],
    nodeaddress_to_channeldepositqueue: Dict[Tuple[str, str], List[ChannelDeposit]],
    channel: Dict[str, Any],
    token_address: str,
    node_to_address: Dict[str, str],
    node_to_endpoint: Dict[str, str]
) -> None:
    node1: str = channel['node1']
    node2: str = channel['node2']
    participant1: str = node_to_address[node1]
    participant2: str = node_to_address[node2]
    minimum_capacity1: int = channel['minimum_capacity1']
    minimum_capacity2: int = channel['minimum_capacity2']
    is_node1_with_less_work: bool = len(nodeaddress_to_channelopenqueue[participant1]) < len(nodeaddress_to_channelopenqueue[participant2])
    if is_node1_with_less_work:
        channelnew_participant: str = participant1
        channelnew_partner: str = participant2
        channelnew_endpoint: str = node_to_endpoint[node1]
        channelnew_minimum_capacity: int = minimum_capacity1
        channeldeposit_partner: str = participant1
        channeldeposit_endpoint: str = node_to_endpoint[node2]
        channeldeposit_minimum_capacity: int = minimum_capacity2
    else:
        channelnew_participant = participant2
        channelnew_partner = participant1
        channelnew_endpoint = node_to_endpoint[node2]
        channelnew_minimum_capacity = minimum_capacity2
        # here, for the deposit, participant remains participant1 for deposit caller
        channeldeposit_partner = participant2
        channeldeposit_endpoint = node_to_endpoint[node1]
        channeldeposit_minimum_capacity = minimum_capacity1
    channel_new: ChannelNew = ChannelNew(
        token_address=token_address,
        participant=channelnew_participant,
        partner=channelnew_partner,
        endpoint=channelnew_endpoint,
        initial_deposit=channelnew_minimum_capacity
    )
    nodeaddress_to_channelopenqueue[channel_new.participant].append(channel_new)
    log.info(f'Queueing {channel_new}')
    channel_deposit: ChannelDeposit = ChannelDeposit(
        token_address=token_address,
        partner=channeldeposit_partner,
        endpoint=channeldeposit_endpoint,
        minimum_capacity=channeldeposit_minimum_capacity
    )
    # Note: key tuple is (token_address, channeldeposit_party) where channeldeposit_party is the depositor.
    # In the if branch, deposit from partner is queued in participant1; in else branch, deposit from participant2.
    nodeaddress_to_channeldepositqueue[(token_address, channelnew_participant)].append(channel_deposit)
    log.info(f'Queueing {channel_deposit}')

def queue_channel_deposit(
    nodeaddress_to_channeldepositqueue: Dict[Tuple[str, str], List[ChannelDeposit]],
    channel: Dict[str, Any],
    current_channel1: Dict[str, Any],
    current_channel2: Dict[str, Any],
    token_address: str,
    node_to_address: Dict[str, str],
    node_to_endpoint: Dict[str, str]
) -> None:
    node1: str = channel['node1']
    node2: str = channel['node2']
    participant1: str = node_to_address[node1]
    participant2: str = node_to_address[node2]
    endpoint1: str = node_to_endpoint[node1]
    endpoint2: str = node_to_endpoint[node2]
    minimum_capacity1: int = channel['minimum_capacity1']
    minimum_capacity2: int = channel['minimum_capacity2']
    deposit1: int = necessary_deposit(current_channel1, minimum_capacity1)
    if deposit1 > 0:
        channel_deposit: ChannelDeposit = ChannelDeposit(
            token_address=token_address,
            partner=participant2,
            endpoint=endpoint1,
            minimum_capacity=minimum_capacity2
        )
        nodeaddress_to_channeldepositqueue[(token_address, participant1)].append(channel_deposit)
        log.info(f'Queueing {channel_deposit}')
    else:
        log.info(f'Channel already with enough capacity {current_channel1}')
    deposit2: int = necessary_deposit(current_channel2, minimum_capacity2)
    if deposit2 > 0:
        channel_deposit = ChannelDeposit(
            token_address=token_address,
            partner=participant1,
            endpoint=endpoint2,
            minimum_capacity=minimum_capacity1
        )
        nodeaddress_to_channeldepositqueue[(token_address, participant2)].append(channel_deposit)
        log.info(f'Queueing {channel_deposit}')
    else:
        log.info(f'Channel already with enough capacity {current_channel2}')

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    with open(args.config, 'r') as handler:
        config: Dict[str, Any] = json.load(handler)
    node_to_endpoint: Dict[str, str] = {}
    node_to_address: Dict[str, str] = {}
    for node_name, node_info in config['nodes'].items():
        if urlsplit(node_info['endpoint']).scheme == '':
            raise ValueError("'endpoint' must have the protocol defined")
        url_deposit: str = f"{node_info['endpoint']}/api/{API_VERSION}/address"
        result: Dict[str, Any] = requests.get(url_deposit).json()
        if result['our_address'] != node_info['address']:
            raise ValueError(f"Address mismatch, configuration {node_info['address']}, API response {result['our_address']}")
        node_to_endpoint[node_name] = node_info['endpoint']
        node_to_address[node_name] = node_info['address']
    nodeaddress_to_channelopenqueue: OpenQueue = defaultdict(list)
    nodeaddress_to_channeldepositqueue: DepositQueue = defaultdict(list)
    for token_address, channels_to_open in config['networks'].items():
        for channel in channels_to_open:
            node1: str = channel['node1']
            node2: str = channel['node2']
            participant1: str = node_to_address[node1]
            participant2: str = node_to_address[node2]
            current_channel1: Optional[Dict[str, Any]] = channel_details(node_to_endpoint[node1], token_address, participant2)
            current_channel2: Optional[Dict[str, Any]] = channel_details(node_to_endpoint[node2], token_address, participant1)
            nodes_are_synchronized: bool = bool(current_channel1) == bool(current_channel2)
            msg: str = f'The channel must exist in both or neither of the nodes.\n{current_channel1}\n{current_channel2}'
            assert nodes_are_synchronized, msg
            if current_channel1 is None:
                queue_channel_open(nodeaddress_to_channelopenqueue, nodeaddress_to_channeldepositqueue, channel, token_address, node_to_address, node_to_endpoint)
            else:
                # current_channel1 and current_channel2 exist
                assert current_channel1 is not None and current_channel2 is not None
                queue_channel_deposit(nodeaddress_to_channeldepositqueue, channel, current_channel1, current_channel2, token_address, node_to_address, node_to_endpoint)
    open_greenlets = {
        gevent.spawn(channel_open, open_queue) for open_queue in nodeaddress_to_channelopenqueue.values()
    }
    deposit_greenlets = {
        gevent.spawn(channel_deposit_with_the_same_token_network, deposit_queue) for deposit_queue in nodeaddress_to_channeldepositqueue.values()
    }
    all_greenlets = set()
    all_greenlets.update(open_greenlets)
    all_greenlets.update(deposit_greenlets)
    gevent.joinall(all_greenlets, raise_error=True)

if __name__ == '__main__':
    main()