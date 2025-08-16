from typing import List, Dict, Any, Tuple

def num_services() -> int:
    return 2

def services(num_services: int, matrix_transports: List[Any]) -> List[str]:
    service_addresses_to_expiry: Dict[str, int] = {factories.make_address(): 9999 for _ in range(num_services)}
    for transport in matrix_transports:
        transport.update_services_addresses(service_addresses_to_expiry)
    return [to_normalized_address(addr) for addr in service_addresses_to_expiry.keys()]

def number_of_transports() -> int:
    return 1

class MessageHandler:

    def __init__(self, bag: Dict[str, Any]):
        self.bag = bag

def get_to_device_broadcast_messages(to_device_mock: MagicMock, expected_receiver_addresses: List[str], device_id: str) -> List[Any]:
    collected_messages = []
    for _, kwargs in to_device_mock.call_args_list:
        assert kwargs['event_type'] == 'm.room.message'
        messages_batch = []
        addresses = []
        for address, to_device_dict in kwargs['messages'].items():
            addresses.append(address.split(':')[0][1:])
            assert to_device_dict.keys() == {device_id}
            messages = to_device_dict[device_id]['body'].split('\n')
            messages = [MessageSerializer.deserialize(message) for message in messages]
            if not messages_batch:
                messages_batch = messages
            else:
                assert messages_batch == messages
        assert len(addresses) == len(expected_receiver_addresses)
        assert set(addresses) == set(expected_receiver_addresses)
        collected_messages += messages_batch
    return collected_messages

def ping_pong_message_success(transport0: Any, transport1: Any) -> bool:
    queueid0 = QueueIdentifier(recipient=transport0._raiden_service.address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
    queueid1 = QueueIdentifier(recipient=transport1._raiden_service.address, canonical_identifier=CANONICAL_IDENTIFIER_UNORDERED_QUEUE)
    transport0_raiden_queues = views.get_all_messagequeues(views.state_from_raiden(transport0._raiden_service))
    transport1_raiden_queues = views.get_all_messagequeues(views.state_from_raiden(transport1._raiden_service))
    transport0_raiden_queues[queueid1] = []
    transport1_raiden_queues[queueid0] = []
    received_messages0 = transport0._raiden_service.message_handler.bag
    received_messages1 = transport1._raiden_service.message_handler.bag
    msg_id = random.randint(100000.0, 900000.0)
    ping_message = Processed(message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE)
    pong_message = Delivered(delivered_message_identifier=MessageID(msg_id), signature=EMPTY_SIGNATURE)
    transport0_raiden_queues[queueid1].append(ping_message)
    transport0._raiden_service.sign(ping_message)
    transport1._raiden_service.sign(pong_message)
    transport0.send_async([MessagesQueue(queueid1, [(ping_message, transport1.address_metadata)])])
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, exception=False):
        all_messages_received = False
        while not all_messages_received:
            all_messages_received = ping_message in received_messages1 and pong_message in received_messages0
            gevent.sleep(0.1)
    assert ping_message in received_messages1
    assert pong_message in received_messages0
    transport0_raiden_queues[queueid1].clear()
    transport1_raiden_queues[queueid0].append(ping_message)
    transport0._raiden_service.sign(pong_message)
    transport1._raiden_service.sign(ping_message)
    transport1.send_async([MessagesQueue(queueid0, [(ping_message, transport0.address_metadata)])])
    with Timeout(TIMEOUT_MESSAGE_RECEIVE, exception=False):
        all_messages_received = False
        while not all_messages_received:
            all_messages_received = ping_message in received_messages0 and pong_message in received_messages1
            gevent.sleep(0.1)
    assert ping_message in received_messages0
    assert pong_message in received_messages1
    transport1_raiden_queues[queueid0].clear()
    return all_messages_received

def is_reachable(transport: Any, address: str) -> None:
    raise NotImplementedError

def _wait_for_peer_reachability(transport: Any, target_address: str, target_reachability: str, timeout: int = 5) -> None:
    raise NotImplementedError
    with Timeout(timeout):
        while True:
            gevent.sleep(0.1)

def wait_for_peer_unreachable(transport: Any, target_address: str, timeout: int = 5) -> None:
    _wait_for_peer_reachability(transport=transport, target_address=target_address, target_reachability=AddressReachability.UNREACHABLE, timeout=timeout)

def wait_for_peer_reachable(transport: Any, target_address: str, timeout: int = 5) -> None:
    _wait_for_peer_reachability(transport=transport, target_address=target_address, target_reachability=AddressReachability.REACHABLE, timeout=timeout)
