from http import HTTPStatus
import grequests
import pytest
from eth_utils import decode_hex, to_checksum_address, to_hex
from raiden.api.rest import APIServer
from raiden.constants import UINT64_MAX
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.integration.api.rest.utils import api_url_for, assert_payment_conflict, assert_payment_secret_and_hash, assert_proper_response, assert_response_with_error, get_json_response
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.transfer import watch_for_unlock_failures
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import List, Secret, Any, Dict, Tuple, Optional, Union
from typing import cast

DEFAULT_AMOUNT: str = '200'
DEFAULT_ID: str = '42'

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_target_error(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    pfs_mock: Any
) -> None:
    _, app1 = raiden_network
    token_address = token_addresses[0]
    target_address = app1.address
    pfs_mock.add_apps(raiden_network)
    app1.stop()
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={'amount': DEFAULT_AMOUNT, 'identifier': DEFAULT_ID}
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    deposit: int,
    pfs_mock: Any
) -> None:
    _, app1 = raiden_network
    amount = 100
    identifier = 42
    token_address = token_addresses[0]
    target_address = app1.address
    pfs_mock.add_apps(raiden_network)
    our_address = api_server_test_instance.rest_api.raiden_api.address
    payment = {
        'initiator_address': to_checksum_address(our_address),
        'target_address': to_checksum_address(target_address),
        'token_address': to_checksum_address(token_address),
        'amount': str(amount),
        'identifier': str(identifier)
    }
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={'amount': str(amount), 'identifier': str(identifier)}
    )
    with watch_for_unlock_failures(*raiden_network):
        response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    assert_payment_secret_and_hash(json_response, payment)
    payment['amount'] = '1'
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={'amount': '1'}
    )
    with watch_for_unlock_failures(*raiden_network):
        response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    assert_payment_secret_and_hash(json_response, payment)
    payment['amount'] = str(deposit)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={'amount': str(deposit)}
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)
    limit = 5
    request = grequests.get(
        api_url_for(
            api_server_test_instance,
            'raideninternaleventsresource',
            limit=limit,
            offset=0
        )
    )
    response = request.send().response
    assert_proper_response(response)
    events = response.json()
    assert len(events) == limit
    assert all(('TimestampedEvent' in event for event in events))

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_without_pfs(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    deposit: int
) -> None:
    _, app1 = raiden_network
    amount = 100
    identifier = 42
    token_address = token_addresses[0]
    target_address = app1.address
    our_address = api_server_test_instance.rest_api.raiden_api.address
    our_metadata = api_server_test_instance.rest_api.raiden_api.raiden.transport.address_metadata
    payment = {
        'initiator_address': to_checksum_address(our_address),
        'target_address': to_checksum_address(target_address),
        'token_address': to_checksum_address(token_address),
        'amount': str(amount),
        'identifier': str(identifier)
    }
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': str(amount),
            'identifier': str(identifier),
            'paths': [{
                'route': [
                    to_checksum_address(our_address),
                    to_checksum_address(app1.address)
                ],
                'address_metadata': {
                    to_checksum_address(our_address): our_metadata,
                    to_checksum_address(app1.address): app1.transport.address_metadata
                }
            }]
        }
    )
    with watch_for_unlock_failures(*raiden_network):
        response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    assert_payment_secret_and_hash(json_response, payment)
    payment['amount'] = '1'
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': str(amount),
            'paths': [{
                'route': [
                    to_checksum_address(our_address),
                    to_checksum_address(app1.address)
                ],
                'address_metadata': {
                    to_checksum_address(our_address): our_metadata,
                    to_checksum_address(app1.address): app1.transport.address_metadata
                }
            }]
        }
    )
    payment['amount'] = str(deposit)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': str(deposit),
            'identifier': str(identifier),
            'paths': [{
                'route': [
                    to_checksum_address(our_address),
                    to_checksum_address(app1.address)
                ],
                'address_metadata': {
                    to_checksum_address(our_address): our_metadata,
                    to_checksum_address(app1.address): app1.transport.address_metadata
                }
            }]
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_without_pfs_failure(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str]
) -> None:
    app0, app1 = raiden_network
    amount = 100
    identifier = 42
    token_address = token_addresses[0]
    target_address = app1.address
    our_address = api_server_test_instance.rest_api.raiden_api.address
    our_metadata = api_server_test_instance.rest_api.raiden_api.raiden.transport.address_metadata

    def send_request(paths: List[Dict[str, Any]]) -> Any:
        request = grequests.post(
            api_url_for(
                api_server_test_instance,
                'token_target_paymentresource',
                token_address=to_checksum_address(token_address),
                target_address=to_checksum_address(target_address)
            ),
            json={
                'amount': str(amount),
                'identifier': str(identifier),
                'paths': paths
            }
        )
        return request.send().response
    paths = [{
        'route': [
            to_checksum_address(our_address),
            to_checksum_address(app0.address)
        ],
        'address_metadata': {
            to_checksum_address(our_address): our_metadata,
            to_checksum_address(app1.address): app1.transport.address_metadata
        }
    }]
    response = send_request(paths)
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)
    paths = [{
        'fake_route': [
            to_checksum_address(our_address),
            to_checksum_address(app0.address)
        ],
        'fake_address_metadata': {
            to_checksum_address(our_address): our_metadata,
            to_checksum_address(app1.address): app1.transport.address_metadata
        }
    }]
    response = send_request(paths)
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)
    paths = [{
        'route': ['fake_app0', 'fake_app1'],
        'address_metadata': {
            'fake_app0': our_metadata,
            'fake_app1': app1.transport.address_metadata
        }
    }]
    response = send_request(paths)
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_secret_hash_errors(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    pfs_mock: Any
) -> None:
    _, app1 = raiden_network
    token_address = token_addresses[0]
    target_address = app1.address
    secret = to_hex(factories.make_secret())
    bad_secret = 'Not Hex String. 0x78c8d676e2f2399aa2a015f3433a2083c55003591a0f3f33'
    bad_secret_hash = 'Not Hex String. 0x78c8d676e2f2399aa2a015f3433a2083c55003591a0f3f33'
    short_secret = '0x123'
    short_secret_hash = 'Short secret hash'
    pfs_mock.add_apps(raiden_network)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret': short_secret
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret': bad_secret
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret_hash': short_secret_hash
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret_hash': bad_secret_hash
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.BAD_REQUEST)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret': secret,
            'secret_hash': secret
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_secret_no_hash(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    pfs_mock: Any
) -> None:
    _, app1 = raiden_network
    token_address = token_addresses[0]
    target_address = app1.address
    secret = to_hex(factories.make_secret())
    our_address = api_server_test_instance.rest_api.raiden_api.address
    pfs_mock.add_apps(raiden_network)
    payment = {
        'initiator_address': to_checksum_address(our_address),
        'target_address': to_checksum_address(target_address),
        'token_address': to_checksum_address(token_address),
        'amount': DEFAULT_AMOUNT,
        'identifier': DEFAULT_ID
    }
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret': secret
        }
    )
    with watch_for_unlock_failures(*raiden_network):
        response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    assert_payment_secret_and_hash(json_response, payment)
    assert secret == json_response['secret']

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_hash_no_secret(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    pfs_mock: Any
) -> None:
    _, app1 = raiden_network
    token_address = token_addresses[0]
    target_address = app1.address
    secret_hash = factories.make_secret_hash()
    pfs_mock.add_apps(raiden_network)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_target_paymentresource',
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(target_address)
        ),
        json={
            'amount': DEFAULT_AMOUNT,
            'identifier': DEFAULT_ID,
            'secret_hash': to_hex(secret_hash)
        }
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.CONFLICT)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_post_without_required_params(
    api_server_test_instance: APIServer,
    token_addresses: List[str]
) -> None:
    token_address = token_addresses[0]
    request = grequests.post(
        api_url_for(api_server_test_instance, 'paymentresource')
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.METHOD_NOT_ALLOWED)
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            'token_paymentresource',
            token_address=to_checksum_address(token_address)
        )
    )
    response = request.send().response
    assert_proper_response(response, status_code=HTTPStatus.METHOD_NOT