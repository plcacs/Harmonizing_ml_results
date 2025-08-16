from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Union

import gevent
import grequests
import pytest
from eth_utils import to_canonical_address, to_checksum_address
from gevent import Greenlet
from requests import Response

from raiden.api.rest import APIServer
from raiden.constants import BLOCK_ID_LATEST, NULL_ADDRESS_HEX
from raiden.raiden_service import RaidenService
from raiden.tests.integration.api.rest.test_rest import DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT
from raiden.tests.integration.api.rest.utils import (
    api_url_for,
    assert_proper_response,
    assert_response_with_code,
    assert_response_with_error,
    get_json_response,
)
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import check_dict_nested_attrs
from raiden.transfer import views
from raiden.transfer.state import ChannelState
from raiden.utils.typing import TokenAmount, TokenNetworkRegistryAddress
from raiden.waiting import wait_for_participant_deposit
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN


@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [1])
@pytest.mark.parametrize("channels_per_node", [0])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_channel_status_channel_nonexistant(
    api_server_test_instance: APIServer, token_addresses: List[str]
) -> None:
    partner_address = "0x61C808D82A3Ac53231750daDc13c777b59310bD9"
    token_address = token_addresses[0]

    request = grequests.get(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=partner_address,
        )
    )
    response = request.send().response
    assert_proper_response(response, HTTPStatus.NOT_FOUND)
    assert get_json_response(response)["errors"] == (
        "Channel with partner '{}' for token '{}' could not be found.".format(
            to_checksum_address(partner_address), to_checksum_address(token_address)
        )
    )


@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [1])
@pytest.mark.parametrize("channels_per_node", [0])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_channel_open_and_deposit(
    api_server_test_instance: APIServer, token_addresses: List[str], reveal_timeout: int
) -> None:
    first_partner_address = "0x61C808D82A3Ac53231750daDc13c777b59310bD9"
    token_address = token_addresses[0]
    token_address_hex = to_checksum_address(token_address)
    settle_timeout = 1650
    channel_data_obj: Dict[str, Any] = {
        "partner_address": first_partner_address,
        "token_address": token_address_hex,
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
    }
    # First let's try to create channel with the null address and see error is handled
    channel_data_obj["partner_address"] = NULL_ADDRESS_HEX
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response
    assert_response_with_error(response, status_code=HTTPStatus.BAD_REQUEST)
    # now let's really create a new channel
    channel_data_obj["partner_address"] = first_partner_address
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response

    assert_proper_response(response, HTTPStatus.CREATED)
    first_channel_id = 1
    json_response = get_json_response(response)
    expected_response = channel_data_obj.copy()
    expected_response.update(
        {
            "balance": "0",
            "state": ChannelState.STATE_OPENED.value,
            "channel_identifier": "1",
            "total_deposit": "0",
        }
    )
    assert check_dict_nested_attrs(json_response, expected_response)
    token_network_address = json_response["token_network_address"]

    # Now let's try to open the same channel again, because it is possible for
    # the participants to race on the channel creation, this is not considered
    # an error.
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response
    assert_proper_response(response, HTTPStatus.OK)
    json_response = get_json_response(response)
    assert check_dict_nested_attrs(json_response, expected_response)

    # now let's open a channel and make a deposit too
    second_partner_address = "0x29FA6cf0Cce24582a9B20DB94Be4B6E017896038"
    total_deposit = 100
    channel_data_obj = {
        "partner_address": second_partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
        "total_deposit": str(total_deposit),
    }
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response

    assert_proper_response(response, HTTPStatus.CREATED)
    second_channel_id = 2
    json_response = get_json_response(response)
    expected_response = channel_data_obj.copy()
    expected_response.update(
        {
            "balance": str(total_deposit),
            "state": ChannelState.STATE_OPENED.value,
            "channel_identifier": str(second_channel_id),
            "token_network_address": token_network_address,
            "total_deposit": str(total_deposit),
        }
    )
    assert check_dict_nested_attrs(json_response, expected_response)

    # assert depositing again with less than the initial deposit returns 409
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=second_partner_address,
        ),
        json={"total_deposit": "99"},
    )
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CONFLICT)

    # assert depositing negative amount fails
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=first_partner_address,
        ),
        json={"total_deposit": "-1000"},
    )
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CONFLICT)

    # let's deposit on the first channel
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=first_partner_address,
        ),
        json={"total_deposit": str(total_deposit)},
    )
    response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    expected_response = {
        "channel_identifier": str(first_channel_id),
        "partner_address": first_partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
        "state": ChannelState.STATE_OPENED.value,
        "balance": str(total_deposit),
        "total_deposit": str(total_deposit),
        "token_network_address": token_network_address,
    }
    assert check_dict_nested_attrs(json_response, expected_response)

    # let's try querying for the second channel
    request = grequests.get(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=second_partner_address,
        )
    )

    response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    expected_response = {
        "channel_identifier": str(second_channel_id),
        "partner_address": second_partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
        "state": ChannelState.STATE_OPENED.value,
        "balance": str(total_deposit),
        "total_deposit": str(total_deposit),
        "token_network_address": token_network_address,
    }
    assert check_dict_nested_attrs(json_response, expected_response)

    # finally let's burn all eth and try to open another channel
    burn_eth(api_server_test_instance.rest_api.raiden_api.raiden.rpc_client)
    channel_data_obj = {
        "partner_address": "0xf3AF96F89b3d7CdcBE0C083690A28185Feb0b3CE",
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
    }
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response
    assert_proper_response(response, HTTPStatus.PAYMENT_REQUIRED)
    json_response = get_json_response(response)
    assert "The account balance is below the estimated amount" in json_response["errors"]


@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [1])
@pytest.mark.parametrize("channels_per_node", [0])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_channel_open_and_deposit_race(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[str],
    reveal_timeout: int,
    token_network_registry_address: TokenNetworkRegistryAddress,
    retry_timeout: int,
) -> None:
    """Tests that a race for the same deposit from the API is handled properly

    The proxy's approve_and_set_total_deposit is raising a
    RaidenRecoverableError in case of races. That needs to be properly handled
    and not allowed to bubble out of the greenlet.

    Regression test for https://github.com/raiden-network/raiden/issues/4937
    """
    app0 = raiden_network[0]
    # let's create a new channel
    first_partner_address = "0x61C808D82A3Ac53231750daDc13c777b59310bD9"
    token_address = token_addresses[0]
    settle_timeout = 1650
    channel_data_obj: Dict[str, Any] = {
        "partner_address": first_partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
    }

    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response

    assert_proper_response(response, HTTPStatus.CREATED)
    json_response = get_json_response(response)
    expected_response = channel_data_obj.copy()
    expected_response.update(
        {
            "balance": "0",
            "state": ChannelState.STATE_OPENED.value,
            "channel_identifier": "1",
            "total_deposit": "0",
        }
    )
    assert check_dict_nested_attrs(json_response, expected_response)

    # Prepare the deposit api call
    deposit_amount = TokenAmount(99)
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=first_partner_address,
        ),
        json={"total_deposit": str(deposit_amount)},
    )

    # Spawn two greenlets doing the same deposit request
    greenlets: List[Greenlet] = [gevent.spawn(request.send), gevent.spawn(request.send)]
    gevent.joinall(set(greenlets), raise_error=True)
    # Make sure that both responses are fine
    g1_response = greenlets[0].get().response
    assert_proper_response(g1_response, HTTPStatus.OK)
    json_response = get_json_response(g1_response)
    expected_response.update(
        {"total_deposit": str(deposit_amount), "balance": str(deposit_amount)}
    )
    assert check_dict_nested_attrs(json_response, expected_response)
    g2_response = greenlets[0].get().response
    assert_proper_response(g2_response, HTTPStatus.OK)
    json_response = get_json_response(g2_response)
    assert check_dict_nested_attrs(json_response, expected_response)

    # Wait for the deposit to be seen
    timeout_seconds = 20
    exception = Exception(f"Expected deposit not seen within {timeout_seconds}")
    with gevent.Timeout(seconds=timeout_seconds, exception=exception):
        wait_for_participant_deposit(
            raiden=app0,
            token_network_registry_address=token_network_registry_address,
            token_address=token_address,
            partner_address=to_canonical_address(first_partner_address),
            target_address=app0.address,
            target_balance=deposit_amount,
            retry_timeout=retry_timeout,
        )

    request = grequests.get(api_url_for(api_server_test_instance, "channelsresource"))
    response = request.send().response
    assert_proper_response(response, HTTPStatus.OK)
    json_response = get_json_response(response)
    channel_info = json_response[0]
    assert channel_info["token_address"] == to_checksum_address(token_address)
    assert channel_info["total_deposit"] == str(deposit_amount)


@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [1])
@pytest.mark.parametrize("channels_per_node", [0])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_channel_open_close_and_settle(
    api_server_test_instance: APIServer,
    token_addresses: List[str],
    reveal_timeout: int,
) -> None:
    # let's create a new channel
    partner_address = "0x61C808D82A3Ac53231750daDc13c777b59310bD9"
    token_address = token_addresses[0]
    settle_timeout = 1650
    channel_data_obj: Dict[str, Any] = {
        "partner_address": partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
    }
    request = grequests.put(
        api_url_for(api_server_test_instance, "channelsresource"), json=channel_data_obj
    )
    response = request.send().response

    balance = 0
    assert_proper_response(response, status_code=HTTPStatus.CREATED)
    channel_identifier = 1
    json_response = get_json_response(response)
    expected_response = channel_data_obj.copy()
    expected_response.update(
        {
            "balance": str(balance),
            "state": ChannelState.STATE_OPENED.value,
            "reveal_timeout": str(reveal_timeout),
            "channel_identifier": str(channel_identifier),
            "total_deposit": "0",
        }
    )
    assert check_dict_nested_attrs(json_response, expected_response)

    token_network_address = json_response["token_network_address"]

    # let's close the channel
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=partner_address,
        ),
        json={"state": ChannelState.STATE_CLOSED.value},
    )
    response = request.send().response
    assert_proper_response(response)
    expected_response = {
        "token_network_address": token_network_address,
        "channel_identifier": str(channel_identifier),
        "partner_address": partner_address,
        "token_address": to_checksum_address(token_address),
        "settle_timeout": str(settle_timeout),
        "reveal_timeout": str(reveal_timeout),
        "state": ChannelState.STATE_CLOSED.value,
        "balance": str(balance),
        "total_deposit": str(balance),
    }
    assert check_dict_nested_attrs(get_json_response(response), expected_response)

    # try closing the channel again
    request = grequests.patch(
        api_url_for(
            api_server_test_instance,
            "channelsresourcebytokenandpartneraddress",
            token_address=token_address,
            partner_address=partner_address,
        ),
        json={"state": ChannelState.STATE_CLOSED.value},
    )
    # Closing the channel again should not work
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CONFLICT)

    # Try sending a payment when channel is closed
    request = grequests.post(
        api_url_for(
            api_server_test_instance,
            "token_target_paymentresource",
            token_address=to_checksum_address(token_address),
            target_address=to_checksum_address(partner_address),
        ),
        json={"amount": "1"},
    )
    # Payment should not work since channel is closing
    response = request