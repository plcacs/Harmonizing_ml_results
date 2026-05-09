from typing import Any, List, Optional, Tuple
from eth_utils import Address
from raiden.settings import BlockTimeout
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.transfer.state import ChannelState
from raiden.utils.typing import BlockNumber, PaymentAmount, TokenAmount, TargetAddress

def test_register_token(raiden_network: List[Any], retry_timeout: int, unregistered_token: Address) -> None:
    ...

def test_register_token_insufficient_eth(raiden_network: List[Any], retry_timeout: int, unregistered_token: Address) -> None:
    ...

def test_token_registered_race(raiden_chain: Tuple[Any, Any], retry_timeout: int, unregistered_token: Address) -> None:
    ...

def test_deposit_updates_balance_immediately(raiden_chain: Tuple[Any, Any], token_addresses: List[Address]) -> None:
    ...

def test_transfer_with_invalid_address_type(raiden_network: List[Any], token_addresses: List[Address]) -> None:
    ...

def test_insufficient_funds(raiden_network: List[Any], token_addresses: List[Address], deposit: int) -> None:
    ...

def test_funds_check_for_openchannel(raiden_network: List[Any], token_addresses: List[Address], deposit: int) -> None:
    ...

def test_payment_timing_out_if_partner_does_not_respond(raiden_network: List[Any], token_addresses: List[Address], reveal_timeout: int, retry_timeout: int) -> None:
    ...

def test_participant_deposit_amount_must_be_smaller_than_the_limit(raiden_network: List[Any], contract_manager: Any, retry_timeout: int) -> None:
    ...

def test_deposit_amount_must_be_smaller_than_the_token_network_limit(raiden_network: List[Any], contract_manager: Any, retry_timeout: int) -> None:
    ...

def test_token_addresses(raiden_network: List[Any], token_addresses: List[Address]) -> None:
    ...

def run_test_token_addresses(raiden_network: List[Any], token_addresses: List[Address]) -> None:
    ...

def test_raidenapi_channel_lifecycle(raiden_network: List[Any], token_addresses: List[Address], deposit: int, retry_timeout: int, settle_timeout_max: BlockTimeout) -> None:
    ...