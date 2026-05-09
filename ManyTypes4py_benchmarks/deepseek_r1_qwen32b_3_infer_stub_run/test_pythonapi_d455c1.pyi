from typing import List, Optional, Tuple, Union
from eth_utils import Address
from raiden.settings import BlockTimeout
from raiden.tests.utils.protocol import HoldRaidenEventHandler
from raiden.transfer.state import ChannelState
from raiden.utils.typing import (
    BlockNumber,
    PaymentAmount,
    TargetAddress,
    TokenAddress,
    TokenAmount,
)
from raiden_contracts.contract_manager import ContractManager


def test_register_token(
    raiden_network: List[RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None:
    ...


def test_register_token_insufficient_eth(
    raiden_network: List[RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None:
    ...


def test_token_registered_race(
    raiden_chain: Tuple[RaidenService, RaidenService],
    retry_timeout: float,
    unregistered_token: TokenAddress,
) -> None:
    ...


def test_deposit_updates_balance_immediately(
    raiden_chain: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    ...


def test_transfer_with_invalid_address_type(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    ...


def test_insufficient_funds(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None:
    ...


def test_funds_check_for_openchannel(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    contract_manager: ContractManager,
    retry_timeout: float,
) -> None:
    ...


def test_payment_timing_out_if_partner_does_not_respond(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    reveal_timeout: BlockTimeout,
    retry_timeout: float,
) -> None:
    ...


def test_participant_deposit_amount_must_be_smaller_than_the_limit(
    raiden_network: List[RaidenService],
    contract_manager: ContractManager,
    retry_timeout: float,
) -> None:
    ...


def test_deposit_amount_must_be_smaller_than_the_token_network_limit(
    raiden_network: List[RaidenService],
    contract_manager: ContractManager,
    retry_timeout: float,
) -> None:
    ...


def test_token_addresses(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    ...


def run_test_token_addresses(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    ...


def test_raidenapi_channel_lifecycle(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    settle_timeout_max: BlockTimeout,
) -> None:
    ...


def test_same_addresses_for_payment(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    ...