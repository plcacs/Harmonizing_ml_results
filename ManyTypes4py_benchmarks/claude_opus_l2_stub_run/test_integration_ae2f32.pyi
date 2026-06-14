import pytest
from _pytest.logging import LogCaptureFixture
from pytest_mock import MockerFixture


def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: dict,
    ticker: dict,
    fee: object,
    mocker: MockerFixture,
) -> None: ...

@pytest.mark.parametrize("balance_ratio,result1", [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(
    default_conf: dict,
    ticker: dict,
    fee: object,
    mocker: MockerFixture,
    balance_ratio: int | float,
    result1: int,
) -> None: ...

def test_dca_buying(
    default_conf_usdt: dict,
    ticker_usdt: object,
    fee: object,
    mocker: MockerFixture,
) -> None: ...

def test_dca_short(
    default_conf_usdt: dict,
    ticker_usdt: object,
    fee: object,
    mocker: MockerFixture,
) -> None: ...

@pytest.mark.parametrize("leverage", [1, 2])
def test_dca_order_adjust(
    default_conf_usdt: dict,
    ticker_usdt: object,
    leverage: int,
    fee: object,
    mocker: MockerFixture,
) -> None: ...

@pytest.mark.parametrize("leverage", [1, 2])
@pytest.mark.parametrize("is_short", [False, True])
def test_dca_order_adjust_entry_replace_fails(
    default_conf_usdt: dict,
    ticker_usdt: object,
    fee: object,
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    is_short: bool,
    leverage: int,
) -> None: ...

@pytest.mark.parametrize("leverage", [1, 2])
def test_dca_exiting(
    default_conf_usdt: dict,
    ticker_usdt: object,
    fee: object,
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    leverage: int,
) -> None: ...

@pytest.mark.parametrize("leverage", [1, 2])
@pytest.mark.parametrize("is_short", [False, True])
def test_dca_handle_similar_open_order(
    default_conf_usdt: dict,
    ticker_usdt: object,
    is_short: bool,
    leverage: int,
    fee: object,
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
) -> None: ...