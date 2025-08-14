# pragma pylint: disable=missing-docstring,W0212,C0103
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, MagicMock, PropertyMock

import pandas as pd
import pytest
from filelock import Timeout
from skopt.space import Integer

from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_hyperopt
from freqtrade.data.history import load_data
from freqtrade.enums import ExitType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.optimize.space import SKDecimal
from freqtrade.strategy import IntParameter
from freqtrade.util import dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    get_args,
    get_markets,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)


def generate_result_metrics() -> Dict[str, Any]:
    return {
        "trade_count": 1,
        "total_trades": 1,
        "avg_profit": 0.1,
        "total_profit": 0.001,
        "profit": 0.01,
        "duration": 20.0,
        "wins": 1,
        "draws": 0,
        "losses": 0,
        "profit_mean": 0.01,
        "profit_total_abs": 0.001,
        "profit_total": 0.01,
        "holding_avg": timedelta(minutes=20),
        "max_drawdown_account": 0.001,
        "max_drawdown_abs": 0.001,
        "loss": 0.001,
        "is_initial_point": 0.001,
        "is_random": False,
        "is_best": 1,
    }


def test_setup_hyperopt_configuration_without_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
    ]

    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config

    assert "position_stacking" not in config
    assert not log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" not in config
    assert "runmode" in config
    assert config["runmode"] == RunMode.HYPEROPT


def test_setup_hyperopt_configuration_with_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--datadir",
        "/foo/bar",
        "--timeframe",
        "1m",
        "--timerange",
        ":100",
        "--enable-position-stacking",
        "--epochs",
        "1000",
        "--spaces",
        "default",
        "--print-all",
    ]

    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert config["runmode"] == RunMode.HYPEROPT

    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert log_has("Parameter -i/--timeframe detected ... Using timeframe: 1m ...", caplog)

    assert "position_stacking" in config
    assert log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" in config
    assert log_has("Parameter --timerange detected: {} ...".format(config["timerange"]), caplog)

    assert "epochs" in config
    assert log_has(
        "Parameter --epochs detected ... Will run Hyperopt with for 1000 epochs ...", caplog
    )

    assert "spaces" in config
    assert log_has("Parameter -s/--spaces detected: {}".format(config["spaces"]), caplog)
    assert "print_all" in config
    assert log_has("Parameter --print-all detected ...", caplog)


def test_setup_hyperopt_configuration_stake_amount(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--stake-amount",
        "1",
        "--starting-balance",
        "2",
    ]
    conf: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf, dict)

    args = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "0.5",
    ]
    with pytest.raises(OperationalException, match=r"Starting balance .* smaller .*"):
        setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)


def test_start_not_installed(mocker: Any, default_conf: Dict[str, Any], import_fails: Any) -> None:
    start_mock: MagicMock = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)

    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.start", start_mock)
    patch_exchange(mocker)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--epochs",
        "5",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
    ]
    pargs: Any = get_args(args)

    with pytest.raises(OperationalException, match=r"Please ensure that the hyperopt dependencies"):
        start_hyperopt(pargs)


def test_start_no_hyperopt_allowed(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    start_mock: MagicMock = MagicMock()
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.start", start_mock)
    patch_exchange(mocker)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--hyperopt",
        "HyperoptTestSepFile",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs: Any = get_args(args)
    with pytest.raises(OperationalException, match=r"Using separate Hyperopt files has been.*"):
        start_hyperopt(pargs)


def test_start_no_data(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path) -> None:
    hyperopt_conf["user_data_dir"] = tmp_path
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.data.history.load_pair_history", MagicMock(return_value=pd.DataFrame))
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )

    patch_exchange(mocker)
    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs: Any = get_args(args)
    with pytest.raises(OperationalException, match="No data found. Terminating."):
        start_hyperopt(pargs)

    # Cleanup since that failed hyperopt start leaves a lockfile.
    try:
        Path(Hyperopt.get_lock_filename(hyperopt_conf)).unlink()
    except Exception:
        pass


def test_start_filelock(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    hyperopt_mock: MagicMock = MagicMock(side_effect=Timeout(Hyperopt.get_lock_filename(hyperopt_conf)))
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch("freqtrade.optimize.hyperopt.Hyperopt.__init__", hyperopt_mock)
    patch_exchange(mocker)

    args: List[str] = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        "HyperoptableStrategy",
        "--hyperopt-loss",
        "SharpeHyperOptLossDaily",
        "--epochs",
        "5",
    ]
    pargs: Any = get_args(args)
    start_hyperopt(pargs)
    assert log_has("Another running instance of freqtrade Hyperopt detected.", caplog)


def test_log_results_if_loss_improves(hyperopt: Hyperopt, capsys: Any) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.total_epochs = 2

    hyperopt.print_results(
        {
            "loss": 1,
            "results_metrics": generate_result_metrics(),
            "total_profit": 0,
            "current_epoch": 2,  # This starts from 1 (in a human-friendly manner)
            "is_initial_point": False,
            "is_random": False,
            "is_best": True,
        }
    )
    hyperopt._hyper_out.print()
    out, _err = capsys.readouterr()
    assert all(
        x in out for x in ["Best", "2/2", "1", "0.10%", "0.00100000 BTC    (1.00%)", "0:20:00"]
    )


def test_no_log_if_loss_does_not_improve(hyperopt: Hyperopt, caplog: Any) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.print_results(
        {
            "is_best": False,
            "loss": 3,
            "current_epoch": 1,
        }
    )
    assert caplog.record_tuples == []


def test_roi_table_generation(hyperopt: Hyperopt) -> None:
    params: Dict[str, Any] = {
        "roi_t1": 5,
        "roi_t2": 10,
        "roi_t3": 15,
        "roi_p1": 1,
        "roi_p2": 2,
        "roi_p3": 3,
    }

    assert hyperopt.hyperopter.custom_hyperopt.generate_roi_table(params) == {
        0: 6,
        15: 3,
        25: 1,
        30: 0,
    }


def test_params_no_optimize_details(hyperopt: Hyperopt) -> None:
    hyperopt.hyperopter.config["spaces"] = ["buy"]
    res: Dict[str, Any] = hyperopt.hyperopter._get_no_optimize_details()
    assert isinstance(res, dict)
    assert "trailing" in res
    assert res["trailing"]["trailing_stop"] is False
    assert "roi" in res
    assert res["roi"]["0"] == 0.04
    assert "stoploss" in res
    assert res["stoploss"]["stoploss"] == -0.1
    assert "max_open_trades" in res
    assert res["max_open_trades"]["max_open_trades"] == 1


def test_start_calls_optimizer(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: MagicMock = mocker.patch("freqtrade.optimize.hyperopt.hyperopt_optimizer.dump")
    dumper2: MagicMock = mocker.patch("freqtrade.optimize.hyperopt.Hyperopt._save_result")
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change", return_value=1.5
    )
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.file_dump_json")

    mocker.patch(
        "freqtrade.optimize.backtesting.Backtesting.load_bt_data",
        MagicMock(return_value=(MagicMock(), None)),
    )
    mocker.patch(
        "freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange",
        MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))),
    )
    # Dummy-reduce points to ensure scikit-learn is forced to generate new values
    mocker.patch("freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS", 2)

    parallel: MagicMock = mocker.patch(
        "freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel",
        MagicMock(
            return_value=[
                {
                    "loss": 1,
                    "results_explanation": "foo result",
                    "params": {"buy": {}, "sell": {}, "roi": {}, "stoploss": 0.0},
                    "results_metrics": generate_result_metrics(),
                }
            ]
        ),
    )
    patch_exchange(mocker)
    # Co-test loading timeframe from strategy
    del hyperopt_conf["timeframe"]

    hyperopt: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})

    hyperopt.start()

    parallel.assert_called_once()

    out, _err = capsys.readouterr()
    assert "Best result:\n\n*    1/1: foo result Objective: 1.00000\n" in out
    # Should be called for historical candle data
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_exit")
    assert hasattr(hyperopt.hyperopter.backtesting.strategy, "advise_entry")
    assert (
        hyperopt.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf["max_open_trades"]
    )
    assert hasattr(hyperopt.hyperopter.backtesting, "_position_stacking")


def test_hyperopt_format_results(hyperopt: Hyperopt) -> None:
    bt_result: Dict[str, Any] = {
        "results": pd.DataFrame(
            {
                "pair": ["UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC", "UNITTEST/BTC"],
                "profit_ratio": [0.003312, 0.010801, 0.013803, 0.002780],
                "profit_abs": [0.000003, 0.000011, 0.000014, 0.000003],
                "open_date": [
                    dt_utc(2017, 11, 14, 19, 32, 00),
                    dt_utc(2017, 11, 14, 21, 36, 00),
                    dt_utc(2017, 11, 14, 22, 12, 00),
                    dt_utc(2017, 11, 14, 22, 44, 00),
                ],
                "close_date": [
                    dt_utc(2017, 11, 14, 21, 35, 00),
                    dt_utc(2017, 11, 14, 22, 10, 00),
                    dt_utc(2017, 11, 14, 22, 43, 00),
                    dt_utc(2017, 11, 14, 22, 58, 00),
                ],
                "open_rate": [0.002543, 0.003003, 0.003089, 0.003214],
                "close_rate": [0.002546, 0.003014, 0.003103, 0.003217],
                "trade_duration": [123, 34, 31, 14],
                "is_open": [False, False, False, True],
                "is_short": [False, False, False, False],
                "stake_amount": [0.01, 0.01, 