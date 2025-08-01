from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List
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
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, get_args, get_markets, log_has, log_has_re, patch_exchange, patched_configuration_load_config_file


def generate_result_metrics() -> Dict[str, Any]:
    return {
        'trade_count': 1,
        'total_trades': 1,
        'avg_profit': 0.1,
        'total_profit': 0.001,
        'profit': 0.01,
        'duration': 20.0,
        'wins': 1,
        'draws': 0,
        'losses': 0,
        'profit_mean': 0.01,
        'profit_total_abs': 0.001,
        'profit_total': 0.01,
        'holding_avg': timedelta(minutes=20),
        'max_drawdown_account': 0.001,
        'max_drawdown_abs': 0.001,
        'loss': 0.001,
        'is_initial_point': 0.001,
        'is_random': False,
        'is_best': 1
    }


def test_setup_hyperopt_configuration_without_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy']
    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' not in config
    assert 'runmode' in config
    assert config['runmode'] == RunMode.HYPEROPT


def test_setup_hyperopt_configuration_with_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy', '--datadir', '/foo/bar', '--timeframe', '1m', '--timerange', ':100', '--enable-position-stacking', '--epochs', '1000', '--spaces', 'default', '--print-all']
    config: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.HYPEROPT
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog)
    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)
    assert 'epochs' in config
    assert log_has('Parameter --epochs detected ... Will run Hyperopt with for 1000 epochs ...', caplog)
    assert 'spaces' in config
    assert log_has('Parameter -s/--spaces detected: {}'.format(config['spaces']), caplog)
    assert 'print_all' in config
    assert log_has('Parameter --print-all detected ...', caplog)


def test_setup_hyperopt_configuration_stake_amount(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy', '--stake-amount', '1', '--starting-balance', '2']
    conf: Dict[str, Any] = setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)
    assert isinstance(conf, dict)
    args = ['hyperopt', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '0.5']
    with pytest.raises(OperationalException, match='Starting balance .* smaller .*'):
        setup_optimize_configuration(get_args(args), RunMode.HYPEROPT)


def test_start_not_installed(mocker: Any, default_conf: Dict[str, Any], import_fails: Any) -> None:
    start_mock: Any = MagicMock()
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy', '--epochs', '5', '--hyperopt-loss', 'SharpeHyperOptLossDaily']
    pargs: Any = get_args(args)
    with pytest.raises(OperationalException, match='Please ensure that the hyperopt dependencies'):
        start_hyperopt(pargs)


def test_start_no_hyperopt_allowed(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    start_mock: Any = MagicMock()
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.start', start_mock)
    patch_exchange(mocker)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--hyperopt', 'HyperoptTestSepFile', '--hyperopt-loss', 'SharpeHyperOptLossDaily', '--epochs', '5']
    pargs: Any = get_args(args)
    with pytest.raises(OperationalException, match='Using separate Hyperopt files has been.*'):
        start_hyperopt(pargs)


def test_start_no_data(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path) -> None:
    hyperopt_conf['user_data_dir'] = tmp_path
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.data.history.load_pair_history', MagicMock(return_value=pd.DataFrame))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    patch_exchange(mocker)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy', '--hyperopt-loss', 'SharpeHyperOptLossDaily', '--epochs', '5']
    pargs: Any = get_args(args)
    with pytest.raises(OperationalException, match='No data found. Terminating.'):
        start_hyperopt(pargs)
    try:
        Path(Hyperopt.get_lock_filename(hyperopt_conf)).unlink()
    except Exception:
        pass


def test_start_filelock(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    hyperopt_mock: Any = MagicMock(side_effect=Timeout(Hyperopt.get_lock_filename(hyperopt_conf)))
    patched_configuration_load_config_file(mocker, hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.__init__', hyperopt_mock)
    patch_exchange(mocker)
    args: List[str] = ['hyperopt', '--config', 'config.json', '--strategy', 'HyperoptableStrategy', '--hyperopt-loss', 'SharpeHyperOptLossDaily', '--epochs', '5']
    pargs: Any = get_args(args)
    start_hyperopt(pargs)
    assert log_has('Another running instance of freqtrade Hyperopt detected.', caplog)


def test_log_results_if_loss_improves(hyperopt: Hyperopt, capsys: Any) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.total_epochs = 2
    hyperopt.print_results({
        'loss': 1,
        'results_metrics': generate_result_metrics(),
        'total_profit': 0,
        'current_epoch': 2,
        'is_initial_point': False,
        'is_random': False,
        'is_best': True
    })
    hyperopt._hyper_out.print()
    out, _err = capsys.readouterr()
    assert all((x in out for x in ['Best', '2/2', '1', '0.10%', '0.00100000 BTC    (1.00%)', '0:20:00']))


def test_no_log_if_loss_does_not_improve(hyperopt: Hyperopt, caplog: Any) -> None:
    hyperopt.current_best_loss = 2
    hyperopt.print_results({'is_best': False, 'loss': 3, 'current_epoch': 1})
    assert caplog.record_tuples == []


def test_roi_table_generation(hyperopt: Hyperopt) -> None:
    params: Dict[str, Any] = {
        'roi_t1': 5,
        'roi_t2': 10,
        'roi_t3': 15,
        'roi_p1': 1,
        'roi_p2': 2,
        'roi_p3': 3
    }
    assert hyperopt.hyperopter.custom_hyperopt.generate_roi_table(params) == {0: 6, 15: 3, 25: 1, 30: 0}


def test_params_no_optimize_details(hyperopt: Hyperopt) -> None:
    hyperopt.hyperopter.config['spaces'] = ['buy']
    res: Dict[str, Any] = hyperopt.hyperopter._get_no_optimize_details()
    assert isinstance(res, dict)
    assert 'trailing' in res
    assert res['trailing']['trailing_stop'] is False
    assert 'roi' in res
    assert res['roi']['0'] == 0.04
    assert 'stoploss' in res
    assert res['stoploss']['stoploss'] == -0.1
    assert 'max_open_trades' in res
    assert res['max_open_trades']['max_open_trades'] == 1


def test_start_calls_optimizer(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS', 2)
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {'buy': {}, 'sell': {}, 'roi': {}, 'stoploss': 0.0},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    del hyperopt_conf['timeframe']
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_exit')
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_entry')
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt_inst.hyperopter.backtesting, '_position_stacking')


def test_hyperopt_format_results(hyperopt: Hyperopt) -> None:
    bt_result: Dict[str, Any] = {
        'results': pd.DataFrame({
            'pair': ['UNITTEST/BTC', 'UNITTEST/BTC', 'UNITTEST/BTC', 'UNITTEST/BTC'],
            'profit_ratio': [0.003312, 0.010801, 0.013803, 0.00278],
            'profit_abs': [3e-06, 1.1e-05, 1.4e-05, 3e-06],
            'open_date': [dt_utc(2017, 11, 14, 19, 32, 0), dt_utc(2017, 11, 14, 21, 36, 0), dt_utc(2017, 11, 14, 22, 12, 0), dt_utc(2017, 11, 14, 22, 44, 0)],
            'close_date': [dt_utc(2017, 11, 14, 21, 35, 0), dt_utc(2017, 11, 14, 22, 10, 0), dt_utc(2017, 11, 14, 22, 43, 0), dt_utc(2017, 11, 14, 22, 58, 0)],
            'open_rate': [0.002543, 0.003003, 0.003089, 0.003214],
            'close_rate': [0.002546, 0.003014, 0.003103, 0.003217],
            'trade_duration': [123, 34, 31, 14],
            'is_open': [False, False, False, True],
            'is_short': [False, False, False, False],
            'stake_amount': [0.01, 0.01, 0.01, 0.01],
            'exit_reason': [ExitType.ROI.value, ExitType.STOP_LOSS.value, ExitType.ROI.value, ExitType.FORCE_EXIT.value]
        }),
        'config': hyperopt.config,
        'locks': [],
        'final_balance': 0.02,
        'rejected_signals': 2,
        'timedout_entry_orders': 0,
        'timedout_exit_orders': 0,
        'canceled_trade_entries': 0,
        'canceled_entry_orders': 0,
        'replaced_entry_orders': 0,
        'backtest_start_time': 1619718665,
        'backtest_end_time': 1619718665
    }
    results_metrics: Dict[str, Any] = generate_strategy_stats(
        ['XRP/BTC'],
        '',
        bt_result,
        dt_utc(2017, 11, 14, 19, 32, 0),
        dt_utc(2017, 12, 14, 19, 32, 0),
        market_change=0
    )
    results_explanation: str = HyperoptTools.format_results_explanation_string(results_metrics, 'BTC')
    total_profit: Any = results_metrics['profit_total_abs']
    results: Dict[str, Any] = {
        'loss': 0.0,
        'params_dict': None,
        'params_details': None,
        'results_metrics': results_metrics,
        'results_explanation': results_explanation,
        'total_profit': total_profit,
        'current_epoch': 1,
        'is_initial_point': True
    }
    result: str = HyperoptTools._format_explanation_string(results, 1)
    assert ' 0.71%' in result
    assert 'Total profit  0.00003100 BTC' in result
    assert '0:50:00 min' in result


def test_populate_indicators(hyperopt: Hyperopt, testdatadir: Any) -> None:
    data: Any = load_data(testdatadir, '1m', ['UNITTEST/BTC'], fill_up_missing=True)
    dataframes: Any = hyperopt.hyperopter.backtesting.strategy.advise_all_indicators(data)
    dataframe: Any = dataframes['UNITTEST/BTC']
    assert 'adx' in dataframe
    assert 'macd' in dataframe
    assert 'rsi' in dataframe


def test_generate_optimizer(mocker: Any, hyperopt_conf: Dict[str, Any]) -> None:
    hyperopt_conf.update({'spaces': 'all', 'hyperopt_min_trades': 1})
    backtest_result: Dict[str, Any] = {
        'results': pd.DataFrame({
            'pair': ['UNITTEST/BTC', 'UNITTEST/BTC', 'UNITTEST/BTC', 'UNITTEST/BTC'],
            'profit_ratio': [0.003312, 0.010801, 0.013803, 0.00278],
            'profit_abs': [3e-06, 1.1e-05, 1.4e-05, 3e-06],
            'open_date': [dt_utc(2017, 11, 14, 19, 32, 0), dt_utc(2017, 11, 14, 21, 36, 0), dt_utc(2017, 11, 14, 22, 12, 0), dt_utc(2017, 11, 14, 22, 44, 0)],
            'close_date': [dt_utc(2017, 11, 14, 21, 35, 0), dt_utc(2017, 11, 14, 22, 10, 0), dt_utc(2017, 11, 14, 22, 43, 0), dt_utc(2017, 11, 14, 22, 58, 0)],
            'open_rate': [0.002543, 0.003003, 0.003089, 0.003214],
            'close_rate': [0.002546, 0.003014, 0.003103, 0.003217],
            'trade_duration': [123, 34, 31, 14],
            'is_open': [False, False, False, True],
            'is_short': [False, False, False, False],
            'stake_amount': [0.01, 0.01, 0.01, 0.01],
            'exit_reason': [ExitType.ROI.value, ExitType.STOP_LOSS.value, ExitType.ROI.value, ExitType.FORCE_EXIT.value]
        }),
        'config': hyperopt_conf,
        'locks': [],
        'rejected_signals': 20,
        'timedout_entry_orders': 0,
        'timedout_exit_orders': 0,
        'canceled_trade_entries': 0,
        'canceled_entry_orders': 0,
        'replaced_entry_orders': 0,
        'final_balance': 1000
    }
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.Backtesting.backtest', return_value=backtest_result)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', return_value=(dt_utc(2017, 12, 10), dt_utc(2017, 12, 13)))
    patch_exchange(mocker)
    mocker.patch.object(Path, 'open')
    mocker.patch('freqtrade.configuration.config_validation.validate_config_schema')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.load', return_value={'XRP/BTC': None})
    optimizer_param: Dict[str, Any] = {
        'buy_plusdi': 0.02,
        'buy_rsi': 35,
        'sell_minusdi': 0.02,
        'sell_rsi': 75,
        'protection_cooldown_lookback': 20,
        'protection_enabled': True,
        'roi_t1': 60.0,
        'roi_t2': 30.0,
        'roi_t3': 20.0,
        'roi_p1': 0.01,
        'roi_p2': 0.01,
        'roi_p3': 0.1,
        'stoploss': -0.4,
        'trailing_stop': True,
        'trailing_stop_positive': 0.02,
        'trailing_stop_positive_offset_p1': 0.05,
        'trailing_only_offset_is_reached': False,
        'max_open_trades': 3
    }
    response_expected: Dict[str, Any] = {
        'loss': 1.9147239021396234,
        'results_explanation': '     4 trades. 4/0/0 Wins/Draws/Losses. Avg profit   0.77%. Median profit   0.71%. Total profit  0.00003100 BTC (   0.00%). Avg duration 0:50:00 min.',
        'params_details': {
            'buy': {'buy_plusdi': 0.02, 'buy_rsi': 35},
            'roi': {'0': 0.12000000000000001, '20.0': 0.02, '50.0': 0.01, '110.0': 0},
            'protection': {'protection_cooldown_lookback': 20, 'protection_enabled': True},
            'sell': {'sell_minusdi': 0.02, 'sell_rsi': 75},
            'stoploss': {'stoploss': -0.4},
            'trailing': {'trailing_only_offset_is_reached': False, 'trailing_stop': True, 'trailing_stop_positive': 0.02, 'trailing_stop_positive_offset': 0.07},
            'max_open_trades': {'max_open_trades': 3}
        },
        'params_dict': optimizer_param,
        'params_not_optimized': {'buy': {}, 'protection': {}, 'sell': {}},
        'results_metrics': ANY,
        'total_profit': 3.1e-08
    }
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.min_date = dt_utc(2017, 12, 10)
    hyperopt_inst.hyperopter.max_date = dt_utc(2017, 12, 13)
    hyperopt_inst.hyperopter.init_spaces()
    generate_optimizer_value: Dict[str, Any] = hyperopt_inst.hyperopter.generate_optimizer(list(optimizer_param.values()))
    assert generate_optimizer_value == response_expected


def test_clean_hyperopt(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file', MagicMock(return_value={}))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.Path.is_file', MagicMock(return_value=True))
    unlinkmock: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt.Path.unlink', MagicMock())
    h: Hyperopt = Hyperopt(hyperopt_conf)
    assert unlinkmock.call_count == 2
    assert log_has(f'Removing `{h.data_pickle_file}`.', caplog)


def test_print_json_spaces_all(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {},
        'params_details': {'buy': {'mfi-value': None}, 'sell': {'sell-mfi-value': None}, 'roi': {}, 'stoploss': {'stoploss': None}, 'trailing': {'trailing_stop': None}, 'max_open_trades': {'max_open_trades': None}},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'all', 'hyperopt_jobs': 1, 'print_json': True})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    result_str: str = '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi":{},"stoploss":null,"trailing_stop":null,"max_open_trades":null}'
    assert result_str in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_print_json_spaces_default(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {},
        'params_details': {'buy': {'mfi-value': None}, 'sell': {'sell-mfi-value': None}, 'roi': {}, 'stoploss': {'stoploss': None}},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'print_json': True})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert '{"params":{"mfi-value":null,"sell-mfi-value":null},"minimal_roi":{},"stoploss":null}' in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_print_json_spaces_roi_stoploss(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {},
        'params_details': {'roi': {}, 'stoploss': {'stoploss': None}},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'roi stoploss', 'hyperopt_jobs': 1, 'print_json': True})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert '{"minimal_roi":{},"stoploss":null}' in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1


def test_simplified_interface_roi_stoploss(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {'stoploss': 0.0},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'roi stoploss'})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_exit')
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_entry')
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt_inst.hyperopter.backtesting, '_position_stacking')


def test_simplified_interface_all_failed(mocker: Any, hyperopt_conf: Dict[str, Any], caplog: Any) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_auto.HyperOptAuto._generate_indicator_space', return_value=[])
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'all'})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    with pytest.raises(OperationalException, match=f"The 'protection' space is included into *"):
        hyperopt_inst.start()
    hyperopt_inst.config['hyperopt_ignore_missing_space'] = True
    caplog.clear()
    hyperopt_inst.hyperopter.init_spaces()
    assert log_has_re("The 'protection' space is included into *", caplog)
    assert hyperopt_inst.hyperopter.protection_space == []


def test_simplified_interface_buy(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'buy'})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_exit')
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_entry')
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt_inst.hyperopter.backtesting, '_position_stacking')


def test_simplified_interface_sell(mocker: Any, hyperopt_conf: Dict[str, Any], capsys: Any) -> None:
    dumper: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump')
    dumper2: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt._save_result')
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.calculate_market_change', return_value=1.5)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    parallel: Any = mocker.patch('freqtrade.optimize.hyperopt.Hyperopt.run_optimizer_parallel', MagicMock(return_value=[{
        'loss': 1,
        'results_explanation': 'foo result',
        'params': {},
        'results_metrics': generate_result_metrics()
    }]))
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': 'sell'})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    hyperopt_inst.start()
    parallel.assert_called_once()
    out, _err = capsys.readouterr()
    assert 'Best result:\n\n*    1/1: foo result Objective: 1.00000\n' in out
    assert dumper.called
    assert dumper.call_count == 1
    assert dumper2.call_count == 1
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_exit')
    assert hasattr(hyperopt_inst.hyperopter.backtesting.strategy, 'advise_entry')
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == hyperopt_conf['max_open_trades']
    assert hasattr(hyperopt_inst.hyperopter.backtesting, '_position_stacking')


@pytest.mark.parametrize('space', ['buy', 'sell', 'protection'])
def test_simplified_interface_failed(mocker: Any, hyperopt_conf: Dict[str, Any], space: str) -> None:
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.dump', MagicMock())
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.file_dump_json')
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.load_bt_data', MagicMock(return_value=(MagicMock(), None)))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.get_timerange', MagicMock(return_value=(datetime(2017, 12, 10), datetime(2017, 12, 13))))
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_auto.HyperOptAuto._generate_indicator_space', return_value=[])
    patch_exchange(mocker)
    hyperopt_conf.update({'spaces': space})
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    hyperopt_inst.hyperopter.backtesting.strategy.advise_all_indicators = MagicMock()
    hyperopt_inst.hyperopter.custom_hyperopt.generate_roi_table = MagicMock(return_value={})
    with pytest.raises(OperationalException, match=f"The '{space}' space is included into *"):
        hyperopt_inst.start()


def test_in_strategy_auto_hyperopt(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any) -> None:
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS', 2)
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['all']
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    opt: Any = hyperopt_inst.hyperopter
    opt.backtesting.exchange.get_max_leverage = MagicMock(return_value=1.0)
    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_started is True
    assert opt.backtesting.strategy.bot_loop_started is False
    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    assert opt.backtesting.strategy.max_open_trades == 1
    buy_rsi_range: range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    assert len(list(buy_rsi_range)) == 51
    hyperopt_inst.start()
    assert opt.backtesting.strategy.protection_cooldown_lookback.value != 30
    assert opt.backtesting.strategy.buy_rsi.value != 35
    assert opt.backtesting.strategy.sell_rsi.value != 74
    assert opt.backtesting.strategy.max_open_trades != 1
    opt.custom_hyperopt.generate_estimator = lambda *args, **kwargs: 'ET1'
    with pytest.raises(OperationalException, match='Estimator ET1 not supported.'):
        opt.get_optimizer(2, 42, 2, 2)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_in_strategy_auto_hyperopt_with_parallel(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any) -> None:
    mocker.patch(f'{EXMS}.validate_config', MagicMock())
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.reload_markets')
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=get_markets()))
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt.INITIAL_POINTS', 2)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['all'],
        'epochs': 2,
        'hyperopt_jobs': 2,
        'fee': fee.return_value
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    opt: Any = hyperopt_inst.hyperopter
    opt.backtesting.exchange.get_max_leverage = lambda *x, **xx: 1.0
    opt.backtesting.exchange.get_min_pair_stake_amount = lambda *x, **xx: 1e-05
    opt.backtesting.exchange.get_max_pair_stake_amount = lambda *x, **xx: 100.0
    opt.backtesting.exchange._markets = get_markets()
    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_started is True
    assert opt.backtesting.strategy.bot_loop_started is False
    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    buy_rsi_range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    assert len(list(buy_rsi_range)) == 51
    hyperopt_inst.start()


def test_in_strategy_auto_hyperopt_per_epoch(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any) -> None:
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_fee', fee)
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['all'],
        'epochs': 3,
        'analyze_per_epoch': True
    })
    go: Any = mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperOptimizer.generate_optimizer', return_value={
        'loss': 0.05,
        'results_explanation': 'foo result',
        'params': {},
        'results_metrics': generate_result_metrics()
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    opt: Any = hyperopt_inst.hyperopter
    opt.backtesting.exchange.get_max_leverage = MagicMock(return_value=1.0)
    assert isinstance(opt.custom_hyperopt, HyperOptAuto)
    assert isinstance(opt.backtesting.strategy.buy_rsi, IntParameter)
    assert opt.backtesting.strategy.bot_loop_started is False
    assert opt.backtesting.strategy.bot_started is True
    assert opt.backtesting.strategy.buy_rsi.in_space is True
    assert opt.backtesting.strategy.buy_rsi.value == 35
    assert opt.backtesting.strategy.sell_rsi.value == 74
    assert opt.backtesting.strategy.protection_cooldown_lookback.value == 30
    buy_rsi_range = opt.backtesting.strategy.buy_rsi.range
    assert isinstance(buy_rsi_range, range)
    assert len(list(buy_rsi_range)) == 51
    hyperopt_inst.start()
    assert go.call_count == 3


def test_SKDecimal() -> None:
    space: SKDecimal = SKDecimal(1, 2, decimals=2)
    assert 1.5 in space
    assert 2.5 not in space
    assert space.low == 100
    assert space.high == 200
    assert space.inverse_transform([200]) == [2.0]
    assert space.inverse_transform([100]) == [1.0]
    assert space.inverse_transform([150, 160]) == [1.5, 1.6]
    assert space.transform([1.5]) == [150]
    assert space.transform([2.0]) == [200]
    assert space.transform([1.0]) == [100]
    assert space.transform([1.5, 1.6]) == [150, 160]


def test_stake_amount_unlimited_max_open_trades(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any) -> None:
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_fee', fee)
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['trades'],
        'stake_amount': 'unlimited'
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperOptimizer._get_params_dict', return_value={'max_open_trades': -1})
    assert isinstance(hyperopt_inst.hyperopter.custom_hyperopt, HyperOptAuto)
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == 1
    hyperopt_inst.start()
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == 1


def test_max_open_trades_dump(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any, capsys: Any) -> None:
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_fee', fee)
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['trades']
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperOptimizer._get_params_dict', return_value={'max_open_trades': -1})
    assert isinstance(hyperopt_inst.hyperopter.custom_hyperopt, HyperOptAuto)
    hyperopt_inst.start()
    out, _err = capsys.readouterr()
    assert 'max_open_trades = -1' in out
    assert 'max_open_trades = inf' not in out
    hyperopt_conf.update({'print_json': True})
    hyperopt_inst = Hyperopt(hyperopt_conf)
    mocker.patch('freqtrade.optimize.hyperopt.hyperopt_optimizer.HyperOptimizer._get_params_dict', return_value={'max_open_trades': -1})
    assert isinstance(hyperopt_inst.hyperopter.custom_hyperopt, HyperOptAuto)
    hyperopt_inst.start()
    out, _err = capsys.readouterr()
    assert '"max_open_trades":-1' in out


def test_max_open_trades_consistency(mocker: Any, hyperopt_conf: Dict[str, Any], tmp_path: Path, fee: Any) -> None:
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_fee', return_value=0)
    (tmp_path / 'hyperopt_results').mkdir(parents=True)
    hyperopt_conf.update({
        'strategy': 'HyperoptableStrategy',
        'user_data_dir': tmp_path,
        'hyperopt_random_state': 42,
        'spaces': ['trades'],
        'stake_amount': 'unlimited',
        'dry_run_wallet': 8,
        'available_capital': 8,
        'dry_run': True,
        'epochs': 1
    })
    hyperopt_inst: Hyperopt = Hyperopt(hyperopt_conf)
    assert isinstance(hyperopt_inst.hyperopter.custom_hyperopt, HyperOptAuto)
    hyperopt_inst.hyperopter.custom_hyperopt.max_open_trades_space = lambda: [Integer(1, 10, name='max_open_trades')]
    first_time_evaluated: bool = False

    def stake_amount_interceptor(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal first_time_evaluated
            stake_amount = func(*args, **kwargs)
            if first_time_evaluated is False:
                assert stake_amount == 1
                first_time_evaluated = True
            return stake_amount
        return wrapper

    hyperopt_inst.hyperopter.backtesting.wallets._calculate_unlimited_stake_amount = stake_amount_interceptor(
        hyperopt_inst.hyperopter.backtesting.wallets._calculate_unlimited_stake_amount
    )
    hyperopt_inst.start()
    assert hyperopt_inst.hyperopter.backtesting.strategy.max_open_trades == 8
    assert hyperopt_inst.config['max_open_trades'] == 8