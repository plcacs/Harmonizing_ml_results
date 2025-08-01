#!/usr/bin/env python3
import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unittest.mock import MagicMock

import pytest
from jsonschema import ValidationError
from freqtrade.commands import Arguments
from freqtrade.configuration import Configuration, validate_config_consistency
from freqtrade.configuration.config_secrets import sanitize_config
from freqtrade.configuration.config_validation import validate_config_schema
from freqtrade.configuration.deprecated_settings import (
    check_conflicting_settings,
    process_deprecated_setting,
    process_removed_setting,
    process_temporary_deprecated_settings,
)
from freqtrade.configuration.environment_vars import _flat_vars_to_nested_dict
from freqtrade.configuration.load_config import (
    load_config_file,
    load_file,
    load_from_files,
    log_config_error_range,
)
from freqtrade.constants import DEFAULT_DB_DRYRUN_URL, DEFAULT_DB_PROD_URL, ENV_VAR_PREFIX
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException
from tests.conftest import CURRENT_TEST_STRATEGY, log_has, log_has_re, patched_configuration_load_config_file

@pytest.fixture(scope='function')
def all_conf() -> Dict[str, Any]:
    config_file: Path = Path(__file__).parents[1] / 'config_examples/config_full.example.json'
    conf: Dict[str, Any] = load_config_file(str(config_file))
    return conf

def test_load_config_missing_attributes(default_conf: Dict[str, Any]) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf.pop('exchange')
    with pytest.raises(ValidationError, match=".*'exchange' is a required property.*"):
        validate_config_schema(conf)
    conf = deepcopy(default_conf)
    conf.pop('stake_currency')
    conf['runmode'] = RunMode.DRY_RUN
    with pytest.raises(ValidationError, match=".*'stake_currency' is a required property.*"):
        validate_config_schema(conf)

def test_load_config_incorrect_stake_amount(default_conf: Dict[str, Any]) -> None:
    default_conf['stake_amount'] = 'fake'
    with pytest.raises(ValidationError, match=".*'fake' does not match 'unlimited'.*"):
        validate_config_schema(default_conf)

def test_load_config_file(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    del default_conf['user_data_dir']
    default_conf['datadir'] = str(default_conf['datadir'])
    file_mock = mocker.patch(
        'freqtrade.configuration.load_config.Path.open',
        mocker.mock_open(read_data=json.dumps(default_conf))
    )
    validated_conf: Dict[str, Any] = load_config_file('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()

def test_load_config_file_error(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    del default_conf['user_data_dir']
    default_conf['datadir'] = str(default_conf['datadir'])
    filedata: str = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch('freqtrade.configuration.load_config.Path.open', mocker.mock_open(read_data=filedata))
    mocker.patch.object(Path, 'read_text', MagicMock(return_value=filedata))
    with pytest.raises(OperationalException, match='.*Please verify the following segment.*'):
        load_config_file('somefile')

def test_load_config_file_error_range(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    del default_conf['user_data_dir']
    default_conf['datadir'] = str(default_conf['datadir'])
    filedata: str = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch.object(Path, 'read_text', MagicMock(return_value=filedata))
    x: str = log_config_error_range('somefile', 'Parse error at offset 64: Invalid value.')
    assert isinstance(x, str)
    assert x == '{"max_open_trades": 1, "stake_currency": "BTC", "stake_amount": .001, "fiat_display_currency": "USD", "timeframe": "5m", "dry_run": true, "cance'
    filedata = json.dumps(default_conf, indent=2).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch.object(Path, 'read_text', MagicMock(return_value=filedata))
    x = log_config_error_range('somefile', 'Parse error at offset 4: Invalid value.')
    assert isinstance(x, str)
    assert x == '  "max_open_trades": 1,\n  "stake_currency": "BTC",\n  "stake_amount": .001,'
    x = log_config_error_range('-', '')
    assert x == ''

def test_load_file_error(tmp_path: Path) -> None:
    testpath: Path = tmp_path / 'config.json'
    with pytest.raises(OperationalException, match='File .* not found!'):
        load_file(testpath)

def test__args_to_config(caplog: Any) -> None:
    arg_list: List[str] = ['trade', '--strategy-path', 'TestTest']
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        configuration._args_to_config(config, argname='strategy_path', logstring='DeadBeef')
        assert len(w) == 0
        assert log_has('DeadBeef', caplog)
        assert config['strategy_path'] == 'TestTest'
    configuration = Configuration(args)
    config = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        configuration._args_to_config(
            config, argname='strategy_path', logstring='DeadBeef', deprecated_msg='Going away soon!'
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert 'DEPRECATED: Going away soon!' in str(w[-1].message)
        assert log_has('DeadBeef', caplog)
        assert config['strategy_path'] == 'TestTest'

def test_load_config_max_open_trades_zero(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf['max_open_trades'] = 0
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf['max_open_trades'] == 0
    assert 'internals' in validated_conf

def test_load_config_combine_dicts(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    conf1: Dict[str, Any] = deepcopy(default_conf)
    conf2: Dict[str, Any] = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']
    config_files: List[Dict[str, Any]] = [conf1, conf2]
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.load_config.load_config_file', configsmock)
    arg_list: List[str] = ['trade', '-c', 'test_conf.json', '--config', 'test2_conf.json']
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    exchange_conf: Dict[str, Any] = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']
    assert 'internals' in validated_conf

def test_from_config(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    conf1: Dict[str, Any] = deepcopy(default_conf)
    conf2: Dict[str, Any] = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']
    conf2['fiat_display_currency'] = 'EUR'
    config_files: List[Dict[str, Any]] = [conf1, conf2]
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.load_config.load_config_file', configsmock)
    validated_conf: Dict[str, Any] = Configuration.from_files(['test_conf.json', 'test2_conf.json'])
    exchange_conf: Dict[str, Any] = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']
    assert validated_conf['fiat_display_currency'] == 'EUR'
    assert 'internals' in validated_conf
    assert isinstance(validated_conf['user_data_dir'], Path)

def test_from_recursive_files(testdatadir: Path) -> None:
    files: Path = testdatadir / 'testconfigs/testconfig.json'
    conf: Dict[str, Any] = Configuration.from_files([files])
    assert conf
    assert conf['exchange']
    assert conf['entry_pricing']
    assert conf['entry_pricing']['price_side'] == 'same'
    assert conf['exit_pricing']
    assert conf['exit_pricing']['price_side'] == 'same'
    assert len(conf['config_files']) == 4
    assert 'testconfig.json' in conf['config_files'][0]
    assert 'test_pricing_conf.json' in conf['config_files'][1]
    assert 'test_base_config.json' in conf['config_files'][2]
    assert 'test_pricing2_conf.json' in conf['config_files'][3]
    files = testdatadir / 'testconfigs/recursive.json'
    with pytest.raises(OperationalException, match='Config loop detected.'):
        load_from_files([files])

def test_print_config(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    conf1: Dict[str, Any] = deepcopy(default_conf)
    del conf1['user_data_dir']
    conf1['datadir'] = str(conf1['datadir'])
    config_files: List[Dict[str, Any]] = [conf1]
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    mocker.patch('freqtrade.configuration.configuration.load_from_files', configsmock)
    validated_conf: Dict[str, Any] = Configuration.from_files(['test_conf.json'])
    assert isinstance(validated_conf['user_data_dir'], Path)
    assert 'user_data_dir' in validated_conf
    assert 'original_config' in validated_conf
    assert isinstance(json.dumps(validated_conf['original_config']), str)

def test_load_config_max_open_trades_minus_one(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf['max_open_trades'] = -1
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf['max_open_trades'] > 999999999
    assert validated_conf['max_open_trades'] == float('inf')
    assert 'runmode' in validated_conf
    assert validated_conf['runmode'] == RunMode.DRY_RUN

def test_load_config_file_exception(mocker: Any) -> None:
    mocker.patch('freqtrade.configuration.configuration.Path.open', MagicMock(side_effect=FileNotFoundError('File not found')))
    with pytest.raises(OperationalException, match='.*Config file "somefile" not found!*'):
        load_config_file('somefile')

def test_load_config(default_conf: Dict[str, Any], mocker: Any) -> None:
    del default_conf['strategy_path']
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf.get('strategy_path') is None
    assert 'edge' not in validated_conf

def test_load_config_with_params(default_conf: Dict[str, Any], mocker: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path', '--db-url', 'sqlite:///someurl']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf.get('strategy') == 'TestStrategy'
    assert validated_conf.get('strategy_path') == '/some/path'
    assert validated_conf.get('db_url') == 'sqlite:///someurl'
    
    conf: Dict[str, Any] = default_conf.copy()
    conf['dry_run'] = False
    conf['db_url'] = 'sqlite:///path/to/db.sqlite'
    patched_configuration_load_config_file(mocker, conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == 'sqlite:///path/to/db.sqlite'
    
    conf = default_conf.copy()
    conf['dry_run'] = True
    conf['db_url'] = 'sqlite:///path/to/db.sqlite'
    patched_configuration_load_config_file(mocker, conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == 'sqlite:///path/to/db.sqlite'
    
    conf = default_conf.copy()
    conf['dry_run'] = False
    del conf['db_url']
    patched_configuration_load_config_file(mocker, conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == DEFAULT_DB_PROD_URL
    assert 'runmode' in validated_conf
    assert validated_conf['runmode'] == RunMode.LIVE
    
    conf = default_conf.copy()
    conf['dry_run'] = True
    conf['db_url'] = DEFAULT_DB_PROD_URL
    patched_configuration_load_config_file(mocker, conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == DEFAULT_DB_DRYRUN_URL

@pytest.mark.parametrize('config_value,expected,arglist', [
    (True, True, ['trade', '--dry-run']),
    (False, True, ['trade', '--dry-run']),
    (False, False, ['trade']),
    (True, True, ['trade'])
])
def test_load_dry_run(default_conf: Dict[str, Any], mocker: Any, config_value: bool, expected: bool, arglist: List[str]) -> None:
    default_conf['dry_run'] = config_value
    patched_configuration_load_config_file(mocker, default_conf)
    configuration = Configuration(Arguments(arglist).get_parsed_arg())
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf['dry_run'] is expected
    assert validated_conf['runmode'] == (RunMode.DRY_RUN if expected else RunMode.LIVE)

def test_load_custom_strategy(default_conf: Dict[str, Any], mocker: Any, tmp_path: Path) -> None:
    default_conf.update({'strategy': 'CustomStrategy', 'strategy_path': f'{tmp_path}/strategies'})
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf.get('strategy') == 'CustomStrategy'
    assert validated_conf.get('strategy_path') == f'{tmp_path}/strategies'

def test_show_info(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['trade', '--strategy', 'TestStrategy', '--db-url', 'sqlite:///tmp/testdb']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    configuration.get_config()
    assert log_has('Using DB: "sqlite:///tmp/testdb"', caplog)
    assert log_has('Dry run is enabled', caplog)

def test_setup_configuration_without_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert 'user_data_dir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert not log_has('Parameter -i/--timeframe detected ...', caplog)
    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' not in config

def test_setup_configuration_with_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any, tmp_path: Path) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    mocker.patch('freqtrade.configuration.configuration.create_userdata_dir', lambda x, *args, **kwargs: Path(x))
    arglist: List[str] = [
        'backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY,
        '--datadir', '/foo/bar', '--userdir', f'{tmp_path}/freqtrade', '--timeframe', '1m',
        '--enable-position-stacking', '--timerange', ':100', '--export', 'trades',
        '--stake-amount', 'unlimited'
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format('/foo/bar'), caplog)
    assert log_has(f'Using user-data directory: {tmp_path / "freqtrade"} ...', caplog)
    assert 'user_data_dir' in config
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog)
    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)
    assert 'export' in config
    assert log_has('Parameter --export detected: {} ...'.format(config['export']), caplog)
    assert 'stake_amount' in config
    assert config['stake_amount'] == 'unlimited'

def test_setup_configuration_with_stratlist(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    """
    Test setup_configuration() function
    """
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = [
        'backtesting', '--config', 'config.json', '--timeframe', '1m', '--export', 'trades',
        '--strategy-list', CURRENT_TEST_STRATEGY, 'TestStrategy'
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args, RunMode.BACKTEST)
    config: Dict[str, Any] = configuration.get_config()
    assert config['runmode'] == RunMode.BACKTEST
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog)
    assert 'strategy_list' in config
    assert log_has('Using strategy list of 2 strategies', caplog)
    assert 'position_stacking' not in config
    assert 'timerange' not in config
    assert 'export' in config
    assert log_has('Parameter --export detected: {} ...'.format(config['export']), caplog)

def test_hyperopt_with_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['hyperopt', '--epochs', '10', '--spaces', 'all']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args, RunMode.HYPEROPT)
    config: Dict[str, Any] = configuration.get_config()
    assert 'epochs' in config
    assert int(config['epochs']) == 10
    assert log_has('Parameter --epochs detected ... Will run Hyperopt with for 10 epochs ...', caplog)
    assert 'spaces' in config
    assert config['spaces'] == ['all']
    assert log_has("Parameter -s/--spaces detected: ['all']", caplog)
    assert 'runmode' in config
    assert config['runmode'] == RunMode.HYPEROPT

def test_cli_verbose_with_params(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.loggers.set_loggers', MagicMock)
    arglist: List[str] = ['trade', '-vvv']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf.get('verbosity') == 3
    assert log_has('Verbosity set to 3', caplog)

def test_set_logfile(default_conf: Dict[str, Any], mocker: Any, tmp_path: Path) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    f: Path = tmp_path / 'test_file.log'
    assert not f.is_file()
    arglist: List[str] = ['trade', '--logfile', str(f)]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf['logfile'] == str(f)
    assert f.is_file()
    try:
        f.unlink()
    except Exception:
        pass

def test_load_config_warn_forcebuy(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf['force_entry_enable'] = True
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf: Dict[str, Any] = configuration.load_config()
    assert validated_conf.get('force_entry_enable')
    assert log_has('`force_entry_enable` RPC message enabled.', caplog)

def test_validate_default_conf(default_conf: Dict[str, Any]) -> None:
    validate_config_schema(default_conf)

@pytest.mark.parametrize('fiat', ['EUR', 'USD', '', None])
def test_validate_fiat_currency_options(default_conf: Dict[str, Any], fiat: Optional[str]) -> None:
    if fiat is not None:
        default_conf['fiat_display_currency'] = fiat
    else:
        del default_conf['fiat_display_currency']
    validate_config_schema(default_conf)

def test_validate_max_open_trades(default_conf: Dict[str, Any]) -> None:
    default_conf['max_open_trades'] = float('inf')
    default_conf['stake_amount'] = 'unlimited'
    with pytest.raises(OperationalException, match='`max_open_trades` and `stake_amount` cannot both be unlimited.'):
        validate_config_consistency(default_conf)

def test_validate_price_side(default_conf: Dict[str, Any]) -> None:
    default_conf['order_types'] = {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}
    validate_config_consistency(default_conf)
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['order_types']['entry'] = 'market'
    with pytest.raises(OperationalException, match='Market entry orders require entry_pricing.price_side = "other".'):
        validate_config_consistency(conf)
    conf = deepcopy(default_conf)
    conf['order_types']['exit'] = 'market'
    with pytest.raises(OperationalException, match='Market exit orders require exit_pricing.price_side = "other".'):
        validate_config_consistency(conf)
    conf = deepcopy(default_conf)
    conf['order_types']['exit'] = 'market'
    conf['order_types']['entry'] = 'market'
    conf['exit_pricing']['price_side'] = 'bid'
    conf['entry_pricing']['price_side'] = 'ask'
    validate_config_consistency(conf)

def test_validate_tsl(default_conf: Dict[str, Any]) -> None:
    default_conf['stoploss'] = 0.0
    with pytest.raises(OperationalException, match='The config stoploss needs to be different from 0 to avoid problems with sell orders.'):
        validate_config_consistency(default_conf)
    default_conf['stoploss'] = -0.1
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0
    default_conf['trailing_stop_positive_offset'] = 0
    default_conf['trailing_only_offset_is_reached'] = True
    with pytest.raises(OperationalException, match='The config trailing_only_offset_is_reached needs trailing_stop_positive_offset to be more than 0 in your config.'):
        validate_config_consistency(default_conf)
    default_conf['trailing_stop_positive_offset'] = 0.01
    default_conf['trailing_stop_positive'] = 0.015
    with pytest.raises(OperationalException, match='The config trailing_stop_positive_offset needs to be greater than trailing_stop_positive in your config.'):
        validate_config_consistency(default_conf)
    default_conf['trailing_stop_positive'] = 0.01
    default_conf['trailing_stop_positive_offset'] = 0.015
    validate_config_consistency(default_conf)
    default_conf['trailing_stop_positive'] = 0
    default_conf['trailing_stop_positive_offset'] = 0.02
    default_conf['trailing_only_offset_is_reached'] = False
    with pytest.raises(OperationalException, match='The config trailing_stop_positive needs to be different from 0 to avoid problems with sell orders'):
        validate_config_consistency(default_conf)

def test_validate_edge2(edge_conf: Dict[str, Any]) -> None:
    edge_conf.update({'use_exit_signal': True})
    validate_config_consistency(edge_conf)
    edge_conf.update({'use_exit_signal': False})
    with pytest.raises(OperationalException, match='Edge requires `use_exit_signal` to be True, otherwise no sells will happen.'):
        validate_config_consistency(edge_conf)

def test_validate_whitelist(default_conf: Dict[str, Any]) -> None:
    default_conf['runmode'] = RunMode.DRY_RUN
    validate_config_consistency(default_conf)
    conf: Dict[str, Any] = deepcopy(default_conf)
    del conf['exchange']['pair_whitelist']
    with pytest.raises(OperationalException, match='StaticPairList requires pair_whitelist to be set.'):
        validate_config_consistency(conf)
    conf = deepcopy(default_conf)
    conf.update({'pairlists': [{'method': 'VolumePairList'}]})
    validate_config_consistency(conf)
    del conf['exchange']['pair_whitelist']
    validate_config_consistency(conf)

def test_validate_ask_orderbook(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['exit_pricing']['use_order_book'] = True
    conf['exit_pricing']['order_book_min'] = 2
    conf['exit_pricing']['order_book_max'] = 2
    validate_config_consistency(conf)
    assert log_has_re('DEPRECATED: Please use `order_book_top` instead of.*', caplog)
    assert conf['exit_pricing']['order_book_top'] == 2
    conf['exit_pricing']['order_book_max'] = 5
    with pytest.raises(OperationalException, match='Using order_book_max != order_book_min in exit_pricing.*'):
        validate_config_consistency(conf)

def test_validate_time_in_force(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['order_time_in_force'] = {'buy': 'gtc', 'sell': 'GTC'}
    validate_config_consistency(conf)
    assert log_has_re("DEPRECATED: Using 'buy' and 'sell' for time_in_force is.*", caplog)
    assert conf['order_time_in_force']['entry'] == 'gtc'
    assert conf['order_time_in_force']['exit'] == 'GTC'
    conf = deepcopy(default_conf)
    conf['order_time_in_force'] = {'buy': 'GTC', 'sell': 'GTC'}
    conf['trading_mode'] = 'futures'
    with pytest.raises(OperationalException, match="Please migrate your time_in_force settings .* 'entry' and 'exit'\\."):
        validate_config_consistency(conf)

def test__validate_order_types(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['order_types'] = {'buy': 'limit', 'sell': 'market', 'forcesell': 'market', 'forcebuy': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    validate_config_consistency(conf)
    assert log_has_re("DEPRECATED: Using 'buy' and 'sell' for order_types is.*", caplog)
    assert conf['order_types']['entry'] == 'limit'
    assert conf['order_types']['exit'] == 'market'
    assert conf['order_types']['force_entry'] == 'limit'
    assert 'buy' not in conf['order_types']
    assert 'sell' not in conf['order_types']
    assert 'forcebuy' not in conf['order_types']
    assert 'forcesell' not in conf['order_types']
    conf = deepcopy(default_conf)
    conf['order_types'] = {'buy': 'limit', 'sell': 'market', 'forcesell': 'market', 'forcebuy': 'limit', 'stoploss': 'market', 'stoploss_on_exchange': False}
    conf['trading_mode'] = 'futures'
    with pytest.raises(OperationalException, match='Please migrate your order_types settings to use the new wording\\.'):
        validate_config_consistency(conf)

def test__validate_unfilledtimeout(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['unfilledtimeout'] = {'buy': 30, 'sell': 35}
    validate_config_consistency(conf)
    assert log_has_re("DEPRECATED: Using 'buy' and 'sell' for unfilledtimeout is.*", caplog)
    assert conf['unfilledtimeout']['entry'] == 30
    assert conf['unfilledtimeout']['exit'] == 35
    assert 'buy' not in conf['unfilledtimeout']
    assert 'sell' not in conf['unfilledtimeout']
    conf = deepcopy(default_conf)
    conf['unfilledtimeout'] = {'buy': 30, 'sell': 35}
    conf['trading_mode'] = 'futures'
    with pytest.raises(OperationalException, match='Please migrate your unfilledtimeout settings to use the new wording\\.'):
        validate_config_consistency(conf)

def test__validate_pricing_rules(default_conf: Dict[str, Any], caplog: Any) -> None:
    def_conf: Dict[str, Any] = deepcopy(default_conf)
    del def_conf['entry_pricing']
    del def_conf['exit_pricing']
    def_conf['ask_strategy'] = {'price_side': 'ask', 'use_order_book': True, 'bid_last_balance': 0.5}
    def_conf['bid_strategy'] = {'price_side': 'bid', 'use_order_book': False, 'ask_last_balance': 0.7}
    conf: Dict[str, Any] = deepcopy(def_conf)
    validate_config_consistency(conf)
    assert log_has_re("DEPRECATED: Using 'ask_strategy' and 'bid_strategy' is.*", caplog)
    assert conf['exit_pricing']['price_side'] == 'ask'
    assert conf['exit_pricing']['use_order_book'] is True
    assert conf['exit_pricing']['price_last_balance'] == 0.5
    assert conf['entry_pricing']['price_side'] == 'bid'
    assert conf['entry_pricing']['use_order_book'] is False
    assert conf['entry_pricing']['price_last_balance'] == 0.7
    assert 'ask_strategy' not in conf
    assert 'bid_strategy' not in conf
    conf = deepcopy(def_conf)
    conf['trading_mode'] = 'futures'
    with pytest.raises(OperationalException, match='Please migrate your pricing settings to use the new wording\\.'):
        validate_config_consistency(conf)

def test__validate_freqai_include_timeframes(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf.update({
        'freqai': {
            'enabled': True,
            'feature_parameters': {'include_timeframes': ['1m', '5m'], 'include_corr_pairlist': []},
            'data_split_parameters': {},
            'model_training_parameters': {}
        }
    })
    with pytest.raises(OperationalException, match='Main timeframe of .*'):
        validate_config_consistency(conf)
    conf.update({'timeframe': '1m'})
    validate_config_consistency(conf)
    conf['freqai']['feature_parameters']['include_timeframes'] = ['5m', '15m']
    validate_config_consistency(conf)
    assert conf['freqai']['feature_parameters']['include_timeframes'] == ['1m', '5m', '15m']
    conf.update({'analyze_per_epoch': True})
    with pytest.raises(OperationalException, match='Using analyze-per-epoch .* not supported with a FreqAI strategy.'):
        validate_config_consistency(conf)

def test__validate_consumers(default_conf: Dict[str, Any], caplog: Any) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf.update({'external_message_consumer': {'enabled': True, 'producers': []}})
    with pytest.raises(OperationalException, match='You must specify at least 1 Producer to connect to.'):
        validate_config_consistency(conf)
    conf = deepcopy(default_conf)
    conf.update({'external_message_consumer': {'enabled': True, 'producers': [
        {'name': 'default', 'host': '127.0.0.1', 'port': 8081, 'ws_token': 'secret_ws_t0ken.'},
        {'name': 'default', 'host': '127.0.0.1', 'port': 8080, 'ws_token': 'secret_ws_t0ken.'}
    ]}})
    with pytest.raises(OperationalException, match='Producer names must be unique. Duplicate: default'):
        validate_config_consistency(conf)
    conf = deepcopy(default_conf)
    conf.update({'process_only_new_candles': True, 'external_message_consumer': {'enabled': True, 'producers': [
        {'name': 'default', 'host': '127.0.0.1', 'port': 8081, 'ws_token': 'secret_ws_t0ken.'}
    ]}})
    validate_config_consistency(conf)
    assert log_has_re('To receive best performance with external data.*', caplog)

def test__validate_orderflow(default_conf: Dict[str, Any]) -> None:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['exchange']['use_public_trades'] = True
    with pytest.raises(ConfigurationError, match='Orderflow is a required configuration key when using public trades.'):
        validate_config_consistency(conf)
    conf.update({'orderflow': {'scale': 0.5, 'stacked_imbalance_range': 3, 'imbalance_volume': 100, 'imbalance_ratio': 3}})
    validate_config_consistency(conf)

def test_load_config_test_comments() -> None:
    """
    Load config with comments
    """
    config_file: Path = Path(__file__).parents[0] / 'config_test_comments.json'
    conf: Dict[str, Any] = load_config_file(str(config_file))
    assert conf

def test_load_config_default_exchange(all_conf: Dict[str, Any]) -> None:
    """
    config['exchange'] subtree has required options in it
    so it cannot be omitted in the config
    """
    del all_conf['exchange']
    assert 'exchange' not in all_conf
    with pytest.raises(ValidationError, match="'exchange' is a required property"):
        validate_config_schema(all_conf)

def test_load_config_default_exchange_name(all_conf: Dict[str, Any]) -> None:
    """
    config['exchange']['name'] option is required
    so it cannot be omitted in the config
    """
    del all_conf['exchange']['name']
    assert 'name' not in all_conf['exchange']
    with pytest.raises(ValidationError, match="'name' is a required property"):
        validate_config_schema(all_conf)

def test_load_config_stoploss_exchange_limit_ratio(all_conf: Dict[str, Any]) -> None:
    all_conf['order_types']['stoploss_on_exchange_limit_ratio'] = 1.15
    with pytest.raises(ValidationError, match='1.15 is greater than the maximum'):
        validate_config_schema(all_conf)

@pytest.mark.parametrize('keys', [('exchange', 'key', ''), ('exchange', 'secret', ''), ('exchange', 'password', '')])
def test_load_config_default_subkeys(all_conf: Dict[str, Any], keys: Tuple[str, str, str]) -> None:
    """
    Test for parameters with default values in sub-paths
    so they can be omitted in the config and the default value
    should is added to the config.
    """
    key_str: str = keys[0]
    subkey: str = keys[1]
    del all_conf[key_str][subkey]
    assert subkey not in all_conf[key_str]
    validate_config_schema(all_conf)
    assert subkey in all_conf[key_str]
    assert all_conf[key_str][subkey] == keys[2]

def test_pairlist_resolving() -> None:
    arglist: List[str] = ['download-data', '--pairs', 'ETH/BTC', 'XRP/BTC', '--exchange', 'binance']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args, RunMode.OTHER)
    config: Dict[str, Any] = configuration.get_config()
    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['pair_whitelist'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == 'binance'

def test_pairlist_resolving_with_config(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['download-data', '--config', 'config.json']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    assert config['pairs'] == default_conf['exchange']['pair_whitelist']
    assert config['exchange']['name'] == default_conf['exchange']['name']
    arglist = ['download-data', '--config', 'config.json', '--pairs', 'ETH/BTC', 'XRP/BTC']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config = configuration.get_config()
    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == default_conf['exchange']['name']

def test_pairlist_resolving_with_config_pl(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['download-data', '--config', 'config.json', '--pairs-file', 'tests/testdata/pairs.json']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    assert len(config['pairs']) == 23
    assert 'ETH/BTC' in config['pairs']
    assert 'XRP/BTC' in config['pairs']
    assert config['exchange']['name'] == default_conf['exchange']['name']

def test_pairlist_resolving_with_config_pl_not_exists(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist: List[str] = ['download-data', '--config', 'config.json', '--pairs-file', 'tests/testdata/pairs_doesnotexist.json']
    args = Arguments(arglist).get_parsed_arg()
    with pytest.raises(OperationalException, match='No pairs file found with path.*'):
        configuration = Configuration(args)
        configuration.get_config()

def test_pairlist_resolving_fallback(mocker: Any, tmp_path: Path) -> None:
    mocker.patch.object(Path, 'exists', MagicMock(return_value=True))
    mocker.patch.object(Path, 'open', MagicMock(return_value=MagicMock()))
    mocker.patch('freqtrade.configuration.configuration.load_file', MagicMock(return_value=['XRP/BTC', 'ETH/BTC']))
    arglist: List[str] = ['download-data', '--exchange', 'binance']
    args = Arguments(arglist).get_parsed_arg()
    args['config'] = None
    configuration = Configuration(args, RunMode.OTHER)
    config: Dict[str, Any] = configuration.get_config()
    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == 'binance'
    assert config['datadir'] == tmp_path / 'user_data/data/binance'

@pytest.mark.parametrize('setting', [
    ('webhook', 'webhookbuy', 'testWEbhook', 'webhook', 'webhookentry', 'testWEbhook'),
    ('ask_strategy', 'ignore_buying_expired_candle_after', 5, None, 'ignore_buying_expired_candle_after', 6)
])
def test_process_temporary_deprecated_settings(mocker: Any, default_conf: Dict[str, Any], setting: Tuple[Any, Any, Any, Any, Any, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    default_conf[setting[0]] = {}
    default_conf[setting[3]] = {}
    default_conf[setting[0]][setting[1]] = setting[2]
    if setting[3]:
        default_conf[setting[3]][setting[4]] = setting[5]
    else:
        default_conf[setting[4]] = setting[5]
    with pytest.raises(OperationalException, match='DEPRECATED'):
        process_temporary_deprecated_settings(default_conf)
    caplog.clear()
    if setting[3]:
        del default_conf[setting[3]][setting[4]]
    else:
        del default_conf[setting[4]]
    process_temporary_deprecated_settings(default_conf)
    assert log_has_re('DEPRECATED', caplog)
    if setting[3]:
        assert default_conf[setting[3]][setting[4]] == setting[2]
    else:
        assert default_conf[setting[4]] == setting[2]

@pytest.mark.parametrize('setting', [
    ('experimental', 'use_sell_signal', False),
    ('experimental', 'sell_profit_only', True),
    ('experimental', 'ignore_roi_if_buy_signal', True)
])
def test_process_removed_settings(mocker: Any, default_conf: Dict[str, Any], setting: Tuple[str, str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    default_conf[setting[0]] = {}
    default_conf[setting[0]][setting[1]] = setting[2]
    with pytest.raises(OperationalException, match='Setting .* has been moved'):
        process_temporary_deprecated_settings(default_conf)

def test_process_deprecated_setting_edge(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, edge_conf)
    edge_conf.update({'edge': {'enabled': True, 'capital_available_percentage': 0.5}})
    with pytest.raises(OperationalException, match="DEPRECATED.*Using 'edge.capital_available_percentage'*"):
        process_temporary_deprecated_settings(edge_conf)

def test_check_conflicting_settings(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    default_conf['sectionA'] = {}
    default_conf['sectionB'] = {}
    default_conf['sectionA']['new_setting'] = 'valA'
    default_conf['sectionB']['deprecated_setting'] = 'valB'
    with pytest.raises(OperationalException, match='DEPRECATED'):
        check_conflicting_settings(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    caplog.clear()
    del default_conf['sectionA']['new_setting']
    check_conflicting_settings(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    assert not log_has_re('DEPRECATED', caplog)
    assert 'new_setting' not in default_conf['sectionA']
    caplog.clear()
    default_conf['sectionA']['new_setting'] = 'valA'
    del default_conf['sectionB']['deprecated_setting']
    check_conflicting_settings(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    assert not log_has_re('DEPRECATED', caplog)
    assert default_conf['sectionA']['new_setting'] == 'valA'

def test_process_deprecated_setting(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    default_conf['sectionA'] = {}
    default_conf['sectionB'] = {}
    default_conf['sectionB']['deprecated_setting'] = 'valB'
    process_deprecated_setting(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    assert log_has_re('DEPRECATED', caplog)
    assert default_conf['sectionA']['new_setting'] == 'valB'
    assert 'deprecated_setting' not in default_conf['sectionB']
    caplog.clear()
    del default_conf['sectionA']['new_setting']
    default_conf['sectionB']['deprecated_setting'] = 'valB'
    process_deprecated_setting(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    assert log_has_re('DEPRECATED', caplog)
    assert default_conf['sectionA']['new_setting'] == 'valB'
    caplog.clear()
    default_conf['sectionA']['new_setting'] = 'valA'
    default_conf['sectionB'].pop('deprecated_setting', None)
    process_deprecated_setting(default_conf, 'sectionB', 'deprecated_setting', 'sectionA', 'new_setting')
    assert not log_has_re('DEPRECATED', caplog)
    assert default_conf['sectionA']['new_setting'] == 'valA'
    caplog.clear()
    default_conf['sectionB']['deprecated_setting2'] = 'DeadBeef'
    process_deprecated_setting(default_conf, 'sectionB', 'deprecated_setting2', None, 'new_setting')
    assert log_has_re('DEPRECATED', caplog)
    assert default_conf['new_setting']

def test_process_removed_setting(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    default_conf['sectionA'] = {}
    default_conf['sectionB'] = {}
    default_conf['sectionB']['somesetting'] = 'valA'
    process_removed_setting(default_conf, 'sectionA', 'somesetting', 'sectionB', 'somesetting')
    default_conf['sectionA']['somesetting'] = 'valB'
    with pytest.raises(OperationalException, match='Setting .* has been moved'):
        process_removed_setting(default_conf, 'sectionA', 'somesetting', 'sectionB', 'somesetting')

def test_process_deprecated_ticker_interval(default_conf: Dict[str, Any], caplog: Any) -> None:
    message: str = "DEPRECATED: Please use 'timeframe' instead of 'ticker_interval."
    config: Dict[str, Any] = deepcopy(default_conf)
    process_temporary_deprecated_settings(config)
    assert not log_has(message, caplog)
    del config['timeframe']
    config['ticker_interval'] = '15m'
    with pytest.raises(OperationalException, match="DEPRECATED: 'ticker_interval' detected. Please use.*"):
        process_temporary_deprecated_settings(config)

def test_process_deprecated_protections(default_conf: Dict[str, Any], caplog: Any) -> None:
    message: str = "DEPRECATED: Setting 'protections' in the configuration is deprecated."
    config: Dict[str, Any] = deepcopy(default_conf)
    process_temporary_deprecated_settings(config)
    assert not log_has(message, caplog)
    config['protections'] = []
    with pytest.raises(ConfigurationError, match=message):
        process_temporary_deprecated_settings(config)

def test_flat_vars_to_nested_dict(caplog: Any) -> None:
    test_args: Dict[str, str] = {
        'FREQTRADE__EXCHANGE__SOME_SETTING': 'true',
        'FREQTRADE__EXCHANGE__SOME_FALSE_SETTING': 'false',
        'FREQTRADE__EXCHANGE__CONFIG__whatever': 'sometime',
        'FREQTRADE__EXIT_PRICING__PRICE_SIDE': 'bid',
        'FREQTRADE__EXIT_PRICING__cccc': '500',
        'FREQTRADE__STAKE_AMOUNT': '200.05',
        'FREQTRADE__TELEGRAM__CHAT_ID': '2151',
        'NOT_RELEVANT': '200.0',
        'FREQTRADE__ARRAY': '[{"name":"default","host":"xxx"}]',
        'FREQTRADE__EXCHANGE__PAIR_WHITELIST': '["BTC/USDT", "ETH/USDT"]',
        'FREQTRADE__ARRAY_TRAIL_COMMA': '[{"name":"default","host":"xxx",}]',
        'FREQTRADE__OBJECT': '{"name":"default","host":"xxx"}'
    }
    expected: Dict[str, Any] = {
        'stake_amount': 200.05,
        'exit_pricing': {'price_side': 'bid', 'cccc': 500},
        'exchange': {
            'config': {'whatever': 'sometime'},
            'some_setting': True,
            'some_false_setting': False,
            'pair_whitelist': ['BTC/USDT', 'ETH/USDT']
        },
        'telegram': {'chat_id': '2151'},
        'array': [{'name': 'default', 'host': 'xxx'}],
        'object': '{"name":"default","host":"xxx"}',
        'array_trail_comma': '[{"name":"default","host":"xxx",}]'
    }
    res: Dict[str, Any] = _flat_vars_to_nested_dict(test_args, ENV_VAR_PREFIX)
    assert res == expected
    assert log_has("Loading variable 'FREQTRADE__EXCHANGE__SOME_SETTING'", caplog)
    assert not log_has("Loading variable 'NOT_RELEVANT'", caplog)

def test_setup_hyperopt_freqai(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    mocker.patch('freqtrade.configuration.configuration.create_userdata_dir', lambda x, *args, **kwargs: Path(x))
    arglist: List[str] = [
        'hyperopt', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY,
        '--timerange', '20220801-20220805', '--freqaimodel', 'LightGBMRegressorMultiTarget',
        '--analyze-per-epoch'
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    config['freqai'] = {'enabled': True}
    with pytest.raises(OperationalException, match='.*analyze-per-epoch parameter is not supported.*'):
        validate_config_consistency(config)

def test_setup_freqai_backtesting(mocker: Any, default_conf: Dict[str, Any]) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    mocker.patch('freqtrade.configuration.configuration.create_userdata_dir', lambda x, *args, **kwargs: Path(x))
    arglist: List[str] = [
        'backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY,
        '--timerange', '20220801-20220805', '--freqaimodel', 'LightGBMRegressorMultiTarget',
        '--freqai-backtest-live-models'
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    config: Dict[str, Any] = configuration.get_config()
    config['runmode'] = RunMode.BACKTEST
    with pytest.raises(OperationalException, match='.*--freqai-backtest-live-models parameter is only.*'):
        validate_config_consistency(config)
    conf: Dict[str, Any] = deepcopy(config)
    conf['freqai'] = {'enabled': True}
    with pytest.raises(OperationalException, match='.* timerange parameter is not supported with .*'):
        validate_config_consistency(conf)
    conf['timerange'] = None
    conf['freqai_backtest_live_models'] = False
    with pytest.raises(OperationalException, match='.* pass --timerange if you intend to use FreqAI .*'):
        validate_config_consistency(conf)

def test_sanitize_config(default_conf_usdt: Dict[str, Any]) -> None:
    assert default_conf_usdt['exchange']['key'] != 'REDACTED'
    res: Dict[str, Any] = sanitize_config(default_conf_usdt)
    assert default_conf_usdt['exchange']['key'] != 'REDACTED'
    assert 'accountId' not in default_conf_usdt['exchange']
    assert res['exchange']['key'] == 'REDACTED'
    assert res['exchange']['secret'] == 'REDACTED'
    assert 'accountId' not in res['exchange']
    res = sanitize_config(default_conf_usdt, show_sensitive=True)
    assert res['exchange']['key'] == default_conf_usdt['exchange']['key']
    assert res['exchange']['secret'] == default_conf_usdt['exchange']['secret']