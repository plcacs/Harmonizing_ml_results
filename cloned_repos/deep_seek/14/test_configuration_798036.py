import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
    config_file = Path(__file__).parents[1] / 'config_examples/config_full.example.json'
    conf = load_config_file(str(config_file))
    return conf

def test_load_config_missing_attributes(default_conf: Dict[str, Any]) -> None:
    conf = deepcopy(default_conf)
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
    file_mock = mocker.patch('freqtrade.configuration.load_config.Path.open', mocker.mock_open(read_data=json.dumps(default_conf)))
    validated_conf = load_config_file('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()

def test_load_config_file_error(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    del default_conf['user_data_dir']
    default_conf['datadir'] = str(default_conf['datadir'])
    filedata = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch('freqtrade.configuration.load_config.Path.open', mocker.mock_open(read_data=filedata))
    mocker.patch.object(Path, 'read_text', MagicMock(return_value=filedata))
    with pytest.raises(OperationalException, match='.*Please verify the following segment.*'):
        load_config_file('somefile')

def test_load_config_file_error_range(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    del default_conf['user_data_dir']
    default_conf['datadir'] = str(default_conf['datadir'])
    filedata = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch.object(Path, 'read_text', MagicMock(return_value=filedata))
    x = log_config_error_range('somefile', 'Parse error at offset 64: Invalid value.')
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
    testpath = tmp_path / 'config.json'
    with pytest.raises(OperationalException, match='File .* not found!'):
        load_file(testpath)

def test__args_to_config(caplog: Any) -> None:
    arg_list = ['trade', '--strategy-path', 'TestTest']
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    config = {}
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
        configuration._args_to_config(config, argname='strategy_path', logstring='DeadBeef', deprecated_msg='Going away soon!')
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
    validated_conf = configuration.load_config()
    assert validated_conf['max_open_trades'] == 0
    assert 'internals' in validated_conf

def test_load_config_combine_dicts(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']
    config_files = [conf1, conf2]
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.load_config.load_config_file', configsmock)
    arg_list = ['trade', '-c', 'test_conf.json', '--config', 'test2_conf.json']
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    exchange_conf = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']
    assert 'internals' in validated_conf

def test_from_config(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']
    conf2['fiat_display_currency'] = 'EUR'
    config_files = [conf1, conf2]
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.load_config.load_config_file', configsmock)
    validated_conf = Configuration.from_files(['test_conf.json', 'test2_conf.json'])
    exchange_conf = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']
    assert validated_conf['fiat_display_currency'] == 'EUR'
    assert 'internals' in validated_conf
    assert isinstance(validated_conf['user_data_dir'], Path)

def test_from_recursive_files(testdatadir: Path) -> None:
    files = testdatadir / 'testconfigs/testconfig.json'
    conf = Configuration.from_files([files])
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
    conf1 = deepcopy(default_conf)
    del conf1['user_data_dir']
    conf1['datadir'] = str(conf1['datadir'])
    config_files = [conf1]
    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    mocker.patch('freqtrade.configuration.configuration.load_from_files', configsmock)
    validated_conf = Configuration.from_files(['test_conf.json'])
    assert isinstance(validated_conf['user_data_dir'], Path)
    assert 'user_data_dir' in validated_conf
    assert 'original_config' in validated_conf
    assert isinstance(json.dumps(validated_conf['original_config']), str)

def test_load_config_max_open_trades_minus_one(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf['max_open_trades'] = -1
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
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
    validated_conf = configuration.load_config()
    assert validated_conf.get('strategy_path') is None
    assert 'edge' not in validated_conf

def test_load_config_with_params(default_conf: Dict[str, Any], mocker: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--strategy-path', '/some/path', '--db-url', 'sqlite:///someurl']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('strategy') == 'TestStrategy'
    assert validated_conf.get('strategy_path') == '/some/path'
    assert validated_conf.get('db_url') == 'sqlite:///someurl'
    conf = default_conf.copy()
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

@pytest.mark.parametrize('config_value,expected,arglist', [(True, True, ['trade', '--dry-run']), (False, True, ['trade', '--dry-run']), (False, False, ['trade']), (True, True, ['trade'])])
def test_load_dry_run(default_conf: Dict[str, Any], mocker: Any, config_value: bool, expected: bool, arglist: List[str]) -> None:
    default_conf['dry_run'] = config_value
    patched_configuration_load_config_file(mocker, default_conf)
    configuration = Configuration(Arguments(arglist).get_parsed_arg())
    validated_conf = configuration.load_config()
    assert validated_conf['dry_run'] is expected
    assert validated_conf['runmode'] == (RunMode.DRY_RUN if expected else RunMode.LIVE)

def test_load_custom_strategy(default_conf: Dict[str, Any], mocker: Any, tmp_path: Path) -> None:
    default_conf.update({'strategy': 'CustomStrategy', 'strategy_path': f'{tmp_path}/strategies'})
    patched_configuration_load_config_file(mocker, default_conf)
    args = Arguments(['trade']).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('strategy') == 'CustomStrategy'
    assert validated_conf.get('strategy_path') == f'{tmp_path}/strategies'

def test_show_info(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist = ['trade', '--strategy', 'TestStrategy', '--db-url', 'sqlite:///tmp/testdb']
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    configuration.get_config()
    assert log_has('Using DB: "sqlite:///tmp/testdb"', caplog)
    assert log_has('Dry run is enabled', caplog)

def test_setup_configuration_without_arguments(mocker: Any, default_conf: Dict[str, Any], caplog: Any) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    arglist = ['backtesting',