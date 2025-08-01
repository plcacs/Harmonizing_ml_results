import logging
from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Union
import numpy as np
import rapidjson
from pandas import isna, json_normalize
from freqtrade.constants import FTHYPT_FILEVERSION, Config
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts, round_dict, safe_value_fallback2
from freqtrade.optimize.hyperopt_epoch_filters import hyperopt_filter_epochs

logger = logging.getLogger(__name__)
NON_OPT_PARAM_APPENDIX = '  # value loaded from strategy'
HYPER_PARAMS_FILE_FORMAT = rapidjson.NM_NATIVE | rapidjson.NM_NAN

def hyperopt_serializer(x: Any) -> Union[int, bool, str]:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)

class HyperoptStateContainer:
    """Singleton class to track state of hyperopt"""
    state: HyperoptState = HyperoptState.OPTIMIZE

    @classmethod
    def set_state(cls, value: HyperoptState) -> None:
        cls.state = value

class HyperoptTools:

    @staticmethod
    def get_strategy_filename(config: Config, strategy_name: str) -> Optional[Path]:
        """
        Get Strategy-location (filename) from strategy_name
        """
        from freqtrade.resolvers.strategy_resolver import StrategyResolver
        strategy_objs: List[Dict[str, Any]] = StrategyResolver.search_all_objects(config, False, config.get('recursive_strategy_search', False))
        strategies: List[Dict[str, Any]] = [s for s in strategy_objs if s['name'] == strategy_name]
        if strategies:
            strategy: Dict[str, Any] = strategies[0]
            return Path(strategy['location'])
        return None

    @staticmethod
    def export_params(params: Dict[str, Any], strategy_name: str, filename: Path) -> None:
        """
        Generate files
        """
        final_params: Dict[str, Any] = deepcopy(params['params_not_optimized'])
        final_params = deep_merge_dicts(params['params_details'], final_params)
        final_params = {
            'strategy_name': strategy_name,
            'params': final_params,
            'ft_stratparam_v': 1,
            'export_time': datetime.now(timezone.utc)
        }
        logger.info(f'Dumping parameters to {filename}')
        with filename.open('w') as f:
            rapidjson.dump(final_params, f, indent=2, default=hyperopt_serializer, number_mode=HYPER_PARAMS_FILE_FORMAT)

    @staticmethod
    def load_params(filename: Path) -> Any:
        """
        Load parameters from file
        """
        with filename.open('r') as f:
            params: Any = rapidjson.load(f, number_mode=HYPER_PARAMS_FILE_FORMAT)
        return params

    @staticmethod
    def try_export_params(config: Config, strategy_name: str, params: Dict[str, Any]) -> None:
        if params.get(FTHYPT_FILEVERSION, 1) >= 2 and (not config.get('disableparamexport', False)):
            fn: Optional[Path] = HyperoptTools.get_strategy_filename(config, strategy_name)
            if fn:
                HyperoptTools.export_params(params, strategy_name, fn.with_suffix('.json'))
            else:
                logger.warning('Strategy not found, not exporting parameter file.')

    @staticmethod
    def has_space(config: Dict[str, Any], space: str) -> bool:
        """
        Tell if the space value is contained in the configuration
        """
        if space in ('trailing', 'protection', 'trades'):
            return any((s in config['spaces'] for s in [space, 'all']))
        else:
            return any((s in config['spaces'] for s in [space, 'all', 'default']))

    @staticmethod
    def _read_results(results_file: Path, batch_size: int = 10) -> Iterator[List[Any]]:
        """
        Stream hyperopt results from file
        """
        logger.info(f"Reading epochs from '{results_file}'")
        with results_file.open('r') as f:
            data: List[Any] = []
            for line in f:
                data += [rapidjson.loads(line)]
                if len(data) >= batch_size:
                    yield data
                    data = []
        yield data

    @staticmethod
    def _test_hyperopt_results_exist(results_file: Path) -> bool:
        if results_file.is_file() and results_file.stat().st_size > 0:
            if results_file.suffix == '.pickle':
                raise OperationalException('Legacy hyperopt results are no longer supported.Please rerun hyperopt or use an older version to load this file.')
            return True
        else:
            return False

    @staticmethod
    def load_filtered_results(results_file: Path, config: Dict[str, Any]) -> Tuple[List[Any], int]:
        filteroptions: Dict[str, Any] = {
            'only_best': config.get('hyperopt_list_best', False),
            'only_profitable': config.get('hyperopt_list_profitable', False),
            'filter_min_trades': config.get('hyperopt_list_min_trades', 0),
            'filter_max_trades': config.get('hyperopt_list_max_trades', 0),
            'filter_min_avg_time': config.get('hyperopt_list_min_avg_time'),
            'filter_max_avg_time': config.get('hyperopt_list_max_avg_time'),
            'filter_min_avg_profit': config.get('hyperopt_list_min_avg_profit'),
            'filter_max_avg_profit': config.get('hyperopt_list_max_avg_profit'),
            'filter_min_total_profit': config.get('hyperopt_list_min_total_profit'),
            'filter_max_total_profit': config.get('hyperopt_list_max_total_profit'),
            'filter_min_objective': config.get('hyperopt_list_min_objective'),
            'filter_max_objective': config.get('hyperopt_list_max_objective')
        }
        if not HyperoptTools._test_hyperopt_results_exist(results_file):
            logger.warning(f'Hyperopt file {results_file} not found.')
            return ([], 0)
        epochs: List[Any] = []
        total_epochs: int = 0
        for epochs_tmp in HyperoptTools._read_results(results_file):
            if total_epochs == 0 and epochs_tmp[0].get('is_best') is None:
                raise OperationalException('The file with HyperoptTools results is incompatible with this version of Freqtrade and cannot be loaded.')
            total_epochs += len(epochs_tmp)
            epochs += hyperopt_filter_epochs(epochs_tmp, filteroptions, log=False)
        logger.info(f'Loaded {total_epochs} previous evaluations from disk.')
        epochs = hyperopt_filter_epochs(epochs, filteroptions, log=True)
        return (epochs, total_epochs)

    @staticmethod
    def show_epoch_details(results: Dict[str, Any], total_epochs: int, print_json: bool, no_header: bool = False, header_str: Optional[str] = None) -> None:
        """
        Display details of the hyperopt result
        """
        params: Dict[str, Any] = results.get('params_details', {})
        non_optimized: Dict[str, Any] = results.get('params_not_optimized', {})
        if header_str is None:
            header_str = 'Best result'
        if not no_header:
            explanation_str: str = HyperoptTools._format_explanation_string(results, total_epochs)
            print(f'\n{header_str}:\n\n{explanation_str}\n')
        if print_json:
            result_dict: Dict[str, Any] = {}
            for s in ['buy', 'sell', 'protection', 'roi', 'stoploss', 'trailing', 'max_open_trades']:
                HyperoptTools._params_update_for_json(result_dict, params, non_optimized, s)
            print(rapidjson.dumps(result_dict, default=str, number_mode=HYPER_PARAMS_FILE_FORMAT))
        else:
            HyperoptTools._params_pretty_print(params, 'buy', 'Buy hyperspace params:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'sell', 'Sell hyperspace params:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'protection', 'Protection hyperspace params:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'roi', 'ROI table:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'stoploss', 'Stoploss:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'trailing', 'Trailing stop:', non_optimized)
            HyperoptTools._params_pretty_print(params, 'max_open_trades', 'Max Open Trades:', non_optimized)

    @staticmethod
    def _params_update_for_json(result_dict: Dict[str, Any], params: Dict[str, Any], non_optimized: Dict[str, Any], space: str) -> None:
        if space in params or space in non_optimized:
            space_params: Dict[str, Any] = HyperoptTools._space_params(params, space)
            space_non_optimized: Dict[str, Any] = HyperoptTools._space_params(non_optimized, space)
            all_space_params: Dict[str, Any] = space_params
            if len(space_non_optimized) > 0:
                all_space_params = {**space_params, **space_non_optimized}
            if space in ['buy', 'sell']:
                result_dict.setdefault('params', {}).update(all_space_params)
            elif space == 'roi':
                result_dict['minimal_roi'] = {str(k): v for k, v in all_space_params.items()}
            else:
                result_dict.update(all_space_params)

    @staticmethod
    def _params_pretty_print(params: Dict[str, Any], space: str, header: str, non_optimized: Optional[Dict[str, Any]] = None) -> None:
        if space in params or (non_optimized and space in non_optimized):
            space_params: Dict[str, Any] = HyperoptTools._space_params(params, space, 5)
            no_params: Dict[str, Any] = HyperoptTools._space_params(non_optimized, space, 5) if non_optimized else {}
            appendix: str = ''
            if not space_params and (not no_params):
                return
            if not space_params:
                appendix = NON_OPT_PARAM_APPENDIX
            result: str = f'\n# {header}\n'
            if space == 'stoploss':
                stoploss: Any = safe_value_fallback2(space_params, no_params, space, space)
                result += f'stoploss = {stoploss}{appendix}'
            elif space == 'max_open_trades':
                max_open_trades: Any = safe_value_fallback2(space_params, no_params, space, space)
                result += f'max_open_trades = {max_open_trades}{appendix}'
            elif space == 'roi':
                result = result[:-1] + f'{appendix}\n'
                minimal_roi_result: str = rapidjson.dumps({str(k): v for k, v in (space_params or no_params).items()}, default=str, indent=4, number_mode=rapidjson.NM_NATIVE)
                result += f'minimal_roi = {minimal_roi_result}'
            elif space == 'trailing':
                for k, v in (space_params or no_params).items():
                    result += f'{k} = {v}{appendix}\n'
            else:
                result += f'{space}_params = {HyperoptTools._pprint_dict(space_params, no_params)}'
            result = result.replace('\n', '\n    ')
            print(result)

    @staticmethod
    def _space_params(params: Dict[str, Any], space: str, r: Optional[int] = None) -> Dict[str, Any]:
        d: Optional[Dict[str, Any]] = params.get(space)
        if d:
            return round_dict(d, r) if r else d
        return {}

    @staticmethod
    def _pprint_dict(params: Dict[str, Any], non_optimized: Dict[str, Any], indent: int = 4) -> str:
        """
        Pretty-print hyperopt results (based on 2 dicts - with add. comment)
        """
        p: Dict[str, Any] = params.copy()
        p.update(non_optimized)
        result: str = '{\n'
        for k, param in p.items():
            result += ' ' * indent + f'"{k}": '
            result += f'"{param}",' if isinstance(param, str) else f'{param},'
            if k in non_optimized:
                result += NON_OPT_PARAM_APPENDIX
            result += '\n'
        result += '}'
        return result

    @staticmethod
    def is_best_loss(results: Dict[str, Any], current_best_loss: float) -> bool:
        return bool(results['loss'] < current_best_loss)

    @staticmethod
    def format_results_explanation_string(results_metrics: Dict[str, Any], stake_currency: str) -> str:
        """
        Return the formatted results explanation in a string
        """
        return (
            f"{results_metrics['total_trades']:6d} trades. "
            f"{results_metrics['wins']}/{results_metrics['draws']}/{results_metrics['losses']} Wins/Draws/Losses. "
            f"Avg profit {results_metrics['profit_mean']:7.2%}. "
            f"Median profit {results_metrics['profit_median']:7.2%}. "
            f"Total profit {results_metrics['profit_total_abs']:11.8f} {stake_currency} "
            f"({results_metrics['profit_total']:8.2%}). "
            f"Avg duration {results_metrics['holding_avg']} min."
        )

    @staticmethod
    def _format_explanation_string(results: Dict[str, Any], total_epochs: int) -> str:
        return ('*' if results['is_initial_point'] else ' ') + f"{results['current_epoch']:5d}/{total_epochs}: " + f"{results['results_explanation']} " + f"Objective: {results['loss']:.5f}"

    @staticmethod
    def export_csv_file(config: Dict[str, Any], results: List[Dict[str, Any]], csv_file: Union[str, Path]) -> None:
        """
        Log result to csv-file
        """
        if not results:
            return
        csv_path: Path = Path(csv_file) if isinstance(csv_file, str) else csv_file
        if csv_path.is_file():
            logger.error(f'CSV file already exists: {csv_path}')
            return
        try:
            csv_path.open('w+').close()
        except OSError:
            logger.error(f'Failed to create CSV file: {csv_path}')
            return
        trials = json_normalize(results, max_level=1)
        trials['Best'] = ''
        base_metrics: List[str] = [
            'Best', 'current_epoch', 'results_metrics.total_trades', 'results_metrics.profit_mean', 
            'results_metrics.profit_median', 'results_metrics.profit_total', 'results_metrics.stake_currency', 
            'results_metrics.profit_total_abs', 'results_metrics.holding_avg', 'results_metrics.trade_count_long', 
            'results_metrics.trade_count_short', 'results_metrics.max_drawdown_abs', 'results_metrics.max_drawdown_account', 
            'loss', 'is_initial_point', 'is_best'
        ]
        perc_multi: int = 100
        param_metrics: List[str] = ['params_dict.' + param for param in results[0]['params_dict'].keys()]
        trials = trials[base_metrics + param_metrics]
        base_columns: List[str] = [
            'Best', 'Epoch', 'Trades', 'Avg profit', 'Median profit', 'Total profit', 
            'Stake currency', 'Profit', 'Avg duration', 'Trade count long', 'Trade count short', 
            'Max drawdown', 'Max drawdown percent', 'Objective', 'is_initial_point', 'is_best'
        ]
        param_columns: List[str] = list(results[0]['params_dict'].keys())
        trials.columns = base_columns + param_columns
        trials['is_profit'] = False
        trials.loc[trials['is_initial_point'], 'Best'] = '*'
        trials.loc[trials['is_best'], 'Best'] = 'Best'
        trials.loc[trials['is_initial_point'] & trials['is_best'], 'Best'] = '* Best'
        trials.loc[trials['Total profit'] > 0, 'is_profit'] = True
        trials['Epoch'] = trials['Epoch'].astype(str)
        trials['Trades'] = trials['Trades'].astype(str)
        trials['Median profit'] = trials['Median profit'] * perc_multi
        trials['Total profit'] = trials['Total profit'].apply(lambda x: f'{x:,.8f}' if x != 0.0 else '')
        trials['Profit'] = trials['Profit'].apply(lambda x: f'{x:,.2f}' if not isna(x) else '')
        trials['Avg profit'] = trials['Avg profit'].apply(lambda x: f'{x * perc_multi:,.2f}%' if not isna(x) else '')
        trials['Max drawdown percent'] = trials['Max drawdown percent'].apply(lambda x: f'{x * perc_multi:,.2f}%' if not isna(x) else '')
        trials['Objective'] = trials['Objective'].apply(lambda x: f'{x:,.5f}' if x != 100000 else '')
        trials = trials.drop(columns=['is_initial_point', 'is_best', 'is_profit'])
        trials.to_csv(csv_path, index=False, header=True, mode='w', encoding='UTF-8')
        logger.info(f'CSV file created: {csv_path}')