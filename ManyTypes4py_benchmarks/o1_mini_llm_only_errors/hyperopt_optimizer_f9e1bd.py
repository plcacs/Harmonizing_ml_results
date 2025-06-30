"""
This module contains the hyperopt optimizer class, which needs to be pickled
and will be sent to the hyperopt worker processes.
"""
import logging
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from joblib import dump, load
from joblib.externals import cloudpickle
from pandas import DataFrame
from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer, HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.util.dry_run_wallet import get_dry_run_wallet
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

logger = logging.getLogger(__name__)
MAX_LOSS = 100000


class HyperOptimizer:
    """
    HyperoptOptimizer class
    This class is sent to the hyperopt worker processes.
    """

    buy_space: List[Dimension]
    sell_space: List[Dimension]
    protection_space: List[Dimension]
    roi_space: List[Dimension]
    stoploss_space: List[Dimension]
    trailing_space: List[Dimension]
    max_open_trades_space: List[Dimension]
    dimensions: List[Dimension]
    config: Config
    backtesting: Backtesting
    pairlist: Any
    analyze_per_epoch: bool
    custom_hyperopt: HyperOptAuto
    custom_hyperoptloss: IHyperOptLoss
    calculate_loss: Callable[..., float]
    data_pickle_file: Path
    market_change: float
    min_date: Optional[datetime]
    max_date: Optional[datetime]
    timerange: Any  # Replace with appropriate type if known

    def __init__(self, config: Config) -> None:
        self.buy_space = []
        self.sell_space = []
        self.protection_space = []
        self.roi_space = []
        self.stoploss_space = []
        self.trailing_space = []
        self.max_open_trades_space = []
        self.dimensions = []
        self.config = config
        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist
        self.analyze_per_epoch = self.config.get('analyze_per_epoch', False)
        if not self.config.get('hyperopt'):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            raise OperationalException(
                'Using separate Hyperopt files has been removed in 2021.9. Please convert your existing Hyperopt file to the new Hyperoptable strategy interface'
            )
        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy
        self.hyperopt_pickle_magic(self.backtesting.strategy.__class__.__bases__)
        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        self.data_pickle_file = self.config['user_data_dir'] / 'hyperopt_results' / 'hyperopt_tickerdata.pkl'
        self.market_change = 0.0
        if HyperoptTools.has_space(self.config, 'sell'):
            self.config['use_exit_signal'] = True
        self.min_date = None
        self.max_date = None

    def prepare_hyperopt(self) -> None:
        self.init_spaces()
        self.prepare_hyperopt_data()
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None
        self.backtesting.exchange._loop_lock = None
        self.backtesting.exchange._cache_lock = None
        self.backtesting.pairlists = None

    def get_strategy_name(self) -> str:
        return self.backtesting.strategy.get_strategy_name()

    def hyperopt_pickle_magic(self, bases: tuple) -> None:
        """
        Hyperopt magic to allow strategy inheritance across files.
        For this to properly work, we need to register the module of the imported class
        to pickle as value.
        """
        for module in bases:
            if module.__name__ != 'IStrategy':
                imported_module = sys.modules.get(module.__module__)
                if imported_module:
                    cloudpickle.register_pickle_by_value(imported_module)
                self.hyperopt_pickle_magic(module.__bases__)

    def _get_params_dict(self, dimensions: List[Dimension], raw_params: List[Any]) -> Dict[str, Any]:
        if len(raw_params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions.')
        return {d.name: v for d, v in zip(dimensions, raw_params, strict=False)}

    def _get_params_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return the params for each space
        """
        result: Dict[str, Any] = {}
        if HyperoptTools.has_space(self.config, 'buy'):
            result['buy'] = {p.name: params.get(p.name) for p in self.buy_space}
        if HyperoptTools.has_space(self.config, 'sell'):
            result['sell'] = {p.name: params.get(p.name) for p in self.sell_space}
        if HyperoptTools.has_space(self.config, 'protection'):
            result['protection'] = {p.name: params.get(p.name) for p in self.protection_space}
        if HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in self.custom_hyperopt.generate_roi_table(params).items()}
        if HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {p.name: params.get(p.name) for p in self.stoploss_space}
        if HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = self.custom_hyperopt.generate_trailing_params(params)
        if HyperoptTools.has_space(self.config, 'trades'):
            result['max_open_trades'] = {
                'max_open_trades': self.backtesting.strategy.max_open_trades
                if self.backtesting.strategy.max_open_trades != float('inf')
                else -1
            }
        return result

    def _get_no_optimize_details(self) -> Dict[str, Any]:
        """
        Get non-optimized parameters
        """
        result: Dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {'stoploss': strategy.stoploss}
        if not HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = {
                'trailing_stop': strategy.trailing_stop,
                'trailing_stop_positive': strategy.trailing_stop_positive,
                'trailing_stop_positive_offset': strategy.trailing_stop_positive_offset,
                'trailing_only_offset_is_reached': strategy.trailing_only_offset_is_reached,
            }
        if not HyperoptTools.has_space(self.config, 'trades'):
            result['max_open_trades'] = {'max_open_trades': strategy.max_open_trades}
        return result

    def init_spaces(self) -> None:
        """
        Assign the dimensions in the hyperoptimization space.
        """
        if HyperoptTools.has_space(self.config, 'protection'):
            logger.debug("Hyperopt has 'protection' space")
            self.config['enable_protections'] = True
            self.backtesting.enable_protections = True
            self.protection_space = self.custom_hyperopt.protection_space()
        if HyperoptTools.has_space(self.config, 'buy'):
            logger.debug("Hyperopt has 'buy' space")
            self.buy_space = self.custom_hyperopt.buy_indicator_space()
        if HyperoptTools.has_space(self.config, 'sell'):
            logger.debug("Hyperopt has 'sell' space")
            self.sell_space = self.custom_hyperopt.sell_indicator_space()
        if HyperoptTools.has_space(self.config, 'roi'):
            logger.debug("Hyperopt has 'roi' space")
            self.roi_space = self.custom_hyperopt.roi_space()
        if HyperoptTools.has_space(self.config, 'stoploss'):
            logger.debug("Hyperopt has 'stoploss' space")
            self.stoploss_space = self.custom_hyperopt.stoploss_space()
        if HyperoptTools.has_space(self.config, 'trailing'):
            logger.debug("Hyperopt has 'trailing' space")
            self.trailing_space = self.custom_hyperopt.trailing_space()
        if HyperoptTools.has_space(self.config, 'trades'):
            logger.debug("Hyperopt has 'trades' space")
            self.max_open_trades_space = self.custom_hyperopt.max_open_trades_space()
        self.dimensions = (
            self.buy_space
            + self.sell_space
            + self.protection_space
            + self.roi_space
            + self.stoploss_space
            + self.trailing_space
            + self.max_open_trades_space
        )

    def assign_params(self, params_dict: Dict[str, Any], category: str) -> None:
        """
        Assign hyperoptable parameters
        """
        for attr_name, attr in self.backtesting.strategy.enumerate_parameters(category):
            if attr.optimize:
                attr.value = params_dict[attr_name]

    def generate_optimizer(
        self, raw_params: List[Any]
    ) -> Dict[str, Any]:
        """
        Used Optimize function.
        Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
        backtest_start_time = datetime.now(timezone.utc)
        params_dict = self._get_params_dict(self.dimensions, raw_params)
        if HyperoptTools.has_space(self.config, 'buy'):
            self.assign_params(params_dict, 'buy')
        if HyperoptTools.has_space(self.config, 'sell'):
            self.assign_params(params_dict, 'sell')
        if HyperoptTools.has_space(self.config, 'protection'):
            self.assign_params(params_dict, 'protection')
        if HyperoptTools.has_space(self.config, 'roi'):
            self.backtesting.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(params_dict)
        if HyperoptTools.has_space(self.config, 'stoploss'):
            self.backtesting.strategy.stoploss = params_dict['stoploss']
        if HyperoptTools.has_space(self.config, 'trailing'):
            trailing_params = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = trailing_params['trailing_stop']
            self.backtesting.strategy.trailing_stop_positive = trailing_params['trailing_stop_positive']
            self.backtesting.strategy.trailing_stop_positive_offset = trailing_params['trailing_stop_positive_offset']
            self.backtesting.strategy.trailing_only_offset_is_reached = trailing_params[
                'trailing_only_offset_is_reached'
            ]
        if HyperoptTools.has_space(self.config, 'trades'):
            if self.config['stake_amount'] == 'unlimited' and (
                params_dict['max_open_trades'] == -1 or params_dict['max_open_trades'] == 0
            ):
                params_dict.update({'max_open_trades': self.config['max_open_trades']})
            updated_max_open_trades: Union[int, float] = (
                int(params_dict['max_open_trades'])
                if params_dict['max_open_trades'] != -1 and params_dict['max_open_trades'] != 0
                else float('inf')
            )
            self.config.update({'max_open_trades': updated_max_open_trades})
            self.backtesting.strategy.max_open_trades = updated_max_open_trades
        with self.data_pickle_file.open('rb') as f:
            processed: DataFrame = load(f, mmap_mode='r')
            if self.analyze_per_epoch:
                processed = self.advise_and_trim(processed)
        bt_results = self.backtesting.backtest(
            processed=processed,
            start_date=self.min_date,  # type: ignore
            end_date=self.max_date,    # type: ignore
        )
        backtest_end_time = datetime.now(timezone.utc)
        bt_results.update(
            {
                'backtest_start_time': int(backtest_start_time.timestamp()),
                'backtest_end_time': int(backtest_end_time.timestamp()),
            }
        )
        return self._get_results_dict(
            backtesting_results=bt_results,
            min_date=self.min_date,  # type: ignore
            max_date=self.max_date,  # type: ignore
            params_dict=params_dict,
            processed=processed,
        )

    def _get_results_dict(
        self,
        backtesting_results: Dict[str, Any],
        min_date: datetime,
        max_date: datetime,
        params_dict: Dict[str, Any],
        processed: DataFrame,
    ) -> Dict[str, Any]:
        params_details = self._get_params_details(params_dict)
        strat_stats = generate_strategy_stats(
            self.pairlist,
            self.backtesting.strategy.get_strategy_name(),
            backtesting_results,
            min_date,
            max_date,
            market_change=self.market_change,
            is_hyperopt=True,
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config['stake_currency']
        )
        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(not_optimized, self._get_no_optimize_details())
        trade_count: int = strat_stats['total_trades']
        total_profit: float = strat_stats['profit_total']
        loss: float = MAX_LOSS
        if trade_count >= self.config['hyperopt_min_trades']:
            loss = self.calculate_loss(
                results=backtesting_results['results'],
                trade_count=trade_count,
                min_date=min_date,
                max_date=max_date,
                config=self.config,
                processed=processed,
                backtest_stats=strat_stats,
                starting_balance=get_dry_run_wallet(self.config),
            )
        return {
            'loss': loss,
            'params_dict': params_dict,
            'params_details': params_details,
            'params_not_optimized': not_optimized,
            'results_metrics': strat_stats,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def get_optimizer(
        self,
        cpu_count: int,
        random_state: int,
        initial_points: int,
        model_queue_size: int,
    ) -> Optimizer:
        dimensions = self.dimensions
        estimator = self.custom_hyperopt.generate_estimator(dimensions=dimensions)
        acq_optimizer: str = 'sampling'
        if isinstance(estimator, str):
            if estimator not in ('GP', 'RF', 'ET', 'GBRT'):
                raise OperationalException(f'Estimator {estimator} not supported.')
            else:
                acq_optimizer = 'auto'
        logger.info(f'Using estimator {estimator}.')
        return Optimizer(
            dimensions,
            base_estimator=estimator,
            acq_optimizer=acq_optimizer,
            n_initial_points=initial_points,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            random_state=random_state,
            model_queue_size=model_queue_size,
        )

    def advise_and_trim(self, data: DataFrame) -> DataFrame:
        preprocessed = self.backtesting.strategy.advise_all_indicators(data)
        trimmed = trim_dataframes(preprocessed, self.timerange, self.backtesting.required_startup)
        self.min_date, self.max_date = get_timerange(trimmed)
        if not self.market_change:
            self.market_change = calculate_market_change(trimmed, 'close')
        return preprocessed

    def prepare_hyperopt_data(self) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
        self.backtesting.load_bt_data_detail()
        logger.info('Dataload complete. Calculating indicators')
        if not self.analyze_per_epoch:
            HyperoptStateContainer.set_state(HyperoptState.INDICATORS)
            preprocessed = self.advise_and_trim(data)
            logger.info(
                f'Hyperopting with data from {self.min_date.strftime(DATETIME_PRINT_FORMAT)} up to '
                f'{self.max_date.strftime(DATETIME_PRINT_FORMAT)} ({(self.max_date - self.min_date).days} days)..'
            )
            dump(preprocessed, self.data_pickle_file)
        else:
            dump(data, self.data_pickle_file)
