import asyncio
import inspect
import logging
import signal
from collections.abc import Generator
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import floor, isnan
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import ccxt
import ccxt.pro as ccxt_pro
from cachetools import TTLCache
from ccxt import TICK_SIZE
from dateutil import parser
from pandas import DataFrame, concat
from freqtrade.constants import DEFAULT_AMOUNT_RESERVE_PERCENT, DEFAULT_TRADES_COLUMNS, NON_OPEN_EXCHANGE_STATES, BidAsk, BuySell, Config, EntryExit, ExchangeConfig, ListPairsWithTimeframes, MakerTaker, OBLiteral, PairWithTimeframe
from freqtrade.data.converter import clean_ohlcv_dataframe, ohlcv_to_dataframe, trades_df_remove_duplicates, trades_dict_to_list, trades_list_to_df
from freqtrade.enums import OPTIMIZE_MODES, TRADE_MODES, CandleType, MarginMode, PriceType, RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError, DDosProtection, ExchangeError, InsufficientFundsError, InvalidOrderException, OperationalException, PricingError, RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_FETCH_ORDER_RETRY_COUNT, remove_exchange_credentials, retrier, retrier_async
from freqtrade.exchange.exchange_types import CcxtBalances, CcxtOrder, CcxtPosition, FtHas, OHLCVResponse, OrderBook, Ticker, Tickers
from freqtrade.exchange.exchange_utils import ROUND, ROUND_DOWN, ROUND_UP, amount_to_contract_precision, amount_to_contracts, amount_to_precision, contracts_to_amount, date_minus_candles, is_exchange_known_ccxt, market_is_active, price_to_precision
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes, timeframe_to_msecs, timeframe_to_next_date, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.exchange_ws import ExchangeWS
from freqtrade.misc import chunks, deep_merge_dicts, file_dump_json, file_load_json, safe_value_fallback2
from freqtrade.util import dt_from_ts, dt_now
from freqtrade.util.datetime_helpers import dt_humanize_delta, dt_ts, format_ms_time
from freqtrade.util.periodic_cache import PeriodicCache

logger = logging.getLogger(__name__)
T = Any

class Exchange:
    _params: Dict[str, Any] = {}
    _ccxt_params: Dict[str, Any] = {}
    _ft_has_default: Dict[str, Any] = {
        'has': None,
        'ohlcv_candle_limit': 1000,
        'trades_limit': 1000,
        'trades_pagination': 'time',
        'order_types': {},
        'order_time_in_force': ['GTC'],
        'stoploss_on_exchange': False,
        'stoploss_order_types': {},
        'mark_ohlcv_price': 'mark',
        'mark_ohlcv_timeframe': '1m',
        'funding_fee_timeframe': '1h',
        'ccxt_futures_name': 'futures',
        'floor_leverage': False,
        'ohlcv_require_since': False,
        'l2_limit_range': None,
        'l2_limit_range_required': True,
        'proxy_coin_mapping': {}
    }
    _ft_has: Dict[str, Any] = {}
    _ft_has_futures: Dict[str, Any] = {}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = []

    def __init__(self, config: Dict[str, Any]) -> None:
        self._ws_async: Optional[Any] = None
        self._exchange_ws: Optional[ExchangeWS] = None
        self._markets: Dict[str, Any] = {}
        self._trading_fees: Dict[str, Any] = {}
        self._leverage_tiers: Dict[str, Any] = {}
        self._loop_lock: Lock = Lock()
        self.loop: asyncio.AbstractEventLoop = self._init_async_loop()
        self._config: Dict[str, Any] = {}
        self._config.update(config)
        self._pairs_last_refresh_time: Dict[Tuple[Any, ...], int] = {}
        self._last_markets_refresh: int = 0
        self._cache_lock: Lock = Lock()
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=4, ttl=60 * 10)
        self._exit_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._entry_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._klines: Dict[Any, DataFrame] = {}
        self._expiring_candle_cache: Dict[Any, PeriodicCache] = {}
        self._trades: Dict[Any, DataFrame] = {}
        self._dry_run_open_orders: Dict[str, dict] = {}
        if self._config.get('dry_run'):
            logger.info('Instance is running with dry_run enabled')
        logger.info(f'Using CCXT {ccxt.__version__}')
        exchange_conf: Dict[str, Any] = self._config.get('exchange', {})
        remove_exchange_credentials(exchange_conf, self._config.get('dry_run', False))
        self.log_responses: bool = exchange_conf.get('log_responses', False)
        self.trading_mode: TradingMode = self._config.get('trading_mode', TradingMode.SPOT)
        self.margin_mode: MarginMode = MarginMode(self._config.get('margin_mode')) if self._config.get('margin_mode') else MarginMode.NONE
        self.liquidation_buffer: float = self._config.get('liquidation_buffer', 0.05)
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if self.trading_mode == TradingMode.FUTURES:
            self._ft_has = deep_merge_dicts(self._ft_has_futures, self._ft_has)
        if exchange_conf.get('_ft_has_params'):
            self._ft_has = deep_merge_dicts(exchange_conf.get('_ft_has_params'), self._ft_has)
            logger.info('Overriding exchange._ft_has with config params, result: %s', self._ft_has)
        self._ohlcv_partial_candle: bool = self._ft_has['ohlcv_partial_candle'] if 'ohlcv_partial_candle' in self._ft_has else True
        self._max_trades_limit: int = self._ft_has['trades_limit']
        self._trades_pagination: str = self._ft_has['trades_pagination']
        self._trades_pagination_arg: str = self._ft_has.get('trades_pagination_arg', 'since')
        ccxt_config: Dict[str, Any] = {}
        self._api: Any = self._init_ccxt(exchange_conf, True, ccxt_config)
        ccxt_async_config: Dict[str, Any] = {}
        self._api_async: Any = self._init_ccxt(exchange_conf, False, ccxt_async_config)
        self._has_watch_ohlcv: bool = self.exchange_has('watchOHLCV') and self._ft_has.get('ws_enabled', False)
        if self._config.get('runmode') in TRADE_MODES and exchange_conf.get('enable_ws', True) and self._has_watch_ohlcv:
            self._ws_async = self._init_ccxt(exchange_conf, False, ccxt_async_config)
            self._exchange_ws = ExchangeWS(self._config, self._ws_async)
        logger.info(f'Using Exchange "{self.name}"')
        self.required_candle_call_count: int = 1
        self.markets_refresh_interval: int = exchange_conf.get('markets_refresh_interval', 60) * 60 * 1000
        self.reload_markets(True, load_leverage_tiers=False)
        self.validate_config(self._config)
        self._startup_candle_count: int = self._config.get('startup_candle_count', 0)
        self.required_candle_call_count = self.validate_required_startup_candles(self._startup_candle_count, self._config.get('timeframe', ''))
        if self.trading_mode != TradingMode.SPOT and self._config.get('load_leverage_tiers', False):
            self.fill_leverage_tiers()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._exchange_ws:
            self._exchange_ws.cleanup()
        logger.debug('Exchange object destroyed, closing async loop')
        if getattr(self, '_api_async', None) and inspect.iscoroutinefunction(self._api_async.close) and getattr(self._api_async, 'session', None):
            logger.debug('Closing async ccxt session.')
            self.loop.run_until_complete(self._api_async.close())
        if self._ws_async and inspect.iscoroutinefunction(self._ws_async.close) and getattr(self._ws_async, 'session', None):
            logger.debug('Closing ws ccxt session.')
            self.loop.run_until_complete(self._ws_async.close())
        if self.loop and (not self.loop.is_closed()):
            self.loop.close()

    def _init_async_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def validate_config(self, config: Dict[str, Any]) -> None:
        self.validate_timeframes(config.get('timeframe', ''))
        self.validate_stakecurrency(config['stake_currency'])
        self.validate_ordertypes(config.get('order_types', {}))
        self.validate_order_time_in_force(config.get('order_time_in_force', {}))
        self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
        self.validate_pricing(config['exit_pricing'])
        self.validate_pricing(config['entry_pricing'])
        self.validate_orderflow(config['exchange'])
        self.validate_freqai(config)

    def _init_ccxt(self, exchange_config: Dict[str, Any], sync: bool, ccxt_kwargs: Dict[str, Any]) -> Any:
        name = exchange_config['name']
        ccxt_module = ccxt if sync else ccxt_pro
        if not is_exchange_known_ccxt(name, ccxt_module):
            import ccxt.async_support as ccxt_async
            ccxt_module = ccxt_async
        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f'Exchange {name} is not supported by ccxt')
        ex_config: Dict[str, Any] = {
            'apiKey': exchange_config.get('api_key', exchange_config.get('apiKey', exchange_config.get('key'))),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
            'accountId': exchange_config.get('account_id', exchange_config.get('accountId', '')),
            'walletAddress': exchange_config.get('wallet_address', exchange_config.get('walletAddress')),
            'privateKey': exchange_config.get('private_key', exchange_config.get('privateKey'))
        }
        if ccxt_kwargs:
            logger.info('Applying additional ccxt config: %s', ccxt_kwargs)
        if self._ccxt_params:
            ccxt_kwargs = deep_merge_dicts(self._ccxt_params, deepcopy(ccxt_kwargs))
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:
            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f'Exchange {name} is not supported') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Initialization of ccxt failed. Reason: {e}') from e
        return api

    @property
    def _ccxt_config(self) -> Dict[str, Any]:
        if self.trading_mode == TradingMode.MARGIN:
            return {'options': {'defaultType': 'margin'}}
        elif self.trading_mode == TradingMode.FUTURES:
            return {'options': {'defaultType': self._ft_has.get('ccxt_futures_name')}}
        else:
            return {}

    @property
    def name(self) -> str:
        return self._api.name

    @property
    def id(self) -> str:
        return self._api.id

    @property
    def timeframes(self) -> List[str]:
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> Dict[str, Any]:
        if not self._markets:
            logger.info('Markets were not loaded. Loading them now..')
            self.reload_markets(True)
        return self._markets

    @property
    def precisionMode(self) -> Any:
        return self._api.precisionMode

    @property
    def precision_mode_price(self) -> Any:
        return self._api.precisionMode

    def additional_exchange_init(self) -> None:
        pass

    def _log_exchange_response(self, endpoint: str, response: Any, *, add_info: Optional[Any] = None) -> None:
        if self.log_responses:
            add_info_str = '' if add_info is None else f' {add_info}: '
            logger.info(f'API {endpoint}:{add_info_str}{response}')

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        ccxt_val = self.features('spot' if candle_type == CandleType.SPOT else 'futures', 'fetchOHLCV', 'limit', 500)
        if not isinstance(ccxt_val, (float, int)):
            ccxt_val = 500
        fallback_val = self._ft_has.get('ohlcv_candle_limit', ccxt_val)
        if candle_type == CandleType.FUNDING_RATE:
            fallback_val = self._ft_has.get('funding_fee_candle_limit', fallback_val)
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {}).get(timeframe, str(fallback_val)))

    def get_markets(
        self,
        base_currencies: Optional[List[str]] = None,
        quote_currencies: Optional[List[str]] = None,
        spot_only: bool = False,
        margin_only: bool = False,
        futures_only: bool = False,
        tradable_only: bool = True,
        active_only: bool = False
    ) -> Dict[str, Any]:
        markets = self.markets
        if not markets:
            raise OperationalException('Markets were not loaded.')
        if base_currencies:
            markets = {k: v for k, v in markets.items() if v['base'] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v['quote'] in quote_currencies}
        if tradable_only:
            markets = {k: v for k, v in markets.items() if self.market_is_tradable(v)}
        if spot_only:
            markets = {k: v for k, v in markets.items() if self.market_is_spot(v)}
        if margin_only:
            markets = {k: v for k, v in markets.items() if self.market_is_margin(v)}
        if futures_only:
            markets = {k: v for k, v in markets.items() if self.market_is_future(v)}
        if active_only:
            markets = {k: v for k, v in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> List[str]:
        markets = self.markets
        return sorted(set([x['quote'] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get('quote', '')

    def get_pair_base_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get('base', '')

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        return market.get(self._ft_has.get('ccxt_futures_name'), False) is True and market.get('type', False) == 'swap' and (market.get('linear', False) is True)

    def market_is_spot(self, market: Dict[str, Any]) -> bool:
        return market.get('spot', False) is True

    def market_is_margin(self, market: Dict[str, Any]) -> bool:
        return market.get('margin', False) is True

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        return (
            market.get('quote') is not None and
            market.get('base') is not None and
            (self.precisionMode != TICK_SIZE or market.get('precision', {}).get('price') is None or market.get('precision', {}).get('price') > 1e-11) and
            ((self.trading_mode == TradingMode.SPOT and self.market_is_spot(market)) or
             (self.trading_mode == TradingMode.MARGIN and self.market_is_margin(market)) or
             (self.trading_mode == TradingMode.FUTURES and self.market_is_future(market)))
        )

    def klines(self, pair_interval: Any, copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def trades(self, pair_interval: Any, copy: bool = True) -> DataFrame:
        if pair_interval in self._trades:
            return self._trades[pair_interval].copy() if copy else self._trades[pair_interval]
        else:
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

    def get_contract_size(self, pair: str) -> Optional[float]:
        if self.trading_mode == TradingMode.FUTURES:
            market = self.markets.get(pair, {})
            contract_size: float = 1.0
            if not market:
                return None
            if market.get('contractSize') is not None:
                contract_size = float(market['contractSize'])
            return contract_size
        else:
            return 1

    def _trades_contracts_to_amount(self, trades: List[dict]) -> List[dict]:
        if len(trades) > 0 and 'symbol' in trades[0]:
            contract_size = self.get_contract_size(trades[0]['symbol'])
            if contract_size != 1:
                for trade in trades:
                    trade['amount'] = trade['amount'] * contract_size
        return trades

    def _order_contracts_to_amount(self, order: dict) -> dict:
        if 'symbol' in order and order['symbol'] is not None:
            contract_size = self.get_contract_size(order['symbol'])
            if contract_size != 1:
                for prop in self._ft_has.get('order_props_in_contracts', []):
                    if prop in order and order[prop] is not None:
                        order[prop] = order[prop] * contract_size
        return order

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        contract_size = self.get_contract_size(pair)
        return amount_to_contracts(amount, contract_size)

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:
        contract_size = self.get_contract_size(pair)
        return contracts_to_amount(num_contracts, contract_size)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        contract_size = self.get_contract_size(pair)
        return amount_to_contract_precision(amount, self.get_precision_amount(pair), self.precisionMode, contract_size)

    def ws_connection_reset(self) -> None:
        if self._exchange_ws:
            self._exchange_ws.reset_connections()

    async def _api_reload_markets(self, reload: bool = False) -> Any:
        try:
            return await self._api_async.load_markets(reload=reload, params={})
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in reload_markets due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise TemporaryError(e) from e

    def _load_async_markets(self, reload: bool = False) -> Any:
        try:
            with self._loop_lock:
                markets = self.loop.run_until_complete(self._api_reload_markets(reload=reload))
            if isinstance(markets, Exception):
                raise markets
            return markets
        except asyncio.TimeoutError as e:
            logger.warning('Could not load markets. Reason: %s', e)
            raise TemporaryError from e

    def reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        is_initial = self._last_markets_refresh == 0
        if not force and self._last_markets_refresh > 0 and (self._last_markets_refresh + self.markets_refresh_interval > dt_ts()):
            return
        logger.debug('Performing scheduled market reload..')
        try:
            retries: int = 3 if force else 0
            self._markets = retrier(self._load_async_markets, retries=retries)(reload=True)
            self._api.set_markets(self._api_async.markets, self._api_async.currencies)
            self._api.options = self._api_async.options
            if self._exchange_ws:
                self._ws_async.set_markets(self._api.markets, self._api.currencies)
                self._ws_async.options = self._api.options
            self._last_markets_refresh = dt_ts()
            if is_initial and self._ft_has.get('needs_trading_fees', False):
                self._trading_fees = self.fetch_trading_fees()
            if load_leverage_tiers and self.trading_mode == TradingMode.FUTURES:
                self.fill_leverage_tiers()
        except (ccxt.BaseError, TemporaryError):
            logger.exception('Could not load markets.')

    def validate_stakecurrency(self, stake_currency: str) -> None:
        if not self._markets:
            raise OperationalException('Could not load markets, therefore cannot start. Please investigate the above error for more details.')
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise ConfigurationError(f'{stake_currency} is not available as stake on {self.name}. Available currencies are: {", ".join(quote_currencies)}')

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> Generator[str, None, None]:
        yielded = False
        for pair in (f'{curr_1}/{curr_2}', f'{curr_2}/{curr_1}', f'{curr_1}/{curr_2}:{curr_2}', f'{curr_2}/{curr_1}:{curr_1}'):
            if pair in self.markets and self.markets[pair].get('active'):
                yielded = True
                yield pair
        if not yielded:
            raise ValueError(f'Could not combine {curr_1} and {curr_2} to get a valid pair.')

    def validate_timeframes(self, timeframe: str) -> None:
        if not hasattr(self._api, 'timeframes') or self._api.timeframes is None:
            raise OperationalException(f'The ccxt library does not provide the list of timeframes for the exchange {self.name} and this exchange is therefore not supported. ccxt fetchOHLCV: {self.exchange_has("fetchOHLCV")}')
        if timeframe and timeframe not in self.timeframes:
            raise ConfigurationError(f"Invalid timeframe '{timeframe}'. This exchange supports: {self.timeframes}")
        if timeframe and self._config.get('runmode') != RunMode.UTIL_EXCHANGE and (timeframe_to_minutes(timeframe) < 1):
            raise ConfigurationError('Timeframes < 1m are currently not supported by Freqtrade.')

    def validate_ordertypes(self, order_types: Dict[str, Any]) -> None:
        if any((v == 'market' for _, v in order_types.items())):
            if not self.exchange_has('createMarketOrder'):
                raise ConfigurationError(f'Exchange {self.name} does not support market orders.')
        self.validate_stop_ordertypes(order_types)

    def validate_stop_ordertypes(self, order_types: Dict[str, Any]) -> None:
        if order_types.get('stoploss_on_exchange') and (not self._ft_has.get('stoploss_on_exchange', False)):
            raise ConfigurationError(f'On exchange stoploss is not supported for {self.name}.')
        if self.trading_mode == TradingMode.FUTURES:
            price_mapping = self._ft_has.get('stop_price_type_value_mapping', {}).keys()
            if order_types.get('stoploss_on_exchange', False) is True and 'stoploss_price_type' in order_types and (order_types['stoploss_price_type'] not in price_mapping):
                raise ConfigurationError(f'On exchange stoploss price type is not supported for {self.name}.')

    def validate_pricing(self, pricing: Dict[str, Any]) -> None:
        if pricing.get('use_order_book', False) and (not self.exchange_has('fetchL2OrderBook')):
            raise ConfigurationError(f'Orderbook not available for {self.name}.')
        if not pricing.get('use_order_book', False) and (not self.exchange_has('fetchTicker') or not self._ft_has.get('tickers_have_price', False)):
            raise ConfigurationError(f'Ticker pricing not available for {self.name}.')

    def validate_order_time_in_force(self, order_time_in_force: Dict[str, Any]) -> None:
        if any((v.upper() not in self._ft_has.get('order_time_in_force', []) for _, v in order_time_in_force.items())):
            raise ConfigurationError(f'Time in force policies are not supported for {self.name} yet.')

    def validate_orderflow(self, exchange: Dict[str, Any]) -> None:
        if exchange.get('use_public_trades', False) and (not self.exchange_has('fetchTrades') or not self._ft_has.get('trades_has_history', False)):
            raise ConfigurationError(f"Trade data not available for {self.name}. Can't use orderflow feature.")

    def validate_freqai(self, config: Dict[str, Any]) -> None:
        freqai_enabled = config.get('freqai', {}).get('enabled', False)
        if freqai_enabled and (not self._ft_has.get('ohlcv_has_history', False)):
            raise ConfigurationError(f"Historic OHLCV data not available for {self.name}. Can't use freqAI.")

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        candle_limit = self.ohlcv_candle_limit(timeframe, self._config.get('candle_type_def'), dt_ts(date_minus_candles(timeframe, startup_candles)) if timeframe else None)
        candle_count = startup_candles + 1
        required_candle_call_count: int = int(candle_count / candle_limit + (0 if candle_count % candle_limit == 0 else 1))
        if self._ft_has.get('ohlcv_has_history', False):
            if required_candle_call_count > 5:
                raise ConfigurationError(f'This strategy requires {startup_candles} candles to start, which is more than 5x the amount of candles {self.name} provides for {timeframe}.')
        elif required_candle_call_count > 1:
            raise ConfigurationError(f'This strategy requires {startup_candles} candles to start, which is more than the amount of candles {self.name} provides for {timeframe}.')
        if required_candle_call_count > 1:
            logger.warning(f'Using {required_candle_call_count} calls to get OHLCV. This can result in slower operations for the bot. Please check if you really need {startup_candles} candles for your strategy')
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(self, trading_mode: TradingMode, margin_mode: MarginMode) -> None:
        if trading_mode != TradingMode.SPOT and (trading_mode, margin_mode) not in self._supported_trading_mode_margin_pairs:
            mm_value = margin_mode.value if margin_mode else ''
            raise OperationalException(f'Freqtrade does not support {mm_value} {trading_mode} on {self.name}')

    def get_option(self, param: str, default: Any = None) -> Any:
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        if endpoint in self._ft_has.get('exchange_has_overrides', {}):
            return self._ft_has['exchange_has_overrides'][endpoint]
        return endpoint in self._api_async.has and self._api_async.has[endpoint]

    def features(self, market_type: str, endpoint: str, attribute: str, default: Any) -> Any:
        feat = self._api_async.features.get('spot', {}) if market_type == 'spot' else self._api_async.features.get('swap', {}).get('linear', {})
        return feat.get(endpoint, {}).get(attribute, default)

    def get_precision_amount(self, pair: str) -> Optional[int]:
        return self.markets.get(pair, {}).get('precision', {}).get('amount', None)

    def get_precision_price(self, pair: str) -> Optional[int]:
        return self.markets.get(pair, {}).get('precision', {}).get('price', None)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode=ROUND) -> float:
        return price_to_precision(price, self.get_precision_price(pair), self.precision_mode_price, rounding_mode=rounding_mode)

    def price_get_one_pip(self, pair: str, price: float) -> float:
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(self, pair: str, price: float, stoploss: float, leverage: float = 1.0) -> Any:
        return self._get_stake_amount_limit(pair, price, stoploss, 'min', leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> Any:
        max_stake_amount = self._get_stake_amount_limit(pair, price, 0.0, 'max', leverage)
        if max_stake_amount is None:
            raise OperationalException(f'{self.name}.get_max_pair_stake_amount should never set max_stake_amount to None')
        return max_stake_amount

    def _get_stake_amount_limit(self, pair: str, price: float, stoploss: float, limit: str, leverage: float = 1.0) -> Optional[float]:
        isMin: bool = limit == 'min'
        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")
        if isMin:
            margin_reserve: float = 1.0 + self._config.get('amount_reserve_percent', DEFAULT_AMOUNT_RESERVE_PERCENT)
            stoploss_reserve: float = margin_reserve / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
            stoploss_reserve = max(min(stoploss_reserve, 1.5), 1)
        else:
            margin_reserve = 1.0
            stoploss_reserve = 1.0
        stake_limits: List[float] = []
        limits = market['limits']
        if limits['cost'][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits['cost'][limit]) * stoploss_reserve)
        if limits['amount'][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits['amount'][limit]) * price * margin_reserve)
        if not stake_limits:
            return None if isMin else float('inf')
        return self._get_stake_amount_considering_leverage(max(stake_limits) if isMin else min(stake_limits), leverage or 1.0)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        return stake_amount / leverage

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float, params: Optional[dict] = None, stop_loss: bool = False) -> dict:
        now: datetime = dt_now()
        order_id: str = f'dry_run_{side}_{pair}_{now.timestamp()}'
        _amount: float = self._contracts_to_amount(pair, self.amount_to_precision(pair, self._amount_to_contracts(pair, amount)))
        dry_order: dict = {
            'id': order_id, 'symbol': pair, 'price': rate, 'average': rate,
            'amount': _amount, 'cost': _amount * rate, 'type': ordertype,
            'side': side, 'filled': 0, 'remaining': _amount,
            'datetime': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'timestamp': dt_ts(now), 'status': 'open', 'fee': None, 'info': {}
        }
        if stop_loss:
            dry_order['info'] = {'stopPrice': dry_order['price']}
            dry_order[self._ft_has.get('stop_price_prop', '')] = dry_order['price']
            dry_order['ft_order_type'] = 'stoploss'
        orderbook: Optional[dict] = None
        if self.exchange_has('fetchL2OrderBook'):
            orderbook = self.fetch_l2_order_book(pair, 20)
        if ordertype == 'limit' and orderbook:
            allowed_diff: float = 0.01
            if self._dry_is_price_crossed(pair, side, rate, orderbook, allowed_diff):
                logger.info(f'Converted order {pair} to market order due to price {rate} crossing spread by more than {allowed_diff:.2%}.')
                dry_order['type'] = 'market'
        if dry_order['type'] == 'market' and (not dry_order.get('ft_order_type')):
            average = self.get_dry_market_fill_price(pair, side, amount, rate, orderbook)
            dry_order.update({'average': average, 'filled': _amount, 'remaining': 0.0, 'status': 'closed', 'cost': _amount * average})
            dry_order = self.add_dry_order_fee(pair, dry_order, 'taker')
        dry_order = self.check_dry_limit_order_filled(dry_order, immediate=True, orderbook=orderbook)
        self._dry_run_open_orders[dry_order['id']] = dry_order
        return dry_order

    def add_dry_order_fee(self, pair: str, dry_order: dict, taker_or_maker: str) -> dict:
        fee: float = self.get_fee(pair, taker_or_maker=taker_or_maker)
        dry_order.update({'fee': {'currency': self.get_pair_quote_currency(pair), 'cost': dry_order['cost'] * fee, 'rate': fee}})
        return dry_order

    def get_dry_market_fill_price(self, pair: str, side: str, amount: float, rate: float, orderbook: Optional[dict]) -> float:
        if self.exchange_has('fetchL2OrderBook'):
            if not orderbook:
                orderbook = self.fetch_l2_order_book(pair, 20)
            ob_type: str = 'asks' if side == 'buy' else 'bids'
            slippage: float = 0.05
            max_slippage_val: float = rate * (1 + slippage if side == 'buy' else 1 - slippage)
            remaining_amount: float = amount
            filled_value: float = 0.0
            book_entry_price: float = 0.0
            for book_entry in orderbook[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:
                        filled_value += remaining_amount * book_entry_price
                        break
                    else:
                        filled_value += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:
                filled_value += remaining_amount * book_entry_price
            forecast_avg_filled_price: float = max(filled_value, 0) / amount
            if side == 'buy':
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)
            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)
            return self.price_to_precision(pair, forecast_avg_filled_price)
        return rate

    def _dry_is_price_crossed(self, pair: str, side: str, limit: float, orderbook: Optional[dict] = None, offset: float = 0.0) -> bool:
        if not self.exchange_has('fetchL2OrderBook'):
            return True
        if not orderbook:
            orderbook = self.fetch_l2_order_book(pair, 1)
        try:
            if side == 'buy':
                price = orderbook['asks'][0][0]
                if limit * (1 - offset) >= price:
                    return True
            else:
                price = orderbook['bids'][0][0]
                if limit * (1 + offset) <= price:
                    return True
        except IndexError:
            pass
        return False

    def check_dry_limit_order_filled(self, order: dict, immediate: bool = False, orderbook: Optional[dict] = None) -> dict:
        if order['status'] != 'closed' and order['type'] in ['limit'] and (not order.get('ft_order_type')):
            pair: str = order['symbol']
            if self._dry_is_price_crossed(pair, order['side'], order['price'], orderbook):
                order.update({'status': 'closed', 'filled': order['amount'], 'remaining': 0})
                self.add_dry_order_fee(pair, order, 'taker' if immediate else 'maker')
        return order

    def fetch_dry_run_order(self, order_id: str) -> dict:
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            from freqtrade.persistence import Order
            order_obj = Order.order_by_id(order_id)
            if order_obj:
                ccxt_order = order_obj.to_ccxt_object(self._ft_has.get('stop_price_prop', ''))
                self._dry_run_open_orders[order_id] = ccxt_order
                return ccxt_order
            raise InvalidOrderException(f'Tried to get an invalid dry-run-order (id: {order_id}). Message: {e}') from e

    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != TradingMode.SPOT:
            self.set_margin_mode(pair, self.margin_mode, accept_fail)
            self._set_leverage(leverage, pair, accept_fail)

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> dict:
        params: dict = self._params.copy()
        if time_in_force != 'GTC' and ordertype != 'market':
            params.update({'timeInForce': time_in_force.upper()})
        if reduceOnly:
            params.update({'reduceOnly': True})
        return params

    def _order_needs_price(self, side: str, ordertype: str) -> bool:
        return ordertype != 'market' or (side == 'buy' and self._api.options.get('createMarketBuyOrderRequiresPrice', False)) or self._ft_has.get('marketOrderRequiresPrice', False)

    def create_order(
        self, *, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float,
        reduceOnly: bool = False, time_in_force: str = 'GTC'
    ) -> dict:
        if self._config.get('dry_run'):
            dry_order = self.create_dry_run_order(pair, ordertype, side, amount, self.price_to_precision(pair, rate), leverage)
            return dry_order
        params = self._get_params(side, ordertype, leverage, reduceOnly, time_in_force)
        try:
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            needs_price = self._order_needs_price(side, ordertype)
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None
            if not reduceOnly:
                self._lev_prep(pair, leverage, side)
            order = self._api.create_order(pair, ordertype, side, amount, rate_for_order, params)
            if order.get('status') is None:
                order['status'] = 'open'
            if order.get('type') is None:
                order['type'] = ordertype
            self._log_exchange_response('create_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(f'Insufficient funds to create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {rate}.Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Could not create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {rate}. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def stoploss_adjust(self, stop_loss: float, order: dict, side: str) -> bool:
        if not self._ft_has.get('stoploss_on_exchange'):
            raise OperationalException(f'stoploss is not implemented for {self.name}.')
        price_param = self._ft_has.get('stop_price_prop', '')
        return order.get(price_param, None) is None or (side == 'sell' and stop_loss > float(order[price_param]) or (side == 'buy' and stop_loss < float(order[price_param])))

    def _get_stop_order_type(self, user_order_type: str) -> Tuple[str, str]:
        available_order_Types = self._ft_has['stoploss_order_types']
        if user_order_type in available_order_Types.keys():
            ordertype = available_order_Types[user_order_type]
        else:
            ordertype = list(available_order_Types.values())[0]
            user_order_type = list(available_order_Types.keys())[0]
        return (ordertype, user_order_type)

    def _get_stop_limit_rate(self, stop_price: float, order_types: dict, side: str) -> float:
        limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
        if side == 'sell':
            limit_rate = stop_price * limit_price_pct
        else:
            limit_rate = stop_price * (2 - limit_price_pct)
        bad_stop_price = stop_price < limit_rate if side == 'sell' else stop_price > limit_rate
        if bad_stop_price:
            raise InvalidOrderException(f'In stoploss limit order, stop price should be more than limit price. Stop price: {stop_price}, Limit price: {limit_rate}, Limit Price pct: {limit_price_pct}')
        return limit_rate

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> dict:
        params = self._params.copy()
        params.update({self._ft_has.get('stop_price_param', ''): stop_price})
        return params

    @retrier(retries=0)
    def create_stoploss(self, pair: str, amount: float, stop_price: float, order_types: dict, side: str, leverage: float) -> dict:
        if not self._ft_has.get('stoploss_on_exchange'):
            raise OperationalException(f'stoploss is not implemented for {self.name}.')
        user_order_type = order_types.get('stoploss', 'market')
        ordertype, user_order_type = self._get_stop_order_type(user_order_type)
        round_mode = ROUND_DOWN if side == 'buy' else ROUND_UP
        stop_price_norm = self.price_to_precision(pair, stop_price, rounding_mode=round_mode)
        limit_rate: Optional[float] = None
        if user_order_type == 'limit':
            limit_rate = self._get_stop_limit_rate(stop_price, order_types, side)
            limit_rate = self.price_to_precision(pair, limit_rate, rounding_mode=round_mode)
        if self._config.get('dry_run'):
            dry_order = self.create_dry_run_order(pair, ordertype, side, amount, stop_price_norm, stop_loss=True, leverage=leverage)
            return dry_order
        try:
            params = self._get_stop_params(side=side, ordertype=ordertype, stop_price=stop_price_norm)
            if self.trading_mode == TradingMode.FUTURES:
                params['reduceOnly'] = True
                if 'stoploss_price_type' in order_types and 'stop_price_type_field' in self._ft_has:
                    price_type = self._ft_has['stop_price_type_value_mapping'][order_types.get('stoploss_price_type', PriceType.LAST)]
                    params[self._ft_has['stop_price_type_field']] = price_type
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            self._lev_prep(pair, leverage, side, accept_fail=True)
            order = self._api.create_order(symbol=pair, type=ordertype, side=side, amount=amount, price=limit_rate, params=params)
            self._log_exchange_response('create_stoploss_order', order)
            order = self._order_contracts_to_amount(order)
            logger.info(f'stoploss {user_order_type} order added for {pair}. stop price: {stop_price}. limit: {limit_rate}')
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(f'Insufficient funds to create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {limit_rate} with stop-price {stop_price_norm}. Message: {e}') from e
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationRejected) as e:
            raise InvalidOrderException(f'Could not create {ordertype} {side} order on market {pair}. Tried to {side} amount {amount} at rate {limit_rate} with stop-price {stop_price_norm}. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not place stoploss order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_order_emulated(self, order_id: str, pair: str, params: dict) -> dict:
        try:
            order = self._api.fetch_open_order(order_id, pair, params=params)
            self._log_exchange_response('fetch_open_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound:
            try:
                order = self._api.fetch_closed_order(order_id, pair, params=params)
                self._log_exchange_response('fetch_closed_order', order)
                order = self._order_contracts_to_amount(order)
                return order
            except ccxt.OrderNotFound as e:
                raise RetryableOrderError(f'Order not found (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Tried to get an invalid order (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str, params: Optional[dict] = None) -> dict:
        if self._config.get('dry_run'):
            return self.fetch_dry_run_order(order_id)
        if params is None:
            params = {}
        try:
            if not self.exchange_has('fetchOrder'):
                return self.fetch_order_emulated(order_id, pair, params)
            order = self._api.fetch_order(order_id, pair, params=params)
            self._log_exchange_response('fetch_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(f'Order not found (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Tried to get an invalid order (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[dict] = None) -> dict:
        return self.fetch_order(order_id, pair, params)

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str, stoploss_order: bool = False) -> dict:
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: dict) -> bool:
        return order.get('status') in NON_OPEN_EXCHANGE_STATES and order.get('filled') == 0.0

    @retrier
    def cancel_order(self, order_id: str, pair: str, params: Optional[dict] = None) -> dict:
        if self._config.get('dry_run'):
            try:
                order = self.fetch_dry_run_order(order_id)
                order.update({'status': 'canceled', 'filled': 0.0, 'remaining': order['amount']})
                return order
            except InvalidOrderException:
                return {}
        if params is None:
            params = {}
        try:
            order = self._api.cancel_order(order_id, pair, params=params)
            self._log_exchange_response('cancel_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f'Could not cancel order. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[dict] = None) -> dict:
        return self.cancel_order(order_id, pair, params)

    def is_cancel_order_result_suitable(self, corder: Any) -> bool:
        if not isinstance(corder, dict):
            return False
        required = ('fee', 'status', 'amount')
        return all((corder.get(k, None) is not None for k in required))

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> dict:
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            logger.warning(f'Could not cancel order {order_id} for {pair}.')
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f'Could not fetch cancelled order {order_id}.')
            order = {'id': order_id, 'status': 'canceled', 'amount': amount, 'filled': 0.0, 'fee': {}, 'info': {}}
        return order

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> dict:
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f'Could not fetch cancelled stoploss order {order_id}.')
            order = {'id': order_id, 'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}
        return order

    @retrier
    def get_balances(self) -> dict:
        try:
            balances = self._api.fetch_balance()
            balances.pop('info', None)
            balances.pop('free', None)
            balances.pop('total', None)
            balances.pop('used', None)
            self._log_exchange_response('fetch_balances', balances)
            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_positions(self, pair: Optional[str] = None) -> List[Any]:
        if self._config.get('dry_run') or self.trading_mode != TradingMode.FUTURES:
            return []
        try:
            symbols: List[str] = []
            if pair:
                symbols.append(pair)
            positions = self._api.fetch_positions(symbols)
            self._log_exchange_response('fetch_positions', positions)
            return positions
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get positions due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> List[Any]:
        orders: List[Any] = []
        if self.exchange_has('fetchClosedOrders'):
            orders = self._api.fetch_closed_orders(pair, since=since_ms)
            if self.exchange_has('fetchOpenOrders'):
                orders_open = self._api.fetch_open_orders(pair, since=since_ms)
                orders.extend(orders_open)
        return orders

    @retrier(retries=0)
    def fetch_orders(self, pair: str, since: datetime, params: Optional[dict] = None) -> List[Any]:
        if self._config.get('dry_run'):
            return []
        try:
            since_ms: int = int((since.timestamp() - 10) * 1000)
            if self.exchange_has('fetchOrders'):
                if not params:
                    params = {}
                try:
                    orders = self._api.fetch_orders(pair, since=since_ms, params=params)
                except ccxt.NotSupported:
                    orders = self._fetch_orders_emulate(pair, since_ms)
            else:
                orders = self._fetch_orders_emulate(pair, since_ms)
            self._log_exchange_response('fetch_orders', orders)
            orders = [self._order_contracts_to_amount(o) for o in orders]
            return orders
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch positions due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_trading_fees(self) -> dict:
        if self._config.get('dry_run') or self.trading_mode != TradingMode.FUTURES or (not self.exchange_has('fetchTradingFees')):
            return {}
        try:
            trading_fees = self._api.fetch_trading_fees()
            self._log_exchange_response('fetch_trading_fees', trading_fees)
            return trading_fees
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch trading fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_bids_asks(self, symbols: Optional[List[str]] = None, *, cached: bool = False) -> dict:
        if not self.exchange_has('fetchBidsAsks'):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_bids_asks')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_bids_asks(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache['fetch_bids_asks'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching bids/asks in batch. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load bids/asks due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self, symbols: Optional[List[str]] = None, *, cached: bool = False, market_type: Optional[str] = None) -> dict:
        if not self.exchange_has('fetchTickers'):
            return {}
        cache_key = f'fetch_tickers_{market_type}' if market_type else 'fetch_tickers'
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get(cache_key)
            if tickers:
                return tickers
        try:
            market_types = {TradingMode.FUTURES: 'swap'}
            params: dict = {'type': market_types.get(market_type, market_type)} if market_type else {}
            tickers = self._api.fetch_tickers(symbols, params)
            with self._cache_lock:
                self._fetch_tickers_cache[cache_key] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching tickers in batch. Message: {e}') from e
        except ccxt.BadSymbol as e:
            logger.warning(f'Could not load tickers due to {e.__class__.__name__}. Message: {e} .Reloading markets.')
            self.reload_markets(True)
            raise TemporaryError from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load tickers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_proxy_coin(self) -> str:
        return self._config['stake_currency']

    def get_conversion_rate(self, coin: str, currency: str) -> Optional[float]:
        if (proxy_coin := self._ft_has['proxy_coin_mapping'].get(coin, None)) is not None:
            coin = proxy_coin
        if (proxy_currency := self._ft_has['proxy_coin_mapping'].get(currency, None)) is not None:
            currency = proxy_currency
        if coin == currency:
            return 1.0
        tickers = self.get_tickers(cached=True)
        try:
            for pair in self.get_valid_pair_combination(coin, currency):
                ticker = tickers.get(pair, None)
                if not ticker:
                    tickers_other = self.get_tickers(cached=True, market_type=TradingMode.SPOT if self.trading_mode != TradingMode.SPOT else TradingMode.FUTURES)
                    ticker = tickers_other.get(pair, None)
                if ticker:
                    rate = safe_value_fallback2(ticker, ticker, 'last', 'ask', None)
                    if rate and pair.startswith(currency) and (not pair.endswith(currency)):
                        rate = 1.0 / rate
                    return rate
        except ValueError:
            return None
        return None

    @retrier
    def fetch_ticker(self, pair: str) -> dict:
        try:
            if pair not in self.markets or self.markets[pair].get('active', False) is False:
                raise ExchangeError(f'Pair {pair} not available')
            data = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load ticker due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[List[int]], range_required: bool = True) -> Optional[int]:
        if not limit_range:
            return limit
        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict:
        limit1 = self.get_next_limit_in_list(limit, self._ft_has.get('l2_limit_range'), self._ft_has.get('l2_limit_range_required', True))
        try:
            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching order book. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get order book due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_price_side(self, side: str, is_short: bool, conf_strategy: dict) -> str:
        price_side = conf_strategy['price_side']
        if price_side in ('same', 'other'):
            price_map = {
                ('entry', 'long', 'same'): 'bid', ('entry', 'long', 'other'): 'ask', 
                ('entry', 'short', 'same'): 'ask', ('entry', 'short', 'other'): 'bid', 
                ('exit', 'long', 'same'): 'ask', ('exit', 'long', 'other'): 'bid', 
                ('exit', 'short', 'same'): 'bid', ('exit', 'short', 'other'): 'ask'
            }
            price_side = price_map[(side, 'short' if is_short else 'long', price_side)]
        return price_side

    def get_rate(self, pair: str, refresh: bool, side: str, is_short: bool, order_book: Optional[dict] = None, ticker: Optional[dict] = None) -> float:
        name: str = side.capitalize()
        strat_name: str = 'entry_pricing' if side == 'entry' else 'exit_pricing'
        cache_rate: TTLCache = self._entry_rate_cache if side == 'entry' else self._exit_rate_cache
        if not refresh:
            with self._cache_lock:
                rate_cached = cache_rate.get(pair)
            if rate_cached:
                logger.debug(f'Using cached {side} rate for {pair}.')
                return rate_cached
        conf_strategy = self._config.get(strat_name, {})
        price_side = self._get_price_side(side, is_short, conf_strategy)
        if conf_strategy.get('use_order_book', False):
            order_book_top = conf_strategy.get('order_book_top', 1)
            if order_book is None:
                order_book = self.fetch_l2_order_book(pair, order_book_top)
            rate = self._get_rate_from_ob(pair, side, order_book, name, price_side, order_book_top)
        else:
            logger.debug(f'Using Last {price_side.capitalize()} / Last Price')
            if ticker is None:
                ticker = self.fetch_ticker(pair)
            rate = self._get_rate_from_ticker(side, ticker, conf_strategy, price_side)
        if rate is None:
            raise PricingError(f'{name}-Rate for {pair} was empty.')
        with self._cache_lock:
            cache_rate[pair] = rate
        return rate

    def _get_rate_from_ticker(self, side: str, ticker: dict, conf_strategy: dict, price_side: str) -> float:
        ticker_rate = ticker[price_side]
        if ticker['last'] and ticker_rate:
            if side == 'entry' and ticker_rate > ticker['last']:
                balance = conf_strategy.get('price_last_balance', 0.0)
                ticker_rate = ticker_rate + balance * (ticker['last'] - ticker_rate)
            elif side == 'exit' and ticker_rate < ticker['last']:
                balance = conf_strategy.get('price_last_balance', 0.0)
                ticker_rate = ticker_rate - balance * (ticker_rate - ticker['last'])
        return ticker_rate

    def _get_rate_from_ob(self, pair: str, side: str, order_book: dict, name: str, price_side: str, order_book_top: int) -> float:
        logger.debug('order_book %s', order_book)
        try:
            obside: str = 'bids' if price_side == 'bid' else 'asks'
            rate = order_book[obside][order_book_top - 1][0]
        except (IndexError, KeyError) as e:
            logger.warning(f'{pair} - {name} Price at location {order_book_top} from orderbook could not be determined. Orderbook: {order_book}')
            raise PricingError from e
        logger.debug(f'{pair} - {name} price from orderbook {price_side.capitalize()}side - top {order_book_top} order book {side} rate {rate:.8f}')
        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        entry_rate: Optional[float] = None
        exit_rate: Optional[float] = None
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate:
                logger.debug(f'Using cached buy rate for {pair}.')
            if exit_rate:
                logger.debug(f'Using cached sell rate for {pair}.')
        entry_pricing = self._config.get('entry_pricing', {})
        exit_pricing = self._config.get('exit_pricing', {})
        order_book: Optional[dict] = None
        ticker: Optional[dict] = None
        if not entry_rate and entry_pricing.get('use_order_book', False):
            order_book_top = max(entry_pricing.get('order_book_top', 1), exit_pricing.get('order_book_top', 1))
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, order_book=order_book)
        elif not entry_rate:
            ticker = self.fetch_ticker(pair)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, ticker=ticker)
        if not exit_rate:
            exit_rate = self.get_rate(pair, refresh, 'exit', is_short, order_book=order_book, ticker=ticker)
        return (entry_rate, exit_rate)

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime, params: Optional[dict] = None) -> List[dict]:
        if self._config.get('dry_run'):
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            _params = params if params else {}
            my_trades = self._api.fetch_my_trades(pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000), params=_params)
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]
            self._log_exchange_response('get_trades_for_order', matched_trades)
            matched_trades = self._trades_contracts_to_amount(matched_trades)
            return matched_trades
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get trades due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_order_id_conditional(self, order: dict) -> Any:
        return order['id']

    @retrier
    def get_fee(self, symbol: str, order_type: str = '', side: str = '', amount: float = 1, price: float = 1, taker_or_maker: str = 'maker') -> float:
        if order_type and order_type == 'market':
            taker_or_maker = 'taker'
        try:
            if self._config.get('dry_run') and self._config.get('fee', None) is not None:
                return self._config['fee']
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets(params={})
            fee = self._api.calculate_fee(symbol=symbol, type=order_type, side=side, amount=amount, price=price, takerOrMaker=taker_or_maker)
            return fee['rate']
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get fee info due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def order_has_fee(order: Any) -> bool:
        if not isinstance(order, dict):
            return False
        return 'fee' in order and order['fee'] is not None and (order['fee'].keys() >= {'currency', 'cost'}) and (order['fee']['currency'] is not None) and (order['fee']['cost'] is not None)

    def calculate_fee_rate(self, fee: dict, symbol: str, cost: float, amount: float) -> Optional[float]:
        if fee.get('rate') is not None:
            return fee.get('rate')
        fee_curr = fee.get('currency')
        if fee_curr is None:
            return None
        fee_cost = float(fee['cost'])
        if fee_curr == self.get_pair_base_currency(symbol):
            return round(fee_cost / amount, 8)
        elif fee_curr == self.get_pair_quote_currency(symbol):
            return round(fee_cost / cost, 8) if cost else None
        else:
            if not cost:
                return None
            try:
                fee_to_quote_rate = self.get_conversion_rate(fee_curr, self._config['stake_currency'])
                if not fee_to_quote_rate:
                    raise ValueError('Conversion rate not found.')
            except (ValueError, ExchangeError):
                fee_to_quote_rate = self._config['exchange'].get('unknown_fee_rate', None)
                if not fee_to_quote_rate:
                    return None
            return round(fee_cost * fee_to_quote_rate / cost, 8)

    def extract_cost_curr_rate(self, fee: dict, symbol: str, cost: float, amount: float) -> Tuple[float, str, Optional[float]]:
        return (float(fee['cost']), fee['currency'], self.calculate_fee_rate(fee, symbol, cost, amount))

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool = False, until_ms: Optional[int] = None) -> DataFrame:
        with self._loop_lock:
            pair_ret, _, _, data, _ = self.loop.run_until_complete(self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=since_ms, until_ms=until_ms, candle_type=candle_type))
        logger.debug(f'Downloaded data for {pair} from ccxt with length {len(data)}.')
        return ohlcv_to_dataframe(data, timeframe, pair, fill_missing=False, drop_incomplete=True)

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, raise_: bool = False, until_ms: Optional[int] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
        logger.debug('one_call: %s msecs (%s)', one_call, dt_humanize_delta(dt_now() - timedelta(milliseconds=one_call)))
        input_coroutines = [self._async_get_candle_history(pair, timeframe, candle_type, since) for since in range(since_ms, until_ms or dt_ts(), one_call)]
        data: List[List[Any]] = []
        for input_coro in chunks(input_coroutines, 100):
            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning(f'Async code raised an exception: {repr(res)}')
                    if raise_:
                        raise res
                    continue
                else:
                    p, _, c, new_data, _ = res
                    if p == pair and c == candle_type:
                        data.extend(new_data)
        data = sorted(data, key=lambda x: x[0])
        return (pair, timeframe, candle_type, data, self._ohlcv_partial_candle)

    def _build_coroutine(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: Optional[int], cache: bool) -> Any:
        not_all_data = cache and self.required_candle_call_count > 1
        if cache and candle_type in (CandleType.SPOT, CandleType.FUTURES):
            if self._has_watch_ohlcv and self._exchange_ws:
                self._exchange_ws.schedule_ohlcv(pair, timeframe, candle_type)
        if cache and (pair, timeframe, candle_type) in self._klines:
            candle_limit = self.ohlcv_candle_limit(timeframe, candle_type)
            min_ts = dt_ts(date_minus_candles(timeframe, candle_limit - 5))
            if self._exchange_ws:
                candle_ts = dt_ts(timeframe_to_prev_date(timeframe))
                prev_candle_ts = dt_ts(date_minus_candles(timeframe, 1))
                candles = self._exchange_ws.ohlcvs(pair, timeframe)
                half_candle = int(candle_ts - (candle_ts - prev_candle_ts) * 0.5)
                last_refresh_time = int(self._exchange_ws.klines_last_refresh.get((pair, timeframe, candle_type), 0))
                if candles and candles[-1][0] >= prev_candle_ts and (last_refresh_time >= half_candle):
                    logger.debug(f'reuse watch result for {pair}, {timeframe}, {last_refresh_time}')
                    return self._exchange_ws.get_ohlcv(pair, timeframe, candle_type, candle_ts)
                logger.info(f'Failed to reuse watch {pair}, {timeframe}, {candle_ts < last_refresh_time}, {candle_ts}, {last_refresh_time}, {format_ms_time(candle_ts)}, {format_ms_time(last_refresh_time)} ')
            if min_ts < self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0):
                not_all_data = False
            else:
                logger.info(f'Time jump detected. Evicting cache for {pair}, {timeframe}, {candle_type}')
                del self._klines[pair, timeframe, candle_type]
        if not since_ms and (self._ft_has.get('ohlcv_require_since') or not_all_data):
            one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
            move_to = one_call * self.required_candle_call_count
            now = timeframe_to_next_date(timeframe)
            since_ms = dt_ts(now - timedelta(seconds=move_to // 1000))
        if since_ms:
            return self._async_get_historic_ohlcv(pair, timeframe, since_ms=since_ms, raise_=True, candle_type=candle_type)
        else:
            return self._async_get_candle_history(pair, timeframe, candle_type, since_ms=since_ms)

    def _build_ohlcv_dl_jobs(self, pair_list: List[Tuple[Any, Any, Any]], since_ms: int, cache: bool) -> Tuple[List[Any], List[Tuple[Any, Any, Any]]]:
        input_coroutines: List[Any] = []
        cached_pairs: List[Tuple[Any, Any, Any]] = []
        for pair, timeframe, candle_type in set(pair_list):
            if timeframe not in self.timeframes and candle_type in (CandleType.SPOT, CandleType.FUTURES):
                logger.warning(f'Cannot download ({pair}, {timeframe}) combination as this timeframe is not available on {self.name}. Available timeframes are {", ".join(self.timeframes)}.')
                continue
            if (pair, timeframe, candle_type) not in self._klines or not cache or self._now_is_time_to_refresh(pair, timeframe, candle_type):
                input_coroutines.append(self._build_coroutine(pair, timeframe, candle_type, since_ms, cache))
            else:
                logger.debug(f'Using cached candle (OHLCV) data for {pair}, {timeframe}, {candle_type} ...')
                cached_pairs.append((pair, timeframe, candle_type))
        return (input_coroutines, cached_pairs)

    def _process_ohlcv_df(self, pair: str, timeframe: str, c_type: Any, ticks: List[Any], cache: bool, drop_incomplete: bool) -> DataFrame:
        if ticks and cache:
            idx = -2 if drop_incomplete and len(ticks) > 1 else -1
            self._pairs_last_refresh_time[pair, timeframe, c_type] = ticks[idx][0]
        ohlcv_df = ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True, drop_incomplete=drop_incomplete)
        if cache:
            if (pair, timeframe, c_type) in self._klines:
                old = self._klines[pair, timeframe, c_type]
                ohlcv_df = clean_ohlcv_dataframe(concat([old, ohlcv_df], axis=0), timeframe, pair, fill_missing=True, drop_incomplete=False)
                candle_limit = self.ohlcv_candle_limit(timeframe, self._config.get('candle_type_def'))
                ohlcv_df = ohlcv_df.tail(candle_limit + self._startup_candle_count)
                ohlcv_df = ohlcv_df.reset_index(drop=True)
                self._klines[pair, timeframe, c_type] = ohlcv_df
            else:
                self._klines[pair, timeframe, c_type] = ohlcv_df
        return ohlcv_df

    def refresh_latest_ohlcv(self, pair_list: List[Tuple[Any, Any, Any]], *, since_ms: Optional[int] = None, cache: bool = True, drop_incomplete: Optional[bool] = None) -> Dict[Tuple[Any, Any, Any], DataFrame]:
        logger.debug('Refreshing candle (OHLCV) data for %d pairs', len(pair_list))
        ohlcv_dl_jobs, cached_pairs = self._build_ohlcv_dl_jobs(pair_list, since_ms if since_ms is not None else 0, cache)
        results_df: Dict[Tuple[Any, Any, Any], DataFrame] = {}
        for dl_jobs_batch in chunks(ohlcv_dl_jobs, 100):

            async def gather_coroutines(coro: List[Any]) -> List[Any]:
                return await asyncio.gather(*coro, return_exceptions=True)
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_jobs_batch))
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f'Async code raised an exception: {repr(res)}')
                    continue
                pair_ret, timeframe_ret, c_type_ret, ticks, drop_hint = res
                drop_incomplete_ = drop_hint if drop_incomplete is None else drop_incomplete
                ohlcv_df = self._process_ohlcv_df(pair_ret, timeframe_ret, c_type_ret, ticks, cache, drop_incomplete_)
                results_df[(pair_ret, timeframe_ret, c_type_ret)] = ohlcv_df
        for pair, timeframe, c_type in cached_pairs:
            results_df[(pair, timeframe, c_type)] = self.klines((pair, timeframe, c_type), copy=False)
        return results_df

    def refresh_ohlcv_with_cache(self, pairs: List[Any], since_ms: int) -> Dict[Any, Any]:
        timeframes = {p[1] for p in pairs}
        for timeframe in timeframes:
            if (timeframe, since_ms) not in self._expiring_candle_cache:
                timeframe_in_sec = timeframe_to_seconds(timeframe)
                self._expiring_candle_cache[(timeframe, since_ms)] = PeriodicCache(ttl=timeframe_in_sec, maxsize=1000)
        candles = {c: self._expiring_candle_cache[(c[1], since_ms)].get(c, None) for c in pairs if c in self._expiring_candle_cache[(c[1], since_ms)]}
        pairs_to_download = [p for p in pairs if p not in candles]
        if pairs_to_download:
            candles_downloaded = self.refresh_latest_ohlcv(pairs_to_download, since_ms=since_ms, cache=False)
            for c, val in candles_downloaded.items():
                self._expiring_candle_cache[(c[1], since_ms)][c] = val
                candles[c] = val
        return candles

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: Any) -> bool:
        interval_in_sec = timeframe_to_msecs(timeframe)
        plr = self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0) + interval_in_sec
        now = dt_ts(timeframe_to_prev_date(timeframe))
        return plr < now

    @retrier_async
    async def _async_get_candle_history(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        try:
            s = '(' + dt_from_ts(since_ms).isoformat() + ') ' if since_ms is not None else ''
            logger.debug('Fetching pair %s, %s, interval %s, since %s %s...', pair, candle_type, timeframe, since_ms, s)
            params = deepcopy(self._ft_has.get('ohlcv_params', {}))
            candle_limit = self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
            if candle_type and candle_type != CandleType.SPOT:
                params.update({'price': candle_type.value})
            if candle_type != CandleType.FUNDING_RATE:
                data = await self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=candle_limit, params=params)
            else:
                data = await self._fetch_funding_rate_history(pair=pair, timeframe=timeframe, limit=candle_limit, since_ms=since_ms)
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception('Error loading %s. Result was %s.', pair, data)
                return (pair, timeframe, candle_type, [], self._ohlcv_partial_candle)
            logger.debug('Done fetching pair %s, %s interval %s...', pair, candle_type, timeframe)
            return (pair, timeframe, candle_type, data, self._ohlcv_partial_candle)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does not support fetching historical candle (OHLCV) data. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch historical candle (OHLCV) data for {pair}, {timeframe}, {candle_type} due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch historical candle (OHLCV) data for {pair}, {timeframe}, {candle_type}. Message: {e}') from e

    async def _fetch_funding_rate_history(self, pair: str, timeframe: str, limit: int, since_ms: Optional[int] = None) -> List[List[Any]]:
        data = await self._api_async.fetch_funding_rate_history(pair, since=since_ms, limit=limit)
        data = [[x['timestamp'], x['fundingRate'], 0, 0, 0, 0] for x in data]
        return data

    def needed_candle_for_trades_ms(self, timeframe: str, candle_type: CandleType) -> int:
        candle_limit = self.ohlcv_candle_limit(timeframe, candle_type)
        tf_s = timeframe_to_seconds(timeframe)
        candles_fetched = candle_limit * self.required_candle_call_count
        max_candles = self._config['orderflow']['max_candles']
        required_candles = min(max_candles, candles_fetched)
        move_to = tf_s * candle_limit * required_candles if required_candles > candle_limit else (max_candles + 1) * tf_s
        now = timeframe_to_next_date(timeframe)
        return int((now - timedelta(seconds=move_to)).timestamp() * 1000)

    def _process_trades_df(self, pair: str, timeframe: str, c_type: Any, ticks: List[Any], cache: bool, first_required_candle_date: int) -> DataFrame:
        trades_df = trades_list_to_df(ticks, True)
        if cache:
            if (pair, timeframe, c_type) in self._trades:
                old = self._trades[pair, timeframe, c_type]
                combined_df = concat([old, trades_df], axis=0)
                logger.debug(f'Clean duplicated ticks from Trades data {pair}')
                trades_df = DataFrame(trades_df_remove_duplicates(combined_df), columns=combined_df.columns)
                trades_df = trades_df[first_required_candle_date < trades_df['timestamp']]
                trades_df = trades_df.reset_index(drop=True)
            self._trades[pair, timeframe, c_type] = trades_df
        return trades_df

    async def _build_trades_dl_jobs(self, pairwt: Tuple[str, str, Any], data_handler: Any, cache: bool) -> Tuple[Tuple[str, str, Any], Optional[DataFrame]]:
        pair, timeframe, candle_type = pairwt
        since_ms: Optional[int] = None
        new_ticks: List[Any] = []
        all_stored_ticks_df: DataFrame = DataFrame(columns=DEFAULT_TRADES_COLUMNS + ['date'])
        first_candle_ms = self.needed_candle_for_trades_ms(timeframe, candle_type)
        is_in_cache = (pair, timeframe, candle_type) in self._trades
        if not is_in_cache or not cache or self._now_is_time_to_refresh_trades(pair, timeframe, candle_type):
            logger.debug(f'Refreshing TRADES data for {pair}')
            try:
                until: Optional[int] = None
                from_id: Optional[Any] = None
                if is_in_cache:
                    from_id = self._trades[pair, timeframe, candle_type].iloc[-1]['id']
                    until = dt_ts()
                else:
                    until = int(timeframe_to_prev_date(timeframe).timestamp()) * 1000
                    all_stored_ticks_df = data_handler.trades_load(f'{pair}-cached', self.trading_mode)
                    if not all_stored_ticks_df.empty:
                        if all_stored_ticks_df.iloc[-1]['timestamp'] > first_candle_ms and all_stored_ticks_df.iloc[0]['timestamp'] <= first_candle_ms:
                            last_cached_ms = all_stored_ticks_df.iloc[-1]['timestamp']
                            from_id = all_stored_ticks_df.iloc[-1]['id']
                            since_ms = last_cached_ms if last_cached_ms > first_candle_ms else first_candle_ms
                        else:
                            all_stored_ticks_df = DataFrame(columns=DEFAULT_TRADES_COLUMNS + ['date'])
                [_, new_ticks] = await self._async_get_trade_history(pair, since=since_ms if since_ms else first_candle_ms, until=until, from_id=from_id)
            except Exception:
                logger.exception(f'Refreshing TRADES data for {pair} failed')
                return (pairwt, None)
            if new_ticks:
                all_stored_ticks_list = list(all_stored_ticks_df[DEFAULT_TRADES_COLUMNS].values.tolist())
                all_stored_ticks_list.extend(new_ticks)
                trades_df = self._process_trades_df(pair, timeframe, candle_type, all_stored_ticks_list, cache, first_required_candle_date=first_candle_ms)
                data_handler.trades_store(f'{pair}-cached', trades_df[DEFAULT_TRADES_COLUMNS], self.trading_mode)
                return (pairwt, trades_df)
            else:
                logger.error(f'No new ticks for {pair}')
        return (pairwt, None)

    def refresh_latest_trades(self, pair_list: List[Tuple[str, str, Any]], *, cache: bool = True) -> Dict[Tuple[str, str, Any], DataFrame]:
        from freqtrade.data.history import get_datahandler
        data_handler = get_datahandler(self._config['datadir'], data_format=self._config['dataformat_trades'])
        logger.debug('Refreshing TRADES data for %d pairs', len(pair_list))
        results_df: Dict[Tuple[str, str, Any], DataFrame] = {}
        trades_dl_jobs: List[Any] = []
        for pair_wt in set(pair_list):
            trades_dl_jobs.append(self._build_trades_dl_jobs(pair_wt, data_handler, cache))
        async def gather_coroutines(coro: List[Any]) -> List[Any]:
            return await asyncio.gather(*coro, return_exceptions=True)
        for dl_job_chunk in chunks(trades_dl_jobs, 100):
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_job_chunk))
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f'Async code raised an exception: {repr(res)}')
                    continue
                pairwt, trades_df = res
                if trades_df is not None:
                    results_df[pairwt] = trades_df
        return results_df

    def _now_is_time_to_refresh_trades(self, pair: str, timeframe: str, candle_type: Any) -> bool:
        trades_df: DataFrame = self.trades((pair, timeframe, candle_type), False)
        pair_last_refreshed = int(trades_df.iloc[-1]['timestamp'])
        full_candle = int(timeframe_to_next_date(timeframe, dt_from_ts(pair_last_refreshed)).timestamp()) * 1000
        now = dt_ts()
        return full_candle <= now

    @retrier_async
    async def _async_fetch_trades(self, pair: str, since: Optional[int] = None, params: Optional[dict] = None) -> Tuple[List[dict], Any]:
        try:
            trades_limit = self._max_trades_limit
            if params:
                logger.debug('Fetching trades for pair %s, params: %s ', pair, params)
                trades = await self._api_async.fetch_trades(pair, params=params, limit=trades_limit)
            else:
                logger.debug('Fetching trades for pair %s, since %s %s...', pair, since, '(' + dt_from_ts(since).isoformat() + ') ' if since is not None else '')
                trades = await self._api_async.fetch_trades(pair, since=since, limit=trades_limit)
            trades = self._trades_contracts_to_amount(trades)
            pagination_value = self._get_trade_pagination_next_value(trades)
            return (trades_dict_to_list(trades), pagination_value)
        except ccxt.NotSupported as e:
            raise OperationalException(f'Exchange {self._api.name} does support fetching historical trade data.Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    def _valid_trade_pagination_id(self, pair: str, from_id: Any) -> bool:
        return True

    def _get_trade_pagination_next_value(self, trades: List[dict]) -> Any:
        if not trades:
            return None
        if self._trades_pagination == 'id':
            return trades[-1].get('id')
        else:
            return trades[-1].get('timestamp')

    async def _async_get_trade_history_id_startup(self, pair: str, since: int) -> Tuple[List[dict], Any]:
        return await self._async_fetch_trades(pair, since=since)

    async def _async_get_trade_history_id(self, pair: str, *, until: int, since: int, from_id: Optional[Any] = None) -> Tuple[str, List[dict]]:
        trades: List[dict] = []
        has_overlap = self._ft_has.get('trades_pagination_overlap', True)
        x = slice(None, -1) if has_overlap else slice(None)
        if not from_id or not self._valid_trade_pagination_id(pair, from_id):
            t, from_id = await self._async_get_trade_history_id_startup(pair, since=since)
            trades.extend(t[x])
        while True:
            try:
                t, from_id_next = await self._async_fetch_trades(pair, params={self._trades_pagination_arg: from_id})
                if t:
                    trades.extend(t[x])
                    if from_id == from_id_next or t[-1][0] > until:
                        logger.debug(f'Stopping because from_id did not change. Reached {t[-1][0]} > {until}')
                        if has_overlap:
                            trades.extend(t[-1:])
                        break
                    from_id = from_id_next
                else:
                    logger.debug('Stopping as no more trades were returned.')
                    break
            except asyncio.CancelledError:
                logger.debug('Async operation Interrupted, breaking trades DL loop.')
                break
        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int, since: int) -> Tuple[str, List[dict]]:
        trades: List[dict] = []
        while True:
            try:
                t, since_next = await self._async_fetch_trades(pair, since=since)
                if t:
                    if since == since_next and len(t) == 1:
                        logger.debug('Stopping because no more trades are available.')
                        break
                    since = since_next
                    trades.extend(t)
                    if until and since_next > until:
                        logger.debug(f'Stopping because until was reached. {since_next} > {until}')
                        break
                else:
                    logger.debug('Stopping as no more trades were returned.')
                    break
            except asyncio.CancelledError:
                logger.debug('Async operation Interrupted, breaking trades DL loop.')
                break
        return (pair, trades)

    async def _async_get_trade_history(self, pair: str, since: int, until: Optional[int] = None, from_id: Optional[Any] = None) -> Tuple[str, List[dict]]:
        logger.debug(f'_async_get_trade_history(), pair: {pair}, since: {since}, until: {until}, from_id: {from_id}')
        if until is None:
            until = ccxt.Exchange.milliseconds()
            logger.debug(f'Exchange milliseconds: {until}')
        if self._trades_pagination == 'time':
            return await self._async_get_trade_history_time(pair=pair, since=since, until=until)
        elif self._trades_pagination == 'id':
            return await self._async_get_trade_history_id(pair=pair, since=since, until=until, from_id=from_id)
        else:
            raise OperationalException(f'Exchange {self.name} does use neither time, nor id based pagination')

    def get_historic_trades(self, pair: str, since: datetime, until: Optional[int] = None, from_id: Optional[Any] = None) -> List[dict]:
        if not self.exchange_has('fetchTrades'):
            raise OperationalException('This exchange does not support downloading Trades.')
        with self._loop_lock:
            task = asyncio.ensure_future(self._async_get_trade_history(pair=pair, since=int(since.timestamp() * 1000), until=until, from_id=from_id))
            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    self.loop.add_signal_handler(sig, task.cancel)
                except NotImplementedError:
                    pass
            return self.loop.run_until_complete(task)[1]

    @retrier
    def _get_funding_fees_from_exchange(self, pair: str, since: Union[int, datetime]) -> float:
        if not self.exchange_has('fetchFundingHistory'):
            raise OperationalException(f'fetch_funding_history() is not available using {self.name}')
        if isinstance(since, datetime):
            since = dt_ts(since)
        try:
            funding_history = self._api.fetch_funding_history(symbol=pair, since=since)
            self._log_exchange_response('funding_history', funding_history, add_info=f'pair: {pair}, since: {since}')
            return sum((fee['amount'] for fee in funding_history))
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not get funding fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_leverage_tiers(self) -> Any:
        try:
            return self._api.fetch_leverage_tiers()
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load leverage tiers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier_async
    async def get_market_leverage_tiers(self, symbol: str) -> Tuple[str, Any]:
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return (symbol, tier)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load leverage tiers for {symbol} due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def load_leverage_tiers(self) -> Any:
        if self.trading_mode == TradingMode.FUTURES:
            if self.exchange_has('fetchLeverageTiers'):
                return self.get_leverage_tiers()
            elif self.exchange_has('fetchMarketLeverageTiers'):
                markets = self.markets
                symbols = [symbol for symbol, market in markets.items() if self.market_is_future(market) and market['quote'] == self._config['stake_currency']]
                tiers: Dict[str, Any] = {}
                tiers_cached = self.load_cached_leverage_tiers(self._config['stake_currency'])
                if tiers_cached:
                    tiers = tiers_cached
                coros = [self.get_market_leverage_tiers(symbol) for symbol in sorted(symbols) if symbol not in tiers]
                if coros:
                    logger.info(f'Initializing leverage_tiers for {len(symbols)} markets. This will take about a minute.')
                else:
                    logger.info('Using cached leverage_tiers.')
                async def gather_results(input_coro: List[Any]) -> List[Any]:
                    return await asyncio.gather(*input_coro, return_exceptions=True)
                for input_coro in chunks(coros, 100):
                    with self._loop_lock:
                        results = self.loop.run_until_complete(gather_results(input_coro))
                    for res in results:
                        if isinstance(res, Exception):
                            logger.warning(f'Leverage tier exception: {repr(res)}')
                            continue
                        symbol_res, tier = res
                        tiers[symbol_res] = tier
                if len(coros) > 0:
                    self.cache_leverage_tiers(tiers, self._config['stake_currency'])
                logger.info(f'Done initializing {len(symbols)} markets.')
                return tiers
        return {}

    def cache_leverage_tiers(self, tiers: Any, stake_currency: str) -> None:
        filename = self._config['datadir'] / 'futures' / f'leverage_tiers_{stake_currency}.json'
        if not filename.parent.is_dir():
            filename.parent.mkdir(parents=True)
        data = {'updated': datetime.now(timezone.utc), 'data': tiers}
        file_dump_json(filename, data)

    def load_cached_leverage_tiers(self, stake_currency: str, cache_time: Optional[timedelta] = None) -> Optional[Any]:
        if not cache_time:
            cache_time = timedelta(weeks=4)
        filename = self._config['datadir'] / 'futures' / f'leverage_tiers_{stake_currency}.json'
        if filename.is_file():
            try:
                tiers = file_load_json(filename)
                updated = tiers.get('updated')
                if updated:
                    updated_dt = parser.parse(updated)
                    if updated_dt < datetime.now(timezone.utc) - cache_time:
                        logger.info('Cached leverage tiers are outdated. Will update.')
                        return None
                return tiers.get('data')
            except Exception:
                logger.exception('Error loading cached leverage tiers. Refreshing.')
        return None

    def fill_leverage_tiers(self) -> None:
        leverage_tiers = self.load_leverage_tiers()
        for pair, tiers in leverage_tiers.items():
            pair_tiers: List[Any] = []
            for tier in tiers:
                pair_tiers.append(self.parse_leverage_tier(tier))
            self._leverage_tiers[pair] = pair_tiers

    def parse_leverage_tier(self, tier: dict) -> dict:
        info = tier.get('info', {})
        return {'minNotional': tier['minNotional'], 'maxNotional': tier['maxNotional'], 'maintenanceMarginRate': tier['maintenanceMarginRate'], 'maxLeverage': tier['maxLeverage'], 'maintAmt': float(info['cum']) if 'cum' in info else None}

    def get_max_leverage(self, pair: str, stake_amount: float) -> float:
        if self.trading_mode == TradingMode.SPOT:
            return 1.0
        if self.trading_mode == TradingMode.FUTURES:
            if stake_amount is None:
                raise OperationalException(f'{self.name}.get_max_leverage requires argument stake_amount')
            if pair not in self._leverage_tiers:
                return 1.0
            pair_tiers = self._leverage_tiers[pair]
            if stake_amount == 0:
                return self._leverage_tiers[pair][0]['maxLeverage']
            for tier_index in range(len(pair_tiers)):
                tier = pair_tiers[tier_index]
                lev = tier['maxLeverage']
                if tier_index < len(pair_tiers) - 1:
                    next_tier = pair_tiers[tier_index + 1]
                    next_floor = next_tier['minNotional'] / next_tier['maxLeverage']
                    if next_floor > stake_amount:
                        return min(tier['maxNotional'] / stake_amount, lev)
                elif stake_amount > tier['maxNotional']:
                    raise InvalidOrderException(f'Amount {stake_amount} too high for {pair}')
                else:
                    return tier['maxLeverage']
            raise OperationalException('Looped through all tiers without finding a max leverage. Should never be reached')
        elif self.trading_mode == TradingMode.MARGIN:
            market = self.markets[pair]
            if market['limits']['leverage']['max'] is not None:
                return market['limits']['leverage']['max']
            else:
                return 1.0
        else:
            return 1.0

    @retrier
    def _set_leverage(self, leverage: float, pair: Optional[str] = None, accept_fail: bool = False) -> None:
        if self._config.get('dry_run') or not self.exchange_has('setLeverage'):
            return
        if self._ft_has.get('floor_leverage', False) is True:
            leverage = floor(leverage)
        try:
            res = self._api.set_leverage(symbol=pair, leverage=leverage)
            self._log_exchange_response('set_leverage', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.BadRequest, ccxt.OperationRejected, ccxt.InsufficientFunds) as e:
            if not accept_fail:
                raise TemporaryError(f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_interest_rate(self) -> float:
        return 0.0

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        return open_date.minute == 0 and open_date.second == 0

    @retrier
    def set_margin_mode(self, pair: str, margin_mode: MarginMode, accept_fail: bool = False, params: Optional[dict] = None) -> None:
        if self._config.get('dry_run') or not self.exchange_has('setMarginMode'):
            return
        if params is None:
            params = {}
        try:
            res = self._api.set_margin_mode(margin_mode.value, pair, params)
            self._log_exchange_response('set_margin_mode', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.BadRequest, ccxt.OperationRejected) as e:
            if not accept_fail:
                raise TemporaryError(f'Could not set margin mode due to {e.__class__.__name__}. Message: {e}') from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not set margin mode due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: Optional[datetime] = None) -> float:
        if self.funding_fee_cutoff(open_date):
            open_date = timeframe_to_prev_date('1h', open_date)
        timeframe = self._ft_has.get('mark_ohlcv_timeframe', '1h')
        timeframe_ff = self._ft_has.get('funding_fee_timeframe', '1h')
        mark_price_type = CandleType.from_string(self._ft_has.get('mark_ohlcv_price', 'mark'))
        if not close_date:
            close_date = datetime.now(timezone.utc)
        since_ms = dt_ts(timeframe_to_prev_date(timeframe, open_date))
        mark_comb = (pair, timeframe, mark_price_type)
        funding_comb = (pair, timeframe_ff, CandleType.FUNDING_RATE)
        candle_histories = self.refresh_latest_ohlcv([mark_comb, funding_comb], since_ms=since_ms, cache=False, drop_incomplete=False)
        try:
            funding_rates = candle_histories[funding_comb]
            mark_rates = candle_histories[mark_comb]
        except KeyError:
            raise ExchangeError('Could not find funding rates.')
        funding_mark_rates = self.combine_funding_and_mark(funding_rates, mark_rates)
        return self.calculate_funding_fees(funding_mark_rates, amount=amount, is_short=is_short, open_date=open_date, close_date=close_date)

    @staticmethod
    def combine_funding_and_mark(funding_rates: DataFrame, mark_rates: DataFrame, futures_funding_rate: Optional[float] = None) -> DataFrame:
        if futures_funding_rate is None:
            return mark_rates.merge(funding_rates, on='date', how='inner', suffixes=['_mark', '_fund'])
        elif len(funding_rates) == 0:
            mark_rates['open_fund'] = futures_funding_rate
            return mark_rates.rename(columns={'open': 'open_mark', 'close': 'close_mark', 'high': 'high_mark', 'low': 'low_mark', 'volume': 'volume_mark'})
        else:
            combined = mark_rates.merge(funding_rates, on='date', how='left', suffixes=['_mark', '_fund'])
            combined['open_fund'] = combined['open_fund'].fillna(futures_funding_rate)
            return combined

    def calculate_funding_fees(self, df: DataFrame, amount: float, is_short: bool, open_date: datetime, close_date: datetime, time_in_ratio: Optional[Any] = None) -> float:
        fees: float = 0
        if not df.empty:
            df1 = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df1['open_fund'] * df1['open_mark'] * amount)
        if isnan(fees):
            fees = 0.0
        return fees if is_short else -fees

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            try:
                if self._config.get('dry_run'):
                    funding_fees = self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
                else:
                    funding_fees = self._get_funding_fees_from_exchange(pair, open_date)
                return funding_fees
            except ExchangeError:
                logger.warning(f'Could not update funding fees for {pair}.')
        return 0.0

    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: Optional[List[Any]] = None) -> Optional[float]:
        if self.trading_mode == TradingMode.SPOT:
            return None
        elif self.trading_mode != TradingMode.FUTURES:
            raise OperationalException(f'{self.name} does not support {self.margin_mode} {self.trading_mode}')
        liquidation_price: Optional[float] = None
        if self._config.get('dry_run') or not self.exchange_has('fetchPositions'):
            liquidation_price = self.dry_run_liquidation_price(pair=pair, open_rate=open_rate, is_short=is_short, amount=amount, stake_amount=stake_amount, leverage=leverage, wallet_balance=wallet_balance, open_trades=open_trades or [])
        else:
            positions = self.fetch_positions(pair)
            if len(positions) > 0:
                pos = positions[0]
                liquidation_price = pos['liquidationPrice']
        if liquidation_price is not None:
            buffer_amount = abs(open_rate - liquidation_price) * self.liquidation_buffer
            liquidation_price_buffer = liquidation_price - buffer_amount if is_short else liquidation_price + buffer_amount
            return max(liquidation_price_buffer, 0.0)
        else:
            return None

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Any]) -> float:
        market = self.markets[pair]
        taker_fee_rate = market['taker']
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if market['inverse']:
                raise OperationalException('Freqtrade does not yet support inverse contracts')
            value = wallet_balance / amount
            mm_ratio_taker = mm_ratio + taker_fee_rate
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def get_maintenance_ratio_and_amt(self, pair: str, notional_value: float) -> Tuple[float, Any]:
        if self._config.get('runmode') in OPTIMIZE_MODES or self.exchange_has('fetchLeverageTiers') or self.exchange_has('fetchMarketLeverageTiers'):
            if pair not in self._leverage_tiers:
                raise InvalidOrderException(f'Maintenance margin rate for {pair} is unavailable for {self.name}')
            pair_tiers = self._leverage_tiers[pair]
            for tier in reversed(pair_tiers):
                if notional_value >= tier['minNotional']:
                    return (tier['maintenanceMarginRate'], tier['maintAmt'])
            raise ExchangeError('nominal value can not be lower than 0')
        else:
            raise ExchangeError(f'Cannot get maintenance ratio using {self.name}')

    # The rest of the methods are already defined above.
    # No additional annotations are added here.
