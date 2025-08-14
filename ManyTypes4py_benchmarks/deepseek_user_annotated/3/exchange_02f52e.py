from typing import (
    Any, Dict, List, Optional, Tuple, Union, Generator, Coroutine, TypeVar, Literal, TypeGuard
)
from collections.abc import Coroutine as CoroutineABC, Generator as GeneratorABC
from datetime import datetime, timedelta, timezone
from math import floor, isnan
from threading import Lock
import asyncio
import inspect
import logging
import signal
from copy import deepcopy

import ccxt
import ccxt.pro as ccxt_pro
from cachetools import TTLCache
from ccxt import TICK_SIZE
from dateutil import parser
from pandas import DataFrame, concat

from freqtrade.constants import (
    DEFAULT_AMOUNT_RESERVE_PERCENT,
    DEFAULT_TRADES_COLUMNS,
    NON_OPEN_EXCHANGE_STATES,
    BidAsk,
    BuySell,
    Config,
    EntryExit,
    ExchangeConfig,
    ListPairsWithTimeframes,
    MakerTaker,
    OBLiteral,
    PairWithTimeframe,
)
from freqtrade.data.converter import (
    clean_ohlcv_dataframe,
    ohlcv_to_dataframe,
    trades_df_remove_duplicates,
    trades_dict_to_list,
    trades_list_to_df,
)
from freqtrade.enums import (
    OPTIMIZE_MODES,
    TRADE_MODES,
    CandleType,
    MarginMode,
    PriceType,
    RunMode,
    TradingMode,
)
from freqtrade.exceptions import (
    ConfigurationError,
    DDosProtection,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange.common import (
    API_FETCH_ORDER_RETRY_COUNT,
    remove_exchange_credentials,
    retrier,
    retrier_async,
)
from freqtrade.exchange.exchange_types import (
    CcxtBalances,
    CcxtOrder,
    CcxtPosition,
    FtHas,
    OHLCVResponse,
    OrderBook,
    Ticker,
    Tickers,
)
from freqtrade.exchange.exchange_utils import (
    ROUND,
    ROUND_DOWN,
    ROUND_UP,
    amount_to_contract_precision,
    amount_to_contracts,
    amount_to_precision,
    contracts_to_amount,
    date_minus_candles,
    is_exchange_known_ccxt,
    market_is_active,
    price_to_precision,
)
from freqtrade.exchange.exchange_utils_timeframe import (
    timeframe_to_minutes,
    timeframe_to_msecs,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    timeframe_to_seconds,
)
from freqtrade.exchange.exchange_ws import ExchangeWS
from freqtrade.misc import (
    chunks,
    deep_merge_dicts,
    file_dump_json,
    file_load_json,
    safe_value_fallback2,
)
from freqtrade.util import dt_from_ts, dt_now
from freqtrade.util.datetime_helpers import dt_humanize_delta, dt_ts, format_ms_time
from freqtrade.util.periodic_cache import PeriodicCache

logger = logging.getLogger(__name__)

T = TypeVar("T")

class Exchange:
    _params: Dict[str, Any] = {}
    _ccxt_params: Dict[str, Any] = {}
    _ft_has_default: FtHas = {
        "stoploss_on_exchange": False,
        "stop_price_param": "stopLossPrice",
        "stop_price_prop": "stopLossPrice",
        "stoploss_order_types": {},
        "order_time_in_force": ["GTC"],
        "ohlcv_params": {},
        "ohlcv_has_history": True,
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        "ohlcv_volume_currency": "base",
        "tickers_have_quoteVolume": True,
        "tickers_have_percentage": True,
        "tickers_have_bid_ask": True,
        "tickers_have_price": True,
        "trades_limit": 1000,
        "trades_pagination": "time",
        "trades_pagination_arg": "since",
        "trades_has_history": False,
        "l2_limit_range": None,
        "l2_limit_range_required": True,
        "mark_ohlcv_price": "mark",
        "mark_ohlcv_timeframe": "8h",
        "funding_fee_timeframe": "8h",
        "ccxt_futures_name": "swap",
        "needs_trading_fees": False,
        "order_props_in_contracts": ["amount", "filled", "remaining"],
        "marketOrderRequiresPrice": False,
        "exchange_has_overrides": {},
        "proxy_coin_mapping": {},
        "ws_enabled": False,
    }
    _ft_has: FtHas = {}
    _ft_has_futures: FtHas = {}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = []

    def __init__(
        self,
        config: Config,
        *,
        exchange_config: Optional[ExchangeConfig] = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> None:
        self._api: ccxt.Exchange
        self._api_async: ccxt_pro.Exchange
        self._ws_async: Optional[ccxt_pro.Exchange] = None
        self._exchange_ws: Optional[ExchangeWS] = None
        self._markets: Dict[str, Any] = {}
        self._trading_fees: Dict[str, Any] = {}
        self._leverage_tiers: Dict[str, List[Dict[str, Any]]] = {}
        self._loop_lock = Lock()
        self.loop = self._init_async_loop()
        self._config: Config = {}
        self._config.update(config)
        self._pairs_last_refresh_time: Dict[PairWithTimeframe, int] = {}
        self._last_markets_refresh: int = 0
        self._cache_lock = Lock()
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=4, ttl=60 * 10)
        self._exit_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._entry_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._klines: Dict[PairWithTimeframe, DataFrame] = {}
        self._expiring_candle_cache: Dict[Tuple[str, int], PeriodicCache] = {}
        self._trades: Dict[PairWithTimeframe, DataFrame] = {}
        self._dry_run_open_orders: Dict[str, Any] = {}
        
        if config["dry_run"]:
            logger.info("Instance is running with dry_run enabled")
        logger.info(f"Using CCXT {ccxt.__version__}")
        exchange_conf: Dict[str, Any] = exchange_config if exchange_config else config["exchange"]
        remove_exchange_credentials(exchange_conf, config.get("dry_run", False))
        self.log_responses = exchange_conf.get("log_responses", False)

        self.trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
        self.margin_mode: MarginMode = (
            MarginMode(config.get("margin_mode")) if config.get("margin_mode") else MarginMode.NONE
        )
        self.liquidation_buffer = config.get("liquidation_buffer", 0.05)

        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if self.trading_mode == TradingMode.FUTURES:
            self._ft_has = deep_merge_dicts(self._ft_has_futures, self._ft_has)
        if exchange_conf.get("_ft_has_params"):
            self._ft_has = deep_merge_dicts(exchange_conf.get("_ft_has_params"), self._ft_has)
            logger.info("Overriding exchange._ft_has with config params, result: %s", self._ft_has)

        self._ohlcv_partial_candle = self._ft_has["ohlcv_partial_candle"]
        self._max_trades_limit = self._ft_has["trades_limit"]
        self._trades_pagination = self._ft_has["trades_pagination"]
        self._trades_pagination_arg = self._ft_has["trades_pagination_arg"]

        ccxt_config = self._ccxt_config
        ccxt_config = deep_merge_dicts(exchange_conf.get("ccxt_config", {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_conf.get("ccxt_sync_config", {}), ccxt_config)

        self._api = self._init_ccxt(exchange_conf, True, ccxt_config)

        ccxt_async_config = self._ccxt_config
        ccxt_async_config = deep_merge_dicts(
            exchange_conf.get("ccxt_config", {}), ccxt_async_config
        )
        ccxt_async_config = deep_merge_dicts(
            exchange_conf.get("ccxt_async_config", {}), ccxt_async_config
        )
        self._api_async = self._init_ccxt(exchange_conf, False, ccxt_async_config)
        self._has_watch_ohlcv = self.exchange_has("watchOHLCV") and self._ft_has["ws_enabled"]
        if (
            self._config["runmode"] in TRADE_MODES
            and exchange_conf.get("enable_ws", True)
            and self._has_watch_ohlcv
        ):
            self._ws_async = self._init_ccxt(exchange_conf, False, ccxt_async_config)
            self._exchange_ws = ExchangeWS(self._config, self._ws_async)

        logger.info(f'Using Exchange "{self.name}"')
        self.required_candle_call_count = 1
        self.markets_refresh_interval: int = (
            exchange_conf.get("markets_refresh_interval", 60) * 60 * 1000
        )

        if validate:
            self.reload_markets(True, load_leverage_tiers=False)
            self.validate_config(config)
            self._startup_candle_count: int = config.get("startup_candle_count", 0)
            self.required_candle_call_count = self.validate_required_startup_candles(
                self._startup_candle_count, config.get("timeframe", "")
            )

        if self.trading_mode != TradingMode.SPOT and load_leverage_tiers:
            self.fill_leverage_tiers()
        self.additional_exchange_init()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._exchange_ws:
            self._exchange_ws.cleanup()
        logger.debug("Exchange object destroyed, closing async loop")
        if (
            getattr(self, "_api_async", None)
            and inspect.iscoroutinefunction(self._api_async.close)
            and self._api_async.session
        ):
            logger.debug("Closing async ccxt session.")
            self.loop.run_until_complete(self._api_async.close())
        if (
            self._ws_async
            and inspect.iscoroutinefunction(self._ws_async.close)
            and self._ws_async.session
        ):
            logger.debug("Closing ws ccxt session.")
            self.loop.run_until_complete(self._ws_async.close())

        if self.loop and not self.loop.is_closed():
            self.loop.close()

    def _init_async_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    def validate_config(self, config: Config) -> None:
        self.validate_timeframes(config.get("timeframe"))
        self.validate_stakecurrency(config["stake_currency"])
        self.validate_ordertypes(config.get("order_types", {}))
        self.validate_order_time_in_force(config.get("order_time_in_force", {}))
        self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
        self.validate_pricing(config["exit_pricing"])
        self.validate_pricing(config["entry_pricing"])
        self.validate_orderflow(config["exchange"])
        self.validate_freqai(config)

    def _init_ccxt(
        self, exchange_config: Dict[str, Any], sync: bool, ccxt_kwargs: Dict[str, Any]
    ) -> ccxt.Exchange:
        name = exchange_config["name"]
        if sync:
            ccxt_module = ccxt
        else:
            ccxt_module = ccxt_pro
            if not is_exchange_known_ccxt(name, ccxt_module):
                import ccxt.async_support as ccxt_async
                ccxt_module = ccxt_async

        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f"Exchange {name} is not supported by ccxt")

        ex_config = {
            "apiKey": exchange_config.get(
                "api_key", exchange_config.get("apiKey", exchange_config.get("key"))
            ),
            "secret": exchange_config.get("secret"),
            "password": exchange_config.get("password"),
            "uid": exchange_config.get("uid", ""),
            "accountId": exchange_config.get("account_id", exchange_config.get("accountId", "")),
            "walletAddress": exchange_config.get(
                "wallet_address", exchange_config.get("walletAddress")
            ),
            "privateKey": exchange_config.get("private_key", exchange_config.get("privateKey")),
        }
        if ccxt_kwargs:
            logger.info("Applying additional ccxt config: %s", ccxt_kwargs)
        if self._ccxt_params:
            ccxt_kwargs = deep_merge_dicts(self._ccxt_params, deepcopy(ccxt_kwargs))
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:
            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f"Exchange {name} is not supported") from e
        except ccxt.BaseError as e:
            raise OperationalException(f"Initialization of ccxt failed. Reason: {e}") from e

        return api

    @property
    def _ccxt_config(self) -> Dict[str, Any]:
        if self.trading_mode == TradingMode.MARGIN:
            return {"options": {"defaultType": "margin"}}
        elif self.trading_mode == TradingMode.FUTURES:
            return {"options": {"defaultType": self._ft_has["ccxt_futures_name"]}}
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
            logger.info("Markets were not loaded. Loading them now..")
            self.reload_markets(True)
        return self._markets

    @property
    def precisionMode(self) -> int:
        return self._api.precisionMode

    @property
    def precision_mode_price(self) -> int:
        return self._api.precisionMode

    def additional_exchange_init(self) -> None:
        pass

    def _log_exchange_response(self, endpoint: str, response: Any, *, add_info: Optional[str] = None) -> None:
        if self.log_responses:
            add_info_str = "" if add_info is None else f" {add_info}: "
            logger.info(f"API {endpoint}: {add_info_str}{response}")

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None
    ) -> int:
        ccxt_val = self.features(
            "spot" if candle_type == CandleType.SPOT else "futures", "fetchOHLCV", "limit", 500
        )
        if not isinstance(ccxt_val, (float, int)):
            ccxt_val = 500
        fallback_val = self._ft_has.get("ohlcv_candle_limit", ccxt_val)
        if candle_type == CandleType.FUNDING_RATE:
            fallback_val = self._ft_has.get("funding_fee_candle_limit", fallback_val)
        return int(
            self._ft_has.get("ohlcv_candle_limit_per_timeframe", {}).get(
                timeframe, str(fallback_val)
            )
        )

    def get_markets(
        self,
        base_currencies: Optional[List[str]] = None,
        quote_currencies: Optional[List[str]] = None,
        spot_only: bool = False,
        margin_only: bool = False,
        futures_only: bool = False,
        tradable_only: bool = True,
        active_only: bool = False,
    ) -> Dict[str, Any]:
        markets = self.markets
        if not markets:
            raise OperationalException("Markets were not loaded.")

        if base_currencies:
            markets = {k: v for k, v in markets.items() if v["base"] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v["quote"] in quote_currencies}
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
        return sorted(set([