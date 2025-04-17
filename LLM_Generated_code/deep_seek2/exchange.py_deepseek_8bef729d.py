from typing import Any, Dict, List, Optional, Tuple, Union, Generator, Coroutine, TypeVar, Literal
from datetime import datetime, timedelta, timezone
from math import floor, isnan
from threading import Lock
from collections.abc import Coroutine as CoroutineABC, Generator as GeneratorABC
from copy import deepcopy
import asyncio
import inspect
import logging
import signal
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
        self._leverage_tiers: Dict[str, List[Dict]] = {}
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
        return sorted(set([x["quote"] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get("quote", "")

    def get_pair_base_currency(self, pair: str) -> str:
        return self.markets.get(pair, {}).get("base", "")

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        return (
            market.get(self._ft_has["ccxt_futures_name"], False) is True
            and market.get("type", False) == "swap"
            and market.get("linear", False) is True
        )

    def market_is_spot(self, market: Dict[str, Any]) -> bool:
        return market.get("spot", False) is True

    def market_is_margin(self, market: Dict[str, Any]) -> bool:
        return market.get("margin", False) is True

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        return (
            market.get("quote", None) is not None
            and market.get("base", None) is not None
            and (
                self.precisionMode != TICK_SIZE
                or market.get("precision", {}).get("price") is None
                or market.get("precision", {}).get("price") > 1e-11
            )
            and (
                (self.trading_mode == TradingMode.SPOT and self.market_is_spot(market))
                or (self.trading_mode == TradingMode.MARGIN and self.market_is_margin(market))
                or (self.trading_mode == TradingMode.FUTURES and self.market_is_future(market))
            )
        )

    def klines(self, pair_interval: PairWithTimeframe, copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def trades(self, pair_interval: PairWithTimeframe, copy: bool = True) -> DataFrame:
        if pair_interval in self._trades:
            if copy:
                return self._trades[pair_interval].copy()
            else:
                return self._trades[pair_interval]
        else:
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

    def get_contract_size(self, pair: str) -> Optional[float]:
        if self.trading_mode == TradingMode.FUTURES:
            market = self.markets.get(pair, {})
            contract_size: float = 1.0
            if not market:
                return None
            if market.get("contractSize") is not None:
                contract_size = float(market["contractSize"])
            return contract_size
        else:
            return 1

    def _trades_contracts_to_amount(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(trades) > 0 and "symbol" in trades[0]:
            contract_size = self.get_contract_size(trades[0]["symbol"])
            if contract_size != 1:
                for trade in trades:
                    trade["amount"] = trade["amount"] * contract_size
        return trades

    def _order_contracts_to_amount(self, order: CcxtOrder) -> CcxtOrder:
        if "symbol" in order and order["symbol"] is not None:
            contract_size = self.get_contract_size(order["symbol"])
            if contract_size != 1:
                for prop in self._ft_has.get("order_props_in_contracts", []):
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
        return amount_to_contract_precision(
            amount, self.get_precision_amount(pair), self.precisionMode, contract_size
        )

    def ws_connection_reset(self) -> None:
        if self._exchange_ws:
            self._exchange_ws.reset_connections()

    async def _api_reload_markets(self, reload: bool = False) -> Dict[str, Any]:
        try:
            return await self._api_async.load_markets(reload=reload, params={})
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in reload_markets due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise TemporaryError(e) from e

    def _load_async_markets(self, reload: bool = False) -> Dict[str, Any]:
        try:
            with self._loop_lock:
                markets = self.loop.run_until_complete(self._api_reload_markets(reload=reload))
            if isinstance(markets, Exception):
                raise markets
            return markets
        except asyncio.TimeoutError as e:
            logger.warning("Could not load markets. Reason: %s", e)
            raise TemporaryError from e

    def reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        if (
            not force
            and self._last_markets_refresh > 0
            and (self._last_markets_refresh + self.markets_refresh_interval > dt_ts())
        ):
            return None
        logger.debug("Performing scheduled market reload..")
        try:
            retries: int = 3 if force else 0
            self._markets = retrier(self._load_async_markets, retries=retries)(reload=True)
            self._api.set_markets(self._api_async.markets, self._api_async.currencies)
            self._api.options = self._api_async.options
            if self._exchange_ws:
                self._ws_async.set_markets(self._api.markets, self._api.currencies)
                self._ws_async.options = self._api.options
            self._last_markets_refresh = dt_ts()
            if self._ft_has["needs_trading_fees"]:
                self._trading_fees = self.fetch_trading_fees()
            if load_leverage_tiers and self.trading_mode == TradingMode.FUTURES:
                self.fill_leverage_tiers()
        except (ccxt.BaseError, TemporaryError):
            logger.exception("Could not load markets.")

    def validate_stakecurrency(self, stake_currency: str) -> None:
        if not self._markets:
            raise OperationalException(
                "Could not load markets, therefore cannot start. "
                "Please investigate the above error for more details."
            )
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise ConfigurationError(
                f"{stake_currency} is not available as stake on {self.name}. "
                f"Available currencies are: {', '.join(quote_currencies)}"
            )

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> Generator[str, None, None]:
        yielded = False
        for pair in (
            f"{curr_1}/{curr_2}",
            f"{curr_2}/{curr_1}",
            f"{curr_1}/{curr_2}:{curr_2}",
            f"{curr_2}/{curr_1}:{curr_1}",
        ):
            if pair in self.markets and self.markets[pair].get("active"):
                yielded = True
                yield pair
        if not yielded:
            raise ValueError(f"Could not combine {curr_1} and {curr_2} to get a valid pair.")

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        if not hasattr(self._api, "timeframes") or self._api.timeframes is None:
            raise OperationalException(
                f"The ccxt library does not provide the list of timeframes "
                f"for the exchange {self.name} and this exchange "
                f"is therefore not supported. ccxt fetchOHLCV: {self.exchange_has('fetchOHLCV')}"
            )
        if timeframe and (timeframe not in self.timeframes):
            raise ConfigurationError(
                f"Invalid timeframe '{timeframe}'. This exchange supports: {self.timeframes}"
            )
        if (
            timeframe
            and self._config["runmode"] != RunMode.UTIL_EXCHANGE
            and timeframe_to_minutes(timeframe) < 1
        ):
            raise ConfigurationError("Timeframes < 1m are currently not supported by Freqtrade.")

    def validate_ordertypes(self, order_types: Dict[str, str]) -> None:
        if any(v == "market" for k, v in order_types.items()):
            if not self.exchange_has("createMarketOrder"):
                raise ConfigurationError(f"Exchange {self.name} does not support market orders.")
        self.validate_stop_ordertypes(order_types)

    def validate_stop_ordertypes(self, order_types: Dict[str, str]) -> None:
        if order_types.get("stoploss_on_exchange") and not self._ft_has.get(
            "stoploss_on_exchange", False
        ):
            raise ConfigurationError(f"On exchange stoploss is not supported for {self.name}.")
        if self.trading_mode == TradingMode.FUTURES:
            price_mapping = self._ft_has.get("stop_price_type_value_mapping", {}).keys()
            if (
                order_types.get("stoploss_on_exchange", False) is True
                and "stoploss_price_type" in order_types
                and order_types["stoploss_price_type"] not in price_mapping
            ):
                raise ConfigurationError(
                    f"On exchange stoploss price type is not supported for {self.name}."
                )

    def validate_pricing(self, pricing: Dict[str, Any]) -> None:
        if pricing.get("use_order_book", False) and not self.exchange_has("fetchL2OrderBook"):
            raise ConfigurationError(f"Orderbook not available for {self.name}.")
        if not pricing.get("use_order_book", False) and (
            not self.exchange_has("fetchTicker") or not self._ft_has["tickers_have_price"]
        ):
            raise ConfigurationError(f"Ticker pricing not available for {self.name}.")

    def validate_order_time_in_force(self, order_time_in_force: Dict[str, str]) -> None:
        if any(
            v.upper() not in self._ft_has["order_time_in_force"]
            for k, v in order_time_in_force.items()
        ):
            raise ConfigurationError(
                f"Time in force policies are not supported for {self.name} yet."
            )

    def validate_orderflow(self, exchange: Dict[str, Any]) -> None:
        if exchange.get("use_public_trades", False) and (
            not self.exchange_has("fetchTrades") or not self._ft_has["trades_has_history"]
        ):
            raise ConfigurationError(
                f"Trade data not available for {self.name}. Can't use orderflow feature."
            )

    def validate_freqai(self, config: Config) -> None:
        freqai_enabled = config.get("freqai", {}).get("enabled", False)
        if freqai_enabled and not self._ft_has["ohlcv_has_history"]:
            raise ConfigurationError(
                f"Historic OHLCV data not available for {self.name}. Can't use freqAI."
            )

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        candle_limit = self.ohlcv_candle_limit(
            timeframe,
            self._config["candle_type_def"],
            dt_ts(date_minus_candles(timeframe, startup_candles)) if timeframe else None,
        )
        candle_count = startup_candles + 1
        required_candle_call_count = int(
            (candle_count / candle_limit) + (0 if candle_count % candle_limit == 0 else 1)
        if self._ft_has["ohlcv_has_history"]:
            if required_candle_call_count > 5:
                raise ConfigurationError(
                    f"This strategy requires {startup_candles} candles to start, "
                    "which is more than 5x "
                    f"the amount of candles {self.name} provides for {timeframe}."
                )
        elif required_candle_call_count > 1:
            raise ConfigurationError(
                f"This strategy requires {startup_candles} candles to start, which is more than "
                f"the amount of candles {self.name} provides for {timeframe}."
            )
        if required_candle_call_count > 1:
            logger.warning(
                f"Using {required_candle_call_count} calls to get OHLCV. "
                f"This can result in slower operations for the bot. Please check "
                f"if you really need {startup_candles} candles for your strategy"
            )
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(
        self,
        trading_mode: TradingMode,
        margin_mode: Optional[MarginMode],
    ):
        if trading_mode != TradingMode.SPOT and (
            (trading_mode, margin_mode) not in self._supported_trading_mode_margin_pairs
        ):
            mm_value = margin_mode and margin_mode.value
            raise OperationalException(
                f"Freqtrade does not support {mm_value} {trading_mode} on {self.name}"
            )

    def get_option(self, param: str, default: Optional[Any] = None) -> Any:
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        if endpoint in self._ft_has.get("exchange_has_overrides", {}):
            return self._ft_has["exchange_has_overrides"][endpoint]
        return endpoint in self._api_async.has and self._api_async.has[endpoint]

    def features(
        self, market_type: Literal["spot", "futures"], endpoint: str, attribute: str, default: T
    ) -> T:
        feat = (
            self._api_async.features.get("spot", {})
            if market_type == "spot"
            else self._api_async.features.get("swap", {}).get("linear", {})
        )
        return feat.get(endpoint, {}).get(attribute, default)

    def get_precision_amount(self, pair: str) -> Optional[float]:
        return self.markets.get(pair, {}).get("precision", {}).get("amount", None)

    def get_precision_price(self, pair: str) -> Optional[float]:
        return self.markets.get(pair, {}).get("precision", {}).get("price", None)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode: int = ROUND) -> float:
        return price_to_precision(
            price,
            self.get_precision_price(pair),
            self.precision_mode_price,
            rounding_mode=rounding_mode,
        )

    def price_get_one_pip(self, pair: str, price: float) -> float:
        precision = self.markets[pair]["precision"]["price"]
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(
        self, pair: str, price: float, stoploss: float, leverage: Optional[float] = 1.0
    ) -> Optional[float]:
        return self._get_stake_amount_limit(pair, price, stoploss, "min", leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        max_stake_amount = self._get_stake_amount_limit(pair, price, 0.0, "max", leverage)
        if max_stake_amount is None:
            raise OperationalException(
                f"{self.name}.get_max_pair_stake_amount should never set max_stake_amount to None"
            )
        return max_stake_amount

    def _get_stake_amount_limit(
        self,
        pair: str,
        price: float,
        stoploss: float,
        limit: Literal["min", "max"],
        leverage: Optional[float] = 1.0,
    ) -> Optional[float]:
        isMin = limit == "min"
        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")
        if isMin:
            margin_reserve: float = 1.0 + self._config.get(
                "amount_reserve_percent", DEFAULT_AMOUNT_RESERVE_PERCENT
            )
            stoploss_reserve = margin