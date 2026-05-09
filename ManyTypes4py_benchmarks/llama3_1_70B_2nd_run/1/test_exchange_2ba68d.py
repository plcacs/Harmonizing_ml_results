from typing import Dict, List, Tuple, Optional, Any, Union
from freqtrade.exchange.common import API_FETCH_ORDER_RETRY_COUNT
from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode

class Exchange:
    _ft_has_default = {
        'ohlcv_has_history': True,
        'ohlcv_candle_limit': 500,
        'ohlcv_timeframe': '1m',
        'ohlcv_trading_limit': 500,
        'order_time_in_force': ['GTC', 'FOK', 'IOC'],
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_limit_ratio': 0.99,
        'trades_pagination': 'time',
        'trades_pagination_arg': 'since',
        'l2_limit_range': [5, 10, 20, 50, 100, 500, 1000],
        'exchange_has_overrides': {},
    }

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api = None
        self._api_async = None
        self._ft_has = Exchange._ft_has_default.copy()
        self._trading_fees = {}
        self._leverage_tiers = {}
        self._params = {}
        self._startup_candle_count = 0
        self._klines = {}
        self._trades = {}
        self._last_markets_refresh = 0
        self._pairs_last_refresh_time = {}
        self._expiring_candle_cache = {}
        self._async_loop = None
        self._async_task = None
        self._dry_run_open_orders = {}
        self._ccxt_config = {}
        self._headers = {}
        self.required_candle_call_count = 1
        self.trading_mode = TradingMode.SPOT
        self.margin_mode = MarginMode.CROSS

    async def _async_get_candle_history(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: int = None, limit: int = None, params: Dict[str, Any] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        return await self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit, params=params)

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, until_ms: int = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        return await self._async_get_candle_history(pair, timeframe, candle_type, since_ms, until_ms=until_ms)

    async def _async_fetch_trades(self, pair: str, since: int = None, params: Dict[str, Any] = None) -> Tuple[List[List[Any]], Union[str, int]]:
        return await self._api_async.fetch_trades(pair, since=since, params=params)

    async def _async_get_trade_history_id(self, pair: str, since: int, until: int) -> Tuple[str, List[List[Any]]]:
        return await self._async_fetch_trades(pair, since=since, params={self._trades_pagination_arg: since})

    async def _async_get_trade_history_time(self, pair: str, since: int, until: int) -> Tuple[str, List[List[Any]]]:
        return await self._async_fetch_trades(pair, since=since)

    def get_historic_trades(self, pair: str, since: int, until: int) -> Tuple[str, List[List[Any]]]:
        if self.exchange_has('fetchTrades'):
            return self._async_get_trade_history_id(pair, since, until)
        else:
            return self._async_get_trade_history_time(pair, since, until)

    def get_next_limit_in_list(self, value: int, limit_range: List[int] = None, round_up: bool = True) -> Optional[int]:
        if limit_range is None:
            return value
        if not round_up:
            return min([x for x in limit_range if x >= value])
        return min([x for x in limit_range if x > value])

    def refresh_latest_ohlcv(self, pairs: List[Tuple[str, str, CandleType]], cache: bool = True) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        if cache:
            return self._refresh_latest_ohlcv_cache(pairs)
        return self._refresh_latest_ohlcv(pairs)

    def _refresh_latest_ohlcv_cache(self, pairs: List[Tuple[str, str, CandleType]]) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        result = {}
        for pair in pairs:
            if pair in self._expiring_candle_cache:
                result[pair] = self._expiring_candle_cache[pair]
        return result

    def _refresh_latest_ohlcv(self, pairs: List[Tuple[str, str, CandleType]]) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        result = {}
        for pair in pairs:
            if self._now_is_time_to_refresh(pair[0], pair[1], pair[2]):
                result[pair] = self.get_historic_ohlcv(pair[0], pair[1], dt_ts(), pair[2])
                self._pairs_last_refresh_time[pair] = result[pair][-2][0]
                self._expiring_candle_cache[pair] = result[pair]
        return result

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType = CandleType.SPOT) -> List[List[Any]]:
        since = dt_ts(since_ms)
        limit = self.ohlcv_candle_limit(timeframe, candle_type)
        calls = self.required_candle_call_count
        ohlcv = []
        while since < dt_ts():
            since_tmp = since
            tmp_ohlcv = self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_tmp, limit=limit, params={'partial': False})
            since = tmp_ohlcv[-1][0] + 1
            ohlcv += tmp_ohlcv
            calls -= 1
            if calls <= 0:
                break
        return ohlcv

    def refresh_latest_trades(self, pairs: List[Tuple[str, str, CandleType]], cache: bool = True) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        if cache:
            return self._refresh_latest_trades_cache(pairs)
        return self._refresh_latest_trades(pairs)

    def _refresh_latest_trades_cache(self, pairs: List[Tuple[str, str, CandleType]]) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        result = {}
        for pair in pairs:
            if pair in self._trades:
                result[pair] = self._trades[pair]
        return result

    def _refresh_latest_trades(self, pairs: List[Tuple[str, str, CandleType]]) -> Dict[Tuple[str, str, CandleType], List[List[Any]]]:
        result = {}
        for pair in pairs:
            if self._now_is_time_to_refresh(pair[0], pair[1], pair[2]):
                result[pair] = self.get_historic_trades(pair[0], dt_ts(), pair[2])
                self._trades[pair] = result[pair]
        return result

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        last_closed_candle = self._pairs_last_refresh_time.get((pair, timeframe, candle_type))
        if last_closed_candle is None:
            return True
        timeframe_mins = int(timeframe[:-1])
        last_closed_candle += timeframe_mins * 60 * 1000
        return last_closed_candle <= dt_ts()

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: int = None) -> int:
        if self._ft_has['ohlcv_candle_limit'] is not None:
            return self._ft_has['ohlcv_candle_limit']
        if since_ms is not None:
            since = dt_ts(since_ms)
        else:
            since = dt_ts()
        limit = self.features(self.trading_mode.value, 'fetchOHLCV', 'limit', 500)
        timeframe_mins = int(timeframe[:-1])
        timeframe_ms = timeframe_mins * 60 * 1000
        calls = self.required_candle_call_count
        while since < dt_ts() and calls > 0:
            since += timeframe_ms * limit
            calls -= 1
        return limit

    def features(self, trading_mode: str, endpoint: str, key: str, default: int) -> int:
        if trading_mode in self._api_async.features and endpoint in self._api_async.features[trading_mode] and key in self._api_async.features[trading_mode][endpoint]:
            return self._api_async.features[trading_mode][endpoint][key]
        return default

    def _load_async_markets(self) -> None:
        self._api_async.load_markets()

    def _load_markets(self) -> Dict[str, Any]:
        return self._api.load_markets()

    def reload_markets(self, force: bool = False) -> None:
        if self._last_markets_refresh + self._config['exchange']['markets_refresh_interval'] < dt_ts():
            self._load_async_markets()
            self._last_markets_refresh = dt_ts()
        elif force:
            self._load_async_markets()

    def validate_timeframes(self) -> None:
        if 'timeframes' not in dir(self._api):
            raise OperationalException(f'The ccxt library does not provide the list of timeframes for the exchange {self._api.id} and this exchange is therefore not supported.')
        if self._config['timeframe'] not in self._api.timeframes:
            raise ConfigurationError(f'Invalid timeframe {self._config["timeframe"]}. This exchange supports {list(self._api.timeframes.keys())}.')

    def validate_pricing(self) -> None:
        if not self._api.has['fetchTicker']:
            raise OperationalException(f'Ticker pricing not available for {self.name}.')
        if self._config['exit_pricing']['use_order_book']:
            if not self._api.has['fetchL2OrderBook']:
                raise OperationalException(f'Orderbook not available for {self.name}.')

    def validate_ordertypes(self) -> None:
        if not self._api.has['createMarketOrder']:
            raise OperationalException(f'Exchange {self.name} does not support market orders.')
        if self._config['order_types']['stoploss_on_exchange'] and self._config['order_types']['stoploss_price_type'] not in self._ft_has['order_time_in_force']:
            raise OperationalException(f'On exchange stoploss price type {self._config["order_types"]["stoploss_price_type"]} is not supported for {self.name}.')

    def validate_order_time_in_force(self, tif: Dict[str, str]) -> None:
        for side, value in tif.items():
            if value not in self._ft_has['order_time_in_force']:
                raise OperationalException(f'Time in force {value} not supported for {side} orders on {self.name}.')

    def validate_orderflow(self, orderflow: Dict[str, bool]) -> None:
        if orderflow['use_public_trades']:
            if not self._ft_has['ohlcv_has_history']:
                raise ConfigurationError(f'Trade data not available for {self.name}.')

    def validate_freqai(self, conf: Dict[str, Any]) -> None:
        if conf['freqai']['enabled']:
            if not self._ft_has['ohlcv_has_history']:
                raise ConfigurationError(f'Historic OHLCV data not available for {self.name}.')

    def validate_stakecurrency(self) -> None:
        stake = self._config['stake_currency']
        quote_currencies = self.get_quote_currencies()
        if stake not in quote_currencies:
            raise ConfigurationError(f'{stake} is not available as stake on {self.name}. Available currencies are: {quote_currencies}.')

    def validate_trading_mode_and_margin_mode(self, trading_mode: TradingMode, margin_mode: MarginMode) -> None:
        if trading_mode == TradingMode.MARGIN and margin_mode == MarginMode.ISOLATED:
            raise OperationalException(f'{self.name} does not support isolated margin.')
        if trading_mode == TradingMode.FUTURES and margin_mode == MarginMode.CROSS:
            raise OperationalException(f'{self.name} does not support cross margin.')

    def get_quote_currencies(self) -> List[str]:
        return list(set([v['quote'] for v in self.markets.values() if 'quote' in v]))

    def get_pair_quote_currency(self, pair: str) -> str:
        market = self.markets.get(pair)
        if market and 'quote' in market:
            return market['quote']
        return ''

    def get_pair_base_currency(self, pair: str) -> str:
        market = self.markets.get(pair)
        if market and 'base' in market:
            return market['base']
        return ''

    def get_conversion_rate(self, from_currency: str, to_currency: str) -> float:
        if from_currency == to_currency:
            return 1
        if from_currency not in self._config['exchange']['quote_currencies']:
            return None
        if to_currency not in self._config['exchange']['quote_currencies']:
            return None
        tick = self.get_tickers()
        if from_currency not in tick:
            return None
        if to_currency not in tick:
            return None
        return tick[from_currency]['last'] / tick[to_currency]['last']

    def get_tickers(self, cached: bool = True) -> Dict[str, Dict[str, Any]]:
        if cached:
            return self._api.fetch_tickers()
        else:
            return self._api.fetch_tickers(params={'ignoreInvalid': True})

    def fetch_ticker(self, pair: str) -> Dict[str, Any]:
        return self._api.fetch_ticker(pair)

    def get_bids_asks(self, cached: bool = True) -> Dict[str, Dict[str, Any]]:
        if cached:
            return self._api.fetch_bids_asks()
        else:
            return self._api.fetch_bids_asks(params={'ignoreInvalid': True})

    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> Dict[str, List[List[Any]]]:
        limit = self.get_next_limit_in_list(limit, self.get_option('l2_limit_range'))
        return self._api.fetch_l2_order_book(pair, limit=limit)

    def get_min_pair_stake_amount(self, pair: str, rate: float, stoploss: float, stake_amount: float = None) -> float:
        market = self.markets.get(pair)
        if market is None:
            raise ValueError(f'Could not get market information for {pair}.')
        if 'limits' not in market:
            return None
        if 'cost' in market['limits'] and market['limits']['cost']['min'] is not None:
            min_cost = market['limits']['cost']['min']
        elif 'amount' in market['limits'] and market['limits']['amount']['min'] is not None:
            min_cost = market['limits']['amount']['min'] * rate
        else:
            return None
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            contract_size = self.get_contract_size(pair)
            if contract_size is not None:
                min_cost /= contract_size
        if stake_amount is not None:
            min_cost /= stake_amount
        if stoploss != 0:
            min_cost *= (1 + abs(stoploss)) / (1 - abs(stoploss))
        return min_cost

    def get_max_pair_stake_amount(self, pair: str, rate: float, stake_amount: float = None) -> float:
        market = self.markets.get(pair)
        if market is None:
            raise ValueError(f'Could not get market information for {pair}.')
        if 'limits' not in market:
            return float('inf')
        if 'cost' in market['limits'] and market['limits']['cost']['max'] is not None:
            max_cost = market['limits']['cost']['max']
        elif 'amount' in market['limits'] and market['limits']['amount']['max'] is not None:
            max_cost = market['limits']['amount']['max'] * rate
        else:
            return float('inf')
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            contract_size = self.get_contract_size(pair)
            if contract_size is not None:
                max_cost /= contract_size
        if stake_amount is not None:
            max_cost /= stake_amount
        return max_cost

    def get_balances(self) -> Dict[str, Dict[str, Any]]:
        return self._api.fetch_balance()

    def fetch_orders(self, pair: str, since: datetime) -> List[Dict[str, Any]]:
        if self.exchange_has('fetchOrders'):
            return self._api.fetch_orders(pair=pair, since=since)
        else:
            return self._api.fetch_open_orders(pair=pair) + self._api.fetch_closed_orders(pair=pair, since=since)

    def fetch_trading_fees(self) -> Dict[str, Dict[str, Any]]:
        return self._api.fetch_trading_fees()

    def fetch_positions(self) -> List[Dict[str, Any]]:
        return self._api.fetch_positions()

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float, rate: float, leverage: float = 1.0) -> Dict[str, Any]:
        order_id = f'dry_run_{side}_{randint(0, 10 ** 6)}'
        return {
            'id': order_id,
            'info': {},
            'symbol': pair,
            'amount': amount,
            'cost': amount * rate,
            'filled': amount,
            'remaining': 0,
            'fee': None,
            'fees': [],
            'trades': None,
            'type': ordertype,
            'side': side,
            'status': 'closed',
            'average': rate,
        }

    def create_order(self, pair: str, ordertype: str, side: str, amount: float, rate: float = None, leverage: float = 1.0, time_in_force: str = 'GTC') -> Dict[str, Any]:
        params = self._get_params(side, ordertype, False, time_in_force, leverage)
        if rate is not None:
            params['price'] = rate
        amount = self.amount_to_precision(pair, amount)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            amount = self._amount_to_contracts(pair, amount)
        params['amount'] = amount
        if self._order_needs_price(side, ordertype):
            params['price'] = rate
        order = self._api.create_order(pair, ordertype, side, amount, params=params)
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            order = self._order_contracts_to_amount(order)
        return order

    def buy_dry_run(self, pair: str, ordertype: str, amount: float, rate: float = None, time_in_force: str = 'GTC', leverage: float = 1.0) -> Dict[str, Any]:
        return self.create_dry_run_order(pair, ordertype, 'buy', amount, rate, leverage)

    def buy_prod(self, pair: str, ordertype: str, amount: float, rate: float = None, time_in_force: str = 'GTC', leverage: float = 1.0) -> Dict[str, Any]:
        return self.create_order(pair, ordertype, 'buy', amount, rate, leverage, time_in_force)

    def sell_dry_run(self, pair: str, ordertype: str, amount: float, rate: float = None, time_in_force: str = 'GTC', leverage: float = 1.0) -> Dict[str, Any]:
        return self.create_dry_run_order(pair, ordertype, 'sell', amount, rate, leverage)

    def sell_prod(self, pair: str, ordertype: str, amount: float, rate: float = None, time_in_force: str = 'GTC', leverage: float = 1.0) -> Dict[str, Any]:
        return self.create_order(pair, ordertype, 'sell', amount, rate, leverage, time_in_force)

    def cancel_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        return self._api.cancel_order(order_id, pair)

    def cancel_stoploss_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        return self._api.cancel_order(order_id, pair)

    def fetch_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        if self.exchange_has('fetchOrder'):
            return self._api.fetch_order(order_id, pair)
        else:
            open_order = self._api.fetch_open_order(order_id, pair)
            if open_order:
                return open_order
            return self._api.fetch_closed_order(order_id, pair)

    def fetch_stoploss_order(self, order_id: str, pair: str) -> Dict[str, Any]:
        if self._config['exchange']['name'] == 'okx':
            return self._api.fetch_order(order_id, pair)
        return self._api.fetch_order(order_id, pair)

    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List[Dict[str, Any]]:
        if self.exchange_has('fetchMyTrades'):
            return self._api.fetch_my_trades(pair, since=since)
        else:
            return []

    def get_fee(self, symbol: str, taker_or_maker: str = 'taker') -> float:
        if 'fee' in self._config:
            return self._config['fee']
        return self._api.calculate_fee(symbol, taker_or_maker)

    def exchange_has(self, endpoint: str) -> bool:
        return self._api.has[endpoint]

    def validate_required_startup_candles(self, startup_candle_count: int, timeframe: str) -> int:
        if startup_candle_count == 0:
            return 1
        timeframe_mins = int(timeframe[:-1])
        timeframe_ms = timeframe_mins * 60 * 1000
        limit = self.ohlcv_candle_limit(timeframe)
        calls = self.required_candle_call_count
        total_candles = 0
        while total_candles < startup_candle_count and calls > 0:
            total_candles += limit
            calls -= 1
        if total_candles < startup_candle_count:
            raise OperationalException(f'This strategy requires {startup_candle_count} candles, which is more than the amount {total_candles} that can be downloaded in {calls} calls.')
        return calls

    def get_option(self, option: str) -> Any:
        return self._ft_has.get(option)

    def merge_ft_has_dict(self) -> None:
        if 'ft_has_params' in self._config['exchange']:
            self._ft_has.update(self._config['exchange']['ft_has_params'])

    def get_valid_pair_combination(self, base: str, quote: str) -> str:
        for pair, market in self.markets.items():
            if market['base'] == base and market['quote'] == quote:
                yield pair
            if market['base'] == quote and market['quote'] == base:
                yield pair

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        if market['spot']:
            return self.trading_mode == TradingMode.SPOT
        if market['future']:
            return self.trading_mode == TradingMode.FUTURES
        if market['margin']:
            return self.trading_mode == TradingMode.MARGIN
        return False

    def market_is_active(self, market: Dict[str, Any]) -> bool:
        return market.get('active', True)

    def order_has_fee(self, order: Dict[str, Any]) -> bool:
        return 'fee' in order and 'cost' in order['fee'] and order['fee']['cost'] is not None

    def extract_cost_curr_rate(self, fee: Dict[str, Any], symbol: str, cost: float, amount: float) -> Tuple[float, str, float]:
        cost_currency = fee['currency']
        if cost_currency in [symbol.split('/')[0], symbol.split('/')[1]]:
            fee_rate = fee['cost'] / cost
        else:
            fee_rate = fee['cost'] / (cost * self.get_conversion_rate(cost_currency, symbol.split('/')[1]))
        return fee['cost'], cost_currency, fee_rate

    def calculate_fee_rate(self, fee: Dict[str, Any], symbol: str, cost: float, amount: float) -> float:
        if 'rate' in fee:
            return fee['rate']
        if 'cost' in fee and fee['cost'] is not None:
            return self.extract_cost_curr_rate(fee, symbol, cost, amount)[2]
        if self._config['exchange']['unknown_fee_rate'] is not None:
            return self._config['exchange']['unknown_fee_rate']
        return None

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        if refresh:
            self._entry_last_rate_refresh = dt_ts()
        if self._entry_last_rate_refresh + 300 < dt_ts():
            self._entry_last_rate_refresh = dt_ts()
        if pair not in self._entry_last_rate:
            self._entry_last_rate[pair] = self.get_rate(pair, 'entry', is_short, refresh)
        if pair not in self._exit_last_rate:
            self._exit_last_rate[pair] = self.get_rate(pair, 'exit', is_short, refresh)
        return self._entry_last_rate[pair], self._exit_last_rate[pair]

    def get_rate(self, pair: str, side: str, is_short: bool, refresh: bool) -> float:
        if refresh:
            if side == 'entry':
                self._entry_last_rate_refresh = dt_ts()
            else:
                self._exit_last_rate_refresh = dt_ts()
        if side == 'entry' and self._entry_last_rate_refresh + 300 < dt_ts():
            self._entry_last_rate_refresh = dt_ts()
        if side == 'exit' and self._exit_last_rate_refresh + 300 < dt_ts():
            self._exit_last_rate_refresh = dt_ts()
        if pair not in self._entry_last_rate:
            self._entry_last_rate[pair] = self.get_rate(pair, 'entry', is_short, refresh)
        if pair not in self._exit_last_rate:
            self._exit_last_rate[pair] = self.get_rate(pair, 'exit', is_short, refresh)
        if side == 'entry':
            return self._entry_last_rate[pair]
        else:
            return self._exit_last_rate[pair]

    def get_entry_rate(self, pair: str, is_short: bool, refresh: bool) -> float:
        return self.get_rate(pair, 'entry', is_short, refresh)

    def get_exit_rate(self, pair: str, is_short: bool, refresh: bool) -> float:
        return self.get_rate(pair, 'exit', is_short, refresh)

    def _get_stop_limit_rate(self, rate: float, order_types: Dict[str, Any], side: str) -> float:
        ratio = order_types.get('stoploss_on_exchange_limit_ratio')
        if ratio is None:
            return rate
        if side == 'buy':
            return rate * (1 + ratio)
        if side == 'sell':
            return rate * (1 - ratio)
        raise InvalidOrderException('Invalid side.')

    def _dry_is_price_crossed(self, pair: str, side: str, rate: float, is_short: bool) -> bool:
        if side == 'buy' and is_short:
            return rate > self.get_exit_rate(pair, is_short, refresh=False)
        if side == 'sell' and not is_short:
            return rate < self.get_entry_rate(pair, is_short, refresh=False)
        return False

    def _valid_trade_pagination_id(self, pair: str, trade_id: str) -> bool:
        if self._trades_pagination == 'id':
            try:
                int(trade_id)
                return True
            except ValueError:
                return False
        return True

    def _async_kucoin_get_candle_history(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: int = None, limit: int = None, params: Dict[str, Any] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        return self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit, params=params)

    def _async_get_candle_history_sort(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: int = None, limit: int = None, params: Dict[str, Any] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        ohlcv = self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit, params=params)
        return pair, timeframe, candle_type, sorted(ohlcv, key=lambda x: x[0]), True

    def _async_get_candle_history_empty(self, pair: str, timeframe: str, candle_type: CandleType, since_ms: int = None, limit: int = None, params: Dict[str, Any] = None) -> Tuple[str, str, CandleType, List[List[Any]], bool]:
        return pair, timeframe, candle_type, [], True

    def _async_fetch_trades_contract_size(self, pair: str, since: int = None, params: Dict[str, Any] = None) -> Tuple[List[List[Any]], Union[str, int]]:
        trades = self._api_async.fetch_trades(pair, since=since, params=params)
        for trade in trades:
            if 'contractSize' in trade['info']:
                trade['amount'] *= float(trade['info']['contractSize'])
        return trades

    def _async_get_trade_history_id(self, pair: str, since: int, until: int) -> Tuple[str, List[List[Any]]]:
        return self._async_fetch_trades(pair, since=since, params={self._trades_pagination_arg: since})

    def _async_get_trade_history_time(self, pair: str, since: int, until: int) -> Tuple[str, List[List[Any]]]:
        return self._async_fetch_trades(pair, since=since)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        return stake_amount / leverage

    def _set_leverage(self, leverage: float) -> None:
        self._api.set_leverage(leverage)

    def set_margin_mode(self, margin_mode: MarginMode) -> None:
        self._api.set_margin_mode(margin_mode)

    def _get_funding_fees_from_exchange(self, pair: str, since: Union[int, datetime]) -> float:
        funding_rates = self._api.fetch_funding_rate_history(pair, since=since)
        return sum([float(rate['amount']) for rate in funding_rates])

    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: datetime) -> float:
        funding_rates = self._api.fetch_funding_rate_history(pair, since=open_date)
        mark_ohlcv = self._api.fetch_ohlcv(pair, timeframe='1h', since=open_date)
        df = self.combine_funding_and_mark(funding_rates, mark_ohlcv)
        return self.calculate_funding_fees(df, amount, is_short, open_date, close_date)

    def combine_funding_and_mark(self, funding_rates: DataFrame, mark_ohlcv: DataFrame, futures_funding_rate: float = None) -> DataFrame:
        funding_rates = funding_rates.set_index('date')
        mark_ohlcv = mark_ohlcv.set_index('date')
        df = funding_rates.join(mark_ohlcv, how='outer', lsuffix='_fund', rsuffix='_mark')
        if futures_funding_rate is not None:
            df['open_fund'] = df['open_fund'].fillna(futures_funding_rate)
        return df

    def calculate_funding_fees(self, df: DataFrame, amount: float, is_short: bool, open_date: datetime, close_date: datetime, time_in_ratio: float = None) -> float:
        if is_short:
            funding_rates = df.loc[open_date:close_date, 'open_fund']
            mark_rates = df.loc[open_date:close_date, 'open_mark']
            funding_fees = funding_rates * amount * mark_rates
            if time_in_ratio is not None:
                funding_fees *= time_in_ratio
            return funding_fees.sum()
        else:
            funding_rates = df.loc[open_date:close_date, 'open_fund']
            funding_fees = funding_rates * amount
            if time_in_ratio is not None:
                funding_fees *= time_in_ratio
            return funding_fees.sum()

    def get_contract_size(self, pair: str) -> float:
        market = self.markets.get(pair)
        if market and 'contractSize' in market:
            return float(market['contractSize'])
        return None

    def _order_contracts_to_amount(self, order: Dict[str, Any]) -> Dict[str, Any]:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            contract_size = self.get_contract_size(order['symbol'])
            if contract_size is not None:
                order['amount'] *= contract_size
                order['cost'] *= contract_size
        return order

    def _trades_contracts_to_amount(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            for trade in trades:
                contract_size = self.get_contract_size(trade['symbol'])
                if contract_size is not None:
                    trade['amount'] *= contract_size
        return trades

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            contract_size = self.get_contract_size(pair)
            if contract_size is not None:
                return amount / contract_size
        return amount

    def _contracts_to_amount(self, pair: str, contracts: float) -> float:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            contract_size = self.get_contract_size(pair)
            if contract_size is not None:
                return contracts * contract_size
        return contracts

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            return self.amount_to_precision(pair, amount)
        return self.amount_to_precision(pair, amount, round=False)

    def _get_stop_limit_rate(self, rate: float, order_types: Dict[str, Any], side: str) -> float:
        ratio = order_types.get('stoploss_on_exchange_limit_ratio')
        if ratio is None:
            return rate
        if side == 'buy':
            return rate * (1 + ratio)
        if side == 'sell':
            return rate * (1 - ratio)
        raise InvalidOrderException('Invalid side.')

    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Dict[str, Any]]) -> float:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            positions = self._api.fetch_positions()
            for position in positions:
                if position['symbol'] == pair:
                    liq_price = position['liquidationPrice']
                    if self._config['liquidation_buffer'] > 0:
                        liq_price += self._config['liquidation_buffer'] * abs(open_rate - liq_price) if is_short else -self._config['liquidation_buffer'] * abs(open_rate - liq_price)
                    return liq_price
        elif self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.CROSS:
            if self.name in ['gate', 'okx']:
                position = 0.0
                for trade in open_trades:
                    if trade['symbol'] == pair:
                        position += trade['amount']
                wb = wallet_balance
                cum_b = 0.0
                side_1 = -1.0 if is_short else 1.0
                ep1 = open_rate
                mmr_b = self.get_maintenance_ratio_and_amt(pair, stake_amount)[0]
                liq_price = (ep1 + (wb / position)) / (1.0 + (mmr_b + self.get_fee(pair)))
                if self._config['liquidation_buffer'] > 0:
                    liq_price += self._config['liquidation_buffer'] * abs(open_rate - liq_price) if is_short else -self._config['liquidation_buffer'] * abs(open_rate - liq_price)
                return liq_price
            elif self.name == 'binance':
                position = 0.0
                for trade in open_trades:
                    if trade['symbol'] == pair:
                        position += trade['amount']
                wb = wallet_balance
                cum_b = 0.0
                side_1 = -1.0 if is_short else 1.0
                ep1 = open_rate
                mmr_b = self.get_maintenance_ratio_and_amt(pair, stake_amount)[0]
                liq_price = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
                if self._config['liquidation_buffer'] > 0:
                    liq_price += self._config['liquidation_buffer'] * abs(open_rate - liq_price) if is_short else -self._config['liquidation_buffer'] * abs(open_rate - liq_price)
                return liq_price
            elif self.name == 'bybit':
                position = 0.0
                for trade in open_trades:
                    if trade['symbol'] == pair:
                        position += trade['amount']
                wb = wallet_balance
                cum_b = 0.0
                side_1 = -1.0 if is_short else 1.0
                ep1 = open_rate
                mmr_b = self.get_maintenance_ratio_and_amt(pair, stake_amount)[0]
                liq_price = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
                if self._config['liquidation_buffer'] > 0:
                    liq_price += self._config['liquidation_buffer'] * abs(open_rate - liq_price) if is_short else -self._config['liquidation_buffer'] * abs(open_rate - liq_price)
                return liq_price
        return None

    def get_maintenance_ratio_and_amt(self, pair: str, nominal_value: float) -> Tuple[float, float]:
        if nominal_value < 0:
            raise DependencyException('nominal value can not be lower than 0')
        for tier in self._leverage_tiers.get(pair, []):
            if nominal_value >= tier['minNotional'] and nominal_value <= tier['maxNotional']:
                return tier['maintenanceMarginRate'], tier['maintAmt']
        raise InvalidOrderException(f'Maintenance margin rate for {pair} is unavailable for {nominal_value}')

    def get_max_leverage(self, pair: str, nominal_value: float) -> float:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            for tier in self._leverage_tiers.get(pair, []):
                if nominal_value >= tier['minNotional'] and nominal_value <= tier['maxNotional']:
                    return tier['maxLeverage']
        else:
            return self.get_max_leverage_from_margin(pair, nominal_value)

    def get_max_leverage_from_margin(self, pair: str, nominal_value: float) -> float:
        market = self.markets.get(pair)
        if market and 'limits' in market and 'cost' in market['limits'] and market['limits']['cost']['min'] is not None:
            return nominal_value / market['limits']['cost']['min']
        return 1.0

    def _get_params(self, side: str, ordertype: str, reduceOnly: bool, time_in_force: str, leverage: float) -> Dict[str, Any]:
        params = self._params.copy()
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            if self.name in ['kraken', 'okx']:
                params['leverage'] = leverage
            if self.name == 'okx':
                params['tdMode'] = 'isolated'
                params['posSide'] = 'net'
            if self.name == 'bybit':
                params['position_idx'] = 0
        if reduceOnly:
            params['reduceOnly'] = True
        if time_in_force:
            params['timeInForce'] = time_in_force
        return params

    def parse_leverage_tier(self, tier: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'minNotional': tier['minNotional'],
            'maxNotional': tier['maxNotional'],
            'maintenanceMarginRate': tier['maintenanceMarginRate'],
            'maxLeverage': tier['maxLeverage'],
            'maintAmt': tier.get('maintAmt'),
        }

    def load_leverage_tiers(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:
            leverage_tiers = self._api.fetch_leverage_tiers()
            self._leverage_tiers = {pair: [self.parse_leverage_tier(tier) for tier in tiers] for pair, tiers in leverage_tiers.items()}
        return self._leverage_tiers

    def get_market_leverage_tiers(self, symbol: str) -> List[Dict[str, Any]]:
        return self._api.fetch_market_leverage_tiers(symbol)

    def _ccxt_config(self) -> Dict[str, Any]:
        if self.trading_mode == TradingMode.SPOT:
            return {}
        if self.trading_mode == TradingMode.MARGIN:
            return {'options': {'defaultType': 'margin'}}
        if self.trading_mode == TradingMode.FUTURES:
            return {'options': {'defaultType': 'swap'}}

    def check_order_canceled_empty(self, order: Dict[str, Any]) -> bool:
        return order['status'] in ['closed', 'canceled'] and order['filled'] == 0.0

    def is_cancel_order_result_suitable(self, order: Dict[str, Any]) -> bool:
        return order['status'] in ['closed', 'canceled'] and 'amount' in order and 'fee' in order

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict[str, Any]:
        cancel_result = self.cancel_order(order_id, pair)
        if self.is_cancel_order_result_suitable(cancel_result):
            return cancel_result
        else:
            return {'id': order_id, 'amount': amount, 'symbol': pair, 'status': 'canceled', 'fee': {}, 'info': {}}

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict[str, Any]:
        cancel_result = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(cancel_result):
            return cancel_result
        else:
            return {'id': order_id, 'amount': amount, 'symbol': pair, 'status': 'canceled', 'fee': {}, 'info': {}}
