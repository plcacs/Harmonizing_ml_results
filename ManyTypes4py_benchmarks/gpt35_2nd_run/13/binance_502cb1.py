from typing import Dict, List, Union
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.binance_public_data import concat_safe, download_archive_ohlcv
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas, Tickers
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts
from pandas import DataFrame

class Binance(Exchange):
    _ft_has: Dict[str, Union[bool, str, Dict[str, str], List[str], List[int]]] = {'stoploss_on_exchange': True, 'stop_price_param': 'stopPrice', 'stop_price_prop': 'stopPrice', 'stoploss_order_types': {'limit': 'stop_loss_limit'}, 'order_time_in_force': ['GTC', 'FOK', 'IOC', 'PO'], 'trades_pagination': 'id', 'trades_pagination_arg': 'fromId', 'trades_has_history': True, 'l2_limit_range': [5, 10, 20, 50, 100, 500, 1000], 'ws_enabled': True}
    _ft_has_futures: Dict[str, Union[int, bool, Dict[str, str], List[str], bool, Dict[str, str]] = {'funding_fee_candle_limit': 1000, 'stoploss_order_types': {'limit': 'stop', 'market': 'stop_market'}, 'order_time_in_force': ['GTC', 'FOK', 'IOC'], 'tickers_have_price': False, 'floor_leverage': True, 'stop_price_type_field': 'workingType', 'order_props_in_contracts': ['amount', 'cost', 'filled', 'remaining'], 'stop_price_type_value_mapping': {PriceType.LAST: 'CONTRACT_PRICE', PriceType.MARK: 'MARK_PRICE'}, 'ws_enabled': False, 'proxy_coin_mapping': {'BNFCR': 'USDC', 'BFUSD': 'USDT'}}
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [(TradingMode.FUTURES, MarginMode.CROSS), (TradingMode.FUTURES, MarginMode.ISOLATED)]

    def get_proxy_coin(self) -> str:
        ...

    def get_tickers(self, symbols=None, *, cached=False, market_type=None) -> Dict[str, Dict[str, float]]:
        ...

    @retrier
    def additional_exchange_init(self) -> None:
        ...

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool = False, until_ms: int = None) -> DataFrame:
        ...

    def get_historic_ohlcv_fast(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, is_new_pair: bool = False, until_ms: int = None) -> DataFrame:
        ...

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        ...

    def fetch_funding_rates(self, symbols=None) -> Dict[str, float]:
        ...

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Trade]) -> float:
        ...

    def load_leverage_tiers(self) -> Dict[str, Dict[str, float]]:
        ...
