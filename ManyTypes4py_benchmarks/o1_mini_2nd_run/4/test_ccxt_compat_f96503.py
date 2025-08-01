"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange import timeframe_to_msecs
from freqtrade.util import dt_floor_day, dt_now, dt_ts
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE, EXCHANGES
import pandas as pd

class ExchangeAPI:
    markets: Dict[str, Any]
    _api: Any
    loop: Any
    _ft_has: Dict[str, Any]

    def market_is_spot(self, market: Dict[str, Any]) -> bool:
        ...

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        ...

    def validate_ordertypes(self, ordertypes: Dict[str, str]) -> None:
        ...

    def ohlcv_candle_limit(self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        ...

    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        ...

    def fetch_ticker(self, pair: str) -> Dict[str, Any]:
        ...

    def fetch_l2_order_book(self, pair: str, limit: Optional[int] = None) -> Dict[str, Any]:
        ...

    def refresh_latest_ohlcv(self, pair_tfs: List[Tuple[str, str, CandleType]], since_ms: Optional[int] = None, drop_incomplete: bool = False) -> Dict[Tuple[str, str, CandleType], pd.DataFrame]:
        ...

    def klines(self, pair_tf: Tuple[str, str, CandleType]) -> pd.DataFrame:
        ...

    def get_next_limit_in_list(self, val: int, limit_range: List[int], limit_required: bool) -> Optional[int]:
        ...

    def get_fee(self, pair: str, order_type: str, side: str) -> float:
        ...

    def get_max_leverage(self, pair: str, some_parameter: int) -> float:
        ...

    def get_contract_size(self, pair: str) -> float:
        ...

    def load_leverage_tiers(self) -> Dict[str, List[Dict[str, float]]]:
        ...

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: List[Any]) -> float:
        ...

    async def _async_get_candle_history(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType) -> Tuple[str, str, CandleType, List[List[Union[int, float]]]]:
        ...

    async def _async_get_trade_history(self, pair: str, since: int, param1: Any, param2: Any) -> Tuple[str, List[Any]]:
        ...

class Exchange:
    _api: ExchangeAPI
    loop: Any
    _ft_has: Dict[str, Any]

@pytest.mark.longrun
class TestCCXTExchange:

    def test_load_markets(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        markets: Dict[str, Any] = exch._api.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exch._api.market_is_spot(markets[pair])

    def test_has_validations(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        exch._api.validate_ordertypes({'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit'})
        if exchangename == 'gate':
            return
        exch._api.validate_ordertypes({'entry': 'market', 'exit': 'market', 'stoploss': 'market'})

    def test_ohlcv_limit(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        expected_count: Optional[int] = EXCHANGES[exchangename].get('candle_count')
        if not expected_count:
            pytest.skip('No expected candle count for exchange')
        assert exch._api.ohlcv_candle_limit('1m', CandleType.SPOT) == expected_count

    def test_ohlcv_limit_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange_futures
        expected_count: Optional[int] = EXCHANGES[exchangename].get('candle_count')
        if not expected_count:
            pytest.skip('No expected candle count for exchange')
        assert exch._api.ohlcv_candle_limit('1m', CandleType.SPOT) == expected_count

    def test_load_markets_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exchange, exchangename = exchange_futures
        pair: str = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        markets: Dict[str, Any] = exchange._api.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exchange._api.market_is_future(markets[pair])

    def test_ccxt_order_parse(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchange_name = exchange
        sample_orders: Optional[List[Dict[str, Any]]] = EXCHANGES[exchange_name].get('sample_order')
        if sample_orders:
            for order in sample_orders:
                pair: str = order['pair']
                exchange_response: Dict[str, Any] = order['exchange_response']
                market: Dict[str, Any] = exch._api._api.markets[pair]
                po: Dict[str, Any] = exch._api._api.parse_order(exchange_response, market)
                expected: Dict[str, Any] = order['expected']
                assert isinstance(po['id'], str)
                assert po['id'] is not None
                if len(exchange_response.keys()) < 5:
                    assert po['status'] is None
                    continue
                assert po['timestamp'] == expected['timestamp']
                assert isinstance(po['datetime'], str)
                assert isinstance(po['timestamp'], int)
                assert isinstance(po['price'], float)
                assert po['price'] == expected['price']
                if po['status'] == 'closed':
                    assert isinstance(po['average'], float)
                    assert po['average'] == 15.5
                assert po['symbol'] == pair
                assert isinstance(po['amount'], float)
                assert po['amount'] == expected['amount']
                assert isinstance(po['status'], str)
        else:
            pytest.skip(f'No sample order available for exchange {exchange_name}')

    def test_ccxt_my_trades_parse(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchange_name = exchange
        sample_trades: Optional[List[Any]] = EXCHANGES[exchange_name].get('sample_my_trades')
        if sample_trades:
            pair: str = 'SOL/USDT'
            for trade in sample_trades:
                po: Dict[str, Any] = exch._api._api.parse_trade(trade)
                assert po['symbol'] == pair
                assert isinstance(po['id'], str)
                assert isinstance(po['side'], str)
                assert isinstance(po['amount'], float)
                assert isinstance(po['price'], float)
                assert isinstance(po['datetime'], str)
                assert isinstance(po['timestamp'], int)
                fees: Optional[List[Dict[str, Any]]] = po.get('fees')
                if fees:
                    assert isinstance(fees, list)
                    for fee in fees:
                        assert isinstance(fee, dict)
                        assert isinstance(fee['cost'], float)
                        assert isinstance(fee['currency'], str)
        else:
            pytest.skip(f'No sample Trades available for exchange {exchange_name}')

    def test_ccxt_balances_parse(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchange_name = exchange
        balance_response: Optional[Dict[str, Any]] = EXCHANGES[exchange_name].get('sample_balances')
        if balance_response:
            balances: Dict[str, Dict[str, Any]] = exch._api._api.parse_balance(balance_response['exchange_response'])
            expected: Dict[str, Dict[str, Any]] = balance_response['expected']
            for currency, balance in expected.items():
                assert currency in balances
                assert isinstance(balance, dict)
                assert balance == balances[currency]
            pass
        else:
            pytest.skip(f'No sample Balances available for exchange {exchange_name}')

    def test_ccxt_fetch_tickers(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        tickers: Dict[str, Dict[str, Any]] = exch._api.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert 'bid' in tickers[pair]
        if EXCHANGES[exchangename].get('tickers_have_bid_ask'):
            assert tickers[pair]['bid'] is not None
            assert tickers[pair]['ask'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_tickers_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange_futures
        if not exch or exchangename in 'gate':
            return
        pair: str = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        tickers: Dict[str, Dict[str, Any]] = exch._api.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert tickers[pair]['ask'] is not None
        assert 'bid' in tickers[pair]
        assert tickers[pair]['bid'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolumeFutures'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_ticker(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        ticker: Dict[str, Any] = exch._api.fetch_ticker(pair)
        assert 'ask' in ticker
        assert 'bid' in ticker
        if EXCHANGES[exchangename].get('tickers_have_bid_ask'):
            assert ticker['ask'] is not None
            assert ticker['bid'] is not None
        assert 'quoteVolume' in ticker
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert ticker['quoteVolume'] is not None

    def test_ccxt_fetch_l2_orderbook(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        l2: Dict[str, Any] = exch._api.fetch_l2_order_book(pair)
        orderbook_max_entries: Optional[int] = EXCHANGES[exchangename].get('orderbook_max_entries')
        assert 'asks' in l2
        assert 'bids' in l2
        assert len(l2['asks']) >= 1
        assert len(l2['bids']) >= 1
        l2_limit_range: List[int] = exch._api._ft_has['l2_limit_range']
        l2_limit_range_required: bool = exch._api._ft_has['l2_limit_range_required']
        if exchangename == 'gate':
            return
        for val in [1, 2, 5, 25, 50, 100]:
            if orderbook_max_entries and val > orderbook_max_entries:
                continue
            l2 = exch._api.fetch_l2_order_book(pair, val)
            if not l2_limit_range or val in l2_limit_range:
                if val > 50:
                    assert val - 5 < len(l2['asks']) <= val
                    assert val - 5 < len(l2['bids']) <= val
                else:
                    assert len(l2['asks']) == val
                    assert len(l2['bids']) == val
            else:
                next_limit: Optional[int] = exch._api.get_next_limit_in_list(val, l2_limit_range, l2_limit_range_required)
                if next_limit is None:
                    assert len(l2['asks']) > 100
                    assert len(l2['bids']) > 100
                elif next_limit > 200:
                    assert len(l2['asks']) > 200
                    assert len(l2['bids']) > 200
                else:
                    assert len(l2['asks']) == next_limit
                    assert len(l2['bids']) == next_limit

    def test_ccxt_fetch_ohlcv(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe, CandleType.SPOT)
        ohlcv: Dict[Tuple[str, str, CandleType], pd.DataFrame] = exch._api.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch._api.klines(pair_tf))
        assert len(exch._api.klines(pair_tf)) > exch._api.ohlcv_candle_limit(timeframe, CandleType.SPOT) * 0.9
        now: datetime = datetime.now(timezone.utc) - timedelta(minutes=timeframe_to_minutes(timeframe) * 2)
        assert exch._api.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)

    def test_ccxt_fetch_ohlcv_startdate(self, exchange: Tuple[Exchange, str]) -> None:
        """
        Test that pair data starts at the provided startdate
        """
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = '1d'
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe, CandleType.SPOT)
        since_ms: int = dt_ts(dt_floor_day(dt_now()) - timedelta(days=6))
        ohlcv: Dict[Tuple[str, str, CandleType], pd.DataFrame] = exch._api.refresh_latest_ohlcv([pair_tf], since_ms=since_ms)
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch._api.klines(pair_tf))
        now: datetime = datetime.now(timezone.utc) - timedelta(minutes=timeframe_to_minutes(timeframe) * 2)
        assert exch._api.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)
        assert exch._api.klines(pair_tf)['date'].astype(int).iloc[0] // 1000000.0 == since_ms

    def ccxt__async_get_candle_history(
        self, 
        exchange: Exchange, 
        exchangename: str, 
        pair: str, 
        timeframe: str, 
        candle_type: CandleType, 
        factor: float = 0.9
    ) -> None:
        timeframe_ms: int = timeframe_to_msecs(timeframe)
        now: datetime = timeframe_to_prev_date(timeframe, datetime.now(timezone.utc))
        for offset in (360, 120, 30, 10, 5, 2):
            since: datetime = now - timedelta(days=offset)
            since_ms: int = int(since.timestamp() * 1000)
            res: Tuple[str, str, CandleType, List[List[Union[int, float]]]] = exchange.loop.run_until_complete(
                exchange._api._async_get_candle_history(pair=pair, timeframe=timeframe, since_ms=since_ms, candle_type=candle_type)
            )
            assert res
            assert res[0] == pair
            assert res[1] == timeframe
            assert res[2] == candle_type
            candles: List[List[Union[int, float]]] = res[3]
            candle_count: float = exchange._api.ohlcv_candle_limit(timeframe, candle_type, since_ms) * factor
            candle_count1: float = (now.timestamp() * 1000 - since_ms) // timeframe_ms * factor
            assert len(candles) >= min(candle_count, candle_count1), f'{len(candles)} < {candle_count} in {timeframe}, Offset: {offset} {factor}'
            assert candles[0][0] == since_ms or since_ms + timeframe_ms

    def test_ccxt__async_get_candle_history(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        if not exch._api._ft_has.get('ohlcv_has_history', False):
            pytest.skip('Exchange does not support candle history')
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        self.ccxt__async_get_candle_history(exch, exchangename, pair, timeframe, CandleType.SPOT)

    @pytest.mark.parametrize('candle_type', [CandleType.FUTURES, CandleType.FUNDING_RATE, CandleType.MARK])
    def test_ccxt__async_get_candle_history_futures(
        self, 
        exchange_futures: Tuple[Exchange, str], 
        candle_type: CandleType
    ) -> None:
        exchange, exchangename = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        if candle_type == CandleType.FUNDING_RATE:
            timeframe = exchange._api._ft_has.get('funding_fee_timeframe', exchange._api._ft_has['mark_ohlcv_timeframe'])
        self.ccxt__async_get_candle_history(exchange, exchangename, pair=pair, timeframe=timeframe, candle_type=candle_type)

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exchange, exchangename = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since: int = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        timeframe_ff: str = exchange._api._ft_has.get('funding_fee_timeframe', exchange._api._ft_has.get('mark_ohlcv_timeframe', '1h'))
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe_ff, CandleType.FUNDING_RATE)
        funding_ohlcv: Dict[Tuple[str, str, CandleType], pd.DataFrame] = exchange._api.refresh_latest_ohlcv([pair_tf], since_ms=since, drop_incomplete=False)
        assert isinstance(funding_ohlcv, dict)
        rate: pd.DataFrame = funding_ohlcv[pair_tf]
        this_hour: datetime = timeframe_to_prev_date(timeframe_ff)
        hour1: datetime = timeframe_to_prev_date(timeframe_ff, this_hour - timedelta(minutes=1))
        hour2: datetime = timeframe_to_prev_date(timeframe_ff, hour1 - timedelta(minutes=1))
        hour3: datetime = timeframe_to_prev_date(timeframe_ff, hour2 - timedelta(minutes=1))
        val0: float = rate[rate['date'] == this_hour].iloc[0]['open']
        val1: float = rate[rate['date'] == hour1].iloc[0]['open']
        val2: float = rate[rate['date'] == hour2].iloc[0]['open']
        val3: float = rate[rate['date'] == hour3].iloc[0]['open']
        assert val0 != 0.0 or val1 != 0.0 or val2 != 0.0 or (val3 != 0.0)
        assert rate['open'].max() != 0.0 or rate['open'].min() != 0.0 or rate['open'].min() != rate['open'].max()

    def test_ccxt_fetch_mark_price_history(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exchange, exchangename = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since: int = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        pair_tf: Tuple[str, str, CandleType] = (pair, '1h', CandleType.MARK)
        mark_ohlcv: Dict[Tuple[str, str, CandleType], pd.DataFrame] = exchange._api.refresh_latest_ohlcv([pair_tf], since_ms=since, drop_incomplete=False)
        assert isinstance(mark_ohlcv, dict)
        expected_tf: str = '1h'
        mark_candles: pd.DataFrame = mark_ohlcv[pair_tf]
        this_hour: datetime = timeframe_to_prev_date(expected_tf)
        prev_hour: datetime = timeframe_to_prev_date(expected_tf, this_hour - timedelta(minutes=1))
        assert mark_candles[mark_candles['date'] == prev_hour].iloc[0]['open'] != 0.0
        assert mark_candles[mark_candles['date'] == this_hour].iloc[0]['open'] != 0.0

    def test_ccxt__calculate_funding_fees(self, exchange_futures: Tuple[Exchange, str]) -> None:
        exchange, exchangename = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since: datetime = datetime.now(timezone.utc) - timedelta(days=5)
        funding_fee: float = exchange._api._fetch_and_calculate_funding_fees(pair, 20, is_short=False, open_date=since)
        assert isinstance(funding_fee, float)

    def test_ccxt__async_get_trade_history(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        lookback: Optional[int] = EXCHANGES[exchangename].get('trades_lookback_hours')
        if not lookback:
            pytest.skip('test_fetch_trades not enabled for this exchange')
        pair: str = EXCHANGES[exchangename]['pair']
        since: int = int((datetime.now(timezone.utc) - timedelta(hours=lookback)).timestamp() * 1000)
        res: Tuple[str, List[Any]] = exch.loop.run_until_complete(
            exch._api._async_get_trade_history(pair, since, None, None)
        )
        assert len(res) == 2
        res_pair: str = res[0]
        res_trades: List[Any] = res[1]
        assert res_pair == pair
        assert isinstance(res_trades, list)
        assert res_trades[0][0] >= since
        assert len(res_trades) > 1200

    def test_ccxt_get_fee(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        threshold: float = 0.01
        assert 0 < exch._api.get_fee(pair, 'limit', 'buy') < threshold
        assert 0 < exch._api.get_fee(pair, 'limit', 'sell') < threshold
        assert 0 < exch._api.get_fee(pair, 'market', 'buy') < threshold
        assert 0 < exch._api.get_fee(pair, 'market', 'sell') < threshold

    def test_ccxt_get_max_leverage_spot(self, exchange: Tuple[Exchange, str]) -> None:
        spot, spot_name = exchange
        if spot:
            leverage_in_market_spot: Optional[bool] = EXCHANGES[spot_name].get('leverage_in_spot_market')
            if leverage_in_market_spot:
                spot_pair: str = EXCHANGES[spot_name].get('pair', EXCHANGES[spot_name]['pair'])
                spot_leverage: Union[float, int] = spot._api.get_max_leverage(spot_pair, 20)
                assert isinstance(spot_leverage, (float, int))
                assert spot_leverage >= 1.0

    def test_ccxt_get_max_leverage_futures(self, exchange_futures: Tuple[Exchange, str]) -> None:
        futures, futures_name = exchange_futures
        leverage_tiers_public: Optional[bool] = EXCHANGES[futures_name].get('leverage_tiers_public')
        if leverage_tiers_public:
            futures_pair: str = EXCHANGES[futures_name].get('futures_pair', EXCHANGES[futures_name]['pair'])
            futures_leverage: Union[float, int] = futures._api.get_max_leverage(futures_pair, 20)
            assert isinstance(futures_leverage, (float, int))
            assert futures_leverage >= 1.0

    def test_ccxt_get_contract_size(self, exchange_futures: Tuple[Exchange, str]) -> None:
        futures, futures_name = exchange_futures
        futures_pair: str = EXCHANGES[futures_name].get('futures_pair', EXCHANGES[futures_name]['pair'])
        contract_size: Union[float, int] = futures._api.get_contract_size(futures_pair)
        assert isinstance(contract_size, (float, int))
        assert contract_size >= 0.0

    def test_ccxt_load_leverage_tiers(self, exchange_futures: Tuple[Exchange, str]) -> None:
        futures, futures_name = exchange_futures
        if EXCHANGES[futures_name].get('leverage_tiers_public'):
            leverage_tiers: Dict[str, List[Dict[str, float]]] = futures._api.load_leverage_tiers()
            futures_pair: str = EXCHANGES[futures_name].get('futures_pair', EXCHANGES[futures_name]['pair'])
            assert isinstance(leverage_tiers, dict)
            assert futures_pair in leverage_tiers
            pair_tiers: List[Dict[str, float]] = leverage_tiers[futures_pair]
            assert len(pair_tiers) > 0
            oldLeverage: float = float('inf')
            oldMaintenanceMarginRate: float = -1
            oldminNotional: float = -1
            oldmaxNotional: float = -1
            for tier in pair_tiers:
                for key in ['maintenanceMarginRate', 'minNotional', 'maxNotional', 'maxLeverage']:
                    assert key in tier
                    assert tier[key] >= 0.0
                assert tier['maxNotional'] > tier['minNotional']
                assert tier['maxLeverage'] <= oldLeverage
                assert tier['maintenanceMarginRate'] >= oldMaintenanceMarginRate
                assert tier['minNotional'] > oldminNotional
                assert tier['maxNotional'] > oldmaxNotional
                oldLeverage = tier['maxLeverage']
                oldMaintenanceMarginRate = tier['maintenanceMarginRate']
                oldminNotional = tier['minNotional']
                oldmaxNotional = tier['maxNotional']

    def test_ccxt_dry_run_liquidation_price(self, exchange_futures: Tuple[Exchange, str]) -> None:
        futures, futures_name = exchange_futures
        if EXCHANGES[futures_name].get('leverage_tiers_public'):
            futures_pair: str = EXCHANGES[futures_name].get('futures_pair', EXCHANGES[futures_name]['pair'])
            liquidation_price: float = futures._api.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
                open_trades=[]
            )
            assert isinstance(liquidation_price, float)
            assert liquidation_price >= 0.0
            liquidation_price = futures._api.dry_run_liquidation_price(
                pair=futures_pair,
                open_rate=40000,
                is_short=False,
                amount=100,
                stake_amount=100,
                leverage=5,
                wallet_balance=100,
                open_trades=[]
            )
            assert isinstance(liquidation_price, float)
            assert liquidation_price >= 0.0

    def test_ccxt_get_max_pair_stake_amount(self, exchange_futures: Tuple[Exchange, str]) -> None:
        futures, futures_name = exchange_futures
        futures_pair: str = EXCHANGES[futures_name].get('futures_pair', EXCHANGES[futures_name]['pair'])
        max_stake_amount: float = futures._api.get_max_pair_stake_amount(futures_pair, 40000)
        assert isinstance(max_stake_amount, float)
        assert max_stake_amount >= 0.0

    def test_private_method_presence(self, exchange: Tuple[Exchange, str]) -> None:
        exch, exchangename = exchange
        for method in EXCHANGES[exchangename].get('private_methods', []):
            assert hasattr(exch._api, method)
