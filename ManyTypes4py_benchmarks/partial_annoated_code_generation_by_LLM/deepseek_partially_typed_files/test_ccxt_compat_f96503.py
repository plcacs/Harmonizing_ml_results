"""
Tests in this file do NOT mock network calls, so they are expected to be fluky at times.

However, these tests should give a good idea to determine if a new exchange is
suitable to run with freqtrade.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import pytest
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange import timeframe_to_msecs
from freqtrade.util import dt_floor_day, dt_now, dt_ts
from tests.exchange_online.conftest import EXCHANGE_FIXTURE_TYPE, EXCHANGES

@pytest.mark.longrun
class TestCCXTExchange:

    def test_load_markets(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        markets: Dict[str, Dict] = exch.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exch.market_is_spot(markets[pair])

    def test_has_validations(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        exch.validate_ordertypes({'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit'})
        if exchangename == 'gate':
            return
        exch.validate_ordertypes({'entry': 'market', 'exit': 'market', 'stoploss': 'market'})

    def test_ohlcv_limit(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        expected_count: Optional[int] = EXCHANGES[exchangename].get('candle_count')
        if not expected_count:
            pytest.skip('No expected candle count for exchange')
        assert exch.ohlcv_candle_limit('1m', CandleType.SPOT) == expected_count

    def test_ohlcv_limit_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange_futures
        expected_count: Optional[int] = EXCHANGES[exchangename].get('candle_count')
        if not expected_count:
            pytest.skip('No expected candle count for exchange')
        assert exch.ohlcv_candle_limit('1m', CandleType.SPOT) == expected_count

    def test_load_markets_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE) -> None:
        (exchange, exchangename) = exchange_futures
        pair: str = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        markets: Dict[str, Dict] = exchange.markets
        assert pair in markets
        assert isinstance(markets[pair], dict)
        assert exchange.market_is_future(markets[pair])

    def test_ccxt_order_parse(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchange_name) = exchange
        if (orders := EXCHANGES[exchange_name].get('sample_order')):
            for order in orders:
                pair: str = order['pair']
                exchange_response: Dict[str, Any] = order['exchange_response']
                market: Dict[str, Any] = exch._api.markets[pair]
                po: Dict[str, Any] = exch._api.parse_order(exchange_response, market)
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

    def test_ccxt_my_trades_parse(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchange_name) = exchange
        if (trades := EXCHANGES[exchange_name].get('sample_my_trades')):
            pair: str = 'SOL/USDT'
            for trade in trades:
                po: Dict[str, Any] = exch._api.parse_trade(trade)
                assert po['symbol'] == pair
                assert isinstance(po['id'], str)
                assert isinstance(po['side'], str)
                assert isinstance(po['amount'], float)
                assert isinstance(po['price'], float)
                assert isinstance(po['datetime'], str)
                assert isinstance(po['timestamp'], int)
                if (fees := po.get('fees')):
                    assert isinstance(fees, list)
                    for fee in fees:
                        assert isinstance(fee, dict)
                        assert isinstance(fee['cost'], float)
                        assert isinstance(fee['currency'], str)
        else:
            pytest.skip(f'No sample Trades available for exchange {exchange_name}')

    def test_ccxt_balances_parse(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchange_name) = exchange
        if (balance_response := EXCHANGES[exchange_name].get('sample_balances')):
            balances: Dict[str, Dict[str, float]] = exch._api.parse_balance(balance_response['exchange_response'])
            expected: Dict[str, Dict[str, float]] = balance_response['expected']
            for (currency, balance) in expected.items():
                assert currency in balances
                assert isinstance(balance, dict)
                assert balance == balances[currency]
            pass
        else:
            pytest.skip(f'No sample Balances available for exchange {exchange_name}')

    def test_ccxt_fetch_tickers(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        tickers: Dict[str, Dict[str, Any]] = exch.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert 'bid' in tickers[pair]
        if EXCHANGES[exchangename].get('tickers_have_bid_ask'):
            assert tickers[pair]['bid'] is not None
            assert tickers[pair]['ask'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_tickers_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange_futures
        if not exch or exchangename in 'gate':
            return
        pair: str = EXCHANGES[exchangename]['pair']
        pair = EXCHANGES[exchangename].get('futures_pair', pair)
        tickers: Dict[str, Dict[str, Any]] = exch.get_tickers()
        assert pair in tickers
        assert 'ask' in tickers[pair]
        assert tickers[pair]['ask'] is not None
        assert 'bid' in tickers[pair]
        assert tickers[pair]['bid'] is not None
        assert 'quoteVolume' in tickers[pair]
        if EXCHANGES[exchangename].get('hasQuoteVolumeFutures'):
            assert tickers[pair]['quoteVolume'] is not None

    def test_ccxt_fetch_ticker(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        ticker: Dict[str, Any] = exch.fetch_ticker(pair)
        assert 'ask' in ticker
        assert 'bid' in ticker
        if EXCHANGES[exchangename].get('tickers_have_bid_ask'):
            assert ticker['ask'] is not None
            assert ticker['bid'] is not None
        assert 'quoteVolume' in ticker
        if EXCHANGES[exchangename].get('hasQuoteVolume'):
            assert ticker['quoteVolume'] is not None

    def test_ccxt_fetch_l2_orderbook(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        l2: Dict[str, List[List[float]]] = exch.fetch_l2_order_book(pair)
        orderbook_max_entries: Optional[int] = EXCHANGES[exchangename].get('orderbook_max_entries')
        assert 'asks' in l2
        assert 'bids' in l2
        assert len(l2['asks']) >= 1
        assert len(l2['bids']) >= 1
        l2_limit_range: Optional[List[int]] = exch._ft_has['l2_limit_range']
        l2_limit_range_required: bool = exch._ft_has['l2_limit_range_required']
        if exchangename == 'gate':
            return
        for val in [1, 2, 5, 25, 50, 100]:
            if orderbook_max_entries and val > orderbook_max_entries:
                continue
            l2 = exch.fetch_l2_order_book(pair, val)
            if not l2_limit_range or val in l2_limit_range:
                if val > 50:
                    assert val - 5 < len(l2['asks']) <= val
                    assert val - 5 < len(l2['bids']) <= val
                else:
                    assert len(l2['asks']) == val
                    assert len(l2['bids']) == val
            else:
                next_limit: Optional[int] = exch.get_next_limit_in_list(val, l2_limit_range, l2_limit_range_required)
                if next_limit is None:
                    assert len(l2['asks']) > 100
                    assert len(l2['asks']) > 100
                elif next_limit > 200:
                    assert len(l2['asks']) > 200
                    assert len(l2['asks']) > 200
                else:
                    assert len(l2['asks']) == next_limit
                    assert len(l2['asks']) == next_limit

    def test_ccxt_fetch_ohlcv(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe, CandleType.SPOT)
        ohlcv: Dict[Tuple[str, str, CandleType], Any] = exch.refresh_latest_ohlcv([pair_tf])
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        assert len(exch.klines(pair_tf)) > exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) * 0.9
        now: datetime = datetime.now(timezone.utc) - timedelta(minutes=timeframe_to_minutes(timeframe) * 2)
        assert exch.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)

    def test_ccxt_fetch_ohlcv_startdate(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        """
        Test that pair data starts at the provided startdate
        """
        (exch, exchangename) = exchange
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = '1d'
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe, CandleType.SPOT)
        since_ms: int = dt_ts(dt_floor_day(dt_now()) - timedelta(days=6))
        ohlcv: Dict[Tuple[str, str, CandleType], Any] = exch.refresh_latest_ohlcv([pair_tf], since_ms=since_ms)
        assert isinstance(ohlcv, dict)
        assert len(ohlcv[pair_tf]) == len(exch.klines(pair_tf))
        now: datetime = datetime.now(timezone.utc) - timedelta(minutes=timeframe_to_minutes(timeframe) * 2)
        assert exch.klines(pair_tf).iloc[-1]['date'] >= timeframe_to_prev_date(timeframe, now)
        assert exch.klines(pair_tf)['date'].astype(int).iloc[0] // 1000000.0 == since_ms

    def ccxt__async_get_candle_history(self, exchange: Any, exchangename: str, pair: str, timeframe: str, candle_type: CandleType, factor: float = 0.9) -> None:
        timeframe_ms: int = timeframe_to_msecs(timeframe)
        now: datetime = timeframe_to_prev_date(timeframe, datetime.now(timezone.utc))
        for offset in (360, 120, 30, 10, 5, 2):
            since: datetime = now - timedelta(days=offset)
            since_ms: int = int(since.timestamp() * 1000)
            res: Tuple[str, str, CandleType, List[List[Union[int, float]]]] = exchange.loop.run_until_complete(exchange._async_get_candle_history(pair=pair, timeframe=timeframe, since_ms=since_ms, candle_type=candle_type))
            assert res
            assert res[0] == pair
            assert res[1] == timeframe
            assert res[2] == candle_type
            candles: List[List[Union[int, float]]] = res[3]
            candle_count: float = exchange.ohlcv_candle_limit(timeframe, candle_type, since_ms) * factor
            candle_count1: float = (now.timestamp() * 1000 - since_ms) // timeframe_ms * factor
            assert len(candles) >= min(candle_count, candle_count1), f'{len(candles)} < {candle_count} in {timeframe}, Offset: {offset} {factor}'
            assert candles[0][0] == since_ms or since_ms + timeframe_ms

    def test_ccxt__async_get_candle_history(self, exchange: EXCHANGE_FIXTURE_TYPE) -> None:
        (exc, exchangename) = exchange
        if not exc._ft_has['ohlcv_has_history']:
            pytest.skip('Exchange does not support candle history')
        pair: str = EXCHANGES[exchangename]['pair']
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        self.ccxt__async_get_candle_history(exc, exchangename, pair, timeframe, CandleType.SPOT)

    @pytest.mark.parametrize('candle_type', [CandleType.FUTURES, CandleType.FUNDING_RATE, CandleType.MARK])
    def test_ccxt__async_get_candle_history_futures(self, exchange_futures: EXCHANGE_FIXTURE_TYPE, candle_type: CandleType) -> None:
        (exchange, exchangename) = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        timeframe: str = EXCHANGES[exchangename]['timeframe']
        if candle_type == CandleType.FUNDING_RATE:
            timeframe = exchange._ft_has.get('funding_fee_timeframe', exchange._ft_has['mark_ohlcv_timeframe'])
        self.ccxt__async_get_candle_history(exchange, exchangename, pair=pair, timeframe=timeframe, candle_type=candle_type)

    def test_ccxt_fetch_funding_rate_history(self, exchange_futures: EXCHANGE_FIXTURE_TYPE) -> None:
        (exchange, exchangename) = exchange_futures
        pair: str = EXCHANGES[exchangename].get('futures_pair', EXCHANGES[exchangename]['pair'])
        since: int = int((datetime.now(timezone.utc) - timedelta(days=5)).timestamp() * 1000)
        timeframe_ff: str = exchange._ft_has.get('funding_fee_timeframe', exchange._ft_has['mark_ohlcv_timeframe'])
        pair_tf: Tuple[str, str, CandleType] = (pair, timeframe_ff, CandleType.FUNDING_RATE)
        funding_ohlcv: Dict[Tuple[str, str, CandleType], Any] = exchange.refresh_latest_ohlcv([pair_tf], since_ms=since, drop_incomplete=False)
        assert isinstance(funding_ohlcv, dict)
        rate: Any = funding_ohlcv[pair_tf]
        this_hour: datetime = timeframe_to_prev_date(timeframe_ff)
        hour1: datetime = timeframe_to_prev_date(timeframe_ff, this_hour - timedelta(minutes=1))
        hour2: datetime = timeframe_to_prev_date(timeframe_ff, hour1 - timedelta(minutes=1))
        hour3: datetime = timeframe_to_prev_date(timeframe_ff, hour2 - timedelta(minutes=1))
        val0: float = rate[rate['date'] == this_hour].iloc[0]['