from typing import Any

# === Internal dependency: freqtrade.enums.CandleType ===
FUNDING_RATE: Any
FUTURES: Any
MARK: Any
SPOT: Any

# === Internal dependency: freqtrade.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_minutes
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_prev_date

# === Internal dependency: freqtrade.exchange.exchange ===
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_msecs

# === Internal dependency: freqtrade.util ===
from freqtrade.util.datetime_helpers import dt_floor_day
from freqtrade.util.datetime_helpers import dt_now
from freqtrade.util.datetime_helpers import dt_ts

# === Third-party dependency: pytest ===
# Used symbols: mark, skip

# === Internal dependency: tests.exchange_online.conftest ===
EXCHANGES = {'binance': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'use_ci_proxy': True, 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 1000, 'futures': True, 'futures_pair': 'BTC/USDT:USDT', ...}, 'binanceus': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 1000, 'futures': False, 'skip_ws_tests': True, 'sample_order': [{'exchange_response': {'symbol': 'SOLUSDT', 'orderId': 3551312894, 'orderListId': -1, 'clientOrderId': 'x-R4DD3S8297c73a11ccb9dc8f2811ba', 'transactTime': 1674493798550, 'price': '15.50000000', 'origQty': '1.10000000', 'executedQty': '0.00000000', ...}, 'pair': 'SOL/USDT', 'expected': {'symbol': 'SOL/USDT', 'orderId': '3551312894', 'timestamp': 1674493798550, 'datetime': '2023-03-25T15:49:58.550Z', 'price': 15.5, 'status': 'open', 'amount': 1.1}}]}, 'kraken': {'pair': 'BTC/USD', 'stake_currency': 'USD', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 720, 'leverage_tiers_public': False, 'leverage_in_spot_market': True, 'trades_lookback_hours': 12, ...}, 'kucoin': {'pair': 'XRP/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 1500, 'leverage_tiers_public': False, 'leverage_in_spot_market': True, 'sample_order': [{'exchange_response': {'id': '63d6742d0adc5570001d2bbf7'}, 'pair': 'SOL/USDT', 'expected': {'symbol': 'SOL/USDT', 'orderId': '3551312894', 'timestamp': 1674493798550, 'datetime': '2023-03-25T15:49:58.550Z', 'price': 15.5, 'status': 'open', 'amount': 1.1}}, {'exchange_response': {'id': '63d6742d0adc5570001d2bbf7', 'symbol': 'SOL-USDT', 'opType': 'DEAL', 'type': 'limit', 'side': 'buy', 'price': '15.5', 'size': '1.1', 'funds': '0', ...}, 'pair': 'SOL/USDT', 'expected': {'symbol': 'SOL/USDT', 'orderId': '3551312894', 'timestamp': 1674493798550, 'datetime': '2023-03-25T15:49:58.550Z', 'price': 15.5, 'status': 'open', 'amount': 1.1}}]}, 'gate': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 1000, 'futures': True, 'futures_pair': 'BTC/USDT:USDT', 'hasQuoteVolumeFutures': True, ...}, 'okx': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 300, 'futures': True, 'futures_pair': 'BTC/USDT:USDT', 'hasQuoteVolumeFutures': False, ...}, 'bybit': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'use_ci_proxy': True, 'timeframe': '1h', 'candle_count': 1000, 'futures_pair': 'BTC/USDT:USDT', 'futures': True, ...}, 'bitmart': {'pair': 'BTC/USDT', 'stake_currency': 'USDT', 'hasQuoteVolume': True, 'timeframe': '1h', 'candle_count': 200, 'orderbook_max_entries': 50}, ...}