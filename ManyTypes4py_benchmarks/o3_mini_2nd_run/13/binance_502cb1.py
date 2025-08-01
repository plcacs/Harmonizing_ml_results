#!/usr/bin/env python3
"""Binance exchange subclass"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
from pandas import DataFrame
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.binance_public_data import concat_safe, download_archive_ohlcv
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas, Tickers
from freqtrade.exchange.exchange_utils_timeframe import timeframe_to_msecs
from freqtrade.misc import deep_merge_dicts, json_load
from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts

logger = logging.getLogger(__name__)


class Binance(Exchange):
    _ft_has: Dict[str, Any] = {
        'stoploss_on_exchange': True,
        'stop_price_param': 'stopPrice',
        'stop_price_prop': 'stopPrice',
        'stoploss_order_types': {'limit': 'stop_loss_limit'},
        'order_time_in_force': ['GTC', 'FOK', 'IOC', 'PO'],
        'trades_pagination': 'id',
        'trades_pagination_arg': 'fromId',
        'trades_has_history': True,
        'l2_limit_range': [5, 10, 20, 50, 100, 500, 1000],
        'ws_enabled': True
    }
    _ft_has_futures: Dict[str, Any] = {
        'funding_fee_candle_limit': 1000,
        'stoploss_order_types': {'limit': 'stop', 'market': 'stop_market'},
        'order_time_in_force': ['GTC', 'FOK', 'IOC'],
        'tickers_have_price': False,
        'floor_leverage': True,
        'stop_price_type_field': 'workingType',
        'order_props_in_contracts': ['amount', 'cost', 'filled', 'remaining'],
        'stop_price_type_value_mapping': {PriceType.LAST: 'CONTRACT_PRICE', PriceType.MARK: 'MARK_PRICE'},
        'ws_enabled': False,
        'proxy_coin_mapping': {'BNFCR': 'USDC', 'BFUSD': 'USDT'}
    }
    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def get_proxy_coin(self) -> str:
        """
        Get the proxy coin for the given coin
        Falls back to the stake currency if no proxy coin is found
        :return: Proxy coin or stake currency
        """
        if self.margin_mode == MarginMode.CROSS:
            return self._config.get('proxy_coin', self._config['stake_currency'])
        return self._config['stake_currency']

    def get_tickers(self, symbols: Optional[List[str]] = None, *, cached: bool = False, market_type: Optional[str] = None) -> Tickers:
        tickers: Tickers = super().get_tickers(symbols=symbols, cached=cached, market_type=market_type)
        if self.trading_mode == TradingMode.FUTURES:
            bidsasks: Dict[str, Any] = self.fetch_bids_asks(symbols, cached=cached)
            tickers = deep_merge_dicts(bidsasks, tickers, allow_null_overrides=False)
        return tickers

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and (not self._config['dry_run']):
                position_side: Dict[str, Any] = self._api.fapiPrivateGetPositionSideDual()
                self._log_exchange_response('position_side_setting', position_side)
                assets_margin: Dict[str, Any] = self._api.fapiPrivateGetMultiAssetsMargin()
                self._log_exchange_response('multi_asset_margin', assets_margin)
                msg: str = ''
                if position_side.get('dualSidePosition') is True:
                    msg += "\nHedge Mode is not supported by freqtrade. Please change 'Position Mode' on your binance futures account."
                if assets_margin.get('multiAssetsMargin') is True and self.margin_mode != MarginMode.CROSS:
                    msg += "\nMulti-Asset Mode is not supported by freqtrade. Please change 'Asset Mode' on your binance futures account."
                if msg:
                    raise OperationalException(msg)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_historic_ohlcv(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, 
                             is_new_pair: bool = False, until_ms: Optional[int] = None) -> DataFrame:
        """
        Overwrite to introduce "fast new pair" functionality by detecting the pair's listing date
        Does not work for other exchanges, which don't return the earliest data when called with "0"
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        if is_new_pair:
            with self._loop_lock:
                x: Any = self.loop.run_until_complete(
                    self._async_get_candle_history(pair, timeframe, candle_type, 0)
                )
            if x and x[3] and x[3][0] and (x[3][0][0] > since_ms):
                since_ms = x[3][0][0]
                logger.info(f'Candle-data for {pair} available starting with {datetime.fromtimestamp(since_ms // 1000, tz=timezone.utc).isoformat()}.')
                if until_ms and since_ms >= until_ms:
                    logger.warning(f'No available candle-data for {pair} before {dt_from_ts(until_ms).isoformat()}')
                    return DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)
        if self._config['exchange'].get('only_from_ccxt', False) or not (
                candle_type == CandleType.SPOT and timeframe in ['1s', '1m', '3m', '5m'] or 
                (candle_type == CandleType.FUTURES and timeframe in ['1m', '3m', '5m', '15m', '30m'])):
            return super().get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=since_ms, candle_type=candle_type, is_new_pair=is_new_pair, until_ms=until_ms)
        else:
            return self.get_historic_ohlcv_fast(pair=pair, timeframe=timeframe, since_ms=since_ms, candle_type=candle_type, is_new_pair=is_new_pair, until_ms=until_ms)

    def get_historic_ohlcv_fast(self, pair: str, timeframe: str, since_ms: int, candle_type: CandleType, 
                                is_new_pair: bool = False, until_ms: Optional[int] = None) -> DataFrame:
        """
        Fastly fetch OHLCV data by leveraging https://data.binance.vision.
        """
        with self._loop_lock:
            df: DataFrame = self.loop.run_until_complete(
                download_archive_ohlcv(candle_type=candle_type, pair=pair, timeframe=timeframe, since_ms=since_ms, until_ms=until_ms, markets=self.markets)
            )
        if df.empty:
            rest_since_ms: int = since_ms
        else:
            rest_since_ms = dt_ts(df.iloc[-1].date) + timeframe_to_msecs(timeframe)
        if until_ms and rest_since_ms > until_ms:
            rest_df: DataFrame = DataFrame()
        else:
            rest_df = super().get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=rest_since_ms, candle_type=candle_type, is_new_pair=is_new_pair, until_ms=until_ms)
        all_df: DataFrame = concat_safe([df, rest_df])
        return all_df

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        """
        Funding fees are only charged at full hours (usually every 4-8h).
        Therefore a trade opening at 10:00:01 will not be charged a funding fee until the next hour.
        On binance, this cutoff is 15s.
        https://github.com/freqtrade/freqtrade/pull/5779#discussion_r740175931
        :param open_date: The open date for a trade
        :return: True if the date falls on a full hour, False otherwise
        """
        return open_date.minute == 0 and open_date.second < 15

    def fetch_funding_rates(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch funding rates for the given symbols.
        :param symbols: List of symbols to fetch funding rates for
        :return: Dict of funding rates for the given symbols
        """
        try:
            if self.trading_mode == TradingMode.FUTURES:
                rates: Dict[str, Any] = self._api.fetch_funding_rates(symbols)
                return rates
            return {}
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, 
                                  leverage: float, wallet_balance: float, open_trades: List[Any]) -> float:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!
        MARGIN: https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        PERPETUAL: https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93

        :param pair: Pair to calculate liquidation price for
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param leverage: Leverage used for this position.
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance
        :param open_trades: List of open trades in the same wallet
        :return: The calculated liquidation price.
        """
        cross_vars: float = 0.0
        mm_ratio, maintenance_amt = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.margin_mode == MarginMode.CROSS:
            mm_ex_1: float = 0.0
            upnl_ex_1: float = 0.0
            pairs: List[str] = [trade.pair for trade in open_trades]
            if self._config['runmode'] in ('live', 'dry_run'):
                funding_rates: Dict[str, Any] = self.fetch_funding_rates(pairs)
            for trade in open_trades:
                if trade.pair == pair:
                    continue
                if self._config['runmode'] in ('live', 'dry_run'):
                    mark_price: float = funding_rates[trade.pair]['markPrice']
                else:
                    mark_price = trade.open_rate
                mm_ratio1, maint_amnt1 = self.get_maintenance_ratio_and_amt(trade.pair, trade.stake_amount)
                maint_margin: float = trade.amount * mark_price * mm_ratio1 - maint_amnt1
                mm_ex_1 += maint_margin
                upnl_ex_1 += trade.amount * mark_price - trade.amount * trade.open_rate
            cross_vars = upnl_ex_1 - mm_ex_1
        side_1: int = -1 if is_short else 1
        if maintenance_amt is None:
            raise OperationalException(f'Parameter maintenance_amt is required by Binance.liquidation_pricefor {self.trading_mode}')
        if self.trading_mode == TradingMode.FUTURES:
            return (wallet_balance + cross_vars + maintenance_amt - side_1 * amount * open_rate) / (amount * mm_ratio - side_1 * amount)
        else:
            raise OperationalException('Freqtrade only supports isolated futures for leverage trading')

    def load_leverage_tiers(self) -> Dict[str, Any]:
        if self.trading_mode == TradingMode.FUTURES:
            if self._config['dry_run']:
                leverage_tiers_path: Path = Path(__file__).parent / 'binance_leverage_tiers.json'
                with leverage_tiers_path.open() as json_file:
                    return json_load(json_file)
            else:
                return self.get_leverage_tiers()
        else:
            return {}

    async def _async_get_trade_history_id_startup(self, pair: str, since: int) -> Tuple[List[Any], str]:
        """
        override for initial call

        Binance only provides a limited set of historic trades data.
        Using from_id=0, we can get the earliest available trades.
        So if we don't get any data with the provided "since", we can assume to
        download all available data.
        """
        t, from_id = await self._async_fetch_trades(pair, since=since)
        if not t:
            return ([], '0')
        return (t, from_id)