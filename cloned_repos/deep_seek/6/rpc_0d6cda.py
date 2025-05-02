"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from collections.abc import Generator, Sequence
from datetime import date, datetime, timedelta, timezone
from math import isnan
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Literal
import psutil
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzlocal
from numpy import inf, int64, mean, nan
from pandas import DataFrame, NaT
from sqlalchemy import func, select
from freqtrade import __version__
from freqtrade.configuration.timerange import TimeRange
from freqtrade.constants import CANCEL_REASON, DEFAULT_DATAFRAME_COLUMNS, Config
from freqtrade.data.history import load_data
from freqtrade.data.metrics import DrawDownResult, calculate_expectancy, calculate_max_drawdown
from freqtrade.enums import CandleType, ExitCheckTuple, ExitType, MarketDirection, SignalDirection, State, TradingMode
from freqtrade.exceptions import ExchangeError, PricingError
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_msecs
from freqtrade.exchange.exchange_utils import price_to_precision
from freqtrade.loggers import bufferHandler
from freqtrade.persistence import KeyStoreKeys, KeyValueStore, PairLocks, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.rpc.rpc_types import RPCSendMsg
from freqtrade.util import decimals_per_coin, dt_from_ts, dt_humanize_delta, dt_now, dt_ts, dt_ts_def, format_date, shorten_date
from freqtrade.wallets import PositionWallet, Wallet
logger = logging.getLogger(__name__)

class RPCException(Exception):
    """
    Should be raised with a rpc-formatted message in an _rpc_* method
    if the required state is wrong, i.e.:

    raise RPCException('*Status:* `no active trade`')
    """

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self) -> str:
        return self.message

    def __json__(self) -> Dict[str, str]:
        return {'msg': self.message}

class RPCHandler:

    def __init__(self, rpc: 'RPC', config: Config) -> None:
        """
        Initializes RPCHandlers
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._rpc = rpc
        self._config = config

    @property
    def name(self) -> str:
        """Returns the lowercase name of the implementation"""
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup pending module resources"""

    @abstractmethod
    def send_msg(self, msg: RPCSendMsg) -> None:
        """Sends a message to all registered rpc modules"""

class RPC:
    """
    RPC class can be used to have extra feature, like bot data, and access to DB data
    """
    _fiat_converter: Optional[CryptoToFiatConverter] = None
    if TYPE_CHECKING:
        from freqtrade.freqtradebot import FreqtradeBot

    def __init__(self, freqtrade: 'FreqtradeBot') -> None:
        """
        Initializes all enabled rpc modules
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        self._freqtrade = freqtrade
        self._config = freqtrade.config
        if self._config.get('fiat_display_currency'):
            self._fiat_converter = CryptoToFiatConverter(self._config)

    @staticmethod
    def _rpc_show_config(config: Config, botstate: State, strategy_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a dict of config options.
        Explicitly does NOT return the full config to avoid leakage of sensitive
        information via rpc.
        """
        val = {'version': __version__, 'strategy_version': strategy_version, 'dry_run': config['dry_run'], 'trading_mode': config.get('trading_mode', 'spot'), 'short_allowed': config.get('trading_mode', 'spot') != 'spot', 'stake_currency': config['stake_currency'], 'stake_currency_decimals': decimals_per_coin(config['stake_currency']), 'stake_amount': str(config['stake_amount']), 'available_capital': config.get('available_capital'), 'max_open_trades': config.get('max_open_trades', 0) if config.get('max_open_trades', 0) != float('inf') else -1, 'minimal_roi': config['minimal_roi'].copy() if 'minimal_roi' in config else {}, 'stoploss': config.get('stoploss'), 'stoploss_on_exchange': config.get('order_types', {}).get('stoploss_on_exchange', False), 'trailing_stop': config.get('trailing_stop'), 'trailing_stop_positive': config.get('trailing_stop_positive'), 'trailing_stop_positive_offset': config.get('trailing_stop_positive_offset'), 'trailing_only_offset_is_reached': config.get('trailing_only_offset_is_reached'), 'unfilledtimeout': config.get('unfilledtimeout'), 'use_custom_stoploss': config.get('use_custom_stoploss'), 'order_types': config.get('order_types'), 'bot_name': config.get('bot_name', 'freqtrade'), 'timeframe': config.get('timeframe'), 'timeframe_ms': timeframe_to_msecs(config['timeframe']) if 'timeframe' in config else 0, 'timeframe_min': timeframe_to_minutes(config['timeframe']) if 'timeframe' in config else 0, 'exchange': config['exchange']['name'], 'strategy': config['strategy'], 'force_entry_enable': config.get('force_entry_enable', False), 'exit_pricing': config.get('exit_pricing', {}), 'entry_pricing': config.get('entry_pricing', {}), 'state': str(botstate), 'runmode': config['runmode'].value, 'position_adjustment_enable': config.get('position_adjustment_enable', False), 'max_entry_position_adjustment': config.get('max_entry_position_adjustment', -1) if config.get('max_entry_position_adjustment') != float('inf') else -1}
        return val

    def _rpc_trade_status(self, trade_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        if trade_ids:
            trades = Trade.get_trades(trade_filter=Trade.id.in_(trade_ids)).all()
        else:
            trades = Trade.get_open_trades()
        if not trades:
            raise RPCException('no active trade')
        else:
            results = []
            for trade in trades:
                current_profit_fiat = None
                total_profit_fiat = None
                oo_details = ''
                oo_details_lst = [f'({oo.order_type} {oo.side} rem={oo.safe_remaining:.8f})' for oo in trade.open_orders if oo.ft_order_side not in ['stoploss']]
                oo_details = ', '.join(oo_details_lst)
                total_profit_abs = 0.0
                total_profit_ratio = None
                if trade.is_open:
                    try:
                        current_rate = self._freqtrade.exchange.get_rate(trade.pair, side='exit', is_short=trade.is_short, refresh=False)
                    except (ExchangeError, PricingError):
                        current_rate = nan
                    if len(trade.select_filled_orders(trade.entry_side)) > 0:
                        current_profit = current_profit_abs = current_profit_fiat = nan
                        if not isnan(current_rate):
                            prof = trade.calculate_profit(current_rate)
                            current_profit = prof.profit_ratio
                            current_profit_abs = prof.profit_abs
                            total_profit_abs = prof.total_profit
                            total_profit_ratio = prof.total_profit_ratio
                    else:
                        current_profit = current_profit_abs = current_profit_fiat = 0.0
                else:
                    current_rate = trade.close_rate or 0.0
                    current_profit = trade.close_profit or 0.0
                    current_profit_abs = trade.close_profit_abs or 0.0
                if not isnan(current_profit_abs) and self._fiat_converter:
                    current_profit_fiat = self._fiat_converter.convert_amount(current_profit_abs, self._freqtrade.config['stake_currency'], self._freqtrade.config['fiat_display_currency'])
                    total_profit_fiat = self._fiat_converter.convert_amount(total_profit_abs, self._freqtrade.config['stake_currency'], self._freqtrade.config['fiat_display_currency'])
                stop_entry = trade.calculate_profit(trade.stop_loss)
                stoploss_entry_dist = stop_entry.profit_abs
                stoploss_entry_dist_ratio = stop_entry.profit_ratio
                stoploss_current_dist = price_to_precision(trade.stop_loss - current_rate, trade.price_precision, trade.precision_mode_price)
                stoploss_current_dist_ratio = stoploss_current_dist / current_rate
                trade_dict = trade.to_json()
                trade_dict.update(dict(close_profit=trade.close_profit if not trade.is_open else None, current_rate=current_rate, profit_ratio=current_profit, profit_pct=round(current_profit * 100, 2), profit_abs=current_profit_abs, profit_fiat=current_profit_fiat, total_profit_abs=total_profit_abs, total_profit_fiat=total_profit_fiat, total_profit_ratio=total_profit_ratio, stoploss_current_dist=stoploss_current_dist, stoploss_current_dist_ratio=round(stoploss_current_dist_ratio, 8), stoploss_current_dist_pct=round(stoploss_current_dist_ratio * 100, 2), stoploss_entry_dist=stoploss_entry_dist, stoploss_entry_dist_ratio=round(stoploss_entry_dist_ratio, 8), open_orders=oo_details, nr_of_successful_entries=trade.nr_of_successful_entries))
                results.append(trade_dict)
            return results

    def _rpc_status_table(self, stake_currency: str, fiat_display_currency: str) -> Tuple[List[List[str]], List[str], float, float]:
        """
        :return: list of trades, list of columns, sum of fiat profit
        """
        nonspot = self._config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT
        if not Trade.get_open_trades():
            raise RPCException('no active trade')
        trades_list = []
        fiat_profit_sum = nan
        fiat_total_profit_sum = nan
        for trade in self._rpc_trade_status():
            profit = f'{trade['profit_ratio']:.2%}'
            fiat_profit = trade.get('profit_fiat', None)
            if fiat_profit is None or isnan(fiat_profit):
                fiat_profit = trade.get('profit_abs', 0.0)
            if not isnan(fiat_profit):
                profit += f' ({fiat_profit:.2f})'
                fiat_profit_sum = fiat_profit if isnan(fiat_profit_sum) else fiat_profit_sum + fiat_profit
            total_profit = trade.get('total_profit_fiat', None)
            if total_profit is None or isnan(total_profit):
                total_profit = trade.get('total_profit_abs', 0.0)
            if not isnan(total_profit):
                fiat_total_profit_sum = total_profit if isnan(fiat_total_profit_sum) else fiat_total_profit_sum + total_profit
            active_order_side = ''
            orders = trade.get('orders', [])
            if orders:
                active_order_side = '.'.join(('*' if o.get('is_open') and o.get('ft_is_entry') else '**' for o in orders if o.get('is_open') and o.get('ft_order_side') != 'stoploss'))
            direction_str = ''
            if nonspot:
                leverage = trade.get('leverage', 1.0)
                direction_str = f'{('S' if trade.get('is_short') else 'L')} {leverage:.3g}x'
            detail_trade = [f'{trade['trade_id']} {direction_str}', f'{trade['pair']}{active_order_side}', shorten_date(dt_humanize_delta(dt_from_ts(trade['open_timestamp']))), profit]
            if self._config.get('position_adjustment_enable', False):
                max_entry_str = ''
                if self._config.get('max_entry_position_adjustment', -1) > 0:
                    max_entry_str = f'/{self._config['max_entry_position_adjustment'] + 1}'
                filled_entries = trade.get('nr_of_successful_entries', 0)
                detail_trade.append(f'{filled_entries}{max_entry_str}')
            trades_list.append(detail_trade)
        columns = ['ID L/S' if nonspot else 'ID', 'Pair', 'Since', f'Profit ({(fiat_display_currency if self._fiat_converter else stake_currency)})']
        if self._config.get('position_adjustment_enable', False):
            columns.append('# Entries')
        return (trades_list, columns, fiat_profit_sum, fiat_total_profit_sum)

    def _rpc_timeunit_profit(self, timescale: int, stake_currency: str, fiat_display_currency: str, timeunit: Literal['days', 'weeks', 'months'] = 'days') -> Dict[str, Any]:
        """
        :param timeunit: Valid entries are 'days', 'weeks', 'months'
        """
        start_date = datetime.now(timezone.utc).date()
        if timeunit == 'weeks':
            start_date = start_date - timedelta(days=start_date.weekday())
        if timeunit == 'months':
            start_date = start_date.replace(day=1)

        def time_offset(step: int) -> Union[timedelta, relativedelta]:
            if timeunit == 'months':
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})
        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException('timescale must be an integer greater than 0')
        profit_units = {}
        daily_stake = self._freqtrade.wallets.get_total_stake_amount()
        for day in range(0, timescale):
            profitday = start_date - time_offset(day)
            trades = Trade.session.execute(select(Trade.close_profit_abs).filter(Trade.is_open.is_(False), Trade.close_date >= profitday, Trade.close_date < profitday + time_offset(1)).order_by(Trade.close_date)).all()
            curdayprofit = sum((trade.close_profit_abs for trade in trades if trade.close_profit_abs is not None))
            daily_stake = daily_stake - curdayprofit
            profit_units[profitday] = {'amount': curdayprofit, 'daily_stake': daily_stake, 'rel_profit': round(curdayprofit / daily_stake, 8) if daily_stake > 0 else 0, 'trades': len(trades)}
        data = [{'date': key, 'abs_profit': value['amount'], 'starting_balance': value['daily_stake'], 'rel_profit': value['rel_profit'], 'fiat_value': self._fiat_converter.convert_amount(value['amount'], stake_currency, fiat_display_currency) if self._fiat_converter else 0, 'trade_count': value['trades']} for key, value in profit_units.items()]
        return {'stake_currency': stake_currency, 'fiat_display_currency': fiat_display_currency, 'data': data}

    def _rpc_trade_history(self, limit: Optional[int], offset: int = 0, order_by_id: bool = False) -> Dict[str, Any]:
        """Returns the X last trades"""
        order_by = Trade.id if order_by_id else Trade.close_date.desc()
        if limit:
            trades = Trade.session.scalars(Trade.get_trades_query([Trade.is_open.is_(False)]).order_by(order_by).limit(limit).offset(offset))
        else:
            trades = Trade.session.scalars(Trade.get_trades_query([Trade.is_open.is_(False)]).order_by(Trade.close_date.desc()))
        output = [trade.to_json() for trade in trades]
        total_trades = Trade.session.scalar(select(func.count(Trade.id)).filter(Trade.is_open.is_(False)))
        return {'trades': output, 'trades_count': len(output), 'offset': offset, 'total_trades': total_trades}

    def _rpc_stats(self) -> Dict[str, Any]:
        """
        Generate generic stats for trades in database
        """

        def trade_win_loss(trade: Trade) -> Literal['wins', 'losses', 'draws']:
            if trade.close_profit > 0:
                return 'wins'
            elif trade.close_profit < 0:
                return 'losses'
            else:
                return 'draws'
        trades = Trade.get_trades([Trade.is_open.is_(False)], include_orders=False)
        dur = {'wins': [], 'draws