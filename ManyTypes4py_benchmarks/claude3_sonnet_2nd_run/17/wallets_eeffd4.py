"""Wallet"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Literal, NamedTuple, Optional, Union, Any, cast
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT, Config, IntOrInf
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.misc import safe_value_fallback
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.util.datetime_helpers import dt_now

logger = logging.getLogger(__name__)

class Wallet(NamedTuple):
    currency: str
    free: float = 0
    used: float = 0
    total: float = 0

class PositionWallet(NamedTuple):
    symbol: str
    position: float = 0
    leverage: Optional[float] = 0
    collateral: float = 0
    side: str = 'long'

class Wallets:

    def __init__(self, config: Config, exchange: Exchange, is_backtest: bool = False) -> None:
        self._config: Config = config
        self._is_backtest: bool = is_backtest
        self._exchange: Exchange = exchange
        self._wallets: Dict[str, Wallet] = {}
        self._positions: Dict[str, PositionWallet] = {}
        self._start_cap: Dict[str, float] = {}
        self._stake_currency: str = self._exchange.get_proxy_coin()
        if isinstance((_start_cap := config['dry_run_wallet']), (float, int)):
            self._start_cap[self._stake_currency] = float(_start_cap)
        else:
            self._start_cap = _start_cap
        self._last_wallet_refresh: Optional[datetime] = None
        self.update()

    def get_free(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.free:
            return balance.free
        else:
            return 0

    def get_used(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.used:
            return balance.used
        else:
            return 0

    def get_total(self, currency: str) -> float:
        balance = self._wallets.get(currency)
        if balance and balance.total:
            return balance.total
        else:
            return 0

    def get_collateral(self) -> float:
        """
        Get total collateral for liquidation price calculation.
        """
        if self._config.get('margin_mode') == 'cross':
            return self.get_free(self._stake_currency) + sum((pos.collateral for pos in self._positions.values()))
        return self.get_total(self._stake_currency)

    def get_owned(self, pair: str, base_currency: str) -> float:
        """
        Get currently owned value.
        Designed to work across both spot and futures.
        """
        if self._config.get('trading_mode', 'spot') != TradingMode.FUTURES:
            return self.get_total(base_currency) or 0
        if (pos := self._positions.get(pair)):
            return pos.position
        return 0

    def _update_dry(self) -> None:
        """
        Update from database in dry-run mode
        - Apply profits of closed trades on top of stake amount
        - Subtract currently tied up stake_amount in open trades
        - update balances for currencies currently in trades
        """
        _wallets: Dict[str, Wallet] = {}
        _positions: Dict[str, PositionWallet] = {}
        open_trades = Trade.get_trades_proxy(is_open=True)
        if not self._is_backtest:
            tot_profit = Trade.get_total_closed_profit()
        else:
            tot_profit = LocalTrade.bt_total_profit
        tot_profit += sum((trade.realized_profit for trade in open_trades))
        tot_in_trades = sum((trade.stake_amount for trade in open_trades))
        used_stake = 0.0
        if self._config.get('trading_mode', 'spot') != TradingMode.FUTURES:
            for trade in open_trades:
                curr = self._exchange.get_pair_base_currency(trade.pair)
                used_stake += sum((o.stake_amount for o in trade.open_orders if o.ft_order_side == trade.entry_side))
                pending = sum((o.amount for o in trade.open_orders if o.amount and o.ft_order_side == trade.exit_side))
                curr_wallet_bal = self._start_cap.get(curr, 0)
                _wallets[curr] = Wallet(curr, curr_wallet_bal + trade.amount - pending, pending, trade.amount + curr_wallet_bal)
        else:
            for position in open_trades:
                _positions[position.pair] = PositionWallet(position.pair, position=position.amount, leverage=position.leverage, collateral=position.stake_amount, side=position.trade_direction)
            used_stake = tot_in_trades
        cross_margin = 0.0
        if self._config.get('margin_mode') == 'cross':
            for curr, bal in self._start_cap.items():
                if curr == self._stake_currency:
                    continue
                rate = self._exchange.get_conversion_rate(curr, self._stake_currency)
                if rate:
                    cross_margin += bal * rate
        current_stake = self._start_cap.get(self._stake_currency, 0) + tot_profit - tot_in_trades
        total_stake = current_stake + used_stake
        _wallets[self._stake_currency] = Wallet(currency=self._stake_currency, free=current_stake + cross_margin, used=used_stake, total=total_stake)
        for currency, bal in self._start_cap.items():
            if currency not in _wallets:
                _wallets[currency] = Wallet(currency, bal, 0, bal)
        self._wallets = _wallets
        self._positions = _positions

    def _update_live(self) -> None:
        balances = self._exchange.get_balances()
        _wallets: Dict[str, Wallet] = {}
        for currency in balances:
            if isinstance(balances[currency], dict):
                _wallets[currency] = Wallet(currency, balances[currency].get('free', 0), balances[currency].get('used', 0), balances[currency].get('total', 0))
        positions = self._exchange.fetch_positions()
        _parsed_positions: Dict[str, PositionWallet] = {}
        for position in positions:
            symbol = position['symbol']
            if position['side'] is None or position['collateral'] == 0.0:
                continue
            size = self._exchange._contracts_to_amount(symbol, position['contracts'])
            collateral = safe_value_fallback(position, 'collateral', 'initialMargin', 0.0)
            leverage = position.get('leverage')
            _parsed_positions[symbol] = PositionWallet(symbol, position=size, leverage=leverage, collateral=collateral, side=position['side'])
        self._positions = _parsed_positions
        self._wallets = _wallets

    def update(self, require_update: bool = True) -> None:
        """
        Updates wallets from the configured version.
        By default, updates from the exchange.
        Update-skipping should only be used for user-invoked /balance calls, since
        for trading operations, the latest balance is needed.
        :param require_update: Allow skipping an update if balances were recently refreshed
        """
        now = dt_now()
        if require_update or self._last_wallet_refresh is None or self._last_wallet_refresh + timedelta(seconds=3600) < now:
            if not self._config['dry_run'] or self._config.get('runmode') == RunMode.LIVE:
                self._update_live()
            else:
                self._update_dry()
            self._local_log('Wallets synced.')
            self._last_wallet_refresh = dt_now()

    def get_all_balances(self) -> Dict[str, Wallet]:
        return self._wallets

    def get_all_positions(self) -> Dict[str, PositionWallet]:
        return self._positions

    def _check_exit_amount(self, trade: Trade) -> bool:
        if trade.trading_mode != TradingMode.FUTURES:
            wallet_amount = self.get_total(trade.safe_base_currency) * (2 - 0.981)
        else:
            position = self._positions.get(trade.pair)
            if position is None:
                return False
            wallet_amount = position.position
        if wallet_amount >= trade.amount:
            return True
        return False

    def check_exit_amount(self, trade: Trade) -> bool:
        """
        Checks if the exit amount is available in the wallet.
        :param trade: Trade to check
        :return: True if the exit amount is available, False otherwise
        """
        if not self._check_exit_amount(trade):
            self.update()
            return self._check_exit_amount(trade)
        return True

    def get_starting_balance(self) -> float:
        """
        Retrieves starting balance - based on either available capital,
        or by using current balance subtracting
        """
        if 'available_capital' in self._config:
            return cast(float, self._config['available_capital'])
        else:
            tot_profit = Trade.get_total_closed_profit()
            open_stakes = Trade.total_open_trades_stakes()
            available_balance = self.get_free(self._stake_currency)
            return (available_balance - tot_profit + open_stakes) * self._config['tradable_balance_ratio']

    def get_total_stake_amount(self) -> float:
        """
        Return the total currently available balance in stake currency, including tied up stake and
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio
        """
        val_tied_up = Trade.total_open_trades_stakes()
        if 'available_capital' in self._config:
            starting_balance = cast(float, self._config['available_capital'])
            tot_profit = Trade.get_total_closed_profit()
            available_amount = starting_balance + tot_profit
        else:
            available_amount = (val_tied_up + self.get_free(self._stake_currency)) * self._config['tradable_balance_ratio']
        return available_amount

    def get_available_stake_amount(self) -> float:
        """
        Return the total currently available balance in stake currency,
        respecting tradable_balance_ratio.
        Calculated as
        (<open_trade stakes> + free amount) * tradable_balance_ratio - <open_trade stakes>
        """
        free = self.get_free(self._stake_currency)
        return min(self.get_total_stake_amount() - Trade.total_open_trades_stakes(), free)

    def _calculate_unlimited_stake_amount(self, available_amount: float, val_tied_up: float, max_open_trades: IntOrInf) -> float:
        """
        Calculate stake amount for "unlimited" stake amount
        :return: 0 if max number of trades reached, else stake_amount to use.
        """
        if max_open_trades == 0:
            return 0
        possible_stake = (available_amount + val_tied_up) / max_open_trades
        return min(possible_stake, available_amount)

    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:
        """
        Check if stake amount can be fulfilled with the available balance
        for the stake currency
        :return: float: Stake amount
        :raise: DependencyException if balance is lower than stake-amount
        """
        if self._config['amend_last_stake_amount']:
            if available_amount > stake_amount * self._config['last_stake_amount_min_ratio']:
                stake_amount = min(stake_amount, available_amount)
            else:
                stake_amount = 0
        if available_amount < stake_amount:
            raise DependencyException(f'Available balance ({available_amount} {self._config["stake_currency"]}) is lower than stake amount ({stake_amount} {self._config["stake_currency"]})')
        return max(stake_amount, 0)

    def get_trade_stake_amount(self, pair: str, max_open_trades: IntOrInf, edge: Optional[Any] = None, update: bool = True) -> float:
        """
        Calculate stake amount for the trade
        :return: float: Stake amount
        :raise: DependencyException if the available stake amount is too low
        """
        if update:
            self.update()
        val_tied_up = Trade.total_open_trades_stakes()
        available_amount = self.get_available_stake_amount()
        if edge:
            stake_amount = edge.stake_amount(pair, self.get_free(self._stake_currency), self.get_total(self._stake_currency), val_tied_up)
        else:
            stake_amount = self._config['stake_amount']
            if stake_amount == UNLIMITED_STAKE_AMOUNT:
                stake_amount = self._calculate_unlimited_stake_amount(available_amount, val_tied_up, max_open_trades)
        return self._check_available_stake_amount(stake_amount, available_amount)

    def validate_stake_amount(self, pair: str, stake_amount: float, min_stake_amount: Optional[float], max_stake_amount: float, trade_amount: float) -> float:
        if not stake_amount:
            self._local_log(f'Stake amount is {stake_amount}, ignoring possible trade for {pair}.', level='debug')
            return 0
        max_allowed_stake = min(max_stake_amount, self.get_available_stake_amount())
        if trade_amount:
            max_allowed_stake = min(max_allowed_stake, max_stake_amount - trade_amount)
        if min_stake_amount is not None and min_stake_amount > max_allowed_stake:
            self._local_log(f'Minimum stake amount > available balance. {min_stake_amount} > {max_allowed_stake}', level='warning')
            return 0
        if min_stake_amount is not None and stake_amount < min_stake_amount:
            self._local_log(f'Stake amount for pair {pair} is too small ({stake_amount} < {min_stake_amount}), adjusting to {min_stake_amount}.')
            if stake_amount * 1.3 < min_stake_amount:
                self._local_log(f'Adjusted stake amount for pair {pair} is more than 30% bigger than the desired stake amount of ({stake_amount:.8f} * 1.3 = {stake_amount * 1.3:.8f}) < {min_stake_amount}), ignoring trade.')
                return 0
            stake_amount = min_stake_amount
        if stake_amount > max_allowed_stake:
            self._local_log(f'Stake amount for pair {pair} is too big ({stake_amount} > {max_allowed_stake}), adjusting to {max_allowed_stake}.')
            stake_amount = max_allowed_stake
        return stake_amount

    def _local_log(self, msg: str, level: Literal['info', 'warning', 'debug'] = 'info') -> None:
        """
        Log a message to the local log.
        """
        if not self._is_backtest:
            if level == 'warning':
                logger.warning(msg)
            elif level == 'debug':
                logger.debug(msg)
            else:
                logger.info(msg)
