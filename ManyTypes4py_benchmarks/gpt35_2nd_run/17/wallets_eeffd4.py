from typing import Literal, NamedTuple, Dict, Union
from freqtrade.constants import Config
from freqtrade.enums import RunMode, TradingMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.util.datetime_helpers import dt_now

class Wallet(NamedTuple):
    free: float
    used: float
    total: float

class PositionWallet(NamedTuple):
    position: float
    leverage: float
    collateral: float
    side: str

class Wallets:
    def __init__(self, config: Config, exchange: Exchange, is_backtest: bool = False) -> None:
    def get_free(self, currency: str) -> float:
    def get_used(self, currency: str) -> float:
    def get_total(self, currency: str) -> float:
    def get_collateral(self) -> float:
    def get_owned(self, pair: str, base_currency: str) -> float:
    def _update_dry(self) -> None:
    def _update_live(self) -> None:
    def update(self, require_update: bool = True) -> None:
    def get_all_balances(self) -> Dict[str, Wallet]:
    def get_all_positions(self) -> Dict[str, PositionWallet]:
    def _check_exit_amount(self, trade) -> bool:
    def check_exit_amount(self, trade) -> bool:
    def get_starting_balance(self) -> float:
    def get_total_stake_amount(self) -> float:
    def get_available_stake_amount(self) -> float:
    def _calculate_unlimited_stake_amount(self, available_amount: float, val_tied_up: float, max_open_trades: int) -> float:
    def _check_available_stake_amount(self, stake_amount: float, available_amount: float) -> float:
    def get_trade_stake_amount(self, pair: str, max_open_trades: int, edge: Union[Edge, None] = None, update: bool = True) -> float:
    def validate_stake_amount(self, pair: str, stake_amount: float, min_stake_amount: float, max_stake_amount: float, trade_amount: float) -> float:
    def _local_log(self, msg: str, level: Literal['info', 'warning', 'debug'] = 'info') -> None:
