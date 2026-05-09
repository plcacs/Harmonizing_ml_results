import abc
import functools
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Type, Union
import pywinauto
from easytrader.grid_strategies import IGridStrategy
from easytrader.refresh_strategies import IRefreshStrategy

class IClientTrader(abc.ABC):
    @property
    @abc.abstractmethod
    def app(self) -> Any:
        ...
    
    @property
    @abc.abstractmethod
    def main(self) -> Any:
        ...
    
    @property
    @abc.abstractmethod
    def config(self) -> Any:
        ...
    
    @abc.abstractmethod
    def wait(self, seconds: float) -> None:
        ...
    
    @abc.abstractmethod
    def refresh(self) -> None:
        ...
    
    @abc.abstractmethod
    def is_exist_pop_dialog(self) -> bool:
        ...

class ClientTrader(IClientTrader):
    _editor_need_type_keys: bool
    grid_strategy: Type[IGridStrategy]
    _grid_strategy_instance: Optional[IGridStrategy]
    refresh_strategy: Type[IRefreshStrategy]
    
    def __init__(self) -> None:
        ...
    
    @property
    def app(self) -> Any:
        ...
    
    @property
    def main(self) -> Any:
        ...
    
    @property
    def config(self) -> Any:
        ...
    
    def connect(self, exe_path: str = ..., **kwargs: Any) -> None:
        ...
    
    @property
    def broker_type(self) -> str:
        ...
    
    @property
    def balance(self) -> Dict[str, float]:
        ...
    
    @property
    def position(self) -> List[Dict[str, Any]]:
        ...
    
    @property
    def today_entrusts(self) -> List[Dict[str, Any]]:
        ...
    
    @property
    def today_trades(self) -> List[Dict[str, Any]]:
        ...
    
    @property
    def cancel_entrusts(self) -> List[Dict[str, Any]]:
        ...
    
    def cancel_entrust(self, entrust_no: str) -> Dict[str, str]:
        ...
    
    def cancel_all_entrusts(self) -> None:
        ...
    
    def repo(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def reverse_repo(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def buy(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def sell(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_buy(self, security: str, amount: float, ttype: Optional[str] = ..., limit_price: Optional[float] = ..., **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_sell(self, security: str, amount: float, ttype: Optional[str] = ..., limit_price: Optional[float] = ..., **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_trade(self, security: str, amount: float, ttype: Optional[str] = ..., limit_price: Optional[float] = ..., **kwargs: Any) -> Dict[str, str]:
        ...
    
    def _set_market_trade_type(self, ttype: str) -> None:
        ...
    
    def _set_stock_exchange_type(self, ttype: str) -> None:
        ...
    
    def auto_ipo(self) -> Dict[str, str]:
        ...
    
    def _click_grid_by_row(self, row: int) -> None:
        ...
    
    def is_exist_pop_dialog(self) -> bool:
        ...
    
    def close_pop_dialog(self) -> None:
        ...
    
    def wait(self, seconds: float) -> None:
        ...
    
    def exit(self) -> None:
        ...
    
    def _close_prompt_windows(self) -> None:
        ...
    
    def close_pormpt_window_no_wait(self) -> None:
        ...
    
    def trade(self, security: str, price: float, amount: float) -> Dict[str, str]:
        ...
    
    def _click(self, control_id: int) -> None:
        ...
    
    def _submit_trade(self) -> None:
        ...
    
    def __get_top_window_pop_dialog(self) -> Any:
        ...
    
    def _get_pop_dialog_title(self) -> str:
        ...
    
    def _set_trade_params(self, security: str, price: float, amount: float) -> None:
        ...
    
    def _set_market_trade_params(self, security: str, amount: float, limit_price: Optional[float] = ..., **kwargs: Any) -> None:
        ...
    
    def _get_grid_data(self, control_id: int) -> List[Dict[str, Any]]:
        ...
    
    def _type_keys(self, control_id: int, text: str) -> None:
        ...
    
    def _type_edit_control_keys(self, control_id: int, text: str) -> None:
        ...
    
    def type_edit_control_keys(self, editor: Any, text: str) -> None:
        ...
    
    def _collapse_left_menus(self) -> None:
        ...
    
    def _switch_left_menus(self, path: List[str], sleep: float = ...) -> None:
        ...
    
    def _switch_left_menus_by_shortcut(self, shortcut: str, sleep: float = ...) -> None:
        ...
    
    @functools.lru_cache()
    def _get_left_menus_handle(self) -> Any:
        ...
    
    def _cancel_entrust_by_double_click(self, row: int) -> None:
        ...
    
    def refresh(self) -> None:
        ...
    
    def _handle_pop_dialogs(self, handler_class: Any = ...) -> Dict[str, str]:
        ...

class BaseLoginClientTrader(ClientTrader):
    @abc.abstractmethod
    def login(self, user: str, password: str, exe_path: str, comm_password: Optional[str] = ..., **kwargs: Any) -> None:
        ...
    
    def prepare(self, config_path: Optional[str] = ..., user: Optional[str] = ..., password: Optional[str] = ..., exe_path: Optional[str] = ..., comm_password: Optional[str] = ..., **kwargs: Any) -> None:
        ...