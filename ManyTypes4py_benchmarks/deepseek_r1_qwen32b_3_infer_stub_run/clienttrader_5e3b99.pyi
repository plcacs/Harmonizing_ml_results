import abc
import functools
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union
import pywinauto
from pywinauto import Application
from easytrader.grid_strategies import IGridStrategy
from easytrader.refresh_strategies import IRefreshStrategy

class IClientTrader(abc.ABC):
    @property
    def app(self) -> Application:
        ...
    
    @property
    def main(self) -> Any:
        ...
    
    @property
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
    grid_strategy: Union[IGridStrategy, type[IGridStrategy]]
    _grid_strategy_instance: Optional[IGridStrategy]
    refresh_strategy: IRefreshStrategy

    def __init__(self) -> None:
        ...

    @property
    def app(self) -> Application:
        ...

    @property
    def main(self) -> Any:
        ...

    @property
    def config(self) -> Any:
        ...

    def connect(self, exe_path: Optional[str] = None, **kwargs: Any) -> None:
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

    def market_buy(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None, **kwargs: Any) -> Dict[str, str]:
        ...

    def market_sell(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None, **kwargs: Any) -> Dict[str, str]:
        ...

    def market_trade(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None, **kwargs: Any) -> Dict[str, str]:
        ...

    def auto_ipo(self) -> Dict[str, str]:
        ...

    def is_exist_pop_dialog(self) -> bool:
        ...

    def close_pop_dialog(self) -> None:
        ...

    def exit(self) -> None:
        ...

    def refresh(self) -> None:
        ...

    def _handle_pop_dialogs(self, handler_class: Any = None) -> Dict[str, str]:
        ...

class BaseLoginClientTrader(ClientTrader):
    @abc.abstractmethod
    def login(self, user: str, password: str, exe_path: str, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...

    def prepare(self, config_path: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, exe_path: Optional[str] = None, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...