from typing import Any, Dict, List, Optional, Type, Union
from abc import ABCMeta
from pywinauto import Application, Window
from easytrader.config import client
from easytrader.grid_strategies import IGridStrategy
from easytrader.log import logger
from easytrader.refresh_strategies import IRefreshStrategy
from easytrader.utils.perf import perf_clock

class IClientTrader(metaclass=ABCMeta):
    @property
    def app(self) -> Any:
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
    def __init__(self) -> None:
        ...
    
    @property
    def app(self) -> Application:
        ...
    
    @property
    def main(self) -> Window:
        ...
    
    @property
    def config(self) -> client.ClientConfig:
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
    
    def connect(self, exe_path: str, **kwargs: Any) -> None:
        ...
    
    def cancel_entrust(self, entrust_no: str) -> Dict[str, str]:
        ...
    
    def cancel_all_entrusts(self) -> None:
        ...
    
    def repo(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def reverse_repo(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def buy(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def sell(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_buy(self, security: str, amount: int, ttype: Optional[str] = None, limit_price: Optional[str] = None, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_sell(self, security: str, amount: int, ttype: Optional[str] = None, limit_price: Optional[str] = None, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def market_trade(self, security: str, amount: int, ttype: Optional[str] = None, limit_price: Optional[str] = None, **kwargs: Any) -> Dict[str, str]:
        ...
    
    def auto_ipo(self) -> Dict[str, str]:
        ...
    
    def is_exist_pop_dialog(self) -> bool:
        ...
    
    def close_pop_dialog(self) -> None:
        ...
    
    def wait(self, seconds: float) -> None:
        ...
    
    def exit(self) -> None:
        ...
    
    def _get_grid_data(self, control_id: int) -> List[Dict[str, Any]]:
        ...
    
    def _type_edit_control_keys(self, control_id: int, text: str) -> None:
        ...
    
    def _submit_trade(self) -> None:
        ...
    
    def _handle_pop_dialogs(self, handler_class: Optional[Type[pop_dialog_handler.PopDialogHandler]] = None) -> Dict[str, str]:
        ...

class BaseLoginClientTrader(ClientTrader):
    @abc.abstractmethod
    def login(self, user: str, password: str, exe_path: str, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...
    
    def prepare(self, config_path: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, exe_path: Optional[str] = None, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...