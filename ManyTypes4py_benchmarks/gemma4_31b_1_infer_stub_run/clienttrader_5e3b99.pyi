import abc
from typing import Any, Optional, Union, Dict, List, Type, overload
from pywinauto import Application, WindowSpecification
from easytrader.grid_strategies import IGridStrategy
from easytrader.refresh_strategies import IRefreshStrategy

class IClientTrader(abc.ABC):
    @property
    @abc.abstractmethod
    def app(self) -> Optional[Application]:
        """Return current app instance"""
        ...

    @property
    @abc.abstractmethod
    def main(self) -> Optional[WindowSpecification]:
        """Return current main window instance"""
        ...

    @property
    @abc.abstractmethod
    def config(self) -> Any:
        """Return current config instance"""
        ...

    @abc.abstractmethod
    def wait(self, seconds: float) -> None:
        """Wait for operation return"""
        ...

    @abc.abstractmethod
    def refresh(self) -> None:
        """Refresh data"""
        ...

    @abc.abstractmethod
    def is_exist_pop_dialog(self) -> bool:
        ...

class ClientTrader(IClientTrader):
    _editor_need_type_keys: bool
    grid_strategy: Any
    _grid_strategy_instance: Optional[IGridStrategy]
    refresh_strategy: IRefreshStrategy

    def enable_type_keys_for_editor(self) -> None:
        ...

    @property
    def grid_strategy_instance(self) -> IGridStrategy:
        ...

    def __init__(self) -> None:
        ...

    @property
    def app(self) -> Optional[Application]:
        ...

    @property
    def main(self) -> Optional[WindowSpecification]:
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

    def _init_toolbar(self) -> None:
        ...

    def _get_balance_from_statics(self) -> Dict[str, float]:
        ...

    @property
    def position(self) -> List[Any]:
        ...

    @property
    def today_entrusts(self) -> List[Any]:
        ...

    @property
    def today_trades(self) -> List[Any]:
        ...

    @property
    def cancel_entrusts(self) -> List[Any]:
        ...

    def cancel_entrust(self, entrust_no: Union[str, int]) -> Dict[str, Any]:
        ...

    def cancel_all_entrusts(self) -> None:
        ...

    def repo(self, security: str, price: float, amount: Union[int, float], **kwargs: Any) -> Dict[str, Any]:
        ...

    def reverse_repo(self, security: str, price: float, amount: Union[int, float], **kwargs: Any) -> Dict[str, Any]:
        ...

    def buy(self, security: str, price: float, amount: Union[int, float], **kwargs: Any) -> Dict[str, Any]:
        ...

    def sell(self, security: str, price: float, amount: Union[int, float], **kwargs: Any) -> Dict[str, Any]:
        ...

    def market_buy(self, security: str, amount: Union[int, float], ttype: Optional[str] = None, limit_price: Optional[Union[str, float]] = None, **kwargs: Any) -> Dict[str, Any]:
        ...

    def market_sell(self, security: str, amount: Union[int, float], ttype: Optional[str] = None, limit_price: Optional[Union[str, float]] = None, **kwargs: Any) -> Dict[str, Any]:
        ...

    def market_trade(self, security: str, amount: Union[int, float], ttype: Optional[str] = None, limit_price: Optional[Union[str, float]] = None, **kwargs: Any) -> Dict[str, Any]:
        ...

    def _set_market_trade_type(self, ttype: str) -> None:
        ...

    def _set_stock_exchange_type(self, ttype: str) -> None:
        ...

    def auto_ipo(self) -> Dict[str, Any]:
        ...

    def _click_grid_by_row(self, row: int) -> None:
        ...

    def is_exist_pop_dialog(self) -> bool:
        ...

    def close_pop_dialog(self) -> None:
        ...

    def _run_exe_path(self, exe_path: str) -> str:
        ...

    def wait(self, seconds: float) -> None:
        ...

    def exit(self) -> None:
        ...

    def _close_prompt_windows(self) -> None:
        ...

    def close_pormpt_window_no_wait(self) -> None:
        ...

    def trade(self, security: str, price: float, amount: Union[int, float]) -> Dict[str, Any]:
        ...

    def _click(self, control_id: int) -> None:
        ...

    def _submit_trade(self) -> None:
        ...

    def __get_top_window_pop_dialog(self) -> Any:
        ...

    def _get_pop_dialog_title(self) -> str:
        ...

    def _set_trade_params(self, security: str, price: float, amount: Union[int, float]) -> None:
        ...

    def _set_market_trade_params(self, security: str, amount: Union[int, float], limit_price: Optional[Union[str, float]] = None) -> None:
        ...

    def _get_grid_data(self, control_id: int) -> List[Any]:
        ...

    def _type_keys(self, control_id: int, text: str) -> None:
        ...

    def _type_edit_control_keys(self, control_id: int, text: str) -> None:
        ...

    def type_edit_control_keys(self, editor: Any, text: str) -> None:
        ...

    def _collapse_left_menus(self) -> None:
        ...

    def _switch_left_menus(self, path: List[str], sleep: float = 0.2) -> None:
        ...

    def _switch_left_menus_by_shortcut(self, shortcut: str, sleep: float = 0.5) -> None:
        ...

    def _get_left_menus_handle(self) -> WindowSpecification:
        ...

    def _cancel_entrust_by_double_click(self, row: int) -> None:
        ...

    def refresh(self) -> None:
        ...

    def _handle_pop_dialogs(self, handler_class: Type[Any] = Any) -> Dict[str, Any]:
        ...

class BaseLoginClientTrader(ClientTrader):
    @abc.abstractmethod
    def login(self, user: str, password: str, exe_path: str, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...

    def prepare(self, config_path: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, exe_path: Optional[str] = None, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        ...