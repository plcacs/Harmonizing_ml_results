#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import functools
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Type, Union

import hashlib
import binascii

import easyutils
from pywinauto import findwindows, timings

from easytrader import grid_strategies, pop_dialog_handler, refresh_strategies
from easytrader.config import client
from easytrader.grid_strategies import IGridStrategy
from easytrader.log import logger
from easytrader.refresh_strategies import IRefreshStrategy
from easytrader.utils.misc import file2dict
from easytrader.utils.perf import perf_clock

if not sys.platform.startswith("darwin"):
    import pywinauto
    import pywinauto.clipboard


class IClientTrader(abc.ABC):
    @property
    @abc.abstractmethod
    def app(self) -> Any:
        """Return current app instance"""
        pass

    @property
    @abc.abstractmethod
    def main(self) -> Any:
        """Return current main window instance"""
        pass

    @property
    @abc.abstractmethod
    def config(self) -> Any:
        """Return current config instance"""
        pass

    @abc.abstractmethod
    def wait(self, seconds: float) -> None:
        """Wait for operation return"""
        pass

    @abc.abstractmethod
    def refresh(self) -> None:
        """Refresh data"""
        pass

    @abc.abstractmethod
    def is_exist_pop_dialog(self) -> bool:
        pass


class ClientTrader(IClientTrader):
    _editor_need_type_keys: bool = False
    # The strategy to use for getting grid data
    grid_strategy: Union[IGridStrategy, Type[IGridStrategy]] = grid_strategies.Copy
    _grid_strategy_instance: Optional[IGridStrategy] = None
    refresh_strategy: IRefreshStrategy = refresh_strategies.Switch()

    def enable_type_keys_for_editor(self) -> None:
        """
        有些客户端无法通过 set_edit_text 方法输入内容，可以通过使用 type_keys 方法绕过
        """
        self._editor_need_type_keys = True

    @property
    def grid_strategy_instance(self) -> IGridStrategy:
        if self._grid_strategy_instance is None:
            self._grid_strategy_instance = (
                self.grid_strategy
                if isinstance(self.grid_strategy, IGridStrategy)
                else self.grid_strategy()  # type: ignore
            )
            self._grid_strategy_instance.set_trader(self)
        return self._grid_strategy_instance

    def __init__(self) -> None:
        self._config = client.create(self.broker_type)
        self._app: Optional[Any] = None
        self._main: Optional[Any] = None
        self._toolbar: Optional[Any] = None

    @property
    def app(self) -> Any:
        return self._app

    @property
    def main(self) -> Any:
        return self._main

    @property
    def config(self) -> Any:
        return self._config

    def connect(self, exe_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        直接连接登陆后的客户端
        :param exe_path: 客户端路径类似 r'C:\\htzqzyb2\\xiadan.exe', 默认 r'C:\\htzqzyb2\\xiadan.exe'
        :return:
        """
        connect_path: Optional[str] = exe_path or self._config.DEFAULT_EXE_PATH
        if connect_path is None:
            raise ValueError(
                "参数 exe_path 未设置，请设置客户端对应的 exe 地址,类似 C:\\客户端安装目录\\xiadan.exe"
            )

        self._app = pywinauto.Application().connect(path=connect_path, timeout=10)
        self._close_prompt_windows()
        self._main = self._app.top_window()
        self._init_toolbar()

    @property
    def broker_type(self) -> str:
        return "ths"

    @property
    def balance(self) -> Dict[str, float]:
        self._switch_left_menus(["查询[F4]", "资金股票"])
        return self._get_balance_from_statics()

    def _init_toolbar(self) -> None:
        self._toolbar = self._main.child_window(class_name="ToolbarWindow32")

    def _get_balance_from_statics(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, control_id in self._config.BALANCE_CONTROL_ID_GROUP.items():
            result[key] = float(
                self._main.child_window(
                    control_id=control_id, class_name="Static"
                ).window_text()
            )
        return result

    @property
    def position(self) -> Any:
        self._switch_left_menus(["查询[F4]", "资金股票"])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def today_entrusts(self) -> Any:
        self._switch_left_menus(["查询[F4]", "当日委托"])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def today_trades(self) -> Any:
        self._switch_left_menus(["查询[F4]", "当日成交"])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def cancel_entrusts(self) -> Any:
        self.refresh()
        self._switch_left_menus(["撤单[F3]"])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @perf_clock
    def cancel_entrust(self, entrust_no: Any) -> Dict[str, Any]:
        self.refresh()
        for i, entrust in enumerate(self.cancel_entrusts):
            if entrust[self._config.CANCEL_ENTRUST_ENTRUST_FIELD] == entrust_no:
                self._cancel_entrust_by_double_click(i)
                return self._handle_pop_dialogs()
        return {"message": "委托单状态错误不能撤单, 该委托单可能已经成交或者已撤"}

    def cancel_all_entrusts(self) -> None:
        self.refresh()
        self._switch_left_menus(["撤单[F3]"])

        # 点击全部撤销控件
        self._app.top_window().child_window(
            control_id=self._config.TRADE_CANCEL_ALL_ENTRUST_CONTROL_ID, class_name="Button", title_re=r"""全撤.*"""
        ).click()
        self.wait(0.2)

        # 等待出现 确认兑换框
        if self.is_exist_pop_dialog():
            # 点击是 按钮
            w = self._app.top_window()
            if w is not None:
                btn = w["是(Y)"]
                if btn is not None:
                    btn.click()
                    self.wait(0.2)

        # 如果出现了确认窗口
        self.close_pop_dialog()

    @perf_clock
    def repo(self, security: str, price: Union[float, str], amount: Union[int, float], **kwargs: Any) -> Any:
        self._switch_left_menus(["债券回购", "融资回购（正回购）"])
        return self.trade(security, price, amount)

    @perf_clock
    def reverse_repo(self, security: str, price: Union[float, str], amount: Union[int, float], **kwargs: Any) -> Any:
        self._switch_left_menus(["债券回购", "融劵回购（逆回购）"])
        return self.trade(security, price, amount)

    @perf_clock
    def buy(self, security: str, price: Union[float, str], amount: Union[int, float], **kwargs: Any) -> Any:
        self._switch_left_menus(["买入[F1]"])
        return self.trade(security, price, amount)

    @perf_clock
    def sell(self, security: str, price: Union[float, str], amount: Union[int, float], **kwargs: Any) -> Any:
        self._switch_left_menus(["卖出[F2]"])
        return self.trade(security, price, amount)

    @perf_clock
    def market_buy(self, security: str, amount: Union[int, float], ttype: Optional[str] = None,
                     limit_price: Optional[Union[float, str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        市价买入
        :param security: 六位证券代码
        :param amount: 交易数量
        :param ttype: 市价委托类型，默认客户端默认选择，
                     深市可选 ['对手方最优价格', '本方最优价格', '即时成交剩余撤销', '最优五档即时成交剩余 '全额成交或撤销']
                     沪市可选 ['最优五档成交剩余撤销', '最优五档成交剩余转限价']
        :param limit_price: 科创板 限价
        :return: {'entrust_no': '委托单号'}
        """
        self._switch_left_menus(["市价委托", "买入"])
        return self.market_trade(security, amount, ttype, limit_price=limit_price)

    @perf_clock
    def market_sell(self, security: str, amount: Union[int, float], ttype: Optional[str] = None,
                      limit_price: Optional[Union[float, str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        市价卖出
        :param security: 六位证券代码
        :param amount: 交易数量
        :param ttype: 市价委托类型，默认客户端默认选择，
                     深市可选 ['对手方最优价格', '本方最优价格', '即时成交剩余撤销', '最优五档即时成交剩余 '全额成交或撤销']
                     沪市可选 ['最优五档成交剩余撤销', '最优五档成交剩余转限价']
        :param limit_price: 科创板 限价
        :return: {'entrust_no': '委托单号'}
        """
        self._switch_left_menus(["市价委托", "卖出"])
        return self.market_trade(security, amount, ttype, limit_price=limit_price)

    def market_trade(self, security: str, amount: Union[int, float], ttype: Optional[str] = None,
                     limit_price: Optional[Union[float, str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        市价交易
        :param security: 六位证券代码
        :param amount: 交易数量
        :param ttype: 市价委托类型，默认客户端默认选择，
                     深市可选 ['对手方最优价格', '本方最优价格', '即时成交剩余撤销', '最优五档即时成交剩余 '全额成交或撤销']
                     沪市可选 ['最优五档成交剩余撤销', '最优五档成交剩余转限价']
        :return: {'entrust_no': '委托单号'}
        """
        code: str = security[-6:]
        self._type_edit_control_keys(self._config.TRADE_SECURITY_CONTROL_ID, code)
        if ttype is not None:
            retry: int = 0
            retry_max: int = 10
            while retry < retry_max:
                try:
                    self._set_market_trade_type(ttype)
                    break
                except Exception:
                    retry += 1
                    self.wait(0.1)
        self._set_market_trade_params(security, amount, limit_price=limit_price)
        self._submit_trade()
        return self._handle_pop_dialogs(
            handler_class=pop_dialog_handler.TradePopDialogHandler
        )

    def _set_market_trade_type(self, ttype: str) -> None:
        """根据选择的市价交易类型选择对应的下拉选项"""
        selects = self._main.child_window(
            control_id=self._config.TRADE_MARKET_TYPE_CONTROL_ID, class_name="ComboBox"
        )
        for i, text in enumerate(selects.texts()):
            # skip 0 index, because 0 index is current select index
            if i == 0:
                if re.search(ttype, text):  # 当前已经选中
                    return
                else:
                    continue
            if re.search(ttype, text):
                selects.select(i - 1)
                return
        raise TypeError("不支持对应的市价类型: {}".format(ttype))

    def _set_stock_exchange_type(self, ttype: str) -> None:
        """根据选择的市价交易类型选择对应的下拉选项"""
        selects = self._main.child_window(
            control_id=self._config.TRADE_STOCK_EXCHANGE_CONTROL_ID, class_name="ComboBox"
        )
        for i, text in enumerate(selects.texts()):
            # skip 0 index, because 0 index is current select index
            if i == 0:
                if ttype.strip() == text.strip():  # 当前已经选中
                    return
                else:
                    continue
            if ttype.strip() == text.strip():
                selects.select(i - 1)
                return
        raise TypeError("不支持对应的市场类型: {}".format(ttype))

    def auto_ipo(self) -> Dict[str, Any]:
        self._switch_left_menus(self._config.AUTO_IPO_MENU_PATH)
        stock_list: List[Any] = self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)
        if len(stock_list) == 0:
            return {"message": "今日无新股"}
        invalid_list_idx: List[int] = [
            i for i, v in enumerate(stock_list) if v[self.config.AUTO_IPO_NUMBER] <= 0
        ]
        if len(stock_list) == len(invalid_list_idx):
            return {"message": "没有发现可以申购的新股"}
        self._click(self._config.AUTO_IPO_SELECT_ALL_BUTTON_CONTROL_ID)
        self.wait(0.1)
        for row in invalid_list_idx:
            self._click_grid_by_row(row)
        self.wait(0.1)
        self._click(self._config.AUTO_IPO_BUTTON_CONTROL_ID)
        self.wait(0.1)
        return self._handle_pop_dialogs()

    def _click_grid_by_row(self, row: int) -> None:
        x: int = self._config.COMMON_GRID_LEFT_MARGIN
        y: int = (
            self._config.COMMON_GRID_FIRST_ROW_HEIGHT
            + self._config.COMMON_GRID_ROW_HEIGHT * row
        )
        self._app.top_window().child_window(
            control_id=self._config.COMMON_GRID_CONTROL_ID,
            class_name="CVirtualGridCtrl",
        ).click(coords=(x, y))

    @perf_clock
    def is_exist_pop_dialog(self) -> bool:
        self.wait(0.5)  # wait dialog display
        try:
            return (
                self._main.wrapper_object() != self._app.top_window().wrapper_object()
            )
        except (
            findwindows.ElementNotFoundError,
            timings.TimeoutError,
            RuntimeError,
        ) as ex:
            logger.exception("check pop dialog timeout")
            return False

    @perf_clock
    def close_pop_dialog(self) -> None:
        try:
            if self._main.wrapper_object() != self._app.top_window().wrapper_object():
                w = self._app.top_window()
                if w is not None:
                    w.close()
                    self.wait(0.2)
        except (
            findwindows.ElementNotFoundError,
            timings.TimeoutError,
            RuntimeError,
        ) as ex:
            pass

    def _run_exe_path(self, exe_path: str) -> str:
        return os.path.join(os.path.dirname(exe_path), "xiadan.exe")

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def exit(self) -> None:
        self._app.kill()

    def _close_prompt_windows(self) -> None:
        self.wait(1)
        for window in self._app.windows(class_name="#32770", visible_only=True):
            title: str = window.window_text()
            if title != self._config.TITLE:
                logging.info("close " + title)
                window.close()
                self.wait(0.2)
        self.wait(1)

    def close_pormpt_window_no_wait(self) -> None:
        for window in self._app.windows(class_name="#32770"):
            if window.window_text() != self._config.TITLE:
                window.close()

    def trade(self, security: str, price: Union[float, str], amount: Union[int, float]) -> Dict[str, Any]:
        self._set_trade_params(security, price, amount)
        self._submit_trade()
        return self._handle_pop_dialogs(
            handler_class=pop_dialog_handler.TradePopDialogHandler
        )

    def _click(self, control_id: int) -> None:
        self._app.top_window().child_window(
            control_id=control_id, class_name="Button"
        ).click()

    @perf_clock
    def _submit_trade(self) -> None:
        time.sleep(0.2)
        self._main.child_window(
            control_id=self._config.TRADE_SUBMIT_CONTROL_ID, class_name="Button"
        ).click()

    @perf_clock
    def __get_top_window_pop_dialog(self) -> Any:
        return self._app.top_window().window(
            control_id=self._config.POP_DIALOD_TITLE_CONTROL_ID
        )

    @perf_clock
    def _get_pop_dialog_title(self) -> str:
        return (
            self._app.top_window()
            .child_window(control_id=self._config.POP_DIALOD_TITLE_CONTROL_ID)
            .window_text()
        )

    def _set_trade_params(self, security: str, price: Union[float, str], amount: Union[int, float]) -> None:
        code: str = security[-6:]
        self._type_edit_control_keys(self._config.TRADE_SECURITY_CONTROL_ID, code)
        # wait security input finish
        self.wait(0.1)
        # 设置交易所
        if security.lower().startswith("sz"):
            self._set_stock_exchange_type("深圳Ａ股")
        if security.lower().startswith("sh"):
            self._set_stock_exchange_type("上海Ａ股")
        self.wait(0.1)
        self._type_edit_control_keys(
            self._config.TRADE_PRICE_CONTROL_ID,
            easyutils.round_price_by_code(price, code),
        )
        self._type_edit_control_keys(
            self._config.TRADE_AMOUNT_CONTROL_ID, str(int(amount))
        )

    def _set_market_trade_params(self, security: str, amount: Union[int, float],
                                 *, limit_price: Optional[Union[float, str]] = None) -> None:
        self._type_edit_control_keys(
            self._config.TRADE_AMOUNT_CONTROL_ID, str(int(amount))
        )
        self.wait(0.1)
        price_control = None
        if str(security).startswith("68"):  # 科创板存在限价
            try:
                price_control = self._main.child_window(
                    control_id=self._config.TRADE_PRICE_CONTROL_ID, class_name="Edit"
                )
            except Exception:
                pass
        if price_control is not None:
            price_control.set_edit_text(limit_price)

    def _get_grid_data(self, control_id: int) -> Any:
        return self.grid_strategy_instance.get(control_id)

    def _type_keys(self, control_id: int, text: str) -> None:
        self._main.child_window(control_id=control_id, class_name="Edit").set_edit_text(text)

    def _type_edit_control_keys(self, control_id: int, text: str) -> None:
        if not self._editor_need_type_keys:
            self._main.child_window(
                control_id=control_id, class_name="Edit"
            ).set_edit_text(text)
        else:
            editor = self._main.child_window(control_id=control_id, class_name="Edit")
            editor.select()
            editor.type_keys(text)

    def type_edit_control_keys(self, editor: Any, text: str) -> None:
        if not self._editor_need_type_keys:
            editor.set_edit_text(text)
        else:
            editor.select()
            editor.type_keys(text)

    def _collapse_left_menus(self) -> None:
        items = self._get_left_menus_handle().roots()
        for item in items:
            item.collapse()

    @perf_clock
    def _switch_left_menus(self, path: Union[str, List[str]], sleep: float = 0.2) -> None:
        self.close_pop_dialog()
        self._get_left_menus_handle().get_item(path).select()
        self._app.top_window().type_keys('{F5}')
        self.wait(sleep)

    def _switch_left_menus_by_shortcut(self, shortcut: str, sleep: float = 0.5) -> None:
        self.close_pop_dialog()
        self._app.top_window().type_keys(shortcut)
        self.wait(sleep)

    @functools.lru_cache()
    def _get_left_menus_handle(self) -> Any:
        count: int = 2
        while True:
            try:
                handle = self._main.child_window(
                    control_id=129, class_name="SysTreeView32"
                )
                if count <= 0:
                    return handle
                # sometime can't find handle ready, must retry
                handle.wait("ready", 2)
                return handle
            except Exception as ex:
                logger.exception("error occurred when trying to get left menus")
            count = count - 1

    def _cancel_entrust_by_double_click(self, row: int) -> None:
        x: int = self._config.CANCEL_ENTRUST_GRID_LEFT_MARGIN
        y: int = (
            self._config.CANCEL_ENTRUST_GRID_FIRST_ROW_HEIGHT
            + self._config.CANCEL_ENTRUST_GRID_ROW_HEIGHT * row
        )
        self._app.top_window().child_window(
            control_id=self._config.COMMON_GRID_CONTROL_ID,
            class_name="CVirtualGridCtrl",
        ).double_click(coords=(x, y))

    def refresh(self) -> None:
        self.refresh_strategy.set_trader(self)
        self.refresh_strategy.refresh()

    @perf_clock
    def _handle_pop_dialogs(self, handler_class: Type[pop_dialog_handler.PopDialogHandler] = pop_dialog_handler.PopDialogHandler) -> Dict[str, Any]:
        handler = handler_class(self._app)
        while self.is_exist_pop_dialog():
            try:
                title: str = self._get_pop_dialog_title()
            except pywinauto.findwindows.ElementNotFoundError:
                return {"message": "success"}
            result: Optional[Dict[str, Any]] = handler.handle(title)
            if result:
                return result
        return {"message": "success"}


class BaseLoginClientTrader(ClientTrader):
    @abc.abstractmethod
    def login(self, user: str, password: str, exe_path: str, comm_password: Optional[str] = None, **kwargs: Any) -> None:
        """Login Client Trader"""
        pass

    def prepare(
        self,
        config_path: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        exe_path: Optional[str] = None,
        comm_password: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        登陆客户端
        :param config_path: 登陆配置文件，跟参数登陆方式二选一
        :param user: 账号
        :param password: 明文密码
        :param exe_path: 客户端路径类似 r'C:\\htzqzyb2\\xiadan.exe', 默认 r'C:\\htzqzyb2\\xiadan.exe'
        :param comm_password: 通讯密码
        :return:
        """
        if config_path is not None:
            account: Dict[str, Any] = file2dict(config_path)
            user = account["user"]
            password = account["password"]
            comm_password = account.get("comm_password")
            exe_path = account.get("exe_path")
        self.login(
            user,  # type: ignore
            password,  # type: ignore
            exe_path or self._config.DEFAULT_EXE_PATH,  # type: ignore
            comm_password,
            **kwargs
        )
        self._init_toolbar()