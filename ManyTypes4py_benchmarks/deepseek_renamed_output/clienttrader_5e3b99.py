import abc
import functools
import logging
import os
import re
import sys
import time
from typing import Type, Union, Any, Dict, List, Optional, Callable
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

if not sys.platform.startswith('darwin'):
    import pywinauto
    import pywinauto.clipboard


class IClientTrader(abc.ABC):
    @property
    @abc.abstractmethod
    def func_1g383z97(self) -> Any:
        """Return current app instance"""
        pass

    @property
    @abc.abstractmethod
    def func_d571n4xg(self) -> Any:
        """Return current main window instance"""
        pass

    @property
    @abc.abstractmethod
    def func_rfcjfcvg(self) -> Any:
        """Return current config instance"""
        pass

    @abc.abstractmethod
    def func_m3pig290(self, seconds: float) -> None:
        """Wait for operation return"""
        pass

    @abc.abstractmethod
    def func_om1km6i8(self) -> None:
        """Refresh data"""
        pass

    @abc.abstractmethod
    def func_wjnh18ay(self) -> bool:
        pass


class ClientTrader(IClientTrader):
    _editor_need_type_keys: bool = False
    grid_strategy: Type[IGridStrategy] = grid_strategies.Copy
    _grid_strategy_instance: Optional[IGridStrategy] = None
    refresh_strategy: IRefreshStrategy = refresh_strategies.Switch()

    def func_luqr1t0i(self) -> None:
        """
        有些客户端无法通过 set_edit_text 方法输入内容，可以通过使用 type_keys 方法绕过
        """
        self._editor_need_type_keys = True

    @property
    def func_iav83svr(self) -> IGridStrategy:
        if self._grid_strategy_instance is None:
            self._grid_strategy_instance = self.grid_strategy if isinstance(
                self.grid_strategy, IGridStrategy) else self.grid_strategy()
            self._grid_strategy_instance.set_trader(self)
        return self._grid_strategy_instance

    def __init__(self) -> None:
        self._config: Any = client.create(self.broker_type)
        self._app: Optional[Any] = None
        self._main: Optional[Any] = None
        self._toolbar: Optional[Any] = None

    @property
    def func_1g383z97(self) -> Optional[Any]:
        return self._app

    @property
    def func_d571n4xg(self) -> Optional[Any]:
        return self._main

    @property
    def func_rfcjfcvg(self) -> Any:
        return self._config

    def func_8egbjqyb(self, exe_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        直接连接登陆后的客户端
        :param exe_path: 客户端路径类似 r'C:\\htzqzyb2\\xiadan.exe', 默认 r'C:\\htzqzyb2\\xiadan.exe'
        :return:
        """
        connect_path = exe_path or self._config.DEFAULT_EXE_PATH
        if connect_path is None:
            raise ValueError(
                '参数 exe_path 未设置，请设置客户端对应的 exe 地址,类似 C:\\客户端安装目录\\xiadan.exe')
        self._app = pywinauto.Application().connect(path=connect_path,
            timeout=10)
        self._close_prompt_windows()
        self._main = self._app.top_window()
        self._init_toolbar()

    @property
    def func_bj2j0xvu(self) -> str:
        return 'ths'

    @property
    def func_w5bd06s0(self) -> Dict[str, float]:
        self._switch_left_menus(['查询[F4]', '资金股票'])
        return self._get_balance_from_statics()

    def func_jv32y9yh(self) -> None:
        self._toolbar = self._main.child_window(class_name='ToolbarWindow32')

    def func_v2dw8zga(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, control_id in self._config.BALANCE_CONTROL_ID_GROUP.items():
            result[key] = float(self._main.child_window(control_id=
                control_id, class_name='Static').window_text())
        return result

    @property
    def func_7g7osy5y(self) -> List[Dict[str, str]]:
        self._switch_left_menus(['查询[F4]', '资金股票'])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def func_e1zbuocp(self) -> List[Dict[str, str]]:
        self._switch_left_menus(['查询[F4]', '当日委托'])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def func_rlc7zvun(self) -> List[Dict[str, str]]:
        self._switch_left_menus(['查询[F4]', '当日成交'])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @property
    def func_4uhqergp(self) -> List[Dict[str, str]]:
        self.refresh()
        self._switch_left_menus(['撤单[F3]'])
        return self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)

    @perf_clock
    def func_h9w7q8m8(self, entrust_no: str) -> Dict[str, str]:
        self.refresh()
        for i, entrust in enumerate(self.cancel_entrusts):
            if entrust[self._config.CANCEL_ENTRUST_ENTRUST_FIELD
                ] == entrust_no:
                self._cancel_entrust_by_double_click(i)
                return self._handle_pop_dialogs()
        return {'message': '委托单状态错误不能撤单, 该委托单可能已经成交或者已撤'}

    def func_emnvtunz(self) -> None:
        self.refresh()
        self._switch_left_menus(['撤单[F3]'])
        self._app.top_window().child_window(control_id=self._config.
            TRADE_CANCEL_ALL_ENTRUST_CONTROL_ID, class_name='Button',
            title_re='全撤.*').click()
        self.wait(0.2)
        if self.is_exist_pop_dialog():
            w = self._app.top_window()
            if w is not None:
                btn = w['是(Y)']
                if btn is not None:
                    btn.click()
                    self.wait(0.2)
        self.close_pop_dialog()

    @perf_clock
    def func_q5dn36ga(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        self._switch_left_menus(['债券回购', '融资回购（正回购）'])
        return self.trade(security, price, amount)

    @perf_clock
    def func_wo9iy0sx(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        self._switch_left_menus(['债券回购', '融劵回购（逆回购）'])
        return self.trade(security, price, amount)

    @perf_clock
    def func_668breqb(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        self._switch_left_menus(['买入[F1]'])
        return self.trade(security, price, amount)

    @perf_clock
    def func_4wdlx54z(self, security: str, price: float, amount: float, **kwargs: Any) -> Dict[str, str]:
        self._switch_left_menus(['卖出[F2]'])
        return self.trade(security, price, amount)

    @perf_clock
    def func_fx2r4xyd(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None,
        **kwargs: Any) -> Dict[str, str]:
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
        self._switch_left_menus(['市价委托', '买入'])
        return self.market_trade(security, amount, ttype, limit_price=
            limit_price)

    @perf_clock
    def func_7qxop5pt(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None,
        **kwargs: Any) -> Dict[str, str]:
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
        self._switch_left_menus(['市价委托', '卖出'])
        return self.market_trade(security, amount, ttype, limit_price=
            limit_price)

    def func_dfm36zar(self, security: str, amount: float, ttype: Optional[str] = None, limit_price: Optional[float] = None,
        **kwargs: Any) -> Dict[str, str]:
        """
        市价交易
        :param security: 六位证券代码
        :param amount: 交易数量
        :param ttype: 市价委托类型，默认客户端默认选择，
                     深市可选 ['对手方最优价格', '本方最优价格', '即时成交剩余撤销', '最优五档即时成交剩余 '全额成交或撤销']
                     沪市可选 ['最优五档成交剩余撤销', '最优五档成交剩余转限价']

        :return: {'entrust_no': '委托单号'}
        """
        code = security[-6:]
        self._type_edit_control_keys(self._config.TRADE_SECURITY_CONTROL_ID,
            code)
        if ttype is not None:
            retry = 0
            retry_max = 10
            while retry < retry_max:
                try:
                    self._set_market_trade_type(ttype)
                    break
                except:
                    retry += 1
                    self.wait(0.1)
        self._set_market_trade_params(security, amount, limit_price=limit_price
            )
        self._submit_trade()
        return self._handle_pop_dialogs(handler_class=pop_dialog_handler.
            TradePopDialogHandler)

    def func_baq3kxnc(self, ttype: str) -> None:
        """根据选择的市价交易类型选择对应的下拉选项"""
        selects = self._main.child_window(control_id=self._config.
            TRADE_MARKET_TYPE_CONTROL_ID, class_name='ComboBox')
        for i, text in enumerate(selects.texts()):
            if i == 0:
                if re.search(ttype, text):
                    return
                else:
                    continue
            if re.search(ttype, text):
                selects.select(i - 1)
                return
        raise TypeError('不支持对应的市价类型: {}'.format(ttype))

    def func_6kmzolhh(self, ttype: str) -> None:
        """根据选择的市价交易类型选择对应的下拉选项"""
        selects = self._main.child_window(control_id=self._config.
            TRADE_STOCK_EXCHANGE_CONTROL_ID, class_name='ComboBox')
        for i, text in enumerate(selects.texts()):
            if i == 0:
                if ttype.strip() == text.strip():
                    return
                else:
                    continue
            if ttype.strip() == text.strip():
                selects.select(i - 1)
                return
        raise TypeError('不支持对应的市场类型: {}'.format(ttype))

    def func_707ufey9(self) -> Dict[str, str]:
        self._switch_left_menus(self._config.AUTO_IPO_MENU_PATH)
        stock_list = self._get_grid_data(self._config.COMMON_GRID_CONTROL_ID)
        if len(stock_list) == 0:
            return {'message': '今日无新股'}
        invalid_list_idx = [i for i, v in enumerate(stock_list) if v[self.
            config.AUTO_IPO_NUMBER] <= 0]
        if len(stock_list) == len(invalid_list_idx):
            return {'message': '没有发现可以申购的新股'}
        self._click(self._config.AUTO_IPO_SELECT_ALL_BUTTON_CONTROL_ID)
        self.wait(0.1)
        for row in invalid_list_idx:
            self._click_grid_by_row(row)
        self.wait(0.1)
        self._click(self._config.AUTO_IPO_BUTTON_CONTROL_ID)
        self.wait(0.1)
        return self._handle_pop_dialogs()

    def func_uihjtpyz(self, row: int) -> None:
        x = self._config.COMMON_GRID_LEFT_MARGIN
        y = (self._config.COMMON_GRID_FIRST_ROW_HEIGHT + self._config.
            COMMON_GRID_ROW_HEIGHT * row)
        self._app.top_window().child_window(control_id=self._config.
            COMMON_GRID_CONTROL_ID, class_name='CVirtualGridCtrl').click(coords
            =(x, y))

    @perf_clock
    def func_wjnh18ay(self) -> bool:
        self.wait(0.5)
        try:
            return self._main.wrapper_object() != self._app.top_window(
                ).wrapper_object()
        except (findwindows.ElementNotFoundError, timings.TimeoutError,
            RuntimeError) as ex:
            logger.exception('check pop dialog timeout')
            return False

    @perf_clock
    def func_jy8x8zgu(self) -> None:
        try:
            if self._main.wrapper_object() != self._app.top_window(
                ).wrapper_object():
                w = self._app.top_window()
                if w is not None:
                    w.close()
                    self.wait(0.2)
        except (findwindows.ElementNotFoundError, timings.TimeoutError,
            RuntimeError) as ex:
            pass

    def func_0g8f8u6x(self, exe_path: str) -> str:
        return os.path.join(os.path.dirname(exe_path), 'xiadan.exe')

    def func_m3pig290(self, seconds: float) -> None:
        time.sleep(seconds)

    def func_vuyhoea9(self) -> None:
        self._app.kill()

    def func_6ekypu09(self) -> None:
        self.wait(1)
        for window in self._app.windows(class_name='#32770', visible_only=True
            ):
            title = window.window_text()
            if title != self._config.TITLE:
                logging.info('close ' + title)
                window.close()
                self.wait(0.2)
        self.wait(1)

    def func_waav6od5(self) -> None:
        for window in self._app.windows(class_name='#32770'):
            if window.window_text() != self._config.TITLE:
                window.close()

    def func_6pc1ckwb(self, security: str, price: float, amount: float) -> Dict[str, str]:
        self._set_trade_params(security, price, amount)
        self._submit_trade()
        return self._handle_pop_dialogs(handler_class=pop_dialog_handler.
            TradePopDialogHandler)

    def func_fhmq2vok(self, control_id: int) -> None:
        self._app.top_window().child_window(control_id=control_id,
            class_name='Button').click()

    @perf_clock
    def func_n4dyqbsu(self) -> None:
        time.sleep(0.2)
        self._main.child_window(control_id=self._config.
            TRADE_SUBMIT_CONTROL_ID, class_name='Button').click()

    @perf_clock
    def __get_top_window_pop_dialog(self) -> Any:
        return self._app.top_window().window(control_id=self._config.
            POP_DIALOD_TITLE_CONTROL_ID)

    @perf_clock
    def func_hj1e27wr(self) -> str:
        return self._app.top_window().child_window(control_id=self._config.
            POP_DIALOD_TITLE_CONTROL_ID).window_text()

    def func_wusrmdoh(self, security: str, price: float, amount: float) -> None:
        code = security[-6:]
        self._type_edit_control_keys(self._config.TRADE_SECURITY_CONTROL_ID,
            code)
        self.wait(0.1)
        if security.lower().startswith('sz'):
            self._set_stock_exchange