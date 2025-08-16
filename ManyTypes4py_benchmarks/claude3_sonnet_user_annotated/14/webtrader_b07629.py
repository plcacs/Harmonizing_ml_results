# -*- coding: utf-8 -*-
import abc
import logging
import os
import re
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Union, TypeVar, Type

import requests
import requests.exceptions

from easytrader import exceptions
from easytrader.log import logger
from easytrader.utils.misc import file2dict, str2num
from easytrader.utils.stock import get_30_date


T = TypeVar('T', bound='WebTrader')

# noinspection PyIncorrectDocstring
class WebTrader(metaclass=abc.ABCMeta):
    global_config_path: str = os.path.dirname(__file__) + "/config/global.json"
    config_path: str = ""

    def __init__(self, debug: bool = True) -> None:
        self.__read_config()
        self.trade_prefix: str = self.config["prefix"]
        self.account_config: Union[Dict[str, Any], str] = ""
        self.heart_active: bool = True
        self.heart_thread: Thread = Thread(target=self.send_heartbeat)
        self.heart_thread.setDaemon(True)

        self.log_level: int = logging.DEBUG if debug else logging.INFO
        self.config: Dict[str, Any] = {}
        self.global_config: Dict[str, Any] = {}

    def read_config(self, path: str) -> None:
        try:
            self.account_config = file2dict(path)
        except ValueError:
            logger.error("配置文件格式有误，请勿使用记事本编辑，推荐 sublime text")
        for value in self.account_config:
            if isinstance(value, int):
                logger.warning("配置文件的值最好使用双引号包裹，使用字符串，否则可能导致不可知问题")

    def prepare(self, config_file: Optional[str] = None, user: Optional[str] = None, 
                password: Optional[str] = None, **kwargs: Any) -> None:
        """登录的统一接口
        :param config_file 登录数据文件，若无则选择参数登录模式
        :param user: 各家券商的账号
        :param password: 密码, 券商为加密后的密码
        :param kwargs: 其他参数
        """
        if config_file is not None:
            self.read_config(config_file)
        else:
            self._prepare_account(user, password, **kwargs)
        self.autologin()

    def _prepare_account(self, user: Optional[str], password: Optional[str], **kwargs: Any) -> None:
        """映射用户名密码到对应的字段"""
        raise Exception("支持参数登录需要实现此方法")

    def autologin(self, limit: int = 10) -> None:
        """实现自动登录
        :param limit: 登录次数限制
        """
        for _ in range(limit):
            if self.login():
                break
        else:
            raise exceptions.NotLoginError(
                "登录失败次数过多, 请检查密码是否正确 / 券商服务器是否处于维护中 / 网络连接是否正常"
            )
        self.keepalive()

    def login(self) -> bool:
        pass

    def keepalive(self) -> None:
        """启动保持在线的进程 """
        if self.heart_thread.is_alive():
            self.heart_active = True
        else:
            self.heart_thread.start()

    def send_heartbeat(self) -> None:
        """每隔10秒查询指定接口保持 token 的有效性"""
        while True:
            if self.heart_active:
                self.check_login()
            else:
                time.sleep(1)

    def check_login(self, sleepy: int = 30) -> None:
        logger.setLevel(logging.ERROR)
        try:
            response = self.heartbeat()
            self.check_account_live(response)
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.RequestException as e:
            logger.setLevel(self.log_level)
            logger.error("心跳线程发现账户出现错误: %s %s, 尝试重新登陆", e.__class__, e)
            self.autologin()
        finally:
            logger.setLevel(self.log_level)
        time.sleep(sleepy)

    def heartbeat(self) -> Any:
        return self.balance

    def check_account_live(self, response: Any) -> None:
        pass

    def exit(self) -> None:
        """结束保持 token 在线的进程"""
        self.heart_active = False

    def __read_config(self) -> None:
        """读取 config"""
        self.config = file2dict(self.config_path)
        self.global_config = file2dict(self.global_config_path)
        self.config.update(self.global_config)

    @property
    def balance(self) -> Any:
        return self.get_balance()

    def get_balance(self) -> Any:
        """获取账户资金状况"""
        return self.do(self.config["balance"])

    @property
    def position(self) -> Any:
        return self.get_position()

    def get_position(self) -> Any:
        """获取持仓"""
        return self.do(self.config["position"])

    @property
    def entrust(self) -> Any:
        return self.get_entrust()

    def get_entrust(self) -> Any:
        """获取当日委托列表"""
        return self.do(self.config["entrust"])

    @property
    def current_deal(self) -> Any:
        return self.get_current_deal()

    def get_current_deal(self) -> Any:
        """获取当日委托列表"""
        # return self.do(self.config['current_deal'])
        logger.warning("目前仅在 佣金宝/银河子类 中实现, 其余券商需要补充")

    @property
    def exchangebill(self) -> Any:
        """
        默认提供最近30天的交割单, 通常只能返回查询日期内最新的 90 天数据。
        :return:
        """
        # TODO 目前仅在 华泰子类 中实现
        start_date, end_date = get_30_date()
        return self.get_exchangebill(start_date, end_date)

    def get_exchangebill(self, start_date: str, end_date: str) -> Any:
        """
        查询指定日期内的交割单
        :param start_date: 20160211
        :param end_date: 20160211
        :return:
        """
        logger.warning("目前仅在 华泰子类 中实现, 其余券商需要补充")

    def get_ipo_limit(self, stock_code: str) -> Any:
        """
        查询新股申购额度申购上限
        :param stock_code: 申购代码 ID
        :return:
        """
        logger.warning("目前仅在 佣金宝子类 中实现, 其余券商需要补充")

    def do(self, params: Dict[str, Any]) -> Any:
        """发起对 api 的请求并过滤返回结果
        :param params: 交易所需的动态参数"""
        request_params = self.create_basic_params()
        request_params.update(params)
        response_data = self.request(request_params)
        try:
            format_json_data = self.format_response_data(response_data)
        # pylint: disable=broad-except
        except Exception:
            # Caused by server force logged out
            return None
        return_data = self.fix_error_data(format_json_data)
        try:
            self.check_login_status(return_data)
        except exceptions.NotLoginError:
            self.autologin()
        return return_data

    def create_basic_params(self) -> Dict[str, Any]:
        """生成基本的参数"""
        return {}

    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """请求并获取 JSON 数据
        :param params: Get 参数"""
        return {}

    def format_response_data(self, data: Any) -> Any:
        """格式化返回的 json 数据
        :param data: 请求返回的数据 """
        return data

    def fix_error_data(self, data: Any) -> Any:
        """若是返回错误移除外层的列表
        :param data: 需要判断是否包含错误信息的数据"""
        return data

    def format_response_data_type(self, response_data: Union[List[Dict[str, Any]], Any]) -> Any:
        """格式化返回的值为正确的类型
        :param response_data: 返回的数据
        """
        if isinstance(response_data, list) and not isinstance(
            response_data, str
        ):
            return response_data

        int_match_str = "|".join(self.config["response_format"]["int"])
        float_match_str = "|".join(self.config["response_format"]["float"])
        for item in response_data:
            for key in item:
                try:
                    if re.search(int_match_str, key) is not None:
                        item[key] = str2num(item[key], "int")
                    elif re.search(float_match_str, key) is not None:
                        item[key] = str2num(item[key], "float")
                except ValueError:
                    continue
        return response_data

    def check_login_status(self, return_data: Any) -> None:
        pass
