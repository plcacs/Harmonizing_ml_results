import abc
import logging
import os
import re
import time
from threading import Thread
import requests
import requests.exceptions
from easytrader import exceptions
from easytrader.log import logger
from easytrader.utils.misc import file2dict, str2num
from easytrader.utils.stock import get_30_date
from typing import Any, Dict, List, Optional, Union

class WebTrader(metaclass=abc.ABCMeta):
    global_config_path: str = os.path.dirname(__file__) + '/config/global.json'
    config_path: str = ''

    def __init__(self, debug: bool = True) -> None:
        self.__read_config()
        self.trade_prefix: str = self.config['prefix']
        self.account_config: Union[str, Dict[str, Any]] = ''
        self.heart_active: bool = True
        self.heart_thread: Thread = Thread(target=self.send_heartbeat)
        self.heart_thread.setDaemon(True)
        self.log_level: int = logging.DEBUG if debug else logging.INFO

    def read_config(self, path: str) -> None:
        try:
            self.account_config = file2dict(path)
        except ValueError:
            logger.error('配置文件格式有误，请勿使用记事本编辑，推荐 sublime text')
        for value in self.account_config:
            if isinstance(value, int):
                logger.warning('配置文件的值最好使用双引号包裹，使用字符串，否则可能导致不可知问题')

    def prepare(self, config_file: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, **kwargs: Any) -> None:
        if config_file is not None:
            self.read_config(config_file)
        else:
            self._prepare_account(user, password, **kwargs)
        self.autologin()

    def _prepare_account(self, user: Optional[str], password: Optional[str], **kwargs: Any) -> None:
        raise Exception('支持参数登录需要实现此方法')

    def autologin(self, limit: int = 10) -> None:
        for _ in range(limit):
            if self.login():
                break
        else:
            raise exceptions.NotLoginError('登录失败次数过多, 请检查密码是否正确 / 券商服务器是否处于维护中 / 网络连接是否正常')
        self.keepalive()

    def login(self) -> bool:
        pass

    def keepalive(self) -> None:
        if self.heart_thread.is_alive():
            self.heart_active = True
        else:
            self.heart_thread.start()

    def send_heartbeat(self) -> None:
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
            logger.error('心跳线程发现账户出现错误: %s %s, 尝试重新登陆', e.__class__, e)
            self.autologin()
        finally:
            logger.setLevel(self.log_level)
        time.sleep(sleepy)

    def heartbeat(self) -> Any:
        return self.balance

    def check_account_live(self, response: Any) -> None:
        pass

    def exit(self) -> None:
        self.heart_active = False

    def __read_config(self) -> None:
        self.config = file2dict(self.config_path)
        self.global_config = file2dict(self.global_config_path)
        self.config.update(self.global_config)

    @property
    def balance(self) -> Any:
        return self.get_balance()

    def get_balance(self) -> Any:
        return self.do(self.config['balance'])

    @property
    def position(self) -> Any:
        return self.get_position()

    def get_position(self) -> Any:
        return self.do(self.config['position'])

    @property
    def entrust(self) -> Any:
        return self.get_entrust()

    def get_entrust(self) -> Any:
        return self.do(self.config['entrust'])

    @property
    def current_deal(self) -> Any:
        return self.get_current_deal()

    def get_current_deal(self) -> None:
        logger.warning('目前仅在 佣金宝/银河子类 中实现, 其余券商需要补充')

    @property
    def exchangebill(self) -> Any:
        start_date, end_date = get_30_date()
        return self.get_exchangebill(start_date, end_date)

    def get_exchangebill(self, start_date: str, end_date: str) -> None:
        logger.warning('目前仅在 华泰子类 中实现, 其余券商需要补充')

    def get_ipo_limit(self, stock_code: str) -> None:
        logger.warning('目前仅在 佣金宝子类 中实现, 其余券商需要补充')

    def do(self, params: Dict[str, Any]) -> Any:
        request_params = self.create_basic_params()
        request_params.update(params)
        response_data = self.request(request_params)
        try:
            format_json_data = self.format_response_data(response_data)
        except Exception:
            return None
        return_data = self.fix_error_data(format_json_data)
        try:
            self.check_login_status(return_data)
        except exceptions.NotLoginError:
            self.autologin()
        return return_data

    def create_basic_params(self) -> Dict[str, Any]:
        return {}

    def request(self, params: Dict[str, Any]) -> Any:
        return {}

    def format_response_data(self, data: Any) -> Any:
        return data

    def fix_error_data(self, data: Any) -> Any:
        return data

    def format_response_data_type(self, response_data: Any) -> Any:
        if isinstance(response_data, list) and (not isinstance(response_data, str)):
            return response_data
        int_match_str = '|'.join(self.config['response_format']['int'])
        float_match_str = '|'.join(self.config['response_format']['float'])
        for item in response_data:
            for key in item:
                try:
                    if re.search(int_match_str, key) is not None:
                        item[key] = str2num(item[key], 'int')
                    elif re.search(float_match_str, key) is not None:
                        item[key] = str2num(item[key], 'float')
                except ValueError:
                    continue
        return response_data

    def check_login_status(self, return_data: Any) -> None:
        pass
