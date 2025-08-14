#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from typing import Any, Dict, Optional
from easytrader.utils.misc import file2dict


def use(broker: str, host: str, port: int = 1430, **kwargs: Any) -> "RemoteClient":
    return RemoteClient(broker, host, port, **kwargs)


class RemoteClient:
    def __init__(self, broker: str, host: str, port: int = 1430, **kwargs: Any) -> None:
        self._s: requests.Session = requests.session()
        self._api: str = "http://{}:{}".format(host, port)
        self._broker: str = broker

    def prepare(
        self,
        config_path: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        exe_path: Optional[str] = None,
        comm_password: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        登陆客户端
        :param config_path: 登陆配置文件，跟参数登陆方式二选一
        :param user: 账号
        :param password: 明文密码
        :param exe_path: 客户端路径类似 r'C:\\htzqzyb2\\xiadan.exe',
            默认 r'C:\\htzqzyb2\\xiadan.exe'
        :param comm_password: 通讯密码
        :return:
        """
        params: Dict[str, Any] = locals().copy()
        params.pop("self")

        if config_path is not None:
            account: Dict[str, Any] = file2dict(config_path)
            params["user"] = account["user"]
            params["password"] = account["password"]

        params["broker"] = self._broker

        response: requests.Response = self._s.post(self._api + "/prepare", json=params)
        if response.status_code >= 300:
            raise Exception(response.json()["error"])
        return response.json()

    @property
    def balance(self) -> Any:
        return self.common_get("balance")

    @property
    def position(self) -> Any:
        return self.common_get("position")

    @property
    def today_entrusts(self) -> Any:
        return self.common_get("today_entrusts")

    @property
    def today_trades(self) -> Any:
        return self.common_get("today_trades")

    @property
    def cancel_entrusts(self) -> Any:
        return self.common_get("cancel_entrusts")

    def auto_ipo(self) -> Any:
        return self.common_get("auto_ipo")

    def exit(self) -> Any:
        return self.common_get("exit")

    def common_get(self, endpoint: str) -> Any:
        response: requests.Response = self._s.get(self._api + "/" + endpoint)
        if response.status_code >= 300:
            raise Exception(response.json()["error"])
        return response.json()

    def buy(self, security: str, price: float, amount: int, **kwargs: Any) -> Any:
        params: Dict[str, Any] = locals().copy()
        params.pop("self")

        response: requests.Response = self._s.post(self._api + "/buy", json=params)
        if response.status_code >= 300:
            raise Exception(response.json()["error"])
        return response.json()

    def sell(self, security: str, price: float, amount: int, **kwargs: Any) -> Any:
        params: Dict[str, Any] = locals().copy()
        params.pop("self")

        response: requests.Response = self._s.post(self._api + "/sell", json=params)
        if response.status_code >= 300:
            raise Exception(response.json()["error"])
        return response.json()

    def cancel_entrust(self, entrust_no: int) -> Any:
        params: Dict[str, Any] = locals().copy()
        params.pop("self")

        response: requests.Response = self._s.post(self._api + "/cancel_entrust", json=params)
        if response.status_code >= 300:
            raise Exception(response.json()["error"])
        return response.json()