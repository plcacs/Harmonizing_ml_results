import requests
from easytrader.utils.misc import file2dict
from typing import Dict, Any, Union

def use(broker: str, host: str, port: int = 1430, **kwargs: Any) -> 'RemoteClient':
    return RemoteClient(broker, host, port)

class RemoteClient:
    def __init__(self, broker: str, host: str, port: int = 1430, **kwargs: Any) -> None:
        self._s = requests.session()
        self._api = 'http://{}:{}'.format(host, port)
        self._broker = broker

    def prepare(self, config_path: str = None, user: str = None, password: str = None, exe_path: str = None, comm_password: str = None, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Union[str, int]] = locals().copy()
        params.pop('self')
        if config_path is not None:
            account = file2dict(config_path)
            params['user'] = account['user']
            params['password'] = account['password']
        params['broker'] = self._broker
        response = self._s.post(self._api + '/prepare', json=params)
        if response.status_code >= 300:
            raise Exception(response.json()['error'])
        return response.json()

    @property
    def balance(self) -> Dict[str, Any]:
        return self.common_get('balance')

    @property
    def position(self) -> Dict[str, Any]:
        return self.common_get('position')

    @property
    def today_entrusts(self) -> Dict[str, Any]:
        return self.common_get('today_entrusts')

    @property
    def today_trades(self) -> Dict[str, Any]:
        return self.common_get('today_trades')

    @property
    def cancel_entrusts(self) -> Dict[str, Any]:
        return self.common_get('cancel_entrusts')

    def auto_ipo(self) -> Dict[str, Any]:
        return self.common_get('auto_ipo')

    def exit(self) -> Dict[str, Any]:
        return self.common_get('exit')

    def common_get(self, endpoint: str) -> Dict[str, Any]:
        response = self._s.get(self._api + '/' + endpoint)
        if response.status_code >= 300:
            raise Exception(response.json()['error'])
        return response.json()

    def buy(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Union[str, float, int]] = locals().copy()
        params.pop('self')
        response = self._s.post(self._api + '/buy', json=params)
        if response.status_code >= 300:
            raise Exception(response.json()['error'])
        return response.json()

    def sell(self, security: str, price: float, amount: int, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Union[str, float, int]] = locals().copy()
        params.pop('self')
        response = self._s.post(self._api + '/sell', json=params)
        if response.status_code >= 300:
            raise Exception(response.json()['error'])
        return response.json()

    def cancel_entrust(self, entrust_no: str) -> Dict[str, Any]:
        params: Dict[str, str] = locals().copy()
        params.pop('self')
        response = self._s.post(self._api + '/cancel_entrust', json=params)
        if response.status_code >= 300:
            raise Exception(response.json()['error'])
        return response.json()
