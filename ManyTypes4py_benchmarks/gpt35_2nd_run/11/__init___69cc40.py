from requests import RequestException

class VenstarColorTouchMock:
    def __init__(self, addr: str, timeout: int, user: str = None, password: str = None, pin: str = None, proto: str = 'http', SSLCert: bool = False) -> None:
        self.status: dict = {}
        self.model: str = 'COLORTOUCH'
        self._api_ver: int = 7
        self._firmware_ver: tuple = (5, 28)
        self.name: str = 'TestVenstar'
        self._info: dict = {}
        self._sensors: dict = {}
        self.alerts: dict = {}
        self.MODE_OFF: int = 0
        self.MODE_HEAT: int = 1
        self.MODE_COOL: int = 2
        self.MODE_AUTO: int = 3
        self._type: str = 'residential'

    def login(self) -> bool:
        return True

    def _request(self, path: str, data: dict = None) -> None:
        self.status = {}

    def update(self) -> bool:
        return True

    def update_info(self) -> bool:
        self.name = 'username'
        return True

    def broken_update_info(self) -> None:
        raise RequestException

    def update_sensors(self) -> bool:
        return True

    def update_runtimes(self) -> bool:
        return True

    def update_alerts(self) -> bool:
        return True

    def get_runtimes(self) -> dict:
        return {}
