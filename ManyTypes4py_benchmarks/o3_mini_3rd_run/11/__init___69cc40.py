from typing import Optional, Tuple, Any, Dict
from requests import RequestException

class VenstarColorTouchMock:
    """Mock Venstar Library."""

    def __init__(
        self,
        addr: str,
        timeout: int,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pin: Optional[str] = None,
        proto: str = 'http',
        SSLCert: bool = False
    ) -> None:
        """Initialize the Venstar library."""
        self.status: Dict[Any, Any] = {}
        self.model: str = 'COLORTOUCH'
        self._api_ver: int = 7
        self._firmware_ver: Tuple[int, int] = (5, 28)
        self.name: str = 'TestVenstar'
        self._info: Dict[Any, Any] = {}
        self._sensors: Dict[Any, Any] = {}
        self.alerts: Dict[Any, Any] = {}
        self.MODE_OFF: int = 0
        self.MODE_HEAT: int = 1
        self.MODE_COOL: int = 2
        self.MODE_AUTO: int = 3
        self._type: str = 'residential'

    def login(self) -> bool:
        """Mock login."""
        return True

    def _request(self, path: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Mock request."""
        self.status = {}

    def update(self) -> bool:
        """Mock update."""
        return True

    def update_info(self) -> bool:
        """Mock update_info."""
        self.name = 'username'
        return True

    def broken_update_info(self) -> None:
        """Mock a update_info that raises Exception."""
        raise RequestException

    def update_sensors(self) -> bool:
        """Mock update_sensors."""
        return True

    def update_runtimes(self) -> bool:
        """Mock update_runtimes."""
        return True

    def update_alerts(self) -> bool:
        """Mock update_alerts."""
        return True

    def get_runtimes(self) -> Dict[Any, Any]:
        """Mock get runtimes."""
        return {}