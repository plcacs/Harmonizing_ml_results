from typing import Dict, Any, List

class WebTrader(metaclass=abc.ABCMeta):
    global_config_path: str = os.path.dirname(__file__) + '/config/global.json'
    config_path: str = ''

    def __init__(self, debug: bool = True) -> None:
        self.trade_prefix: str
        self.account_config: Dict[str, Any]
        self.heart_active: bool = True
        self.heart_thread: Thread
        self.log_level: int

    def read_config(self, path: str) -> None:
        pass

    def prepare(self, config_file: str = None, user: str = None, password: str = None, **kwargs: Any) -> None:
        pass

    def _prepare_account(self, user: str, password: str, **kwargs: Any) -> None:
        pass

    def autologin(self, limit: int = 10) -> None:
        pass

    def login(self) -> None:
        pass

    def keepalive(self) -> None:
        pass

    def send_heartbeat(self) -> None:
        pass

    def check_login(self, sleepy: int = 30) -> None:
        pass

    def heartbeat(self) -> Any:
        pass

    def check_account_live(self, response: Any) -> None:
        pass

    def exit(self) -> None:
        pass

    def __read_config(self) -> None:
        pass

    @property
    def balance(self) -> Any:
        pass

    def get_balance(self) -> Any:
        pass

    @property
    def position(self) -> Any:
        pass

    def get_position(self) -> Any:
        pass

    @property
    def entrust(self) -> Any:
        pass

    def get_entrust(self) -> Any:
        pass

    @property
    def current_deal(self) -> None:
        pass

    def get_current_deal(self) -> None:
        pass

    @property
    def exchangebill(self) -> Any:
        pass

    def get_exchangebill(self, start_date: str, end_date: str) -> Any:
        pass

    def get_ipo_limit(self, stock_code: str) -> None:
        pass

    def do(self, params: Dict[str, Any]) -> Any:
        pass

    def create_basic_params(self) -> Dict[str, Any]:
        pass

    def request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def format_response_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def fix_error_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def format_response_data_type(self, response_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def check_login_status(self, return_data: Dict[str, Any]) -> None:
        pass
