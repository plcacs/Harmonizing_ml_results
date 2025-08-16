from typing import List, Dict

class BaseFollower(metaclass=abc.ABCMeta):
    LOGIN_PAGE: str = ''
    LOGIN_API: str = ''
    TRANSACTION_API: str = ''
    CMD_CACHE_FILE: str = 'cmd_cache.pk'
    WEB_REFERER: str = ''
    WEB_ORIGIN: str = ''

    def __init__(self) -> None:
        self.trade_queue: queue.Queue = queue.Queue()
        self.expired_cmds: set = set()
        self.s: requests.Session = requests.Session()
        self.s.verify: bool = False
        self.slippage: float = 0.0

    def login(self, user: str = None, password: str = None, **kwargs: Dict) -> None:
        ...

    def _generate_headers(self) -> Dict[str, str]:
        ...

    def check_login_success(self, rep: requests.Response) -> None:
        ...

    def create_login_params(self, user: str, password: str, **kwargs: Dict) -> Dict:
        ...

    def follow(self, users: List, strategies: List, track_interval: int = 1, trade_cmd_expire_seconds: int = 120, cmd_cache: bool = True, slippage: float = 0.0, **kwargs: Dict) -> None:
        ...

    def _calculate_price_by_slippage(self, action: str, price: float) -> float:
        ...

    def load_expired_cmd_cache(self) -> None:
        ...

    def start_trader_thread(self, users: List, trade_cmd_expire_seconds: int, entrust_prop: str = 'limit', send_interval: int = 0) -> None:
        ...

    @staticmethod
    def warp_list(value) -> List:
        ...

    @staticmethod
    def extract_strategy_id(strategy_url: str) -> str:
        ...

    def extract_strategy_name(self, strategy_url: str) -> str:
        ...

    def track_strategy_worker(self, strategy: str, name: str, interval: int = 10, **kwargs: Dict) -> None:
        ...

    @staticmethod
    def generate_expired_cmd_key(cmd: Dict) -> str:
        ...

    def is_cmd_expired(self, cmd: Dict) -> bool:
        ...

    def add_cmd_to_expired_cmds(self, cmd: Dict) -> None:
        ...

    @staticmethod
    def _is_number(s: str) -> bool:
        ...

    def _execute_trade_cmd(self, trade_cmd: Dict, users: List, expire_seconds: int, entrust_prop: str, send_interval: int) -> None:
        ...

    def trade_worker(self, users: List, expire_seconds: int = 120, entrust_prop: str = 'limit', send_interval: int = 0) -> None:
        ...

    def query_strategy_transaction(self, strategy: str, **kwargs: Dict) -> List:
        ...

    def extract_transactions(self, history: Dict) -> List:
        ...

    def create_query_transaction_params(self, strategy: str) -> Dict:
        ...

    @staticmethod
    def re_find(pattern: str, string: str, dtype) -> str:
        ...

    @staticmethod
    def re_search(pattern: str, string: str, dtype) -> str:
        ...

    def project_transactions(self, transactions: List, **kwargs: Dict) -> List:
        ...

    def order_transactions_sell_first(self, transactions: List) -> List:
        ...
