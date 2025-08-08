from __future__ import division, print_function, unicode_literals
import json
import re
from datetime import datetime
from numbers import Number
from threading import Thread
from easytrader.follower import BaseFollower
from easytrader.log import logger
from easytrader.utils.misc import parse_cookies_str

class XueQiuFollower(BaseFollower):
    LOGIN_PAGE: str = 'https://www.xueqiu.com'
    LOGIN_API: str = 'https://xueqiu.com/snowman/login'
    TRANSACTION_API: str = 'https://xueqiu.com/cubes/rebalancing/history.json'
    PORTFOLIO_URL: str = 'https://xueqiu.com/p/'
    WEB_REFERER: str = 'https://www.xueqiu.com'

    def __init__(self) -> None:
        super().__init__()
        self._adjust_sell: bool = None
        self._users: list = None

    def login(self, user: str = None, password: str = None, **kwargs) -> None:
        ...

    def follow(self, users: list, strategies: list, total_assets: list = 10000, initial_assets: list = None, adjust_sell: bool = False, track_interval: int = 10, trade_cmd_expire_seconds: int = 120, cmd_cache: bool = True, slippage: float = 0.0) -> None:
        ...

    def calculate_assets(self, strategy_url: str, total_assets: Number = None, initial_assets: Number = None) -> Number:
        ...

    @staticmethod
    def extract_strategy_id(strategy_url: str) -> str:
        ...

    def extract_strategy_name(self, strategy_url: str) -> str:
        ...

    def extract_transactions(self, history: dict) -> list:
        ...

    def create_query_transaction_params(self, strategy: str) -> dict:
        ...

    def none_to_zero(self, data) -> int:
        ...

    def project_transactions(self, transactions: list, assets: Number) -> None:
        ...

    def _adjust_sell_amount(self, stock_code: str, amount: int) -> int:
        ...

    def _get_portfolio_info(self, portfolio_code: str) -> dict:
        ...

    def _get_portfolio_net_value(self, portfolio_code: str) -> float:
        ...
