    def market_is_future(self, market: dict) -> bool:
    
    def additional_exchange_init(self) -> None:
    
    def _lev_prep(self, pair: str, leverage: float, side: str, accept_fail: bool = False) -> None:
    
    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = 'GTC') -> dict:
    
    def _order_needs_price(self, side: str, ordertype: str) -> bool:
    
    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: list) -> float:
    
    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
    
    def fetch_orders(self, pair: str, since: datetime, params: dict = None) -> list:
    
    def fetch_order(self, order_id: str, pair: str, params: dict = None) -> dict:
    
    def get_leverage_tiers(self) -> list:
