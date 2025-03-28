```python
# -*- coding: utf-8 -*-

from datetime import datetime
from threading import Thread
from typing import List, Union, Dict, Any

from easytrader.follower import BaseFollower
from easytrader.log import logger


class RiceQuantFollower(BaseFollower):
    def __init__(self) -> None:
        super().__init__()
        self.client = None

    def login(self, user: str = None, password: str = None, **kwargs: Any) -> None:
        from rqopen_client import RQOpenClient

        self.client = RQOpenClient(user, password, logger=logger)

    def follow(
        self,
        users: Union[List[Any], Any],
        run_id: Union[List[str], str],
        track_interval: int = 1,
        trade_cmd_expire_seconds: int = 120,
        cmd_cache: bool = True,
        entrust_prop: str = "limit",
        send_interval: int = 0,
    ) -> None:
        users = self.warp_list(users)
        run_ids = self.warp_list(run_id)

        if cmd_cache:
            self.load_expired_cmd_cache()

        self.start_trader_thread(
            users, trade_cmd_expire_seconds, entrust_prop, send_interval
        )

        workers: List[Thread] = []
        for id_ in run_ids:
            strategy_name = self.extract_strategy_name(id_)
            strategy_worker = Thread(
                target=self.track_strategy_worker,
                args=[id_, strategy_name],
                kwargs={"interval": track_interval},
            )
            strategy_worker.start()
            workers.append(strategy_worker)
            logger.info("开始跟踪策略: %s", strategy_name)
        for worker in workers:
            worker.join()

    def extract_strategy_name(self, run_id: str) -> str:
        ret_json = self.client.get_positions(run_id)
        if ret_json["code"] != 200:
            logger.error(
                "fetch data from run_id %s fail, msg %s",
                run_id,
                ret_json["msg"],
            )
            raise RuntimeError(ret_json["msg"])
        return ret_json["resp"]["name"]

    def extract_day_trades(self, run_id: str) -> List[Dict[str, Any]]:
        ret_json = self.client.get_day_trades(run_id)
        if ret_json["code"] != 200:
            logger.error(
                "fetch day trades from run_id %s fail, msg %s",
                run_id,
                ret_json["msg"],
            )
            raise RuntimeError(ret_json["msg"])
        return ret_json["resp"]["trades"]

    def query_strategy_transaction(self, strategy: str, **kwargs: Any) -> List[Dict[str, Any]]:
        transactions = self.extract_day_trades(strategy)
        transactions = self.project_transactions(transactions, **kwargs)
        return self.order_transactions_sell_first(transactions)

    @staticmethod
    def stock_shuffle_to_prefix(stock: str) -> str:
        assert (
            len(stock) == 11
        ), "stock {} must like 123456.XSHG or 123456.XSHE".format(stock)
        code = stock[:6]
        if stock.find("XSHG") != -1:
            return "sh" + code
        if stock.find("XSHE") != -1:
            return "sz" + code
        raise TypeError("not valid stock code: {}".format(code))

    def project_transactions(self, transactions: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        new_transactions: List[Dict[str, Any]] = []
        for transaction in transactions:
            new_transaction: Dict[str, Any] = {}
            new_transaction["price"] = transaction["price"]
            new_transaction["amount"] = int(abs(transaction["quantity"]))
            new_transaction["datetime"] = datetime.strptime(
                transaction["time"], "%Y-%m-%d %H:%M:%S"
            )
            new_transaction["stock_code"] = self.stock_shuffle_to_prefix(
                transaction["order_book_id"]
            )
            new_transaction["action"] = (
                "buy" if transaction["quantity"] > 0 else "sell"
            )
            new_transactions.append(new_transaction)

        return new_transactions
```