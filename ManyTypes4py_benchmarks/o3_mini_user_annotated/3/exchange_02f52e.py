from __future__ import annotations
import asyncio
import signal
from collections.abc import Generator
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import floor, isnan
from typing import Any, Optional, Union

import ccxt
import ccxt.pro as ccxt_pro
import pandas as pd
from dateutil import parser

# Assume DEFAULT_TRADES_COLUMNS is defined somewhere and imported as needed.

class Exchange:
    # … (other parts of the class may be here)

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> Generator[str, None, None]:
        # implementation
        ...

    def get_quote_currencies(self) -> list[str]:
        markets: dict[str, Any] = self.markets
        return sorted({market["quote"] for market in markets.values()})

    def get_pair_quote_currency(self, pair: str) -> str:
        markets: dict[str, Any] = self.markets
        return markets.get(pair, {}).get("quote", "")

    def get_pair_base_currency(self, pair: str) -> str:
        markets: dict[str, Any] = self.markets
        return markets.get(pair, {}).get("base", "")

    def market_is_future(self, market: dict[str, Any]) -> bool:
        return bool(market.get("future", False))

    def market_is_spot(self, market: dict[str, Any]) -> bool:
        return bool(market.get("spot", False))

    def market_is_margin(self, market: dict[str, Any]) -> bool:
        return bool(market.get("margin", False))

    def market_is_tradable(self, market: dict[str, Any]) -> bool:
        # Your existing code logic here.
        ...

    def klines(self, pair_interval: tuple[str, str, Any], copy: bool = True) -> pd.DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return pd.DataFrame()

    def trades(self, pair_interval: tuple[str, str, Any], copy: bool = True) -> pd.DataFrame:
        if pair_interval in self._trades:
            return self._trades[pair_interval].copy() if copy else self._trades[pair_interval]
        else:
            return pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS)

    def get_contract_size(self, pair: str) -> Optional[float]:
        market: dict[str, Any] = self.markets.get(pair, {})
        return market.get("contractSize")

    def _trades_contracts_to_amount(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Implementation details…
        return trades

    def _order_contracts_to_amount(self, order: dict[str, Any]) -> dict[str, Any]:
        # Implementation details…
        return order

    def _amount_to_contracts(self, pair: str, amount: float) -> float:
        contract_size: Optional[float] = self.get_contract_size(pair)
        if contract_size is None:
            return amount
        return amount * contract_size

    def _contracts_to_amount(self, pair: str, contracts: float) -> float:
        contract_size: Optional[float] = self.get_contract_size(pair)
        if contract_size is None or contract_size == 0:
            return contracts
        return contracts / contract_size

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        # Assume amount_to_contract_precision is defined elsewhere
        contract_size: Optional[float] = self.get_contract_size(pair)
        precision = self.get_precision_amount(pair)
        return amount_to_contract_precision(amount, precision, self.precisionMode, contract_size)

    def ws_connection_reset(self) -> None:
        if self._exchange_ws:
            self._exchange_ws.reset_connections()

    async def _api_reload_markets(self, reload: bool = False) -> dict[str, Any]:
        # Async call to reload markets.
        ...

    def _load_async_markets(self, reload: bool = False) -> dict[str, Any]:
        try:
            with self._loop_lock:
                markets: dict[str, Any] = self.loop.run_until_complete(self._api_reload_markets(reload=reload))
            return markets
        except asyncio.TimeoutError as e:
            raise Exception(f"Timeout error: {e}")

    def reload_markets(self, force: bool = False, *, load_leverage_tiers: bool = True) -> None:
        # Code to reload markets
        ...

    def validate_stakecurrency(self, stake_currency: str) -> None:
        if not self._markets:
            raise Exception("Markets not loaded.")
        if stake_currency not in self.get_quote_currencies():
            raise Exception(f"{stake_currency} is not available.")

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> Generator[str, None, None]:
        # Implementation here...
        ...

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        if timeframe and timeframe not in self.timeframes:
            raise Exception(f"Invalid timeframe: {timeframe}")

    def validate_ordertypes(self, order_types: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_stop_ordertypes(self, order_types: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_pricing(self, pricing: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_order_time_in_force(self, order_time_in_force: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_orderflow(self, exchange: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_freqai(self, config: dict[str, Any]) -> None:
        # Implementation here...
        ...

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        candle_limit: int = self.ohlcv_candle_limit(timeframe, self._config["candle_type_def"])
        candle_count: int = startup_candles + 1
        required_candle_call_count: int = int(candle_count / candle_limit + (0 if candle_count % candle_limit == 0 else 1))
        if required_candle_call_count > 1:
            return required_candle_call_count
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(self, trading_mode: Any, margin_mode: Optional[Any]) -> None:
        # Implementation here...
        ...

    def get_option(self, param: str, default: Optional[Any] = None) -> Any:
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        if endpoint in self._ft_has.get("exchange_has_overrides", {}):
            return bool(self._ft_has["exchange_has_overrides"][endpoint])
        return bool(endpoint in self._api_async.has and self._api_async.has[endpoint])

    def features(self, market_type: Literal["spot", "futures"], endpoint: str, attribute: str, default: T) -> T:
        feat: dict[str, Any] = self._api_async.features.get("spot", {}) if market_type == "spot" \
            else self._api_async.features.get("swap", {}).get("linear", {})
        return feat.get(endpoint, {}).get(attribute, default)

    def get_precision_amount(self, pair: str) -> Optional[float]:
        return self.markets.get(pair, {}).get("precision", {}).get("amount")

    def get_precision_price(self, pair: str) -> Optional[float]:
        return self.markets.get(pair, {}).get("precision", {}).get("price")

    def amount_to_precision(self, pair: str, amount: float) -> float:
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float, *, rounding_mode: int = ROUND) -> float:
        return price_to_precision(price, self.get_precision_price(pair), self.precision_mode_price, rounding_mode=rounding_mode)

    def price_get_one_pip(self, pair: str, price: float) -> float:
        precision: float = self.markets[pair]["precision"]["price"]
        if self.precisionMode == TICK_SIZE:
            return precision
        return 1 / (10 ** precision)

    def get_min_pair_stake_amount(self, pair: str, price: float, stoploss: float, leverage: Optional[float] = 1.0) -> Optional[float]:
        return self._get_stake_amount_limit(pair, price, stoploss, "min", leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        max_stake_amount: Optional[float] = self._get_stake_amount_limit(pair, price, 0.0, "max", leverage)
        if max_stake_amount is None:
            raise Exception("max_stake_amount cannot be None")
        return max_stake_amount

    def _get_stake_amount_limit(self, pair: str, price: float, stoploss: float, limit: Literal["min", "max"], leverage: Optional[float] = 1.0) -> Optional[float]:
        isMin: bool = (limit == "min")
        market: dict[str, Any] = self.markets.get(pair)
        if market is None:
            raise ValueError(f"Market information for {pair} not found")
        # ... calculation logic ...
        stake_limits: list[float] = []
        # For example, using market limits:
        limits: dict[str, Any] = market["limits"]
        if limits["cost"][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits["cost"][limit]))
        if limits["amount"][limit] is not None:
            stake_limits.append(self._contracts_to_amount(pair, limits["amount"][limit]) * price)
        if not stake_limits:
            return None if isMin else float("inf")
        return self._get_stake_amount_considering_leverage(max(stake_limits) if isMin else min(stake_limits), leverage or 1.0)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        return stake_amount / leverage

    # Dry-run methods

    def create_dry_run_order(
        self,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float,
        leverage: float,
        params: Optional[dict[str, Any]] = None,
        stop_loss: bool = False,
    ) -> dict[str, Any]:
        now: datetime = dt_now()
        order_id: str = f"dry_run_{side}_{pair}_{now.timestamp()}"
        _amount: float = self._contracts_to_amount(pair, self.amount_to_precision(pair, self._amount_to_contracts(pair, amount)))
        dry_order: dict[str, Any] = {
            "id": order_id,
            "symbol": pair,
            "price": rate,
            "average": rate,
            "amount": _amount,
            "cost": _amount * rate,
            "type": ordertype,
            "side": side,
            "filled": 0,
            "remaining": _amount,
            "datetime": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "timestamp": dt_ts(now),
            "status": "open",
            "fee": None,
            "info": {},
        }
        if stop_loss:
            dry_order["info"] = {"stopPrice": dry_order["price"]}
            dry_order[self._ft_has["stop_price_prop"]] = dry_order["price"]
            dry_order["ft_order_type"] = "stoploss"
        orderbook: Optional[dict[str, Any]] = None
        if self.exchange_has("fetchL2OrderBook"):
            orderbook = self.fetch_l2_order_book(pair, 20)
        if ordertype == "limit" and orderbook:
            allowed_diff: float = 0.01
            if self._dry_is_price_crossed(pair, side, rate, orderbook, allowed_diff):
                dry_order["type"] = "market"
        if dry_order["type"] == "market" and not dry_order.get("ft_order_type"):
            average: float = self.get_dry_market_fill_price(pair, side, amount, rate, orderbook)
            dry_order.update({
                "average": average,
                "filled": _amount,
                "remaining": 0.0,
                "status": "closed",
                "cost": (_amount * average),
            })
            dry_order = self.add_dry_order_fee(pair, dry_order, "taker")
        dry_order = self.check_dry_limit_order_filled(dry_order, immediate=True, orderbook=orderbook)
        self._dry_run_open_orders[dry_order["id"]] = dry_order
        return dry_order

    def add_dry_order_fee(
        self,
        pair: str,
        dry_order: dict[str, Any],
        taker_or_maker: str,
    ) -> dict[str, Any]:
        fee: float = self.get_fee(pair, taker_or_maker=taker_or_maker)
        dry_order.update({
            "fee": {
                "currency": self.get_pair_quote_currency(pair),
                "cost": dry_order["cost"] * fee,
                "rate": fee,
            }
        })
        return dry_order

    def get_dry_market_fill_price(
        self, pair: str, side: str, amount: float, rate: float, orderbook: Optional[dict[str, Any]]
    ) -> float:
        if self.exchange_has("fetchL2OrderBook"):
            if orderbook is None:
                orderbook = self.fetch_l2_order_book(pair, 20)
            ob_type: str = "asks" if side == "buy" else "bids"
            slippage: float = 0.05
            max_slippage_val: float = rate * ((1 + slippage) if side == "buy" else (1 - slippage))
            remaining_amount: float = amount
            filled_value: float = 0.0
            book_entry_price: float = 0.0
            for book_entry in orderbook[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume: float = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:
                        filled_value += remaining_amount * book_entry_price
                        break
                    else:
                        filled_value += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:
                filled_value += remaining_amount * book_entry_price
            forecast_avg_filled_price: float = max(filled_value, 0) / amount
            if side == "buy":
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)
            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)
            return self.price_to_precision(pair, forecast_avg_filled_price)
        return rate

    def _dry_is_price_crossed(
        self,
        pair: str,
        side: str,
        limit: float,
        orderbook: Optional[dict[str, Any]] = None,
        offset: float = 0.0,
    ) -> bool:
        if not self.exchange_has("fetchL2OrderBook"):
            return True
        if orderbook is None:
            orderbook = self.fetch_l2_order_book(pair, 1)
        try:
            if side == "buy":
                price = orderbook["asks"][0][0]
                if limit * (1 - offset) >= price:
                    return True
            else:
                price = orderbook["bids"][0][0]
                if limit * (1 + offset) <= price:
                    return True
        except IndexError:
            pass
        return False

    def check_dry_limit_order_filled(
        self, order: dict[str, Any], immediate: bool = False, orderbook: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        if order["status"] != "closed" and order["type"] in ["limit"] and not order.get("ft_order_type"):
            pair: str = order["symbol"]
            if self._dry_is_price_crossed(pair, order["side"], order["price"], orderbook):
                order.update({
                    "status": "closed",
                    "filled": order["amount"],
                    "remaining": 0,
                })
                self.add_dry_order_fee(pair, order, "taker" if immediate else "maker")
        return order

    def fetch_dry_run_order(self, order_id: str) -> dict[str, Any]:
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            from freqtrade.persistence import Order
            order_obj = Order.order_by_id(order_id)
            if order_obj:
                ccxt_order = order_obj.to_ccxt_object(self._ft_has["stop_price_prop"])
                self._dry_run_open_orders[order_id] = ccxt_order
                return ccxt_order
            raise Exception(f"Invalid dry-run order (id: {order_id}). {e}") from e

    def _lev_prep(self, pair: Optional[str], leverage: float, side: str, accept_fail: bool = False) -> None:
        if self.trading_mode != "SPOT":
            self.set_margin_mode(pair, self.margin_mode, accept_fail)
            self._set_leverage(leverage, pair, accept_fail)

    def _get_params(self, side: str, ordertype: str, leverage: float, reduceOnly: bool, time_in_force: str = "GTC") -> dict[str, Any]:
        params: dict[str, Any] = self._params.copy()
        if time_in_force != "GTC" and ordertype != "market":
            params.update({"timeInForce": time_in_force.upper()})
        if reduceOnly:
            params.update({"reduceOnly": True})
        return params

    def _order_needs_price(self, side: str, ordertype: str) -> bool:
        return ordertype != "market" or (side == "buy" and self._api.options.get("createMarketBuyOrderRequiresPrice", False)) or self._ft_has.get("marketOrderRequiresPrice", False)

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float,
        leverage: float,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        if self._config["dry_run"]:
            return self.create_dry_run_order(pair, ordertype, side, amount, self.price_to_precision(pair, rate), leverage)
        params: dict[str, Any] = self._get_params(side, ordertype, leverage, reduceOnly, time_in_force)
        try:
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            needs_price: bool = self._order_needs_price(side, ordertype)
            rate_for_order: Optional[float] = self.price_to_precision(pair, rate) if needs_price else None
            if not reduceOnly:
                self._lev_prep(pair, leverage, side)
            order: dict[str, Any] = self._api.create_order(pair, ordertype, side, amount, rate_for_order, params)
            if order.get("status") is None:
                order["status"] = "open"
            if order.get("type") is None:
                order["type"] = ordertype
            self._log_exchange_response("create_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InsufficientFunds as e:
            raise Exception(f"Insufficient funds: {e}") from e
        except ccxt.InvalidOrder as e:
            raise Exception(f"Invalid order: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS Protection: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error: {e}") from e

    def stoploss_adjust(self, stop_loss: float, order: dict[str, Any], side: str) -> bool:
        if not self._ft_has.get("stoploss_on_exchange"):
            raise Exception(f"stoploss is not implemented for {self.name}.")
        price_param: str = self._ft_has["stop_price_prop"]
        return order.get(price_param) is None or ((side == "sell" and stop_loss > float(order[price_param])) or (side == "buy" and stop_loss < float(order[price_param])))

    def _get_stop_order_type(self, user_order_type: str) -> tuple[str, str]:
        available_order_Types: dict[str, str] = self._ft_has["stoploss_order_types"]
        if user_order_type in available_order_Types:
            ordertype = available_order_Types[user_order_type]
        else:
            ordertype = list(available_order_Types.values())[0]
            user_order_type = list(available_order_Types.keys())[0]
        return ordertype, user_order_type

    def _get_stop_limit_rate(self, stop_price: float, order_types: dict[str, Any], side: str) -> float:
        limit_price_pct: float = order_types.get("stoploss_on_exchange_limit_ratio", 0.99)
        limit_rate: float = stop_price * limit_price_pct if side == "sell" else stop_price * (2 - limit_price_pct)
        bad_stop_price: bool = (stop_price < limit_rate) if side == "sell" else (stop_price > limit_rate)
        if bad_stop_price:
            raise Exception(f"Stop price {stop_price} is less than limit price {limit_rate}.")
        return limit_rate

    def _get_stop_params(self, side: str, ordertype: str, stop_price: float) -> dict[str, Any]:
        params: dict[str, Any] = self._params.copy()
        params.update({self._ft_has["stop_price_param"]: stop_price})
        return params

    def create_stoploss(
        self,
        pair: str,
        amount: float,
        stop_price: float,
        order_types: dict[str, Any],
        side: str,
        leverage: float,
    ) -> dict[str, Any]:
        if not self._ft_has["stoploss_on_exchange"]:
            raise Exception(f"stoploss is not implemented for {self.name}.")
        user_order_type: str = order_types.get("stoploss", "market")
        ordertype, _ = self._get_stop_order_type(user_order_type)
        round_mode: int = ROUND_DOWN if side == "buy" else ROUND_UP
        stop_price_norm: float = self.price_to_precision(pair, stop_price, rounding_mode=round_mode)
        limit_rate: Optional[float] = None
        if user_order_type == "limit":
            limit_rate = self._get_stop_limit_rate(stop_price, order_types, side)
            limit_rate = self.price_to_precision(pair, limit_rate, rounding_mode=round_mode)
        if self._config["dry_run"]:
            return self.create_dry_run_order(pair, ordertype, side, amount, stop_price_norm, leverage, stop_loss=True)
        try:
            params: dict[str, Any] = self._get_stop_params(side, ordertype, stop_price_norm)
            if self.trading_mode == "FUTURES":
                params["reduceOnly"] = True
                if "stoploss_price_type" in order_types and "stop_price_type_field" in self._ft_has:
                    price_type = self._ft_has["stop_price_type_value_mapping"][order_types.get("stoploss_price_type", "LAST")]
                    params[self._ft_has["stop_price_type_field"]] = price_type
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            self._lev_prep(pair, leverage, side, accept_fail=True)
            order: dict[str, Any] = self._api.create_order(symbol=pair, type=ordertype, side=side, amount=amount, price=limit_rate, params=params)
            self._log_exchange_response("create_stoploss_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InsufficientFunds as e:
            raise Exception(f"Insufficient funds for stoploss order: {e}") from e
        except (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationRejected) as e:
            raise Exception(f"Invalid stoploss order: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in stoploss: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in stoploss: {e}") from e

    def fetch_order_emulated(self, order_id: str, pair: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            order = self._api.fetch_open_order(order_id, pair, params=params)
            self._log_exchange_response("fetch_open_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound:
            try:
                order = self._api.fetch_closed_order(order_id, pair, params=params)
                self._log_exchange_response("fetch_closed_order", order)
                order = self._order_contracts_to_amount(order)
                return order
            except ccxt.OrderNotFound as e:
                raise Exception(f"Order not found: {e}") from e
        except ccxt.InvalidOrder as e:
            raise Exception(f"Invalid order fetched: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_order_emulated: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_order_emulated: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_order_emulated: {e}") from e

    def fetch_order(self, order_id: str, pair: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)
        if params is None:
            params = {}
        try:
            if not self.exchange_has("fetchOrder"):
                return self.fetch_order_emulated(order_id, pair, params)
            order: dict[str, Any] = self._api.fetch_order(order_id, pair, params=params)
            self._log_exchange_response("fetch_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound as e:
            raise Exception(f"Order not found: {e}") from e
        except ccxt.InvalidOrder as e:
            raise Exception(f"Invalid order: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_order: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_order: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_order: {e}") from e

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self.fetch_order(order_id, pair, params)

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str, stoploss_order: bool = False) -> dict[str, Any]:
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: dict[str, Any]) -> bool:
        return order.get("status") in ("canceled", "closed") and order.get("filled") == 0.0

    def cancel_order(self, order_id: str, pair: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if self._config["dry_run"]:
            try:
                order = self.fetch_dry_run_order(order_id)
                order.update({"status": "canceled", "filled": 0.0, "remaining": order["amount"]})
                return order
            except Exception:
                return {}
        if params is None:
            params = {}
        try:
            order: dict[str, Any] = self._api.cancel_order(order_id, pair, params=params)
            self._log_exchange_response("cancel_order", order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InvalidOrder as e:
            raise Exception(f"Invalid cancel order: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in cancel_order: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in cancel_order: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in cancel_order: {e}") from e

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self.cancel_order(order_id, pair, params)

    def is_cancel_order_result_suitable(self, corder: Any) -> bool:
        if not isinstance(corder, dict):
            return False
        required = ("fee", "status", "amount")
        return all(corder.get(k) is not None for k in required)

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> dict[str, Any]:
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except Exception:
            pass
        try:
            order = self.fetch_order(order_id, pair)
        except Exception:
            order = {"id": order_id, "status": "canceled", "amount": amount, "filled": 0.0, "fee": {}, "info": {}}
        return order

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> dict[str, Any]:
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except Exception:
            order = {"id": order_id, "fee": {}, "status": "canceled", "amount": amount, "info": {}}
        return order

    def get_balances(self) -> dict[str, Any]:
        try:
            balances: dict[str, Any] = self._api.fetch_balance()
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)
            self._log_exchange_response("fetch_balances", balances)
            return balances
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_balances: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_balances: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_balances: {e}") from e

    def fetch_positions(self, pair: Optional[str] = None) -> list[dict[str, Any]]:
        if self._config["dry_run"] or self.trading_mode != "FUTURES":
            return []
        try:
            symbols: list[str] = [pair] if pair else []
            positions: list[dict[str, Any]] = self._api.fetch_positions(symbols)
            self._log_exchange_response("fetch_positions", positions)
            return positions
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_positions: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_positions: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_positions: {e}") from e

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        if self.exchange_has("fetchClosedOrders"):
            orders = self._api.fetch_closed_orders(pair, since=since_ms)
            if self.exchange_has("fetchOpenOrders"):
                orders_open: list[dict[str, Any]] = self._api.fetch_open_orders(pair, since=since_ms)
                orders.extend(orders_open)
        return orders

    def fetch_orders(self, pair: str, since: datetime, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if self._config["dry_run"]:
            return []
        try:
            since_ms: int = int((since.timestamp() - 10) * 1000)
            if self.exchange_has("fetchOrders"):
                if params is None:
                    params = {}
                try:
                    orders: list[dict[str, Any]] = self._api.fetch_orders(pair, since=since_ms, params=params)
                except ccxt.NotSupported:
                    orders = self._fetch_orders_emulate(pair, since_ms)
            else:
                orders = self._fetch_orders_emulate(pair, since_ms)
            self._log_exchange_response("fetch_orders", orders)
            orders = [self._order_contracts_to_amount(o) for o in orders]
            return orders
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_orders: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_orders: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_orders: {e}") from e

    def fetch_trading_fees(self) -> dict[str, Any]:
        if self._config["dry_run"] or self.trading_mode != "FUTURES" or not self.exchange_has("fetchTradingFees"):
            return {}
        try:
            trading_fees: dict[str, Any] = self._api.fetch_trading_fees()
            self._log_exchange_response("fetch_trading_fees", trading_fees)
            return trading_fees
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_trading_fees: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_trading_fees: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_trading_fees: {e}") from e

    def fetch_bids_asks(self, symbols: Optional[list[str]] = None, *, cached: bool = False) -> dict[str, Any]:
        if not self.exchange_has("fetchBidsAsks"):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get("fetch_bids_asks")
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_bids_asks(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache["fetch_bids_asks"] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise Exception(f"Exchange does not support fetchBidsAsks: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_bids_asks: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_bids_asks: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_bids_asks: {e}") from e

    def get_tickers(self, symbols: Optional[list[str]] = None, *, cached: bool = False, market_type: Optional[str] = None) -> dict[str, Any]:
        if not self.exchange_has("fetchTickers"):
            return {}
        cache_key: str = f"fetch_tickers_{market_type}" if market_type else "fetch_tickers"
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get(cache_key)
            if tickers:
                return tickers
        try:
            market_types: dict[str, str] = {"FUTURES": "swap"}
            params: dict[str, Any] = {"type": market_types.get(market_type, market_type)} if market_type else {}
            tickers: dict[str, Any] = self._api.fetch_tickers(symbols, params)
            with self._cache_lock:
                self._fetch_tickers_cache[cache_key] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise Exception(f"Exchange does not support fetchTickers: {e}") from e
        except ccxt.BadSymbol as e:
            self.reload_markets(True)
            raise Exception("BadSymbol exception") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_tickers: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_tickers: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_tickers: {e}") from e

    def get_proxy_coin(self) -> str:
        return self._config["stake_currency"]

    def get_conversion_rate(self, coin: str, currency: str) -> Optional[float]:
        if (proxy_coin := self._ft_has["proxy_coin_mapping"].get(coin)) is not None:
            coin = proxy_coin
        if (proxy_currency := self._ft_has["proxy_coin_mapping"].get(currency)) is not None:
            currency = proxy_currency
        if coin == currency:
            return 1.0
        tickers: dict[str, Any] = self.get_tickers(cached=True)
        try:
            for pair in self.get_valid_pair_combination(coin, currency):
                ticker: Optional[dict[str, Any]] = tickers.get(pair)
                if not ticker:
                    tickers_other = self.get_tickers(cached=True, market_type=("SPOT" if self.trading_mode != "SPOT" else "FUTURES"))
                    ticker = tickers_other.get(pair)
                if ticker:
                    rate: Optional[float] = ticker.get("last") or ticker.get("ask")
                    if rate and pair.startswith(currency) and not pair.endswith(currency):
                        rate = 1.0 / rate
                    return rate
        except ValueError:
            return None
        return None

    def fetch_ticker(self, pair: str) -> dict[str, Any]:
        try:
            if pair not in self.markets or not self.markets[pair].get("active", False):
                raise Exception(f"Pair {pair} not available")
            data: dict[str, Any] = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_ticker: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_ticker: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_ticker: {e}") from e

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[list[int]], range_required: bool = True) -> Union[int, None]:
        if not limit_range:
            return limit
        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            return None
        return result

    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict[str, Any]:
        limit1: Union[int, None] = self.get_next_limit_in_list(limit, self._ft_has.get("l2_limit_range"), self._ft_has.get("l2_limit_range_required", True))
        try:
            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise Exception(f"Exchange does not support fetch_l2_order_book: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in fetch_l2_order_book: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in fetch_l2_order_book: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in fetch_l2_order_book: {e}") from e

    def _get_price_side(self, side: str, is_short: bool, conf_strategy: dict[str, Any]) -> str:
        price_side: str = conf_strategy["price_side"]
        if price_side in ("same", "other"):
            price_map = {
                ("entry", "long", "same"): "bid",
                ("entry", "long", "other"): "ask",
                ("entry", "short", "same"): "ask",
                ("entry", "short", "other"): "bid",
                ("exit", "long", "same"): "ask",
                ("exit", "long", "other"): "bid",
                ("exit", "short", "same"): "bid",
                ("exit", "short", "other"): "ask",
            }
            price_side = price_map[(side, "short" if is_short else "long", price_side)]
        return price_side

    def get_rate(self, pair: str, refresh: bool, side: str, is_short: bool, order_book: Optional[dict[str, Any]] = None, ticker: Optional[dict[str, Any]] = None) -> float:
        name: str = side.capitalize()
        strat_name: str = "entry_pricing" if side == "entry" else "exit_pricing"
        cache_rate = self._entry_rate_cache if side == "entry" else self._exit_rate_cache
        if not refresh:
            with self._cache_lock:
                rate_cached = cache_rate.get(pair)
            if rate_cached:
                return rate_cached
        conf_strategy: dict[str, Any] = self._config.get(strat_name, {})
        price_side: str = self._get_price_side(side, is_short, conf_strategy)
        if conf_strategy.get("use_order_book", False):
            order_book_top: int = conf_strategy.get("order_book_top", 1)
            if order_book is None:
                order_book = self.fetch_l2_order_book(pair, order_book_top)
            rate = self._get_rate_from_ob(pair, side, order_book, name, price_side, order_book_top)
        else:
            if ticker is None:
                ticker = self.fetch_ticker(pair)
            rate = self._get_rate_from_ticker(side, ticker, conf_strategy, price_side)
        if rate is None:
            raise Exception(f"{name}-Rate for {pair} was empty.")
        with self._cache_lock:
            cache_rate[pair] = rate
        return rate

    def _get_rate_from_ticker(self, side: str, ticker: dict[str, Any], conf_strategy: dict[str, Any], price_side: str) -> Optional[float]:
        ticker_rate: Optional[float] = ticker.get(price_side)
        if ticker.get("last") and ticker_rate:
            if side == "entry" and ticker_rate > ticker["last"]:
                balance: float = conf_strategy.get("price_last_balance", 0.0)
                ticker_rate = ticker_rate + balance * (ticker["last"] - ticker_rate)
            elif side == "exit" and ticker_rate < ticker["last"]:
                balance: float = conf_strategy.get("price_last_balance", 0.0)
                ticker_rate = ticker_rate - balance * (ticker_rate - ticker["last"])
        return ticker_rate

    def _get_rate_from_ob(self, pair: str, side: str, order_book: dict[str, Any], name: str, price_side: str, order_book_top: int) -> float:
        try:
            obside: str = "bids" if price_side == "bid" else "asks"
            rate = order_book[obside][order_book_top - 1][0]
        except (IndexError, KeyError) as e:
            raise Exception(f"Price from orderbook could not be determined for {pair}") from e
        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> tuple[float, float]:
        entry_rate: Optional[float] = None
        exit_rate: Optional[float] = None
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
        entry_pricing: dict[str, Any] = self._config.get("entry_pricing", {})
        exit_pricing: dict[str, Any] = self._config.get("exit_pricing", {})
        order_book: Optional[dict[str, Any]] = None
        ticker: Optional[dict[str, Any]] = None
        if not entry_rate and entry_pricing.get("use_order_book", False):
            order_book_top: int = max(entry_pricing.get("order_book_top", 1), exit_pricing.get("order_book_top", 1))
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            entry_rate = self.get_rate(pair, refresh, "entry", is_short, order_book=order_book)
        elif not entry_rate:
            ticker = self.fetch_ticker(pair)
            entry_rate = self.get_rate(pair, refresh, "entry", is_short, ticker=ticker)
        if not exit_rate:
            exit_rate = self.get_rate(pair, refresh, "exit", is_short, order_book=order_book, ticker=ticker)
        return entry_rate, exit_rate

    def get_trades_for_order(self, order_id: str, pair: str, since: datetime, params: Optional[dict[str, Any]] = None) -> list[Any]:
        if self._config["dry_run"]:
            return []
        if not self.exchange_has("fetchMyTrades"):
            return []
        try:
            _params: dict[str, Any] = params if params else {}
            my_trades: list[dict[str, Any]] = self._api.fetch_my_trades(pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000), params=_params)
            matched_trades = [trade for trade in my_trades if trade["order"] == order_id]
            self._log_exchange_response("get_trades_for_order", matched_trades)
            matched_trades = self._trades_contracts_to_amount(matched_trades)
            return matched_trades
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_trades_for_order: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_trades_for_order: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_trades_for_order: {e}") from e

    def get_order_id_conditional(self, order: dict[str, Any]) -> str:
        return order["id"]

    def get_fee(self, symbol: str, order_type: str = "", side: str = "", amount: float = 1, price: float = 1, taker_or_maker: str = "maker") -> float:
        if order_type and order_type == "market":
            taker_or_maker = "taker"
        try:
            if self._config["dry_run"] and self._config.get("fee") is not None:
                return self._config["fee"]
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets(params={})
            fee_info = self._api.calculate_fee(symbol=symbol, type=order_type, side=side, amount=amount, price=price, takerOrMaker=taker_or_maker)
            return fee_info["rate"]
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_fee: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_fee: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_fee: {e}") from e

    @staticmethod
    def order_has_fee(order: Any) -> bool:
        if not isinstance(order, dict):
            return False
        return ("fee" in order and order["fee"] is not None and (set(order["fee"].keys()) >= {"currency", "cost"}) and order["fee"]["currency"] is not None and order["fee"]["cost"] is not None)

    def calculate_fee_rate(self, fee: dict[str, Any], symbol: str, cost: float, amount: float) -> Optional[float]:
        if fee.get("rate") is not None:
            return fee.get("rate")
        fee_curr: Optional[str] = fee.get("currency")
        if fee_curr is None:
            return None
        fee_cost: float = float(fee["cost"])
        if fee_curr == self.get_pair_base_currency(symbol):
            return round(fee_cost / amount, 8)
        elif fee_curr == self.get_pair_quote_currency(symbol):
            return round(fee_cost / cost, 8) if cost else None
        else:
            if not cost:
                return None
            try:
                fee_to_quote_rate: Optional[float] = self.get_conversion_rate(fee_curr, self._config["stake_currency"])
                if not fee_to_quote_rate:
                    raise ValueError("Conversion rate not found.")
            except Exception:
                fee_to_quote_rate = self._config["exchange"].get("unknown_fee_rate")
                if not fee_to_quote_rate:
                    return None
            return round((fee_cost * fee_to_quote_rate) / cost, 8)

    def extract_cost_curr_rate(self, fee: dict[str, Any], symbol: str, cost: float, amount: float) -> tuple[float, str, Optional[float]]:
        return (float(fee["cost"]), fee["currency"], self.calculate_fee_rate(fee, symbol, cost, amount))

    def get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: Any,
        is_new_pair: bool = False,
        until_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        with self._loop_lock:
            pair_returned, _, _, data, _ = self.loop.run_until_complete(
                self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe, since_ms=since_ms, until_ms=until_ms, candle_type=candle_type)
            )
        return ohlcv_to_dataframe(data, timeframe, pair, fill_missing=False, drop_incomplete=True)

    async def _async_get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: Any,
        raise_: bool = False,
        until_ms: Optional[int] = None,
    ) -> tuple[str, str, Any, list[Any], bool]:
        one_call: int = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
        input_coroutines = [self._async_get_candle_history(pair, timeframe, candle_type, since) for since in range(since_ms, until_ms or dt_ts(), one_call)]
        data: list = []
        for input_coro in chunks(input_coroutines, 100):
            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    if raise_:
                        raise res
                    continue
                else:
                    p, _, c, new_data, _ = res
                    if p == pair and c == candle_type:
                        data.extend(new_data)
        data = sorted(data, key=lambda x: x[0])
        return pair, timeframe, candle_type, data, self._ohlcv_partial_candle

    def _build_coroutine(
        self,
        pair: str,
        timeframe: str,
        candle_type: Any,
        since_ms: Optional[int],
        cache: bool,
    ) -> asyncio.coroutines.Coroutine[Any, Any, tuple[str, str, Any, list[Any], bool]]:
        not_all_data = cache and self.required_candle_call_count > 1
        if cache and candle_type in ("SPOT", "FUTURES"):
            if self._has_watch_ohlcv and self._exchange_ws:
                self._exchange_ws.schedule_ohlcv(pair, timeframe, candle_type)
        if cache and (pair, timeframe, candle_type) in self._klines:
            candle_limit: int = self.ohlcv_candle_limit(timeframe, candle_type)
            min_ts: int = dt_ts(date_minus_candles(timeframe, candle_limit - 5))
            if self._exchange_ws:
                candle_ts: int = dt_ts(timeframe_to_prev_date(timeframe))
                prev_candle_ts: int = dt_ts(date_minus_candles(timeframe, 1))
                candles = self._exchange_ws.ohlcvs(pair, timeframe)
                half_candle: int = int(candle_ts - (candle_ts - prev_candle_ts) * 0.5)
                last_refresh_time: int = int(self._exchange_ws.klines_last_refresh.get((pair, timeframe, candle_type), 0))
                if candles and candles[-1][0] >= prev_candle_ts and last_refresh_time >= half_candle:
                    return self._exchange_ws.get_ohlcv(pair, timeframe, candle_type, candle_ts)
                logger.info("Failed to reuse watch for %s", pair)
            if min_ts < self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0):
                not_all_data = False
            else:
                del self._klines[(pair, timeframe, candle_type)]
        if not since_ms and (self._ft_has.get("ohlcv_require_since") or not_all_data):
            one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
            move_to: int = one_call * self.required_candle_call_count
            now = timeframe_to_next_date(timeframe)
            since_ms = dt_ts(now - timedelta(seconds=move_to // 1000))
        if since_ms:
            return self._async_get_historic_ohlcv(pair, timeframe, since_ms=since_ms, raise_=True, candle_type=candle_type)
        else:
            return self._async_get_candle_history(pair, timeframe, candle_type, since_ms)

    def _build_ohlcv_dl_jobs(self, pair_list: list[tuple[str, str, Any]], since_ms: Optional[int], cache: bool) -> tuple[list[asyncio.coroutines.Coroutine[Any, Any, tuple[str, str, Any, list[Any], bool]]], list[tuple[str, str, Any]]]:
        input_coroutines: list = []
        cached_pairs: list[tuple[str, str, Any]] = []
        for pair, timeframe, candle_type in set(pair_list):
            if timeframe not in self.timeframes and candle_type in ("SPOT", "FUTURES"):
                logger.warning("Cannot download (%s, %s) because timeframe not available.", pair, timeframe)
                continue
            if ((pair, timeframe, candle_type) not in self._klines) or (not cache) or self._now_is_time_to_refresh(pair, timeframe, candle_type):
                input_coroutines.append(self._build_coroutine(pair, timeframe, candle_type, since_ms, cache))
            else:
                cached_pairs.append((pair, timeframe, candle_type))
        return input_coroutines, cached_pairs

    def _process_ohlcv_df(self, pair: str, timeframe: str, c_type: Any, ticks: list[list], cache: bool, drop_incomplete: bool) -> pd.DataFrame:
        if ticks and cache:
            idx: int = -2 if drop_incomplete and len(ticks) > 1 else -1
            self._pairs_last_refresh_time[(pair, timeframe, c_type)] = ticks[idx][0]
        ohlcv_df: pd.DataFrame = ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True, drop_incomplete=drop_incomplete)
        if cache:
            if (pair, timeframe, c_type) in self._klines:
                old = self._klines[(pair, timeframe, c_type)]
                ohlcv_df = clean_ohlcv_dataframe(pd.concat([old, ohlcv_df], axis=0), timeframe, pair, fill_missing=True, drop_incomplete=False)
                candle_limit: int = self.ohlcv_candle_limit(timeframe, self._config["candle_type_def"])
                ohlcv_df = ohlcv_df.tail(candle_limit + self._startup_candle_count)
                ohlcv_df = ohlcv_df.reset_index(drop=True)
                self._klines[(pair, timeframe, c_type)] = ohlcv_df
            else:
                self._klines[(pair, timeframe, c_type)] = ohlcv_df
        return ohlcv_df

    def refresh_latest_ohlcv(self, pair_list: list[tuple[str, str, Any]], *, since_ms: Optional[int] = None, cache: bool = True, drop_incomplete: Optional[bool] = None) -> dict[tuple[str, str, Any], pd.DataFrame]:
        logger.debug("Refreshing OHLCV for %d pairs", len(pair_list))
        ohlcv_dl_jobs, cached_pairs = self._build_ohlcv_dl_jobs(pair_list, since_ms, cache)
        results_df: dict[tuple[str, str, Any], pd.DataFrame] = {}
        for dl_jobs_batch in chunks(ohlcv_dl_jobs, 100):
            async def gather_coroutines(coro: list) -> list:
                return await asyncio.gather(*coro, return_exceptions=True)
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_jobs_batch))
            for res in results:
                if isinstance(res, Exception):
                    continue
                pair, timeframe, c_type, ticks, drop_hint = res
                drop_incomplete_ = drop_hint if drop_incomplete is None else drop_incomplete
                ohlcv_df = self._process_ohlcv_df(pair, timeframe, c_type, ticks, cache, drop_incomplete_)
                results_df[(pair, timeframe, c_type)] = ohlcv_df
        for pair, timeframe, c_type in cached_pairs:
            results_df[(pair, timeframe, c_type)] = self.klines((pair, timeframe, c_type), copy=False)
        return results_df

    def refresh_ohlcv_with_cache(self, pairs: list[tuple[str, str, Any]], since_ms: int) -> dict[tuple[str, str, Any], pd.DataFrame]:
        timeframes = {p[1] for p in pairs}
        for timeframe in timeframes:
            if (timeframe, since_ms) not in self._expiring_candle_cache:
                timeframe_in_sec: int = timeframe_to_seconds(timeframe)
                self._expiring_candle_cache[(timeframe, since_ms)] = PeriodicCache(ttl=timeframe_in_sec, maxsize=1000)
        candles: dict[tuple[str, str, Any], Optional[pd.DataFrame]] = {c: self._expiring_candle_cache[(c[1], since_ms)].get(c) for c in pairs if c in self._expiring_candle_cache[(c[1], since_ms)]}
        pairs_to_download = [p for p in pairs if p not in candles]
        if pairs_to_download:
            new_candles = self.refresh_latest_ohlcv(pairs_to_download, since_ms=since_ms, cache=False)
            for c, val in new_candles.items():
                self._expiring_candle_cache[(c[1], since_ms)][c] = val
                candles[c] = val
        return candles

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: Any) -> bool:
        interval_in_sec: int = timeframe_to_msecs(timeframe)
        plr: int = self._pairs_last_refresh_time.get((pair, timeframe, candle_type), 0) + interval_in_sec
        now: int = dt_ts(timeframe_to_prev_date(timeframe))
        return plr < now

    async def _async_get_candle_history(self, pair: str, timeframe: str, candle_type: Any, since_ms: Optional[int]) -> tuple[str, str, Any, list[Any], bool]:
        try:
            s: str = "(" + dt_from_ts(since_ms).isoformat() + ") " if since_ms is not None else ""
            params: dict[str, Any] = deepcopy(self._ft_has.get("ohlcv_params", {}))
            candle_limit: int = self.ohlcv_candle_limit(timeframe, candle_type, since_ms)
            if candle_type and candle_type != "FUNDING_RATE":
                data = await self._api_async.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=candle_limit, params=params)
            else:
                data = await self._fetch_funding_rate_history(pair=pair, timeframe=timeframe, limit=candle_limit, since_ms=since_ms)
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                return pair, timeframe, candle_type, [], self._ohlcv_partial_candle
            return pair, timeframe, candle_type, data, self._ohlcv_partial_candle
        except ccxt.NotSupported as e:
            raise Exception(f"Exchange does not support fetching OHLCV: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in _async_get_candle_history: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in _async_get_candle_history: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in _async_get_candle_history: {e}") from e

    async def _fetch_funding_rate_history(self, pair: str, timeframe: str, limit: int, since_ms: Optional[int]) -> list[list]:
        data = await self._api_async.fetch_funding_rate_history(pair, since=since_ms, limit=limit)
        data = [[x["timestamp"], x["fundingRate"], 0, 0, 0, 0] for x in data]
        return data

    def needed_candle_for_trades_ms(self, timeframe: str, candle_type: Any) -> int:
        candle_limit: int = self.ohlcv_candle_limit(timeframe, candle_type)
        tf_s: int = timeframe_to_seconds(timeframe)
        candles_fetched: int = candle_limit * self.required_candle_call_count
        max_candles: int = self._config["orderflow"]["max_candles"]
        required_candles: int = min(max_candles, candles_fetched)
        move_to: int = tf_s * candle_limit * required_candles if required_candles > candle_limit else (max_candles + 1) * tf_s
        now = timeframe_to_next_date(timeframe)
        return int((now - timedelta(seconds=move_to)).timestamp() * 1000)

    def _process_trades_df(self, pair: str, timeframe: str, c_type: Any, ticks: list[list], cache: bool, first_required_candle_date: int) -> pd.DataFrame:
        trades_df: pd.DataFrame = trades_list_to_df(ticks, True)
        if cache:
            if (pair, timeframe, c_type) in self._trades:
                old = self._trades[(pair, timeframe, c_type)]
                combined_df = pd.concat([old, trades_df], axis=0)
                trades_df = pd.DataFrame(trades_df_remove_duplicates(combined_df), columns=combined_df.columns)
                trades_df = trades_df[first_required_candle_date < trades_df["timestamp"]]
                trades_df = trades_df.reset_index(drop=True)
            self._trades[(pair, timeframe, c_type)] = trades_df
        return trades_df

    async def _build_trades_dl_jobs(self, pairwt: tuple[str, str, Any], data_handler: Any, cache: bool) -> tuple[tuple[str, str, Any], Optional[pd.DataFrame]]:
        pair, timeframe, candle_type = pairwt
        since_ms: Optional[int] = None
        new_ticks: list = []
        all_stored_ticks_df: pd.DataFrame = pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS + ["date"])
        first_candle_ms: int = self.needed_candle_for_trades_ms(timeframe, candle_type)
        is_in_cache: bool = (pair, timeframe, candle_type) in self._trades
        if (not is_in_cache) or (not cache) or self._now_is_time_to_refresh_trades(pair, timeframe, candle_type):
            try:
                until: int = None
                from_id: Optional[str] = None
                if is_in_cache:
                    from_id = self._trades[(pair, timeframe, candle_type)].iloc[-1]["id"]
                    until = dt_ts()
                else:
                    until = int(timeframe_to_prev_date(timeframe).timestamp()) * 1000
                    all_stored_ticks_df = data_handler.trades_load(f"{pair}-cached", self.trading_mode)
                    if not all_stored_ticks_df.empty:
                        if all_stored_ticks_df.iloc[-1]["timestamp"] > first_candle_ms and all_stored_ticks_df.iloc[0]["timestamp"] <= first_candle_ms:
                            last_cached_ms = all_stored_ticks_df.iloc[-1]["timestamp"]
                            from_id = all_stored_ticks_df.iloc[-1]["id"]
                            since_ms = last_cached_ms if last_cached_ms > first_candle_ms else first_candle_ms
                        else:
                            all_stored_ticks_df = pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS + ["date"])
                [_, new_ticks] = await self._async_get_trade_history(pair, since=since_ms if since_ms else first_candle_ms, until=until, from_id=from_id)
            except Exception:
                return pairwt, None
            if new_ticks:
                all_stored_ticks_list = all_stored_ticks_df[DEFAULT_TRADES_COLUMNS].values.tolist()
                all_stored_ticks_list.extend(new_ticks)
                trades_df = self._process_trades_df(pair, timeframe, candle_type, all_stored_ticks_list, cache, first_required_candle_date=first_candle_ms)
                data_handler.trades_store(f"{pair}-cached", trades_df[DEFAULT_TRADES_COLUMNS], self.trading_mode)
                return pairwt, trades_df
            else:
                logger.error("No new ticks for %s", pair)
        return pairwt, None

    def refresh_latest_trades(self, pair_list: list[tuple[str, str, Any]], *, cache: bool = True) -> dict[tuple[str, str, Any], pd.DataFrame]:
        from freqtrade.data.history import get_datahandler
        data_handler = get_datahandler(self._config["datadir"], data_format=self._config["dataformat_trades"])
        results_df: dict[tuple[str, str, Any], pd.DataFrame] = {}
        trades_dl_jobs = [self._build_trades_dl_jobs(pairwt, data_handler, cache) for pairwt in set(pair_list)]
        async def gather_coroutines(coro: list) -> list:
            return await asyncio.gather(*coro, return_exceptions=True)
        for dl_job_chunk in chunks(trades_dl_jobs, 100):
            with self._loop_lock:
                results = self.loop.run_until_complete(gather_coroutines(dl_job_chunk))
            for res in results:
                if isinstance(res, Exception):
                    continue
                pairwt, trades_df = res
                if trades_df is not None:
                    results_df[pairwt] = trades_df
        return results_df

    def _now_is_time_to_refresh_trades(self, pair: str, timeframe: str, candle_type: Any) -> bool:
        trades_df: pd.DataFrame = self.trades((pair, timeframe, candle_type), False)
        pair_last_refreshed: int = int(trades_df.iloc[-1]["timestamp"])
        full_candle: int = int(timeframe_to_next_date(timeframe, dt_from_ts(pair_last_refreshed)).timestamp()) * 1000
        now: int = dt_ts()
        return full_candle <= now

    async def _async_fetch_trades(self, pair: str, since: Optional[int] = None, params: Optional[dict[str, Any]] = None) -> tuple[list[list], Any]:
        try:
            trades_limit: int = self._max_trades_limit
            if params:
                trades = await self._api_async.fetch_trades(pair, params=params, limit=trades_limit)
            else:
                trades = await self._api_async.fetch_trades(pair, since=since, limit=trades_limit)
            trades = self._trades_contracts_to_amount(trades)
            pagination_value = self._get_trade_pagination_next_value(trades)
            return trades_dict_to_list(trades), pagination_value
        except ccxt.NotSupported as e:
            raise Exception(f"Exchange does not support fetching trades: {e}") from e
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in _async_fetch_trades: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in _async_fetch_trades: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in _async_fetch_trades: {e}") from e

    def _valid_trade_pagination_id(self, pair: str, from_id: str) -> bool:
        return True

    def _get_trade_pagination_next_value(self, trades: list[dict]) -> Any:
        if not trades:
            return None
        if self._trades_pagination == "id":
            return trades[-1].get("id")
        else:
            return trades[-1].get("timestamp")

    async def _async_get_trade_history_id_startup(self, pair: str, since: int) -> tuple[list[list], str]:
        return await self._async_fetch_trades(pair, since=since)

    async def _async_get_trade_history_id(self, pair: str, *, until: int, since: int, from_id: Optional[str] = None) -> tuple[str, list[list]]:
        trades: list[list] = []
        has_overlap: bool = self._ft_has.get("trades_pagination_overlap", True)
        x = slice(None, -1) if has_overlap else slice(None)
        if not from_id or not self._valid_trade_pagination_id(pair, from_id):
            t, from_id = await self._async_get_trade_history_id_startup(pair, since=since)
            trades.extend(t[x])
        while True:
            try:
                t, from_id_next = await self._async_fetch_trades(pair, params={self._trades_pagination_arg: from_id})
                if t:
                    trades.extend(t[x])
                    if from_id == from_id_next or t[-1][0] > until:
                        if has_overlap:
                            trades.extend(t[-1:])
                        break
                    from_id = from_id_next
                else:
                    break
            except asyncio.CancelledError:
                break
        return pair, trades

    async def _async_get_trade_history_time(self, pair: str, until: int, since: int) -> tuple[str, list[list]]:
        trades: list[list] = []
        while True:
            try:
                t, since_next = await self._async_fetch_trades(pair, since=since)
                if t:
                    if since == since_next and len(t) == 1:
                        break
                    since = since_next
                    trades.extend(t)
                    if until and since_next > until:
                        break
                else:
                    break
            except asyncio.CancelledError:
                break
        return pair, trades

    async def _async_get_trade_history(self, pair: str, *, since: int, until: Optional[int] = None, from_id: Optional[str] = None) -> tuple[str, list[list]]:
        if until is None:
            until = ccxt.Exchange.milliseconds()
        if self._trades_pagination == "time":
            return await self._async_get_trade_history_time(pair=pair, since=since, until=until)
        elif self._trades_pagination == "id":
            return await self._async_get_trade_history_id(pair=pair, since=since, until=until, from_id=from_id)
        else:
            raise Exception(f"Exchange {self.name} does not use supported pagination")

    def get_historic_trades(self, pair: str, since: int, until: Optional[int] = None, from_id: Optional[str] = None) -> tuple[str, list]:
        if not self.exchange_has("fetchTrades"):
            raise Exception("This exchange does not support fetching Trades.")
        with self._loop_lock:
            task = asyncio.ensure_future(self._async_get_trade_history(pair=pair, since=since, until=until, from_id=from_id))
            for sig in [signal.SIGINT, signal.SIGTERM]:
                try:
                    self.loop.add_signal_handler(sig, task.cancel)
                except NotImplementedError:
                    pass
            return self.loop.run_until_complete(task)

    def _get_funding_fees_from_exchange(self, pair: str, since: Union[datetime, int]) -> float:
        if not self.exchange_has("fetchFundingHistory"):
            raise Exception(f"fetchFundingHistory() is not available for {self.name}")
        if isinstance(since, datetime):
            since = dt_ts(since)
        try:
            funding_history = self._api.fetch_funding_history(symbol=pair, since=since)
            self._log_exchange_response("funding_history", funding_history, add_info=f"pair: {pair}, since: {since}")
            return sum(fee["amount"] for fee in funding_history)
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in _get_funding_fees_from_exchange: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in _get_funding_fees_from_exchange: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in _get_funding_fees_from_exchange: {e}") from e

    def get_leverage_tiers(self) -> dict[str, list[dict]]:
        try:
            return self._api.fetch_leverage_tiers()
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_leverage_tiers: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_leverage_tiers: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_leverage_tiers: {e}") from e

    async def get_market_leverage_tiers(self, symbol: str) -> tuple[str, list[dict]]:
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return symbol, tier
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in get_market_leverage_tiers: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in get_market_leverage_tiers: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in get_market_leverage_tiers: {e}") from e

    def load_leverage_tiers(self) -> dict[str, list[dict]]:
        if self.trading_mode == "FUTURES":
            if self.exchange_has("fetchLeverageTiers"):
                return self.get_leverage_tiers()
            elif self.exchange_has("fetchMarketLeverageTiers"):
                markets = self.markets
                symbols = [symbol for symbol, market in markets.items() if self.market_is_future(market) and market["quote"] == self._config["stake_currency"]]
                tiers: dict[str, list[dict]] = {}
                tiers_cached = self.load_cached_leverage_tiers(self._config["stake_currency"])
                if tiers_cached:
                    tiers = tiers_cached
                coros = [self.get_market_leverage_tiers(symbol) for symbol in sorted(symbols) if symbol not in tiers]
                if coros:
                    logger.info(f"Initializing leverage_tiers for {len(symbols)} markets. This may take a minute.")
                else:
                    logger.info("Using cached leverage_tiers.")
                async def gather_results(input_coro: list) -> list:
                    return await asyncio.gather(*input_coro, return_exceptions=True)
                for input_coro in chunks(coros, 100):
                    with self._loop_lock:
                        results = self.loop.run_until_complete(gather_results(input_coro))
                    for res in results:
                        if isinstance(res, Exception):
                            logger.warning(f"Leverage tier exception: {res}")
                            continue
                        symbol, tier = res
                        tiers[symbol] = tier
                if coros:
                    self.cache_leverage_tiers(tiers, self._config["stake_currency"])
                logger.info(f"Done initializing {len(symbols)} markets.")
                return tiers
        return {}

    def cache_leverage_tiers(self, tiers: dict[str, list[dict]], stake_currency: str) -> None:
        filename = self._config["datadir"] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if not filename.parent.is_dir():
            filename.parent.mkdir(parents=True)
        data = {"updated": datetime.now(timezone.utc), "data": tiers}
        file_dump_json(filename, data)

    def load_cached_leverage_tiers(self, stake_currency: str, cache_time: Optional[timedelta] = None) -> Optional[dict[str, list[dict]]]:
        if not cache_time:
            cache_time = timedelta(weeks=4)
        filename = self._config["datadir"] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if filename.is_file():
            try:
                tiers = file_load_json(filename)
                updated = tiers.get("updated")
                if updated:
                    updated_dt = parser.parse(updated)
                    if updated_dt < datetime.now(timezone.utc) - cache_time:
                        logger.info("Cached leverage tiers are outdated. Will update.")
                        return None
                return tiers.get("data")
            except Exception:
                logger.exception("Error loading cached leverage tiers. Refreshing.")
        return None

    def fill_leverage_tiers(self) -> None:
        leverage_tiers: dict[str, list[dict]] = self.load_leverage_tiers()
        for pair, tiers in leverage_tiers.items():
            pair_tiers: list[dict] = []
            for tier in tiers:
                pair_tiers.append(self.parse_leverage_tier(tier))
            self._leverage_tiers[pair] = pair_tiers

    def parse_leverage_tier(self, tier: dict[str, Any]) -> dict[str, Any]:
        info = tier.get("info", {})
        return {
            "minNotional": tier["minNotional"],
            "maxNotional": tier["maxNotional"],
            "maintenanceMarginRate": tier["maintenanceMarginRate"],
            "maxLeverage": tier["maxLeverage"],
            "maintAmt": float(info["cum"]) if "cum" in info else None,
        }

    def get_max_leverage(self, pair: str, stake_amount: Optional[float]) -> float:
        if self.trading_mode == "SPOT":
            return 1.0
        if self.trading_mode == "FUTURES":
            if stake_amount is None:
                raise Exception(f"{self.name}.get_max_leverage requires stake_amount")
            if pair not in self._leverage_tiers:
                return 1.0
            pair_tiers = self._leverage_tiers[pair]
            if stake_amount == 0:
                return pair_tiers[0]["maxLeverage"]
            for tier_index in range(len(pair_tiers)):
                tier = pair_tiers[tier_index]
                lev = tier["maxLeverage"]
                if tier_index < len(pair_tiers) - 1:
                    next_tier = pair_tiers[tier_index + 1]
                    next_floor = next_tier["minNotional"] / next_tier["maxLeverage"]
                    if next_floor > stake_amount:
                        return min((tier["maxNotional"] / stake_amount), lev)
                else:
                    if stake_amount > tier["maxNotional"]:
                        raise Exception(f"Amount {stake_amount} too high for {pair}")
                    else:
                        return tier["maxLeverage"]
            raise Exception("No tier found for max leverage")
        elif self.trading_mode == "MARGIN":
            market = self.markets[pair]
            if market["limits"]["leverage"]["max"] is not None:
                return market["limits"]["leverage"]["max"]
            else:
                return 1.0
        else:
            return 1.0

    def _set_leverage(self, leverage: float, pair: Optional[str] = None, accept_fail: bool = False) -> None:
        if self._config["dry_run"] or not self.exchange_has("setLeverage"):
            return
        if self._ft_has.get("floor_leverage", False) is True:
            leverage = floor(leverage)
        try:
            res = self._api.set_leverage(symbol=pair, leverage=leverage)
            self._log_exchange_response("set_leverage", res)
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in _set_leverage: {e}") from e
        except (ccxt.BadRequest, ccxt.OperationRejected, ccxt.InsufficientFunds) as e:
            if not accept_fail:
                raise Exception(f"Could not set leverage: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in _set_leverage: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in _set_leverage: {e}") from e

    def get_interest_rate(self) -> float:
        return 0.0

    def funding_fee_cutoff(self, open_date: datetime) -> bool:
        return open_date.minute == 0 and open_date.second == 0

    def set_margin_mode(self, pair: str, margin_mode: Any, accept_fail: bool = False, params: Optional[dict[str, Any]] = None) -> None:
        if self._config["dry_run"] or not self.exchange_has("setMarginMode"):
            return
        if params is None:
            params = {}
        try:
            res = self._api.set_margin_mode(margin_mode.value, pair, params)
            self._log_exchange_response("set_margin_mode", res)
        except ccxt.DDoSProtection as e:
            raise Exception(f"DDoS error in set_margin_mode: {e}") from e
        except (ccxt.BadRequest, ccxt.OperationRejected) as e:
            if not accept_fail:
                raise Exception(f"Could not set margin mode: {e}") from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise Exception(f"Temporary error in set_margin_mode: {e}") from e
        except ccxt.BaseError as e:
            raise Exception(f"Operational error in set_margin_mode: {e}") from e

    def _fetch_and_calculate_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime, close_date: Optional[datetime] = None) -> float:
        if self.funding_fee_cutoff(open_date):
            open_date = timeframe_to_prev_date("1h", open_date)
        timeframe: str = self._ft_has["mark_ohlcv_timeframe"]
        timeframe_ff: str = self._ft_has["funding_fee_timeframe"]
        mark_price_type = self._ft_has["mark_ohlcv_price"]
        if not close_date:
            close_date = datetime.now(timezone.utc)
        since_ms: int = dt_ts(date_minus_candles(timeframe, self.ohlcv_candle_limit(timeframe, self._config["candle_type_def"])))
        mark_comb: tuple[str, str, Any] = (pair, timeframe, mark_price_type)
        funding_comb: tuple[str, str, Any] = (pair, timeframe_ff, "FUNDING_RATE")
        candle_histories: dict[tuple[str, str, Any], pd.DataFrame] = self.refresh_latest_ohlcv([mark_comb, funding_comb], since_ms=since_ms, cache=False, drop_incomplete=False)
        try:
            funding_rates = candle_histories[funding_comb]
            mark_rates = candle_histories[mark_comb]
        except KeyError:
            raise Exception("Could not find funding rates.")
        funding_mark_rates = self.combine_funding_and_mark(funding_rates, mark_rates)
        return self.calculate_funding_fees(funding_mark_rates, amount=amount, is_short=is_short, open_date=open_date, close_date=close_date)

    @staticmethod
    def combine_funding_and_mark(funding_rates: pd.DataFrame, mark_rates: pd.DataFrame, futures_funding_rate: Optional[int] = None) -> pd.DataFrame:
        if futures_funding_rate is None:
            return mark_rates.merge(funding_rates, on="date", how="inner", suffixes=["_mark", "_fund"])
        else:
            if funding_rates.empty:
                mark_rates["open_fund"] = futures_funding_rate
                return mark_rates.rename(columns={"open": "open_mark", "close": "close_mark", "high": "high_mark", "low": "low_mark", "volume": "volume_mark"})
            else:
                combined = mark_rates.merge(funding_rates, on="date", how="left", suffixes=["_mark", "_fund"])
                combined["open_fund"] = combined["open_fund"].fillna(futures_funding_rate)
                return combined

    def calculate_funding_fees(self, df: pd.DataFrame, amount: float, is_short: bool, open_date: datetime, close_date: datetime, time_in_ratio: Optional[float] = None) -> float:
        fees: float = 0.0
        if not df.empty:
            df1 = df[(df["date"] >= open_date) & (df["date"] <= close_date)]
            fees = sum(df1["open_fund"] * df1["open_mark"] * amount)
        if isnan(fees):
            fees = 0.0
        return fees if is_short else -fees

    def get_funding_fees(self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        if self.trading_mode == "FUTURES":
            try:
                if self._config["dry_run"]:
                    funding_fees = self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
                else:
                    funding_fees = self._get_funding_fees_from_exchange(pair, open_date)
                return funding_fees
            except Exception:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0

    def get_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: Optional[list] = None) -> Optional[float]:
        if self.trading_mode == "SPOT":
            return None
        elif self.trading_mode != "FUTURES":
            raise Exception(f"{self.name} does not support {self.margin_mode} {self.trading_mode}")
        liquidation_price: Optional[float] = None
        if self._config["dry_run"] or not self.exchange_has("fetchPositions"):
            liquidation_price = self.dry_run_liquidation_price(pair=pair, open_rate=open_rate, is_short=is_short, amount=amount, stake_amount=stake_amount, leverage=leverage, wallet_balance=wallet_balance, open_trades=open_trades or [])
        else:
            positions = self.fetch_positions(pair)
            if positions:
                pos = positions[0]
                liquidation_price = pos["liquidationPrice"]
        if liquidation_price is not None:
            buffer_amount: float = abs(open_rate - liquidation_price) * self.liquidation_buffer
            liquidation_price_buffer: float = liquidation_price - buffer_amount if is_short else liquidation_price + buffer_amount
            return max(liquidation_price_buffer, 0.0)
        else:
            return None

    def dry_run_liquidation_price(self, pair: str, open_rate: float, is_short: bool, amount: float, stake_amount: float, leverage: float, wallet_balance: float, open_trades: list) -> Optional[float]:
        market = self.markets[pair]
        taker_fee_rate: float = market["taker"]
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)
        if self.trading_mode == "FUTURES" and self.margin_mode == "ISOLATED":
            if market.get("inverse"):
                raise Exception("Inverse contracts not supported")
            value: float = wallet_balance / amount
            mm_ratio_taker: float = mm_ratio + taker_fee_rate
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise Exception("Only isolated futures supported")

    def get_maintenance_ratio_and_amt(self, pair: str, notional_value: float) -> tuple[float, Optional[float]]:
        if self._config.get("runmode") in ("OPTIMIZE",) or self.exchange_has("fetchLeverageTiers") or self.exchange_has("fetchMarketLeverageTiers"):
            if pair not in self._leverage_tiers:
                raise Exception(f"Maintenance margin rate for {pair} is unavailable for {self.name}")
            pair_tiers = self._leverage_tiers[pair]
            for tier in reversed(pair_tiers):
                if notional_value >= tier["minNotional"]:
                    return (tier["maintenanceMarginRate"], tier["maintAmt"])
            raise Exception("Notional value cannot be lower than 0")
        else:
            raise Exception(f"Cannot get maintenance ratio using {self.name}")

    # _log_exchange_response and other helper methods assumed implemented elsewhere.
    def _log_exchange_response(self, endpoint: str, response: Any, *, add_info: Optional[Any] = None) -> None:
        logger.info(f"API {endpoint}: {response}")

# End of annotated code.
