import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import inspect, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.engine.reflection import Inspector

from freqtrade.exceptions import OperationalException
from freqtrade.persistence.trade_model import Order, Trade

logger = logging.getLogger(__name__)


def get_table_names_for_table(inspector: Inspector, tabletype: str) -> List[str]:
    return [t for t in inspector.get_table_names() if t.startswith(tabletype)]


def has_column(columns: List[Dict[str, Any]], searchname: str) -> bool:
    return len(list(filter(lambda x: x["name"] == searchname, columns))) == 1


def get_column_def(columns: List[Dict[str, Any]], column: str, default: str) -> str:
    return default if not has_column(columns, column) else column


def get_backup_name(tabs: List[str], backup_prefix: str) -> str:
    table_back_name: str = backup_prefix
    for i, _ in enumerate(tabs):
        table_back_name = f"{backup_prefix}{i}"
        logger.debug(f"trying {table_back_name}")

    return table_back_name


def get_last_sequence_ids(engine: Engine, trade_back_name: str, order_back_name: str) -> Tuple[Optional[int], Optional[int]]:
    order_id: Optional[int] = None
    trade_id: Optional[int] = None

    if engine.name == "postgresql":
        with engine.begin() as connection:
            trade_id = connection.execute(text("select nextval('trades_id_seq')")).fetchone()[0]
            order_id = connection.execute(text("select nextval('orders_id_seq')")).fetchone()[0]
        with engine.begin() as connection:
            connection.execute(
                text(f"ALTER SEQUENCE orders_id_seq rename to {order_back_name}_id_seq_bak")
            )
            connection.execute(
                text(f"ALTER SEQUENCE trades_id_seq rename to {trade_back_name}_id_seq_bak")
            )
    return order_id, trade_id


def set_sequence_ids(engine: Engine, order_id: Optional[int], trade_id: Optional[int], pairlock_id: Optional[int] = None) -> None:
    if engine.name == "postgresql":
        with engine.begin() as connection:
            if order_id:
                connection.execute(text(f"ALTER SEQUENCE orders_id_seq RESTART WITH {order_id}"))
            if trade_id:
                connection.execute(text(f"ALTER SEQUENCE trades_id_seq RESTART WITH {trade_id}"))
            if pairlock_id:
                connection.execute(
                    text(f"ALTER SEQUENCE pairlocks_id_seq RESTART WITH {pairlock_id}")
                )


def drop_index_on_table(engine: Engine, inspector: Inspector, table_bak_name: str) -> None:
    with engine.begin() as connection:
        for index in inspector.get_indexes(table_bak_name):
            if engine.name == "mysql":
                connection.execute(text(f"drop index {index['name']} on {table_bak_name}"))
            else:
                connection.execute(text(f"drop index {index['name']}"))


def migrate_trades_and_orders_table(
    decl_base: Any,
    inspector: Inspector,
    engine: Engine,
    trade_back_name: str,
    cols: List[Dict[str, Any]],
    order_back_name: str,
    cols_order: List[Dict[str, Any]],
) -> None:
    base_currency: str = get_column_def(cols, "base_currency", "null")
    stake_currency: str = get_column_def(cols, "stake_currency", "null")
    fee_open: str = get_column_def(cols, "fee_open", "fee")
    fee_open_cost: str = get_column_def(cols, "fee_open_cost", "null")
    fee_open_currency: str = get_column_def(cols, "fee_open_currency", "null")
    fee_close: str = get_column_def(cols, "fee_close", "fee")
    fee_close_cost: str = get_column_def(cols, "fee_close_cost", "null")
    fee_close_currency: str = get_column_def(cols, "fee_close_currency", "null")
    open_rate_requested: str = get_column_def(cols, "open_rate_requested", "null")
    close_rate_requested: str = get_column_def(cols, "close_rate_requested", "null")
    stop_loss: str = get_column_def(cols, "stop_loss", "0.0")
    stop_loss_pct: str = get_column_def(cols, "stop_loss_pct", "null")
    initial_stop_loss: str = get_column_def(cols, "initial_stop_loss", "0.0")
    initial_stop_loss_pct: str = get_column_def(cols, "initial_stop_loss_pct", "null")
    is_stop_loss_trailing: str = get_column_def(
        cols,
        "is_stop_loss_trailing",
        f"coalesce({stop_loss_pct}, 0.0) <> coalesce({initial_stop_loss_pct}, 0.0)",
    )
    max_rate: str = get_column_def(cols, "max_rate", "0.0")
    min_rate: str = get_column_def(cols, "min_rate", "null")
    exit_reason: str = get_column_def(cols, "sell_reason", get_column_def(cols, "exit_reason", "null"))
    strategy: str = get_column_def(cols, "strategy", "null")
    enter_tag: str = get_column_def(cols, "buy_tag", get_column_def(cols, "enter_tag", "null"))
    realized_profit: str = get_column_def(cols, "realized_profit", "0.0")
    trading_mode: str = get_column_def(cols, "trading_mode", "null")
    leverage: str = get_column_def(cols, "leverage", "1.0")
    liquidation_price: str = get_column_def(
        cols, "liquidation_price", get_column_def(cols, "isolated_liq", "null")
    )
    if engine.name == "postgresql":
        is_short: str = get_column_def(cols, "is_short", "false")
    else:
        is_short: str = get_column_def(cols, "is_short", "0")
    interest_rate: str = get_column_def(cols, "interest_rate", "0.0")
    funding_fees: str = get_column_def(cols, "funding_fees", "0.0")
    funding_fee_running: str = get_column_def(cols, "funding_fee_running", "null")
    max_stake_amount: str = get_column_def(cols, "max_stake_amount", "stake_amount")

    if has_column(cols, "ticker_interval"):
        timeframe: str = get_column_def(cols, "timeframe", "ticker_interval")
    else:
        timeframe: str = get_column_def(cols, "timeframe", "null")

    open_trade_value: str = get_column_def(
        cols, "open_trade_value", f"amount * open_rate * (1 + {fee_open})"
    )
    close_profit_abs: str = get_column_def(
        cols, "close_profit_abs", f"(amount * close_rate * (1 - {fee_close})) - {open_trade_value}"
    )
    exit_order_status: str = get_column_def(
        cols, "exit_order_status", get_column_def(cols, "sell_order_status", "null")
    )
    amount_requested: str = get_column_def(cols, "amount_requested", "amount")
    amount_precision: str = get_column_def(cols, "amount_precision", "null")
    price_precision: str = get_column_def(cols, "price_precision", "null")
    precision_mode: str = get_column_def(cols, "precision_mode", "null")
    contract_size: str = get_column_def(cols, "contract_size", "null")
    precision_mode_price: str = get_column_def(
        cols, "precision_mode_price", get_column_def(cols, "precision_mode", "null")
    )

    with engine.begin() as connection:
        connection.execute(text(f"alter table trades rename to {trade_back_name}"))

    drop_index_on_table(engine, inspector, trade_back_name)

    order_id, trade_id = get_last_sequence_ids(engine, trade_back_name, order_back_name)

    drop_orders_table(engine, order_back_name)

    decl_base.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""insert into trades
            (id, exchange, pair, base_currency, stake_currency, is_open,
            fee_open, fee_open_cost, fee_open_currency,
            fee_close, fee_close_cost, fee_close_currency, open_rate,
            open_rate_requested, close_rate, close_rate_requested, close_profit,
            stake_amount, amount, amount_requested, open_date, close_date,
            stop_loss, stop_loss_pct, initial_stop_loss, initial_stop_loss_pct,
            is_stop_loss_trailing,
            max_rate, min_rate, exit_reason, exit_order_status, strategy, enter_tag,
            timeframe, open_trade_value, close_profit_abs,
            trading_mode, leverage, liquidation_price, is_short,
            interest_rate, funding_fees, funding_fee_running, realized_profit,
            amount_precision, price_precision, precision_mode, precision_mode_price, contract_size,
            max_stake_amount
            )
        select id, lower(exchange), pair, {base_currency} base_currency,
            {stake_currency} stake_currency,
            is_open, {fee_open} fee_open, {fee_open_cost} fee_open_cost,
            {fee_open_currency} fee_open_currency, {fee_close} fee_close,
            {fee_close_cost} fee_close_cost, {fee_close_currency} fee_close_currency,
            open_rate, {open_rate_requested} open_rate_requested, close_rate,
            {close_rate_requested} close_rate_requested, close_profit,
            stake_amount, amount, {amount_requested}, open_date, close_date,
            {stop_loss} stop_loss, {stop_loss_pct} stop_loss_pct,
            {initial_stop_loss} initial_stop_loss,
            {initial_stop_loss_pct} initial_stop_loss_pct,
            {is_stop_loss_trailing} is_stop_loss_trailing,
            {max_rate} max_rate, {min_rate} min_rate,
            case when {exit_reason} = 'sell_signal' then 'exit_signal'
                 when {exit_reason} = 'custom_sell' then 'custom_exit'
                 when {exit_reason} = 'force_sell' then 'force_exit'
                 when {exit_reason} = 'emergency_sell' then 'emergency_exit'
                 else {exit_reason}
            end exit_reason,
            {exit_order_status} exit_order_status,
            {strategy} strategy, {enter_tag} enter_tag, {timeframe} timeframe,
            {open_trade_value} open_trade_value, {close_profit_abs} close_profit_abs,
            {trading_mode} trading_mode, {leverage} leverage, {liquidation_price} liquidation_price,
            {is_short} is_short, {interest_rate} interest_rate,
            {funding_fees} funding_fees, {funding_fee_running} funding_fee_running,
            {realized_profit} realized_profit,
            {amount_precision} amount_precision, {price_precision} price_precision,
            {precision_mode} precision_mode, {precision_mode_price} precision_mode_price,
            {contract_size} contract_size, {max_stake_amount} max_stake_amount
            from {trade_back_name}
            """
            )
        )

    migrate_orders_table(engine, order_back_name, cols_order)
    set_sequence_ids(engine, order_id, trade_id)


def drop_orders_table(engine: Engine, table_back_name: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"create table {table_back_name} as select * from orders"))
        connection.execute(text("drop table orders"))


def migrate_orders_table(engine: Engine, table_back_name: str, cols_order: List[Dict[str, Any]]) -> None:
    ft_fee_base: str = get_column_def(cols_order, "ft_fee_base", "null")
    average: str = get_column_def(cols_order, "average", "null")
    stop_price: str = get_column_def(cols_order, "stop_price", "null")
    funding_fee: str = get_column_def(cols_order, "funding_fee", "0.0")
    ft_amount: str = get_column_def(cols_order, "ft_amount", "coalesce(amount, 0.0)")
    ft_price: str = get_column_def(cols_order, "ft_price", "coalesce(price, 0.0)")
    ft_cancel_reason: str = get_column_def(cols_order, "ft_cancel_reason", "null")
    ft_order_tag: str = get_column_def(cols_order, "ft_order_tag", "null")

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
            insert into orders (id, ft_trade_id, ft_order_side, ft_pair, ft_is_open, order_id,
            status, symbol, order_type, side, price, amount, filled, average, remaining, cost,
            stop_price, order_date, order_filled_date, order_update_date, ft_fee_base, funding_fee,
            ft_amount, ft_price, ft_cancel_reason, ft_order_tag
            )
            select id, ft_trade_id, ft_order_side, ft_pair, ft_is_open, order_id,
            status, symbol, order_type, side, price, amount, filled, {average} average, remaining,
            cost, {stop_price} stop_price, order_date, order_filled_date,
            order_update_date, {ft_fee_base} ft_fee_base, {funding_fee} funding_fee,
            {ft_amount} ft_amount, {ft_price} ft_price, {ft_cancel_reason} ft_cancel_reason,
            {ft_order_tag} ft_order_tag
            from {table_back_name}
            """
            )
        )


def migrate_pairlocks_table(
    decl_base: Any, inspector: Inspector, engine: Engine, pairlock_back_name: str, cols: List[Dict[str, Any]]
) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"alter table pairlocks rename to {pairlock_back_name}"))

    drop_index_on_table(engine, inspector, pairlock_back_name)

    side: str = get_column_def(cols, "side", "'*'")

    decl_base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"""insert into pairlocks
        (id, pair, side, reason, lock_time,
         lock_end_time, active)
        select id, pair, {side} side, reason, lock_time,
         lock_end_time, active
        from {pairlock_back_name}
        """
            )
        )


def set_sqlite_to_wal(engine: Engine) -> None:
    if engine.name == "sqlite" and str(engine.url) != "sqlite://":
        with engine.begin() as connection:
            connection.execute(text("PRAGMA journal_mode=wal"))


def fix_old_dry_orders(engine: Engine) -> None:
    with engine.begin() as connection:
        stmt = (
            update(Order)
            .where(
                Order.ft_is_open.is_(True),
                Order.ft_order_side == "stoploss",
                Order.order_id.like("dry%"),
            )
            .values(ft_is_open=False)
        )
        connection.execute(stmt)

        stmt = (
            update(Order)
            .where(
                Order.ft_is_open.is_(True),
                Order.ft_trade_id.not_in(select(Trade.id).where(Trade.is_open.is_(True))),
                Order.ft_order_side != "stoploss",
                Order.order_id.like("dry%"),
            )
            .values(ft_is_open=False)
        )
        connection.execute(stmt)


def check_migrate(engine: Engine, decl_base: Any, previous_tables: Set[str]) -> None:
    inspector: Inspector = inspect(engine)

    cols_trades: List[Dict[str, Any]] = inspector.get_columns("trades")
    cols_orders: List[Dict[str, Any]] = inspector.get_columns("orders")
    cols_pairlocks: List[Dict[str, Any]] = inspector.get_columns("pairlocks")
    tabs: List[str] = get_table_names_for_table(inspector, "trades")
    table_back_name: str = get_backup_name(tabs, "trades_bak")
    order_tabs: List[str] = get_table_names_for_table(inspector, "orders")
    order_table_bak_name: str = get_backup_name(order_tabs, "orders_bak")
    pairlock_tabs: List[str] = get_table_names_for_table(inspector, "pairlocks")
    pairlock_table_bak_name: str = get_backup_name(pairlock_tabs, "pairlocks_bak")

    migrating: bool = False
    if not has_column(cols_trades, "precision_mode_price"):
        migrating = True
        logger.info(
            f"Running database migration for trades - "
            f"backup: {table_back_name}, {order_table_bak_name}"
        )
        migrate_trades_and_orders_table(
            decl_base,
            inspector,
            engine,
            table_back_name,
            cols_trades,
            order_table_bak_name,
            cols_orders,
        )

    if not has_column(cols_pairlocks, "side"):
        migrating = True
        logger.info(f"Running database migration for pairlocks - backup: {pairlock_table_bak_name}")
        migrate_pairlocks_table(
            decl_base, inspector, engine, pairlock_table_bak_name, cols_pairlocks
        )
    if "orders" not in previous_tables and "trades" in previous_tables:
        raise OperationalException(
            "Your database seems to be very old. "
            "Please update to freqtrade 2022.3 to migrate this database or "
            "start with a fresh database."
        )

    set_sqlite_to_wal(engine)
    fix_old_dry_orders(engine)

    if migrating:
        logger.info("Database migration finished.")