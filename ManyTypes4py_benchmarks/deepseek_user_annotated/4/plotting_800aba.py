import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.btanalysis import (
    analyze_trade_parallelism,
    extract_trades_of_period,
    load_trades,
)
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange, load_data
from freqtrade.data.metrics import (
    calculate_max_drawdown,
    calculate_underwater,
    combine_dataframes_with_mean,
    create_cum_profit,
)
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.misc import pair_to_filename
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import get_dry_run_wallet


logger = logging.getLogger(__name__)


try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots
except ImportError:
    logger.exception("Module plotly not found \n Please install using `pip3 install plotly`")
    exit(1)


def init_plotscript(config: Config, markets: List[Dict[str, Any]], startup_candles: int = 0) -> Dict[str, Any]:
    """
    Initialize objects needed for plotting
    :return: Dict with candle (OHLCV) data, trades and pairs
    """

    if "pairs" in config:
        pairs = expand_pairlist(config["pairs"], markets)
    else:
        pairs = expand_pairlist(config["exchange"]["pair_whitelist"], markets)

    # Set timerange to use
    timerange = TimeRange.parse_timerange(config.get("timerange"))

    data = load_data(
        datadir=config.get("datadir"),
        pairs=pairs,
        timeframe=config["timeframe"],
        timerange=timerange,
        startup_candles=startup_candles,
        data_format=config["dataformat_ohlcv"],
        candle_type=config.get("candle_type_def", CandleType.SPOT),
    )

    if startup_candles and data:
        min_date, max_date = get_timerange(data)
        logger.info(f"Loading data from {min_date} to {max_date}")
        timerange.adjust_start_if_necessary(
            timeframe_to_seconds(config["timeframe"]), startup_candles, min_date
        )

    no_trades = False
    filename = config.get("exportfilename")
    if config.get("no_trades", False):
        no_trades = True
    elif config["trade_source"] == "file":
        if not filename.is_dir() and not filename.is_file():
            logger.warning("Backtest file is missing skipping trades.")
            no_trades = True
    try:
        trades = load_trades(
            config["trade_source"],
            db_url=config.get("db_url"),
            exportfilename=filename,
            no_trades=no_trades,
            strategy=config.get("strategy"),
        )
    except ValueError as e:
        raise OperationalException(e) from e
    if not trades.empty:
        trades = trim_dataframe(trades, timerange, df_date_col="open_date")

    return {
        "ohlcv": data,
        "trades": trades,
        "pairs": pairs,
        "timerange": timerange,
    }


def add_indicators(fig: make_subplots, row: int, indicators: Dict[str, Dict[str, Any]], data: pd.DataFrame) -> make_subplots:
    """
    Generate all the indicators selected by the user for a specific row, based on the configuration
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param indicators: Dict of Indicators with configuration options.
                       Dict key must correspond to dataframe column.
    :param data: candlestick DataFrame
    """
    plot_kinds = {
        "scatter": go.Scatter,
        "bar": go.Bar,
    }
    for indicator, conf in indicators.items():
        logger.debug(f"indicator {indicator} with config {conf}")
        if indicator in data:
            kwargs = {"x": data["date"], "y": data[indicator].values, "name": indicator}

            plot_type = conf.get("type", "scatter")
            color = conf.get("color")
            if plot_type == "bar":
                kwargs.update(
                    {
                        "marker_color": color or "DarkSlateGrey",
                        "marker_line_color": color or "DarkSlateGrey",
                    }
                )
            else:
                if color:
                    kwargs.update({"line": {"color": color}})
                kwargs["mode"] = "lines"
                if plot_type != "scatter":
                    logger.warning(
                        f"Indicator {indicator} has unknown plot trace kind {plot_type}"
                        f', assuming "scatter".'
                    )

            kwargs.update(conf.get("plotly", {}))
            trace = plot_kinds[plot_type](**kwargs)
            fig.add_trace(trace, row, 1)
        else:
            logger.info(
                'Indicator "%s" ignored. Reason: This indicator is not found in your strategy.',
                indicator,
            )

    return fig


def add_profit(fig: make_subplots, row: int, data: pd.DataFrame, column: str, name: str) -> make_subplots:
    """
    Add profit-plot
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param data: candlestick DataFrame
    :param column: Column to use for plot
    :param name: Name to use
    :return: fig with added profit plot
    """
    profit = go.Scatter(
        x=data.index,
        y=data[column],
        name=name,
    )
    fig.add_trace(profit, row, 1)

    return fig


def add_max_drawdown(
    fig: make_subplots,
    row: int,
    trades: pd.DataFrame,
    df_comb: pd.DataFrame,
    timeframe: str,
    starting_balance: float,
) -> make_subplots:
    """
    Add scatter points indicating max drawdown
    """
    try:
        drawdown = calculate_max_drawdown(trades, starting_balance=starting_balance)

        drawdown = go.Scatter(
            x=[drawdown.high_date, drawdown.low_date],
            y=[
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.high_date), "cum_profit"],
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.low_date), "cum_profit"],
            ],
            mode="markers",
            name=f"Max drawdown {drawdown.relative_account_drawdown:.2%}",
            text=f"Max drawdown {drawdown.relative_account_drawdown:.2%}",
            marker=dict(symbol="square-open", size=9, line=dict(width=2), color="green"),
        )
        fig.add_trace(drawdown, row, 1)
    except ValueError:
        logger.warning("No trades found - not plotting max drawdown.")
    return fig


def add_underwater(fig: make_subplots, row: int, trades: pd.DataFrame, starting_balance: float) -> make_subplots:
    """
    Add underwater plots
    """
    try:
        underwater = calculate_underwater(
            trades, value_col="profit_abs", starting_balance=starting_balance
        )

        underwater_plot = go.Scatter(
            x=underwater["date"],
            y=underwater["drawdown"],
            name="Underwater Plot",
            fill="tozeroy",
            fillcolor="#cc362b",
            line={"color": "#cc362b"},
        )

        underwater_plot_relative = go.Scatter(
            x=underwater["date"],
            y=(-underwater["drawdown_relative"]),
            name="Underwater Plot (%)",
            fill="tozeroy",
            fillcolor="green",
            line={"color": "green"},
        )

        fig.add_trace(underwater_plot, row, 1)
        fig.add_trace(underwater_plot_relative, row + 1, 1)
    except ValueError:
        logger.warning("No trades found - not plotting underwater plot")
    return fig


def add_parallelism(fig: make_subplots, row: int, trades: pd.DataFrame, timeframe: str) -> make_subplots:
    """
    Add Chart showing trade parallelism
    """
    try:
        result = analyze_trade_parallelism(trades, timeframe)

        drawdown = go.Scatter(
            x=result.index,
            y=result["open_trades"],
            name="Parallel trades",
            fill="tozeroy",
            fillcolor="#242222",
            line={"color": "#242222"},
        )
        fig.add_trace(drawdown, row, 1)
    except ValueError:
        logger.warning("No trades found - not plotting Parallelism.")
    return fig


def plot_trades(fig: make_subplots, trades: pd.DataFrame) -> make_subplots:
    """
    Add trades to "fig"
    """
    # Trades can be empty
    if trades is not None and len(trades) > 0:
        # Create description for exit summarizing the trade
        trades["desc"] = trades.apply(
            lambda row: f"{row['profit_ratio']:.2%}, "
            + (f"{row['enter_tag']}, " if row["enter_tag"] is not None else "")
            + f"{row['exit_reason']}, "
            + f"{row['trade_duration']} min",
            axis=1,
        )
        trade_entries = go.Scatter(
            x=trades["open_date"],
            y=trades["open_rate"],
            mode="markers",
            name="Trade entry",
            text=trades["desc"],
            marker=dict(symbol="circle-open", size=11, line=dict(width=2), color="cyan"),
        )

        trade_exits = go.Scatter(
            x=trades.loc[trades["profit_ratio"] > 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] > 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] > 0, "desc"],
            mode="markers",
            name="Exit - Profit",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="green"),
        )
        trade_exits_loss = go.Scatter(
            x=trades.loc[trades["profit_ratio"] <= 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] <= 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] <= 0, "desc"],
            mode="markers",
            name="Exit - Loss",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="red"),
        )
        fig.add_trace(trade_entries, 1, 1)
        fig.add_trace(trade_exits, 1, 1)
        fig.add_trace(trade_exits_loss, 1, 1)
    else:
        logger.warning("No trades found.")
    return fig


def create_plotconfig(
    indicators1: List[str], indicators2: List[str], plot_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Combines indicators 1 and indicators 2 into plot_config if necessary
    :param indicators1: List containing Main plot indicators
    :param indicators2: List containing Sub plot indicators
    :param plot_config: Dict of Dicts containing advanced plot configuration
    :return: plot_config - eventually with indicators 1 and 2
    """

    if plot_config:
        if indicators1:
            plot_config["main_plot"] = {ind: {} for ind in indicators1}
        if indicators2:
            plot_config["subplots"] = {"Other": {ind: {} for ind in indicators2}}

    if not plot_config:
        # If no indicators and no plot-config given, use defaults.
        if not indicators1:
            indicators1 = ["sma", "ema3", "ema5"]
        if not indicators2:
            indicators2 = ["macd", "macdsignal"]

        # Create subplot configuration if plot_config is not available.
        plot_config = {
            "main_plot": {ind: {} for ind in indicators1},
            "subplots": {"Other": {ind: {} for ind in indicators2}},
        }
    if "main_plot" not in plot_config:
        plot_config["main_plot"] = {}

    if "subplots" not in plot_config:
        plot_config["subplots"] = {}
    return plot_config


def plot_area(
    fig: make_subplots,
    row: int,
    data: pd.DataFrame,
    indicator_a: str,
    indicator_b: str,
    label: str = "",
    fill_color: str = "rgba(0,176,246,0.2)",
) -> make_subplots:
    """Creates a plot for the area between two traces and adds it to fig.
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param data: candlestick DataFrame
    :param indicator_a: indicator name as populated in strategy
    :param indicator_b: indicator name as populated in strategy
    :param label: label for the filled area
    :param fill_color: color to be used for the filled area
    :return: fig with added  filled_traces plot
    """
    if indicator_a in data and indicator_b in data:
        # make lines invisible to get the area plotted, only.
        line = {"color": "rgba(255,255,255,0)"}
        # TODO: Figure out why scattergl causes problems plotly/plotly.js#2284
        trace_a = go.Scatter(x=data.date, y=data[indicator_a], showlegend=False, line=line)
        trace_b = go.Scatter(
            x=data.date,
            y=data[indicator_b],
            name=label,
            fill="tonexty",
            fillcolor=fill_color,
            line=line,
        )
        fig.add_trace(trace_a, row, 1)
        fig.add_trace(trace_b, row, 1)
    return fig


def add_areas(fig: make_subplots, row: int, data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> make_subplots:
    """Adds all area plots (specified in plot_config) to fig.
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param data: candlestick DataFrame
    :param indicators: dict with indicators. ie.: plot_config['main_plot'] or
                            plot_config['subplots'][subplot_label]
    :return: fig with added  filled_traces plot
    """
    for indicator, ind_conf in indicators.items():
        if "fill_to" in ind_conf:
            indicator_b = ind_conf["fill_to"]
            if indicator in data and indicator_b in data:
                label = ind_conf.get("fill_label", f"{indicator}<>{indicator_b}")
                fill_color = ind_conf.get("fill_color", "rgba(0,176,246,0.2)")
                fig = plot_area(
                    fig, row, data, indicator, indicator_b, label=label, fill_color=fill_color
                )
            elif indicator not in data:
                logger.info(
                    'Indicator "%s" ignored. Reason: This indicator is not found in your strategy.',
                    indicator,
                )
            elif indicator_b not in data:
                logger.info(
                    'fill_to: "%s" ignored. Reason: This indicator is not in your strategy.',
                    indicator_b,
                )
    return fig


def create_scatter(data: pd.DataFrame, column_name: str, color: str, direction: str) -> Optional[go.Scatter]:
    if column_name in data.columns:
        df_short = data[data[column_name] == 1]
        if len(df_short) > 0:
            shorts = go.Scatter(
                x=df_short.date,
                y=df_short.close,
                mode="markers",
                name=column_name,
                marker=dict(
                    symbol=f"triangle-{direction}-dot",
                    size=9,
                    line=dict(width=1),
                    color=color,
                ),
            )
            return shorts
        else:
            logger.warning(f"No {column_name}-signals found.")

    return None


def generate_candlestick_graph(
    pair: str,
    data: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    *,
    indicators1: Optional[List[str]] = None,
    indicators2: Optional[List[str]] = None,
    plot_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Figure:
    """
    Generate the graph from the data generated by Backtesting or from DB
    Volume will always be plotted in row2, so Row 1 and 3 are to our disposal for custom indicators
    :param pair: Pair to Display on the graph
    :param data: OHLCV DataFrame containing indicators and entry/exit signals
    :param trades: All trades created
    :param indicators1: List containing Main plot indicators
    :param indicators2: List containing Sub plot indicators
    :param plot_config: Dict of Dicts containing advanced plot configuration
    :return: Plotly figure
    """
    plot_config = create_plotconfig(
        indicators1 or [],
        indicators2 or [],
        plot_config or {},
    )
    rows = 2 + len(plot_config["subplots"])
    row_widths = [1 for _ in plot_config["subplots"]]
    # Define the graph
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_width=row_widths + [1, 4],
        vertical_spacing=0.0001,
    )
    fig["layout"].update(title=pair)
