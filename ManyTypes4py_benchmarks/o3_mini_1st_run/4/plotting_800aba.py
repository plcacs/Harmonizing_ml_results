#!/usr/bin/env python3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.btanalysis import analyze_trade_parallelism, extract_trades_of_period, load_trades
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

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    from plotly.subplots import make_subplots
except ImportError:
    logger = logging.getLogger(__name__)
    logger.exception('Module plotly not found \n Please install using `pip3 install plotly`')
    exit(1)

logger = logging.getLogger(__name__)


def init_plotscript(config: Dict[str, Any], markets: List[str], startup_candles: int = 0) -> Dict[str, Any]:
    """
    Initialize objects needed for plotting
    :return: Dict with candle (OHLCV) data, trades and pairs
    """
    if 'pairs' in config:
        pairs: List[str] = expand_pairlist(config['pairs'], markets)
    else:
        pairs = expand_pairlist(config['exchange']['pair_whitelist'], markets)
    timerange: TimeRange = TimeRange.parse_timerange(config.get('timerange'))
    data: Dict[str, pd.DataFrame] = load_data(
        datadir=config.get('datadir'),
        pairs=pairs,
        timeframe=config['timeframe'],
        timerange=timerange,
        startup_candles=startup_candles,
        data_format=config['dataformat_ohlcv'],
        candle_type=config.get('candle_type_def', CandleType.SPOT),
    )
    if startup_candles and data:
        min_date, max_date = get_timerange(data)
        logger.info(f'Loading data from {min_date} to {max_date}')
        timerange.adjust_start_if_necessary(timeframe_to_seconds(config['timeframe']), startup_candles, min_date)
    no_trades: bool = False
    filename: Union[Path, Any] = config.get('exportfilename')
    if config.get('no_trades', False):
        no_trades = True
    elif config['trade_source'] == 'file':
        if not filename.is_dir() and (not filename.is_file()):
            logger.warning('Backtest file is missing skipping trades.')
            no_trades = True
    try:
        trades: pd.DataFrame = load_trades(
            config['trade_source'],
            db_url=config.get('db_url'),
            exportfilename=filename,
            no_trades=no_trades,
            strategy=config.get('strategy'),
        )
    except ValueError as e:
        raise OperationalException(e) from e
    if not trades.empty:
        trades = trim_dataframe(trades, timerange, df_date_col='open_date')
    return {'ohlcv': data, 'trades': trades, 'pairs': pairs, 'timerange': timerange}


def add_indicators(
    fig: go.Figure, row: int, indicators: Dict[str, Dict[str, Any]], data: pd.DataFrame
) -> go.Figure:
    """
    Generate all the indicators selected by the user for a specific row, based on the configuration
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param indicators: Dict of Indicators with configuration options.
                       Dict key must correspond to dataframe column.
    :param data: candlestick DataFrame
    """
    plot_kinds: Dict[str, Any] = {'scatter': go.Scatter, 'bar': go.Bar}
    for indicator, conf in indicators.items():
        logger.debug(f'indicator {indicator} with config {conf}')
        if indicator in data:
            kwargs: Dict[str, Any] = {'x': data['date'], 'y': data[indicator].values, 'name': indicator}
            plot_type: str = conf.get('type', 'scatter')
            color: Optional[str] = conf.get('color')
            if plot_type == 'bar':
                kwargs.update({'marker_color': color or 'DarkSlateGrey', 'marker_line_color': color or 'DarkSlateGrey'})
            else:
                if color:
                    kwargs.update({'line': {'color': color}})
                kwargs['mode'] = 'lines'
                if plot_type != 'scatter':
                    logger.warning(f'Indicator {indicator} has unknown plot trace kind {plot_type}, assuming "scatter".')
            kwargs.update(conf.get('plotly', {}))
            trace = plot_kinds[plot_type](**kwargs)
            fig.add_trace(trace, row, 1)
        else:
            logger.info('Indicator "%s" ignored. Reason: This indicator is not found in your strategy.', indicator)
    return fig


def add_profit(fig: go.Figure, row: int, data: pd.DataFrame, column: str, name: str) -> go.Figure:
    """
    Add profit-plot
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param data: candlestick DataFrame
    :param column: Column to use for plot
    :param name: Name to use
    :return: fig with added profit plot
    """
    profit = go.Scatter(x=data.index, y=data[column], name=name)
    fig.add_trace(profit, row, 1)
    return fig


def add_max_drawdown(
    fig: go.Figure, row: int, trades: pd.DataFrame, df_comb: pd.DataFrame, timeframe: str, starting_balance: float
) -> go.Figure:
    """
    Add scatter points indicating max drawdown
    """
    try:
        drawdown = calculate_max_drawdown(trades, starting_balance=starting_balance)
        trace = go.Scatter(
            x=[drawdown.high_date, drawdown.low_date],
            y=[
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.high_date), 'cum_profit'],
                df_comb.loc[timeframe_to_prev_date(timeframe, drawdown.low_date), 'cum_profit'],
            ],
            mode='markers',
            name=f'Max drawdown {drawdown.relative_account_drawdown:.2%}',
            text=f'Max drawdown {drawdown.relative_account_drawdown:.2%}',
            marker=dict(symbol='square-open', size=9, line=dict(width=2), color='green'),
        )
        fig.add_trace(trace, row, 1)
    except ValueError:
        logger.warning('No trades found - not plotting max drawdown.')
    return fig


def add_underwater(fig: go.Figure, row: int, trades: pd.DataFrame, starting_balance: float) -> go.Figure:
    """
    Add underwater plots
    """
    try:
        underwater = calculate_underwater(trades, value_col='profit_abs', starting_balance=starting_balance)
        underwater_plot = go.Scatter(
            x=underwater['date'],
            y=underwater['drawdown'],
            name='Underwater Plot',
            fill='tozeroy',
            fillcolor='#cc362b',
            line={'color': '#cc362b'},
        )
        underwater_plot_relative = go.Scatter(
            x=underwater['date'],
            y=-underwater['drawdown_relative'],
            name='Underwater Plot (%)',
            fill='tozeroy',
            fillcolor='green',
            line={'color': 'green'},
        )
        fig.add_trace(underwater_plot, row, 1)
        fig.add_trace(underwater_plot_relative, row + 1, 1)
    except ValueError:
        logger.warning('No trades found - not plotting underwater plot')
    return fig


def add_parallelism(fig: go.Figure, row: int, trades: pd.DataFrame, timeframe: str) -> go.Figure:
    """
    Add Chart showing trade parallelism
    """
    try:
        result = analyze_trade_parallelism(trades, timeframe)
        trace = go.Scatter(
            x=result.index,
            y=result['open_trades'],
            name='Parallel trades',
            fill='tozeroy',
            fillcolor='#242222',
            line={'color': '#242222'},
        )
        fig.add_trace(trace, row, 1)
    except ValueError:
        logger.warning('No trades found - not plotting Parallelism.')
    return fig


def plot_trades(fig: go.Figure, trades: pd.DataFrame) -> go.Figure:
    """
    Add trades to "fig"
    """
    if trades is not None and len(trades) > 0:
        trades['desc'] = trades.apply(
            lambda row: f'{row["profit_ratio"]:.2%}, '
            + (f'{row["enter_tag"]}, ' if row["enter_tag"] is not None else '')
            + f'{row["exit_reason"]}, '
            + f'{row["trade_duration"]} min',
            axis=1,
        )
        trade_entries = go.Scatter(
            x=trades['open_date'],
            y=trades['open_rate'],
            mode='markers',
            name='Trade entry',
            text=trades['desc'],
            marker=dict(symbol='circle-open', size=11, line=dict(width=2), color='cyan'),
        )
        trade_exits = go.Scatter(
            x=trades.loc[trades['profit_ratio'] > 0, 'close_date'],
            y=trades.loc[trades['profit_ratio'] > 0, 'close_rate'],
            text=trades.loc[trades['profit_ratio'] > 0, 'desc'],
            mode='markers',
            name='Exit - Profit',
            marker=dict(symbol='square-open', size=11, line=dict(width=2), color='green'),
        )
        trade_exits_loss = go.Scatter(
            x=trades.loc[trades['profit_ratio'] <= 0, 'close_date'],
            y=trades.loc[trades['profit_ratio'] <= 0, 'close_rate'],
            text=trades.loc[trades['profit_ratio'] <= 0, 'desc'],
            mode='markers',
            name='Exit - Loss',
            marker=dict(symbol='square-open', size=11, line=dict(width=2), color='red'),
        )
        fig.add_trace(trade_entries, 1, 1)
        fig.add_trace(trade_exits, 1, 1)
        fig.add_trace(trade_exits_loss, 1, 1)
    else:
        logger.warning('No trades found.')
    return fig


def create_plotconfig(
    indicators1: List[str], indicators2: List[str], plot_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combines indicators 1 and indicators 2 into plot_config if necessary
    :param indicators1: List containing Main plot indicators
    :param indicators2: List containing Sub plot indicators
    :param plot_config: Dict of Dicts containing advanced plot configuration
    :return: plot_config - eventually with indicators 1 and 2
    """
    if plot_config:
        if indicators1:
            plot_config['main_plot'] = {ind: {} for ind in indicators1}
        if indicators2:
            plot_config['subplots'] = {'Other': {ind: {} for ind in indicators2}}
    if not plot_config:
        if not indicators1:
            indicators1 = ['sma', 'ema3', 'ema5']
        if not indicators2:
            indicators2 = ['macd', 'macdsignal']
        plot_config = {'main_plot': {ind: {} for ind in indicators1}, 'subplots': {'Other': {ind: {} for ind in indicators2}}}
    if 'main_plot' not in plot_config:
        plot_config['main_plot'] = {}
    if 'subplots' not in plot_config:
        plot_config['subplots'] = {}
    return plot_config


def plot_area(
    fig: go.Figure,
    row: int,
    data: pd.DataFrame,
    indicator_a: str,
    indicator_b: str,
    label: str = '',
    fill_color: str = 'rgba(0,176,246,0.2)',
) -> go.Figure:
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
        line: Dict[str, str] = {'color': 'rgba(255,255,255,0)'}
        trace_a = go.Scatter(x=data.date, y=data[indicator_a], showlegend=False, line=line)
        trace_b = go.Scatter(
            x=data.date,
            y=data[indicator_b],
            name=label,
            fill='tonexty',
            fillcolor=fill_color,
            line=line,
        )
        fig.add_trace(trace_a, row, 1)
        fig.add_trace(trace_b, row, 1)
    return fig


def add_areas(
    fig: go.Figure, row: int, data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]
) -> go.Figure:
    """Adds all area plots (specified in plot_config) to fig.
    :param fig: Plot figure to append to
    :param row: row number for this plot
    :param data: candlestick DataFrame
    :param indicators: dict with indicators. ie.: plot_config['main_plot'] or
                            plot_config['subplots'][subplot_label]
    :return: fig with added  filled_traces plot
    """
    for indicator, ind_conf in indicators.items():
        if 'fill_to' in ind_conf:
            indicator_b: str = ind_conf['fill_to']
            if indicator in data and indicator_b in data:
                label: str = ind_conf.get('fill_label', f'{indicator}<>{indicator_b}')
                fill_color: str = ind_conf.get('fill_color', 'rgba(0,176,246,0.2)')
                fig = plot_area(fig, row, data, indicator, indicator_b, label=label, fill_color=fill_color)
            elif indicator not in data:
                logger.info('Indicator "%s" ignored. Reason: This indicator is not found in your strategy.', indicator)
            elif indicator_b not in data:
                logger.info('fill_to: "%s" ignored. Reason: This indicator is not in your strategy.', indicator_b)
    return fig


def create_scatter(data: pd.DataFrame, column_name: str, color: str, direction: str) -> Optional[go.Scatter]:
    if column_name in data.columns:
        df_filtered: pd.DataFrame = data[data[column_name] == 1]
        if len(df_filtered) > 0:
            scatter = go.Scatter(
                x=df_filtered.date,
                y=df_filtered.close,
                mode='markers',
                name=column_name,
                marker=dict(symbol=f'triangle-{direction}-dot', size=9, line=dict(width=1), color=color),
            )
            return scatter
        else:
            logger.warning(f'No {column_name}-signals found.')
    return None


def generate_candlestick_graph(
    pair: str,
    data: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    *,
    indicators1: Optional[List[str]] = None,
    indicators2: Optional[List[str]] = None,
    plot_config: Optional[Dict[str, Any]] = None
) -> go.Figure:
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
    plot_config = create_plotconfig(indicators1 or [], indicators2 or [], plot_config or {})
    rows: int = 2 + len(plot_config['subplots'])
    row_widths: List[int] = [1 for _ in plot_config['subplots']]
    fig: go.Figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_width=row_widths + [1, 4],
        vertical_spacing=0.0001,
    )
    fig['layout'].update(title=pair)
    fig['layout']['yaxis1'].update(title='Price')
    fig['layout']['yaxis2'].update(title='Volume')
    for i, name in enumerate(plot_config['subplots']):
        fig['layout'][f'yaxis{3 + i}'].update(title=name)
    fig['layout']['xaxis']['rangeslider'].update(visible=False)
    fig.update_layout(modebar_add=['v1hovermode', 'toggleSpikeLines'])
    candles = go.Candlestick(
        x=data.date, open=data.open, high=data.high, low=data.low, close=data.close, name='Price'
    )
    fig.add_trace(candles, 1, 1)
    longs = create_scatter(data, 'enter_long', 'green', 'up')
    exit_longs = create_scatter(data, 'exit_long', 'red', 'down')
    shorts = create_scatter(data, 'enter_short', 'blue', 'down')
    exit_shorts = create_scatter(data, 'exit_short', 'violet', 'up')
    for scatter in [longs, exit_longs, shorts, exit_shorts]:
        if scatter:
            fig.add_trace(scatter, 1, 1)
    fig = plot_area(fig, 1, data, 'bb_lowerband', 'bb_upperband', label='Bollinger Band')
    try:
        del plot_config['main_plot']['bb_lowerband']
        del plot_config['main_plot']['bb_upperband']
    except KeyError:
        pass
    fig = add_indicators(fig=fig, row=1, indicators=plot_config['main_plot'], data=data)
    fig = add_areas(fig, 1, data, plot_config['main_plot'])
    fig = plot_trades(fig, trades)  # type: ignore
    volume = go.Bar(
        x=data['date'],
        y=data['volume'],
        name='Volume',
        marker_color='DarkSlateGrey',
        marker_line_color='DarkSlateGrey',
    )
    fig.add_trace(volume, 2, 1)
    for i, label in enumerate(plot_config['subplots']):
        sub_config: Dict[str, Any] = plot_config['subplots'][label]
        row: int = 3 + i
        fig = add_indicators(fig=fig, row=row, indicators=sub_config, data=data)
        fig = add_areas(fig, row, data, sub_config)
    return fig


def generate_profit_graph(
    pairs: List[str],
    data: Dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    timeframe: str,
    stake_currency: str,
    starting_balance: float,
) -> go.Figure:
    try:
        df_comb: pd.DataFrame = combine_dataframes_with_mean(data, 'close')
    except ValueError:
        raise OperationalException('No data found. Please make sure that data is available for the timerange and pairs selected.')
    trades = extract_trades_of_period(df_comb, trades, date_index=True)
    if len(trades) == 0:
        raise OperationalException('No trades found in selected timerange.')
    df_comb = create_cum_profit(df_comb, trades, 'cum_profit', timeframe)
    avgclose = go.Scatter(x=df_comb.index, y=df_comb['mean'], name='Avg close price')
    fig: go.Figure = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        row_heights=[1, 1, 1, 0.5, 0.75, 0.75],
        vertical_spacing=0.05,
        subplot_titles=[
            'AVG Close Price',
            'Combined Profit',
            'Profit per pair',
            'Parallelism',
            'Underwater',
            'Relative Drawdown'
        ]
    )
    fig['layout'].update(title='Freqtrade Profit plot')
    fig['layout']['yaxis1'].update(title='Price')
    fig['layout']['yaxis2'].update(title=f'Profit {stake_currency}')
    fig['layout']['yaxis3'].update(title=f'Profit {stake_currency}')
    fig['layout']['yaxis4'].update(title='Trade count')
    fig['layout']['yaxis5'].update(title='Underwater Plot')
    fig['layout']['yaxis6'].update(title='Underwater Plot Relative (%)', tickformat=',.2%')
    fig['layout']['xaxis']['rangeslider'].update(visible=False)
    fig.update_layout(modebar_add=['v1hovermode', 'toggleSpikeLines'])
    fig.add_trace(avgclose, 1, 1)
    fig = add_profit(fig, 2, df_comb, 'cum_profit', 'Profit')
    fig = add_max_drawdown(fig, 2, trades, df_comb, timeframe, starting_balance)
    fig = add_parallelism(fig, 4, trades, timeframe)
    fig = add_underwater(fig, 5, trades, starting_balance)
    for pair in pairs:
        profit_col: str = f'cum_profit_{pair}'
        try:
            df_comb = create_cum_profit(df_comb, trades[trades['pair'] == pair], profit_col, timeframe)
            fig = add_profit(fig, 3, df_comb, profit_col, f'Profit {pair}')
        except ValueError:
            pass
    return fig


def generate_plot_filename(pair: str, timeframe: str) -> str:
    """
    Generate filenames per pair/timeframe to be used for storing plots
    """
    pair_s: str = pair_to_filename(pair)
    file_name: str = 'freqtrade-plot-' + pair_s + '-' + timeframe + '.html'
    logger.info('Generate plot file for %s', pair)
    return file_name


def store_plot_file(fig: go.Figure, filename: str, directory: Path, auto_open: bool = False) -> None:
    """
    Generate a plot html file from pre populated fig plotly object
    :param fig: Plotly Figure to plot
    :param filename: Name to store the file as
    :param directory: Directory to store the file in
    :param auto_open: Automatically open files saved
    :return: None
    """
    directory.mkdir(parents=True, exist_ok=True)
    _filename: Path = directory.joinpath(filename)
    plot(fig, filename=str(_filename), auto_open=auto_open)
    logger.info(f'Stored plot as {_filename}')


def load_and_plot_trades(config: Dict[str, Any]) -> None:
    """
    From configuration provided
    - Initializes plot-script
    - Get candle (OHLCV) data
    - Generate Dafaframes populated with indicators and signals based on configured strategy
    - Load trades executed during the selected period
    - Generate Plotly plot objects
    - Generate plot files
    :return: None
    """
    strategy: IStrategy = StrategyResolver.load_strategy(config)
    exchange = ExchangeResolver.load_exchange(config)
    IStrategy.dp = DataProvider(config, exchange)
    strategy.ft_bot_start()
    strategy_safe_wrapper(strategy.bot_loop_start)(current_time=datetime.now(timezone.utc))
    plot_elements: Dict[str, Any] = init_plotscript(config, list(exchange.markets), strategy.startup_candle_count)
    timerange: TimeRange = plot_elements['timerange']
    trades: pd.DataFrame = plot_elements['trades']
    pair_counter: int = 0
    for pair, data in plot_elements['ohlcv'].items():
        pair_counter += 1
        logger.info('analyse pair %s', pair)
        df_analyzed: pd.DataFrame = strategy.analyze_ticker(data, {'pair': pair})
        df_analyzed = trim_dataframe(df_analyzed, timerange)
        if not trades.empty:
            trades_pair: pd.DataFrame = trades.loc[trades['pair'] == pair]
            trades_pair = extract_trades_of_period(df_analyzed, trades_pair)
        else:
            trades_pair = trades
        fig: go.Figure = generate_candlestick_graph(
            pair=pair,
            data=df_analyzed,
            trades=trades_pair,
            indicators1=config.get('indicators1', []),
            indicators2=config.get('indicators2', []),
            plot_config=strategy.plot_config if hasattr(strategy, 'plot_config') else {}
        )
        store_plot_file(
            fig,
            filename=generate_plot_filename(pair, config['timeframe']),
            directory=config['user_data_dir'] / 'plot'
        )
    logger.info('End of plotting process. %s plots generated', pair_counter)


def plot_profit(config: Dict[str, Any]) -> None:
    """
    Plots the total profit for all pairs.
    Note, the profit calculation isn't realistic.
    But should be somewhat proportional, and therefore useful
    in helping out to find a good algorithm.
    """
    if 'timeframe' not in config:
        raise OperationalException('Timeframe must be set in either config or via --timeframe.')
    exchange = ExchangeResolver.load_exchange(config)
    plot_elements: Dict[str, Any] = init_plotscript(config, list(exchange.markets))
    trades: pd.DataFrame = plot_elements['trades']
    trades = trades[trades['pair'].isin(plot_elements['pairs']) & ~trades['close_date'].isnull()]
    if len(trades) == 0:
        raise OperationalException('No trades found, cannot generate Profit-plot without trades from either Backtest result or database.')
    fig: go.Figure = generate_profit_graph(
        plot_elements['pairs'],
        plot_elements['ohlcv'],
        trades,
        config['timeframe'],
        config.get('stake_currency', ''),
        config.get('available_capital', get_dry_run_wallet(config))
    )
    store_plot_file(fig, filename='freqtrade-profit-plot.html', directory=config['user_data_dir'] / 'plot', auto_open=config.get('plot_auto_open', False))