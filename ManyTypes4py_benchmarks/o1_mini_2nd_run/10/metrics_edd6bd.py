import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_market_change(data: Dict[str, pd.DataFrame], column: str = 'close') -> float:
    """
    Calculate market change based on "column".
    Calculation is done by taking the first non-null and the last non-null element of each column
    and calculating the pctchange as "(last - first) / first".
    Then the results per pair are combined as mean.

    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return:
    """
    tmp_means: list[float] = []
    for pair, df in data.items():
        start: float = df[column].dropna().iloc[0]
        end: float = df[column].dropna().iloc[-1]
        tmp_means.append((end - start) / start)
    return float(np.mean(tmp_means))

def combine_dataframes_by_column(data: Dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key.
    :raise: ValueError if no data is provided.
    """
    if not data:
        raise ValueError('No data provided.')
    df_comb = pd.concat(
        [
            data[pair].set_index('date').rename({column: pair}, axis=1)[pair]
            for pair in data
        ],
        axis=1
    )
    return df_comb

def combined_dataframes_with_rel_mean(
    data: Dict[str, pd.DataFrame],
    fromdt: datetime,
    todt: datetime,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    :raise: ValueError if no data is provided.
    """
    df_comb: pd.DataFrame = combine_dataframes_by_column(data, column)
    df_comb = df_comb.iloc[
        (df_comb.index >= fromdt) & (df_comb.index < todt)
    ]
    df_comb['count'] = df_comb.count(axis=1)
    df_comb['mean'] = df_comb.mean(axis=1)
    df_comb['rel_mean'] = df_comb['mean'].pct_change().fillna(0).cumsum()
    return df_comb[['mean', 'rel_mean', 'count']]

def combine_dataframes_with_mean(data: Dict[str, pd.DataFrame], column: str = 'close') -> pd.DataFrame:
    """
    Combine multiple dataframes "column"
    :param data: Dict of Dataframes, dict key should be pair.
    :param column: Column in the original dataframes to use
    :return: DataFrame with the column renamed to the dict key, and a column
        named mean, containing the mean of all pairs.
    :raise: ValueError if no data is provided.
    """
    df_comb: pd.DataFrame = combine_dataframes_by_column(data, column)
    df_comb['mean'] = df_comb.mean(axis=1)
    return df_comb

def create_cum_profit(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    col_name: str,
    timeframe: str
) -> pd.DataFrame:
    """
    Adds a column `col_name` with the cumulative profit for the given trades array.
    :param df: DataFrame with date index
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :param col_name: Column name that will be assigned the results
    :param timeframe: Timeframe used during the operations
    :return: Returns df with one additional column, col_name, containing the cumulative profit.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError('Trade dataframe empty.')
    from freqtrade.exchange import timeframe_to_resample_freq
    timeframe_freq: str = timeframe_to_resample_freq(timeframe)
    _trades_sum: pd.DataFrame = trades.resample(timeframe_freq, on='close_date')[['profit_abs']].sum()
    df.loc[:, col_name] = _trades_sum['profit_abs'].cumsum()
    df.loc[df.iloc[0].name, col_name] = 0
    df[col_name] = df[col_name].ffill()
    return df

def _calc_drawdown_series(
    profit_results: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
    starting_balance: float
) -> pd.DataFrame:
    max_drawdown_df: pd.DataFrame = pd.DataFrame()
    max_drawdown_df['cumulative'] = profit_results[value_col].cumsum()
    max_drawdown_df['high_value'] = max_drawdown_df['cumulative'].cummax()
    max_drawdown_df['drawdown'] = max_drawdown_df['cumulative'] - max_drawdown_df['high_value']
    max_drawdown_df['date'] = profit_results.loc[:, date_col]
    if starting_balance:
        cumulative_balance: pd.Series = starting_balance + max_drawdown_df['cumulative']
        max_balance: pd.Series = starting_balance + max_drawdown_df['high_value']
        max_drawdown_df['drawdown_relative'] = (max_balance - cumulative_balance) / max_balance
    else:
        max_drawdown_df['drawdown_relative'] = (
            (max_drawdown_df['high_value'] - max_drawdown_df['cumulative'])
            / max_drawdown_df['high_value']
        )
    return max_drawdown_df

def calculate_underwater(
    trades: pd.DataFrame,
    *,
    date_col: str = 'close_date',
    value_col: str = 'profit_ratio',
    starting_balance: float = 0.0
) -> pd.DataFrame:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_ratio')
    :return: Tuple (float, highdate, lowdate, highvalue, lowvalue) with absolute max drawdown,
             high and low time and high and low value.
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError('Trade dataframe empty.')
    profit_results: pd.DataFrame = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df: pd.DataFrame = _calc_drawdown_series(
        profit_results,
        date_col=date_col,
        value_col=value_col,
        starting_balance=starting_balance
    )
    return max_drawdown_df

@dataclass
class DrawDownResult:
    drawdown_abs: float = 0.0
    high_date: Optional[datetime] = None
    low_date: Optional[datetime] = None
    high_value: float = 0.0
    low_value: float = 0.0
    relative_account_drawdown: float = 0.0

def calculate_max_drawdown(
    trades: pd.DataFrame,
    *,
    date_col: str = 'close_date',
    value_col: str = 'profit_abs',
    starting_balance: float = 0.0,
    relative: bool = False
) -> DrawDownResult:
    """
    Calculate max drawdown and the corresponding close dates
    :param trades: DataFrame containing trades (requires columns close_date and profit_ratio)
    :param date_col: Column in DataFrame to use for dates (defaults to 'close_date')
    :param value_col: Column in DataFrame to use for values (defaults to 'profit_abs')
    :param starting_balance: Portfolio starting balance - properly calculate relative drawdown.
    :return: DrawDownResult object
             with absolute max drawdown, high and low time and high and low value,
             and the relative account drawdown
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError('Trade dataframe empty.')
    profit_results: pd.DataFrame = trades.sort_values(date_col).reset_index(drop=True)
    max_drawdown_df: pd.DataFrame = _calc_drawdown_series(
        profit_results,
        date_col=date_col,
        value_col=value_col,
        starting_balance=starting_balance
    )
    if relative:
        idxmin: int = max_drawdown_df['drawdown_relative'].idxmax()
    else:
        idxmin: int = max_drawdown_df['drawdown'].idxmin()
    if idxmin == 0:
        raise ValueError('No losing trade, therefore no drawdown.')
    high_date: datetime = profit_results.loc[
        max_drawdown_df.iloc[:idxmin]['high_value'].idxmax(),
        date_col
    ]
    low_date: datetime = profit_results.loc[idxmin, date_col]
    high_val: float = max_drawdown_df.loc[idxmin, 'cumulative']
    low_val: float = max_drawdown_df.loc[idxmin, 'cumulative']
    max_drawdown_rel: float = max_drawdown_df.loc[idxmin, 'drawdown_relative']
    return DrawDownResult(
        drawdown_abs=abs(max_drawdown_df.loc[idxmin, 'drawdown']),
        high_date=high_date,
        low_date=low_date,
        high_value=high_val,
        low_value=low_val,
        relative_account_drawdown=max_drawdown_rel
    )

def calculate_csum(
    trades: pd.DataFrame,
    starting_balance: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate min/max cumsum of trades, to show if the wallet/stake amount ratio is sane
    :param trades: DataFrame containing trades (requires columns close_date and profit_percent)
    :param starting_balance: Add starting balance to results, to show the wallets high / low points
    :return: Tuple (float, float) with cumsum of profit_abs
    :raise: ValueError if trade-dataframe was found empty.
    """
    if len(trades) == 0:
        raise ValueError('Trade dataframe empty.')
    csum_df: pd.DataFrame = pd.DataFrame()
    csum_df['sum'] = trades['profit_abs'].cumsum()
    csum_min: float = csum_df['sum'].min() + starting_balance
    csum_max: float = csum_df['sum'].max() + starting_balance
    return (csum_min, csum_max)

def calculate_cagr(
    days_passed: int,
    starting_balance: float,
    final_balance: float
) -> float:
    """
    Calculate CAGR
    :param days_passed: Days passed between start and ending balance
    :param starting_balance: Starting balance
    :param final_balance: Final balance to calculate CAGR against
    :return: CAGR
    """
    if final_balance < 0:
        return 0.0
    return (final_balance / starting_balance) ** (1 / (days_passed / 365)) - 1

def calculate_expectancy(trades: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate expectancy
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: expectancy, expectancy_ratio
    """
    expectancy: float = 0.0
    expectancy_ratio: float = 100.0
    if len(trades) > 0:
        winning_trades: pd.DataFrame = trades.loc[trades['profit_abs'] > 0]
        losing_trades: pd.DataFrame = trades.loc[trades['profit_abs'] < 0]
        profit_sum: float = winning_trades['profit_abs'].sum()
        loss_sum: float = abs(losing_trades['profit_abs'].sum())
        nb_win_trades: int = len(winning_trades)
        nb_loss_trades: int = len(losing_trades)
        average_win: float = profit_sum / nb_win_trades if nb_win_trades > 0 else 0.0
        average_loss: float = loss_sum / nb_loss_trades if nb_loss_trades > 0 else 0.0
        winrate: float = nb_win_trades / len(trades)
        loserate: float = nb_loss_trades / len(trades)
        expectancy = winrate * average_win - loserate * average_loss
        if average_loss > 0:
            risk_reward_ratio: float = average_win / average_loss
            expectancy_ratio = (1 + risk_reward_ratio) * winrate - 1
    return (expectancy, expectancy_ratio)

def calculate_sortino(
    trades: pd.DataFrame,
    min_date: Optional[datetime],
    max_date: Optional[datetime],
    starting_balance: float
) -> float:
    """
    Calculate sortino
    :param trades: DataFrame containing trades (requires columns profit_abs)
    :return: sortino
    """
    if len(trades) == 0 or min_date is None or max_date is None or (min_date == max_date):
        return 0.0
    total_profit: pd.Series = trades['profit_abs'] / starting_balance
    days_period: int = max(1, (max_date - min_date).days)
    expected_returns_mean: float = total_profit.sum() / days_period
    down_stdev: float = np.std(trades.loc[trades['profit_abs'] < 0, 'profit_abs'] / starting_balance)
    if down_stdev != 0 and (not np.isnan(down_stdev)):
        sortino_ratio: float = expected_returns_mean / down_stdev * math.sqrt(365)
    else:
        sortino_ratio = -100.0
    return sortino_ratio

def calculate_sharpe(
    trades: pd.DataFrame,
    min_date: Optional[datetime],
    max_date: Optional[datetime],
    starting_balance: float
) -> float:
    """
    Calculate sharpe
    :param trades: DataFrame containing trades (requires column profit_abs)
    :return: sharpe
    """
    if len(trades) == 0 or min_date is None or max_date is None or (min_date == max_date):
        return 0.0
    total_profit: pd.Series = trades['profit_abs'] / starting_balance
    days_period: int = max(1, (max_date - min_date).days)
    expected_returns_mean: float = total_profit.sum() / days_period
    up_stdev: float = np.std(total_profit)
    if up_stdev != 0:
        sharp_ratio: float = expected_returns_mean / up_stdev * math.sqrt(365)
    else:
        sharp_ratio = -100.0
    return sharp_ratio

def calculate_calmar(
    trades: pd.DataFrame,
    min_date: Optional[datetime],
    max_date: Optional[datetime],
    starting_balance: float
) -> float:
    """
    Calculate calmar
    :param trades: DataFrame containing trades (requires columns close_date and profit_abs)
    :return: calmar
    """
    if len(trades) == 0 or min_date is None or max_date is None or (min_date == max_date):
        return 0.0
    total_profit: float = trades['profit_abs'].sum() / starting_balance
    days_period: int = max(1, (max_date - min_date).days)
    expected_returns_mean: float = total_profit / days_period * 100
    try:
        drawdown: pd.DataFrame = calculate_max_drawdown(
            trades,
            value_col='profit_abs',
            starting_balance=starting_balance
        )
        max_drawdown: float = drawdown.relative_account_drawdown
    except ValueError:
        max_drawdown = 0.0
    if max_drawdown != 0:
        calmar_ratio: float = expected_returns_mean / max_drawdown * math.sqrt(365)
    else:
        calmar_ratio = -100.0
    return calmar_ratio
