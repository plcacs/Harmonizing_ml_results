from typing import Dict, List
import plotly.graph_objects as go
from freqtrade.configuration import TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import get_dry_run_wallet

def init_plotscript(config: Dict, markets: List[str], startup_candles: int = 0) -> Dict:
    ...

def add_indicators(fig: go.Figure, row: int, indicators: Dict[str, Dict], data: pd.DataFrame) -> go.Figure:
    ...

def add_profit(fig: go.Figure, row: int, data: pd.DataFrame, column: str, name: str) -> go.Figure:
    ...

def add_max_drawdown(fig: go.Figure, row: int, trades: pd.DataFrame, df_comb: pd.DataFrame, timeframe: str, starting_balance: float) -> go.Figure:
    ...

def add_underwater(fig: go.Figure, row: int, trades: pd.DataFrame, starting_balance: float) -> go.Figure:
    ...

def add_parallelism(fig: go.Figure, row: int, trades: pd.DataFrame, timeframe: str) -> go.Figure:
    ...

def plot_trades(fig: go.Figure, trades: pd.DataFrame) -> go.Figure:
    ...

def create_plotconfig(indicators1: List[str], indicators2: List[str], plot_config: Dict) -> Dict:
    ...

def plot_area(fig: go.Figure, row: int, data: pd.DataFrame, indicator_a: str, indicator_b: str, label: str = '', fill_color: str = 'rgba(0,176,246,0.2)') -> go.Figure:
    ...

def add_areas(fig: go.Figure, row: int, data: pd.DataFrame, indicators: Dict[str, Dict]) -> go.Figure:
    ...

def create_scatter(data: pd.DataFrame, column_name: str, color: str, direction: str) -> go.Scatter:
    ...

def generate_candlestick_graph(pair: str, data: pd.DataFrame, trades: pd.DataFrame = None, indicators1: List[str] = None, indicators2: List[str] = None, plot_config: Dict = None) -> go.Figure:
    ...

def generate_profit_graph(pairs: List[str], data: pd.DataFrame, trades: pd.DataFrame, timeframe: str, stake_currency: str, starting_balance: float) -> go.Figure:
    ...

def generate_plot_filename(pair: str, timeframe: str) -> str:
    ...

def store_plot_file(fig: go.Figure, filename: str, directory: Path, auto_open: bool = False) -> None:
    ...

def load_and_plot_trades(config: Dict) -> None:
    ...

def plot_profit(config: Dict) -> None:
    ...
