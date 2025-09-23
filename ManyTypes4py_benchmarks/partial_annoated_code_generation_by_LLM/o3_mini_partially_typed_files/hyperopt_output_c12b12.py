import sys
from os import get_terminal_size
from typing import Any, Optional, List, Dict, Union
from rich.align import Align
from rich.console import Console
from rich.table import Table
from rich.text import Text
from freqtrade.constants import Config
from freqtrade.optimize.optimize_reports import generate_wins_draws_losses
from freqtrade.util import fmt_coin

class HyperoptOutput:
    _results: List[Any]
    _streaming: bool
    table: Table

    def __init__(self, streaming: bool = False) -> None:
        self._results = []
        self._streaming = streaming
        self.__init_table()

    def __call__(self, *args: Any, **kwds: Any) -> Align:
        return Align.center(self.table)

    def __init_table(self) -> None:
        """Initialize table"""
        self.table = Table(title='Hyperopt results')
        self.table.add_column('Best', justify='left')
        self.table.add_column('Epoch', justify='right')
        self.table.add_column('Trades', justify='right')
        self.table.add_column('Win  Draw  Loss  Win%', justify='right')
        self.table.add_column('Avg profit', justify='right')
        self.table.add_column('Profit', justify='right')
        self.table.add_column('Avg duration', justify='right')
        self.table.add_column('Objective', justify='right')
        self.table.add_column('Max Drawdown (Acct)', justify='right')

    def print(self, console: Optional[Console] = None, *, print_colorized: bool = True) -> None:
        if not console:
            console = Console(color_system='auto' if print_colorized else None,
                              width=200 if 'pytest' in sys.modules else None)
        console.print(self.table)

    def add_data(self, config: Config, results: List[Dict[str, Any]], total_epochs: int, highlight_best: bool) -> None:
        """Format one or multiple rows and add them"""
        stake_currency = config['stake_currency']
        self._results.extend(results)
        max_rows: Optional[int] = None
        if self._streaming:
            try:
                ts = get_terminal_size()
                if ts.columns < 148:
                    max_rows = -(int(ts.lines / 2) - 6)
                else:
                    max_rows = -(ts.lines - 6)
            except OSError:
                pass
        self.__init_table()
        for r in self._results[max_rows:]:
            best_prefix: str = ''
            if r.get('is_initial_point', False) or r.get('is_random', False):
                best_prefix += '*'
            if r.get('is_best', False):
                best_prefix += ' Best'
            best_cell: str = best_prefix.lstrip()

            epoch_cell: str = f"{r['current_epoch']}/{total_epochs}"
            trades_cell: str = str(r['results_metrics']['total_trades'])
            win_draw_loss_cell: str = generate_wins_draws_losses(
                r['results_metrics']['wins'],
                r['results_metrics']['draws'],
                r['results_metrics']['losses']
            )
            profit_mean_value = r['results_metrics'].get('profit_mean')
            avg_profit_cell: str = f"{profit_mean_value:.2%}" if profit_mean_value is not None else '--'
            
            profit_total_abs = r['results_metrics'].get('profit_total_abs', 0)
            profit_total = r['results_metrics'].get('profit_total')
            if profit_total_abs != 0.0 and profit_total is not None:
                profit_text = '{} {}'.format(
                    fmt_coin(profit_total_abs, stake_currency, keep_trailing_zeros=True),
                    f"({profit_total:,.2%})".rjust(10, ' ')
                )
                # Choose style only if not the best
                profit_style: str = ''
                if not r.get('is_best', False):
                    profit_style = 'green' if profit_total_abs > 0 else 'red'
                profit_cell: Union[Text, str] = Text(profit_text, style=profit_style)
            else:
                profit_cell = '--'

            holding_avg_cell: str = str(r['results_metrics']['holding_avg'])
            loss_val = r['loss']
            objective_cell: str = f"{loss_val:,.5f}" if loss_val != 100000 else 'N/A'
            max_drawdown_account = r['results_metrics']['max_drawdown_account']
            if max_drawdown_account != 0.0:
                max_drawdown_text = '{} {}'.format(
                    fmt_coin(r['results_metrics']['max_drawdown_abs'], stake_currency, keep_trailing_zeros=True),
                    f"({max_drawdown_account:,.2%})".rjust(10, ' ')
                )
                max_drawdown_cell: str = max_drawdown_text
            else:
                max_drawdown_cell = '--'

            style_list: List[str] = []
            if r.get('is_best', False) and highlight_best:
                style_list.append('bold gold1')
            if r.get('is_initial_point', False):
                style_list.append('italic')
            row_style: str = ' '.join(style_list)

            self.table.add_row(
                best_cell,
                epoch_cell,
                trades_cell,
                win_draw_loss_cell,
                avg_profit_cell,
                profit_cell,
                holding_avg_cell,
                f"{r['loss']:,.5f}" if r['loss'] != 100000 else 'N/A',
                max_drawdown_cell,
                style=row_style
            )