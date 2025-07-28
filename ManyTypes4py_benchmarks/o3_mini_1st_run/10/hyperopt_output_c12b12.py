import sys
from os import get_terminal_size
from typing import Any, Optional, List, Dict
from rich.align import Align
from rich.console import Console
from rich.table import Table
from rich.text import Text
from freqtrade.constants import Config
from freqtrade.optimize.optimize_reports import generate_wins_draws_losses
from freqtrade.util import fmt_coin

class HyperoptOutput:
    def __init__(self, streaming: bool = False) -> None:
        self._results: List[Dict[str, Any]] = []
        self._streaming: bool = streaming
        self.__init_table()

    def __call__(self, *args: Any, **kwds: Any) -> Align:
        return Align.center(self.table)

    def __init_table(self) -> None:
        """Initialize table"""
        self.table: Table = Table(title='Hyperopt results')
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
            console = Console(
                color_system='auto' if print_colorized else None,
                width=200 if 'pytest' in sys.modules else None
            )
        console.print(self.table)

    def add_data(self, config: Config, results: List[Dict[str, Any]], total_epochs: int, highlight_best: bool) -> None:
        stake_currency: Any = config['stake_currency']
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
            best_text: str = (('*' if r['is_initial_point'] or r['is_random'] else '') +
                              (' Best' if r['is_best'] else '')).lstrip()
            epoch_text: str = f"{r['current_epoch']}/{total_epochs}"
            trades_text: str = str(r['results_metrics']['total_trades'])
            win_draw_loss: str = generate_wins_draws_losses(
                r['results_metrics']['wins'],
                r['results_metrics']['draws'],
                r['results_metrics']['losses']
            )
            avg_profit: str = f"{r['results_metrics']['profit_mean']:.2%}" if r['results_metrics']['profit_mean'] is not None else '--'
            if r['results_metrics'].get('profit_total_abs', 0) != 0.0:
                profit_total_str = f"({r['results_metrics']['profit_total']:,.2%})".rjust(10, ' ')
                profit_text = Text(
                    '{} {}'.format(
                        fmt_coin(r['results_metrics']['profit_total_abs'], stake_currency, keep_trailing_zeros=True),
                        profit_total_str
                    ),
                    style=('green' if r['results_metrics'].get('profit_total_abs', 0) > 0 else 'red') if not r['is_best'] else ''
                )
            else:
                profit_text = Text('--')
            avg_duration: str = str(r['results_metrics']['holding_avg'])
            objective: str = f"{r['loss']:,.5f}" if r['loss'] != 100000 else 'N/A'
            if r['results_metrics']['max_drawdown_account'] != 0.0:
                max_drawdown_total = fmt_coin(r['results_metrics']['max_drawdown_abs'], stake_currency, keep_trailing_zeros=True)
                max_drawdown_percent = f"({r['results_metrics']['max_drawdown_account']:,.2%})".rjust(10, ' ')
                max_drawdown: str = '{} {}'.format(max_drawdown_total, max_drawdown_percent)
            else:
                max_drawdown = '--'
            style_list: List[str] = []
            if r['is_best'] and highlight_best:
                style_list.append('bold gold1')
            if r['is_initial_point']:
                style_list.append('italic')
            row_style: str = ' '.join(style_list)

            self.table.add_row(
                best_text,
                epoch_text,
                trades_text,
                win_draw_loss,
                avg_profit,
                profit_text,
                avg_duration,
                objective,
                max_drawdown,
                style=row_style
            )