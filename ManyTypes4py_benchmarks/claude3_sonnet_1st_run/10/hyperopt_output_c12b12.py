import sys
from os import get_terminal_size
from typing import Any, Dict, List, Optional, Union
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
            console = Console(color_system='auto' if print_colorized else None, width=200 if 'pytest' in sys.modules else None)
        console.print(self.table)

    def add_data(self, config: Config, results: List[Dict[str, Any]], total_epochs: int, highlight_best: bool) -> None:
        """Format one or multiple rows and add them"""
        stake_currency: str = config['stake_currency']
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
            self.table.add_row(*[(('*' if r['is_initial_point'] or r['is_random'] else '') + (' Best' if r['is_best'] else '')).lstrip(), 
                               f'{r["current_epoch"]}/{total_epochs}', 
                               str(r['results_metrics']['total_trades']), 
                               generate_wins_draws_losses(r['results_metrics']['wins'], r['results_metrics']['draws'], r['results_metrics']['losses']), 
                               f'{r["results_metrics"]["profit_mean"]:.2%}' if r['results_metrics']['profit_mean'] is not None else '--', 
                               Text('{} {}'.format(fmt_coin(r['results_metrics']['profit_total_abs'], stake_currency, keep_trailing_zeros=True), 
                                                  f'({r["results_metrics"]["profit_total"]:,.2%})'.rjust(10, ' ')) 
                                    if r['results_metrics'].get('profit_total_abs', 0) != 0.0 else '--', 
                                    style=('green' if r['results_metrics'].get('profit_total_abs', 0) > 0 else 'red') if not r['is_best'] else ''), 
                               str(r['results_metrics']['holding_avg']), 
                               f'{r["loss"]:,.5f}' if r['loss'] != 100000 else 'N/A', 
                               '{} {}'.format(fmt_coin(r['results_metrics']['max_drawdown_abs'], stake_currency, keep_trailing_zeros=True), 
                                             f'({r["results_metrics"]["max_drawdown_account"]:,.2%})'.rjust(10, ' ')) 
                               if r['results_metrics']['max_drawdown_account'] != 0.0 else '--'], 
                              style=' '.join(['bold gold1' if r['is_best'] and highlight_best else '', 'italic ' if r['is_initial_point'] else '']))
