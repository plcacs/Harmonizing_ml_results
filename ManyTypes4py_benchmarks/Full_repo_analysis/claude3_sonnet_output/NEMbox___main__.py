import _curses
import argparse
import curses
import sys
import traceback
from typing import Optional
from . import __version__
from .menu import Menu

def start() -> None:
    version: str = __version__
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', help='show this version and exit', action='store_true')
    args: argparse.Namespace = parser.parse_args()
    if args.version:
        latest: Optional[str] = Menu().check_version()
        try:
            curses.endwin()
        except _curses.error:
            pass
        print('NetEase-MusicBox installed version:' + version)
        if latest != version:
            print('NetEase-MusicBox latest version:' + str(latest))
        sys.exit()
    nembox_menu: Menu = Menu()
    try:
        nembox_menu.start_fork(version)
    except (OSError, TypeError, ValueError, KeyError, IndexError):
        try:
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        except _curses.error:
            pass
        traceback.print_exc()

if __name__ == '__main__':
    start()
