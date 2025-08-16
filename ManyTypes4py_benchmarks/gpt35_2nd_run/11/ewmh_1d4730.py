from Xlib import display, X, protocol
import time
from typing import List, Tuple, Union

class EWMH:
    NET_WM_WINDOW_TYPES: Tuple[str, ...]
    NET_WM_ACTIONS: Tuple[str, ...]
    NET_WM_STATES: Tuple[str, ...]

    def __init__(self, _display: display.Display = None, root = None):
        self.display: display.Display
        self.root: display.Window
        self.__getAttrs: dict
        self.__setAttrs: dict

    def setNumberOfDesktops(self, nb: int) -> None:
    
    def setDesktopGeometry(self, w: int, h: int) -> None:
    
    def setDesktopViewport(self, w: int, h: int) -> None:
    
    def setCurrentDesktop(self, i: int) -> None:
    
    def setActiveWindow(self, win) -> None:
    
    def setShowingDesktop(self, show: int) -> None:
    
    def setCloseWindow(self, win) -> None:
    
    def setWmName(self, win, name: str) -> None:
    
    def setWmVisibleName(self, win, name: str) -> None:
    
    def setWmDesktop(self, win, i: int) -> None:
    
    def setMoveResizeWindow(self, win, gravity: int = 0, x: int = None, y: int = None, w: int = None, h: int = None) -> None:
    
    def setWmState(self, win, action: int, state: Union[int, str], state2: int = 0) -> None:
    
    def getClientList(self) -> List[display.Window]:
    
    def getClientListStacking(self) -> List[display.Window]:
    
    def getNumberOfDesktops(self) -> int:
    
    def getDesktopGeometry(self) -> List[int]:
    
    def getDesktopViewPort(self) -> List[List[int]]:
    
    def getCurrentDesktop(self) -> int:
    
    def getActiveWindow(self) -> display.Window:
    
    def getWorkArea(self) -> List[List[int]]:
    
    def getShowingDesktop(self) -> int:
    
    def getWmName(self, win) -> str:
    
    def getWmVisibleName(self, win) -> str:
    
    def getWmDesktop(self, win) -> int:
    
    def getWmWindowType(self, win, str: bool = False) -> List[Union[int, str]]:
    
    def getWmState(self, win, str: bool = False) -> List[Union[int, str]]:
    
    def getWmAllowedActions(self, win, str: bool = False) -> List[Union[int, str]]:
    
    def getWmPid(self, win) -> int:
    
    def _getProperty(self, _type: str, win = None):
    
    def _setProperty(self, _type: str, data, win = None, mask = None):
    
    def _getAtomName(self, atom) -> str:
    
    def _createWindow(self, wId) -> display.Window:
    
    def getReadableProperties(self) -> List[str]:
    
    def getProperty(self, prop: str, *args, **kwargs):
    
    def getWritableProperties(self) -> List[str]:
    
    def setProperty(self, prop: str, *args, **kwargs):
