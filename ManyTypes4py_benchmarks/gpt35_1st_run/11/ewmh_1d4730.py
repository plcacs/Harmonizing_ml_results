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
    
    def setActiveWindow(self, win: display.Window) -> None:
    
    def setShowingDesktop(self, show: int) -> None:
    
    def setCloseWindow(self, win: display.Window) -> None:
    
    def setWmName(self, win: display.Window, name: str) -> None:
    
    def setWmVisibleName(self, win: display.Window, name: str) -> None:
    
    def setWmDesktop(self, win: display.Window, i: int) -> None:
    
    def setMoveResizeWindow(self, win: display.Window, gravity: int = 0, x: int = None, y: int = None, w: int = None, h: int = None) -> None:
    
    def setWmState(self, win: display.Window, action: int, state: Union[int, str], state2: Union[int, str] = 0) -> None:
    
    def getClientList(self) -> List[display.Window]:
    
    def getClientListStacking(self) -> List[display.Window]:
    
    def getNumberOfDesktops(self) -> int:
    
    def getDesktopGeometry(self) -> List[int]:
    
    def getDesktopViewPort(self) -> List[List[int]]:
    
    def getCurrentDesktop(self) -> int:
    
    def getActiveWindow(self) -> Union[display.Window, None]:
    
    def getWorkArea(self) -> List[List[int]]:
    
    def getShowingDesktop(self) -> int:
    
    def getWmName(self, win: display.Window) -> str:
    
    def getWmVisibleName(self, win: display.Window) -> str:
    
    def getWmDesktop(self, win: display.Window) -> Union[int, None]:
    
    def getWmWindowType(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
    
    def getWmState(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
    
    def getWmAllowedActions(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
    
    def getWmPid(self, win: display.Window) -> Union[int, None]:
    
    def _getProperty(self, _type: str, win: display.Window = None) -> Union[List[int], List[str]]:
    
    def _setProperty(self, _type: str, data: Union[str, List[int]], win: display.Window = None, mask = None) -> None:
    
    def _getAtomName(self, atom: int) -> str:
    
    def _createWindow(self, wId: int) -> Union[display.Window, None]:
    
    def getReadableProperties(self) -> List[str]:
    
    def getProperty(self, prop: str, *args, **kwargs) -> Union[List[int], List[str]]:
    
    def getWritableProperties(self) -> List[str]:
    
    def setProperty(self, prop: str, *args, **kwargs) -> None:
