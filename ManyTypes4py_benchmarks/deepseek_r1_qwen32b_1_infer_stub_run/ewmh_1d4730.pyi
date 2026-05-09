from Xlib import display, X
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)
import time

class Window:
    id: int

class EWMH:
    NET_WM_WINDOW_TYPES: Tuple[str, ...]
    NET_WM_ACTIONS: Tuple[str, ...]
    NET_WM_STATES: Tuple[str, ...]

    def __init__(self, _display: Optional[display.Display] = None, root: Optional['Window'] = None) -> None:
        ...

    def setNumberOfDesktops(self, nb: int) -> None:
        ...

    def setDesktopGeometry(self, w: int, h: int) -> None:
        ...

    def setDesktopViewport(self, w: int, h: int) -> None:
        ...

    def setCurrentDesktop(self, i: int) -> None:
        ...

    def setActiveWindow(self, win: 'Window') -> None:
        ...

    def setShowingDesktop(self, show: int) -> None:
        ...

    def setCloseWindow(self, win: 'Window') -> None:
        ...

    def setWmName(self, win: 'Window', name: str) -> None:
        ...

    def setWmVisibleName(self, win: 'Window', name: str) -> None:
        ...

    def setWmDesktop(self, win: 'Window', i: int) -> None:
        ...

    def setMoveResizeWindow(
        self,
        win: 'Window',
        gravity: int = 0,
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ) -> None:
        ...

    def setWmState(
        self,
        win: 'Window',
        action: int,
        state: Union[int, str],
        state2: Union[int, str, int] = 0,
    ) -> None:
        ...

    def getClientList(self) -> List[Optional['Window']]:
        ...

    def getClientListStacking(self) -> List[Optional['Window']]:
        ...

    def getNumberOfDesktops(self) -> int:
        ...

    def getDesktopGeometry(self) -> List[int]:
        ...

    def getDesktopViewPort(self) -> List[List[int]]:
        ...

    def getCurrentDesktop(self) -> int:
        ...

    def getActiveWindow(self) -> Optional['Window']:
        ...

    def getWorkArea(self) -> List[List[int]]:
        ...

    def getShowingDesktop(self) -> int:
        ...

    def getWmName(self, win: 'Window') -> str:
        ...

    def getWmVisibleName(self, win: 'Window') -> str:
        ...

    def getWmDesktop(self, win: 'Window') -> Optional[int]:
        ...

    def getWmWindowType(self, win: 'Window', str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmState(self, win: 'Window', str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmAllowedActions(self, win: 'Window', str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmPid(self, win: 'Window') -> Optional[int]:
        ...

    def _getProperty(self, _type: str, win: Optional['Window'] = None) -> Any:
        ...

    def _setProperty(
        self,
        _type: str,
        data: Union[str, List[Any]],
        win: Optional['Window'] = None,
        mask: Optional[int] = None,
    ) -> None:
        ...

    def _getAtomName(self, atom: int) -> str:
        ...

    def _createWindow(self, wId: int) -> Optional['Window']:
        ...

    def getReadableProperties(self) -> List[str]:
        ...

    def getProperty(self, prop: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def getWritableProperties(self) -> List[str]:
        ...

    def setProperty(self, prop: str, *args: Any, **kwargs: Any) -> None:
        ...