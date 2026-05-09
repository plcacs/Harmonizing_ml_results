from Xlib import display, X, protocol
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

NET_WM_WINDOW_TYPES = Tuple[str, ...]
NET_WM_ACTIONS = Tuple[str, ...]
NET_WM_STATES = Tuple[str, ...]

class EWMH:
    display: display.Display
    root: protocol.Window

    def __init__(self, _display: Optional[display.Display] = None, root: Optional[protocol.Window] = None) -> None:
        ...

    def setNumberOfDesktops(self, nb: int) -> None:
        ...

    def setDesktopGeometry(self, w: int, h: int) -> None:
        ...

    def setDesktopViewport(self, w: int, h: int) -> None:
        ...

    def setCurrentDesktop(self, i: int) -> None:
        ...

    def setActiveWindow(self, win: protocol.Window) -> None:
        ...

    def setShowingDesktop(self, show: int) -> None:
        ...

    def setCloseWindow(self, win: protocol.Window) -> None:
        ...

    def setWmName(self, win: protocol.Window, name: str) -> None:
        ...

    def setWmVisibleName(self, win: protocol.Window, name: str) -> None:
        ...

    def setWmDesktop(self, win: protocol.Window, i: int) -> None:
        ...

    def setMoveResizeWindow(
        self,
        win: protocol.Window,
        gravity: int = 0,
        x: Optional[int] = None,
        y: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
    ) -> None:
        ...

    def setWmState(
        self,
        win: protocol.Window,
        action: int,
        state: Union[int, str],
        state2: Union[int, str, int] = 0,
    ) -> None:
        ...

    def getClientList(self) -> List[protocol.Window]:
        ...

    def getClientListStacking(self) -> List[protocol.Window]:
        ...

    def getNumberOfDesktops(self) -> int:
        ...

    def getDesktopGeometry(self) -> List[int]:
        ...

    def getDesktopViewPort(self) -> List[List[int]]:
        ...

    def getCurrentDesktop(self) -> int:
        ...

    def getActiveWindow(self) -> Optional[protocol.Window]:
        ...

    def getWorkArea(self) -> List[List[int]]:
        ...

    def getShowingDesktop(self) -> int:
        ...

    def getWmName(self, win: protocol.Window) -> str:
        ...

    def getWmVisibleName(self, win: protocol.Window) -> str:
        ...

    def getWmDesktop(self, win: protocol.Window) -> Optional[int]:
        ...

    def getWmWindowType(self, win: protocol.Window, str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmState(self, win: protocol.Window, str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmAllowedActions(self, win: protocol.Window, str: bool = False) -> Union[List[int], List[str]]:
        ...

    def getWmPid(self, win: protocol.Window) -> Optional[int]:
        ...

    def _getProperty(self, _type: str, win: Optional[protocol.Window] = None) -> Any:
        ...

    def _setProperty(
        self,
        _type: str,
        data: Union[str, List[Any]],
        win: Optional[protocol.Window] = None,
        mask: Optional[int] = None,
    ) -> None:
        ...

    def _getAtomName(self, atom: int) -> str:
        ...

    def _createWindow(self, wId: int) -> Optional[protocol.Window]:
        ...

    def getReadableProperties(self) -> List[str]:
        ...

    def getProperty(self, prop: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def getWritableProperties(self) -> List[str]:
        ...

    def setProperty(self, prop: str, *args: Any, **kwargs: Any) -> None:
        ...

    __getAttrs: Dict[str, Callable[..., Any]]
    __setAttrs: Dict[str, Callable[..., Any]]