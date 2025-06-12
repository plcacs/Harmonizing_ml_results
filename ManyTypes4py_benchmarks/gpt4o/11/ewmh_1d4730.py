from Xlib import display, X, protocol
import time
from typing import List, Optional, Union

class EWMH:
    NET_WM_WINDOW_TYPES: tuple = ('_NET_WM_WINDOW_TYPE_DESKTOP', '_NET_WM_WINDOW_TYPE_DOCK', '_NET_WM_WINDOW_TYPE_TOOLBAR', '_NET_WM_WINDOW_TYPE_MENU', '_NET_WM_WINDOW_TYPE_UTILITY', '_NET_WM_WINDOW_TYPE_SPLASH', '_NET_WM_WINDOW_TYPE_DIALOG', '_NET_WM_WINDOW_TYPE_DROPDOWN_MENU', '_NET_WM_WINDOW_TYPE_POPUP_MENU', '_NET_WM_WINDOW_TYPE_NOTIFICATION', '_NET_WM_WINDOW_TYPE_COMBO', '_NET_WM_WINDOW_TYPE_DND', '_NET_WM_WINDOW_TYPE_NORMAL')
    NET_WM_ACTIONS: tuple = ('_NET_WM_ACTION_MOVE', '_NET_WM_ACTION_RESIZE', '_NET_WM_ACTION_MINIMIZE', '_NET_WM_ACTION_SHADE', '_NET_WM_ACTION_STICK', '_NET_WM_ACTION_MAXIMIZE_HORZ', '_NET_WM_ACTION_MAXIMIZE_VERT', '_NET_WM_ACTION_FULLSCREEN', '_NET_WM_ACTION_CHANGE_DESKTOP', '_NET_WM_ACTION_CLOSE', '_NET_WM_ACTION_ABOVE', '_NET_WM_ACTION_BELOW')
    NET_WM_STATES: tuple = ('_NET_WM_STATE_MODAL', '_NET_WM_STATE_STICKY', '_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ', '_NET_WM_STATE_SHADED', '_NET_WM_STATE_SKIP_TASKBAR', '_NET_WM_STATE_SKIP_PAGER', '_NET_WM_STATE_HIDDEN', '_NET_WM_STATE_FULLSCREEN', '_NET_WM_STATE_ABOVE', '_NET_WM_STATE_BELOW', '_NET_WM_STATE_DEMANDS_ATTENTION')

    def __init__(self, _display: Optional[display.Display] = None, root: Optional[display.Window] = None):
        self.display: display.Display = _display or display.Display()
        self.root: display.Window = root or self.display.screen().root
        self.__getAttrs: dict = {'_NET_CLIENT_LIST': self.getClientList, '_NET_CLIENT_LIST_STACKING': self.getClientListStacking, '_NET_NUMBER_OF_DESKTOPS': self.getNumberOfDesktops, '_NET_DESKTOP_GEOMETRY': self.getDesktopGeometry, '_NET_DESKTOP_VIEWPORT': self.getDesktopViewPort, '_NET_CURRENT_DESKTOP': self.getCurrentDesktop, '_NET_ACTIVE_WINDOW': self.getActiveWindow, '_NET_WORKAREA': self.getWorkArea, '_NET_SHOWING_DESKTOP': self.getShowingDesktop, '_NET_WM_NAME': self.getWmName, '_NET_WM_VISIBLE_NAME': self.getWmVisibleName, '_NET_WM_DESKTOP': self.getWmDesktop, '_NET_WM_WINDOW_TYPE': self.getWmWindowType, '_NET_WM_STATE': self.getWmState, '_NET_WM_ALLOWED_ACTIONS': self.getWmAllowedActions, '_NET_WM_PID': self.getWmPid}
        self.__setAttrs: dict = {'_NET_NUMBER_OF_DESKTOPS': self.setNumberOfDesktops, '_NET_DESKTOP_GEOMETRY': self.setDesktopGeometry, '_NET_DESKTOP_VIEWPORT': self.setDesktopViewport, '_NET_CURRENT_DESKTOP': self.setCurrentDesktop, '_NET_ACTIVE_WINDOW': self.setActiveWindow, '_NET_SHOWING_DESKTOP': self.setShowingDesktop, '_NET_CLOSE_WINDOW': self.setCloseWindow, '_NET_MOVERESIZE_WINDOW': self.setMoveResizeWindow, '_NET_WM_NAME': self.setWmName, '_NET_WM_VISIBLE_NAME': self.setWmVisibleName, '_NET_WM_DESKTOP': self.setWmDesktop, '_NET_WM_STATE': self.setWmState}

    def setNumberOfDesktops(self, nb: int) -> None:
        self._setProperty('_NET_NUMBER_OF_DESKTOPS', [nb])

    def setDesktopGeometry(self, w: int, h: int) -> None:
        self._setProperty('_NET_DESKTOP_GEOMETRY', [w, h])

    def setDesktopViewport(self, w: int, h: int) -> None:
        self._setProperty('_NET_DESKTOP_VIEWPORT', [w, h])

    def setCurrentDesktop(self, i: int) -> None:
        self._setProperty('_NET_CURRENT_DESKTOP', [i, X.CurrentTime])

    def setActiveWindow(self, win: display.Window) -> None:
        self._setProperty('_NET_ACTIVE_WINDOW', [1, X.CurrentTime, win.id], win)

    def setShowingDesktop(self, show: int) -> None:
        self._setProperty('_NET_SHOWING_DESKTOP', [show])

    def setCloseWindow(self, win: display.Window) -> None:
        self._setProperty('_NET_CLOSE_WINDOW', [int(time.mktime(time.localtime())), 1], win)

    def setWmName(self, win: display.Window, name: str) -> None:
        self._setProperty('_NET_WM_NAME', name, win)

    def setWmVisibleName(self, win: display.Window, name: str) -> None:
        self._setProperty('_NET_WM_VISIBLE_NAME', name, win)

    def setWmDesktop(self, win: display.Window, i: int) -> None:
        self._setProperty('_NET_WM_DESKTOP', [i, 1], win)

    def setMoveResizeWindow(self, win: display.Window, gravity: int = 0, x: Optional[int] = None, y: Optional[int] = None, w: Optional[int] = None, h: Optional[int] = None) -> None:
        gravity_flags: int = gravity | 2048
        if x is None:
            x = 0
        else:
            gravity_flags = gravity_flags | 1024
        if y is None:
            y = 0
        else:
            gravity_flags = gravity_flags | 512
        if w is None:
            w = 0
        else:
            gravity_flags = gravity_flags | 256
        if h is None:
            h = 0
        else:
            gravity_flags = gravity_flags | 128
        self._setProperty('_NET_MOVERESIZE_WINDOW', [gravity_flags, x, y, w, h], win)

    def setWmState(self, win: display.Window, action: int, state: Union[int, str], state2: Union[int, str] = 0) -> None:
        if type(state) != int:
            state = self.display.get_atom(state, 1)
        if type(state2) != int:
            state2 = self.display.get_atom(state2, 1)
        self._setProperty('_NET_WM_STATE', [action, state, state2, 1], win)

    def getClientList(self) -> List[display.Window]:
        return [self._createWindow(w) for w in self._getProperty('_NET_CLIENT_LIST')]

    def getClientListStacking(self) -> List[display.Window]:
        return [self._createWindow(w) for w in self._getProperty('_NET_CLIENT_LIST_STACKING')]

    def getNumberOfDesktops(self) -> int:
        return self._getProperty('_NET_NUMBER_OF_DESKTOPS')[0]

    def getDesktopGeometry(self) -> List[int]:
        return self._getProperty('_NET_DESKTOP_GEOMETRY')

    def getDesktopViewPort(self) -> List[List[int]]:
        return self._getProperty('_NET_DESKTOP_VIEWPORT')

    def getCurrentDesktop(self) -> int:
        return self._getProperty('_NET_CURRENT_DESKTOP')[0]

    def getActiveWindow(self) -> Optional[display.Window]:
        active_window = self._getProperty('_NET_ACTIVE_WINDOW')
        if active_window is None:
            return None
        return self._createWindow(active_window[0])

    def getWorkArea(self) -> List[List[int]]:
        return self._getProperty('_NET_WORKAREA')

    def getShowingDesktop(self) -> int:
        return self._getProperty('_NET_SHOWING_DESKTOP')[0]

    def getWmName(self, win: display.Window) -> str:
        return self._getProperty('_NET_WM_NAME', win)

    def getWmVisibleName(self, win: display.Window) -> str:
        return self._getProperty('_NET_WM_VISIBLE_NAME', win)

    def getWmDesktop(self, win: display.Window) -> Optional[int]:
        arr = self._getProperty('_NET_WM_DESKTOP', win)
        return arr[0] if arr else None

    def getWmWindowType(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
        types = self._getProperty('_NET_WM_WINDOW_TYPE', win) or []
        if not str:
            return types
        return [self._getAtomName(t) for t in types]

    def getWmState(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
        states = self._getProperty('_NET_WM_STATE', win) or []
        if not str:
            return states
        return [self._getAtomName(s) for s in states]

    def getWmAllowedActions(self, win: display.Window, str: bool = False) -> List[Union[int, str]]:
        wAllowedActions = self._getProperty('_NET_WM_ALLOWED_ACTIONS', win) or []
        if not str:
            return wAllowedActions
        return [self._getAtomName(a) for a in wAllowedActions]

    def getWmPid(self, win: display.Window) -> Optional[int]:
        arr = self._getProperty('_NET_WM_PID', win)
        return arr[0] if arr else None

    def _getProperty(self, _type: str, win: Optional[display.Window] = None) -> Optional[List[int]]:
        if not win:
            win = self.root
        atom = win.get_full_property(self.display.get_atom(_type), X.AnyPropertyType)
        if atom:
            return atom.value

    def _setProperty(self, _type: str, data: Union[List[int], str], win: Optional[display.Window] = None, mask: Optional[int] = None) -> None:
        if not win:
            win = self.root
        if type(data) is str:
            dataSize = 8
        else:
            data = (data + [0] * (5 - len(data)))[:5]
            dataSize = 32
        ev = protocol.event.ClientMessage(window=win, client_type=self.display.get_atom(_type), data=(dataSize, data))
        if not mask:
            mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask
        self.root.send_event(ev, event_mask=mask)

    def _getAtomName(self, atom: int) -> str:
        try:
            return self.display.get_atom_name(atom)
        except:
            return 'UNKNOWN'

    def _createWindow(self, wId: int) -> Optional[display.Window]:
        if not wId:
            return None
        return self.display.create_resource_object('window', wId)

    def getReadableProperties(self) -> List[str]:
        return list(self.__getAttrs.keys())

    def getProperty(self, prop: str, *args, **kwargs) -> Union[int, str, List[Union[int, str]], display.Window, None]:
        f = self.__getAttrs.get(prop)
        if not f:
            raise KeyError('Unknown readable property: %s' % prop)
        return f(self, *args, **kwargs)

    def getWritableProperties(self) -> List[str]:
        return list(self.__setAttrs.keys())

    def setProperty(self, prop: str, *args, **kwargs) -> None:
        f = self.__setAttrs.get(prop)
        if not f:
            raise KeyError('Unknown writable property: %s' % prop)
        f(self, *args, **kwargs)
