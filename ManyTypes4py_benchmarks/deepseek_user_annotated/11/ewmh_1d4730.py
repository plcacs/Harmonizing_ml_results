# mypy: ignore-errors
# ruff: noqa

"""
pyewmh Copyright (C) 2011-2017 Julien Pagès (parkouss)
Licenced under GNU Lesser General Public License, version 3
https://github.com/parkouss/pyewmh

This module intends to provide an implementation of Extended Window Manager
Hints, based on the Xlib modules for python.

See the freedesktop.org `specification
<http://standards.freedesktop.org/wm-spec/wm-spec-latest.html>`_ for more
information.
"""

from Xlib import display, X, protocol
import time
from typing import Optional, List, Union, Dict, Callable, Any, Tuple, TypeVar

T = TypeVar('T')
Window = Any  # Xlib.X.Window is not exposed in a way we can type properly

class EWMH:
    """
    This class provides the ability to get and set properties defined by the
    EWMH spec.
    """
    NET_WM_WINDOW_TYPES: Tuple[str, ...] = (
        "_NET_WM_WINDOW_TYPE_DESKTOP",
        "_NET_WM_WINDOW_TYPE_DOCK",
        "_NET_WM_WINDOW_TYPE_TOOLBAR",
        "_NET_WM_WINDOW_TYPE_MENU",
        "_NET_WM_WINDOW_TYPE_UTILITY",
        "_NET_WM_WINDOW_TYPE_SPLASH",
        "_NET_WM_WINDOW_TYPE_DIALOG",
        "_NET_WM_WINDOW_TYPE_DROPDOWN_MENU",
        "_NET_WM_WINDOW_TYPE_POPUP_MENU",
        "_NET_WM_WINDOW_TYPE_NOTIFICATION",
        "_NET_WM_WINDOW_TYPE_COMBO",
        "_NET_WM_WINDOW_TYPE_DND",
        "_NET_WM_WINDOW_TYPE_NORMAL",
    )
    """List of strings representing all known window types."""

    NET_WM_ACTIONS: Tuple[str, ...] = (
        "_NET_WM_ACTION_MOVE",
        "_NET_WM_ACTION_RESIZE",
        "_NET_WM_ACTION_MINIMIZE",
        "_NET_WM_ACTION_SHADE",
        "_NET_WM_ACTION_STICK",
        "_NET_WM_ACTION_MAXIMIZE_HORZ",
        "_NET_WM_ACTION_MAXIMIZE_VERT",
        "_NET_WM_ACTION_FULLSCREEN",
        "_NET_WM_ACTION_CHANGE_DESKTOP",
        "_NET_WM_ACTION_CLOSE",
        "_NET_WM_ACTION_ABOVE",
        "_NET_WM_ACTION_BELOW",
    )
    """List of strings representing all known window actions."""

    NET_WM_STATES: Tuple[str, ...] = (
        "_NET_WM_STATE_MODAL",
        "_NET_WM_STATE_STICKY",
        "_NET_WM_STATE_MAXIMIZED_VERT",
        "_NET_WM_STATE_MAXIMIZED_HORZ",
        "_NET_WM_STATE_SHADED",
        "_NET_WM_STATE_SKIP_TASKBAR",
        "_NET_WM_STATE_SKIP_PAGER",
        "_NET_WM_STATE_HIDDEN",
        "_NET_WM_STATE_FULLSCREEN",
        "_NET_WM_STATE_ABOVE",
        "_NET_WM_STATE_BELOW",
        "_NET_WM_STATE_DEMANDS_ATTENTION",
    )
    """List of strings representing all known window states."""

    def __init__(self, _display: Optional[display.Display] = None, root: Optional[Window] = None) -> None:
        self.display: display.Display = _display or display.Display()
        self.root: Window = root or self.display.screen().root
        self.__getAttrs: Dict[str, Callable[..., Any]] = {
            "_NET_CLIENT_LIST": self.getClientList,
            "_NET_CLIENT_LIST_STACKING": self.getClientListStacking,
            "_NET_NUMBER_OF_DESKTOPS": self.getNumberOfDesktops,
            "_NET_DESKTOP_GEOMETRY": self.getDesktopGeometry,
            "_NET_DESKTOP_VIEWPORT": self.getDesktopViewPort,
            "_NET_CURRENT_DESKTOP": self.getCurrentDesktop,
            "_NET_ACTIVE_WINDOW": self.getActiveWindow,
            "_NET_WORKAREA": self.getWorkArea,
            "_NET_SHOWING_DESKTOP": self.getShowingDesktop,
            "_NET_WM_NAME": self.getWmName,
            "_NET_WM_VISIBLE_NAME": self.getWmVisibleName,
            "_NET_WM_DESKTOP": self.getWmDesktop,
            "_NET_WM_WINDOW_TYPE": self.getWmWindowType,
            "_NET_WM_STATE": self.getWmState,
            "_NET_WM_ALLOWED_ACTIONS": self.getWmAllowedActions,
            "_NET_WM_PID": self.getWmPid,
        }
        self.__setAttrs: Dict[str, Callable[..., Any]] = {
            "_NET_NUMBER_OF_DESKTOPS": self.setNumberOfDesktops,
            "_NET_DESKTOP_GEOMETRY": self.setDesktopGeometry,
            "_NET_DESKTOP_VIEWPORT": self.setDesktopViewport,
            "_NET_CURRENT_DESKTOP": self.setCurrentDesktop,
            "_NET_ACTIVE_WINDOW": self.setActiveWindow,
            "_NET_SHOWING_DESKTOP": self.setShowingDesktop,
            "_NET_CLOSE_WINDOW": self.setCloseWindow,
            "_NET_MOVERESIZE_WINDOW": self.setMoveResizeWindow,
            "_NET_WM_NAME": self.setWmName,
            "_NET_WM_VISIBLE_NAME": self.setWmVisibleName,
            "_NET_WM_DESKTOP": self.setWmDesktop,
            "_NET_WM_STATE": self.setWmState,
        }

    # ------------------------ setters properties ------------------------

    def setNumberOfDesktops(self, nb: int) -> None:
        """Set the number of desktops (property _NET_NUMBER_OF_DESKTOPS)."""
        self._setProperty("_NET_NUMBER_OF_DESKTOPS", [nb])

    def setDesktopGeometry(self, w: int, h: int) -> None:
        """Set the desktop geometry (property _NET_DESKTOP_GEOMETRY)"""
        self._setProperty("_NET_DESKTOP_GEOMETRY", [w, h])

    def setDesktopViewport(self, w: int, h: int) -> None:
        """Set the viewport size of the current desktop"""
        self._setProperty("_NET_DESKTOP_VIEWPORT", [w, h])

    def setCurrentDesktop(self, i: int) -> None:
        """Set the current desktop (property _NET_CURRENT_DESKTOP)."""
        self._setProperty("_NET_CURRENT_DESKTOP", [i, X.CurrentTime])

    def setActiveWindow(self, win: Window) -> None:
        """Set the given window active (property _NET_ACTIVE_WINDOW)"""
        self._setProperty("_NET_ACTIVE_WINDOW", [1, X.CurrentTime, win.id], win)

    def setShowingDesktop(self, show: int) -> None:
        """Set/unset the mode Showing desktop (property _NET_SHOWING_DESKTOP)"""
        self._setProperty("_NET_SHOWING_DESKTOP", [show])

    def setCloseWindow(self, win: Window) -> None:
        """Close the given window (property _NET_CLOSE_WINDOW)"""
        self._setProperty("_NET_CLOSE_WINDOW", [int(time.mktime(time.localtime())), 1], win)

    def setWmName(self, win: Window, name: str) -> None:
        """Set the property _NET_WM_NAME"""
        self._setProperty("_NET_WM_NAME", name, win)

    def setWmVisibleName(self, win: Window, name: str) -> None:
        """Set the property _NET_WM_VISIBLE_NAME"""
        self._setProperty("_NET_WM_VISIBLE_NAME", name, win)

    def setWmDesktop(self, win: Window, i: int) -> None:
        """Move the window to the desired desktop by changing the property _NET_WM_DESKTOP."""
        self._setProperty("_NET_WM_DESKTOP", [i, 1], win)

    def setMoveResizeWindow(self, win: Window, gravity: int = 0, x: Optional[int] = None, 
                          y: Optional[int] = None, w: Optional[int] = None, h: Optional[int] = None) -> None:
        """Set the property _NET_MOVERESIZE_WINDOW to move or resize the given window."""
        gravity_flags = gravity | 0b0000100000000000
        if x is None:
            x = 0
        else:
            gravity_flags = gravity_flags | 0b0000010000000000
        if y is None:
            y = 0
        else:
            gravity_flags = gravity_flags | 0b0000001000000000
        if w is None:
            w = 0
        else:
            gravity_flags = gravity_flags | 0b0000000100000000
        if h is None:
            h = 0
        else:
            gravity_flags = gravity_flags | 0b0000000010000000
        self._setProperty("_NET_MOVERESIZE_WINDOW", [gravity_flags, x, y, w, h], win)

    def setWmState(self, win: Window, action: int, state: Union[int, str], state2: Union[int, str] = 0) -> None:
        """Set/unset one or two state(s) for the given window (property _NET_WM_STATE)."""
        if type(state) != int:
            state = self.display.get_atom(state, 1)
        if type(state2) != int:
            state2 = self.display.get_atom(state2, 1)
        self._setProperty("_NET_WM_STATE", [action, state, state2, 1], win)

    # ------------------------ getters properties ------------------------

    def getClientList(self) -> List[Window]:
        """Get the list of windows maintained by the window manager."""
        return [self._createWindow(w) for w in self._getProperty("_NET_CLIENT_LIST")]

    def getClientListStacking(self) -> List[Window]:
        """Get the list of windows maintained by the window manager."""
        return [self._createWindow(w) for w in self._getProperty("_NET_CLIENT_LIST_STACKING")]

    def getNumberOfDesktops(self) -> int:
        """Get the number of desktops (property _NET_NUMBER_OF_DESKTOPS)."""
        return self._getProperty("_NET_NUMBER_OF_DESKTOPS")[0]

    def getDesktopGeometry(self) -> List[int]:
        """Get the desktop geometry (property _NET_DESKTOP_GEOMETRY)."""
        return self._getProperty("_NET_DESKTOP_GEOMETRY")

    def getDesktopViewPort(self) -> List[List[int]]:
        """Get the current viewports of each desktop."""
        return self._getProperty("_NET_DESKTOP_VIEWPORT")

    def getCurrentDesktop(self) -> int:
        """Get the current desktop number (property _NET_CURRENT_DESKTOP)"""
        return self._getProperty("_NET_CURRENT_DESKTOP")[0]

    def getActiveWindow(self) -> Optional[Window]:
        """Get the current active (toplevel) window or None."""
        active_window = self._getProperty("_NET_ACTIVE_WINDOW")
        if active_window is None:
            return None
        return self._createWindow(active_window[0])

    def getWorkArea(self) -> List[List[int]]:
        """Get the work area for each desktop (property _NET_WORKAREA)."""
        return self._getProperty("_NET_WORKAREA")

    def getShowingDesktop(self) -> int:
        """Get the value of "showing the desktop" mode."""
        return self._getProperty("_NET_SHOWING_DESKTOP")[0]

    def getWmName(self, win: Window) -> str:
        """Get the property _NET_WM_NAME for the given window."""
        return self._getProperty("_NET_WM_NAME", win)

    def getWmVisibleName(self, win: Window) -> str:
        """Get the property _NET_WM_VISIBLE_NAME for the given window."""
        return self._getProperty("_NET_WM_VISIBLE_NAME", win)

    def getWmDesktop(self, win: Window) -> Optional[int]:
        """Get the current desktop number of the given window."""
        arr = self._getProperty("_NET_WM_DESKTOP", win)
        return arr[0] if arr else None

    def getWmWindowType(self, win: Window, str: bool = False) -> Union[List[int], List[str]]:
        """Get the list of window types of the given window."""
        types = self._getProperty("_NET_WM_WINDOW_TYPE", win) or []
        if not str:
            return types
        return [self._getAtomName(t) for t in types]

    def getWmState(self, win: Window, str: bool = False) -> Union[List[int], List[str]]:
        """Get the list of states of the given window."""
        states = self._getProperty("_NET_WM_STATE", win) or []
        if not str:
            return states
        return [self._getAtomName(s) for s in states]

    def getWmAllowedActions(self, win: Window, str: bool = False) -> Union[List[int], List[str]]:
        """Get the list of allowed actions for the given window."""
        wAllowedActions = self._getProperty("_NET_WM_ALLOWED_ACTIONS", win) or []
        if not str:
            return wAllowedActions
        return [self._getAtomName(a) for a in wAllowedActions]

    def getWmPid(self, win: Window) -> Optional[int]:
        """Get the pid of the application associated to the given window."""
        arr = self._getProperty("_NET_WM_PID", win)
        return arr[0] if arr else None

    def _getProperty(self, _type: str, win: Optional[Window] = None) -> Any:
        if not win:
            win = self.root
        atom = win.get_full_property(self.display.get_atom(_type), X.AnyPropertyType)
        if atom:
            return atom.value

    def _setProperty(self, _type: str, data: Union[str, List[Any]], win: Optional[Window] = None, mask: Optional[int] = None) -> None:
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
            return "UNKNOWN"

    def _createWindow(self, wId: int) -> Optional[Window]:
        if not wId:
            return None
        return self.display.create_resource_object("window", wId)

    def getReadableProperties(self) -> List[str]:
        """Get all the readable properties' names"""
        return list(self.__getAttrs.keys())

    def getProperty(self, prop: str, *args: Any, **kwargs: Any) -> Any:
        """Get the value of a property."""
        f = self.__getAttrs.get(prop)
        if not f:
            raise KeyError("Unknown readable property: %s" % prop)
        return f(*args, **kwargs)

    def getWritableProperties(self) -> List[str]:
        """Get all the writable properties names"""
        return list(self.__setAttrs.keys())

    def setProperty(self, prop: str, *args: Any, **kwargs: Any) -> None:
        """Set the value of a property by sending an event on the root window."""
        f = self.__setAttrs.get(prop)
        if not f:
            raise KeyError("Unknown writable property: %s" % prop)
        f(*args, **kwargs)
