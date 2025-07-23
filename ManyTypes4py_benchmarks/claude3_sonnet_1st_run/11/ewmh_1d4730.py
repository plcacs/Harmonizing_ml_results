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
from typing import List, Dict, Callable, Union, Optional, Any, Tuple, cast
import time

class EWMH:
    """
    This class provides the ability to get and set properties defined by the
    EWMH spec.

    Each property can be accessed in two ways. For example, to get the active
    window::

      win = ewmh.getActiveWindow()
      # or: win = ewmh.getProperty('_NET_ACTIVE_WINDOW')

    Similarly, to set the active window::

      ewmh.setActiveWindow(myWindow)
      # or: ewmh.setProperty('_NET_ACTIVE_WINDOW', myWindow)

    When a property is written, don't forget to really send the notification by
    flushing requests::

      ewmh.display.flush()

    :param _display: the display to use. If not given, Xlib.display.Display()
                     is used.
    :param root: the root window to use. If not given,
                     self.display.screen().root is used.
    """
    NET_WM_WINDOW_TYPES: Tuple[str, ...] = ('_NET_WM_WINDOW_TYPE_DESKTOP', '_NET_WM_WINDOW_TYPE_DOCK', '_NET_WM_WINDOW_TYPE_TOOLBAR', '_NET_WM_WINDOW_TYPE_MENU', '_NET_WM_WINDOW_TYPE_UTILITY', '_NET_WM_WINDOW_TYPE_SPLASH', '_NET_WM_WINDOW_TYPE_DIALOG', '_NET_WM_WINDOW_TYPE_DROPDOWN_MENU', '_NET_WM_WINDOW_TYPE_POPUP_MENU', '_NET_WM_WINDOW_TYPE_NOTIFICATION', '_NET_WM_WINDOW_TYPE_COMBO', '_NET_WM_WINDOW_TYPE_DND', '_NET_WM_WINDOW_TYPE_NORMAL')
    'List of strings representing all known window types.'
    NET_WM_ACTIONS: Tuple[str, ...] = ('_NET_WM_ACTION_MOVE', '_NET_WM_ACTION_RESIZE', '_NET_WM_ACTION_MINIMIZE', '_NET_WM_ACTION_SHADE', '_NET_WM_ACTION_STICK', '_NET_WM_ACTION_MAXIMIZE_HORZ', '_NET_WM_ACTION_MAXIMIZE_VERT', '_NET_WM_ACTION_FULLSCREEN', '_NET_WM_ACTION_CHANGE_DESKTOP', '_NET_WM_ACTION_CLOSE', '_NET_WM_ACTION_ABOVE', '_NET_WM_ACTION_BELOW')
    'List of strings representing all known window actions.'
    NET_WM_STATES: Tuple[str, ...] = ('_NET_WM_STATE_MODAL', '_NET_WM_STATE_STICKY', '_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ', '_NET_WM_STATE_SHADED', '_NET_WM_STATE_SKIP_TASKBAR', '_NET_WM_STATE_SKIP_PAGER', '_NET_WM_STATE_HIDDEN', '_NET_WM_STATE_FULLSCREEN', '_NET_WM_STATE_ABOVE', '_NET_WM_STATE_BELOW', '_NET_WM_STATE_DEMANDS_ATTENTION')
    'List of strings representing all known window states.'

    def __init__(self, _display: Optional[display.Display] = None, root: Optional[Any] = None) -> None:
        self.display: display.Display = _display or display.Display()
        self.root: Any = root or self.display.screen().root
        self.__getAttrs: Dict[str, Callable] = {'_NET_CLIENT_LIST': self.getClientList, '_NET_CLIENT_LIST_STACKING': self.getClientListStacking, '_NET_NUMBER_OF_DESKTOPS': self.getNumberOfDesktops, '_NET_DESKTOP_GEOMETRY': self.getDesktopGeometry, '_NET_DESKTOP_VIEWPORT': self.getDesktopViewPort, '_NET_CURRENT_DESKTOP': self.getCurrentDesktop, '_NET_ACTIVE_WINDOW': self.getActiveWindow, '_NET_WORKAREA': self.getWorkArea, '_NET_SHOWING_DESKTOP': self.getShowingDesktop, '_NET_WM_NAME': self.getWmName, '_NET_WM_VISIBLE_NAME': self.getWmVisibleName, '_NET_WM_DESKTOP': self.getWmDesktop, '_NET_WM_WINDOW_TYPE': self.getWmWindowType, '_NET_WM_STATE': self.getWmState, '_NET_WM_ALLOWED_ACTIONS': self.getWmAllowedActions, '_NET_WM_PID': self.getWmPid}
        self.__setAttrs: Dict[str, Callable] = {'_NET_NUMBER_OF_DESKTOPS': self.setNumberOfDesktops, '_NET_DESKTOP_GEOMETRY': self.setDesktopGeometry, '_NET_DESKTOP_VIEWPORT': self.setDesktopViewport, '_NET_CURRENT_DESKTOP': self.setCurrentDesktop, '_NET_ACTIVE_WINDOW': self.setActiveWindow, '_NET_SHOWING_DESKTOP': self.setShowingDesktop, '_NET_CLOSE_WINDOW': self.setCloseWindow, '_NET_MOVERESIZE_WINDOW': self.setMoveResizeWindow, '_NET_WM_NAME': self.setWmName, '_NET_WM_VISIBLE_NAME': self.setWmVisibleName, '_NET_WM_DESKTOP': self.setWmDesktop, '_NET_WM_STATE': self.setWmState}

    def setNumberOfDesktops(self, nb: int) -> None:
        """
        Set the number of desktops (property _NET_NUMBER_OF_DESKTOPS).

        :param nb: the number of desired desktops"""
        self._setProperty('_NET_NUMBER_OF_DESKTOPS', [nb])

    def setDesktopGeometry(self, w: int, h: int) -> None:
        """
        Set the desktop geometry (property _NET_DESKTOP_GEOMETRY)

        :param w: desktop width
        :param h: desktop height"""
        self._setProperty('_NET_DESKTOP_GEOMETRY', [w, h])

    def setDesktopViewport(self, w: int, h: int) -> None:
        """
        Set the viewport size of the current desktop
        (property _NET_DESKTOP_VIEWPORT)

        :param w: desktop width
        :param h: desktop height"""
        self._setProperty('_NET_DESKTOP_VIEWPORT', [w, h])

    def setCurrentDesktop(self, i: int) -> None:
        """
        Set the current desktop (property _NET_CURRENT_DESKTOP).

        :param i: the desired desktop number"""
        self._setProperty('_NET_CURRENT_DESKTOP', [i, X.CurrentTime])

    def setActiveWindow(self, win: Any) -> None:
        """
        Set the given window active (property _NET_ACTIVE_WINDOW)

        :param win: the window object"""
        self._setProperty('_NET_ACTIVE_WINDOW', [1, X.CurrentTime, win.id], win)

    def setShowingDesktop(self, show: int) -> None:
        """
        Set/unset the mode Showing desktop (property _NET_SHOWING_DESKTOP)

        :param show: 1 to set the desktop mode, else 0"""
        self._setProperty('_NET_SHOWING_DESKTOP', [show])

    def setCloseWindow(self, win: Any) -> None:
        """
        Close the given window (property _NET_CLOSE_WINDOW)

        :param win: the window object"""
        self._setProperty('_NET_CLOSE_WINDOW', [int(time.mktime(time.localtime())), 1], win)

    def setWmName(self, win: Any, name: str) -> None:
        """
        Set the property _NET_WM_NAME

        :param win: the window object
        :param name: desired name"""
        self._setProperty('_NET_WM_NAME', name, win)

    def setWmVisibleName(self, win: Any, name: str) -> None:
        """
        Set the property _NET_WM_VISIBLE_NAME

        :param win: the window object
        :param name: desired visible name"""
        self._setProperty('_NET_WM_VISIBLE_NAME', name, win)

    def setWmDesktop(self, win: Any, i: int) -> None:
        """
        Move the window to the desired desktop by changing the property
        _NET_WM_DESKTOP.

        :param win: the window object
        :param i: desired desktop number
        """
        self._setProperty('_NET_WM_DESKTOP', [i, 1], win)

    def setMoveResizeWindow(self, win: Any, gravity: int = 0, x: Optional[int] = None, y: Optional[int] = None, w: Optional[int] = None, h: Optional[int] = None) -> None:
        """
        Set the property _NET_MOVERESIZE_WINDOW to move or resize the given
        window. Flags are automatically calculated if x, y, w or h are defined.

        :param win: the window object
        :param gravity: gravity (one of the Xlib.X.*Gravity constant or 0)
        :param x: int or None
        :param y: int or None
        :param w: int or None
        :param h: int or None
        """
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

    def setWmState(self, win: Any, action: int, state: Union[int, str], state2: Union[int, str] = 0) -> None:
        """
        Set/unset one or two state(s) for the given window (property
        _NET_WM_STATE).

        :param win: the window object
        :param action: 0 to remove, 1 to add or 2 to toggle state(s)
        :param state: a state
        :type state: int or str (see :attr:`NET_WM_STATES`)
        :param state2: a state or 0
        :type state2: int or str (see :attr:`NET_WM_STATES`)
        """
        if type(state) != int:
            state = self.display.get_atom(state, 1)
        if type(state2) != int:
            state2 = self.display.get_atom(state2, 1)
        self._setProperty('_NET_WM_STATE', [action, state, state2, 1], win)

    def getClientList(self) -> List[Any]:
        """
        Get the list of windows maintained by the window manager for the
        property _NET_CLIENT_LIST.

        :return: list of Window objects
        """
        return [self._createWindow(w) for w in self._getProperty('_NET_CLIENT_LIST')]

    def getClientListStacking(self) -> List[Any]:
        """
        Get the list of windows maintained by the window manager for the
        property _NET_CLIENT_LIST_STACKING.

        :return: list of Window objects"""
        return [self._createWindow(w) for w in self._getProperty('_NET_CLIENT_LIST_STACKING')]

    def getNumberOfDesktops(self) -> int:
        """
        Get the number of desktops (property _NET_NUMBER_OF_DESKTOPS).

        :return: int"""
        return cast(int, self._getProperty('_NET_NUMBER_OF_DESKTOPS')[0])

    def getDesktopGeometry(self) -> List[int]:
        """
        Get the desktop geometry (property _NET_DESKTOP_GEOMETRY) as an array
        of two integers [width, height].

        :return: [int, int]"""
        return cast(List[int], self._getProperty('_NET_DESKTOP_GEOMETRY'))

    def getDesktopViewPort(self) -> List[List[int]]:
        """
        Get the current viewports of each desktop as a list of [x, y]
        representing the top left corner (property _NET_DESKTOP_VIEWPORT).

        :return: list of [int, int]
        """
        return cast(List[List[int]], self._getProperty('_NET_DESKTOP_VIEWPORT'))

    def getCurrentDesktop(self) -> int:
        """
        Get the current desktop number (property _NET_CURRENT_DESKTOP)

        :return: int
        """
        return cast(int, self._getProperty('_NET_CURRENT_DESKTOP')[0])

    def getActiveWindow(self) -> Optional[Any]:
        """
        Get the current active (toplevel) window or None (property
        _NET_ACTIVE_WINDOW)

        :return: Window object or None
        """
        active_window = self._getProperty('_NET_ACTIVE_WINDOW')
        if active_window is None:
            return None
        return self._createWindow(active_window[0])

    def getWorkArea(self) -> List[List[int]]:
        """
        Get the work area for each desktop (property _NET_WORKAREA) as a list
        of [x, y, width, height]

        :return: a list of [int, int, int, int]
        """
        return cast(List[List[int]], self._getProperty('_NET_WORKAREA'))

    def getShowingDesktop(self) -> int:
        """
        Get the value of "showing the desktop" mode of the window manager
        (property _NET_SHOWING_DESKTOP).  1 means the mode is activated, and 0
        means deactivated.

        :return: int
        """
        return cast(int, self._getProperty('_NET_SHOWING_DESKTOP')[0])

    def getWmName(self, win: Any) -> str:
        """
        Get the property _NET_WM_NAME for the given window as a string.

        :param win: the window object
        :return: str
        """
        return cast(str, self._getProperty('_NET_WM_NAME', win))

    def getWmVisibleName(self, win: Any) -> str:
        """
        Get the property _NET_WM_VISIBLE_NAME for the given window as a string.

        :param win: the window object
        :return: str
        """
        return cast(str, self._getProperty('_NET_WM_VISIBLE_NAME', win))

    def getWmDesktop(self, win: Any) -> Optional[int]:
        """
        Get the current desktop number of the given window (property
        _NET_WM_DESKTOP).

        :param win: the window object
        :return: int
        """
        arr = self._getProperty('_NET_WM_DESKTOP', win)
        return cast(int, arr[0]) if arr else None

    def getWmWindowType(self, win: Any, str: bool = False) -> List[Union[int, str]]:
        """
        Get the list of window types of the given window (property
        _NET_WM_WINDOW_TYPE).

        :param win: the window object
        :param str: True to get a list of string types instead of int
        :return: list of (int|str)
        """
        types = self._getProperty('_NET_WM_WINDOW_TYPE', win) or []
        if not str:
            return cast(List[int], types)
        return [self._getAtomName(t) for t in types]

    def getWmState(self, win: Any, str: bool = False) -> List[Union[int, str]]:
        """
        Get the list of states of the given window (property _NET_WM_STATE).

        :param win: the window object
        :param str: True to get a list of string states instead of int
        :return: list of (int|str)
        """
        states = self._getProperty('_NET_WM_STATE', win) or []
        if not str:
            return cast(List[int], states)
        return [self._getAtomName(s) for s in states]

    def getWmAllowedActions(self, win: Any, str: bool = False) -> List[Union[int, str]]:
        """
        Get the list of allowed actions for the given window (property
        _NET_WM_ALLOWED_ACTIONS).

        :param win: the window object
        :param str: True to get a list of string allowed actions instead of int
        :return: list of (int|str)
        """
        wAllowedActions = self._getProperty('_NET_WM_ALLOWED_ACTIONS', win) or []
        if not str:
            return cast(List[int], wAllowedActions)
        return [self._getAtomName(a) for a in wAllowedActions]

    def getWmPid(self, win: Any) -> Optional[int]:
        """
        Get the pid of the application associated to the given window (property
        _NET_WM_PID)

        :param win: the window object
        """
        arr = self._getProperty('_NET_WM_PID', win)
        return cast(int, arr[0]) if arr else None

    def _getProperty(self, _type: str, win: Optional[Any] = None) -> Any:
        if not win:
            win = self.root
        atom = win.get_full_property(self.display.get_atom(_type), X.AnyPropertyType)
        if atom:
            return atom.value
        return None

    def _setProperty(self, _type: str, data: Union[List[Any], str], win: Optional[Any] = None, mask: Optional[int] = None) -> None:
        """
        Send a ClientMessage event to the root window
        """
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

    def _createWindow(self, wId: int) -> Optional[Any]:
        if not wId:
            return None
        return self.display.create_resource_object('window', wId)

    def getReadableProperties(self) -> List[str]:
        """
        Get all the readable properties' names
        """
        return list(self.__getAttrs.keys())

    def getProperty(self, prop: str, *args: Any, **kwargs: Any) -> Any:
        """
        Get the value of a property. See the corresponding method for the
        required arguments.  For example, for the property _NET_WM_STATE, look
        for :meth:`getWmState`
        """
        f = self.__getAttrs.get(prop)
        if not f:
            raise KeyError('Unknown readable property: %s' % prop)
        return f(self, *args, **kwargs)

    def getWritableProperties(self) -> List[str]:
        """Get all the writable properties names"""
        return list(self.__setAttrs.keys())

    def setProperty(self, prop: str, *args: Any, **kwargs: Any) -> None:
        """
        Set the value of a property by sending an event on the root window.
        See the corresponding method for the required arguments. For example,
        for the property _NET_WM_STATE, look for :meth:`setWmState`
        """
        f = self.__setAttrs.get(prop)
        if not f:
            raise KeyError('Unknown writable property: %s' % prop)
        f(self, *args, **kwargs)
