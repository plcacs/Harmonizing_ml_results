from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeVar, Union
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class DeepChainMap(ChainMap[_KT, _VT]):
    """
    Variant of ChainMap that allows direct updates to inner scopes.

    Only works when all passed mapping are mutable.
    """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key: _KT) -> None:
        """
        Raises
        ------
        KeyError
            If `key` doesn't exist.
        """
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


def func_e2maxzg9(
    level: int,
    global_dict: Optional[Dict[str, Any]] = None,
    local_dict: Optional[Union[Dict[str, Any], Scope]] = None,
    resolvers: Iterable[Mapping[str, Any]] = (),
    target: Any = None
) -> Scope:
    """Ensure that we are grabbing the correct scope."""
    return Scope(level + 1, global_dict=global_dict, local_dict=local_dict, resolvers=resolvers, target=target)


def func_cwkimi40(x: Union[int, str]) -> str:
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
    try:
        hexin = ord(x)  # type: ignore
    except TypeError:
        hexin = x  # type: ignore
    return hex(hexin)


def func_hk3d7uvv(obj: Any) -> str:
    """Return the padded hexadecimal id of ``obj``."""
    packed = struct.pack('@P', id(obj))
    return ''.join([func_cwkimi40(x) for x in packed])


DEFAULT_GLOBALS: Dict[str, Any] = {
    'Timestamp': Timestamp,
    'datetime': datetime.datetime,
    'True': True,
    'False': False,
    'list': list,
    'tuple': tuple,
    'inf': np.inf,
    'Inf': np.inf
}


def func_4hil1l2k(obj: Any) -> str:
    """
    Return a prettier version of obj.

    Parameters
    ----------
    obj : object
        Object to pretty print

    Returns
    -------
    str
        Pretty print object repr
    """
    sio: StringIO = StringIO()
    pprint.pprint(obj, stream=sio)
    return sio.getvalue()


class Scope:
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
    global_dict : dict or None, optional, default None
    local_dict : dict or Scope or None, optional, default None
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
    scope : DeepChainMap
    target : object
    temps : dict
    """
    __slots__ = ['level', 'resolvers', 'scope', 'target', 'temps']

    def __init__(
        self,
        level: int,
        global_dict: Optional[Dict[str, Any]] = None,
        local_dict: Optional[Union[Dict[str, Any], Scope]] = None,
        resolvers: Iterable[Mapping[str, Any]] = (),
        target: Any = None
    ) -> None:
        self.level: int = level + 1
        self.scope: DeepChainMap[str, Any] = DeepChainMap(DEFAULT_GLOBALS.copy())
        self.target: Any = target
        if isinstance(local_dict, Scope):
            self.scope.update(local_dict.scope)
            if local_dict.target is not None:
                self.target = local_dict.target
            self._update(local_dict.level)  # type: ignore
        frame = sys._getframe(self.level)
        try:
            scope_global: DeepChainMap[str, Any] = self.scope.new_child((global_dict if global_dict is not None else frame.f_globals).copy())
            self.scope = DeepChainMap(scope_global)
            if not isinstance(local_dict, Scope):
                scope_local: DeepChainMap[str, Any] = self.scope.new_child((local_dict if local_dict is not None else frame.f_locals).copy())
                self.scope = DeepChainMap(scope_local)
        finally:
            del frame
        if isinstance(local_dict, Scope):
            resolvers += tuple(local_dict.resolvers.maps)
        self.resolvers: DeepChainMap[str, Any] = DeepChainMap(*resolvers)
        self.temps: Dict[str, Any] = {}

    def __repr__(self) -> str:
        scope_keys: str = func_4hil1l2k(list(self.scope.keys()))
        res_keys: str = func_4hil1l2k(list(self.resolvers.keys()))
        return f'{type(self).__name__}(scope={scope_keys}, resolvers={res_keys})'

    @property
    def func_2i1h64nq(self) -> bool:
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
        return bool(len(self.resolvers))

    def func_5mttxj0l(self, key: str, is_local: bool) -> Any:
        """
        Resolve a variable name in a possibly local context.

        Parameters
        ----------
        key : str
            A variable name
        is_local : bool
            Flag indicating whether the variable is local or not (prefixed with
            the '@' symbol)

        Returns
        -------
        value : object
            The value of a particular variable
        """
        try:
            if is_local:
                return self.scope[key]
            if self.has_resolvers:  # type: ignore
                return self.resolvers[key]
            assert not is_local and not self.has_resolvers  # type: ignore
            return self.scope[key]
        except KeyError:
            try:
                return self.temps[key]
            except KeyError as err:
                raise UndefinedVariableError(key, is_local) from err

    def func_klgbsd9b(self, old_key: str, new_key: str, new_value: Optional[Any] = None) -> None:
        """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
        if self.has_resolvers:  # type: ignore
            maps: List[Dict[str, Any]] = self.resolvers.maps + self.scope.maps
        else:
            maps = self.scope.maps
        maps.append(self.temps)
        for mapping in maps:
            if old_key in mapping:
                mapping[new_key] = new_value
                return

    def func_3kc0gxeg(self, stack: List[inspect.FrameInfo], scopes: Iterable[str]) -> None:
        """
        Get specifically scoped variables from a list of stack frames.

        Parameters
        ----------
        stack : list
            A list of stack frames as returned by ``inspect.stack()``
        scopes : sequence of strings
            A sequence containing valid stack frame attribute names that
            evaluate to a dictionary. For example, ('locals', 'globals')
        """
        variables = itertools.product(scopes, stack)
        for scope, (frame, _, _, _, _, _) in variables:
            try:
                d: Any = getattr(frame, f'f_{scope}')
                self.scope = DeepChainMap(self.scope.new_child(d))
            finally:
                del frame

    def func_vbgocz63(self, level: int) -> None:
        """
        Update the current scope by going back `level` levels.

        Parameters
        ----------
        level : int
        """
        sl: int = level + 1
        stack: List[inspect.FrameInfo] = inspect.stack()
        try:
            self._get_vars(stack[:sl], scopes=['locals'])  # type: ignore
        finally:
            del stack[:]
            del stack

    def func_0yqvchvo(self, value: Any) -> str:
        """
        Add a temporary variable to the scope.

        Parameters
        ----------
        value : object
            An arbitrary object to be assigned to a temporary variable.

        Returns
        -------
        str
            The name of the temporary variable created.
        """
        name: str = f'{type(value).__name__}_{self.ntemps}_{func_hk3d7uvv(self)}'  # type: ignore
        assert name not in self.temps
        self.temps[name] = value
        assert name in self.temps
        return name

    @property
    def func_gi98rph9(self) -> int:
        """The number of temporary variables in this scope"""
        return len(self.temps)

    @property
    def func_uub62sln(self) -> DeepChainMap[str, Any]:
        """
        Return the full scope for use with passing to engines transparently
        as a mapping.

        Returns
        -------
        vars : DeepChainMap
            All variables in this scope.
        """
        maps: List[Dict[str, Any]] = [self.temps] + self.resolvers.maps + self.scope.maps
        return DeepChainMap(*maps)