"""
Module for scope operations
"""
from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar
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

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
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


def func_e2maxzg9(level, global_dict=None, local_dict=None, resolvers=(),
    target=None):
    """Ensure that we are grabbing the correct scope."""
    return Scope(level + 1, global_dict=global_dict, local_dict=local_dict,
        resolvers=resolvers, target=target)


def func_cwkimi40(x):
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
    try:
        hexin = ord(x)
    except TypeError:
        hexin = x
    return hex(hexin)


def func_hk3d7uvv(obj):
    """Return the padded hexadecimal id of ``obj``."""
    packed = struct.pack('@P', id(obj))
    return ''.join([func_cwkimi40(x) for x in packed])


DEFAULT_GLOBALS = {'Timestamp': Timestamp, 'datetime': datetime.datetime,
    'True': True, 'False': False, 'list': list, 'tuple': tuple, 'inf': np.
    inf, 'Inf': np.inf}


def func_4hil1l2k(obj):
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
    sio = StringIO()
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

    def __init__(self, level, global_dict=None, local_dict=None, resolvers=
        (), target=None):
        self.level = level + 1
        self.scope = DeepChainMap(DEFAULT_GLOBALS.copy())
        self.target = target
        if isinstance(local_dict, Scope):
            self.scope.update(local_dict.scope)
            if local_dict.target is not None:
                self.target = local_dict.target
            self._update(local_dict.level)
        frame = sys._getframe(self.level)
        try:
            scope_global = self.scope.new_child((global_dict if global_dict
                 is not None else frame.f_globals).copy())
            self.scope = DeepChainMap(scope_global)
            if not isinstance(local_dict, Scope):
                scope_local = self.scope.new_child((local_dict if 
                    local_dict is not None else frame.f_locals).copy())
                self.scope = DeepChainMap(scope_local)
        finally:
            del frame
        if isinstance(local_dict, Scope):
            resolvers += tuple(local_dict.resolvers.maps)
        self.resolvers = DeepChainMap(*resolvers)
        self.temps = {}

    def __repr__(self):
        scope_keys = func_4hil1l2k(list(self.scope.keys()))
        res_keys = func_4hil1l2k(list(self.resolvers.keys()))
        return (
            f'{type(self).__name__}(scope={scope_keys}, resolvers={res_keys})')

    @property
    def func_2i1h64nq(self):
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
        return bool(len(self.resolvers))

    def func_5mttxj0l(self, key, is_local):
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
            if self.has_resolvers:
                return self.resolvers[key]
            assert not is_local and not self.has_resolvers
            return self.scope[key]
        except KeyError:
            try:
                return self.temps[key]
            except KeyError as err:
                raise UndefinedVariableError(key, is_local) from err

    def func_klgbsd9b(self, old_key, new_key, new_value=None):
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
        if self.has_resolvers:
            maps = self.resolvers.maps + self.scope.maps
        else:
            maps = self.scope.maps
        maps.append(self.temps)
        for mapping in maps:
            if old_key in mapping:
                mapping[new_key] = new_value
                return

    def func_3kc0gxeg(self, stack, scopes):
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
                d = getattr(frame, f'f_{scope}')
                self.scope = DeepChainMap(self.scope.new_child(d))
            finally:
                del frame

    def func_vbgocz63(self, level):
        """
        Update the current scope by going back `level` levels.

        Parameters
        ----------
        level : int
        """
        sl = level + 1
        stack = inspect.stack()
        try:
            self._get_vars(stack[:sl], scopes=['locals'])
        finally:
            del stack[:], stack

    def func_0yqvchvo(self, value):
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
        name = f'{type(value).__name__}_{self.ntemps}_{func_hk3d7uvv(self)}'
        assert name not in self.temps
        self.temps[name] = value
        assert name in self.temps
        return name

    @property
    def func_gi98rph9(self):
        """The number of temporary variables in this scope"""
        return len(self.temps)

    @property
    def func_uub62sln(self):
        """
        Return the full scope for use with passing to engines transparently
        as a mapping.

        Returns
        -------
        vars : DeepChainMap
            All variables in this scope.
        """
        maps = [self.temps] + self.resolvers.maps + self.scope.maps
        return DeepChainMap(*maps)
