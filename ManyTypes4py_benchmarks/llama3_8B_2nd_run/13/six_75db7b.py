import functools
import itertools
import operator
import sys
import types
from __future__ import absolute_import

class _LazyDescr(object):
    """Lazy loading of moved objects."""
    def __init__(self, name: str) -> None:
        self.name: str = name

    def __get__(self, obj: object, tp: type) -> object:
        result: object = self._resolve()
        setattr(obj, self.name, result)
        try:
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result

class MovedModule(_LazyDescr):
    """Lazy loading of moved modules."""
    def __init__(self, name: str, old: str, new: str = None) -> None:
        super().__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod: str = new
        else:
            self.mod: str = old

    def _resolve(self) -> object:
        return _import_module(self.mod)

    def __getattr__(self, attr: str) -> object:
        _module = self._resolve()
        value: object = getattr(_module, attr)
        setattr(self, attr, value)
        return value

class _LazyModule(types.ModuleType):
    """Lazy loading of moved modules."""
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self) -> list:
        attrs: list = ['__doc__', '__name__']
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs

class Module_six_moves_urllib_parse(_LazyModule):
    """Lazy loading of moved objects in six.moves.urllib.parse."""
    _urllib_parse_moved_attributes: list = [
        MovedAttribute('ParseResult', 'urlparse', 'urllib.parse'),
        MovedAttribute('SplitResult', 'urlparse', 'urllib.parse'),
        # ... more moved attributes ...
    ]
    for attr in _urllib_parse_moved_attributes:
        setattr(Module_six_moves_urllib_parse, attr.name, attr)
    del attr
    Module_six_moves_urllib_parse._moved_attributes = _urllib_parse_moved_attributes
    _importer._add_module(Module_six_moves_urllib_parse(__name__ + '.moves.urllib.parse'), 'moves.urllib.parse', 'moves.urllib.parse')

class Module_six_moves_urllib_request(_LazyModule):
    """Lazy loading of moved objects in six.moves.urllib.request."""
    _urllib_request_moved_attributes: list = [
        MovedAttribute('urlopen', 'urllib2', 'urllib.request'),
        MovedAttribute('install_opener', 'urllib2', 'urllib.request'),
        # ... more moved attributes ...
    ]
    for attr in _urllib_request_moved_attributes:
        setattr(Module_six_moves_urllib_request, attr.name, attr)
    del attr
    Module_six_moves_urllib_request._moved_attributes = _urllib_request_moved_attributes
    _importer._add_module(Module_six_moves_urllib_request(__name__ + '.moves.urllib.request'), 'moves.urllib.request', 'moves.urllib.request')

# ... more moved modules ...

def add_move(move: MovedModule) -> None:
    """Add an item to six.moves."""
    setattr(_MovedItems, move.name, move)

def remove_move(name: str) -> None:
    """Remove item from six.moves."""
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError('no such move, %r' % (name,))

# ... more functions ...

def assertCountEqual(self, *args, **kwargs) -> None:
    return getattr(self, _assertCountEqual)(*args, **kwargs)

def assertRaisesRegex(self, *args, **kwargs) -> None:
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)

def assertRegex(self, *args, **kwargs) -> None:
    return getattr(self, _assertRegex)(*args, **kwargs)

def reraise(tp: type, value: object, tb: type = None) -> None:
    if value is None:
        value = tp()
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value

def print_( *args, **kwargs) -> None:
    fp = kwargs.pop('file', sys.stdout)
    if fp is None:
        return

    def write(data: str) -> None:
        if not isinstance(data, str):
            data = str(data)
        fp.write(data)

    # ... more print_ function ...

def wraps(wrapped: callable, assigned: tuple = functools.WRAPPER_ASSIGNMENTS, updated: tuple = functools.WRAPPER_UPDATES) -> callable:
    """Create a wrapper function."""
    def wrapper(f: callable) -> callable:
        f = functools.wraps(wrapped, assigned, updated)(f)
        f.__wrapped__ = wrapped
        return f
    return wrapper

def with_metaclass(meta: type, *bases: type) -> type:
    """Create a base class with a metaclass."""
    class metaclass(meta):
        def __new__(cls, name: str, this_bases: tuple, d: dict) -> type:
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass: type) -> callable:
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls: type) -> type:
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper

def python_2_unicode_compatible(klass: type) -> type:
    """A decorator that defines __unicode__ and __str__ methods under Python 2."""
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__()." % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass
