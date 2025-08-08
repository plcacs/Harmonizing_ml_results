from typing import Tuple, Type, Any, Union

PY2: bool
PY3: bool
PY34: bool
string_types: Tuple[Type[str], ...]
integer_types: Tuple[Type[int], ...]
class_types: Tuple[Type, ...]
text_type: Type[str]
binary_type: Type[bytes]
MAXSIZE: int

def _add_doc(func: Any, doc: str) -> None:
    func.__doc__ = doc

def _import_module(name: str) -> Any:
    __import__(name)
    return sys.modules[name]

class _LazyDescr:
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, tp: Type) -> Any:
        result = self._resolve()
        setattr(obj, self.name, result)
        try:
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result

class MovedModule(_LazyDescr):
    def __init__(self, name: str, old: str, new: Union[str, None]) -> None:
        super(MovedModule, self).__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod = new
        else:
            self.mod = old

    def _resolve(self) -> Any:
        return _import_module(self.mod)

    def __getattr__(self, attr: str) -> Any:
        _module = self._resolve()
        value = getattr(_module, attr)
        setattr(self, attr, value)
        return value

class _LazyModule(types.ModuleType):
    def __init__(self, name: str) -> None:
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

class MovedAttribute(_LazyDescr):
    def __init__(self, name: str, old_mod: str, new_mod: Union[str, None], old_attr: Union[str, None], new_attr: Union[str, None]) -> None:
        super(MovedAttribute, self).__init__(name)
        if PY3:
            if new_mod is None:
                new_mod = name
            self.mod = new_mod
            if new_attr is None:
                if old_attr is None:
                    new_attr = name
                else:
                    new_attr = old_attr
            self.attr = new_attr
        else:
            self.mod = old_mod
            if old_attr is None:
                old_attr = name
            self.attr = old_attr

    def _resolve(self) -> Any:
        module = _import_module(self.mod)
        return getattr(module, self.attr)

class _SixMetaPathImporter:
    def __init__(self, six_module_name: str) -> None:
        self.name = six_module_name
        self.known_modules = {}

    def _add_module(self, mod: Any, *fullnames: str) -> None:
        for fullname in fullnames:
            self.known_modules[self.name + '.' + fullname] = mod

    def _get_module(self, fullname: str) -> Any:
        return self.known_modules[self.name + '.' + fullname]

    def find_module(self, fullname: str, path: Any = None) -> Any:
        if fullname in self.known_modules:
            return self
        return None

    def __get_module(self, fullname: str) -> Any:
        try:
            return self.known_modules[fullname]
        except KeyError:
            raise ImportError('This loader does not know module ' + fullname)

    def load_module(self, fullname: str) -> Any:
        try:
            return sys.modules[fullname]
        except KeyError:
            pass
        mod = self.__get_module(fullname)
        if isinstance(mod, MovedModule):
            mod = mod._resolve()
        else:
            mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod

    def is_package(self, fullname: str) -> bool:
        return hasattr(self.__get_module(fullname), '__path__')

    def get_code(self, fullname: str) -> None:
        self.__get_module(fullname)
        return None

_importer = _SixMetaPathImporter(__name__)

class _MovedItems(_LazyModule):
    pass

def add_move(move: Any) -> None:
    setattr(_MovedItems, move.name, move)

def remove_move(name: str) -> None:
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError('no such move, %r' % (name,))

def advance_iterator(it: Any) -> Any:
    return it.next()

def get_unbound_function(unbound: Any) -> Any:
    return unbound.im_func

def create_bound_method(func: Any, obj: Any) -> Any:
    return types.MethodType(func, obj, obj.__class__)

def create_unbound_method(func: Any, cls: Type) -> Any:
    return types.MethodType(func, None, cls)

def iterkeys(d: Any, **kw: Any) -> Any:
    return iter(d.keys(**kw))

def itervalues(d: Any, **kw: Any) -> Any:
    return iter(d.values(**kw))

def iteritems(d: Any, **kw: Any) -> Any:
    return iter(d.items(**kw))

def iterlists(d: Any, **kw: Any) -> Any:
    return iter(d.lists(**kw))

def b(s: str) -> bytes:
    return s.encode('latin-1')

def u(s: str) -> str:
    return s

def assertCountEqual(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertCountEqual)(*args, **kwargs)

def assertRaisesRegex(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)

def assertRegex(self: Any, *args: Any, **kwargs: Any) -> Any:
    return getattr(self, _assertRegex)(*args, **kwargs)

def reraise(tp: Type, value: Any, tb: Any = None) -> None:
    if value is None:
        value = tp()
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value

def exec_(_code_: str, _globs_: Any = None, _locs_: Any = None) -> None:
    if _globs_ is None:
        frame = sys._getframe(1)
        _globs_ = frame.f_globals
        if _locs_ is None:
            _locs_ = frame.f_locals
        del frame
    elif _locs_ is None:
        _locs_ = _globs_
    exec('exec _code_ in _globs_, _locs_')

def raise_from(value: Any, from_value: Any) -> None:
    raise value from from_value

def print_(*args: Any, **kwargs: Any) -> None:
    fp = kwargs.get('file', sys.stdout)
    flush = kwargs.pop('flush', False)
    _print(*args, **kwargs)
    if flush and fp is not None:
        fp.flush()

def wraps(wrapped: Any, assigned: Tuple[str, ...] = functools.WRAPPER_ASSIGNMENTS, updated: Tuple[str, ...] = functools.WRAPPER_UPDATES) -> Any:
    def wrapper(f: Any) -> Any:
        f = functools.wraps(wrapped, assigned, updated)(f)
        f.__wrapped__ = wrapped
        return f
    return wrapper

def with_metaclass(meta: Type, *bases: Type) -> Type:
    class metaclass(meta):
        def __new__(cls, name: str, this_bases: Tuple[Type, ...], d: dict) -> Any:
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass: Type) -> Any:
    def wrapper(cls: Type) -> Any:
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

def python_2_unicode_compatible(klass: Type) -> Type:
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied to %s because it doesn't define __str__()." % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return klass
