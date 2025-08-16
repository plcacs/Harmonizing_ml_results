from typing import Tuple, Type, Union

PY2: bool
PY3: bool
PY34: bool
string_types: Union[Tuple[str], Tuple[Type[str]]]
integer_types: Union[Tuple[int], Tuple[Type[int]]]
class_types: Union[Tuple[type], Tuple[Type[type]]]
text_type: Type[str]
binary_type: Type[bytes]
MAXSIZE: int

def _add_doc(func, doc: str) -> None:
    func.__doc__ = doc

def _import_module(name: str) -> Type[types.ModuleType]:
    __import__(name)
    return sys.modules[name]

class _LazyDescr:
    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, tp):
        result = self._resolve()
        setattr(obj, self.name, result)
        try:
            delattr(obj.__class__, self.name)
        except AttributeError:
            pass
        return result

class MovedModule(_LazyDescr):
    def __init__(self, name: str, old: str, new: str = None):
        super(MovedModule, self).__init__(name)
        if PY3:
            if new is None:
                new = name
            self.mod = new
        else:
            self.mod = old

    def _resolve(self) -> Type[types.ModuleType]:
        return _import_module(self.mod)

    def __getattr__(self, attr: str) -> Type:
        _module = self._resolve()
        value = getattr(_module, attr)
        setattr(self, attr, value)
        return value

class _LazyModule(types.ModuleType):
    def __init__(self, name: str):
        super(_LazyModule, self).__init__(name)
        self.__doc__ = self.__class__.__doc__

    def __dir__(self) -> list:
        attrs = ['__doc__', '__name__']
        attrs += [attr.name for attr in self._moved_attributes]
        return attrs

class MovedAttribute(_LazyDescr):
    def __init__(self, name: str, old_mod: str, new_mod: str, old_attr: str = None, new_attr: str = None):
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

    def _resolve(self):
        module = _import_module(self.mod)
        return getattr(module, self.attr)

class _SixMetaPathImporter:
    def __init__(self, six_module_name: str):
        self.name = six_module_name
        self.known_modules = {}

    def _add_module(self, mod, *fullnames: str) -> None:
        for fullname in fullnames:
            self.known_modules[self.name + '.' + fullname] = mod

    def _get_module(self, fullname: str) -> Type[types.ModuleType]:
        return self.known_modules[self.name + '.' + fullname]

    def find_module(self, fullname: str, path=None) -> Union[Type['_SixMetaPathImporter'], None]:
        if fullname in self.known_modules:
            return self
        return None

    def __get_module(self, fullname: str) -> Type[types.ModuleType]:
        try:
            return self.known_modules[fullname]
        except KeyError:
            raise ImportError('This loader does not know module ' + fullname)

    def load_module(self, fullname: str) -> Type[types.ModuleType]:
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

class Module_six_moves_urllib_parse(_LazyModule):
    pass

class Module_six_moves_urllib_error(_LazyModule):
    pass

class Module_six_moves_urllib_request(_LazyModule):
    pass

class Module_six_moves_urllib_response(_LazyModule):
    pass

class Module_six_moves_urllib_robotparser(_LazyModule):
    pass

class Module_six_moves_urllib(types.ModuleType):
    pass

def add_move(move: MovedAttribute) -> None:
    setattr(_MovedItems, move.name, move)

def remove_move(name: str) -> None:
    try:
        delattr(_MovedItems, name)
    except AttributeError:
        try:
            del moves.__dict__[name]
        except KeyError:
            raise AttributeError('no such move, %r' % (name,))

def get_unbound_function(unbound: Type) -> Type:
    return unbound

def create_bound_method(func: Type, obj: object) -> Type:
    return types.MethodType(func, obj, obj.__class__)

def create_unbound_method(func: Type, cls: Type) -> Type:
    return types.MethodType(func, None, cls)

def iterkeys(d, **kw) -> iter:
    return iter(d.keys(**kw))

def itervalues(d, **kw) -> iter:
    return iter(d.values(**kw))

def iteritems(d, **kw) -> iter:
    return iter(d.items(**kw))

def iterlists(d, **kw) -> iter:
    return iter(d.lists(**kw))

def b(s: str) -> bytes:
    return s.encode('latin-1')

def u(s: str) -> str:
    return s

def assertCountEqual(self, *args, **kwargs) -> None:
    return getattr(self, _assertCountEqual)(*args, **kwargs)

def assertRaisesRegex(self, *args, **kwargs) -> None:
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)

def assertRegex(self, *args, **kwargs) -> None:
    return getattr(self, _assertRegex)(*args, **kwargs)

def reraise(tp: Type, value: Exception, tb: Union[None, Type]) -> None:
    try:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None
        tb = None

def raise_from(value: Exception, from_value: Exception) -> None:
    try:
        raise value from from_value
    finally:
        value = None

def wraps(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES):
    def wrapper(f):
        f = functools.wraps(wrapped, assigned, updated)(f)
        f.__wrapped__ = wrapped
        return f
    return wrapper

def with_metaclass(meta, *bases) -> Type:
    class metaclass(meta):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

def add_metaclass(metaclass) -> Type:
    def wrapper(cls):
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
