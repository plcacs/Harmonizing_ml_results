"""Utilities for generating code at runtime."""
from typing import Any, Callable, Dict, List, Mapping, Tuple, cast
__all__ = ['Function', 'Method', 'InitMethod', 'HashMethod', 'CompareMethod', 'EqMethod', 'NeMethod', 'LeMethod', 'LtMethod', 'GeMethod', 'GtMethod', 'build_function', 'build_function_source', 'reprkwargs', 'reprcall']
MISSING = object()

def Function(name, args, body, *, globals=None, locals=None, return_type=MISSING, argsep=', '):
    """Compile function code object from args and body."""
    return build_function(name=name, source=build_function_source(name=name, args=args, body=body, return_type=return_type, argsep=argsep), return_type=return_type, globals=globals, locals=locals)

def build_closure_source(name, args, body, *, outer_name='__outer__', outer_args=None, closures, return_type=MISSING, indentlevel=0, indentspaces=4, argsep=', '):
    inner_source = build_function_source(name, args, body, return_type=return_type, indentlevel=indentlevel, indentspaces=indentspaces, argsep=argsep)
    closure_vars = [f'{local_name} = {global_name}' for local_name, global_name in closures.items()]
    outer_source = build_function_source(name=outer_name, args=outer_args or [], body=closure_vars + inner_source.split('\n') + [f'return {name}'], return_type=MISSING, indentlevel=indentlevel, indentspaces=indentspaces, argsep=argsep)
    return outer_source

def build_closure(outer_name, source, *args, return_type=MISSING, globals=None, locals=None):
    assert locals is not None
    if return_type is not MISSING:
        locals['_return_type'] = return_type
    exec(source, globals, locals)
    obj = locals[outer_name](*args)
    obj.__sourcecode__ = source
    return cast(Callable, obj)

def build_function(name, source, *, return_type=MISSING, globals=None, locals=None):
    """Generate function from Python from source code string."""
    assert locals is not None
    if return_type is not MISSING:
        locals['_return_type'] = return_type
    exec(source, globals, locals)
    obj = locals[name]
    obj.__sourcecode__ = source
    return cast(Callable, obj)

def build_function_source(name, args, body, *, return_type=MISSING, indentlevel=0, indentspaces=4, argsep=', '):
    """Generate function source code from args and body."""
    indent = ' ' * indentspaces
    curindent = indent * indentlevel
    nextindent = indent * (indentlevel + 1)
    return_annotation = ''
    if return_type is not MISSING:
        return_annotation = '->_return_type'
    bodys = '\n'.join((f'{nextindent}{b}' for b in body))
    return f'{curindent}def {name}({argsep.join(args)}){return_annotation}:\n{bodys}'

def Method(name, args, body, **kwargs):
    """Generate Python method."""
    return Function(name, ['self'] + args, body, **kwargs)

def InitMethod(args, body, **kwargs):
    """Generate ``__init__`` method."""
    return Method('__init__', args, body, return_type='None', **kwargs)

def HashMethod(attrs, **kwargs):
    """Generate ``__hash__`` method."""
    self_tuple = obj_attrs_tuple('self', attrs)
    return Method('__hash__', [], [f'return hash({self_tuple})'], **kwargs)

def EqMethod(fields, **kwargs):
    """Generate ``__eq__`` method."""
    return CompareMethod(name='__eq__', op='==', fields=fields, **kwargs)

def NeMethod(fields, **kwargs):
    """Generate ``__ne__`` method."""
    return CompareMethod(name='__ne__', op='!=', fields=fields, **kwargs)

def GeMethod(fields, **kwargs):
    """Generate ``__ge__`` method."""
    return CompareMethod(name='__ge__', op='>=', fields=fields, **kwargs)

def GtMethod(fields, **kwargs):
    """Generate ``__gt__`` method."""
    return CompareMethod(name='__gt__', op='>', fields=fields, **kwargs)

def LeMethod(fields, **kwargs):
    """Generate ``__le__`` method."""
    return CompareMethod(name='__le__', op='<=', fields=fields, **kwargs)

def LtMethod(fields, **kwargs):
    """Generate ``__lt__`` method."""
    return CompareMethod(name='__lt__', op='<', fields=fields, **kwargs)

def CompareMethod(name, op, fields, **kwargs):
    """Generate object comparison method.

    Excellent for ``__eq__``, ``__le__``, etc.

    Examples:
        The example:

        .. sourcecode:: python

            CompareMethod(
                name='__eq__',
                op='==',
                fields=['x', 'y'],
            )

        Generates a method like this:

        .. sourcecode:: python

           def __eq__(self, other):
              if other.__class__ is self.__class__:
                   return (self.x,self.y) == (other.x,other.y)
               return NotImplemented
    """
    self_tuple = obj_attrs_tuple('self', fields)
    other_tuple = obj_attrs_tuple('other', fields)
    return Method(name, ['other'], ['if other.__class__ is self.__class__:', f' return {self_tuple}{op}{other_tuple}', 'return NotImplemented'], **kwargs)

def obj_attrs_tuple(obj_name, attrs):
    """Draw Python tuple from list of attributes.

    If attrs is ``['x', 'y']`` and ``obj_name`` is 'self',
    returns ``(self.x,self.y)``.
    """
    if not attrs:
        return '()'
    return f'({','.join([f'{obj_name}.{f}' for f in attrs])},)'

def reprkwargs(kwargs, *, sep=', ', fmt='{0}={1}'):
    return sep.join((fmt.format(k, repr(v)) for k, v in kwargs.items()))

def reprcall(name, args=(), kwargs={}, *, sep=', '):
    return '{0}({1}{2}{3})'.format(name, sep.join(map(repr, args or ())), (args and kwargs) and sep or '', reprkwargs(kwargs, sep=sep))