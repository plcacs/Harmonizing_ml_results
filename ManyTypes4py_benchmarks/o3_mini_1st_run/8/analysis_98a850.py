#!/usr/bin/env python3
"""
Module for statical analysis.
"""
from typing import Any, Optional, Tuple, Type
from parso.python import tree
from jedi import debug
from jedi.inference.helpers import is_string

CODES: dict[str, Tuple[int, Type[Exception], Optional[str]]] = {
    'attribute-error': (1, AttributeError, 'Potential AttributeError.'),
    'name-error': (2, NameError, 'Potential NameError.'),
    'import-error': (3, ImportError, 'Potential ImportError.'),
    'type-error-too-many-arguments': (4, TypeError, None),
    'type-error-too-few-arguments': (5, TypeError, None),
    'type-error-keyword-argument': (6, TypeError, None),
    'type-error-multiple-values': (7, TypeError, None),
    'type-error-star-star': (8, TypeError, None),
    'type-error-star': (9, TypeError, None),
    'type-error-operation': (10, TypeError, None),
    'type-error-not-iterable': (11, TypeError, None),
    'type-error-isinstance': (12, TypeError, None),
    'type-error-not-subscriptable': (13, TypeError, None),
    'value-error-too-many-values': (14, ValueError, None),
    'value-error-too-few-values': (15, ValueError, None)
}

class Error:
    def __init__(self, name: str, module_path: str, start_pos: Tuple[int, int], message: Optional[str] = None) -> None:
        self.path: str = module_path
        self._start_pos: Tuple[int, int] = start_pos
        self.name: str = name
        if message is None:
            message = CODES[self.name][2]
        self.message: Optional[str] = message

    @property
    def line(self) -> int:
        return self._start_pos[0]

    @property
    def column(self) -> int:
        return self._start_pos[1]

    @property
    def code(self) -> str:
        first: str = self.__class__.__name__[0]
        return first + str(CODES[self.name][0])

    def __str__(self) -> str:
        return '%s:%s:%s: %s %s' % (self.path, self.line, self.column, self.code, self.message)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Error) and 
            self.path == other.path and 
            self.name == other.name and 
            (self._start_pos == other._start_pos)
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.path, self._start_pos, self.name))

    def __repr__(self) -> str:
        return '<%s %s: %s@%s,%s>' % (self.__class__.__name__, self.name, self.path, self._start_pos[0], self._start_pos[1])

class Warning(Error):
    pass

def add(node_context: Any, error_name: str, node: Any, message: Optional[str] = None, typ: Type[Error] = Error, payload: Optional[Any] = None) -> Optional[Error]:
    exception: Type[Exception] = CODES[error_name][1]
    if _check_for_exception_catch(node_context, node, exception, payload):
        return None
    module_context: Any = node_context.get_root_context()
    module_path: str = module_context.py__file__()
    issue_instance: Error = typ(error_name, module_path, node.start_pos, message)
    debug.warning(str(issue_instance), format=False)
    node_context.inference_state.analysis.append(issue_instance)
    return issue_instance

def _check_for_setattr(instance: Any) -> bool:
    """
    Check if there's any setattr method inside an instance. If so, return True.
    """
    module: Any = instance.get_root_context()
    node: Any = module.tree_node
    if node is None:
        return False
    try:
        stmt_names: Any = node.get_used_names()['setattr']
    except KeyError:
        return False
    return any(
        (node.start_pos < n.start_pos < node.end_pos and 
         (not (n.parent.type == 'funcdef' and n.parent.name == n)))
        for n in stmt_names
    )

def add_attribute_error(name_context: Any, lookup_value: Any, name: Any) -> None:
    message: str = 'AttributeError: %s has no attribute %s.' % (lookup_value, name)
    typ: Type[Error] = Error
    if lookup_value.is_instance() and (not lookup_value.is_compiled()):
        if _check_for_setattr(lookup_value):
            typ = Warning
    payload: tuple[Any, Any] = (lookup_value, name)
    add(name_context, 'attribute-error', name, message, typ, payload)

def _check_for_exception_catch(node_context: Any, jedi_name: Any, exception: Exception, payload: Optional[Any] = None) -> bool:
    """
    Checks if a jedi object (e.g. `Statement`) sits inside a try/catch and
    doesn't count as an error (if equal to `exception`).
    Also checks `hasattr` for AttributeErrors and uses the `payload` to compare
    it.
    Returns True if the exception was caught.
    """

    def check_match(cls: Any, exception: Exception) -> bool:
        if not cls.is_class():
            return False
        for python_cls in exception.mro():
            if cls.py__name__() == python_cls.__name__ and cls.parent_context.is_builtins_module():
                return True
        return False

    def check_try_for_except(obj: Any, exception: Exception) -> bool:
        iterator = iter(obj.children)
        for branch_type in iterator:
            next(iterator)
            suite = next(iterator)
            if branch_type == 'try' and (not branch_type.start_pos < jedi_name.start_pos <= suite.end_pos):
                return False
        for node in obj.get_except_clause_tests():
            if node is None:
                return True
            else:
                except_classes = node_context.infer_node(node)
                for cls in except_classes:
                    from jedi.inference.value import iterable
                    if isinstance(cls, iterable.Sequence) and cls.array_type == 'tuple':
                        for lazy_value in cls.py__iter__():
                            for typ_ in lazy_value.infer():
                                if check_match(typ_, exception):
                                    return True
                    elif check_match(cls, exception):
                        return True
        return False

    def check_hasattr(node: Any, suite: Any) -> bool:
        try:
            assert suite.start_pos <= jedi_name.start_pos < suite.end_pos
            assert node.type in ('power', 'atom_expr')
            base = node.children[0]
            assert base.type == 'name' and base.value == 'hasattr'
            trailer = node.children[1]
            assert trailer.type == 'trailer'
            arglist = trailer.children[1]
            assert arglist.type == 'arglist'
            from jedi.inference.arguments import TreeArguments
            args = TreeArguments(node_context.inference_state, node_context, arglist)
            unpacked_args = list(args.unpack())
            assert len(unpacked_args) == 2
            key, lazy_value = unpacked_args[1]
            names = list(lazy_value.infer())
            assert len(names) == 1 and is_string(names[0])
            assert names[0].get_safe_value() == payload[1].value
            key, lazy_value = unpacked_args[0]
            objects = lazy_value.infer()
            return payload[0] in objects
        except AssertionError:
            return False

    obj: Any = jedi_name
    while obj is not None and (not isinstance(obj, (tree.Function, tree.Class))):
        if isinstance(obj, tree.Flow):
            if obj.type == 'try_stmt' and check_try_for_except(obj, exception):
                return True
            if exception == AttributeError and obj.type in ('if_stmt', 'while_stmt'):
                if check_hasattr(obj.children[1], obj.children[3]):
                    return True
        obj = obj.parent
    return False
