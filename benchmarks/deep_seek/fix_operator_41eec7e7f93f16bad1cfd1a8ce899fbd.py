"""Fixer for operator functions.

operator.isCallable(obj)       -> callable(obj)
operator.sequenceIncludes(obj) -> operator.contains(obj)
operator.isSequenceType(obj)   -> isinstance(obj, collections.abc.Sequence)
operator.isMappingType(obj)    -> isinstance(obj, collections.abc.Mapping)
operator.isNumberType(obj)     -> isinstance(obj, numbers.Number)
operator.repeat(obj, n)        -> operator.mul(obj, n)
operator.irepeat(obj, n)       -> operator.imul(obj, n)
"""
import collections.abc
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from lib2to3.pytree import Node, Leaf
from lib2to3.pgen2 import token
from typing import Any, Dict, Optional, Callable as CallableT, TypeVar, Union

T = TypeVar('T')

def invocation(s: str) -> CallableT[[T], T]:
    def dec(f: T) -> T:
        setattr(f, 'invocation', s)
        return f
    return dec

class FixOperator(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    methods: str = "\n              method=('isCallable'|'sequenceIncludes'\n                     |'isSequenceType'|'isMappingType'|'isNumberType'\n                     |'repeat'|'irepeat')\n              "
    obj: str = "'(' obj=any ')'"
    PATTERN: str = ("\n              power< module='operator'\n                trailer< '.' %(methods)s > trailer< %(obj)s > >\n              |\n              power< %(methods)s trailer< %(obj)s > >\n              " % dict(methods=methods, obj=obj))

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)
        return None

    @invocation('operator.contains(%s)')
    def _sequenceIncludes(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'contains')

    @invocation('callable(%s)')
    def _isCallable(self, node: Node, results: Dict[str, Any]) -> Node:
        obj = results['obj']
        return Call(Name('callable'), [obj.clone()], prefix=node.prefix)

    @invocation('operator.mul(%s)')
    def _repeat(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'mul')

    @invocation('operator.imul(%s)')
    def _irepeat(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'imul')

    @invocation('isinstance(%s, collections.abc.Sequence)')
    def _isSequenceType(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_type2abc(node, results, 'collections.abc', 'Sequence')

    @invocation('isinstance(%s, collections.abc.Mapping)')
    def _isMappingType(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_type2abc(node, results, 'collections.abc', 'Mapping')

    @invocation('isinstance(%s, numbers.Number)')
    def _isNumberType(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_type2abc(node, results, 'numbers', 'Number')

    def _handle_rename(self, node: Node, results: Dict[str, Any], name: str) -> Node:
        method = results['method'][0]
        method.value = name
        method.changed()
        return node

    def _handle_type2abc(self, node: Node, results: Dict[str, Any], module: str, abc: str) -> Node:
        touch_import(None, module, node)
        obj = results['obj']
        args = [obj.clone(), String((', ' + '.'.join([module, abc])))]
        return Call(Name('isinstance'), args, prefix=node.prefix)

    def _check_method(self, node: Node, results: Dict[str, Any]) -> Optional[CallableT[[Node, Dict[str, Any]], Node]]:
        method_name = '_' + results['method'][0].value
        method = getattr(self, method_name, None)
        if callable(method):
            if 'module' in results:
                return method
            else:
                sub = (str(results['obj']),)
                invocation_str = (getattr(method, 'invocation', '') % sub)
                self.warning(node, f"You should use '{invocation_str}' here.")
        return None
