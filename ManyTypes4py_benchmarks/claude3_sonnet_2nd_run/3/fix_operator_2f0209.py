"""Fixer for operator functions.

operator.isCallable(obj)       -> hasattr(obj, '__call__')
operator.sequenceIncludes(obj) -> operator.contains(obj)
operator.isSequenceType(obj)   -> isinstance(obj, collections.Sequence)
operator.isMappingType(obj)    -> isinstance(obj, collections.Mapping)
operator.isNumberType(obj)     -> isinstance(obj, numbers.Number)
operator.repeat(obj, n)        -> operator.mul(obj, n)
operator.irepeat(obj, n)       -> operator.imul(obj, n)
"""
import collections
from typing import Any, Callable, Optional, Dict, Union
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from lib2to3.pytree import Node

def invocation(s: str) -> Callable[[Callable], Callable]:
    def dec(f: Callable) -> Callable:
        f.invocation = s  # type: ignore
        return f
    return dec

class FixOperator(fixer_base.BaseFix):
    BM_compatible = True
    order = 'pre'
    methods = "\n              method=('isCallable'|'sequenceIncludes'\n                     |'isSequenceType'|'isMappingType'|'isNumberType'\n                     |'repeat'|'irepeat')\n              "
    obj = "'(' obj=any ')'"
    PATTERN = "\n              power< module='operator'\n                trailer< '.' %(methods)s > trailer< %(obj)s > >\n              |\n              power< %(methods)s trailer< %(obj)s > >\n              " % dict(methods=methods, obj=obj)

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)
        return None

    @invocation('operator.contains(%s)')
    def _sequenceIncludes(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'contains')

    @invocation("hasattr(%s, '__call__')")
    def _isCallable(self, node: Node, results: Dict[str, Any]) -> Node:
        obj = results['obj']
        args = [obj.clone(), String(', '), String("'__call__'")]
        return Call(Name('hasattr'), args, prefix=node.prefix)

    @invocation('operator.mul(%s)')
    def _repeat(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'mul')

    @invocation('operator.imul(%s)')
    def _irepeat(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_rename(node, results, 'imul')

    @invocation('isinstance(%s, collections.Sequence)')
    def _isSequenceType(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_type2abc(node, results, 'collections', 'Sequence')

    @invocation('isinstance(%s, collections.Mapping)')
    def _isMappingType(self, node: Node, results: Dict[str, Any]) -> Node:
        return self._handle_type2abc(node, results, 'collections', 'Mapping')

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
        args = [obj.clone(), String(', ' + '.'.join([module, abc]))]
        return Call(Name('isinstance'), args, prefix=node.prefix)

    def _check_method(self, node: Node, results: Dict[str, Any]) -> Optional[Callable[[Node, Dict[str, Any]], Node]]:
        method = getattr(self, '_' + results['method'][0].value)
        if isinstance(method, collections.Callable):
            if 'module' in results:
                return method
            else:
                sub = (str(results['obj']),)
                invocation_str = method.invocation % sub  # type: ignore
                self.warning(node, "You should use '%s' here." % invocation_str)
        return None
