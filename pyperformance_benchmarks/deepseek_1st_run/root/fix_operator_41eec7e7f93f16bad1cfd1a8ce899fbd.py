'Fixer for operator functions.\n\noperator.isCallable(obj)       -> callable(obj)\noperator.sequenceIncludes(obj) -> operator.contains(obj)\noperator.isSequenceType(obj)   -> isinstance(obj, collections.abc.Sequence)\noperator.isMappingType(obj)    -> isinstance(obj, collections.abc.Mapping)\noperator.isNumberType(obj)     -> isinstance(obj, numbers.Number)\noperator.repeat(obj, n)        -> operator.mul(obj, n)\noperator.irepeat(obj, n)       -> operator.imul(obj, n)\n'
import collections.abc
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from lib2to3.pytree import Node
from lib2to3.pygram import python_symbols as syms
from typing import Any, Dict, Optional, Callable as TypedCallable, Union

def invocation(s: str) -> TypedCallable[[TypedCallable], TypedCallable]:
    def dec(f: TypedCallable) -> TypedCallable:
        f.invocation = s
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
        if (method is not None):
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

    def _check_method(self, node: Node, results: Dict[str, Any]) -> Optional[TypedCallable[[Node, Dict[str, Any]], Node]]:
        method_name = '_' + results['method'][0].value
        method = getattr(self, method_name, None)
        if callable(method):
            if ('module' in results):
                return method
            else:
                sub = (str(results['obj']),)
                invocation_str = (method.invocation % sub)
                self.warning(node, ("You should use '%s' here." % invocation_str))
        return None
