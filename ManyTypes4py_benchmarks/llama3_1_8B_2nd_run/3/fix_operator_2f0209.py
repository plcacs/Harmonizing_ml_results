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
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from typing import Callable, Dict, List, Optional, Tuple

def invocation(s: str) -> Callable:
    """Decorate a function to store its invocation string."""
    def dec(f: Callable) -> Callable:
        f.invocation = s
        return f
    return dec

class FixOperator(fixer_base.BaseFix):
    """Fixer for operator functions."""
    BM_compatible: bool = True
    order: str = 'pre'
    methods: str = "\n              method=('isCallable'|'sequenceIncludes'\n                     |'isSequenceType'|'isMappingType'|'isNumberType'\n                     |'repeat'|'irepeat')\n              "
    obj: str = "'(' obj=any ')'"
    PATTERN: str = "\n              power< module='operator'\n                trailer< '.' %(methods)s > trailer< %(obj)s > >\n              |\n              power< %(methods)s trailer< %(obj)s > >\n              " % dict(methods=methods, obj=obj)

    def transform(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> Optional[fixer_base.Node]:
        """Transform the node."""
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)

    @invocation('operator.contains(%s)')
    def _sequenceIncludes(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle sequenceIncludes."""
        return self._handle_rename(node, results, 'contains')

    @invocation("hasattr(%s, '__call__')")
    def _isCallable(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle isCallable."""
        obj: fixer_base.Node = results['obj']
        args: List[fixer_base.Node] = [obj.clone(), String(', '), String("'__call__'")]
        return Call(Name('hasattr'), args, prefix=node.prefix)

    @invocation('operator.mul(%s)')
    def _repeat(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle repeat."""
        return self._handle_rename(node, results, 'mul')

    @invocation('operator.imul(%s)')
    def _irepeat(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle irepeat."""
        return self._handle_rename(node, results, 'imul')

    @invocation('isinstance(%s, collections.Sequence)')
    def _isSequenceType(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle isSequenceType."""
        return self._handle_type2abc(node, results, 'collections', 'Sequence')

    @invocation('isinstance(%s, collections.Mapping)')
    def _isMappingType(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle isMappingType."""
        return self._handle_type2abc(node, results, 'collections', 'Mapping')

    @invocation('isinstance(%s, numbers.Number)')
    def _isNumberType(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> fixer_base.Node:
        """Handle isNumberType."""
        return self._handle_type2abc(node, results, 'numbers', 'Number')

    def _handle_rename(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node], name: str) -> fixer_base.Node:
        """Handle renaming."""
        method: fixer_base.Node = results['method'][0]
        method.value = name
        method.changed()
        return node

    def _handle_type2abc(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node], module: str, abc: str) -> fixer_base.Node:
        """Handle type to abc."""
        touch_import(None, module, node)
        obj: fixer_base.Node = results['obj']
        args: List[fixer_base.Node] = [obj.clone(), String(', ' + '.'.join([module, abc]))]
        return Call(Name('isinstance'), args, prefix=node.prefix)

    def _check_method(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> Optional[Callable[[fixer_base.Node, Dict[str, fixer_base.Node]], fixer_base.Node]]:
        """Check the method."""
        method: Optional[Callable[[fixer_base.Node, Dict[str, fixer_base.Node]], fixer_base.Node]] = getattr(self, '_' + results['method'][0].value)
        if isinstance(method, collections.Callable):
            if 'module' in results:
                return method
            else:
                sub: Tuple[str, ...] = (str(results['obj']),)
                invocation_str: str = method.invocation % sub
                self.warning(node, "You should use '%s' here." % invocation_str)
        return None
