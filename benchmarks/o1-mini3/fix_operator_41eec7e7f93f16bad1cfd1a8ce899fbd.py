'Fixer for operator functions.\n\noperator.isCallable(obj)       -> callable(obj)\noperator.sequenceIncludes(obj) -> operator.contains(obj)\noperator.isSequenceType(obj)   -> isinstance(obj, collections.abc.Sequence)\noperator.isMappingType(obj)    -> isinstance(obj, collections.abc.Mapping)\noperator.isNumberType(obj)     -> isinstance(obj, numbers.Number)\noperator.repeat(obj, n)        -> operator.mul(obj, n)\noperator.irepeat(obj, n)       -> operator.imul(obj, n)\n'
import collections.abc
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from typing import Callable, Any, Optional

def invocation(s: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def dec(f: Callable[..., Any]) -> Callable[..., Any]:
        f.invocation = s
        return f
    return dec

class FixOperator(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    methods: str = "\n              method=('isCallable'|'sequenceIncludes'\n                     |'isSequenceType'|'isMappingType'|'isNumberType'\n                     |'repeat'|'irepeat')\n              "
    obj: str = "'(' obj=any ')'"
    PATTERN: str = ("\n              power< module='operator'\n                trailer< '.' %(methods)s > trailer< %(obj)s > >\n              |\n              power< %(methods)s trailer< %(obj)s > >\n              " % dict(methods=methods, obj=obj))

    def transform(self, node: Any, results: Any) -> Optional[Any]:
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)

    @invocation('operator.contains(%s)')
    def _sequenceIncludes(self, node: Any, results: Any) -> Any:
        return self._handle_rename(node, results, 'contains')

    @invocation('callable(%s)')
    def _isCallable(self, node: Any, results: Any) -> Any:
        obj = results['obj']
        return Call(Name('callable'), [obj.clone()], prefix=node.prefix)

    @invocation('operator.mul(%s)')
    def _repeat(self, node: Any, results: Any) -> Any:
        return self._handle_rename(node, results, 'mul')

    @invocation('operator.imul(%s)')
    def _irepeat(self, node: Any, results: Any) -> Any:
        return self._handle_rename(node, results, 'imul')

    @invocation('isinstance(%s, collections.abc.Sequence)')
    def _isSequenceType(self, node: Any, results: Any) -> Any:
        return self._handle_type2abc(node, results, 'collections.abc', 'Sequence')

    @invocation('isinstance(%s, collections.abc.Mapping)')
    def _isMappingType(self, node: Any, results: Any) -> Any:
        return self._handle_type2abc(node, results, 'collections.abc', 'Mapping')

    @invocation('isinstance(%s, numbers.Number)')
    def _isNumberType(self, node: Any, results: Any) -> Any:
        return self._handle_type2abc(node, results, 'numbers', 'Number')

    def _handle_rename(self, node: Any, results: Any, name: str) -> Any:
        method = results['method'][0]
        method.value = name
        method.changed()

    def _handle_type2abc(self, node: Any, results: Any, module: str, abc: str) -> Any:
        touch_import(None, module, node)
        obj = results['obj']
        args = [obj.clone(), String(', '.join([module, abc]))]
        return Call(Name('isinstance'), args, prefix=node.prefix)

    def _check_method(self, node: Any, results: Any) -> Optional[Callable[..., Any]]:
        method = getattr(self, '_' + results['method'][0].value)
        if isinstance(method, collections.abc.Callable):
            if 'module' in results:
                return method
            else:
                sub = (str(results['obj']), )
                invocation_str = method.invocation % sub
                self.warning(node, f"You should use '{invocation_str}' here.")
        return None
