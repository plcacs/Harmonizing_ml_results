import collections.abc
from lib2to3 import fixer_base
from lib2to3.fixer_util import Call, Name, String, touch_import
from typing import Callable, Optional, Dict, Any

def invocation(s: str) -> Callable:
    def dec(f: Callable) -> Callable:
        f.invocation = s
        return f
    return dec

class FixOperator(fixer_base.BaseFix):
    BM_compatible: bool = True
    order: str = 'pre'
    methods: str = "\n              method=('isCallable'|'sequenceIncludes'\n                     |'isSequenceType'|'isMappingType'|'isNumberType'\n                     |'repeat'|'irepeat')\n              "
    obj: str = "'(' obj=any ')'"
    PATTERN: str = ("\n              power< module='operator'\n                trailer< '.' %(methods)s > trailer< %(obj)s > >\n              |\n              power< %(methods)s trailer< %(obj)s > >\n              " % dict(methods=methods, obj=obj))

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        method = self._check_method(node, results)
        if method is not None:
            return method(node, results)
        return None

    @invocation('operator.contains(%s)')
    def _sequenceIncludes(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_rename(node, results, 'contains')

    @invocation('callable(%s)')
    def _isCallable(self, node: Any, results: Dict[str, Any]) -> Any:
        obj = results['obj']
        return Call(Name('callable'), [obj.clone()], prefix=node.prefix)

    @invocation('operator.mul(%s)')
    def _repeat(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_rename(node, results, 'mul')

    @invocation('operator.imul(%s)')
    def _irepeat(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_rename(node, results, 'imul')

    @invocation('isinstance(%s, collections.abc.Sequence)')
    def _isSequenceType(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_type2abc(node, results, 'collections.abc', 'Sequence')

    @invocation('isinstance(%s, collections.abc.Mapping)')
    def _isMappingType(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_type2abc(node, results, 'collections.abc', 'Mapping')

    @invocation('isinstance(%s, numbers.Number)')
    def _isNumberType(self, node: Any, results: Dict[str, Any]) -> Any:
        return self._handle_type2abc(node, results, 'numbers', 'Number')

    def _handle_rename(self, node: Any, results: Dict[str, Any], name: str) -> None:
        method = results['method'][0]
        method.value = name
        method.changed()

    def _handle_type2abc(self, node: Any, results: Dict[str, Any], module: str, abc: str) -> Any:
        touch_import(None, module, node)
        obj = results['obj']
        args = [obj.clone(), String((', ' + '.'.join([module, abc])))]
        return Call(Name('isinstance'), args, prefix=node.prefix)

    def _check_method(self, node: Any, results: Dict[str, Any]) -> Optional[Callable]:
        method = getattr(self, ('_' + results['method'][0].value))
        if isinstance(method, collections.abc.Callable):
            if 'module' in results:
                return method
            else:
                sub = (str(results['obj']),)
                invocation_str = (method.invocation % sub)
                self.warning(node, ("You should use '%s' here." % invocation_str))
        return None
