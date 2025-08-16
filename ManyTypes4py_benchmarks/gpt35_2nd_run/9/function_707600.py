from typing import List, Union

class LambdaName(AbstractNameDefinition):
    string_name: str = '<lambda>'
    api_type: str = 'function'

    def __init__(self, lambda_value: TreeValue) -> None:
        self._lambda_value: TreeValue = lambda_value
        self.parent_context: ValueContext = lambda_value.parent_context

    @property
    def start_pos(self) -> int:
        return self._lambda_value.tree_node.start_pos

    def infer(self) -> ValueSet:
        return ValueSet([self._lambda_value])

class FunctionAndClassBase(TreeValue):

    def get_qualified_names(self) -> Union[None, List[str]]:
        if self.parent_context.is_class():
            n = self.parent_context.get_qualified_names()
            if n is None:
                return None
            return n + (self.py__name__(),)
        elif self.parent_context.is_module():
            return (self.py__name__(),)
        else:
            return None

class FunctionMixin:
    api_type: str = 'function'

    def get_filters(self, origin_scope=None) -> Generator:
        cls = self.py__class__()
        for instance in cls.execute_with_values():
            yield from instance.get_filters(origin_scope=origin_scope)

    def py__get__(self, instance, class_value) -> ValueSet:
        from jedi.inference.value.instance import BoundMethod
        if instance is None:
            return ValueSet([self])
        return ValueSet([BoundMethod(instance, class_value.as_context(), self)])

    def get_param_names(self) -> List[AnonymousParamName]:
        return [AnonymousParamName(self, param.name) for param in self.tree_node.get_params()]

    @property
    def name(self) -> ValueName:
        if self.tree_node.type == 'lambdef':
            return LambdaName(self)
        return ValueName(self, self.tree_node.name)

    def is_function(self) -> bool:
        return True

    def py__name__(self) -> str:
        return self.name.string_name

    def get_type_hint(self, add_class_info=True) -> str:
        return_annotation = self.tree_node.annotation
        if return_annotation is None:

            def param_name_to_str(n) -> str:
                s = n.string_name
                annotation = n.infer().get_type_hint()
                if annotation is not None:
                    s += ': ' + annotation
                if n.default_node is not None:
                    s += '=' + n.default_node.get_code(include_prefix=False)
                return s
            function_execution = self.as_context()
            result = function_execution.infer()
            return_hint = result.get_type_hint()
            body = self.py__name__() + '(%s)' % ', '.join([param_name_to_str(n) for n in function_execution.get_param_names()])
            if return_hint is None:
                return body
        else:
            return_hint = return_annotation.get_code(include_prefix=False)
            body = self.py__name__() + self.tree_node.children[2].get_code(include_prefix=False)
        return body + ' -> ' + return_hint

    def py__call__(self, arguments) -> ValueSet:
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def _as_context(self, arguments=None) -> Union[AnonymousFunctionExecution, FunctionExecutionContext]:
        if arguments is None:
            return AnonymousFunctionExecution(self)
        return FunctionExecutionContext(self, arguments)

    def get_signatures(self) -> List[TreeSignature]:
        return [TreeSignature(f) for f in self.get_signature_functions()]

class FunctionValue(FunctionMixin, FunctionAndClassBase, metaclass=CachedMetaClass):

    @classmethod
    def from_context(cls, context, tree_node) -> Union[FunctionValue, OverloadedFunctionValue]:
        ...

    def py__class__(self) -> ValueSet:
        c, = values_from_qualified_names(self.inference_state, 'types', 'FunctionType')
        return c

    def get_default_param_context(self) -> ValueContext:
        return self.parent_context

    def get_signature_functions(self) -> List[FunctionValue]:
        return [self]

class FunctionNameInClass(NameWrapper):

    def __init__(self, class_context, name) -> None:
        super().__init__(name)
        self._class_context: ValueContext = class_context

    def get_defining_qualified_value(self) -> ValueSet:
        return self._class_context.get_value()

class MethodValue(FunctionValue):

    def __init__(self, inference_state, class_context, *args, **kwargs) -> None:
        super().__init__(inference_state, *args, **kwargs)
        self.class_context: ValueContext = class_context

    def get_default_param_context(self) -> ValueContext:
        return self.class_context

    def get_qualified_names(self) -> Union[None, List[str]]:
        ...

    @property
    def name(self) -> FunctionNameInClass:
        return FunctionNameInClass(self.class_context, super().name)

class BaseFunctionExecutionContext(ValueContext, TreeContextMixin):

    def infer_annotations(self) -> ValueSet:
        raise NotImplementedError

    def get_return_values(self, check_yields=False) -> ValueSet:
        ...

    def _get_yield_lazy_value(self, yield_expr) -> Generator:
        ...

    def get_yield_lazy_values(self, is_async=False) -> Generator:
        ...

    def merge_yield_values(self, is_async=False) -> ValueSet:
        ...

    def is_generator(self) -> bool:
        ...

    def infer(self) -> ValueSet:
        ...

class FunctionExecutionContext(BaseFunctionExecutionContext):

    def __init__(self, function_value, arguments) -> None:
        super().__init__(function_value)
        self._arguments = arguments

    def get_filters(self, until_position=None, origin_scope=None) -> Generator:
        ...

    def infer_annotations(self) -> ValueSet:
        ...

    def get_param_names(self) -> List[ParamName]:
        ...

class AnonymousFunctionExecution(BaseFunctionExecutionContext):

    def infer_annotations(self) -> ValueSet:
        ...

    def get_filters(self, until_position=None, origin_scope=None) -> Generator:
        ...

    def get_param_names(self) -> List[ParamName]:
        ...

class OverloadedFunctionValue(FunctionMixin, ValueWrapper):

    def __init__(self, function, overloaded_functions) -> None:
        super().__init__(function)
        self._overloaded_functions: List[FunctionValue] = overloaded_functions

    def py__call__(self, arguments) -> ValueSet:
        ...

    def get_signature_functions(self) -> List[FunctionValue]:
        ...

    def get_type_hint(self, add_class_info=True) -> str:
        return 'Union[%s]' % ', '.join((f.get_type_hint() for f in self._overloaded_functions))

def _find_overload_functions(context, tree_node) -> Generator:
    ...
