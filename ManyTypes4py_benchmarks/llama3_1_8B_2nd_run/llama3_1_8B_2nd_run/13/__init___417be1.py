class InferenceState:
    def __init__(self, project: object, environment: object = None, script_path: str = None) -> None:
        if environment is None:
            environment = project.get_environment()
        self.environment: object = environment
        self.script_path: str = script_path
        self.compiled_subprocess: object = environment.get_inference_state_subprocess(self)
        self.grammar: object = environment.get_grammar()
        self.latest_grammar: object = parso.load_grammar(version='3.7')
        self.memoize_cache: dict = {}
        self.module_cache: imports.ModuleCache = imports.ModuleCache()
        self.stub_module_cache: dict = {}
        self.compiled_cache: dict = {}
        self.inferred_element_counts: dict = {}
        self.mixed_cache: dict = {}
        self.analysis: list = []
        self.dynamic_params_depth: int = 0
        self.is_analysis: bool = False
        self.project: object = project
        self.access_cache: dict = {}
        self.allow_descriptor_getattr: bool = False
        self.flow_analysis_enabled: bool = True
        self.reset_recursion_limitations()

    def import_module(self, import_names: tuple, sys_path: tuple = None, prefer_stubs: bool = True) -> object:
        return imports.import_module_by_names(self, import_names, sys_path, prefer_stubs=prefer_stubs)

    @staticmethod
    @plugin_manager.decorate()
    def execute(value: object, arguments: object) -> ValueSet:
        debug.dbg('execute: %s %s', value, arguments)
        with debug.increase_indent_cm():
            value_set = value.py__call__(arguments=arguments)
        debug.dbg('execute result: %s in %s', value_set, value)
        return value_set

    @property
    @inference_state_function_cache()
    def builtins_module(self) -> object:
        module_name: str = 'builtins'
        builtins_module, = self.import_module((module_name,), sys_path=())
        return builtins_module

    @property
    @inference_state_function_cache()
    def typing_module(self) -> object:
        typing_module, = self.import_module(('typing',))
        return typing_module

    def reset_recursion_limitations(self) -> None:
        self.recursion_detector: recursion.RecursionDetector = recursion.RecursionDetector()
        self.execution_recursion_detector: recursion.ExecutionRecursionDetector = recursion.ExecutionRecursionDetector(self)

    def get_sys_path(self, **kwargs) -> tuple:
        """Convenience function"""
        return self.project._get_sys_path(self, **kwargs)

    def infer(self, context: object, name: object) -> ValueSet:
        def_ = name.get_definition(import_name_always=True)
        if def_ is not None:
            type_ = def_.type
            is_classdef = type_ == 'classdef'
            if is_classdef or type_ == 'funcdef':
                if is_classdef:
                    c = ClassValue(self, context, name.parent)
                else:
                    c = FunctionValue.from_context(context, name.parent)
                return ValueSet([c])
            if type_ == 'expr_stmt':
                is_simple_name = name.parent.type not in ('power', 'trailer')
                if is_simple_name:
                    return infer_expr_stmt(context, def_, name)
            if type_ == 'for_stmt':
                container_types = context.infer_node(def_.children[3])
                cn = ContextualizedNode(context, def_.children[3])
                for_types = iterate_values(container_types, cn)
                n = TreeNameDefinition(context, name)
                return check_tuple_assignments(n, for_types)
            if type_ in ('import_from', 'import_name'):
                return imports.infer_import(context, name)
            if type_ == 'with_stmt':
                return tree_name_to_values(self, context, name)
            elif type_ == 'param':
                return context.py__getattribute__(name.value, position=name.end_pos)
            elif type_ == 'namedexpr_test':
                return context.infer_node(def_)
        else:
            result = follow_error_node_imports_if_possible(context, name)
            if result is not None:
                return result
        return helpers.infer_call_of_leaf(context, name)

    def parse_and_get_code(self, code: str = None, path: str = None, use_latest_grammar: bool = False, file_io: object = None, **kwargs) -> tuple:
        if code is None:
            if file_io is None:
                file_io = FileIO(path)
            code = file_io.read()
        code = parso.python_bytes_to_unicode(code, encoding='utf-8', errors='replace')
        if len(code) > settings._cropped_file_size:
            code = code[:settings._cropped_file_size]
        grammar = self.latest_grammar if use_latest_grammar else self.grammar
        return (grammar.parse(code=code, path=path, file_io=file_io, **kwargs), code)

    def parse(self, *args, **kwargs) -> object:
        return self.parse_and_get_code(*args, **kwargs)[0]
