import os
import os.path
import sys
import ast
import re
import copy
import datetime
import math
import traceback
import io
import subprocess
import shlex
import shutil
import tokenize
import collections
import json
from contextlib import contextmanager, ExitStack
from org.transcrypt import utils, sourcemaps, minify, static_check, type_check

inIf: bool = False
ecom: bool = True
noecom: bool = False
dataClassDefaultArgTuple: tuple = (['init', True], ['repr', True], ['eq', True], ['order', False], ['unsafe_hash', False], ['frozen', False])

class Program:
    def __init__(self, module_search_dirs: list, symbols: dict, envir: object) -> None:
        utils.set_program(self)
        self.module_search_dirs: list = module_search_dirs
        self.symbols: dict = symbols
        self.envir: object = envir
        self.javascript_version: int = int(utils.command_args.esv) if utils.command_args.esv else 6
        self.module_dict: dict = {}
        self.import_stack: list = []
        self.source_prepath: str = os.path.abspath(utils.command_args.source).replace('\\', '/')
        self.source_dir: str = '/'.join(self.source_prepath.split('/')[:-1])
        self.main_module_name: str = self.source_prepath.split('/')[-1]
        if utils.command_args.outdir:
            if os.path.isabs(utils.command_args.outdir):
                self.target_dir: str = utils.command_args.outdir.replace('\\', '/')
            else:
                self.target_dir: str = f'{self.source_dir}/{utils.command_args.outdir}'.replace('\\', '/')
        else:
            self.target_dir: str = f'{self.source_dir}/__target__'.replace('\\', '/')
        self.project_path: str = f'{self.target_dir}/{self.main_module_name}.project'
        try:
            with open(self.project_path, 'r') as project_file:
                project: dict = json.load(project_file)
        except:
            project: dict = {}
        self.options_changed: bool = utils.command_args.project_options != project.get('options')
        if utils.command_args.build or self.options_changed:
            shutil.rmtree(self.target_dir, ignore_errors=True)
        try:
            self.runtime_module_name: str = 'org.transcrypt.__runtime__'
            self.searched_module_paths: list = []
            self.provide(self.runtime_module_name)
            self.searched_module_paths: list = []
            self.provide(self.main_module_name, '__main__')
        except Exception as exception:
            utils.enhance_exception(exception, message=f'\n\t{exception}')
        project = {'options': utils.command_args.project_options, 'modules': [{'source': module.source_path, 'target': module.target_path} for module in self.module_dict.values()]}
        with utils.create(self.project_path) as project_file:
            json.dump(project, project_file)

    def provide(self, module_name: str, __module_name__: str = None, filter: callable = None) -> object:
        if module_name in self.module_dict:
            return self.module_dict[module_name]
        else:
            return Module(self, module_name, __module_name__, filter)

class Module:
    def __init__(self, program: object, name: str, __name__: str = None, filter: callable = None) -> None:
        self.program: object = program
        self.name: str = name
        self.__name__: str = __name__ if __name__ else self.name
        self.find_paths(filter)
        self.program.import_stack.append([self, None])
        self.program.module_dict[self.name] = self
        self.source_mapper: object = sourcemaps.SourceMapper(self.name, self.program.target_dir, not utils.command_args.nomin, utils.command_args.dmap)
        if utils.command_args.build or self.program.options_changed or (not os.path.isfile(self.target_path)) or (os.path.getmtime(self.source_path) > os.path.getmtime(self.target_path)):
            if self.is_javascript_only:
                self.load_javascript()
                javascript_digest = utils.digest_javascript(self.target_code, self.program.symbols, not utils.command_args.dnostrip, False)
            else:
                if utils.command_args.dstat:
                    try:
                        type_check.run(self.source_path)
                    except Exception as exception:
                        utils.log(True, 'Validating: {} and dependencies\n\tInternal error in static typing validator\n', self.source_path)
                self.parse()
                if utils.command_args.dtree:
                    self.dump_tree()
                if utils.command_args.dcheck:
                    try:
                        static_check.run(self.source_path, self.parse_tree)
                    except Exception as exception:
                        utils.log(True, 'Checking: {}\n\tInternal error in lightweight consistency checker, remainder of module skipped\n', self.source_path)
                self.generate_javascript_and_pretty_map()
                javascript_digest = utils.digest_javascript(self.target_code, self.program.symbols, False, self.generator.allow_debug_map)
            utils.log(True, 'Saving target code in: {}\n', self.target_path)
            file_path = self.target_path if utils.command_args.nomin else self.pretty_target_path
            with utils.create(file_path) as a_file:
                a_file.write(self.target_code)
            if not utils.command_args.nomin:
                utils.log(True, 'Saving minified target code in: {}\n', self.target_path)
                minify.run(self.program.target_dir, self.pretty_target_name, self.target_name, map_file_name=self.shrink_map_name if utils.command_args.map else None)
                if utils.command_args.map:
                    if self.is_javascript_only:
                        if os.path.isfile(self.map_path):
                            os.remove(self.map_path)
                        os.rename(self.shrink_map_path, self.map_path)
                    else:
                        self.source_mapper.generate_multilevel_map()
            with open(self.target_path, 'a') as target_file:
                target_file.write(self.map_ref)
        else:
            self.target_code = open(self.target_path, 'r').read()
            javascript_digest = utils.digest_javascript(self.target_code, self.program.symbols, True, False, refuse_if_appears_minified=True)
            if not javascript_digest:
                minify.run(self.program.target_dir, self.target_name, self.pretty_target_name, prettify=True)
                self.pretty_target_code = open(self.pretty_target_path, 'r').read()
                javascript_digest = utils.digest_javascript(self.pretty_target_code, self.program.symbols, True, False)
        self.target_code = javascript_digest.digested_code
        self.imported_module_names: list = javascript_digest.imported_module_names
        self.exported_names: list = javascript_digest.exported_names
        for imported_module_name in self.imported_module_names:
            self.program.searched_module_paths = []
            self.program.provide(imported_module_name)
        utils.try_remove(self.pretty_target_path)
        utils.try_remove(self.shrink_map_path)
        utils.try_remove(self.pretty_map_path)
        self.program.import_stack.pop()

    def find_paths(self, filter: callable) -> None:
        raw_rel_source_slug: str = self.name.replace('.', '/')
        rel_source_slug: str = filter(raw_rel_source_slug) if filter and utils.command_args.alimod else raw_rel_source_slug
        for search_dir in self.program.module_search_dirs:
            source_slug: str = f'{search_dir}/{rel_source_slug}'
            if os.path.isdir(source_slug):
                self.source_dir = source_slug
                self.source_prename = '__init__'
            else:
                self.source_dir, self.source_prename = source_slug.rsplit('/', 1)
            self.source_prepath = f'{self.source_dir}/{self.source_prename}'
            self.python_source_path = f'{self.source_prepath}.py'
            self.javascript_source_path = f'{self.source_prepath}.js'
            self.target_prepath = f'{self.program.target_dir}/{self.name}'
            self.target_name = f'{self.name}.js'
            self.target_path = f'{self.target_prepath}.js'
            self.pretty_target_name = f'{self.name}.pretty.js'
            self.pretty_target_path = f'{self.target_prepath}.pretty.js'
            self.import_rel_path = f'./{self.name}.js'
            self.tree_path = f'{self.target_prepath}.tree'
            self.map_path = f'{self.target_prepath}.map'
            self.pretty_map_path = f'{self.target_prepath}.shrink.map'
            self.shrink_map_name = f'{self.name}.shrink.map'
            self.shrink_map_path = f'{self.target_prepath}.shrink.map'
            self.map_source_path = f'{self.target_prepath}.py'
            self.map_ref = f'\n//# sourceMappingURL={self.name}.map'
            if os.path.isfile(self.python_source_path) or os.path.isfile(self.javascript_source_path):
                self.is_javascript_only = os.path.isfile(self.javascript_source_path) and (not os.path.isfile(self.python_source_path))
                self.source_path = self.javascript_source_path if self.is_javascript_only else self.python_source_path
                break
            self.program.searched_module_paths.extend([self.python_source_path, self.javascript_source_path])
        else:
            raise utils.Error(message="\n\tImport error, can't find any of:\n\t\t{}".format('\n\t\t'.join(self.program.searched_module_paths)))

    def generate_javascript_and_pretty_map(self) -> None:
        utils.log(False, 'Generating code for module: {}\n', self.target_path)
        self.generator = Generator(self)
        if utils.command_args.map or utils.command_args.anno:
            instrumented_target_lines = ''.join(self.generator.target_fragments).split('\n')
            if utils.command_args.map:
                self.source_line_nrs = []
            target_lines = []
            for target_line in instrumented_target_lines:
                source_line_nr_string = target_line[-sourcemaps.line_nr_length:]
                source_line_nr = int('1' + source_line_nr_string) - sourcemaps.max_nr_of_source_lines_per_module
                target_line = target_line[:-sourcemaps.line_nr_length]
                if target_line.strip() != ';':
                    if self.generator.allow_debug_map:
                        target_line = '/* {} */ {}'.format(source_line_nr_string, target_line)
                    target_lines.append(target_line)
                    if utils.command_args.map:
                        self.source_line_nrs.append(source_line_nr)
            if utils.command_args.map:
                utils.log(False, 'Saving source map in: {}\n', self.map_path)
                self.source_mapper.generate_and_save_pretty_map(self.source_line_nrs)
                shutil.copyfile(self.source_path, self.map_source_path)
        else:
            target_lines = [line for line in ''.join(self.generator.target_fragments).split('\n') if line.strip() != ';']
        self.target_code = '\n'.join(target_lines)

    def load_javascript(self) -> None:
        with tokenize.open(self.source_path) as source_file:
            self.target_code = source_file.read()

    def parse(self) -> None:
        def pragmas_from_comments(source_code: str) -> str:
            tokens = tokenize.tokenize(io.BytesIO(source_code.encode('utf-8')).readline)
            pragma_comment_line_indices = []
            short_pragma_comment_line_indices = []
            ecom_pragma_line_indices = []
            noecom_pragma_line_indices = []
            pragma_index = -1000
            for token_index, (token_type, token_string, start_row_column, end_row_column, logical_line) in enumerate(tokens):
                if token_type == tokenize.COMMENT:
                    stripped_comment = token_string[1:].lstrip()
                    if stripped_comment.startswith('__pragma__'):
                        pragma_comment_line_indices.append(start_row_column[0] - 1)
                    elif stripped_comment.replace(' ', '').replace('\t', '').startswith('__:'):
                        short_pragma_comment_line_indices.append(start_row_column[0] - 1)
                if token_type == tokenize.NAME and token_string == '__pragma__':
                    pragma_index = token_index
                if token_index - pragma_index == 2:
                    pragma_kind = token_string[1:-1]
                    if pragma_kind == 'ecom':
                        ecom_pragma_line_indices.append(start_row_column[0] - 1)
                    elif pragma_kind == 'noecom':
                        noecom_pragma_line_indices.append(start_row_column[0] - 1)
            source_lines = source_code.split('\n')
            for ecom_pragma_line_index in ecom_pragma_line_indices:
                source_lines[ecom_pragma_line_index] = ecom
            for noecom_pragma_line_index in noecom_pragma_line_indices:
                source_lines[noecom_pragma_line_index] = noecom
            allow_executable_comments = utils.command_args.ecom
            for pragma_comment_line_index in pragma_comment_line_indices:
                indentation, separator, tail = source_lines[pragma_comment_line_index].partition('#')
                pragma, separator, comment = tail.partition('#')
                pragma = pragma.replace(' ', '').replace('\t', '')
                if "('ecom')" in pragma or '("ecom")' in pragma:
                    allow_executable_comments = True
                    source_lines[pragma_comment_line_index] = ecom
                elif "('noecom')" in pragma or '("noecom")' in pragma:
                    allow_executable_comments = False
                    source_lines[pragma_comment_line_index] = noecom
                else:
                    source_lines[pragma_comment_line_index] = indentation + tail.lstrip()
            for short_pragma_comment_line_index in short_pragma_comment_line_indices:
                head, tail = source_lines[short_pragma_comment_line_index].rsplit('#', 1)
                stripped_head = head.lstrip()
                indent = head[:len(head) - len(stripped_head)]
                pragma_name = tail.replace(' ', '').replace('\t', '')[3:]
                if pragma_name == 'ecom':
                    source_lines[pragma_comment_line_index] = ecom
                elif pragma_name == 'noecom':
                    source_lines[short_pragma_comment_line_index] = noecom
                elif pragma_name.startswith('no'):
                    source_lines[short_pragma_comment_line_index] = "{}__pragma__ ('{}'); {}; __pragma__ ('{}')".format(indent, pragma_name, head, pragma_name[2:])
                else:
                    source_lines[short_pragma_comment_line_index] = "{}__pragma__ ('{}'); {}; __pragma__ ('no{}')".format(indent, pragma_name, head, pragma_name)
            uncommented_source_lines = []
            for source_line in source_lines:
                if source_line == ecom:
                    allow_executable_comments = True
                elif source_line == noecom:
                    allow_executable_comments = False
                elif allow_executable_comments:
                    l_stripped_source_line = source_line.lstrip()
                    if not l_stripped_source_line[:4] in {"'''?", "?'''", '"""?', '?"""'}:
                        uncommented_source_lines.append(source_line.replace('#?', '', 1) if l_stripped_source_line.startswith('#?') else source_line)
                else:
                    uncommented_source_lines.append(source_line)
            return '\n'.join(uncommented_source_lines)
        try:
            utils.log(False, 'Parsing module: {}\n', self.source_path)
            with tokenize.open(self.source_path) as source_file:
                self.source_code = utils.extra_lines + source_file.read()
            self.parse_tree = ast.parse(pragmas_from_comments(self.source_code))
            for node in ast.walk(self.parse_tree):
                for child_node in ast.iter_child_nodes(node):
                    child_node.parentNode = node
        except SyntaxError as syntax_error:
            utils.enhance_exception(syntax_error, line_nr=syntax_error.lineno, message='\n\t{} [<-SYNTAX FAULT] {}'.format(syntax_error.text[:syntax_error.offset].lstrip(), syntax_error.text[syntax_error.offset:].rstrip()) if syntax_error.text else syntax_error.args[0])

    def dump_tree(self) -> None:
        utils.log(False, 'Dumping syntax tree for module: {}\n', self.source_path)

        def walk(name: str, value: object, tab_level: int) -> None:
            self.tree_fragments.append('\n{0}{1}: {2} '.format(tab_level * '\t', name, type(value).__name__))
            if isinstance(value, ast.AST):
                for field in ast.iter_fields(value):
                    walk(field[0], field[1], tab_level + 1)
            elif isinstance(value, list):
                for element in value:
                    walk('element', element, tab_level + 1)
            else:
                self.tree_fragments.append('= {0}'.format(value))
        self.tree_fragments = []
        walk('file', self.parse_tree, 0)
        self.text_tree = ''.join(self.tree_fragments)[1:]
        with utils.create(self.tree_path) as tree_file:
            tree_file.write(self.text_tree)

class Generator(ast.NodeVisitor):
    def __init__(self, module: object) -> None:
        self.module: object = module
        self.target_fragments: list = []
        self.fragment_index: int = 0
        self.indent_level: int = 0
        self.scopes: list = []
        self.import_heads: set = set()
        self.import_hoist_memos: list = []
        self.all_own_names: set = set()
        self.all_imported_names: set = set()
        self.expecting_non_overloaded_lhs_index: bool = False
        self.line_nr: int = 1
        self.property_accessor_list: list = []
        self.merge_list: list = []
        self.aliases: list = [('js_and', 'and'), ('arguments', 'py_arguments'), ('js_arguments', 'arguments'), ('case', 'py_case'), ('clear', 'py_clear'), ('js_clear', 'clear'), ('js_conjugate', 'conjugate'), ('default', 'py_default'), ('del', 'py_del'), ('js_del', 'del'), ('false', 'py_false'), ('js_from', 'from'), ('get', 'py_get'), ('js_get', 'get'), ('js_global', 'global'), ('Infinity', 'py_Infinity'), ('js_Infinity', 'Infinity'), ('is', 'py_is'), ('js_is', 'is'), ('isNaN', 'py_isNaN'), ('js_isNaN', 'isNaN'), ('iter', 'py_iter'), ('js_iter', 'iter'), ('items', 'py_items'), ('js_items', 'items'), ('keys', 'py_keys'), ('js_keys', 'keys'), ('name', 'py_name'), ('js_name', 'name'), ('NaN', 'py_NaN'), ('js_NaN', 'NaN'), ('new', 'py_new'), ('next', 'py_next'), ('js_next', 'next'), ('js_not', 'not'), ('js_or', 'or'), ('pop', 'py_pop'), ('js_pop', 'pop'), ('popitem', 'py_popitem'), ('js_popitem', 'popitem'), ('replace', 'py_replace'), ('js_replace', 'replace'), ('selector', 'py_selector'), ('js_selector', 'selector'), ('sort', 'py_sort'), ('js_sort', 'sort'), ('split', 'py_split'), ('js_split', 'split'), ('switch', 'py_switch'), ('type', 'py_metatype'), ('js_type', 'type'), ('TypeError', 'py_TypeError'), ('js_TypeError', 'TypeError'), ('update', 'py_update'), ('js_update', 'update'), ('values', 'py_values'), ('js_values', 'values'), ('reversed', 'py_reversed'), ('js_reversed', 'reversed'), ('setdefault', 'py_setdefault'), ('js_setdefault', 'setdefault'), ('js_super', 'super'), ('true', 'py_true'), ('undefined', 'py_undefined'), ('js_undefined', 'undefined')]
        self.id_filtering: bool = True
        self.temp_indices: dict = {}
        self.skipped_temps: set = set()
        self.stubs_name: str = 'org.{}.stubs.'.format(self.module.program.envir.transpiler_name)
        self.name_consts: dict = {None: 'null', True: 'true', False: 'false'}
        self.operators: dict = {ast.Not: ('!', 16), ast.Invert: ('~', 16), ast.UAdd: ('+', 16), ast.USub: ('-', 16), ast.Pow: (None, 15), ast.Mult: ('*', 14), ast.MatMult: (None, 14), ast.Div: ('/', 14), ast.FloorDiv: (None, 14), ast.Mod: ('%', 14), ast.Add: ('+', 13), ast.Sub: ('-', 13), ast.LShift: ('<<', 12), ast.RShift: ('>>', 12), ast.Lt: ('<', 11), ast.LtE: ('<=', 11), ast.Gt: ('>', 11), ast.GtE: ('>=', 11), ast.In: (None, 11), ast.NotIn: (None, 11), ast.Eq: ('==', 10), ast.NotEq: ('!=', 10), ast.Is: ('===', 10), ast.IsNot: ('!==', 10), ast.BitAnd: ('&', 9), ast.BitOr: ('|', 8), ast.BitXor: ('^', 7), ast.And: ('&&', 6), ast.Or: ('||', 5)}
        self.allow_keyword_args: bool = utils.command_args.kwargs
        self.allow_operator_overloading: bool = utils.command_args.opov
        self.allow_conversion_to_iterable: bool = utils.command_args.iconv
        self.allow_conversion_to_truth_value: bool = utils.command_args.tconv
        self.allow_key_check: bool = utils.command_args.keycheck
        self.allow_debug_map: bool = utils.command_args.anno and (not self.module.source_path.endswith('.js'))
        self.allow_doc_attribs: bool = utils.command_args.docat
        self.allow_globals: bool = utils.command_args.xglobs
        self.allow_javascript_iter: bool = False
        self.allow_javascript_call: bool = utils.command_args.jscall
        self.allow_javascript_keys: bool = utils.command_args.jskeys
        self.allow_javascript_mod: bool = utils.command_args.jsmod
        self.allow_memoize_calls: bool = utils.command_args.fcall
        self.noskip_code_generation: bool = True
        self.conditional_code_generation: bool = True
        self.strip_tuple: bool = False
        self.strip_tuples: bool = False
        self.replace_send: bool = False
        try:
            self.visit(self.module.parse_tree)
            self.target_fragments.append(self.line_nr_string)
        except Exception as exception:
            utils.enhance_exception(exception, line_nr=self.line_nr)
        if self.temp_indices:
            raise utils.Error(message='\n\tTemporary variables leak in code generator: {}'.format(self.temp_indices))

    def visit_sub_expr(self, node: object, child: object) -> None:
        def get_priority(expr_node: object) -> int:
            if type(expr_node) in (ast.BinOp, ast.BoolOp):
                return self.operators[type(expr_node.op)][1]
            elif type(expr_node) == ast.Compare:
                return self.operators[type(expr_node.ops[0])][1]
            elif type(expr_node) == ast.Yield:
                return -1000000
            else:
                return 1000000
        if get_priority(child) <= get_priority(node):
            self.emit('(')
            self.visit(child)
            self.emit(')')
        else:
            self.visit(child)

    def filter_id(self, qualified_id: str) -> str:
        if not self.id_filtering or (qualified_id.startswith('__') and qualified_id.endswith('__')):
            return qualified_id
        else:
            for alias in self.aliases:
                qualified_id = re.sub(f'(^|(?P<pre_dunder>__)|(?<=[./])){alias[0]}((?P<post_dunder>__)|(?=[./])|$)', lambda match_object: ('=' if match_object.group('pre_dunder') else '') + alias[1] + ('=' if match_object.group('post_dunder') else ''), qualified_id)
                qualified_id = re.sub(f'(^|(?<=[./=])){alias[0]}((?=[./=])|$)', alias[1], qualified_id)
            return qualified_id.replace('=', '')

    def tabs(self, indent_level: int = None) -> str:
        if indent_level == None:
            indent_level = self.indent_level
        return indent_level * '\t'

    def emit(self, fragment: str, *formatter: object) -> None:
        if not self.target_fragments or (self.target_fragments and self.target_fragments[self.fragment_index - 1].endswith('\n')):
            self.target_fragments.insert(self.fragment_index, self.tabs())
            self.fragment_index += 1
        fragment = fragment[:-1].replace('\n', '\n' + self.tabs()) + fragment[-1]
        self.target_fragments.insert(self.fragment_index, fragment.format(*formatter).replace('\n', self.line_nr_string + '\n'))
        self.fragment_index += 1

    def indent(self) -> None:
        self.indent_level += 1

    def dedent(self) -> None:
        self.indent_level -= 1

    def inscope(self, node: object) -> None:
        self.scopes.append(utils.Any(node=node, nonlocals=set(), contains_yield=False))

    def descope(self) -> None:
        self.scopes.pop()

    def get_scope(self, *node_types: object) -> object:
        if node_types:
            for scope in reversed(self.scopes):
                if type(scope.node) in node_types:
                    return scope
        else:
            return self.scopes[-1]

    def get_adjacent_class_scopes(self, in_method: bool = False) -> list:
        reversed_class_scopes = []
        for scope in reversed(self.scopes):
            if in_method:
                if type(scope.node) in (ast.FunctionDef, ast.AsyncFunctionDef):
                    continue
                else:
                    in_method = False
            if type(scope.node) != ast.ClassDef:
                break
            reversed_class_scopes.append(scope)
        return reversed(reversed_class_scopes)

    def emit_comma(self, index: int, blank: bool = True) -> None:
        if self.noskip_code_generation and self.conditional_code_generation and index:
            self.emit(', ' if blank else ',')

    def emit_begin_truthy(self) -> None:
        if self.allow_conversion_to_truth_value:
            self.emit('__t__ (')

    def emit_end_truthy(self) -> None:
        if self.allow_conversion_to_truth_value:
            self.emit(')')

    def adapt_line_nr_string(self, node: object = None, offset: int = 0) -> None:
        if utils.command_args.map or utils.command_args.anno:
            if node:
                if hasattr(node, 'lineno'):
                    line_nr = node.lineno + offset
                else:
                    line_nr = self.line_nr + offset
            else:
                line_nr = 1 + offset
            self.line_nr_string = str(sourcemaps.max_nr_of_source_lines_per_module + line_nr)[1:]
        else:
            self.line_nr_string = ''

    def is_comment_string(self, statement: object) -> bool:
        return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and (type(statement.value.value) == str)

    def emit_body(self, body: object) -> None:
        for statement in body:
            if self.is_comment_string(statement):
                pass
            else:
                self.visit(statement)
                self.emit(';\n')

    def emit_subscript_assign(self, target: object, value: object, emit_path_indices: callable = None) -> None:
        if type(target.slice) == ast.Index:
            if type(target.slice.value) == ast.Tuple:
                self.visit(target.value)
                self.emit('.__setitem__ (')
                self.strip_tuple = True
                self.visit(target.slice.value)
                self.emit(')')
            elif self.allow_operator_overloading:
                self.emit('__setitem__ (')
                self.visit(target.value)
                self.emit(', ')
                self.visit(target.slice.value)
                self.emit(', ')
                self.visit(value)
                emit_path_indices()
                self.emit(')')
            else:
                self.expecting_non_overloaded_lhs_index = True
                self.visit(target)
                self.emit(' = ')
                self.visit(value)
                emit_path_indices()
        elif type(target.slice) == ast.Slice:
            if self.allow_operator_overloading:
                self.emit('__setslice__ (')
                self.visit(target.value)
                self.emit(', ')
            else:
                self.visit(target.value)
                self.emit('.__setslice__ (')
            if target.slice.lower == None:
                self.emit('0')
            else:
                self.visit(target.slice.lower)
            self.emit(', ')
            if target.slice.upper == None:
                self.emit('null')
            else:
                self.visit(target.slice.upper)
            self.emit(', ')
            if target.slice.step:
                self.visit(target.slice.step)
            else:
                self.emit('null')
            self.emit(', ')
            self.visit(value)
            self.emit(')')
        elif type(target.slice) == ast.ExtSlice:
            self.visit(target.value)
            self.emit('.__setitem__ (')
            self.emit('[')
            for index, dim in enumerate(target.slice.dims):
                self.emit_comma(index)
                self.visit(dim)
            self.emit(']')
            self.emit(', ')
            self.visit(value)
            self.emit(')')

    def next_temp(self, name: str) -> str:
        if name in self.temp_indices:
            self.temp_indices[name] += 1
        else:
            self.temp_indices[name] = 0
        return self.get_temp(name)

    def skip_temp(self, name: str) -> None:
        self.skipped_temps.add(self.next_temp(name))

    def skipped_temp(self, name: str) -> bool:
        return self.get_temp(name) in self.skipped_temps

    def get_temp(self, name: str) -> str:
        if name in self.temp_indices:
            return '__{}{}__'.format(name, self.temp_indices[name])
        else:
            return None

    def prev_temp(self, name: str) -> None:
        if self.get_temp(name) in self.skipped_temps:
            self.skipped_temps.remove(self.get_temp(name))
        self.temp_indices[name] -= 1
        if self.temp_indices[name] < 0:
            del self.temp_indices[name]

    def use_module(self, name: str) -> object:
        self.module.program.import_stack[-1][1] = self.line_nr
        return self.module.program.provide(name, filter=self.filter_id)

    def is_call(self, node: object, name: str) -> bool:
        return type(node) == ast.Call and type(node.func) == ast.Name and (node.func.id == name)

    def get_pragma_from_expr(self, node: object) -> object:
        return node.value.args if type(node) == ast.Expr and self.is_call(node.value, '__pragma__') else None

    def get_pragma_from_if(self, node: object) -> object:
        return node.test.args if type(node) == ast.If and self.is_call(node.test, '__pragma__') else None

    def visit(self, node: object) -> None:
        try:
            self.line_nr = node.lineno
        except:
            pass
        pragma_in_if = self.get_pragma_from_if(node)
        pragma_in_expr = self.get_pragma_from_expr(node)
        if pragma_in_if:
            if pragma_in_if[0].s == 'defined':
                for symbol in pragma_in_if[1:]:
                    if symbol.s in self.module.program.symbols:
                        defined_in_if = True
                        break
                else:
                    defined_in_if = False
        elif pragma_in_expr:
            if pragma_in_expr[0].s == 'skip':
                self.noskip_code_generation = False
            elif pragma_in_expr[0].s == 'noskip':
                self.noskip_code_generation = True
            if pragma_in_expr[0].s in ('ifdef', 'ifndef'):
                defined_in_expr = eval(compile(ast.Expression(pragma_in_expr[1]), '<string>', 'eval'), {}, {'__envir__': self.module.program.envir}) in self.module.program.symbols
            if pragma_in_expr[0].s == 'ifdef':
                self.conditional_code_generation = defined_in_expr
            elif pragma_in_expr[0].s == 'ifndef':
                self.conditional_code_generation = not defined_in_expr
            elif pragma_in_expr[0].s == 'else':
                self.conditional_code_generation = not self.conditional_code_generation
            elif pragma_in_expr[0].s == 'endif':
                self.conditional_code_generation = True
        if self.noskip_code_generation and self.conditional_code_generation:
            if pragma_in_if:
                if defined_in_if:
                    self.emit_body(node.body)
            else:
                super().visit(node)

    def visit_arg(self, node: object) -> None:
        self.emit(self.filter_id(node.arg))

    def visit_arguments(self, node: object) -> None:
        self.emit('(')
        for index, arg in enumerate(node.args):
            self.emit_comma(index)
            self.visit(arg)
        self.emit(') {{\n')
        self.indent()
        for arg, expr in reversed(list(zip(reversed(node.args), reversed(node.defaults)))):
            if expr:
                self.emit('if (typeof {0} == \'undefined\' || ({0} != null && {0}.hasOwnProperty ("__kwargtrans__"))) {{;\n', self.filter_id(arg.arg))
                self.indent()
                self.emit('var {} = ', self.filter_id(arg.arg))
                self.visit(expr)
                self.emit(';\n')
                self.dedent()
                self.emit('}};\n')
        for arg, expr in zip(node.kwonlyargs, node.kw_defaults):
            if expr:
                self.emit('var {} = ', self.filter_id(arg.arg))
                self.visit(expr)
                self.emit(';\n')
        if self.allow_keyword_args:
            if node.kwarg:
                self.emit('var {} = dict ();\n', self.filter_id(node.kwarg.arg))
            self.emit('if (arguments.length) {{\n')
            self.indent()
            self.emit('var {} = arguments.length - 1;\n', self.next_temp('ilastarg'))
            self.emit('if (arguments [{0}] && arguments [{0}].hasOwnProperty ("__kwargtrans__")) {{\n', self.get_temp('ilastarg'))
            self.indent()
            self.emit('var {} = arguments [{}--];\n', self.next_temp('allkwargs'), self.get_temp('ilastarg'))
            self.emit('for (var {} in {}) {{\n', self.next_temp('attrib'), self.get_temp('allkwargs'))
            self.indent()
            if node.args + node.kwonlyargs or node.kwarg:
                self.emit('switch ({}) {{\n', self.get_temp('attrib'))
                self.indent()
                for arg in node.args + node.kwonlyargs:
                    self.emit("case '{0}': var {0} = {1} [{2}]; break;\n", self.filter_id(arg.arg), self.get_temp('allkwargs'), self.get_temp('attrib'))
                if node.kwarg:
                    self.emit('default: {0} [{1}] = {2} [{1}];\n', self.filter_id(node.kwarg.arg), self.get_temp('attrib'), self.get_temp('allkwargs'))
                self.dedent()
                self.emit('}}\n')
            self.prev_temp('allkwargs')
            self.prev_temp('attrib')
            self.dedent()
            self.emit('}}\n')
            if node.kwarg:
                self.emit('delete {}.__kwargtrans__;\n', self.filter_id(node.kwarg.arg))
            self.dedent()
            self.emit('}}\n')
            if node.vararg:
                self.emit('var {} = tuple ([].slice.apply (arguments).slice ({}, {} + 1));\n', self.filter_id(node.vararg.arg), len(node.args), self.get_temp('ilastarg'))
            self.prev_temp('ilastarg')
            self.dedent()
            self.emit('}}\n')
            self.emit('else {{\n')
            self.indent()
            if node.vararg:
                self.emit('var {} = tuple ();\n', self.filter_id(node.vararg.arg))
            self.dedent()
            self.emit('}}\n')
        elif node.vararg:
            self.emit('var {} = tuple ([].slice.apply (arguments).slice ({}));\n', self.filter_id(node.vararg.arg), len(node.args))

    def visit_ann_assign(self, node: object) -> None:
        if node.value != None:
            self.visit(ast.Assign([node.target], node.value))

    def visit_assert(self, node: object) -> None:
        if utils.command_args.dassert:
            self.emit('assert (')
            self.visit(node.test)
            if node.msg:
                self.emit(', ')
                self.visit(node.msg)
            self.emit(');\n')

    def visit_assign(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        target_leafs = (ast.Attribute, ast.Subscript, ast.Name)

        def assign_target(target: object, value: object, path_indices: list) -> None:
            def emit_path_indices() -> None:
                if path_indices:
                    self.emit(' ')
                    for path_index in path_indices:
                        self.emit('[')
                        self.visit(path_index)
                        self.emit(']')
                else:
                    pass
            if type(target) == ast.Subscript:
                self.emit_subscript_assign(target, value, emit_path_indices)
            elif is_property_assign and target.id != self.get_temp('left'):
                self.emit("Object.defineProperty ({}, '{}', ".format(self.get_scope().node.name, target.id))
                self.visit(value)
                emit_path_indices()
                self.emit(')')
            else:
                if type(target) == ast.Name:
                    if type(self.get_scope().node) == ast.ClassDef and target.id != self.get_temp('left'):
                        self.emit('{}.'.format('.'.join([scope.node.name for scope in self.get_adjacent_class_scopes()])))
                    elif target.id in self.get_scope().nonlocals:
                        pass
                    else:
                        if type(self.get_scope().node) == ast.Module:
                            if hasattr(node, 'parentNode') and type(node.parentNode) == ast.Module and (not target.id in self.all_own_names):
                                self.emit('export ')
                        self.emit('var ')
                self.visit(target)
                self.emit(' = ')
                self.visit(value)
                emit_path_indices()

        def walk_target(expr: object, path_indices: list) -> None:
            if type(expr) in target_leafs:
                self.emit(';\n')
                assign_target(expr, ast.Name(id=self.get_temp('left'), ctx=ast.Load), path_indices)
            else:
                path_indices.append(None)
                for index, elt in enumerate(expr.elts):
                    path_indices[-1] = index
                    walk_target(elt, path_indices)
                path_indices.pop()

        def get_is_property_assign(value: object) -> bool:
            if self.is_call(value, 'property'):
                return True
            else:
                try:
                    return get_is_property_assign(value.elts[0])
                except:
                    return False
        is_property_assign = type(self.get_scope().node) == ast.ClassDef and get_is_property_assign(node.value)
        if len(node.targets) == 1 and type(node.targets[0]) in target_leafs:
            assign_target(node.targets[0], node.value)
        else:
            self.visit(ast.Assign(targets=[ast.Name(id=self.next_temp('left'), ctx=ast.Store)], value=node.value))
            for expr in node.targets:
                walk_target(expr, [])
            self.prev_temp('left')

    def visit_attribute(self, node: object) -> None:
        if type(node.value) in (ast.BinOp, ast.BoolOp, ast.Compare):
            self.emit('(')
        self.visit(node.value)
        if type(node.value) in (ast.BinOp, ast.BoolOp, ast.Compare):
            self.emit(')')
        self.emit('.{}', self.filter_id(node.attr))

    def visit-await(self, node: object) -> None:
        self.emit('await ')
        self.visit(node.value)

    def visit_aug_assign(self, node: object) -> None:
        if self.allow_operator_overloading:
            rhs_function_name = self.filter_id('__ipow__' if type(node.op) == ast.Pow else '__imatmul__' if type(node.op) == ast.MatMult else ('__ijsmod__' if self.allow_javascript_mod else '__imod__') if type(node.op) == ast.Mod else '__imul__' if type(node.op) == ast.Mult else '__idiv__' if type(node.op) == ast.Div else '__iadd__' if type(node.op) == ast.Add else '__isub__' if type(node.op) == ast.Sub else '__ilshift__' if type(node.op) == ast.LShift else '__irshift__' if type(node.op) == ast.RShift else '__ior__' if type(node.op) == ast.BitOr else '__ixor__' if type(node.op) == ast.BitXor else '__iand__' if type(node.op) == ast.BitAnd else 'Never here')
            rhs_call = ast.Call(func=ast.Name(id=rhs_function_name, ctx=ast.Load), args=[node.target, node.value], keywords=[])
            if type(node.target) == ast.Subscript:
                self.emit_subscript_assign(node.target, rhs_call)
            else:
                if type(node.target) == ast.Name and (not node.target.id in self.get_scope().nonlocals):
                    self.emit('var ')
                self.visit(node.target)
                self.emit(' = ')
                self.visit(rhs_call)
        elif type(node.op) in (ast.FloorDiv, ast.MatMult, ast.Pow) or (type(node.op) == ast.Mod and (not self.allow_javascript_mod)) or (type(node.target) == ast.Subscript and (type(node.target.slice) != ast.Index or type(node.target.slice.value) == ast.Tuple)):
            self.visit(ast.Assign(targets=[node.target], value=ast.BinOp(left=node.target, op=node.op, right=node.value)))
        else:
            self.expecting_non_overloaded_lhs_index = True
            self.visit(node.target)
            if type(node.value) == ast.Constant and node.value.value == 1:
                if type(node.op) == ast.Add:
                    self.emit('++')
                    return
                elif type(node.op) == ast.Sub:
                    self.emit('--')
                    return
            elif type(node.value) == ast.UnaryOp and type(node.value.operand) == ast.Constant and (node.value.operand.value == 1):
                if type(node.op) == ast.Add:
                    if type(node.value.op) == ast.UAdd:
                        self.emit('++')
                        return
                    elif type(node.value.op) == ast.USub:
                        self.emit('--')
                        return
                elif type(node.op) == ast.Sub:
                    if type(node.value.op) == ast.UAdd:
                        self.emit('--')
                        return
                    elif type(node.value.op) == ast.USub:
                        self.emit('++')
                        return
            self.emit(' {}= ', self.operators[type(node.op)][0])
            self.visit(node.value)

    def visit_bin_op(self, node: object) -> None:
        if type(node.op) == ast.FloorDiv:
            if self.allow_operator_overloading:
                self.emit('__floordiv__ (')
                self.visit_sub_expr(node, node.left)
                self.emit(', ')
                self.visit_sub_expr(node, node.right)
                self.emit(')')
            else:
                self.emit('Math.floor (')
                self.visit_sub_expr(node, node.left)
                self.emit(' / ')
                self.visit_sub_expr(node, node.right)
                self.emit(')')
        elif type(node.op) in (ast.Pow, ast.MatMult) or (type(node.op) == ast.Mod and (self.allow_operator_overloading or not self.allow_javascript_mod)) or (type(node.op) in (ast.Mult, ast.Div, ast.Add, ast.Sub, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd) and self.allow_operator_overloading):
            self.emit('{} ('.format(self.filter_id(('__floordiv__' if self.allow_operator_overloading else 'Math.floor') if type(node.op) == ast.FloorDiv else ('__pow__' if self.allow_operator_overloading else 'Math.pow') if type(node.op) == ast.Pow else '__matmul__' if type(node.op) == ast.MatMult else ('__jsmod__' if self.allow_javascript_mod else '__mod__') if type(node.op) == ast.Mod else '__mul__' if type(node.op) == ast.Mult else '__truediv__' if type(node.op) == ast.Div else '__add__' if type(node.op) == ast.Add else '__sub__' if type(node.op) == ast.Sub else '__lshift__' if type(node.op) == ast.LShift else '__rshift__' if type(node.op) == ast.RShift else '__or__' if type(node.op) == ast.BitOr else '__xor__' if type(node.op) == ast.BitXor else '__and__' if type(node.op) == ast.BitAnd else 'Never here')))
            self.visit(node.left)
            self.emit(', ')
            self.visit(node.right)
            self.emit(')')
        else:
            self.visit_sub_expr(node, node.left)
            self.emit(' {0} '.format(self.operators[type(node.op)][0]))
            self.visit_sub_expr(node, node.right)

    def visit_bool_op(self, node: object) -> None:
        for index, value in enumerate(node.values):
            if index:
                self.emit(' {} '.format(self.operators[type(node.op)][0]))
            if index < len(node.values) - 1:
                self.emit_begin_truthy()
            self.visit_sub_expr(node, value)
            if index < len(node.values) - 1:
                self.emit_end_truthy()

    def visit_break(self, node: object) -> None:
        if not self.skipped_temp('break'):
            self.emit('{} = true;\n', self.get_temp('break'))
        self.emit('break')

    def visit_call(self, node: object, data_class_arg_dict: dict = None) -> None:
        self.adapt_line_nr_string(node)

        def emit_kwarg_trans() -> None:
            self.emit('__kwargtrans__ (')
            has_separate_key_args = False
            has_kwargs = False
            for keyword in node.keywords:
                if keyword.arg:
                    has_separate_key_args = True
                else:
                    has_kwargs = True
                    break
            if has_separate_key_args:
                if has_kwargs:
                    self.emit('__mergekwargtrans__ (')
                self.emit('{{')
            for keyword_index, keyword in enumerate(node.keywords):
                if keyword.arg:
                    self.emit_comma(keyword_index)
                    self.emit('{}: ', self.filter_id(keyword.arg))
                    self.visit(keyword.value)
                else:
                    if has_separate_key_args:
                        self.emit('}}, ')
                    self.visit(keyword.value)
            if has_separate_key_args:
                if has_kwargs:
                    self.emit(')')
                else:
                    self.emit('}}')
            self.emit(')')

        def include(file_name: str) -> str:
            try:
                searched_include_paths = []
                for search_dir in self.module.program.module_search_dirs:
                    file_path = '{}/{}'.format(search_dir, file_name)
                    if os.path.isfile(file_path):
                        included_code = tokenize.open(file_path).read()
                        if file_name.endswith('.js'):
                            included_code = utils.digest_javascript(included_code, self.module.program.symbols, not utils.command_args.dnostrip or utils.command_args.anno, self.allow_debug_map).digested_code
                        return included_code
                    else:
                        searched_include_paths.append(file_path)
                else:
                    raise utils.Error(line_nr=self.line_nr, message="\n\tAttempt to include file: {}\n\tCan't find any of:\n\t\t{}".format(node.args[0], '\n\t\t'.join(searched_include_paths)))
            except:
                print(traceback.format_exc())

        if type(node.func) == ast.Name:
            if node.func.id == 'type':
                self.emit('py_typeof (')
                self.visit(node.args[0])
                self.emit(')')
                return
            elif node.func.id == 'property':
                self.emit('{0}.call ({1}, {1}.{2}'.format(node.func.id, self.get_scope(ast.ClassDef).node.name, self.filter_id(node.args[0].id)))
                if len(node.args) > 1:
                    self.emit(', {}.{}'.format(self.get_scope(ast.ClassDef).node.name, node.args[1].id))
                self.emit(')')
                return
            elif node.func.id == '__pragma__':
                if node.args[0].s == 'alias':
                    self.aliases.insert(0, (node.args[1].s, node.args[2].s))
                elif node.args[0].s == 'noalias':
                    if len(node.args) == 1:
                        self.aliases = []
                    else:
                        for index in reversed(range(len(self.aliases))):
                            if self.aliases[index][0] == node.args[1].s:
                                self.aliases.pop(index)
                elif node.args[0].s == 'noanno':
                    self.allow_debug_map = False
                elif node.args[0].s == 'fcall':
                    self.allow_memoize_calls = True
                elif node.args[0].s == 'nofcall':
                    self.allow_memoize_calls = False
                elif node.args[0].s == 'docat':
                    self.allow_doc_attribs = True
                elif node.args[0].s == 'nodocat':
                    self.allow_doc_attribs = False
                elif node.args[0].s == 'iconv':
                    self.allow_conversion_to_iterable = True
                elif node.args[0].s == 'noiconv':
                    self.allow_conversion_to_iterable = False
                elif node.args[0].s == 'jsiter':
                    self.allow_javascript_iter = True
                elif node.args[0].s == 'nojsiter':
                    self.allow_javascript_iter = False
                elif node.args[0].s == 'jscall':
                    self.allow_javascript_call = True
                elif node.args[0].s == 'nojscall':
                    self.allow_javascript_call = False
                elif node.args[0].s == 'jskeys':
                    self.allow_javascript_keys = True
                elif node.args[0].s == 'nojskeys':
                    self.allow_javascript_keys = False
                elif node.args[0].s == 'keycheck':
                    self.allow_key_check = True
                elif node.args[0].s == 'nokeycheck':
                    self.allow_key_check = False
                elif node.args[0].s == 'jsmod':
                    self.allow_javascript_mod = True
                elif node.args[0].s == 'nojsmod':
                    self.allow_javascript_mod = False
                elif node.args[0].s == 'gsend':
                    self.replace_send = True
                elif node.args[0].s == 'nogsend':
                    self.replace_send = False
                elif node.args[0].s == 'tconv':
                    self.allow_conversion_to_truth_value = True
                elif node.args[0].s == 'notconv':
                    self.allow_conversion_to_truth_value = False
                elif node.args[0].s == 'run':
                    pass
                elif node.args[0].s == 'norun':
                    pass
                elif node.args[0].s == 'js':
                    try:
                        try:
                            code = node.args[1].s.format(*[eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include}) for arg in node.args[2:]])
                        except:
                            code = node.args[2].s
                        for line in code.split('\n'):
                            self.emit('{}\n', line)
                    except:
                        print(traceback.format_exc())
                elif node.args[0].s == 'xtrans':
                    try:
                        source_code = node.args[2].s.format(*[eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include}) for arg in node.args[3:]])
                        work_dir = '.'
                        for keyword in node.keywords:
                            if keyword.arg == 'cwd':
                                work_dir = keyword.value.s
                        process = subprocess.Popen(shlex.split(node.args[1].s), stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=work_dir)
                        process.stdin.write(source_code.encode('utf8'))
                        process.stdin.close()
                        while process.returncode is None:
                            process.poll()
                        target_code = process.stdout.read().decode('utf8').replace('\r\n', '\n')
                        for line in target_code.split('\n'):
                            self.emit('{}\n', line)
                    except:
                        print(traceback.format_exc())
                elif node.args[0].s == 'xglobs':
                    self.allow_globals = True
                elif node.args[0].s == 'noxglobs':
                    self.allow_globals = False
                elif node.args[0].s == 'kwargs':
                    self.allow_keyword_args = True
                elif node.args[0].s == 'nokwargs':
                    self.allow_keyword_args = False
                elif node.args[0].s == 'opov':
                    self.allow_operator_overloading = True
                elif node.args[0].s == 'noopov':
                    self.allow_operator_overloading = False
                elif node.args[0].s == 'redirect':
                    if node.args[1].s == 'stdout':
                        self.emit("__stdout__ = '{}'", node.args[2])
                elif node.args[0].s == 'noredirect':
                    if node.args[1].s == 'stdout':
                        self.emit("__stdout__ = '__console__'")
                elif node.args[0].s in ('skip', 'noskip', 'defined', 'ifdef', 'ifndef', 'else', 'endif'):
                    pass
                elif node.args[0].s == 'xpath':
                    self.module.program.module_search_dirs[1:1] = [elt.s for elt in node.args[1].elts]
                else:
                    raise utils.Error(line_nr=self.line_nr, message='\n\tUnknown pragma: {}'.format(node.args[0].value if type(node.args[0]) == ast.Constant else node.args[0]))
                return
            elif node.func.id == '__new__':
                self.emit('new ')
                self.visit(node.args[0])
                return
            elif node.func.id == '__typeof__':
                self.emit('typeof ')
                self.visit(node.args[0])
                return
            elif node.func.id == '__preinc__':
                self.emit('++')
                self.visit(node.args[0])
                return
            elif node.func.id == '__postinc__':
                self.visit(node.args[0])
                self.emit('++')
                return
            elif node.func.id == '__predec__':
                self.emit('--')
                self.visit(node.args[0])
                return
            elif node.func.id == '__postdec__':
                self.visit(node.args[0])
                self.emit('--')
                return
        elif type(node.func) == ast.Attribute and node.func.attr == 'conjugate':
            try:
                self.visit(ast.Call(func=ast.Name(id='__conj__', ctx=ast.Load), args=[node.func.value], keywords=[]))
                return
            except:
                print(traceback.format_exc())
        elif type(node.func) == ast.Attribute and self.replace_send and (node.func.attr == 'send'):
            self.emit('(function () {{return ')
            self.visit(ast.Attribute(value=ast.Call(func=ast.Attribute(value=ast.Name(id=node.func.value.id, ctx=ast.Load), attr='js_next', ctx=ast.Load), args=node.args, keywords=node.keywords), attr='value', ctx=ast.Load))
            self.emit('}}) ()')
            return
        elif type(node.func) == ast.Attribute and type(node.func.value) == ast.Call and (type(node.func.value.func) == ast.Name) and (node.func.value.func.id == 'super'):
            if node.func.value.args or node.func.value.keywords:
                raise utils.Error(line_nr=self.line_nr, message="\n\tBuilt in function 'super' with arguments not supported")
            else:
                self.visit(ast.Call(func=ast.Call(func=ast.Name(id='__super__', ctx=ast.Load), args=[ast.Name(id='.'.join([scope.node.name for scope in self.get_adjacent_class_scopes(True)]), ctx=ast.Load), ast.Constant(value=node.func.attr)], keywords=[]), args=[ast.Name(id='self', ctx=ast.Load)] + node.args, keywords=node.keywords))
                return
        if self.allow_operator_overloading and (not (type(node.func) == ast.Name and node.func.id == '__call__')):
            if type(node.func) == ast.Attribute:
                self.emit('(function () {{\n')
                self.inscope(ast.FunctionDef())
                self.indent()
                self.emit('var {} = ', self.next_temp('accu'))
                self.visit(node.func.value)
                self.emit(';\n')
                self.emit('return ')
                self.visit(ast.Call(func=ast.Name(id='__call__', ctx=ast.Load), args=[ast.Attribute(value=ast.Name(id=self.get_temp('accu'), ctx=ast.Load), attr=node.func.attr, ctx=ast.Load), ast.Name(id=self.get_temp('accu'), ctx=ast.Load)] + node.args, keywords=node.keywords))
                self.emit(';\n')
                self.prev_temp('accu')
                self.dedent()
                self.descope()
                self.emit('}}) ()')
            else:
                self.visit(ast.Call(func=ast.Name(id='__call__', ctx=ast.Load), args=[node.func, ast.Constant(value=None)] + node.args, keywords=node.keywords))
            return
        if data_class_arg_dict != None:
            data_class_arg_tuple = copy.deepcopy(dataClassDefaultArgTuple)
            for index, expr in enumerate(node.args):
                value = None
                if expr == ast.Constant:
                    value = True if expr.value == 'True' else False if expr.value == 'False' else None
                if value != None:
                    data_class_arg_tuple[index][1] = value
                else:
                    raise utils.Error(message='Arguments to @dataclass can only be constants True or False')
            data_class_arg_dict.update(dict(data_class_arg_tuple))
            for keyword in node.keywords:
                data_class_arg_dict[keyword.arg] = keyword.value
            return
        self.visit(node.func)
        self.emit(' (')
        for index, expr in enumerate(node.args):
            self.emit_comma(index)
            if type(expr) == ast.Starred:
                self.emit('...')
            self.visit(expr)
        if node.keywords:
            self.emit_comma(len(node.args))
            emit_kwarg_trans()
        self.emit(')')

    def visit_class_def(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        if type(self.get_scope().node) == ast.Module:
            self.emit('export var {} = '.format(self.filter_id(node.name)))
            self.all_own_names.add(node.name)
        elif type(self.get_scope().node) == ast.ClassDef:
            self.emit('\n{}:'.format(self.filter_id(node.name)))
        else:
            self.emit('var {} ='.format(self.filter_id(node.name)))
        is_data_class = False
        if node.decorator_list:
            if type(node.decorator_list[-1]) == ast.Name and node.decorator_list[-1].id == 'dataclass':
                is_data_class = True
                data_class_arg_dict = dict(dataClassDefaultArgTuple)
                node.decorator_list.pop()
            elif type(node.decorator_list[-1]) == ast.Call and node.decorator_list[-1].func.id == 'dataclass':
                is_data_class = True
                data_class_arg_dict = {}
                self.visit_Call(node.decorator_list.pop(), data_class_arg_dict)
        decorators_used = 0
        if node.decorator_list:
            self.emit(' ')
            if self.allow_operator_overloading:
                self.emit('__call__ (')
            for decorator in node.decorator_list:
                if decorators_used > 0:
                    self.emit(' (')
                self.visit(decorator)
                decorators_used += 1
            if self.allow_operator_overloading:
                self.emit(', null, ')
            else:
                self.emit(' (')
        self.emit(" __class__ ('{}', [", self.filter_id(node.name))
        if node.bases:
            for index, expr in enumerate(node.bases):
                try:
                    self.emit_comma(index)
                    self.visit(expr)
                except Exception as exception:
                    utils.enhance_exception(exception, line_nr=self.line_nr, message='\n\tInvalid base class')
        else:
            self.emit('object')
        self.emit('], {{')
        self.inscope(node)
        self.indent()
        self.emit('\n__module__: __name__,')
        inline_assigns = []
        property_assigns = []
        init_assigns = []
        delayed_assigns = []
        repr_assigns = []
        compare_assigns = []
        index = 0
        if is_data_class:
            init_hoist_fragment_index = self.fragment_index
            init_hoist_indent_level = self.indent_level
        for statement in node.body:
            if self.is_comment_string(statement):
                pass
            elif type(statement) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef):
                self.emit_comma(index, False)
                self.visit(statement)
                index += 1
            elif type(statement) == ast.Assign:
                if len(statement.targets) == 1 and type(statement.targets[0]) == ast.Name:
                    if type(statement.value) == ast.Call and type(statement.value.func) == ast.Name and (statement.value.func.id == 'property'):
                        property_assigns.append(statement)
                    else:
                        inline_assigns.append(statement)
                        self.emit_comma(index, False)
                        self.emit('\n{}: '.format(self.filter_id(statement.targets[0].id)))
                        self.visit(statement.value)
                        self.adapt_line_nr_string(statement)
                        index += 1
                else:
                    delayed_assigns.append(statement)
            elif type(statement) == ast.AnnAssign:
                if type(statement.value) == ast.Call and type(statement.value.func) == ast.Name and (statement.value.func.id == 'property'):
                    property_assigns.append(statement)
                    if is_data_class:
                        repr_assigns.append(statement)
                        compare_assigns.append(statement)
                elif is_data_class and type(statement.annotation) == ast.Name and (statement.annotation.id != 'ClassVar'):
                    inline_assigns.append(statement)
                    init_assigns.append(statement)
                    repr_assigns.append(statement)
                    compare_assigns.append(statement)
                    self.emit_comma(index, False)
                    self.emit('\n{}: '.format(self.filter_id(statement.target.id)))
                    self.visit(statement.value)
                    self.adapt_line_nr_string(statement)
                    index += 1
                elif type(statement.target) == ast.Name:
                    try:
                        inline_assigns.append(statement)
                        self.emit_comma(index, False)
                        self.emit('\n{}: '.format(self.filter_id(statement.target.id)))
                        self.visit(statement.value)
                        self.adapt_line_nr_string(statement)
                        index += 1
                    except:
                        print(traceback.format_exc())
                else:
                    delayed_assigns.append(statement)
            elif self.get_pragma_from_expr(statement):
                self.visit(statement)
        self.dedent()
        self.emit('\n}}')
        if node.keywords:
            if node.keywords[0].arg == 'metaclass':
                self.emit(', ')
                self.visit(node.keywords[0].value)
            else:
                raise utils.Error(line_nr=self.line_nr, message='\n\tUnknown keyword argument {} definition of class {}'.format(node.keywords[0].arg, node.name))
        self.emit(')')
        if decorators_used:
            self.emit(')' * decorators_used)
        if self.allow_doc_attribs:
            doc_string = ast.get_docstring(node)
            if doc_string:
                self.emit(" .__setdoc__ ('{}')", doc_string.replace('\n', '\\n '))
        if is_data_class:
            nr_of_fragments_to_jump = self.fragment_index - init_hoist_fragment_index
            self.fragment_index = init_hoist_fragment_index
            original_indent_level = self.indent_level
            self.indent_level = init_hoist_indent_level
            init_args = [((init_assign.targets[0] if type(init_assign) == ast.Assign else init_assign.target).id, init_assign.value) for init_assign in init_assigns]
            repr_names = [(repr_assign.targets[0] if type(repr_assign) == ast.Assign else repr_assign.target).id for repr_assign in repr_assigns]
            compare_names = [(compare_assign.targets[0] if type(compare_assign) == ast.Assign else compare_assign.target).id for compare_assign in compare_assigns]
            if data_class_arg_dict['repr']:
                original_allow_keyword_args = self.allow_keyword_args
                self.allow_keyword_args = True
                self.visit(ast.FunctionDef(name='__init__', args=ast.arguments(args=[ast.arg(arg='self', annotation=None)], vararg=ast.arg(arg='args', annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=ast.arg(arg='kwargs', annotation=None), defaults=[]), body=[ast.Expr(value=ast.Call(func=ast.Name(id='__pragma__', ctx=ast.Load), args=[ast.Constant(value='js'), ast.Constant(value='\nlet names = self.__initfields__.values ();\nfor (let arg of args) {\n    self [names.next () .value] = arg;\n}\nfor (let name of kwargs.py_keys ()) {\n    self [name] = kwargs [name];\n}\n                                        '.strip())], keywords=[]))], decorator_list=[], returns=None, docstring=None))
                self.emit(',')
                self.allow_keyword_args = original_allow_keyword_args
            if data_class_arg_dict['repr']:
                self.visit(ast.FunctionDef(name='__repr__', args=ast.arguments(args=[ast.arg(arg='self', annotation=None)], vararg=None, kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Expr(value=ast.Call(func=ast.Name(id='__pragma__', ctx=ast.Load), args=[ast.Constant(value='js'), ast.Constant(value="\nlet names = self.__reprfields__.values ();\nlet fields = [];\nfor (let name of names) {{\n    fields.push (name + '=' + repr (self [name]));\n}}\nreturn  self.__name__ + '(' + ', '.join (fields) + ')'\n                                        ".strip())], keywords=[]))], decorator_list=[], returns=None, docstring=None))
                self.emit(',')
            comparator_names = []
            if 'eq' in data_class_arg_dict:
                comparator_names += ['__eq__', '__ne__']
            if 'order' in data_class_arg_dict:
                comparator_names += ['__lt__', '__le__', '__gt__', '__ge__']
            for comparator_name in comparator_names:
                self.visit(ast.FunctionDef(name=comparator_name, args=ast.arguments(args=[ast.arg(arg='self', annotation=None), ast.arg(arg='other', annotation=None)], vararg=None, kw_defaults=[], kwarg=None, defaults=[]), body=[ast.Expr(value=ast.Call(func=ast.Name(id='__pragma__', ctx=ast.Load), args=[ast.Constant(value='js'), ast.Constant(value='{}'), ast.Constant(value=('\nlet names = self.__comparefields__.values ();\nlet selfFields = [];\nlet otherFields = [];\nfor (let name of names) {\n    selfFields.push (self [name]);\n    otherFields.push (other [name]);\n}\nreturn list (selfFields).' + comparator_name + '(list (otherFields));\n                                        ').strip())], keywords=[]))], decorator_list=[]))
                returns = (None,)
                self.emit(',')
            self.fragment_index += nr_of_fragments_to_jump
            self.indent_level = original_indent_level
        for assign in delayed_assigns + property_assigns:
            self.emit(';\n')
            self.visit(assign)
        self.merge_list.append(utils.Any(className='.'.join([scope.node.name for scope in self.get_adjacent_class_scopes()]), is_data_class=is_data_class, repr_assigns=repr_assigns, compare_assigns=compare_assigns, init_assigns=init_assigns))
        self.descope()

        def emit_merges() -> None:
            def emit_merge(merge: object) -> None:
                if merge.is_data_class:
                    self.emit('\nfor (let aClass of {}.__bases__) {{\n'.format(self.filter_id(merge.className)))
                    self.indent()
                    self.emit('__mergefields__ ({}, aClass);\n'.format(self.filter_id(merge.className)))
                    self.dedent()
                    self.emit('}}')
                    self.emit(';\n__mergefields__ ({}, {{'.format(self.filter_id(merge.className)))
                    self.emit('__reprfields__: new Set ([{}]), '.format(', '.join(("'{}'".format(repr_assign.target.id) for repr_assign in merge.repr_assigns))))
                    self.emit('__comparefields__: new Set ([{}]), '.format(', '.join(("'{}'".format(compare_assign.target.id) for compare_assign in merge.compare_assigns))))
                    self.emit('__initfields__: new Set ([{}])'.format(', '.join(("'{}'".format(init_assign.target.id) for init_assign in merge.init_assigns))))
                    self.emit('}})')
            for merge in self.merge_list:
                emit_merge(merge)
            self.merge_list = []

        def emit_properties() -> None:
            def emit_property(class_name: str, property_name: str, getter_name: str, setter_name: str = None) -> None:
                self.emit("\nObject.defineProperty ({}, '{}', ".format(class_name, property_name))
                if setter_name:
                    self.emit('property.call ({0}, {0}.{1}, {0}.{2})'.format(class_name, getter_name, setter_name))
                else:
                    self.emit('property.call ({0}, {0}.{1})'.format(class_name, getter_name))
                self.emit(');')
            if self.property_accessor_list:
                self.emit(';')
            while self.property_accessor_list:
                property_accessor = self.property_accessor_list.pop()
                class_name = property_accessor.className
                function_name = property_accessor.functionName
                property_name = function_name[5:]
                is_getter = function_name[:5] == '_get_'
                for property_accessor2 in self.property_accessor_list:
                    class_name2 = property_accessor2.className
                    function_name2 = property_accessor2.functionName
                    property_name2 = function_name2[5:]
                    is_getter2 = function_name2[:5] == '_get_'
                    if class_name == class_name2 and property_name == property_name2 and (is_getter != is_getter2):
                        self.property_accessor_list.remove(property_accessor2)
                        if is_getter:
                            emit_property(class_name, property_name, function_name, function_name2)
                        else:
                            emit_property(class_name, property_name, function_name2, function_name)
                        break
                else:
                    if is_getter:
                        emit_property(class_name, property_name, function_name)
                    else:
                        raise utils.Error(message='\n\tProperty setter declared without getter\n')
        if type(self.get_scope().node) != ast.ClassDef:
            emit_properties()
            emit_merges()

    def visit_compare(self, node: object) -> None:
        if len(node.comparators) > 1:
            self.emit('(')
        left = node.left
        for index, (op, right) in enumerate(zip(node.ops, node.comparators)):
            if index:
                self.emit(' && ')
            if type(op) in (ast.In, ast.NotIn) or (self.allow_operator_overloading and type(op) in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                self.emit('{} ('.format(self.filter_id('__in__' if type(op) == ast.In else '!__in__' if type(op) == ast.NotIn else '__eq__' if type(op) == ast.Eq else '__ne__' if type(op) == ast.NotEq else '__lt__' if type(op) == ast.Lt else '__le__' if type(op) == ast.LtE else '__gt__' if type(op) == ast.Gt else '__ge__' if type(op) == ast.GtE else 'Never here')))
                self.visit_sub_expr(node, left)
                self.emit(', ')
                self.visit_sub_expr(node, right)
                self.emit(')')
            else:
                self.visit_sub_expr(node, left)
                self.emit(' {0} '.format(self.operators[type(op)][0]))
                self.visit_sub_expr(node, right)
            left = right
        if len(node.comparators) > 1:
            self.emit(')')

    def visit_constant(self, node: object) -> None:
        if type(node.value) == str:
            self.emit('{}', repr(node.s))
        elif type(node.value) == bytes:
            self.emit("bytes ('{}')", node.s.decode('ASCII'))
        elif type(node.value) == complex:
            self.emit('complex (0, {})'.format(node.n.imag))
        elif type(node.value) in {float, int}:
            self.emit('{}'.format(node.n))
        else:
            self.emit(self.name_consts[node.value])

    def visit_continue(self, node: object) -> None:
        self.emit('continue')

    def visit_delete(self, node: object) -> None:
        for expr in node.targets:
            if type(expr) != ast.Name:
                self.emit('delete ')
                self.visit(expr)
                self.emit(';\n')

    def visit_dict(self, node: object) -> None:
        if not self.allow_javascript_keys:
            for key in node.keys:
                if not type(key) == ast.Constant:
                    self.emit('dict ([')
                    for index, (key, value) in enumerate(zip(node.keys, node.values)):
                        self.emit_comma(index)
                        self.emit('[')
                        self.visit(key)
                        self.emit(', ')
                        self.visit(value)
                        self.emit(']')
                    self.emit('])')
                    return
        if self.allow_javascript_iter:
            self.emit('{{')
        else:
            self.emit('dict ({{')
        for index, (key, value) in enumerate(zip(node.keys, node.values)):
            self.emit_comma(index)
            self.id_filtering = False
            self.visit(key)
            self.id_filtering = True
            self.emit(': ')
            self.visit(value)
        if self.allow_javascript_iter:
            self.emit('}}')
        else:
            self.emit('}})')

    def visit_dict_comp(self, node: object) -> None:
        self.visit_ListComp(node, is_dict=True)

    def visit_expr(self, node: object) -> None:
        self.visit(node.value)

    def visit_for(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        if node.orelse and (not self.allow_javascript_iter):
            self.emit('var {} = false;\n'.format(self.next_temp('break')))
        else:
            self.skip_temp('break')
        optimize = type(node.target) == ast.Name and self.is_call(node.iter, 'range') and (type(node.iter.args[0]) != ast.Starred) and (len(node.iter.args) < 3 or (type(node.iter.args[2]) == ast.Constant and type(node.iter.args[2].value) == int) or (type(node.iter.args[2]) == ast.UnaryOp and type(node.iter.args[2].operand) == ast.Constant and (type(node.iter.args[2].operand.value) == int)))
        if self.allow_javascript_iter:
            self.emit('for (var ')
            self.visit(node.target)
            self.emit(' in ')
            self.visit(node.iter)
            self.emit(') {{\n')
            self.indent()
        elif optimize:
            step = 1 if len(node.iter.args) <= 2 else node.iter.args[2].value if type(node.iter.args[2]) == ast.Constant else node.iter.args[2].operand.value if type(node.iter.args[2].op) == ast.UAdd else -node.iter.args[2].operand.value
            self.emit('for (var ')
            self.visit(node.target)
            self.emit(' = ')
            self.visit(node.iter.args[0] if len(node.iter.args) > 1 else ast.Constant(value=0))
            self.emit('; ')
            self.visit(node.target)
            self.emit(' < ' if step > 0 else ' > ')
            self.visit(node.iter.args[1] if len(node.iter.args) > 1 else node.iter.args[0])
            self.emit('; ')
            self.visit(node.target)
            if step == 1:
                self.emit('++')
            elif step == -1:
                self.emit('--')
            elif step >= 0:
                self.emit(' += {}', step)
            else:
                self.emit(' -= {}', -step)
            self.emit(') {{\n')
            self.indent()
        elif not self.allow_operator_overloading:
            self.emit('for (var ')
            self.strip_tuples = True
            self.visit(node.target)
            self.strip_tuples = False
            self.emit(' of ')
            if self.allow_conversion_to_iterable:
                self.emit('__i__ (')
            self.visit(node.iter)
            if self.allow_conversion_to_iterable:
                self.emit(')')
            self.emit(') {{\n')
            self.indent()
        else:
            self.emit('var {} = '.format(self.next_temp('iterable')))
            self.visit(node.iter)
            self.emit(';\n')
            if self.allow_conversion_to_iterable:
                self.emit('{0} = __i__ ({0});\n'.format(self.get_temp('iterable')))
            self.emit('for (var {0} = 0; {0} < len ({1}); {0}++) {{\n'.format(self.next_temp('index'), self.get_temp('iterable')))
            self.indent()
            self.visit(ast.Assign(targets=[node.target], value=ast.Subscript(value=ast.Name(id=self.get_temp('iterable'), ctx=ast.Load), slice=ast.Index(value=ast.Name(id=self.get_temp('index'), ctx=ast.Load)), ctx=ast.Load)))
            self.emit(';\n')
        self.emit_body(node.body)
        self.dedent()
        self.emit('}}\n')
        if not (self.allow_javascript_iter or optimize):
            if self.allow_operator_overloading:
                self.prev_temp('index')
                self.prev_temp('iterable')
        if node.orelse:
            self.adapt_line_nr_string(node.orelse, 1)
            self.emit('if (!{}) {{\n'.format(self.get_temp('break')))
            self.prev_temp('break')
            self.indent()
            self.emit_body(node.orelse)
            self.dedent()
            self.emit('}}\n')
        else:
            self.prev_temp('break')

    def visit_formatted_value(self, node: object) -> None:
        self.visit(node.value)

    def visit_async_function_def(self, node: object) -> None:
        self.visit_FunctionDef(node, an_async=True)

    def visit_function_def(self, node: object, an_async: bool = False) -> None:
        def emit_scoped_body() -> None:
            self.inscope(node)
            self.emit_body(node.body)
            self.dedent()
            if self.get_scope(ast.AsyncFunctionDef if an_async else ast.FunctionDef).contains_yield:
                self.target_fragments.insert(yield_star_index, '*')
            self.descope()

        def push_property_accessor(function_name: str) -> None:
            self.property_accessor_list.append(utils.Any(function_name=function_name, className='.'.join([scope.node.name for scope in self.get_adjacent_class_scopes()])))

        node_name = node.name
        if not node_name == '__pragma__':
            is_global = type(self.get_scope().node) == ast.Module
            is_method = not (is_global or type(self.get_scope().node) in (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_method:
                self.emit('\n')
            self.adapt_line_nr_string(node)
            decorate = False
            is_class_method = False
            is_static_method = False
            is_property = False
            getter = '__get__'
            if node.decorator_list:
                for decorator in node.decorator_list:
                    decorator_node = decorator
                    decorator_type = type(decorator_node)
                    name_check = ''
                    while decorator_type != ast.Name:
                        if decorator_type == ast.Call:
                            decorator_node = decorator_node.func
                        elif decorator_type == ast.Attribute:
                            name_check = '.' + decorator_node.attr + name_check
                            decorator_node = decorator_node.value
                        decorator_type = type(decorator_node)
                    name_check = decorator_node.id + name_check
                    if name_check == 'classmethod':
                        is_class_method = True
                        getter = '__getcm__'
                    elif name_check == 'staticmethod':
                        is_static_method = True
                        getter = '__getsm__'
                    elif name_check == 'property':
                        is_property = True
                        node_name = '_get_' + node.name
                        push_property_accessor(node_name)
                    elif re.match('[a-zA-Z0-9_]+\\.setter', name_check):
                        is_property = True
                        node_name = '_set_' + re.match('([a-zA-Z0-9_]+)\\.setter', name_check).group(1)
                        push_property_accessor(node_name)
                    else:
                        decorate = True
            if sum([is_class_method, is_static_method, is_property]) > 1:
                raise utils.Error(line_nr=self.line_nr, message="\n\tstaticmethod, classmethod and property decorators can't be mixed\n")
            js_call = self.allow_javascript_call and node_name != '__init__'
            decorators_used = 0
            if decorate:
                if is_method:
                    if js_call:
                        raise utils.Error(line_nr=self.line_nr, message='\n\tdecorators are not supported with jscall\n')
                        self.emit('{}: '.format(self.filter_id(node_name)))
                    else:
                        self.emit('get {} () {{return {} (this, '.format(self.filter_id(node_name), getter)
                elif is_global:
                    if type(node.parentNode) == ast.Module and (not node_name in self.all_own_names):
                        self.emit('export ')
                    self.emit('var {} = '.format(self.filter_id(node_name)))
                else:
                    self.emit('var {} = '.format(self.filter_id(node_name)))
                if self.allow_operator_overloading:
                    self.emit('__call__ (')
                for decorator in node.decorator_list:
                    if not (type(decorator) == ast.Name and decorator.id in ('classmethod', 'staticmethod')):
                        if decorators_used > 0:
                            self.emit(' (')
                        self.visit(decorator)
                        decorators_used += 1
                if self.allow_operator_overloading:
                    self.emit(', null, ')
                else:
                    self.emit(' (')
                self.emit('{}function'.format('async ' if an_async else ''), )
            elif is_method:
                if js_call:
                    self.emit('{}: function'.format(self.filter_id(node_name), 'async ' if an_async else ''), )
                elif is_static_method:
                    self.emit('get {} () {{return {}function'.format(self.filter_id(node_name), 'async ' if an_async else ''), )
                else:
                    self.emit('get {} () {{return {} (this, {}function'.format(self.filter_id(node_name), getter, 'async ' if an_async else ''), )
            elif is_global:
                if type(node.parentNode) == ast.Module and (not node_name in self.all_own_names):
                    self.emit('export ')
                self.emit('var {} = {}function'.format(self.filter_id(node_name), 'async ' if an_async else ''), )
            else:
                self.emit('var {} = {}function'.format(self.filter_id(node_name), 'async ' if an_async else ''), )
            yield_star_index = self.fragment_index
            self.emit(' ')
            skip_first_arg = js_call and (not (not is_method or is_static_method or is_property))
            if skip_first_arg:
                first_arg = node.args.args[0].arg
                node.args.args = node.args.args[1:]
            self.visit(node.args)
            if skip_first_arg:
                if is_class_method:
                    self.emit("var {} = '__class__' in this ? this.__class__ : this;\n".format(first_arg))
                else:
                    self.emit('var {} = this;\n'.format(first_arg))
            emit_scoped_body()
            self.emit('}}')
            if self.allow_doc_attribs:
                doc_string = ast.get_docstring(node)
                if doc_string:
                    self.emit(" .__setdoc__ ('{}')".format(doc_string.replace('\n', '\\n ')))
            if decorate:
                self.emit(')' * decorators_used)
            if is_method:
                if not js_call:
                    if is_static_method:
                        self.emit(';}}')
                    else:
                        if self.allow_memoize_calls:
                            self.emit(", '{}'", node_name)
                        self.emit(');}}')
                if node_name == '__iter__':
                    self.emit(',\n[Symbol.iterator] () {{return this.__iter__ ()}}')
                if node_name == '__next__':
                    self.emit(',\nnext: __jsUsePyNext__')
            if is_global:
                self.all_own_names.add(node_name)

    def visit_generator_exp(self, node: object) -> None:
        self.visit_ListComp(node, is_gen_exp=True)

    def visit_global(self, node: object) -> None:
        self.get_scope(ast.FunctionDef, ast.AsyncFunctionDef).nonlocals.update(node.names)

    def visit_if(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        self.emit('if (')
        self.emit_begin_truthy()
        global in_if
        in_if = True
        self.visit(node.test)
        in_if = False
        self.emit_end_truthy()
        self.emit(') {{\n')
        self.indent()
        self.emit_body(node.body)
        self.dedent()
        self.emit('}}\n')
        if node.orelse:
            if len(node.orelse) == 1 and node.orelse[0].__class__.__name__ == 'If':
                self.emit('else ')
                self.visit(node.orelse[0])
            else:
                self.adapt_line_nr_string(node.orelse, 1)
                self.emit('else {{\n')
                self.indent()
                self.emit_body(node.orelse)
                self.dedent()
                self.emit('}}\n')

    def visit_if_exp(self, node: object) -> None:
        self.emit('(')
        self.emit_begin_truthy()
        self.visit(node.test)
        self.emit_end_truthy()
        self.emit(' ? ')
        self.visit(node.body)
        self.emit(' : ')
        self.visit(node.orelse)
        self.emit(')')

    def visit_import(self, node: object) -> None:
        self.import_hoist_memos.append(utils.Any(node=node, line_nr=self.line_nr))

    def revisit_import(self, import_hoist_memo: object) -> None:
        self.line_nr = import_hoist_memo.line_nr
        node = import_hoist_memo.node
        self.adapt_line_nr_string(node)
        names = [alias for alias in node.names if not alias.name.startswith(self.stubs_name)]
        if not names:
            return
        for index, alias in enumerate(names):
            try:
                module = self.use_module(alias.name)
            except Exception as exception:
                utils.enhance_exception(exception, line_nr=self.line_nr, message="\n\tCan't import module '{}'".format(alias.name))
            if alias.asname and (not alias.asname in self.all_own_names | self.all_imported_names):
                self.all_imported_names.add(alias.asname)
                self.emit("import * as {} from '{}';\n".format(self.filter_id(alias.asname), module.import_rel_path))
            else:
                self.emit("import * as __module_{}__ from '{}';\n".format(self.filter_id(module.name).replace('.', '_'), module.import_rel_path))
                alias_split = alias.name.split('.', 1)
                head = alias_split[0]
                tail = alias_split[1] if len(alias_split) > 1 else ''
                self.import_heads.add(head)
                self.emit("__nest__ ({}, '{}', __module_{}__);\n".format(self.filter_id(head), self.filter_id(tail), self.filter_id(module.name.replace('.', '_'))))
            if index < len(names) - 1:
                self.emit(';\n')

    def visit_import_from(self, node: object) -> None:
        self.import_hoist_memos.append(utils.Any(node=node, line_nr=self.line_nr))

    def revisit_import_from(self, import_hoist_memo: object) -> None:
        self.line_nr = import_hoist_memo.line_nr
        node = import_hoist_memo.node
        self.adapt_line_nr_string(node)
        if node.module.startswith(self.stubs_name):
            return
        try:
            self.module.program.searched_module_paths = []
            name_pairs = []
            facility_imported = False
            for index, alias in enumerate(node.names):
                if alias.name == '*':
                    if len(node.names) > 1:
                        raise utils.Error(line_nr=self.line_nr, message="\n\tCan't import module '{}'".format(alias.name))
                    module = self.use_module(node.module)
                    for a_name in module.exported_names:
                        name_pairs.append(utils.Any(name=a_name, as_name=None))
                else:
                    try:
                        module = self.use_module('{}.{}'.format(node.module, alias.name))
                        self.emit("import * as {} from '{}';\n".format(self.filter_id(alias.asname) if alias.asname else self.filter_id(alias.name), module.import_rel_path))
                        self.all_imported_names.add(alias.asname or alias.name)
                    except:
                        module = self.use_module(node.module)
                        name_pairs.append(utils.Any(name=alias.name, as_name=alias.asname))
                        facility_imported = True
            if facility_imported:
                module = self.use_module(node.module)
                name_pairs.append(utils.Any(name=alias.name, as_name=alias.asname))
            if name_pairs:
                try:
                    self.emit('import {{')
                    for index, name_pair in enumerate(sorted(name_pairs, key=lambda name_pair: name_pair.as_name if name_pair.as_name else name_pair.name)):
                        if not (name_pair.as_name if name_pair.as_name else name_pair.name) in self.all_own_names | self.all_imported_names:
                            self.emit_comma(index)
                            self.emit(self.filter_id(name_pair.name))
                            if name_pair.as_name:
                                self.emit(' as {}'.format(self.filter_id(name_pair.as_name)))
                                self.all_imported_names.add(name_pair.as_name)
                            else:
                                self.all_imported_names.add(name_pair.name)
                    self.emit("}} from '{}';\n".format(module.import_rel_path))
                except:
                    print('Unexpected import error:', traceback.format_exc())
        except Exception as exception:
            utils.enhance_exception(exception, line_nr=self.line_nr, message="\n\tCan't import from module '{}'".format(node.module))

    def visit_joined_str(self, node: object) -> None:
        self.emit(repr(''.join([value.value if type(value) == ast.Constant else '{{}}' for value in node.values])))
        self.emit('.format (')
        index = 0
        for value in node.values:
            if type(value) == ast.FormattedValue:
                self.emit_comma(index)
                self.visit(value)
                index += 1
        self.emit(')')

    def visit_lambda(self, node: object) -> None:
        self.emit('(function __lambda__ ')
        self.visit(node.args)
        self.emit('return ')
        self.visit(node.body)
        self.dedent()
        self.emit(';\n}})')

    def visit_list(self, node: object) -> None:
        self.emit('[')
        for index, elt in enumerate(node.elts):
            self.emit_comma(index)
            self.visit(elt)
        self.emit(']')

    def visit_list_comp(self, node: object, is_set: bool = False, is_dict: bool = False, is_gen_exp: bool = False) -> None:
        elts = []
        bodies = [[]]

        def nest_loops(generators: list) -> None:
            for comprehension in generators:
                target = comprehension.target
                iter = comprehension.iter
                bodies.append([])
                bodies[-2].append(ast.For(target, iter, bodies[-1], []))
                for expr in comprehension.ifs:
                    test = expr
                    bodies.append([])
                    bodies[-2].append(ast.If(test=test, body=bodies[-1], orelse=[]))
            bodies[-1].append(ast.Call(func=ast.Attribute(value=ast.Name(id=self.get_temp('accu'), ctx=ast.Load), attr='append', ctx=ast.Load), args=[ast.List(elts=[node.key, node.value], ctx=ast.Load) if is_dict else node.elt], keywords=[]))
            self.visit(bodies[0][0])
        self.emit('(function () {{\n')
        self.inscope(ast.FunctionDef())
        self.indent()
        self.emit('var {} = [];\n'.format(self.next_temp('accu')))
        nest_loops(node.generators[:])
        self.emit('return {}{}{};\n'.format('set (' if is_set else 'dict (' if is_dict else '{} ('.format(self.filter_id('iter')) if is_gen_exp else '', self.get_temp('accu'), ')' if is_set or is_dict or is_gen_exp else ''))
        self.prev_temp('accu')
        self.dedent()
        self.descope()
        self.emit('}}) ()')

    def visit_module(self, node: object) -> None:
        self.adapt_line_nr_string()
        self.emit("// {}'ed from Python, {}\n".format(self.module.program.envir.transpiler_name.capitalize(), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.adapt_line_nr_string(node)
        self.inscope(node)
        self.import_hoist_fragment_index = self.fragment_index
        self.emit("var __name__ = '{}';\n".format(self.module.__name__))
        self.all_own_names.add('__name__')
        for statement in node.body:
            if self.is_comment_string(statement):
                pass
            else:
                self.visit(statement)
                self.emit(';\n')
        if self.allow_doc_attribs:
            doc_string = ast.get_docstring(node)
            if doc_string:
                self.all_own_names.add('__doc__')
        self.fragment_index = self.import_hoist_fragment_index
        if self.allow_doc_attribs and doc_string:
            self.emit("export var __doc__ = '{}';\n".format(doc_string.replace('\n', '\\n')))
        "\n        Make the globals () function work as well as possible in conjunction with JavaScript 6 modules rather than closures\n\n        JavaScript 6 module-level variables normally cannot be accessed directly by their name as a string\n        They aren't attributes of any global object, certainly not in strict mode, which is the default for modules\n        By making getters and setters by the same name members of __all__, we can approach globals () as a dictionary\n\n        Limitations:\n        - We can access (read/write) but not create module-level globals this way\n        - If there are a lot of globals (bad style) this mechanism becomes expensive, so it must be under a pragma\n\n        It's possible that future versions of JavaScript facilitate better solutions to this minor problem\n        "
        if self.allow_globals:
            self.emit('var __all__ = dict ({{' + ', '.join([f'get {name} () {{{{return {name};}}}}, set {name} (value) {{{{{name} = value;}}}}' for name in sorted(self.all_own_names)]) + '}});\n')
        self.fragment_index = self.import_hoist_fragment_index
        for import_hoist_memo in reversed(self.import_hoist_memos):
            if type(import_hoist_memo.node) == ast.Import:
                self.revisit_Import(import_hoist_memo)
            else:
                self.revisit_ImportFrom(import_hoist_memo)
        if utils.command_args.xreex or self.module.source_prename == '__init__':
            if self.all_imported_names:
                self.emit('export {{{}}};\n'.format(', '.join([self.filter_id(imported_name) for imported_name in self.all_imported_names])))
        self.fragment_index = self.import_hoist_fragment_index
        if self.module.name != self.module.program.runtimeModuleName:
            runtime_module = self.module.program.module_dict[self.module.program.runtimeModuleName]
            imported_names_from_runtime = ', '.join(sorted([exported_name_from_runtime for exported_name_from_runtime in runtime_module.exported_names if not exported_name_from_runtime in self.all_own_names | self.all_imported_names]))
            self.emit("import {{{}}} from '{}';\n".format(imported_names_from_runtime, runtime_module.import_rel_path))
        self.fragment_index = self.import_hoist_fragment_index
        for import_head in sorted(self.import_heads):
            self.emit('var {} = {{}};\n'.format(self.filter_id(import_head)))
        self.descope()

    def visit_name(self, node: object) -> None:
        if node.id == '__file__':
            self.visit(ast.Constant(value=self.module.source_path))
            return
        elif node.id == '__filename__':
            path = os.path.split(self.module.source_path)
            file_name = path[1]
            if file_name.startswith('__init__'):
                sub_dir = os.path.split(path[0])
                file_name = os.path.join(sub_dir[1], file_name)
            self.visit(ast.Constant(value=file_name))
            return
        elif node.id == '__line__':
            self.visit(ast.Constant(value=self.line_nr))
            return
        elif type(node.ctx) == ast.Store:
            if type(self.get_scope().node) == ast.Module:
                self.all_own_names.add(node.id)
        self.emit(self.filter_id(node.id))

    def visit_nonlocal(self, node: object) -> None:
        self.get_scope(ast.FunctionDef, ast.AsyncFunctionDef).nonlocals.update(node.names)

    def visit_pass(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        self.emit('// pass')

    def visit_raise(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        if node.exc:
            self.emit('var {} = '.format(self.next_temp('except')))
            self.visit(node.exc)
            self.emit(';\n')
        else:
            pass
        self.emit('{}.__cause__ = '.format(self.get_temp('except')))
        if node.cause:
            self.visit(node.cause)
        else:
            self.emit('null')
        self.emit(';\n')
        self.emit('throw {}'.format(self.get_temp('except')))
        if node.exc:
            self.prev_temp('except')

    def visit_return(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        self.emit('return ')
        if node.value:
            self.visit(node.value)

    def visit_set(self, node: object) -> None:
        self.emit('new set ([')
        for index, elt in enumerate(node.elts):
            self.emit_comma(index)
            self.visit(elt)
        self.emit('])')

    def visit_set_comp(self, node: object) -> None:
        self.visit_ListComp(node, is_set=True)

    def visit_slice(self, node: object) -> None:
        self.emit('tuple ([')
        if node.lower == None:
            self.emit('0')
        else:
            self.visit(node.lower)
        self.emit(', ')
        if node.upper == None:
            self.emit('null')
        else:
            self.visit(node.upper)
        self.emit(', ')
        if node.step == None:
            self.emit('1')
        else:
            self.visit(node.step)
        self.emit('])')

    def visit_subscript(self, node: object) -> None:
        if type(node.slice) == ast.Index:
            if type(node.slice.value) == ast.Tuple:
                self.visit(node.value)
                self.emit('.__getitem__ (')
                self.strip_tuple = True
                self.visit(node.slice.value)
                self.emit(')')
            elif self.allow_operator_overloading:
                self.emit('__getitem__ (')
                self.visit(node.value)
                self.emit(', ')
                self.visit(node.slice.value)
                self.emit(')')
            else:
                try:
                    is_rhs_index = not self.expecting_non_overloaded_lhs_index
                    self.expecting_non_overloaded_lhs_index = False
                    if is_rhs_index and self.allow_key_check:
                        self.emit('__k__ (')
                        self.visit(node.value)
                        self.emit(', ')
                        self.visit(node.slice.value)
                        self.emit(')')
                    else:
                        self.visit(node.value)
                        self.emit(' [')
                        self.visit(node.slice.value)
                        self.emit(']')
                except:
                    print(traceback.format_exc())
        elif type(node.slice) == ast.Slice:
            if self.allow_operator_overloading:
                self.emit('__getslice__ (')
                self.visit(node.value)
                self.emit(', ')
            else:
                self.visit(node.value)
                self.emit('.__getslice__ (')
            if node.slice.lower == None:
                self.emit('0')
            else:
                self.visit(node.slice.lower)
            self.emit(', ')
            if node.slice.upper == None:
                self.emit('null')
            else:
                self.visit(node.slice.upper)
            self.emit(', ')
            if node.slice.step:
                self.visit(node.slice.step)
            else:
                self.emit('null')
            self.emit(', ')
            self.visit(node.value)
            self.emit(')')
        elif type(node.slice) == ast.ExtSlice:
            self.visit(node.value)
            self.emit('.__getitem__ (')
            self.emit('[')
            for index, dim in enumerate(node.slice.dims):
                self.emit_comma(index)
                self.visit(dim)
            self.emit(']')
            self.emit(', ')
            self.visit(node.value)
            self.emit(')')

    def visit_try(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        self.emit('try {{\n')
        self.indent()
        self.emit_body(node.body)
        if node.orelse:
            self.emit('try {{\n')
            self.indent()
            self.emit_body(node.orelse)
            self.dedent()
            self.emit('}}\n')
            self.emit('catch ({}) {{\n'.format(self.next_temp('except')))
            self.emit('}}\n')
            self.prev_temp('except')
        self.dedent()
        self.emit('}}\n')
        if node.handlers:
            self.emit('catch ({}) {{\n'.format(self.next_temp('except')))
            self.indent()
            for index, exception_handler in enumerate(node.handlers):
                if index:
                    self.emit('else ')
                if exception_handler.type:
                    self.emit('if (isinstance ({}, '.format(self.get_temp('except')))
                    self.visit(exception_handler.type)
                    self.emit(')) {{\n')
                    self.indent()
                    if exception_handler.name:
                        self.emit('var {} = {};\n'.format(exception_handler.name, self.get_temp('except')))
                    self.emit_body(exception_handler.body)
                    self.dedent()
                    self.emit('}}\n')
                else:
                    self.emit_body(exception_handler.body)
                    break
            else:
                self.emit('else {{\n')
                self.indent()
                self.emit('throw {};\n'.format(self.get_temp('except')))
                self.dedent()
                self.emit('}}\n')
            self.dedent()
            self.prev_temp('except')
            self.emit('}}\n')
        if node.finalbody:
            self.emit('finally {{\n')
            self.indent()
            self.emit_body(node.finalbody)
            self.dedent()
            self.emit('}}\n')

    def visit_tuple(self, node: object) -> None:
        keep_tuple = not (self.strip_tuple or self.strip_tuples)
        self.strip_tuple = False
        if keep_tuple:
            self.emit('tuple (')
        self.emit('[')
        for index, elt in enumerate(node.elts):
            self.emit_comma(index)
            self.visit(elt)
        self.emit(']')
        if keep_tuple:
            self.emit(')')

    def visit_unary_op(self, node: object) -> None:
        if self.allow_operator_overloading and type(node.op) == ast.USub:
            self.emit('{} ('.format(self.filter_id('__neg__')))
            self.visit_sub_expr(node, node.operand)
            self.emit(')')
        else:
            self.emit(self.operators[type(node.op)][0])
            self.emit_begin_truthy()
            self.visit_sub_expr(node, node.operand)
            self.emit_end_truthy()

    def visit_while(self, node: object) -> None:
        self.adapt_line_nr_string(node)
        if node.orelse:
            self.emit('var {} = false;\n'.format(self.next_temp('break')))
        else:
            self.skip_temp('break')
        self.emit('while (')
        self.emit_begin_truthy()
        self.visit(node.test)
        self.emit_end_truthy()
        self.emit(') {{\n')
        self.indent()
        self.emit_body(node.body)
        self.dedent()
        self.emit('}}\n')
        if node.orelse:
            self.adapt_line_nr_string(node.orelse, 1)
            self.emit('if (!{}) {{\n'.format(self.get_temp('break')))
            self.prev_temp('break')
            self.indent()
            self.emit_body(node.orelse)
            self.dedent()
            self.emit('}}\n')
        else:
            self.prev_temp('break')

    def visit_with(self, node: object) -> None:
        from contextlib import contextmanager, ExitStack
        self.adapt_line_nr_string(node)

        @contextmanager
        def item_context(item: object) -> None:
            if not self.noskip_code_generation:
                yield
                return
            self.emit('var ')
            if item.optional_vars:
                self.visit(item.optional_vars)
                with_id = item.optional_vars.id
            else:
                with_id = self.next_temp('withid')
                self.emit(with_id)
            self.emit(' = ')
            self.visit(item.context_expr)
            self.emit(';\n')
            self.emit('try {{\n')
            self.indent()
            self.emit('{}.__enter__ ();\n'.format(with_id))
            yield
            self.emit('{}.__exit__ ();\n'.format(with_id))
            self.dedent()
            self.emit('}}\n')
            self.emit('catch ({}) {{\n'.format(self.next_temp('except')))
            self.indent()
            self.emit('if (! ({0}.__exit__ ({1}.name, {1}, {1}.stack))) {{\n'.format(with_id, self.get_temp('except')))
            self.indent()
            self.emit('throw {};\n'.format(self.get_temp('except')))
            self.dedent()
            self.emit('}}\n')
            self.dedent()
            self.emit('}}\n')
            self.prev_temp('except')
            if with_id == self.get_temp('withid'):
                self.prev_temp('withid')

        @contextmanager
        def pragma_context(item: object) -> None:
            expr = item.context_expr
            name = expr.args[0].s
            if name.startswith('no'):
                rev_name = name[2:]
            else:
                rev_name = 'no' + name
            self.visit(expr)
            yield
            self.visit(ast.Call(expr.func, [ast.Constant(value=rev_name)] + expr.args[1:]))

        @contextmanager
        def skip_context(item: object) -> None:
            self.noskip_code_generation = False
            yield
            self.noskip_code_generation = True
        with ExitStack() as stack:
            for item in node.items:
                expr = item.context_expr
                if self.is_call(expr, '__pragma__'):
                    if expr.args[0].s == 'skip':
                        stack.enter_context(skip_context(item))
                    else:
                        stack.enter_context(pragma_context(item))
                else:
                    stack.enter_context(item_context(item))
            self.emit_body(node.body)

    def visit_yield(self, node: object) -> None:
        self.get_scope(ast.FunctionDef, ast.AsyncFunctionDef).contains_yield = True
        self.emit('yield')
        if node.value != None:
            self.emit(' ')
            self.visit(node.value)

    def visit_yield_from(self, node: object) -> None:
        self.get_scope(ast.FunctionDef, ast.AsyncFunctionDef).contains_yield = True
        self.emit('yield* ')
        self.visit(node.value)
