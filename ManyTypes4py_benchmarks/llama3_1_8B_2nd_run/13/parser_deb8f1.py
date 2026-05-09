from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser

NAME: PythonTokenTypes = PythonTokenTypes.NAME
INDENT: PythonTokenTypes = PythonTokenTypes.INDENT
DEDENT: PythonTokenTypes = PythonTokenTypes.DEDENT

class Parser(BaseParser):
    """
    This class is used to parse a Python file, it then divides them into a
    class structure of different scopes.

    :param pgen_grammar: The grammar object of pgen2. Loaded by load_grammar.
    """
    node_map: dict[str, type] = {
        'expr_stmt': tree.ExprStmt,
        'classdef': tree.Class,
        'funcdef': tree.Function,
        'file_input': tree.Module,
        'import_name': tree.ImportName,
        'import_from': tree.ImportFrom,
        'break_stmt': tree.KeywordStatement,
        'continue_stmt': tree.KeywordStatement,
        'return_stmt': tree.ReturnStmt,
        'raise_stmt': tree.KeywordStatement,
        'yield_expr': tree.YieldExpr,
        'del_stmt': tree.KeywordStatement,
        'pass_stmt': tree.KeywordStatement,
        'global_stmt': tree.GlobalStmt,
        'nonlocal_stmt': tree.KeywordStatement,
        'print_stmt': tree.KeywordStatement,
        'assert_stmt': tree.AssertStmt,
        'if_stmt': tree.IfStmt,
        'with_stmt': tree.WithStmt,
        'for_stmt': tree.ForStmt,
        'while_stmt': tree.WhileStmt,
        'try_stmt': tree.TryStmt,
        'sync_comp_for': tree.SyncCompFor,
        'decorator': tree.Decorator,
        'lambdef': tree.Lambda,
        'lambdef_nocond': tree.Lambda,
        'namedexpr_test': tree.NamedExpr
    }
    default_node: type = tree.PythonNode
    _leaf_map: dict[PythonTokenTypes, type] = {
        PythonTokenTypes.STRING: tree.String,
        PythonTokenTypes.NUMBER: tree.Number,
        PythonTokenTypes.NEWLINE: tree.Newline,
        PythonTokenTypes.ENDMARKER: tree.EndMarker,
        PythonTokenTypes.FSTRING_STRING: tree.FStringString,
        PythonTokenTypes.FSTRING_START: tree.FStringStart,
        PythonTokenTypes.FSTRING_END: tree.FStringEnd
    }

    def __init__(self, pgen_grammar: object, error_recovery: bool = True, start_nonterminal: str = 'file_input') -> None:
        super().__init__(pgen_grammar, start_nonterminal, error_recovery=error_recovery)
        self.syntax_errors: list = []
        self._omit_dedent_list: list = []
        self._indent_counter: int = 0

    def parse(self, tokens: list) -> tree.PythonNode:
        if self._error_recovery:
            if self._start_nonterminal != 'file_input':
                raise NotImplementedError
            tokens = self._recovery_tokenize(tokens)
        return super().parse(tokens)

    def convert_node(self, nonterminal: str, children: list) -> tree.PythonNode:
        """
        Convert raw node information to a PythonBaseNode instance.

        This is passed to the parser driver which calls it whenever a reduction of a
        grammar rule produces a new complete node, so that the tree is build
        strictly bottom-up.
        """
        try:
            node = self.node_map[nonterminal](children)
        except KeyError:
            if nonterminal == 'suite':
                children = [children[0]] + children[2:-1]
            node = self.default_node(nonterminal, children)
        return node

    def convert_leaf(self, type: PythonTokenTypes, value: str, prefix: str, start_pos: int) -> tree.PythonNode:
        if type == NAME:
            if value in self._pgen_grammar.reserved_syntax_strings:
                return tree.Keyword(value, start_pos, prefix)
            else:
                return tree.Name(value, start_pos, prefix)
        return self._leaf_map.get(type, tree.Operator)(value, start_pos, prefix)

    def error_recovery(self, token: tuple[PythonTokenTypes, str, int, str]) -> None:
        tos_nodes = self.stack[-1].nodes
        if tos_nodes:
            last_leaf = tos_nodes[-1].get_last_leaf()
        else:
            last_leaf = None
        if self._start_nonterminal == 'file_input' and (token[0] == PythonTokenTypes.ENDMARKER or (token[0] == DEDENT and (not last_leaf.value.endswith('\n')) and (not last_leaf.value.endswith('\r')))):
            if self.stack[-1].dfa.from_rule == 'simple_stmt':
                try:
                    plan = self.stack[-1].dfa.transitions[PythonTokenTypes.NEWLINE]
                except KeyError:
                    pass
                else:
                    if plan.next_dfa.is_final and (not plan.dfa_pushes):
                        self.stack[-1].dfa = plan.next_dfa
                        self._add_token(token)
                        return
        if not self._error_recovery:
            return super().error_recovery(token)

        def current_suite(stack: list) -> int:
            for until_index, stack_node in reversed(list(enumerate(stack))):
                if stack_node.nonterminal == 'file_input':
                    break
                elif stack_node.nonterminal == 'suite':
                    if len(stack_node.nodes) != 1:
                        break
            return until_index
        until_index = current_suite(self.stack)
        if self._stack_removal(until_index + 1):
            self._add_token(token)
        else:
            typ, value, start_pos, prefix = token
            if typ == INDENT:
                self._omit_dedent_list.append(self._indent_counter)
            error_leaf = tree.PythonErrorLeaf(typ.name, value, start_pos, prefix)
            self.stack[-1].nodes.append(error_leaf)
        tos = self.stack[-1]
        if tos.nonterminal == 'suite':
            try:
                tos.dfa = tos.dfa.arcs['stmt']
            except KeyError:
                pass

    def _stack_removal(self, start_index: int) -> bool:
        all_nodes = [node for stack_node in self.stack[start_index:] for node in stack_node.nodes]
        if all_nodes:
            node = tree.PythonErrorNode(all_nodes)
            self.stack[start_index - 1].nodes.append(node)
        self.stack[start_index:] = []
        return bool(all_nodes)

    def _recovery_tokenize(self, tokens: list) -> list:
        for token in tokens:
            typ = token[0]
            if typ == DEDENT:
                o = self._omit_dedent_list
                if o and o[-1] == self._indent_counter:
                    o.pop()
                    self._indent_counter -= 1
                    continue
                self._indent_counter -= 1
            elif typ == INDENT:
                self._indent_counter += 1
            yield token
