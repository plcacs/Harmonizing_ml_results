from typing import Any, Dict, Optional
from .. import fixer_base
from ..fixer_util import Call, Comma, Name, Node, BlankLine, syms
from lib2to3.pytree import Node

CMP = "(n='!=' | '==' | 'is' | n=comp_op< 'is' 'not' >)"
TYPE = "power< 'type' trailer< '(' x=any ')' > >"


class FixIdioms(fixer_base.BaseFix):
    explicit: bool = True
    PATTERN: str = (
        "\n        isinstance=comparison< %s %s T=any >\n"
        "        |\n"
        "        isinstance=comparison< T=any %s %s >\n"
        "        |\n"
        "        while_stmt< 'while' while='1' ':' any+ >\n"
        "        |\n"
        "        sorted=any<\n"
        "            any*\n"
        "            simple_stmt<\n"
        "              expr_stmt< id1=any '='\n"
        "                     power< list='list' trailer< '(' (not arglist<any+>) any ')' > >\n"
        "              >\n"
        "              '\\n'\n"
        "            >\n"
        "            sort=\n"
        "            simple_stmt<\n"
        "              power< id2=any\n"
        "                     trailer< '.' 'sort' > trailer< '(' ')' >\n"
        "              >\n"
        "              '\\n'\n"
        "            >\n"
        "            next=any*\n"
        "        >\n"
        "        |\n"
        "        sorted=any<\n"
        "            any*\n"
        "            simple_stmt< expr_stmt< id1=any '=' expr=any > '\\n' >\n"
        "            sort=\n"
        "            simple_stmt<\n"
        "              power< id2=any\n"
        "                     trailer< '.' 'sort' > trailer< '(' ')' >\n"
        "              >\n"
        "              '\\n'\n"
        "            >\n"
        "            next=any*\n"
        "        >\n"
        "    " % (TYPE, CMP, CMP, TYPE)
    )

    def match(self, node: Node) -> Optional[Dict[str, Any]]:
        r = super(FixIdioms, self).match(node)
        if r and 'sorted' in r:
            if r.get('id1') == r.get('id2'):
                return r
            return None
        return r

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        if 'isinstance' in results:
            return self.transform_isinstance(node, results)
        elif 'while' in results:
            return self.transform_while(node, results)
        elif 'sorted' in results:
            return self.transform_sort(node, results)
        else:
            raise RuntimeError('Invalid match')

    def transform_isinstance(self, node: Node, results: Dict[str, Any]) -> Node:
        x = results['x'].clone()
        T = results['T'].clone()
        x.prefix = ''
        T.prefix = ' '
        test = Call(Name('isinstance'), [x, Comma(), T])
        if 'n' in results:
            test.prefix = ' '
            test = Node(syms.not_test, [Name('not'), test])
        test.prefix = node.prefix
        return test

    def transform_while(self, node: Node, results: Dict[str, Any]) -> None:
        one = results['while']
        one.replace(Name('True', prefix=one.prefix))

    def transform_sort(self, node: Node, results: Dict[str, Any]) -> None:
        sort_stmt = results['sort']
        next_stmt = results.get('next')
        list_call = results.get('list')
        simple_expr = results.get('expr')
        if list_call:
            list_call.replace(Name('sorted', prefix=list_call.prefix))
        elif simple_expr:
            new = simple_expr.clone()
            new.prefix = ''
            simple_expr.replace(Call(Name('sorted'), [new], prefix=simple_expr.prefix))
        else:
            raise RuntimeError('should not have reached here')
        sort_stmt.remove()
        btwn = sort_stmt.prefix
        if '\n' in btwn:
            if next_stmt:
                prefix_lines = (btwn.rpartition('\n')[0], next_stmt[0].prefix)
                next_stmt[0].prefix = '\n'.join(prefix_lines)
            else:
                assert list_call.parent
                assert list_call.next_sibling is None
                end_line = BlankLine()
                list_call.parent.append_child(end_line)
                assert list_call.next_sibling is end_line
                end_line.prefix = btwn.rpartition('\n')[0]
