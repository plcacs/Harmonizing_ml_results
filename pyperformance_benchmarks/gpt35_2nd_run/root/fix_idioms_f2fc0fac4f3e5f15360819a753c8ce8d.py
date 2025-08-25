from typing import Any

CMP: str = "(n='!=' | '==' | 'is' | n=comp_op< 'is' 'not' >)"
TYPE: str = "power< 'type' trailer< '(' x=Any ')' > >"

class FixIdioms(fixer_base.BaseFix):
    explicit: bool = True
    PATTERN: str = ("\n        isinstance=comparison< %s %s T=Any >\n        |\n        isinstance=comparison< T=Any %s %s >\n        |\n        while_stmt< 'while' while='True' ':' Any+ >\n        |\n        sorted=Any<\n            Any*\n            simple_stmt<\n              expr_stmt< id1=Any '='\n                         power< list='list' trailer< '(' (not arglist<Any+>) Any ')' > >\n              >\n              '\\n'\n            >\n            sort=\n            simple_stmt<\n              power< id2=Any\n                     trailer< '.' 'sort' > trailer< '(' ')' >\n              >\n              '\\n'\n            >\n            next=Any*\n        >\n        |\n        sorted=Any<\n            Any*\n            simple_stmt< expr_stmt< id1=Any '=' expr=Any > '\\n' >\n            sort=\n            simple_stmt<\n              power< id2=Any\n                     trailer< '.' 'sort' > trailer< '(' ')' >\n              >\n              '\\n'\n            >\n            next=Any*\n        >\n    " % (TYPE, CMP, CMP, TYPE))

    def match(self, node: Any) -> Any:
        r: Any = super(FixIdioms, self).match(node)
        if (r and ('sorted' in r)):
            if (r['id1'] == r['id2']):
                return r
            return None
        return r

    def transform(self, node: Any, results: Any) -> Any:
        if ('isinstance' in results):
            return self.transform_isinstance(node, results)
        elif ('while' in results):
            return self.transform_while(node, results)
        elif ('sorted' in results):
            return self.transform_sort(node, results)
        else:
            raise RuntimeError('Invalid match')

    def transform_isinstance(self, node: Any, results: Any) -> Any:
        x: Any = results['x'].clone()
        T: Any = results['T'].clone()
        x.prefix = ''
        T.prefix = ' '
        test: Any = Call(Name('isinstance'), [x, Comma(), T])
        if ('n' in results):
            test.prefix = ' '
            test = Node(syms.not_test, [Name('not'), test])
        test.prefix = node.prefix
        return test

    def transform_while(self, node: Any, results: Any) -> None:
        one: Any = results['while']
        one.replace(Name('True', prefix=one.prefix))

    def transform_sort(self, node: Any, results: Any) -> None:
        sort_stmt: Any = results['sort']
        next_stmt: Any = results['next']
        list_call: Any = results.get('list')
        simple_expr: Any = results.get('expr')
        if list_call:
            list_call.replace(Name('sorted', prefix=list_call.prefix))
        elif simple_expr:
            new: Any = simple_expr.clone()
            new.prefix = ''
            simple_expr.replace(Call(Name('sorted'), [new], prefix=simple_expr.prefix))
        else:
            raise RuntimeError('should not have reached here')
        sort_stmt.remove()
        btwn: str = sort_stmt.prefix
        if ('\n' in btwn):
            if next_stmt:
                prefix_lines: tuple = (btwn.rpartition('\n')[0], next_stmt[0].prefix)
                next_stmt[0].prefix = '\n'.join(prefix_lines)
            else:
                assert list_call.parent
                assert (list_call.next_sibling is None)
                end_line: Any = BlankLine()
                list_call.parent.append_child(end_line)
                assert (list_call.next_sibling is end_line)
                end_line.prefix = btwn.rpartition('\n')[0]
