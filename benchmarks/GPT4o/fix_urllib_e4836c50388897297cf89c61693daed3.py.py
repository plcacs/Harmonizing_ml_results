from typing import Generator, List, Optional, Union
from lib2to3.fixes.fix_imports import alternates, FixImports
from lib2to3.fixer_util import Name, Comma, FromImport, Newline, find_indentation, Node, syms

MAPPING: dict = {
    'urllib': [
        ('urllib.request', ['URLopener', 'FancyURLopener', 'urlretrieve', '_urlopener', 'urlopen', 'urlcleanup', 'pathname2url', 'url2pathname', 'getproxies']),
        ('urllib.parse', ['quote', 'quote_plus', 'unquote', 'unquote_plus', 'urlencode', 'splitattr', 'splithost', 'splitnport', 'splitpasswd', 'splitport', 'splitquery', 'splittag', 'splittype', 'splituser', 'splitvalue']),
        ('urllib.error', ['ContentTooShortError'])
    ],
    'urllib2': [
        ('urllib.request', ['urlopen', 'install_opener', 'build_opener', 'Request', 'OpenerDirector', 'BaseHandler', 'HTTPDefaultErrorHandler', 'HTTPRedirectHandler', 'HTTPCookieProcessor', 'ProxyHandler', 'HTTPPasswordMgr', 'HTTPPasswordMgrWithDefaultRealm', 'AbstractBasicAuthHandler', 'HTTPBasicAuthHandler', 'ProxyBasicAuthHandler', 'AbstractDigestAuthHandler', 'HTTPDigestAuthHandler', 'ProxyDigestAuthHandler', 'HTTPHandler', 'HTTPSHandler', 'FileHandler', 'FTPHandler', 'CacheFTPHandler', 'UnknownHandler']),
        ('urllib.error', ['URLError', 'HTTPError'])
    ]
}
MAPPING['urllib2'].append(MAPPING['urllib'][1])

def build_pattern() -> Generator[str, None, None]:
    for old_module, changes in MAPPING.items():
        for new_module, members in changes:
            members = alternates(members)
            yield ("import_name< 'import' (module=%r\n                                  | dotted_as_names< any* module=%r any* >) >\n                  " % (old_module, old_module))
            yield ("import_from< 'from' mod_member=%r 'import'\n                       ( member=%s | import_as_name< member=%s 'as' any > |\n                         import_as_names< members=any*  >) >\n                  " % (old_module, members, members))
            yield ("import_from< 'from' module_star=%r 'import' star='*' >\n                  " % old_module)
            yield ("import_name< 'import'\n                                  dotted_as_name< module_as=%r 'as' any > >\n                  " % old_module)
            yield ("power< bare_with_attr=%r trailer< '.' member=%s > any* >\n                  " % (old_module, members))

class FixUrllib(FixImports):

    def build_pattern(self) -> str:
        return '|'.join(build_pattern())

    def transform_import(self, node: Node, results: dict) -> None:
        import_mod = results.get('module')
        pref = import_mod.prefix
        names: List[Union[Name, Comma]] = []
        for name in MAPPING[import_mod.value][:-1]:
            names.extend([Name(name[0], prefix=pref), Comma()])
        names.append(Name(MAPPING[import_mod.value][-1][0], prefix=pref))
        import_mod.replace(names)

    def transform_member(self, node: Node, results: dict) -> None:
        mod_member = results.get('mod_member')
        pref = mod_member.prefix
        member = results.get('member')
        if member:
            if isinstance(member, list):
                member = member[0]
            new_name: Optional[str] = None
            for change in MAPPING[mod_member.value]:
                if member.value in change[1]:
                    new_name = change[0]
                    break
            if new_name:
                mod_member.replace(Name(new_name, prefix=pref))
            else:
                self.cannot_convert(node, 'This is an invalid module element')
        else:
            modules: List[str] = []
            mod_dict: dict = {}
            members = results['members']
            for member in members:
                if member.type == syms.import_as_name:
                    as_name = member.children[2].value
                    member_name = member.children[0].value
                else:
                    member_name = member.value
                    as_name = None
                if member_name != ',':
                    for change in MAPPING[mod_member.value]:
                        if member_name in change[1]:
                            if change[0] not in mod_dict:
                                modules.append(change[0])
                            mod_dict.setdefault(change[0], []).append(member)
            new_nodes: List[Node] = []
            indentation = find_indentation(node)
            first = True

            def handle_name(name: Node, prefix: str) -> List[Node]:
                if name.type == syms.import_as_name:
                    kids = [Name(name.children[0].value, prefix=prefix), name.children[1].clone(), name.children[2].clone()]
                    return [Node(syms.import_as_name, kids)]
                return [Name(name.value, prefix=prefix)]

            for module in modules:
                elts = mod_dict[module]
                names: List[Union[Name, Comma]] = []
                for elt in elts[:-1]:
                    names.extend(handle_name(elt, pref))
                    names.append(Comma())
                names.extend(handle_name(elts[-1], pref))
                new = FromImport(module, names)
                if not first or node.parent.prefix.endswith(indentation):
                    new.prefix = indentation
                new_nodes.append(new)
                first = False
            if new_nodes:
                nodes: List[Union[Node, Newline]] = []
                for new_node in new_nodes[:-1]:
                    nodes.extend([new_node, Newline()])
                nodes.append(new_nodes[-1])
                node.replace(nodes)
            else:
                self.cannot_convert(node, 'All module elements are invalid')

    def transform_dot(self, node: Node, results: dict) -> None:
        module_dot = results.get('bare_with_attr')
        member = results.get('member')
        new_name: Optional[str] = None
        if isinstance(member, list):
            member = member[0]
        for change in MAPPING[module_dot.value]:
            if member.value in change[1]:
                new_name = change[0]
                break
        if new_name:
            module_dot.replace(Name(new_name, prefix=module_dot.prefix))
        else:
            self.cannot_convert(node, 'This is an invalid module element')

    def transform(self, node: Node, results: dict) -> None:
        if results.get('module'):
            self.transform_import(node, results)
        elif results.get('mod_member'):
            self.transform_member(node, results)
        elif results.get('bare_with_attr'):
            self.transform_dot(node, results)
        elif results.get('module_star'):
            self.cannot_convert(node, 'Cannot handle star imports.')
        elif results.get('module_as'):
            self.cannot_convert(node, 'This module is now multiple modules')
