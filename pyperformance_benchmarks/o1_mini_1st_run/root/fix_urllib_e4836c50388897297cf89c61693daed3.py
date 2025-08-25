from typing import Any, Dict, Generator, List, Tuple, Optional
from lib2to3.fixes.fix_imports import alternates, FixImports
from lib2to3.fixer_util import (
    Name,
    Comma,
    FromImport,
    Newline,
    find_indentation,
    Node,
    syms,
)

MAPPING: Dict[str, List[Tuple[str, List[str]]]] = {
    'urllib': [
        (
            'urllib.request',
            [
                'URLopener',
                'FancyURLopener',
                'urlretrieve',
                '_urlopener',
                'urlopen',
                'urlcleanup',
                'pathname2url',
                'url2pathname',
                'getproxies',
            ],
        ),
        (
            'urllib.parse',
            [
                'quote',
                'quote_plus',
                'unquote',
                'unquote_plus',
                'urlencode',
                'splitattr',
                'splithost',
                'splitnport',
                'splitpasswd',
                'splitport',
                'splitquery',
                'splittag',
                'splittype',
                'splituser',
                'splitvalue',
            ],
        ),
        ('urllib.error', ['ContentTooShortError']),
    ],
    'urllib2': [
        (
            'urllib.request',
            [
                'urlopen',
                'install_opener',
                'build_opener',
                'Request',
                'OpenerDirector',
                'BaseHandler',
                'HTTPDefaultErrorHandler',
                'HTTPRedirectHandler',
                'HTTPCookieProcessor',
                'ProxyHandler',
                'HTTPPasswordMgr',
                'HTTPPasswordMgrWithDefaultRealm',
                'AbstractBasicAuthHandler',
                'HTTPBasicAuthHandler',
                'ProxyBasicAuthHandler',
                'AbstractDigestAuthHandler',
                'HTTPDigestAuthHandler',
                'ProxyDigestAuthHandler',
                'HTTPHandler',
                'HTTPSHandler',
                'FileHandler',
                'FTPHandler',
                'CacheFTPHandler',
                'UnknownHandler',
            ],
        ),
        ('urllib.error', ['URLError', 'HTTPError']),
    ],
}

MAPPING['urllib2'].append(MAPPING['urllib'][1])


def build_pattern() -> Generator[str, None, None]:
    bare: set = set()
    for old_module, changes in MAPPING.items():
        for change in changes:
            new_module, members = change
            members = alternates(members)
            yield (
                "import_name< 'import' (module=%r\n                                  | dotted_as_names< any* module=%r any* >) >\n                  "
                % (old_module, old_module)
            )
            yield (
                "import_from< 'from' mod_member=%r 'import'\n                       ( member=%s | import_as_name< member=%s 'as' any > |\n                         import_as_names< members=any*  >) >\n                  "
                % (old_module, members, members)
            )
            yield (
                "import_from< 'from' module_star=%r 'import' star='*' >\n                  "
                % old_module
            )
            yield (
                "import_name< 'import'\n                                  dotted_as_name< module_as=%r 'as' any > >\n                  "
                % old_module
            )
            yield (
                "power< bare_with_attr=%r trailer< '.' member=%s > any* >\n                  "
                % (old_module, members)
            )


class FixUrllib(FixImports):

    def build_pattern(self) -> str:
        return '|'.join(build_pattern())

    def transform_import(self, node: Node, results: Dict[str, Any]) -> None:
        'Transform for the basic import case. Replaces the old\n           import name with a comma separated list of its\n           replacements.\n        '
        import_mod: Node = results.get('module')
        assert isinstance(import_mod, Node)
        pref: str = import_mod.prefix
        names: List[Any] = []
        for name_tuple in MAPPING[import_mod.value][:-1]:
            names.extend([Name(name_tuple[0], prefix=pref), Comma()])
        last_name = MAPPING[import_mod.value][-1][0]
        names.append(Name(last_name, prefix=pref))
        import_mod.replace(names)

    def transform_member(self, node: Node, results: Dict[str, Any]) -> None:
        'Transform for imports of specific module elements. Replaces\n           the module to be imported from with the appropriate new\n           module.\n        '
        mod_member: Node = results.get('mod_member')
        assert isinstance(mod_member, Node)
        pref: str = mod_member.prefix
        member: Optional[Any] = results.get('member')
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
            mod_dict: Dict[str, List[Any]] = {}
            members = results['members']
            for member_node in members:
                if member_node.type == syms.import_as_name:
                    as_name = member_node.children[2].value
                    member_name = member_node.children[0].value
                else:
                    member_name = member_node.value
                    as_name = None
                if member_name != ',':
                    for change in MAPPING[mod_member.value]:
                        if member_name in change[1]:
                            if change[0] not in mod_dict:
                                modules.append(change[0])
                            mod_dict.setdefault(change[0], []).append(member_node)
            new_nodes: List[Any] = []
            indentation: str = find_indentation(node)
            first: bool = True

            def handle_name(name_node: Node, prefix: str) -> List[Node]:
                if name_node.type == syms.import_as_name:
                    kids = [
                        Name(name_node.children[0].value, prefix=prefix),
                        name_node.children[1].clone(),
                        name_node.children[2].clone(),
                    ]
                    return [Node(syms.import_as_name, kids)]
                return [Name(name_node.value, prefix=prefix)]

            for module in modules:
                elts = mod_dict[module]
                names_list: List[Any] = []
                for elt in elts[:-1]:
                    names_list.extend(handle_name(elt, pref))
                    names_list.append(Comma())
                names_list.extend(handle_name(elts[-1], pref))
                new = FromImport(module, names_list)
                if not first or node.parent.prefix.endswith(indentation):
                    new.prefix = indentation
                new_nodes.append(new)
                first = False
            if new_nodes:
                nodes: List[Any] = []
                for new_node in new_nodes[:-1]:
                    nodes.extend([new_node, Newline()])
                nodes.append(new_nodes[-1])
                node.replace(nodes)
            else:
                self.cannot_convert(node, 'All module elements are invalid')

    def transform_dot(self, node: Node, results: Dict[str, Any]) -> None:
        'Transform for calls to module members in code.'
        module_dot: Node = results.get('bare_with_attr')
        assert isinstance(module_dot, Node)
        member: Optional[Any] = results.get('member')
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

    def transform(self, node: Node, results: Dict[str, Any]) -> None:
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
