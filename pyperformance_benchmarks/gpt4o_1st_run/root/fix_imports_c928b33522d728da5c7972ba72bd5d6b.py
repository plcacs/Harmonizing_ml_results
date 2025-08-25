from __future__ import annotations
from typing import Generator
from .. import fixer_base
from ..fixer_util import Name, attr_chain

MAPPING: dict[str, str] = {
    'StringIO': 'io', 'cStringIO': 'io', 'cPickle': 'pickle', '__builtin__': 'builtins',
    'copy_reg': 'copyreg', 'Queue': 'queue', 'SocketServer': 'socketserver',
    'ConfigParser': 'configparser', 'repr': 'reprlib', 'FileDialog': 'tkinter.filedialog',
    'tkFileDialog': 'tkinter.filedialog', 'SimpleDialog': 'tkinter.simpledialog',
    'tkSimpleDialog': 'tkinter.simpledialog', 'tkColorChooser': 'tkinter.colorchooser',
    'tkCommonDialog': 'tkinter.commondialog', 'Dialog': 'tkinter.dialog', 'Tkdnd': 'tkinter.dnd',
    'tkFont': 'tkinter.font', 'tkMessageBox': 'tkinter.messagebox', 'ScrolledText': 'tkinter.scrolledtext',
    'Tkconstants': 'tkinter.constants', 'Tix': 'tkinter.tix', 'ttk': 'tkinter.ttk', 'Tkinter': 'tkinter',
    'markupbase': '_markupbase', '_winreg': 'winreg', 'thread': '_thread', 'dummy_thread': '_dummy_thread',
    'dbhash': 'dbm.bsd', 'dumbdbm': 'dbm.dumb', 'dbm': 'dbm.ndbm', 'gdbm': 'dbm.gnu',
    'xmlrpclib': 'xmlrpc.client', 'DocXMLRPCServer': 'xmlrpc.server', 'SimpleXMLRPCServer': 'xmlrpc.server',
    'httplib': 'http.client', 'htmlentitydefs': 'html.entities', 'HTMLParser': 'html.parser',
    'Cookie': 'http.cookies', 'cookielib': 'http.cookiejar', 'BaseHTTPServer': 'http.server',
    'SimpleHTTPServer': 'http.server', 'CGIHTTPServer': 'http.server', 'commands': 'subprocess',
    'UserString': 'collections', 'UserList': 'collections', 'urlparse': 'urllib.parse',
    'robotparser': 'urllib.robotparser'
}

def alternates(members: list[str]) -> str:
    return '(' + '|'.join(map(repr, members)) + ')'

def build_pattern(mapping: dict[str, str] = MAPPING) -> Generator[str, None, None]:
    mod_list = ' | '.join([f"module_name='{key}'" for key in mapping])
    bare_names = alternates(list(mapping.keys()))
    yield (f"name_import=import_name< 'import' (({mod_list}) |\n"
           f"               multiple_imports=dotted_as_names< any* ({mod_list}) any* >) >\n          ")
    yield (f"import_from< 'from' ({mod_list}) 'import' ['(']\n"
           f"              ( any | import_as_name< any 'as' any > |\n"
           f"                import_as_names< any* >)  [')'] >\n          ")
    yield (f"import_name< 'import' (dotted_as_name< ({mod_list}) 'as' any > |\n"
           f"               multiple_imports=dotted_as_names<\n"
           f"                 any* dotted_as_name< ({mod_list}) 'as' any > any* >) >\n          ")
    yield f"power< bare_with_attr=({bare_names}) trailer<'.' any > any* >"

class FixImports(fixer_base.BaseFix):
    BM_compatible: bool = True
    keep_line_order: bool = True
    mapping: dict[str, str] = MAPPING
    run_order: int = 6

    def build_pattern(self) -> str:
        return '|'.join(build_pattern(self.mapping))

    def compile_pattern(self) -> None:
        self.PATTERN = self.build_pattern()
        super(FixImports, self).compile_pattern()

    def match(self, node) -> bool:
        match = super(FixImports, self).match
        results = match(node)
        if results:
            if (('bare_with_attr' not in results) and any((match(obj) for obj in attr_chain(node, 'parent')))):
                return False
            return results
        return False

    def start_tree(self, tree, filename: str) -> None:
        super(FixImports, self).start_tree(tree, filename)
        self.replace: dict[str, str] = {}

    def transform(self, node, results) -> None:
        import_mod = results.get('module_name')
        if import_mod:
            mod_name = import_mod.value
            new_name = self.mapping[mod_name]
            import_mod.replace(Name(new_name, prefix=import_mod.prefix))
            if 'name_import' in results:
                self.replace[mod_name] = new_name
            if 'multiple_imports' in results:
                results = self.match(node)
                if results:
                    self.transform(node, results)
        else:
            bare_name = results['bare_with_attr'][0]
            new_name = self.replace.get(bare_name.value)
            if new_name:
                bare_name.replace(Name(new_name, prefix=bare_name.prefix))
