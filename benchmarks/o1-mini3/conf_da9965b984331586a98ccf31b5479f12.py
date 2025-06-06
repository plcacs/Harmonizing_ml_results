import os
import sys

sys.path.append(os.path.abspath('tools/extensions'))

extensions: list[str] = ['pyspecific', 'sphinx.ext.extlinks']
manpages_url: str = 'https://manpages.debian.org/{path}'
project: str = 'Python'
copyright: str = '2001, Python Software Foundation'
version: str = release: str = sys.version.split(' ', 1)[0]
rst_epilog: str = f'''
.. |python_version_literal| replace:: ``Python {version}``
.. |python_x_dot_y_literal| replace:: ``python{version}``
.. |usr_local_bin_python_x_dot_y_literal| replace:: ``/usr/local/bin/python{version}``
'''
today: str = ''
today_fmt: str = '%B %d, %Y'
highlight_language: str = 'python3'
needs_sphinx: str = '6.2.1'
toc_object_entries: bool = False
smartquotes_excludes: dict[str, list[str]] = {
    'languages': ['ja', 'fr', 'zh_TW', 'zh_CN'],
    'builders': ['man', 'text']
}
root_doc: str = 'contents'
extlinks: dict[str, tuple[str, str]] = {
    'cve': ('https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-%s', 'CVE-%s'),
    'cwe': ('https://cwe.mitre.org/data/definitions/%s.html', 'CWE-%s'),
    'pypi': ('https://pypi.org/project/%s/', '%s'),
    'source': ('https://github.com/python/cpython/tree/3.13/%s', '%s')
}
extlinks_detect_hardcoded_links: bool = True
