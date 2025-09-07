import requests_html
from typing import Dict, List, Tuple, Any, Optional

project: str = 'requests-HTML'
copyright: str = u'MMXVIII. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> Project'
author: str = 'Kenneth Reitz'
version: str = ''
release: str = 'v0.3.4'
extensions: List[str] = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages']
templates_path: List[str] = ['_templates']
source_suffix: str = '.rst'
master_doc: str = 'index'
language: Optional[str] = None
exclude_patterns: List[str] = []
pygments_style: str = 'sphinx'
html_theme: str = 'alabaster'
html_theme_options: Dict[str, Any] = {'show_powered_by': False, 'github_user': 'psf', 'github_repo': 'requests-html', 'github_banner': True, 'show_related': False, 'note_bg': '#FFF59C'}
html_static_path: List[str] = ['_static']
html_sidebars: Dict[str, List[str]] = {'index': ['sidebarintro.html', 'sourcelink.html', 'searchbox.html', 'hacks.html'], '**': ['sidebarlogo.html', 'localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html', 'hacks.html']}
html_show_sphinx: bool = False
html_show_sourcelink: bool = False
htmlhelp_basename: str = 'requests-htmldoc'
latex_elements: Dict[str, Any] = {}
latex_documents: List[Tuple[str, str, str, str, str]] = [(master_doc, 'requests-html.tex', 'requests-html Documentation', 'Kenneth Reitz', 'manual')]
man_pages: List[Tuple[str, str, str, List[str], int]] = [(master_doc, 'requests-html', 'requests-html Documentation', [author], 1)]
texinfo_documents: List[Tuple[str, str, str, str, str, str, str]] = [(master_doc, 'requests-html', 'requests-html Documentation', author, 'requests-html', 'One line description of project.', 'Miscellaneous')]
epub_title: str = project
epub_author: str = author
epub_publisher: str = author
epub_copyright: str = copyright
epub_exclude_files: List[str] = ['search.html']
intersphinx_mapping: Dict[str, Optional[Any]] = {'https://docs.python.org/': None}
todo_include_todos: bool = True
