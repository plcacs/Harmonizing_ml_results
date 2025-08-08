from datetime import datetime
import importlib
import inspect
import logging
import os
import re
import sys
import warnings
import jinja2
from numpydoc.docscrape import NumpyDocString
from sphinx.ext.autosummary import _import_by_name
from sphinx.ext.autodoc import AttributeDocumenter, Documenter, MethodDocumenter
from sphinx.ext.autosummary import Autosummary

logger: logging.Logger = logging.getLogger(__name__)
sys.setrecursionlimit(5000)
sys.path.insert(0, os.path.abspath('../sphinxext'))
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '../..', 'sphinxext'])
extensions: list = ['contributors', 'IPython.sphinxext.ipython_directive', 'IPython.sphinxext.ipython_console_highlighting', 'matplotlib.sphinxext.plot_directive', 'numpydoc', 'sphinx_copybutton', 'sphinx_design', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.coverage', 'sphinx.ext.doctest', 'sphinx.ext.extlinks', 'sphinx.ext.ifconfig', 'sphinx.ext.intersphinx', 'sphinx.ext.linkcode', 'sphinx.ext.mathjax', 'sphinx.ext.todo', 'nbsphinx']
exclude_patterns: list = ['**.ipynb_checkpoints', '**/includes/**']
try:
    import nbconvert
except ImportError:
    logger.warning('nbconvert not installed. Skipping notebooks.')
    exclude_patterns.append('**/*.ipynb')
else:
    try:
        nbconvert.utils.pandoc.get_pandoc_version()
    except nbconvert.utils.pandoc.PandocMissing:
        logger.warning('Pandoc not installed. Skipping notebooks.')
        exclude_patterns.append('**/*.ipynb')
source_path: str = os.path.dirname(os.path.abspath(__file__))
pattern: str = os.environ.get('SPHINX_PATTERN')
single_doc: bool = pattern is not None and pattern not in ('-api', 'whatsnew')
include_api: bool = pattern is None or pattern == 'whatsnew'
if pattern:
    for dirname, dirs, fnames in os.walk(source_path):
        reldir: str = os.path.relpath(dirname, source_path)
        for fname in fnames:
            if os.path.splitext(fname)[-1] in ('.rst', '.ipynb'):
                rel_fname: str = os.path.relpath(os.path.join(dirname, fname), source_path)
                if rel_fname == 'index.rst' and os.path.abspath(dirname) == source_path:
                    continue
                if pattern == '-api' and reldir.startswith('reference'):
                    exclude_patterns.append(rel_fname)
                elif pattern == 'whatsnew' and (not reldir.startswith('reference')) and (reldir != 'whatsnew'):
                    exclude_patterns.append(rel_fname)
                elif single_doc and rel_fname != pattern:
                    exclude_patterns.append(rel_fname)
with open(os.path.join(source_path, 'index.rst.template'), encoding='utf-8') as f:
    t: jinja2.Template = jinja2.Template(f.read())
with open(os.path.join(source_path, 'index.rst'), 'w', encoding='utf-8') as f:
    f.write(t.render(include_api=include_api, single_doc=pattern if single_doc else None))
autosummary_generate: bool = True if include_api else ['index']
autodoc_typehints: str = 'none'
numpydoc_show_class_members: bool = False
numpydoc_show_inherited_class_members: bool = False
numpydoc_attributes_as_param_list: bool = False
plot_include_source: bool = True
plot_formats: list = [('png', 90)]
plot_html_show_formats: bool = False
plot_html_show_source_link: bool = False
plot_pre_code: str = 'import numpy as np\nimport pandas as pd'
nbsphinx_requirejs_path: str = ''
toggleprompt_offset_right: int = 35
templates_path: list = ['../_templates']
source_suffix: list = ['.rst']
source_encoding: str = 'utf-8'
master_doc: str = 'index'
project: str = 'pandas'
copyright: str = f'{datetime.now().year},'
import pandas
version: str = str(pandas.__version__)
release: str = version
language: str = 'en'
exclude_trees: list = []
pygments_style: str = 'sphinx'
html_theme: str = 'pydata_sphinx_theme'
if '.dev' in version:
    switcher_version: str = 'dev'
elif 'rc' in version:
    switcher_version: str = version.split('rc', maxsplit=1)[0] + ' (rc)'
else:
    switcher_version: str = '.'.join(version.split('.')[:2])
html_theme_options: dict = {'external_links': [], 'footer_start': ['pandas_footer', 'sphinx-version'], 'github_url': 'https://github.com/pandas-dev/pandas', 'analytics': {'plausible_analytics_domain': 'pandas.pydata.org', 'plausible_analytics_url': 'https://views.scientific-python.org/js/script.js'}, 'logo': {'image_dark': 'https://pandas.pydata.org/static/img/pandas_white.svg'}, 'navbar_align': 'left', 'navbar_end': ['version-switcher', 'theme-switcher', 'navbar-icon-links'], 'switcher': {'json_url': 'https://pandas.pydata.org/versions.json', 'version_match': switcher_version}, 'show_version_warning_banner': False, 'icon_links': [{'name': 'X', 'url': 'https://x.com/pandas_dev', 'icon': 'fa-brands fa-square-x-twitter'}, {'name': 'Mastodon', 'url': 'https://fosstodon.org/@pandas_dev', 'icon': 'fa-brands fa-mastodon'}]}
html_logo: str = '../../web/pandas/static/img/pandas.svg'
html_static_path: list = ['_static']
html_css_files: list = ['css/getting_started.css', 'css/pandas.css']
html_favicon: str = '../../web/pandas/static/img/favicon.ico'
moved_api_pages: list = [('pandas.core.common.isnull', 'pandas.isna'), ('pandas.core.common.notnull', 'pandas.notna'), ('pandas.core.reshape.get_dummies', 'pandas.get_dummies'), ('pandas.tools.merge.concat', 'pandas.concat'), ('pandas.tools.merge.merge', 'pandas.merge'), ('pandas.tools.pivot.pivot_table', 'pandas.pivot_table'), ('pandas.tseries.tools.to_datetime', 'pandas.to_datetime'), ('pandas.io.clipboard.read_clipboard', 'pandas.read_clipboard'), ('pandas.io.excel.ExcelFile.parse', 'pandas.ExcelFile.parse'), ('pandas.io.excel.read_excel', 'pandas.read_excel'), ('pandas.io.html.read_html', 'pandas.read_html'), ('pandas.io.json.read_json', 'pandas.read_json'), ('pandas.io.parsers.read_csv', 'pandas.read_csv'), ('pandas.io.parsers.read_fwf', 'pandas.read_fwf'), ('pandas.io.parsers.read_table', 'pandas.read_table'), ('pandas.io.pickle.read_pickle', 'pandas.read_pickle'), ('pandas.io.pytables.HDFStore.append', 'pandas.HDFStore.append'), ('pandas.io.pytables.HDFStore.get', 'pandas.HDFStore.get'), ('pandas.io.pytables.HDFStore.put', 'pandas.HDFStore.put'), ('pandas.io.pytables.HDFStore.select', 'pandas.HDFStore.select'), ('pandas.io.pytables.read_hdf', 'pandas.read_hdf'), ('pandas.io.sql.read_sql', 'pandas.read_sql'), ('pandas.io.sql.read_frame', 'pandas.read_frame'), ('pandas.io.sql.write_frame', 'pandas.write_frame'), ('pandas.io.stata.read_stata', 'pandas.read_stata')]
moved_classes: list = [('pandas.tseries.resample.Resampler', 'pandas.core.resample.Resampler'), ('pandas.formats.style.Styler', 'pandas.io.formats.style.Styler')]
for old, new in moved_classes:
    moved_api_pages.append((old, new))
    mod, classname = new.rsplit('.', 1)
    klass = getattr(importlib.import_module(mod), classname)
    methods: list = [x for x in dir(klass) if not x.startswith('_') or x in ('__iter__', '__array__')]
    moved_api_pages.extend(((f'{old}.{method}', f'{new}.{method}') for method in methods)
if include_api:
    html_additional_pages: dict = {'generated/' + page[0]: 'api_redirect.html' for page in moved_api_pages}
header: str = ".. currentmodule:: pandas\n\n.. ipython:: python\n   :suppress:\n\n   import numpy as np\n   import pandas as pd\n\n   np.random.seed(123456)\n   np.set_printoptions(precision=4, suppress=True)\n   pd.options.display.max_rows = 15\n\n   import os\n   os.chdir(r'{os.path.dirname(os.path.dirname(__file__))}')\n"
html_context: dict = {'redirects': dict(moved_api_pages), 'header': header}
html_use_modindex: bool = True
htmlhelp_basename: str = 'pandas'
nbsphinx_allow_errors: bool = True
latex_elements: dict = {}
latex_documents: list = [('index', 'pandas.tex', 'pandas: powerful Python data analysis toolkit', 'Wes McKinney and the pandas Development Team', 'manual')]
if include_api:
    intersphinx_mapping: dict = {'dateutil': ('https://dateutil.readthedocs.io/en/latest/', None), 'matplotlib': ('https://matplotlib.org/stable/', None), 'numpy': ('https://numpy.org/doc/stable/', None), 'python': ('https://docs.python.org/3/', None), 'scipy': ('https://docs.scipy.org/doc/scipy/', None), 'pyarrow': ('https://arrow.apache.org/docs/', None)}
extlinks: dict = {'issue': ('https://github.com/pandas-dev/pandas/issues/%s', 'GH %s')}
ipython_execlines: list = ['import numpy as np', 'import pandas as pd', 'pd.options.display.encoding="utf8"']

class AccessorDocumenter(MethodDocumenter):
    objtype: str = 'accessor'
    directivetype: str = 'method'
    priority: float = 0.6

    def format_signature(self) -> str:
        return ''

class AccessorLevelDocumenter(Documenter):
    def resolve_name(self, modname, parents, path, base) -> tuple:
        if modname is None:
            if path:
                mod_cls: str = path.rstrip('.')
            else:
                mod_cls: str = None
                mod_cls = self.env.temp_data.get('autodoc:class')
                if mod_cls is None:
                    mod_cls = self.env.temp_data.get('py:class')
                if mod_cls is None:
                    return (None, [])
            modname, _, accessor = mod_cls.rpartition('.')
            modname, _, cls = modname.rpartition('.')
            parents: list = [cls, accessor]
            if not modname:
                modname = self.env.temp_data.get('autodoc:module')
            if not modname:
                if sphinx.__version__ > '1.3':
                    modname = self.env.ref_context.get('py:module')
                else:
                    modname = self.env.temp_data.get('py:module')
        return (modname, parents + [base])

class AccessorAttributeDocumenter(AccessorLevelDocumenter, AttributeDocumenter):
    objtype: str = 'accessorattribute'
    directivetype: str = 'attribute'
    priority: float = 0.6

class AccessorMethodDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    objtype: str = 'accessormethod'
    directivetype: str = 'method'
    priority: float = 0.6

class AccessorCallableDocumenter(AccessorLevelDocumenter, MethodDocumenter):
    objtype: str = 'accessorcallable'
    directivetype: str = 'method'
    priority: float = 0.5

    def format_name(self) -> str:
        return MethodDocumenter.format_name(self).removesuffix('.__call__')

class PandasAutosummary(Autosummary):
    def _replace_pandas_items(self, display_name, sig, summary, real_name) -> tuple:
        if display_name == 'DataFrame.plot':
            sig = '([x, y, kind, ax, ....])'
            summary = 'DataFrame plotting accessor and method'
        elif display_name == 'Series.plot':
            sig = '([kind, ax, figsize, ....])'
            summary = 'Series plotting accessor and method'
        return (display_name, sig, summary, real_name)

    @staticmethod
    def _is_deprecated(real_name) -> bool:
        try:
            obj, parent, modname = _import_by_name(real_name)
        except ImportError:
            return False
        doc = NumpyDocString(obj.__doc__ or '')
        summary = ''.join(doc['Summary'] + doc['Extended Summary'])
        return '.. deprecated::' in summary

    def _add_deprecation_prefixes(self, items) -> list:
        for item in items:
            display_name, sig, summary, real_name = item
            if self._is_deprecated(real_name):
                summary = f'(DEPRECATED) {summary}'
            yield (display_name, sig, summary, real_name)

    def get_items(self, names) -> list:
        items = Autosummary.get_items(self, names)
        items = [self._replace_pandas_items(*item) for item in items]
        items = list(self._add_deprecation_prefixes(items))
        return items

def linkcode_resolve(domain, info) -> str:
    if domain != 'py':
        return None
    modname: str = info['module']
    fullname: str = info['fullname']
    submod = sys.modules.get(modname)
    if submod is None:
        return None
    obj = submod
    for part in fullname.split('.'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None
    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None
    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None
    if lineno:
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
    else:
        linespec = ''
    fn = os.path.relpath(fn, start=os.path.dirname(pandas.__file__))
    if '+' in pandas.__version__:
        return f'https://github.com/pandas-dev/pandas/blob/main/pandas/{fn}{linespec}'
    else:
        return f'https://github.com/pandas-dev/pandas/blob/v{pandas.__version__}/pandas/{fn}{linespec}'

def remove_flags_docstring(app, what, name, obj, options, lines) -> None:
    if what == 'attribute' and name.endswith('.flags'):
        del lines[:]

def process_class_docstrings(app, what, name, obj, options, lines) -> None:
    if what == 'class':
        joined: str = '\n'.join(lines)
        templates: list = ['.. rubric:: Attributes\n\n.. autosummary::\n   :toctree:\n\n   None\n', '.. rubric:: Methods\n\n.. autosummary::\n   :toctree:\n\n   None\n']
        for template in templates:
            if template in joined:
                joined = joined.replace(template, '')
        lines[:] = joined.split('\n')

_BUSINED_ALIASES: list = ['pandas.tseries.offsets.' + name for name in ['BDay', 'CDay', 'BMonthEnd', 'BMonthBegin', 'CBMonthEnd', 'CBMonthBegin']]

def process_business_alias_docstrings(app, what, name, obj, options, lines) -> None:
    if name in _BUSINED_ALIASES:
        lines[:] = []

suppress_warnings: list = ['app.add_directive']
if pattern:
    suppress_warnings.append('ref.ref')

def rstjinja(app, docname, source) -> None:
    if app.builder.format != 'html':
        return
    src: str = source[0]
    rendered: str = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered

def setup(app) -> None:
    app.connect('source-read', rstjinja)
    app.connect('autodoc-process-docstring', remove_flags_docstring)
    app.connect('autodoc-process-docstring', process_class_docstrings)
    app.connect('autodoc-process-docstring', process_business_alias_docstrings)
    app.add_autodocumenter(AccessorDocumenter)
    app.add_autodocumenter(AccessorAttributeDocumenter)
    app.add_autodocumenter(AccessorMethodDocumenter)
    app.add_autodocumenter(AccessorCallableDocumenter)
    app.add_directive('autosummary', PandasAutosummary)
linkcheck_ignore: list = ['^http://$', '^https://$', *[re.escape(link) for link in ['http://scatterci.github.io/pydata/pandas', 'http://specs.frictionlessdata.io/json-table-schema/', 'https://crates.io/crates/calamine', 'https://devguide.python.org/setup/#macos', 'https://en.wikipedia.org/wiki/Imputation_statistics', 'https://en.wikipedia.org/wiki/Imputation_(statistics', 'https://github.com/noatamir/pandas-dev', 'https://github.com/pandas-dev/pandas/blob/main/pandas/plotting/__init__.py#L1', 'https://github.com/pandas-dev/pandas/blob/v0.20.2/pandas/core/generic.py#L568', 'https://github.com/pandas-dev/pandas/blob/v0.20.2/pandas/core/frame.py#L1495', 'https://github.com/pandas-dev/pandas/issues/174151', 'https://gitpod.io/#https://github.com/USERNAME/pandas', 'https://manishamde.github.io/blog/2013/03/07/pandas-and-python-top-10/', 'https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.table', 'https://nipunbatra.github.io/blog/visualisation/2013/05/01/aggregation-timeseries.html', 'https://nbviewer.ipy