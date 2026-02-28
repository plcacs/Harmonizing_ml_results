from __future__ import annotations
import importlib
import inspect
import os
import re
import sys
from inspect import getmembers, isclass, isfunction
from pathlib import Path
from click import secho, style
import kedro
from kedro import __version__ as release

project: str = 'kedro'
author: str = 'kedro'
version: str = re.match('^([0-9]+\\.[0-9]+).*', release).group(1)
extensions: list[str] = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'myst_parser',
    'notfound.extension',
    'sphinxcontrib.jquery',
    'sphinx.ext.intersphinx',
    'sphinx_last_updated_by_git',
    'sphinx_favicon',
    'sphinxcontrib.youtube'
]
autosummary_generate: bool = True
autosummary_generate_overwrite: bool = False
napoleon_include_init_with_doc: bool = True
templates_path: list[str] = ['_templates']
intersphinx_mapping: dict[str, tuple[str, None]] = {
    'kedro-viz': ('https://docs.kedro.org/projects/kedro-viz/en/v6.6.1/', None),
    'kedro-datasets': ('https://docs.kedro.org/projects/kedro-datasets/en/kedro-datasets-2.0.0/', None),
    'cpython': ('https://docs.python.org/3.9/', None),
    'ipython': ('https://ipython.readthedocs.io/en/8.21.0/', None),
    'mlflow': ('https://www.mlflow.org/docs/2.12.1/', None),
    'kedro-mlflow': ('https://kedro-mlflow.readthedocs.io/en/0.12.2/', None)
}
source_suffix: dict[str, str] = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc: str = 'index'
favicons: list[str] = ['https://kedro.org/images/favicon.ico']
language: str = 'en'
exclude_patterns: list[str] = [
    '**.ipynb_checkpoints',
    '_templates',
    'modules.rst',
    'source',
    'kedro_docs_style_guide.md'
]
type_targets: dict[str, tuple[str, ...]] = {
    'py:class': (
        'object',
        'bool',
        'int',
        'float',
        'str',
        'tuple',
        'Any',
        'Dict',
        'dict',
        'list',
        'set',
        'typing.Dict',
        'typing.Iterable',
        'typing.List',
        'typing.Tuple',
        'typing.Type',
        'typing.Set',
        'kedro.config.config.ConfigLoader',
        'kedro.io.catalog_config_resolver.CatalogConfigResolver',
        'kedro.io.core.AbstractDataset',
        'kedro.io.core.AbstractVersionedDataset',
        'kedro.io.core.CatalogProtocol',
        'kedro.io.core.DatasetError',
        'kedro.io.core.Version',
        'kedro.io.data_catalog.DataCatalog',
        'kedro.io.kedro_data_catalog.KedroDataCatalog',
        'kedro.io.memory_dataset.MemoryDataset',
        'kedro.io.partitioned_dataset.PartitionedDataset',
        'kedro.pipeline.pipeline.Pipeline',
        'kedro.runner.runner.AbstractRunner',
        'kedro.framework.context.context.KedroContext',
        'kedro.framework.startup.ProjectMetadata',
        'abc.ABC',
        'Path',
        'pathlib.Path',
        'PurePosixPath',
        'pathlib.PurePosixPath',
        'requests.auth.AuthBase',
        'google.oauth2.credentials.Credentials',
        'Exception',
        'CONF_SOURCE',
        'integer -- return number of occurrences of value',
        'integer -- return first index of value.',
        'kedro_datasets.pandas.json_dataset.JSONDataset',
        'pluggy._manager.PluginManager',
        'PluginManager',
        '_DI',
        '_DO',
        'deltalake.table.Metadata',
        'None.  Remove all items from D.',
        'a shallow copy of D',
        "a set-like object providing a view on D's items",
        "a set-like object providing a view on D's keys",
        'v, remove specified key and return the corresponding value.',
        'None.  Update D from dict/iterable E and F.',
        "an object providing a view on D's values",
        '(k, v), remove and return some (key, value) pair',
        'D.get(k,d), also set D[k]=d if k not in D',
        'D[k] if k in D, else d.  d defaults to None.',
        'None.  Update D from mapping/iterable E and F.',
        'Patterns',
        'CatalogConfigResolver',
        'CatalogProtocol',
        'KedroDataCatalog'
    ),
    'py:data': (
        'typing.Any',
        'typing.Callable',
        'typing.Union',
        'typing.Optional',
        'typing.Tuple'
    ),
    'py:exc': (
        'ValueError',
        'BadConfigException',
        'MissingConfigException',
        'DatasetError',
        'ImportError',
        'KedroCliError',
        'Exception',
        'TypeError',
        'SyntaxError',
        'CircularDependencyError',
        'OutputNotUniqueError',
        'ConfirmNotUniqueError',
        'ParserError'
    )
}
nitpick_ignore: list[tuple[str, str]] = [
    (key, value)
    for key in type_targets
    for value in type_targets[key]
]
pygments_style: str = 'sphinx'
html_theme: str = 'kedro-sphinx-theme'
here: Path = Path(__file__).parent.absolute()
html_theme_options: dict[str, bool] = {'collapse_navigation': False, 'style_external_links': True}
html_extra_path: list[str] = [str(here / 'extra_files')]
html_show_copyright: bool = False
linkcheck_ignore: list[str] = [
    'http://127.0.0.1:8787/status',
    'https://datacamp.com/community/tutorials/docstrings-python',
    'https://github.com/argoproj/argo/blob/master/README.md#quickstart',
    'https://console.aws.amazon.com/batch/home#/jobs',
    'https://github.com/EbookFoundation/free-programming-books/blob/master/books/free-programming-books-langs.md#python',
    'https://github.com/jazzband/pip-tools#example-usage-for-pip-compile',
    'https://www.astronomer.io/docs/cloud/stable/get-started/quickstart#',
    'https://eternallybored.org/misc/wget/',
    'https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.from_pandas',
    'https://www.oracle.com/java/technologies/javase-downloads.html',
    'https://www.java.com/en/download/help/download_options.html',
    'https://docs.delta.io/latest/delta-update.html#language-python',
    'https://github.com/kedro-org/kedro/blob/main/kedro/framework/project/default_logging.yml',
    'https://github.com/kedro-org/kedro/blob/main/kedro/framework/project/rich_logging.yml',
    'https://github.com/kedro-org/kedro/blob/main/README.md#the-humans-behind-kedro',
    'https://opensource.org/license/apache2-0-php/',
    'https://docs.github.com/en/rest/overview/other-authentication-methods#via-username-and-password',
    'https://www.educative.io/blog/advanced-yaml-syntax-cheatsheet#anchors',
    'https://www.quora.com/What-is-thread-safety-in-Python'
]
linkcheck_retries: int = 3
html_context: dict[str, bool] = {'display_github': True, 'github_url': 'https://github.com/kedro-org/kedro/tree/main/docs/source'}
html_show_sourcelink: bool = False
htmlhelp_basename: str = 'Kedrodoc'
latex_elements: dict[str, str] = {}
latex_documents: list[tuple[str, str, str, str, str]] = [
    (master_doc, 'Kedro.tex', 'Kedro Documentation', 'Kedro', 'manual')
]
man_pages: list[tuple[str, str, str, list[str], int]] = [
    (master_doc, 'kedro', 'Kedro Documentation', [author], 1)
]
texinfo_documents: list[tuple[str, str, str, str, str, str, str]] = [
    (master_doc, 'Kedro', 'Kedro Documentation', author, 'Kedro', 'Kedro is a Python framework for creating reproducible, maintainable and modular data science code.', 'Data-Science')
]
todo_include_todos: bool = False
KEDRO_MODULES: list[str] = ['kedro.io', 'kedro.pipeline', 'kedro.runner', 'kedro.config', 'kedro_datasets']

def get_classes(module: str) -> list[str]:
    """Get all classes from a module."""
    importlib.import_module(module)
    return [obj[0] for obj in getmembers(sys.modules[module], lambda obj: isclass(obj))]

def get_functions(module: str) -> list[str]:
    """Get all functions from a module."""
    importlib.import_module(module)
    return [obj[0] for obj in getmembers(sys.modules[module], lambda obj: isfunction(obj))]

def remove_arrows_in_examples(lines: list[str]) -> None:
    """Remove '>>>' from examples."""
    for i, line in enumerate(lines):
        lines[i] = line.replace('>>>', '')

def autolink_replacements(what: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """
    Create a list containing replacement tuples of the form:
    (``regex``, ``replacement``, ``obj``) for all classes and methods which are
    imported in ``KEDRO_MODULES`` ``__init__.py`` files. The ``replacement``
    is a reStructuredText link to their documentation.

    For example, if the docstring reads:
        This LambdaDataset loads and saves ...

    Then the word ``LambdaDataset``, will be replaced by
    :class:`~kedro.io.LambdaDataset`

    Works for plural as well, e.g:
        These ``LambdaDataset``s load and save

    Will convert to:
        These :class:`kedro.io.LambdaDataset` load and save

    Args:
        what: The objects to create replacement tuples for. Possible values
            ["class", "func"].

    Returns:
        A list of tuples: (regex, replacement, obj), for all "what" objects
        imported in __init__.py files of ``KEDRO_MODULES``.
    """
    replacements: list[tuple[str, str, str]] = []
    suggestions: list[tuple[str, str, str]] = []
    for module in KEDRO_MODULES:
        if what == 'class':
            objects = get_classes(module)
        elif what == 'func':
            objects = get_functions(module)
        if what == 'class':
            replacements += [(f'``{obj}``s', f':{what}:`~{module}.{obj}`\\\\s', obj) for obj in objects]
        replacements += [(f'``{obj}``', f':{what}:`~{module}.{obj}`', obj) for obj in objects]
        if what == 'class':
            suggestions += [(f'(?<!\\w|`){obj}s(?!\\w|`{{2}})', f'``{obj}``s', obj) for obj in objects]
        suggestions += [(f'(?<!\\w|`){obj}(?!\\w|`{{2}})', f'``{obj}``', obj) for obj in objects]
    return (replacements, suggestions)

def log_suggestions(lines: list[str], name: str) -> None:
    """Use the ``suggestions`` list to log in the terminal places where the
    developer has forgotten to surround with double back-ticks class
    name/function name references.

    Args:
        lines: The docstring lines.
        name: The name of the object whose docstring is contained in lines.
    """
    title_printed: bool = False
    for i in range(len(lines)):
        if '>>>' in lines[i]:
            continue
        for existing, replacement, obj in suggestions:
            new = re.sub(existing, f'{replacement}', lines[i])
            if new == lines[i]:
                continue
            if ':rtype:' in lines[i] or ':type ' in lines[i]:
                continue
            if not title_printed:
                secho('-' * 50 + '\n' + name + ':\n' + '-' * 50, fg='blue')
                title_printed = True
            print('[' + str(i) + '] ' + re.sub(existing, f'{style(obj, fg="magenta")}', lines[i]))
            print('[' + str(i) + '] ' + re.sub(existing, f'``{style(obj, fg="green")}``', lines[i]))
    if title_printed:
        print('\n')

def autolink_classes_and_methods(lines: list[str]) -> None:
    for i in range(len(lines)):
        if '>>>' in lines[i]:
            continue
        for existing, replacement, obj in replacements:
            lines[i] = re.sub(existing, f'{replacement}', lines[i])

def autodoc_process_docstring(app, what: str, name: str, obj, options, lines: list[str]) -> None:
    try:
        log_suggestions(lines, name)
        autolink_classes_and_methods(lines)
    except Exception as e:
        print(style(f'Failed to check for class name mentions that can be converted to reStructuredText links in docstring of {name}. Error is: \n{e!s}', fg='red'))
    remove_arrows_in_examples(lines)

def setup(app) -> None:
    app.connect('autodoc-process-docstring', autodoc_process_docstring)

replacements: list[tuple[str, str, str]] = []
suggestions: list[tuple[str, str, str]] = []
try:
    replacements_f, suggestions_f = autolink_replacements('func')
    replacements_c, suggestions_c = autolink_replacements('class')
    replacements = replacements_f + replacements_c
    suggestions = suggestions_f + suggestions_c
except Exception as e:
    print(style(f'Failed to create list of (regex, reStructuredText link replacement) for class names and method names in docstrings. Error is: \n{e!s}', fg='red'))

user_agent: str = 'Mozilla/5.0 (X11; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0'
myst_heading_anchors: int = 5
myst_enable_extensions: list[str] = ['colon_fence']

def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve a GitHub URL corresponding to a Python object."""
    if domain != 'py':
        return None
    try:
        mod = sys.modules[info['module']]
        obj = mod
        for attr in info['fullname'].split('.'):
            obj = getattr(obj, attr)
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        relpath = os.path.relpath(filename, start=os.path.dirname(kedro.__file__))
        return 'https://github.com/kedro-org/kedro/blob/main/kedro/%s#L%d#L%d' % (relpath, lineno, lineno + len(source) - 1)
    except (KeyError, ImportError, AttributeError, TypeError, OSError, ValueError):
        return None
