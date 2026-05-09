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
sys.path.extend([os.path.join(os.path.dirname(__file__), '..', '../..', 'sphinxext')])

# ... (rest of the code remains the same)
