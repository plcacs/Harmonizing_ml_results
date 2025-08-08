from mimesis.locales import Locale
from mimesis.enums import Gender
from mimesis.providers.base import BaseDataProvider, BaseProvider
from mimesis.exceptions import LocaleError, NonEnumerableError
from pathlib import Path
import pytest
import tempfile
import json
import re
