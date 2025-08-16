from mimesis.enums import Gender, Locale
from mimesis.exceptions import LocaleError, NonEnumerableError
from mimesis.locales import Locale
from mimesis.providers import Code, Cryptographic, Internet, Person
from mimesis.providers.base import BaseDataProvider, BaseProvider
from mimesis.types import MissingSeed
from pathlib import Path
from typing import List, Dict, Any
import json
import pytest
import re
import tempfile

class TestBase:

    def base_data_provider(self) -> BaseDataProvider:
        return BaseDataProvider()

    def test_override_locale(self, locale: Locale, new_locale: Locale) -> None:
    
    def test_update_dataset(self, base_data_provider: BaseDataProvider, data: Dict[str, Any], keys_count: int, values_count: int) -> None:
    
    def test_update_dataset_raises_error(self, base_data_provider: BaseDataProvider, data: Any) -> None:
    
    def test_override_missing_locale_argument(self) -> None:
    
    def test_override_locale_independent(self, provider: BaseProvider) -> None:
    
    def test_load_datafile(self, locale: Locale, city: str) -> None:
    
    def test_load_datafile_raises(self, locale: Locale) -> None:
    
    def test_extract(self, base_data_provider: BaseDataProvider) -> None:
    
    def test_extract_missing_positional_arguments(self, base_data_provider: BaseDataProvider) -> None:
    
    def test_update_dict(self, base_data_provider: BaseDataProvider) -> None:
    
    def test_setup_locale(self, base_data_provider: BaseDataProvider, inp: Locale, out: str) -> None:
    
    def test_setup_locale_unsupported_locale(self) -> None:
    
    def test_str(self, base_data_provider: BaseDataProvider) -> None:
    
    def test_validate_enum(self, base_data_provider: BaseDataProvider, gender: Gender, excepted: Any) -> None:
    
    def test_get_current_locale(self, locale: Locale) -> None:
    
    def test_base_wrong_random_type(self) -> None:
    
    def test_read_global_file(self, base_data_provider: BaseDataProvider) -> None:
    
    def test_custom_data_provider(self) -> None:

class TestSeededBase:

    def _bases(self, seed: Any) -> Tuple[BaseDataProvider, BaseDataProvider]:
    
    def test_base_random(self, _bases: Tuple[BaseDataProvider, BaseDataProvider]) -> None:
    
    def test_per_instance_random(self, seed: Any) -> None:
    
    def test_has_seed_no_global(self, monkeypatch: Any, seed: Any, global_seed: Any) -> None:
    
    def test_has_seed_global(self, monkeypatch: Any, seed: Any) -> None:
