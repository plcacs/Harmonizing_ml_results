import io
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pydantic
from prefect.utilities.annotations import BaseAnnotation, quote
from prefect.utilities.collections import AutoEnum, StopVisiting, deep_merge, deep_merge_dicts, dict_to_flatdict, flatdict_to_dict, get_from_dict, isiterable, remove_nested_keys, set_in_dict, visit_collection

class ExampleAnnotation(BaseAnnotation):
    pass

class Color(AutoEnum):
    RED = AutoEnum.auto()
    BLUE = AutoEnum.auto()

class TestAutoEnum:
    # ...

class TestVisitCollection:
    # ...

class TestRemoveKeys:
    # ...

class TestIsIterable:
    # ...

class TestGetFromDict:
    # ...

class TestSetInDict:
    # ...

class TestDeepMerge:
    # ...

class TestDeepMergeDicts:
    # ...
