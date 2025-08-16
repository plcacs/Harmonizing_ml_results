from __future__ import annotations
import dataclasses
import sys
from typing import TYPE_CHECKING, Any, cast, dataclass_transform, Tuple, List

def _class_fields(cls, kw_only) -> List[Tuple[str, Any, Any]]:

@dataclass_transform(field_specifiers=(dataclasses.field, dataclasses.Field), frozen_default=True, kw_only_default=True)
class FrozenOrThawed(type):

    def _make_dataclass(cls, name, bases, kw_only):

    def __new__(mcs, name, bases, namespace, frozen_or_thawed=False, **kwargs) -> FrozenOrThawed:

    def __init__(cls, name, bases, namespace, **kwargs):
