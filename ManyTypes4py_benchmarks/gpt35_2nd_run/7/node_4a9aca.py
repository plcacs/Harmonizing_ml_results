from __future__ import annotations
import copy
import inspect
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, List, Dict

class Node:
    def __init__(self, func: Callable, inputs: Any, outputs: Any, *, name: str = None, tags: Any = None, confirms: Any = None, namespace: str = None) -> None:
    def _copy(self, **overwrite_params: Any) -> Node:
    @property
    def _logger(self) -> logging.Logger:
    @property
    def _unique_key(self) -> tuple:
    def __eq__(self, other: Any) -> bool:
    def __lt__(self, other: Any) -> bool:
    def __hash__(self) -> int:
    def __str__(self) -> str:
    def __repr__(self) -> str:
    def __call__(self, **kwargs: Any) -> Any:
    @property
    def _func_name(self) -> str:
    @property
    def func(self) -> Callable:
    @func.setter
    def func(self, func: Callable) -> None:
    @property
    def tags(self) -> set:
    def tag(self, tags: Any) -> Node:
    @property
    def name(self) -> str:
    @property
    def short_name(self) -> str:
    @property
    def namespace(self) -> str:
    @property
    def inputs(self) -> List[str]:
    @property
    def outputs(self) -> List[str]:
    @property
    def confirms(self) -> List[str]:
    def run(self, inputs: dict = None) -> dict:
    def _run_with_no_inputs(self, inputs: dict) -> Any:
    def _run_with_one_input(self, inputs: dict, node_input: str) -> Any:
    def _run_with_list(self, inputs: dict, node_inputs: List[str]) -> Any:
    def _run_with_dict(self, inputs: dict, node_inputs: Dict[str, str]) -> Any:
    def _outputs_to_dictionary(self, outputs: Any) -> dict:
    def _validate_inputs(self, func: Callable, inputs: Any) -> None:
    def _validate_unique_outputs(self) -> None:
    def _validate_inputs_dif_than_outputs(self) -> None:
    @staticmethod
    def _process_inputs_for_bind(inputs: Any) -> tuple:
    def _dict_inputs_to_list(func: Callable, inputs: Dict[str, str]) -> List[str]:
    def _to_list(element: Any) -> List[str]:
    def _get_readable_func_name(func: Callable) -> str

def _node_error_message(msg: str) -> str:

def node(func: Callable, inputs: Any, outputs: Any, *, name: str = None, tags: Any = None, confirms: Any = None, namespace: str = None) -> Node:
