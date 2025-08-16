from typing import Any, Callable, List, Mapping, Optional
from snorkel.preprocess import BasePreprocessor
from snorkel.types import DataPoint

class LabelingFunction:
    def __init__(self, name: str, f: Callable, resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[BasePreprocessor]] = None) -> None:
    
    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
    
    def __call__(self, x: DataPoint) -> int:
    
    def __repr__(self) -> str:

class labeling_function:
    def __init__(self, name: Optional[str] = None, resources: Optional[Mapping[str, Any]] = None, pre: Optional[List[BasePreprocessor]] = None) -> None:
    
    def __call__(self, f: Callable) -> LabelingFunction:
