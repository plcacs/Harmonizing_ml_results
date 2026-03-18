```python
import numpy as np
from typing import Any, List, Tuple

def make_sparse_index(
    length: int, 
    indices: np.ndarray, 
    kind: str
) -> Any: ...

class BlockIndex:
    def __init__(
        self, 
        length: int, 
        blocs: List[int], 
        blengths: List[int]
    ) -> None: ...
    
    @property
    def npoints(self) -> int: ...
    
    @property
    def blocs(self) -> np.ndarray: ...
    
    @property
    def blengths(self) -> np.ndarray: ...
    
    def make_union(self, other: Any) -> Any: ...
    
    def intersect(self, other: Any) -> Any: ...
    
    def equals(self, other: Any) -> bool: ...
    
    def to_int_index(self) -> Any: ...
    
    def to_block_index(self) -> Any: ...
    
    def lookup(self, idx: int) -> int: ...
    
    def lookup_array(self, arr: np.ndarray) -> np.ndarray: ...
    
    def check_integrity(self) -> None: ...

class IntIndex:
    def __init__(
        self, 
        length: int, 
        indices: np.ndarray
    ) -> None: ...
    
    @property
    def npoints(self) -> int: ...
    
    @property
    def indices(self) -> np.ndarray: ...
    
    def make_union(self, other: Any) -> Any: ...
    
    def intersect(self, other: Any) -> Any: ...
    
    def equals(self, other: Any) -> bool: ...
    
    def to_int_index(self) -> Any: ...
    
    def to_block_index(self) -> Any: ...
    
    def lookup(self, idx: int) -> int: ...
    
    def lookup_array(self, arr: np.ndarray) -> np.ndarray: ...
    
    def check_integrity(self) -> None: ...

def sparse_add_float64(
    x: np.ndarray, 
    xindex: Any, 
    xfill: float, 
    y: np.ndarray, 
    yindex: Any, 
    yfill: float
) -> Tuple[np.ndarray, Any, float]: ...

def sparse_sub_float64(
    x: np.ndarray, 
    xindex: Any, 
    xfill: float, 
    y: np.ndarray, 
    yindex: Any, 
    yfill: float
) -> Tuple[np.ndarray, Any, float]: ...

def sparse_mul_float64(
    x: np.ndarray, 
    xindex: Any, 
    xfill: float, 
    y: np.ndarray, 
    yindex: Any, 
    yfill: float
) -> Tuple[np.ndarray, Any, float]: ...

def sparse_truediv_float64(
    x: np.ndarray, 
    xindex: Any, 
    xfill: float, 
    y: np.ndarray, 
    yindex: Any, 
    yfill: float
) -> Tuple[np.ndarray, Any, float]: ...

def sparse_floordiv_float64(
    x: np.ndarray, 
    xindex: Any, 
    xfill: float, 
    y: np.ndarray, 
    yindex: Any, 
    yfill: float
) -> Tuple[np.ndarray, Any, float]: ...
```