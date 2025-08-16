from typing import Any, Callable, TypeVar

T = TypeVar('T')

class Shrinker:
    def __init__(self, initial: T, predicate: Callable[[T], bool], *, full: bool = False, debug: bool = False, name: str = None, **kwargs: Any) -> None:
    
    def setup(self, **kwargs: Any) -> None:
    
    def delegate(self, other_class: Type['Shrinker'], convert_to: Callable[[T], Any], convert_from: Callable[[Any], T], **kwargs: Any) -> None:
    
    def call_shrinker(self, other_class: Type['Shrinker'], initial: T, predicate: Callable[[T], bool], **kwargs: Any) -> T:
    
    def debug(self, *args: Any) -> None:
    
    @classmethod
    def shrink(cls, initial: T, predicate: Callable[[T], bool], **kwargs: Any) -> T:
    
    def run(self) -> None:
    
    def incorporate(self, value: T) -> bool:
    
    def consider(self, value: T) -> bool:
    
    def make_immutable(self, value: T) -> T:
    
    def check_invariants(self, value: T) -> None:
    
    def short_circuit(self) -> bool:
    
    def left_is_better(self, left: T, right: T) -> bool:
    
    def run_step(self) -> None:
