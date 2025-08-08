from typing import Callable, Any

class Shrinker:
    def __init__(self, initial: Any, predicate: Callable[[Any], bool], *, full: bool = False, debug: bool = False, name: str = None, **kwargs: Any) -> None:
    
    def setup(self, **kwargs: Any) -> None:
    
    def delegate(self, other_class: type, convert_to: Callable[[Any], Any], convert_from: Callable[[Any], Any], **kwargs: Any) -> None:
    
    def call_shrinker(self, other_class: type, initial: Any, predicate: Callable[[Any], bool], **kwargs: Any) -> Any:
    
    def debug(self, *args: Any) -> None:
    
    @classmethod
    def shrink(cls, initial: Any, predicate: Callable[[Any], bool], **kwargs: Any) -> Any:
    
    def run(self) -> None:
    
    def incorporate(self, value: Any) -> bool:
    
    def consider(self, value: Any) -> bool:
    
    def make_immutable(self, value: Any) -> Any:
    
    def check_invariants(self, value: Any) -> None:
    
    def short_circuit(self) -> bool:
    
    def left_is_better(self, left: Any, right: Any) -> bool:
    
    def run_step(self) -> None:
