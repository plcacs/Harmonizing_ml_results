from typing import Union, Type, Callable, Sequence, overload

class AbstractRouteDef(abc.ABC):

    @abc.abstractmethod
    def register(self, router: Any) -> None:
        """Register itself into the given router."""

_HandlerType = Union[Type[AbstractView], Handler]

class RouteDef(AbstractRouteDef):

    def __repr__(self) -> str:
        ...

    def register(self, router: Any) -> List[Any]:
        ...

class StaticDef(AbstractRouteDef):

    def __repr__(self) -> str:
        ...

    def register(self, router: Any) -> List[Any]:
        ...

def route(method: str, path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def head(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def options(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def get(path: str, handler: _HandlerType, *, name: Optional[str] = None, allow_head: bool = True, **kwargs: Any) -> RouteDef:
    ...

def post(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def put(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def patch(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def delete(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def view(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    ...

def static(prefix: str, path: str, **kwargs: Any) -> StaticDef:
    ...

_Deco = Callable[[_HandlerType], _HandlerType]

class RouteTableDef(Sequence[AbstractRouteDef]):

    def __init__(self):
        ...

    def __repr__(self) -> str:
        ...

    @overload
    def __getitem__(self, index: int) -> AbstractRouteDef:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AbstractRouteDef]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[AbstractRouteDef, Sequence[AbstractRouteDef]]:
        ...

    def __iter__(self) -> Iterator[AbstractRouteDef]:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, item: AbstractRouteDef) -> bool:
        ...

    def route(self, method: str, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:

        def inner(handler: _HandlerType) -> _HandlerType:
            ...
        return inner

    def head(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def get(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def post(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def put(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def patch(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def delete(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def options(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def view(self, path: str, **kwargs: Any) -> Callable[[_HandlerType], _HandlerType]:
        ...

    def static(self, prefix: str, path: str, **kwargs: Any) -> None:
        ...
