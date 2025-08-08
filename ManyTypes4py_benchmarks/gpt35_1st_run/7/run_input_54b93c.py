from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Coroutine, Dict, Generic, Literal, Optional, Set, Type, TypeVar, Union
from uuid import UUID
import anyio
import pydantic
from pydantic import ConfigDict
from typing_extensions import Self

R = TypeVar('R', bound='RunInput')
T = TypeVar('T', bound='object')
Keyset = Dict[Union[Literal['description'], Literal['response'], Literal['schema']], str]

def keyset_from_paused_state(state: State) -> Dict[str, str]:
    ...

def keyset_from_base_key(base_key: str) -> Dict[str, str]:
    ...

class RunInputMetadata(pydantic.BaseModel):
    sender: Optional[Any]

class BaseRunInput(pydantic.BaseModel):
    model_config: ConfigDict
    _description: Optional[str]
    _metadata: pydantic.PrivateAttr

    @property
    def metadata(self) -> RunInputMetadata:
        ...

    @classmethod
    def keyset_from_type(cls) -> Dict[str, str]:
        ...

    @classmethod
    async def save(cls, keyset: Keyset, flow_run_id: Optional[UUID] = None) -> None:
        ...

    @classmethod
    async def load(cls, keyset: Keyset, flow_run_id: Optional[UUID] = None) -> BaseRunInput:
        ...

    @classmethod
    def load_from_flow_run_input(cls, flow_run_input: FlowRunInput) -> BaseRunInput:
        ...

    @classmethod
    def with_initial_data(cls, description: Optional[str] = None, **kwargs: Any) -> Type[BaseRunInput]:
        ...

    async def respond(self, run_input: RunInput, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
        ...

    async def send_to(self, flow_run_id: UUID, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
        ...

class RunInput(BaseRunInput):
    ...

class AutomaticRunInput(BaseRunInput, Generic[T]):
    ...

def run_input_subclass_from_type(_type: Type[T]) -> Type[Union[RunInput, AutomaticRunInput[T]]]:
    ...

class GetInputHandler(Generic[R]):
    ...

class GetAutomaticInputHandler(Generic[T]):
    ...

async def _send_input(flow_run_id: UUID, run_input: Union[RunInput, AutomaticRunInput[T]], sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
    ...

async def send_input(run_input: Union[RunInput, AutomaticRunInput[T]], flow_run_id: UUID, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
    ...

def receive_input(input_type: Type[T], timeout: int = 3600, poll_interval: int = 10, raise_timeout_error: bool = False, exclude_keys: Optional[Set[str]] = None, key_prefix: Optional[str] = None, flow_run_id: Optional[UUID] = None, with_metadata: bool = False) -> Union[GetInputHandler[R], GetAutomaticInputHandler[T]]:
    ...
