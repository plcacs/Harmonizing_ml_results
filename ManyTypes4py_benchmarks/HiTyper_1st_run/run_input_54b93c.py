"""
This module contains functions that allow sending type-checked `RunInput` data
to flows at runtime. Flows can send back responses, establishing two-way
channels with senders. These functions are particularly useful for systems that
require ongoing data transfer or need to react to input quickly.
real-time interaction and efficient data handling. It's designed to facilitate
dynamic communication within distributed or microservices-oriented systems,
making it ideal for scenarios requiring continuous data synchronization and
processing. It's particularly useful for systems that require ongoing data
input and output.

The following is an example of two flows. One sends a random number to the
other and waits for a response. The other receives the number, squares it, and
sends the result back. The sender flow then prints the result.

Sender flow:

```python
import random
from uuid import UUID
from prefect import flow
from prefect.logging import get_run_logger
from prefect.input import RunInput

class NumberData(RunInput):
    number: int


@flow
async def sender_flow(receiver_flow_run_id: UUID):
    logger = get_run_logger()

    the_number = random.randint(1, 100)

    await NumberData(number=the_number).send_to(receiver_flow_run_id)

    receiver = NumberData.receive(flow_run_id=receiver_flow_run_id)
    squared = await receiver.next()

    logger.info(f"{the_number} squared is {squared.number}")
```

Receiver flow:
```python
import random
from uuid import UUID
from prefect import flow
from prefect.logging import get_run_logger
from prefect.input import RunInput

class NumberData(RunInput):
    number: int


@flow
async def receiver_flow():
    async for data in NumberData.receive():
        squared = data.number ** 2
        data.respond(NumberData(number=squared))
```
"""
from __future__ import annotations
import inspect
from inspect import isclass
from typing import TYPE_CHECKING, Any, ClassVar, Coroutine, Dict, Generic, Literal, Optional, Set, Type, TypeVar, Union, cast, overload
from uuid import UUID, uuid4
import anyio
import pydantic
from pydantic import ConfigDict
from typing_extensions import Self
from prefect.input.actions import create_flow_run_input, create_flow_run_input_from_model, ensure_flow_run_id, filter_flow_run_input, read_flow_run_input
from prefect.utilities.asyncutils import sync_compatible
if TYPE_CHECKING:
    from prefect.client.schemas.objects import FlowRunInput
    from prefect.states import State
from prefect._internal.pydantic.v2_schema import create_v2_schema, is_v2_model
R = TypeVar('R', bound='RunInput')
T = TypeVar('T', bound='object')
Keyset = Dict[Union[Literal['description'], Literal['response'], Literal['schema']], str]

def keyset_from_paused_state(state: Union[dict, str, models.characters.states.State]) -> Union[str, typing.Iterable[str]]:
    """
    Get the keyset for the given Paused state.

    Args:
        - state (State): the state to get the keyset for
    """
    if not state.is_paused():
        raise RuntimeError(f'{state.type.value!r} is unsupported.')
    state_name = state.name or ''
    base_key = f'{state_name.lower()}-{str(state.state_details.pause_key)}'
    return keyset_from_base_key(base_key)

def keyset_from_base_key(base_key: Union[str, dict, T]) -> dict[typing.Text, typing.Text]:
    """
    Get the keyset for the given base key.

    Args:
        - base_key (str): the base key to get the keyset for

    Returns:
        - Dict[str, str]: the keyset
    """
    return {'description': f'{base_key}-description', 'response': f'{base_key}-response', 'schema': f'{base_key}-schema'}

class RunInputMetadata(pydantic.BaseModel):
    sender = None

class BaseRunInput(pydantic.BaseModel):
    model_config = ConfigDict(extra='forbid')
    _description = pydantic.PrivateAttr(default=None)
    _metadata = pydantic.PrivateAttr()

    @property
    def metadata(self):
        return self._metadata

    @classmethod
    def keyset_from_type(cls: Union[typing.Type, str]) -> Union[str, bytes]:
        return keyset_from_base_key(cls.__name__.lower())

    @classmethod
    @sync_compatible
    async def save(cls, keyset, flow_run_id=None):
        """
        Save the run input response to the given key.

        Args:
            - keyset (Keyset): the keyset to save the input for
            - flow_run_id (UUID, optional): the flow run ID to save the input for
        """
        if is_v2_model(cls):
            schema = create_v2_schema(cls.__name__, model_base=cls)
        else:
            schema = cls.model_json_schema(by_alias=True)
        coro = create_flow_run_input(key=keyset['schema'], value=schema, flow_run_id=flow_run_id)
        if TYPE_CHECKING:
            assert inspect.iscoroutine(coro)
        await coro
        description = cls._description if isinstance(cls._description, str) else None
        if description:
            coro = create_flow_run_input(key=keyset['description'], value=description, flow_run_id=flow_run_id)
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            await coro

    @classmethod
    @sync_compatible
    async def load(cls, keyset, flow_run_id=None):
        """
        Load the run input response from the given key.

        Args:
            - keyset (Keyset): the keyset to load the input for
            - flow_run_id (UUID, optional): the flow run ID to load the input for
        """
        flow_run_id = ensure_flow_run_id(flow_run_id)
        value = await read_flow_run_input(keyset['response'], flow_run_id=flow_run_id)
        if value:
            instance = cls(**value)
        else:
            instance = cls()
        instance._metadata = RunInputMetadata(key=keyset['response'], sender=None, receiver=flow_run_id)
        return instance

    @classmethod
    def load_from_flow_run_input(cls: Union[Machine, rl_algorithms.utils.config.ConfigDict], flow_run_input: Union[dict[str, set[str]], dict]):
        """
        Load the run input from a FlowRunInput object.

        Args:
            - flow_run_input (FlowRunInput): the flow run input to load the input for
        """
        instance = cls(**flow_run_input.decoded_value)
        instance._metadata = RunInputMetadata(key=flow_run_input.key, sender=flow_run_input.sender, receiver=flow_run_input.flow_run_id)
        return instance

    @classmethod
    def with_initial_data(cls: Union[str, int, list], description: Union[None, str]=None, **kwargs):
        """
        Create a new `RunInput` subclass with the given initial data as field
        defaults.

        Args:
            - description (str, optional): a description to show when resuming
                a flow run that requires input
            - kwargs (Any): the initial data to populate the subclass
        """
        fields = {}
        for key, value in kwargs.items():
            fields[key] = (type(value), value)
        model = pydantic.create_model(cls.__name__, **fields, __base__=cls)
        if description is not None:
            model._description = description
        return model

    @sync_compatible
    async def respond(self, run_input, sender=None, key_prefix=None):
        flow_run_id = None
        if self.metadata.sender and self.metadata.sender.startswith('prefect.flow-run'):
            _, _, id = self.metadata.sender.rpartition('.')
            flow_run_id = UUID(id)
        if not flow_run_id:
            raise RuntimeError('Cannot respond to an input that was not sent by a flow run.')
        await _send_input(flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix)

    @sync_compatible
    async def send_to(self, flow_run_id, sender=None, key_prefix=None):
        await _send_input(flow_run_id=flow_run_id, run_input=self, sender=sender, key_prefix=key_prefix)

class RunInput(BaseRunInput):

    @classmethod
    def receive(cls: Union[int, None, float, str], timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, int, float]=None, key_prefix: Union[None, int, str, typing.Mapping]=None, flow_run_id: Union[None, int, float]=None) -> GetInputHandler:
        if key_prefix is None:
            key_prefix = f'{cls.__name__.lower()}-auto'
        return GetInputHandler(run_input_cls=cls, key_prefix=key_prefix, timeout=timeout, poll_interval=poll_interval, raise_timeout_error=raise_timeout_error, exclude_keys=exclude_keys, flow_run_id=flow_run_id)

    @classmethod
    def subclass_from_base_model_type(cls: Union[typing.Type, typing.Iterable[str], None], model_cls: str) -> typing.Type:
        """
        Create a new `RunInput` subclass from the given `pydantic.BaseModel`
        subclass.

        Args:
            - model_cls (pydantic.BaseModel subclass): the class from which
                to create the new `RunInput` subclass
        """
        return type(f'{model_cls.__name__}RunInput', (RunInput, model_cls), {})

class AutomaticRunInput(BaseRunInput, Generic[T]):

    @classmethod
    @sync_compatible
    async def load(cls, keyset, flow_run_id=None):
        """
        Load the run input response from the given key.

        Args:
            - keyset (Keyset): the keyset to load the input for
            - flow_run_id (UUID, optional): the flow run ID to load the input for
        """
        instance_coro = super().load(keyset, flow_run_id=flow_run_id)
        if TYPE_CHECKING:
            assert inspect.iscoroutine(instance_coro)
        instance = await instance_coro
        return instance.value

    @classmethod
    def subclass_from_type(cls: Union[str, typing.Iterable[str]], _type: Union[str, bytes]):
        """
        Create a new `AutomaticRunInput` subclass from the given type.

        This method uses the type's name as a key prefix to identify related
        flow run inputs. This helps in ensuring that values saved under a type
        (like List[int]) are retrievable under the generic type name (like "list").
        """
        fields = {'value': (_type, ...)}
        type_prefix = getattr(_type, '__name__', getattr(_type, '_name', '')).lower()
        class_name = f'{type_prefix}AutomaticRunInput'
        new_cls = pydantic.create_model(class_name, **fields, __base__=AutomaticRunInput)
        return new_cls

    @classmethod
    def receive(cls: Union[int, None, float, str], timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, int, float]=None, key_prefix: Union[None, int, str, typing.Mapping]=None, flow_run_id: Union[None, int, float]=None, with_metadata=False) -> GetInputHandler:
        key_prefix = key_prefix or f'{cls.__name__.lower()}-auto'
        return GetAutomaticInputHandler(run_input_cls=cls, key_prefix=key_prefix, timeout=timeout, poll_interval=poll_interval, raise_timeout_error=raise_timeout_error, exclude_keys=exclude_keys, flow_run_id=flow_run_id, with_metadata=with_metadata)

def run_input_subclass_from_type(_type: typing.Type) -> Union[bool, typing.Type, None]:
    """
    Create a new `RunInput` subclass from the given type.
    """
    if isclass(_type):
        if issubclass(_type, RunInput):
            return cast(Type[R], _type)
        elif issubclass(_type, pydantic.BaseModel):
            return cast(Type[R], RunInput.subclass_from_base_model_type(_type))
    return cast(Type[AutomaticRunInput[T]], AutomaticRunInput.subclass_from_type(cast(Type[T], _type)))

class GetInputHandler(Generic[R]):

    def __init__(self, run_input_cls: Union[list[str], None, str, int], key_prefix: str, timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, str, typing.Iterable, typing.Type]=None, flow_run_id: Union[None, str, dict[str, typing.Any], list[int]]=None) -> None:
        self.run_input_cls = run_input_cls
        self.key_prefix = key_prefix
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.exclude_keys = set()
        self.raise_timeout_error = raise_timeout_error
        self.flow_run_id = ensure_flow_run_id(flow_run_id)
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> GetInputHandler:
        return self

    def __next__(self) -> int:
        try:
            return cast(R, self.next())
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> GetInputHandler:
        return self

    async def __anext__(self):
        try:
            coro = self.next()
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            return await coro
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self):
        flow_run_inputs_coro = filter_flow_run_input(key_prefix=self.key_prefix, limit=1, exclude_keys=self.exclude_keys, flow_run_id=self.flow_run_id)
        if TYPE_CHECKING:
            assert inspect.iscoroutine(flow_run_inputs_coro)
        flow_run_inputs = await flow_run_inputs_coro
        if flow_run_inputs:
            self.exclude_keys.add(*[i.key for i in flow_run_inputs])
        return flow_run_inputs

    def to_instance(self, flow_run_input: Union[dict[str, typing.Any], str, typing.MutableMapping]) -> Union[str, list["DictDataLoader"], dict]:
        return self.run_input_cls.load_from_flow_run_input(flow_run_input)

    @sync_compatible
    async def next(self):
        flow_run_inputs = await self.filter_for_inputs()
        if flow_run_inputs:
            return self.to_instance(flow_run_inputs[0])
        with anyio.fail_after(self.timeout):
            while True:
                await anyio.sleep(self.poll_interval)
                flow_run_inputs = await self.filter_for_inputs()
                if flow_run_inputs:
                    return self.to_instance(flow_run_inputs[0])

class GetAutomaticInputHandler(Generic[T]):

    def __init__(self, run_input_cls: Union[list[str], None, str, int], key_prefix: str, timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, str, typing.Iterable, typing.Type]=None, flow_run_id: Union[None, str, dict[str, typing.Any], list[int]]=None, with_metadata=False) -> None:
        self.run_input_cls = run_input_cls
        self.key_prefix = key_prefix
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.exclude_keys = set()
        self.raise_timeout_error = raise_timeout_error
        self.flow_run_id = ensure_flow_run_id(flow_run_id)
        self.with_metadata = with_metadata
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> GetInputHandler:
        return self

    def __next__(self) -> int:
        try:
            not_coro = self.next()
            if TYPE_CHECKING:
                assert not isinstance(not_coro, Coroutine)
            return not_coro
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> GetInputHandler:
        return self

    async def __anext__(self):
        try:
            coro = self.next()
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            return cast(Union[T, AutomaticRunInput[T]], await coro)
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self):
        flow_run_inputs_coro = filter_flow_run_input(key_prefix=self.key_prefix, limit=1, exclude_keys=self.exclude_keys, flow_run_id=self.flow_run_id)
        if TYPE_CHECKING:
            assert inspect.iscoroutine(flow_run_inputs_coro)
        flow_run_inputs = await flow_run_inputs_coro
        if flow_run_inputs:
            self.exclude_keys.add(*[i.key for i in flow_run_inputs])
        return flow_run_inputs

    @sync_compatible
    async def next(self):
        flow_run_inputs = await self.filter_for_inputs()
        if flow_run_inputs:
            return self.to_instance(flow_run_inputs[0])
        with anyio.fail_after(self.timeout):
            while True:
                await anyio.sleep(self.poll_interval)
                flow_run_inputs = await self.filter_for_inputs()
                if flow_run_inputs:
                    return self.to_instance(flow_run_inputs[0])

    def to_instance(self, flow_run_input: Union[dict[str, typing.Any], str, typing.MutableMapping]) -> Union[str, list["DictDataLoader"], dict]:
        run_input = self.run_input_cls.load_from_flow_run_input(flow_run_input)
        if self.with_metadata:
            return run_input
        return run_input.value

async def _send_input(flow_run_id, run_input, sender=None, key_prefix=None):
    if isinstance(run_input, RunInput):
        _run_input = run_input
    else:
        input_cls = run_input_subclass_from_type(type(run_input))
        _run_input = input_cls(value=run_input)
    if key_prefix is None:
        key_prefix = f'{_run_input.__class__.__name__.lower()}-auto'
    key = f'{key_prefix}-{uuid4()}'
    coro = create_flow_run_input_from_model(key=key, flow_run_id=flow_run_id, model_instance=_run_input, sender=sender)
    if TYPE_CHECKING:
        assert inspect.iscoroutine(coro)
    await coro

@sync_compatible
async def send_input(run_input, flow_run_id, sender=None, key_prefix=None):
    await _send_input(flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix)

@overload
def receive_input(input_type: Union[dict, bool, typing.Any, None], timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, float, int, str]=None, key_prefix: Union[None, float, int, str]=None, flow_run_id: Union[None, float, int, str]=None, with_metadata: bool=False) -> None:
    ...

@overload
def receive_input(input_type: Union[dict, bool, typing.Any, None], timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, float, int, str]=None, key_prefix: Union[None, float, int, str]=None, flow_run_id: Union[None, float, int, str]=None, with_metadata: bool=False) -> None:
    ...

def receive_input(input_type: Union[dict, bool, typing.Any, None], timeout: int=3600, poll_interval: int=10, raise_timeout_error: bool=False, exclude_keys: Union[None, float, int, str]=None, key_prefix: Union[None, float, int, str]=None, flow_run_id: Union[None, float, int, str]=None, with_metadata: bool=False) -> None:
    input_cls = run_input_subclass_from_type(input_type)
    if issubclass(input_cls, AutomaticRunInput):
        return input_cls.receive(timeout=timeout, poll_interval=poll_interval, raise_timeout_error=raise_timeout_error, exclude_keys=exclude_keys, key_prefix=key_prefix, flow_run_id=flow_run_id, with_metadata=with_metadata)
    else:
        return input_cls.receive(timeout=timeout, poll_interval=poll_interval, raise_timeout_error=raise_timeout_error, exclude_keys=exclude_keys, key_prefix=key_prefix, flow_run_id=flow_run_id)