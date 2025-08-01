from __future__ import annotations
import inspect
from inspect import isclass
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID, uuid4
import anyio
import pydantic
from pydantic import ConfigDict
from typing_extensions import Self
from prefect.input.actions import (
    create_flow_run_input,
    create_flow_run_input_from_model,
    ensure_flow_run_id,
    filter_flow_run_input,
    read_flow_run_input,
)
from prefect.utilities.asyncutils import sync_compatible
from prefect._internal.pydantic.v2_schema import create_v2_schema, is_v2_model

if False:
    from prefect.client.schemas.objects import FlowRunInput  # type: ignore
    from prefect.states import State  # type: ignore

R = TypeVar("R", bound="RunInput")
T = TypeVar("T", bound=object)
Keyset = Dict[Union[Literal["description"], Literal["response"], Literal["schema"]], str]


def keyset_from_paused_state(state: Any) -> Keyset:
    """
    Get the keyset for the given Paused state.

    Args:
        - state (State): the state to get the keyset for
    """
    if not state.is_paused():
        raise RuntimeError(f"{state.type.value!r} is unsupported.")
    state_name: str = state.name or ""
    base_key: str = f"{state_name.lower()}-{str(state.state_details.pause_key)}"
    return keyset_from_base_key(base_key)


def keyset_from_base_key(base_key: str) -> Keyset:
    """
    Get the keyset for the given base key.

    Args:
        - base_key (str): the base key to get the keyset for

    Returns:
        - Dict[str, str]: the keyset
    """
    return {
        "description": f"{base_key}-description",
        "response": f"{base_key}-response",
        "schema": f"{base_key}-schema",
    }


class RunInputMetadata(pydantic.BaseModel):
    sender: Optional[Any] = None


class BaseRunInput(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid")
    _description: Optional[str] = pydantic.PrivateAttr(default=None)
    _metadata: RunInputMetadata = pydantic.PrivateAttr()

    @property
    def metadata(self) -> RunInputMetadata:
        return self._metadata

    @classmethod
    def keyset_from_type(cls) -> Keyset:
        return keyset_from_base_key(cls.__name__.lower())

    @classmethod
    @sync_compatible
    async def save(cls: Type[Self], keyset: Keyset, flow_run_id: Optional[UUID] = None) -> None:
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
        coro: Coroutine[Any, Any, Any] = create_flow_run_input(key=keyset["schema"], value=schema, flow_run_id=flow_run_id)
        await coro
        description: Optional[str] = cls._description if isinstance(cls._description, str) else None
        if description:
            coro = create_flow_run_input(key=keyset["description"], value=description, flow_run_id=flow_run_id)
            await coro

    @classmethod
    @sync_compatible
    async def load(cls: Type[Self], keyset: Keyset, flow_run_id: Optional[UUID] = None) -> Self:
        """
        Load the run input response from the given key.

        Args:
            - keyset (Keyset): the keyset to load the input for
            - flow_run_id (UUID, optional): the flow run ID to load the input for
        """
        flow_run_id = ensure_flow_run_id(flow_run_id)
        value: Optional[Dict[str, Any]] = await read_flow_run_input(keyset["response"], flow_run_id=flow_run_id)
        if value:
            instance: Self = cls(**value)
        else:
            instance = cls()
        instance._metadata = RunInputMetadata(key=keyset["response"], sender=None, receiver=flow_run_id)
        return instance

    @classmethod
    def load_from_flow_run_input(cls: Type[Self], flow_run_input: Any) -> Self:
        """
        Load the run input from a FlowRunInput object.

        Args:
            - flow_run_input (FlowRunInput): the flow run input to load the input for
        """
        instance: Self = cls(**flow_run_input.decoded_value)
        instance._metadata = RunInputMetadata(
            key=flow_run_input.key, sender=flow_run_input.sender, receiver=flow_run_input.flow_run_id
        )
        return instance

    @classmethod
    def with_initial_data(cls: Type[Self], description: Optional[str] = None, **kwargs: Any) -> Type[Self]:
        """
        Create a new `RunInput` subclass with the given initial data as field
        defaults.

        Args:
            - description (str, optional): a description to show when resuming
                a flow run that requires input
            - kwargs (Any): the initial data to populate the subclass
        """
        fields: Dict[str, Any] = {}
        for key, value in kwargs.items():
            fields[key] = (type(value), value)
        model = pydantic.create_model(cls.__name__, **fields, __base__=cls)  # type: ignore
        if description is not None:
            model._description = description  # type: ignore
        return model

    @sync_compatible
    async def respond(self, run_input: Any, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
        flow_run_id: Optional[UUID] = None
        if self.metadata.sender and isinstance(self.metadata.sender, str) and self.metadata.sender.startswith("prefect.flow-run"):
            _, _, id_ = self.metadata.sender.rpartition(".")
            flow_run_id = UUID(id_)
        if not flow_run_id:
            raise RuntimeError("Cannot respond to an input that was not sent by a flow run.")
        await _send_input(flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix)

    @sync_compatible
    async def send_to(self, flow_run_id: UUID, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
        await _send_input(flow_run_id=flow_run_id, run_input=self, sender=sender, key_prefix=key_prefix)


class RunInput(BaseRunInput):
    @classmethod
    def receive(
        cls: Type[R],
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
        flow_run_id: Optional[UUID] = None,
    ) -> GetInputHandler[R]:
        if key_prefix is None:
            key_prefix = f"{cls.__name__.lower()}-auto"
        return GetInputHandler(run_input_cls=cls, key_prefix=key_prefix, timeout=timeout, poll_interval=poll_interval, raise_timeout_error=raise_timeout_error, exclude_keys=exclude_keys, flow_run_id=flow_run_id)

    @classmethod
    def subclass_from_base_model_type(cls: Type[RunInput], model_cls: Type[pydantic.BaseModel]) -> Type[RunInput]:
        """
        Create a new `RunInput` subclass from the given `pydantic.BaseModel`
        subclass.

        Args:
            - model_cls (pydantic.BaseModel subclass): the class from which
                to create the new `RunInput` subclass
        """
        return cast(Type[RunInput], type(f"{model_cls.__name__}RunInput", (RunInput, model_cls), {}))


class AutomaticRunInput(BaseRunInput, Generic[T]):
    @classmethod
    @sync_compatible
    async def load(cls: Type[AutomaticRunInput[T]], keyset: Keyset, flow_run_id: Optional[UUID] = None) -> T:
        """
        Load the run input response from the given key.

        Args:
            - keyset (Keyset): the keyset to load the input for
            - flow_run_id (UUID, optional): the flow run ID to load the input for
        """
        instance_coro: Coroutine[Any, Any, AutomaticRunInput[T]] = super().load(keyset, flow_run_id=flow_run_id)
        instance = await instance_coro
        return instance.value  # type: ignore

    @classmethod
    def subclass_from_type(cls: Type[AutomaticRunInput[T]], _type: Any) -> Type[AutomaticRunInput[T]]:
        """
        Create a new `AutomaticRunInput` subclass from the given type.

        This method uses the type's name as a key prefix to identify related
        flow run inputs. This helps in ensuring that values saved under a type
        (like List[int]) are retrievable under the generic type name (like "list").
        """
        fields: Dict[str, Any] = {"value": (_type, ...)}
        type_prefix: str = getattr(_type, "__name__", getattr(_type, "_name", "")).lower()
        class_name: str = f"{type_prefix}AutomaticRunInput"
        new_cls = pydantic.create_model(class_name, **fields, __base__=AutomaticRunInput)  # type: ignore
        return new_cls

    @classmethod
    def receive(
        cls: Type[AutomaticRunInput[T]],
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
        flow_run_id: Optional[UUID] = None,
        with_metadata: bool = False,
    ) -> GetAutomaticInputHandler[T]:
        key_prefix = key_prefix or f"{cls.__name__.lower()}-auto"
        return GetAutomaticInputHandler(
            run_input_cls=cls,
            key_prefix=key_prefix,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_timeout_error=raise_timeout_error,
            exclude_keys=exclude_keys,
            flow_run_id=flow_run_id,
            with_metadata=with_metadata,
        )


def run_input_subclass_from_type(_type: Any) -> Union[Type[RunInput], Type[AutomaticRunInput[T]]]:
    """
    Create a new `RunInput` subclass from the given type.
    """
    if isclass(_type):
        if issubclass(_type, RunInput):
            return cast(Type[RunInput], _type)
        elif issubclass(_type, pydantic.BaseModel):
            return cast(Type[RunInput], RunInput.subclass_from_base_model_type(_type))
    return cast(Type[AutomaticRunInput[T]], AutomaticRunInput.subclass_from_type(cast(Type[T], _type)))


class GetInputHandler(Generic[R]):
    def __init__(
        self,
        run_input_cls: Type[R],
        key_prefix: str,
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        flow_run_id: Optional[UUID] = None,
    ) -> None:
        self.run_input_cls: Type[R] = run_input_cls
        self.key_prefix: str = key_prefix
        self.timeout: int = timeout
        self.poll_interval: int = poll_interval
        self.exclude_keys: Set[str] = set()
        self.raise_timeout_error: bool = raise_timeout_error
        self.flow_run_id: UUID = ensure_flow_run_id(flow_run_id)
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> Iterator[R]:
        return self

    def __next__(self) -> R:
        try:
            return cast(R, self.next())  # type: ignore
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> AsyncIterator[R]:
        return self

    async def __anext__(self) -> R:
        try:
            coro: Coroutine[Any, Any, R] = self.next()  # type: ignore
            return await coro
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self) -> List[Any]:
        flow_run_inputs = await filter_flow_run_input(
            key_prefix=self.key_prefix,
            limit=1,
            exclude_keys=self.exclude_keys,
            flow_run_id=self.flow_run_id,
        )
        if flow_run_inputs:
            self.exclude_keys.update({i.key for i in flow_run_inputs})
        return flow_run_inputs

    def to_instance(self, flow_run_input: Any) -> R:
        return self.run_input_cls.load_from_flow_run_input(flow_run_input)

    @sync_compatible
    async def next(self) -> R:
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
    def __init__(
        self,
        run_input_cls: Type[AutomaticRunInput[T]],
        key_prefix: str,
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        flow_run_id: Optional[UUID] = None,
        with_metadata: bool = False,
    ) -> None:
        self.run_input_cls: Type[AutomaticRunInput[T]] = run_input_cls
        self.key_prefix: str = key_prefix
        self.timeout: int = timeout
        self.poll_interval: int = poll_interval
        self.exclude_keys: Set[str] = set()
        self.raise_timeout_error: bool = raise_timeout_error
        self.flow_run_id: UUID = ensure_flow_run_id(flow_run_id)
        self.with_metadata: bool = with_metadata
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> Iterator[Union[T, AutomaticRunInput[T]]]:
        return self

    def __next__(self) -> Union[T, AutomaticRunInput[T]]:
        try:
            result = self.next()  # type: ignore
            return result  # type: ignore
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> AsyncIterator[Union[T, AutomaticRunInput[T]]]:
        return self

    async def __anext__(self) -> Union[T, AutomaticRunInput[T]]:
        try:
            coro: Coroutine[Any, Any, Union[T, AutomaticRunInput[T]]] = self.next()  # type: ignore
            return cast(Union[T, AutomaticRunInput[T]], await coro)
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self) -> List[Any]:
        flow_run_inputs = await filter_flow_run_input(
            key_prefix=self.key_prefix,
            limit=1,
            exclude_keys=self.exclude_keys,
            flow_run_id=self.flow_run_id,
        )
        if flow_run_inputs:
            self.exclude_keys.update({i.key for i in flow_run_inputs})
        return flow_run_inputs

    @sync_compatible
    async def next(self) -> Union[T, AutomaticRunInput[T]]:
        flow_run_inputs = await self.filter_for_inputs()
        if flow_run_inputs:
            return self.to_instance(flow_run_inputs[0])
        with anyio.fail_after(self.timeout):
            while True:
                await anyio.sleep(self.poll_interval)
                flow_run_inputs = await self.filter_for_inputs()
                if flow_run_inputs:
                    return self.to_instance(flow_run_inputs[0])

    def to_instance(self, flow_run_input: Any) -> Union[T, AutomaticRunInput[T]]:
        run_input = self.run_input_cls.load_from_flow_run_input(flow_run_input)
        if self.with_metadata:
            return run_input
        return run_input.value  # type: ignore


async def _send_input(flow_run_id: UUID, run_input: Any, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
    if isinstance(run_input, RunInput):
        _run_input: RunInput = run_input
    else:
        input_cls = run_input_subclass_from_type(type(run_input))
        _run_input = input_cls(value=run_input)  # type: ignore
    if key_prefix is None:
        key_prefix = f"{_run_input.__class__.__name__.lower()}-auto"
    key: str = f"{key_prefix}-{uuid4()}"
    coro: Coroutine[Any, Any, Any] = create_flow_run_input_from_model(key=key, flow_run_id=flow_run_id, model_instance=_run_input, sender=sender)
    await coro


@sync_compatible
async def send_input(run_input: Any, flow_run_id: UUID, sender: Optional[Any] = None, key_prefix: Optional[str] = None) -> None:
    await _send_input(flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix)


@overload
def receive_input(
    input_type: Any,
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> GetInputHandler[Any]:
    ...


@overload
def receive_input(
    input_type: Any,
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> GetAutomaticInputHandler[Any]:
    ...


def receive_input(
    input_type: Any,
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> Union[GetInputHandler[Any], GetAutomaticInputHandler[Any]]:
    input_cls = run_input_subclass_from_type(input_type)
    if issubclass(input_cls, AutomaticRunInput):
        return input_cls.receive(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_timeout_error=raise_timeout_error,
            exclude_keys=exclude_keys,
            key_prefix=key_prefix,
            flow_run_id=flow_run_id,
            with_metadata=with_metadata,
        )
    else:
        return input_cls.receive(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_timeout_error=raise_timeout_error,
            exclude_keys=exclude_keys,
            key_prefix=key_prefix,
            flow_run_id=flow_run_id,
        )