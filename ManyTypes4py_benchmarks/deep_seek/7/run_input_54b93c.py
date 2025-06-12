from __future__ import annotations
import inspect
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    ClassVar,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    Literal,
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
from pydantic import ConfigDict, BaseModel
from typing_extensions import Self
from prefect.input.actions import (
    create_flow_run_input,
    create_flow_run_input_from_model,
    ensure_flow_run_id,
    filter_flow_run_input,
    read_flow_run_input,
)
from prefect.utilities.asyncutils import sync_compatible

if TYPE_CHECKING:
    from prefect.client.schemas.objects import FlowRunInput
    from prefect.states import State

R = TypeVar("R", bound="RunInput")
T = TypeVar("T", bound=object)
Keyset = Dict[Union[Literal["description"], Literal["response"], Literal["schema"]], str]


def keyset_from_paused_state(state: "State") -> Keyset:
    if not state.is_paused():
        raise RuntimeError(f"{state.type.value!r} is unsupported.")
    state_name = state.name or ""
    base_key = f"{state_name.lower()}-{str(state.state_details.pause_key)}"
    return keyset_from_base_key(base_key)


def keyset_from_base_key(base_key: str) -> Keyset:
    return {
        "description": f"{base_key}-description",
        "response": f"{base_key}-response",
        "schema": f"{base_key}-schema",
    }


class RunInputMetadata(pydantic.BaseModel):
    key: Optional[str] = None
    sender: Optional[str] = None
    receiver: Optional[UUID] = None


class BaseRunInput(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid")
    _description: ClassVar[Optional[str]] = pydantic.PrivateAttr(default=None)
    _metadata: RunInputMetadata = pydantic.PrivateAttr()

    @property
    def metadata(self) -> RunInputMetadata:
        return self._metadata

    @classmethod
    def keyset_from_type(cls) -> Keyset:
        return keyset_from_base_key(cls.__name__.lower())

    @classmethod
    @sync_compatible
    async def save(cls, keyset: Keyset, flow_run_id: Optional[UUID] = None) -> None:
        if is_v2_model(cls):
            schema = create_v2_schema(cls.__name__, model_base=cls)
        else:
            schema = cls.model_json_schema(by_alias=True)
        coro = create_flow_run_input(
            key=keyset["schema"], value=schema, flow_run_id=flow_run_id
        )
        if TYPE_CHECKING:
            assert inspect.iscoroutine(coro)
        await coro
        description = cls._description if isinstance(cls._description, str) else None
        if description:
            coro = create_flow_run_input(
                key=keyset["description"], value=description, flow_run_id=flow_run_id
            )
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            await coro

    @classmethod
    @sync_compatible
    async def load(cls, keyset: Keyset, flow_run_id: Optional[UUID] = None) -> Self:
        flow_run_id = ensure_flow_run_id(flow_run_id)
        value = await read_flow_run_input(keyset["response"], flow_run_id=flow_run_id)
        if value:
            instance = cls(**value)
        else:
            instance = cls()
        instance._metadata = RunInputMetadata(
            key=keyset["response"], sender=None, receiver=flow_run_id
        )
        return instance

    @classmethod
    def load_from_flow_run_input(cls, flow_run_input: "FlowRunInput") -> Self:
        instance = cls(**flow_run_input.decoded_value)
        instance._metadata = RunInputMetadata(
            key=flow_run_input.key,
            sender=flow_run_input.sender,
            receiver=flow_run_input.flow_run_id,
        )
        return instance

    @classmethod
    def with_initial_data(cls, description: Optional[str] = None, **kwargs: Any) -> Type[Self]:
        fields = {}
        for key, value in kwargs.items():
            fields[key] = (type(value), value)
        model = pydantic.create_model(cls.__name__, **fields, __base__=cls)
        if description is not None:
            model._description = description
        return model

    @sync_compatible
    async def respond(
        self,
        run_input: Union["BaseRunInput", Any],
        sender: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> None:
        flow_run_id = None
        if self.metadata.sender and self.metadata.sender.startswith("prefect.flow-run"):
            _, _, id = self.metadata.sender.rpartition(".")
            flow_run_id = UUID(id)
        if not flow_run_id:
            raise RuntimeError("Cannot respond to an input that was not sent by a flow run.")
        await _send_input(
            flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix
        )

    @sync_compatible
    async def send_to(
        self,
        flow_run_id: UUID,
        sender: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> None:
        await _send_input(
            flow_run_id=flow_run_id, run_input=self, sender=sender, key_prefix=key_prefix
        )


class RunInput(BaseRunInput):
    @classmethod
    def receive(
        cls,
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
        flow_run_id: Optional[UUID] = None,
    ) -> "GetInputHandler[Self]":
        if key_prefix is None:
            key_prefix = f"{cls.__name__.lower()}-auto"
        return GetInputHandler(
            run_input_cls=cls,
            key_prefix=key_prefix,
            timeout=timeout,
            poll_interval=poll_interval,
            raise_timeout_error=raise_timeout_error,
            exclude_keys=exclude_keys,
            flow_run_id=flow_run_id,
        )

    @classmethod
    def subclass_from_base_model_type(cls, model_cls: Type[BaseModel]) -> Type["RunInput"]:
        return type(f"{model_cls.__name__}RunInput", (RunInput, model_cls), {})


class AutomaticRunInput(BaseRunInput, Generic[T]):
    value: T

    @classmethod
    @sync_compatible
    async def load(cls, keyset: Keyset, flow_run_id: Optional[UUID] = None) -> T:
        instance_coro = super().load(keyset, flow_run_id=flow_run_id)
        if TYPE_CHECKING:
            assert inspect.iscoroutine(instance_coro)
        instance = await instance_coro
        return instance.value

    @classmethod
    def subclass_from_type(cls, _type: Type[T]) -> Type["AutomaticRunInput[T]"]:
        fields = {"value": (_type, ...)}
        type_prefix = getattr(_type, "__name__", getattr(_type, "_name", "")).lower()
        class_name = f"{type_prefix}AutomaticRunInput"
        new_cls = pydantic.create_model(class_name, **fields, __base__=AutomaticRunInput)
        return new_cls

    @classmethod
    def receive(
        cls,
        timeout: int = 3600,
        poll_interval: int = 10,
        raise_timeout_error: bool = False,
        exclude_keys: Optional[Set[str]] = None,
        key_prefix: Optional[str] = None,
        flow_run_id: Optional[UUID] = None,
        with_metadata: bool = False,
    ) -> "GetAutomaticInputHandler[T]":
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


def run_input_subclass_from_type(
    _type: Union[Type[R], Type[T]]
) -> Union[Type[R], Type["AutomaticRunInput[T]"]]:
    if isclass(_type):
        if issubclass(_type, RunInput):
            return cast(Type[R], _type)
        elif issubclass(_type, BaseModel):
            return cast(Type[R], RunInput.subclass_from_base_model_type(_type))
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
    ):
        self.run_input_cls = run_input_cls
        self.key_prefix = key_prefix
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.exclude_keys: Set[str] = set()
        self.raise_timeout_error = raise_timeout_error
        self.flow_run_id = ensure_flow_run_id(flow_run_id)
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> Iterator[R]:
        return self

    def __next__(self) -> R:
        try:
            return cast(R, self.next())
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> AsyncIterator[R]:
        return self

    async def __anext__(self) -> R:
        try:
            coro = self.next()
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            return await coro
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self) -> list["FlowRunInput"]:
        flow_run_inputs_coro = filter_flow_run_input(
            key_prefix=self.key_prefix,
            limit=1,
            exclude_keys=self.exclude_keys,
            flow_run_id=self.flow_run_id,
        )
        if TYPE_CHECKING:
            assert inspect.iscoroutine(flow_run_inputs_coro)
        flow_run_inputs = await flow_run_inputs_coro
        if flow_run_inputs:
            self.exclude_keys.update(i.key for i in flow_run_inputs)
        return flow_run_inputs

    def to_instance(self, flow_run_input: "FlowRunInput") -> R:
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
    ):
        self.run_input_cls = run_input_cls
        self.key_prefix = key_prefix
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.exclude_keys: Set[str] = set()
        self.raise_timeout_error = raise_timeout_error
        self.flow_run_id = ensure_flow_run_id(flow_run_id)
        self.with_metadata = with_metadata
        if exclude_keys is not None:
            self.exclude_keys.update(exclude_keys)

    def __iter__(self) -> Iterator[Union[T, AutomaticRunInput[T]]]:
        return self

    def __next__(self) -> Union[T, AutomaticRunInput[T]]:
        try:
            not_coro = self.next()
            if TYPE_CHECKING:
                assert not isinstance(not_coro, Coroutine)
            return not_coro
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopIteration

    def __aiter__(self) -> AsyncIterator[Union[T, AutomaticRunInput[T]]]:
        return self

    async def __anext__(self) -> Union[T, AutomaticRunInput[T]]:
        try:
            coro = self.next()
            if TYPE_CHECKING:
                assert inspect.iscoroutine(coro)
            return cast(Union[T, AutomaticRunInput[T]], await coro)
        except TimeoutError:
            if self.raise_timeout_error:
                raise
            raise StopAsyncIteration

    async def filter_for_inputs(self) -> list["FlowRunInput"]:
        flow_run_inputs_coro = filter_flow_run_input(
            key_prefix=self.key_prefix,
            limit=1,
            exclude_keys=self.exclude_keys,
            flow_run_id=self.flow_run_id,
        )
        if TYPE_CHECKING:
            assert inspect.iscoroutine(flow_run_inputs_coro)
        flow_run_inputs = await flow_run_inputs_coro
        if flow_run_inputs:
            self.exclude_keys.update(i.key for i in flow_run_inputs)
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

    def to_instance(self, flow_run_input: "FlowRunInput") -> Union[T, AutomaticRunInput[T]]:
        run_input = self.run_input_cls.load_from_flow_run_input(flow_run_input)
        if self.with_metadata:
            return run_input
        return run_input.value


async def _send_input(
    flow_run_id: UUID,
    run_input: Union[BaseRunInput, Any],
    sender: Optional[str] = None,
    key_prefix: Optional[str] = None,
) -> None:
    if isinstance(run_input, RunInput):
        _run_input = run_input
    else:
        input_cls = run_input_subclass_from_type(type(run_input))
        _run_input = input_cls(value=run_input)
    if key_prefix is None:
        key_prefix = f"{_run_input.__class__.__name__.lower()}-auto"
    key = f"{key_prefix}-{uuid4()}"
    coro = create_flow_run_input_from_model(
        key=key, flow_run_id=flow_run_id, model_instance=_run_input, sender=sender
    )
    if TYPE_CHECKING:
        assert inspect.iscoroutine(coro)
    await coro


@sync_compatible
async def send_input(
    run_input: Union[BaseRunInput, Any],
    flow_run_id: UUID,
    sender: Optional[str] = None,
    key_prefix: Optional[str] = None,
) -> None:
    await _send_input(flow_run_id=flow_run_id, run_input=run_input, sender=sender, key_prefix=key_prefix)


@overload
def receive_input(
    input_type: Type[R],
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> GetInputHandler[R]:
    ...


@overload
def receive_input(
    input_type: Type[T],
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> GetAutomaticInputHandler[T]:
    ...


def receive_input(
    input_type: Union[Type[R], Type[T]],
    timeout: int = 3600,
    poll_interval: int = 10,
    raise_timeout_error: bool = False,
    exclude_keys: Optional[Set[str]] = None,
    key_prefix: Optional[str] = None,
    flow_run_id: Optional[UUID] = None,
    with_metadata: bool = False,
) -> Union[GetInputHandler[R], GetAutomaticInputHandler[T]]:
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
           