from __future__ import annotations
import contextlib
from collections.abc import Generator
from functools import partial
from typing import Any, Optional, Tuple, Iterator
from pydantic_core import ValidationError
from pydantic import BaseModel, TypeAdapter, create_model, dataclasses, field_validator, validate_call
from pydantic.plugin import (
    PydanticPluginProtocol,
    SchemaTypePath,
    ValidateJsonHandlerProtocol,
    ValidatePythonHandlerProtocol,
    ValidateStringsHandlerProtocol,
)
from pydantic.plugin._loader import _plugins


@contextlib.contextmanager
def install_plugin(plugin: PydanticPluginProtocol) -> Iterator[None]:
    _plugins[plugin.__class__.__qualname__] = plugin
    try:
        yield
    finally:
        _plugins.clear()


def test_on_validate_json_on_success() -> None:
    class CustomOnValidateJson(ValidateJsonHandlerProtocol):
        def on_enter(self, input: Any, *, strict: Any = None, context: Any = None, self_instance: Any = None) -> None:
            assert input == '{"a": 1}'
            assert strict is None
            assert context is None
            assert self_instance is None

        def on_success(self, result: Any) -> None:
            assert isinstance(result, Model)

    class CustomPlugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_on_validate_json_on_success.<locals>.Model')
            assert schema_kind == 'BaseModel'
            return (None, CustomOnValidateJson(), None)

    plugin: PydanticPluginProtocol = CustomPlugin()
    with install_plugin(plugin):

        class Model(BaseModel, metaclass=type('ModelMeta', (), {}), plugin_settings={'observe': 'all'}):  # type: ignore
            pass

        assert Model.model_validate({'a': 1}) == Model(a=1)
        assert Model.model_validate_json('{"a": 1}') == Model(a=1)
        assert Model.__pydantic_validator__.title == 'Model'


def test_on_validate_json_on_error() -> None:
    class CustomOnValidateJson(ValidateJsonHandlerProtocol):
        def on_enter(self, input: Any, *, strict: Any = None, context: Any = None, self_instance: Any = None) -> None:
            assert input == '{"a": "potato"}'
            assert strict is None
            assert context is None
            assert self_instance is None

        def on_error(self, error: Any) -> None:
            assert error.title == 'Model'
            expected = [{
                'input': 'potato',
                'loc': ('a',),
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'type': 'int_parsing',
            }]
            assert error.errors(include_url=False) == expected

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            return (None, CustomOnValidateJson(), None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, metaclass=type('ModelMeta', (), {}), plugin_settings={'observe': 'all'}):  # type: ignore
            pass

        assert Model.model_validate({'a': 1}) == Model(a=1)
        with contextlib.suppress(ValidationError):
            Model.model_validate_json('{"a": "potato"}')


def test_on_validate_python_on_success() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        def on_enter(self, input: Any, *, strict: Any = None, from_attributes: Any = None, context: Any = None, self_instance: Any = None) -> None:
            assert input == {'a': 1}
            assert strict is None
            assert context is None
            assert self_instance is None

        def on_success(self, result: Any) -> None:
            assert isinstance(result, Model)

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_kind == 'BaseModel'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, metaclass=type('ModelMeta', (), {}), plugin_settings={'observe': 'all'}):  # type: ignore
            pass

        assert Model.model_validate({'a': 1}).model_dump() == {'a': 1}
        assert Model.model_validate_json('{"a": 1}').model_dump() == {'a': 1}


def test_on_validate_python_on_error() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        def on_enter(self, input: Any, *, strict: Any = None, from_attributes: Any = None, context: Any = None, self_instance: Any = None) -> None:
            assert input == {'a': 'potato'}
            assert strict is None
            assert context is None
            assert self_instance is None

        def on_error(self, error: Any) -> None:
            assert error.title == 'Model'
            expected = [{
                'input': 'potato',
                'loc': ('a',),
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'type': 'int_parsing',
            }]
            assert error.errors(include_url=False) == expected

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_kind == 'BaseModel'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, metaclass=type('ModelMeta', (), {}), plugin_settings={'observe': 'all'}):  # type: ignore
            pass

        with contextlib.suppress(ValidationError):
            Model.model_validate({'a': 'potato'})
        assert Model.model_validate_json('{"a": 1}').model_dump() == {'a': 1}


def test_stateful_plugin() -> None:
    stack: list[Any] = []

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        def on_enter(self, input: Any, *, strict: Any = None, from_attributes: Any = None, context: Any = None, self_instance: Any = None) -> None:
            stack.append(input)

        def on_success(self, result: Any) -> None:
            stack.pop()

        def on_error(self, error: Any) -> None:
            stack.pop()

        def on_exception(self, exception: Exception) -> None:
            stack.pop()

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()

    class MyException(Exception):
        pass

    with install_plugin(plugin):

        class Model(BaseModel, metaclass=type('ModelMeta', (), {}), plugin_settings={'observe': 'all'}):  # type: ignore
            @field_validator('a')
            def validate_a(cls, v: int) -> int:
                if v < 0:
                    raise MyException
                return v

        with contextlib.suppress(ValidationError):
            Model.model_validate({'a': 'potato'})
        assert not stack
        with contextlib.suppress(MyException):
            Model.model_validate({'a': -1})
        assert not stack
        assert Model.model_validate({'a': 1}).a == 1
        assert not stack


def test_all_handlers() -> None:
    log: list[str] = []

    class Python(ValidatePythonHandlerProtocol):
        def on_enter(self, input: Any, **kwargs: Any) -> None:
            log.append(f'python enter input={input} kwargs={kwargs}')

        def on_success(self, result: Any) -> None:
            log.append(f'python success result={result}')

        def on_error(self, error: Any) -> None:
            log.append(f'python error error={error}')

    class Json(ValidateJsonHandlerProtocol):
        def on_enter(self, input: Any, **kwargs: Any) -> None:
            log.append(f'json enter input={input} kwargs={kwargs}')

        def on_success(self, result: Any) -> None:
            log.append(f'json success result={result}')

        def on_error(self, error: Any) -> None:
            log.append(f'json error error={error}')

    class Strings(ValidateStringsHandlerProtocol):
        def on_enter(self, input: Any, **kwargs: Any) -> None:
            log.append(f'strings enter input={input} kwargs={kwargs}')

        def on_success(self, result: Any) -> None:
            log.append(f'strings success result={result}')

        def on_error(self, error: Any) -> None:
            log.append(f'strings error error={error}')

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            return (Python(), Json(), Strings())

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel):
            pass

        # When creating via __init__
        instance = Model(a=1)
        assert instance.model_dump() == {'a': 1}
        # The plugin for __init__ uses the python handler.
        assert log == [
            "python enter input={'a': 1} kwargs={'self_instance': Model()}",
            'python success result=a=1',
        ]
        log.clear()

        result_json = Model.model_validate_json('{"a": 2}', context={'c': 2})
        assert result_json.model_dump() == {'a': 2}
        assert log == [
            "json enter input={\"a\": 2} kwargs={'strict': None, 'context': {'c': 2}}",
            'json success result=a=2',
        ]
        log.clear()

        result_strings = Model.model_validate_strings({'a': '3'}, strict=True, context={'c': 3})
        assert result_strings.model_dump() == {'a': 3}
        assert log == [
            "strings enter input={'a': '3'} kwargs={'strict': True, 'context': {'c': 3}}",
            'strings success result=a=3',
        ]


def test_plugin_path_dataclass() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert schema_type.__name__ == 'Bar'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_plugin_path_dataclass.<locals>.Bar')
            assert schema_kind == 'dataclass'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        @dataclasses.dataclass
        class Bar:
            pass


def test_plugin_path_type_adapter() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):
        adapter: TypeAdapter[list[str]] = TypeAdapter(list[str])
        adapter.validate_python(['a', 'b'])


def test_plugin_path_type_adapter_with_module() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('provided_module_by_type_adapter', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):
        _ = TypeAdapter(list[str], module='provided_module_by_type_adapter')


def test_plugin_path_type_adapter_without_name_in_globals() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):
        code: str = '\nimport pydantic\npydantic.TypeAdapter(list[str])\n'
        exec(code, {'bar': 'baz'})


def test_plugin_path_validate_call() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin1(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert schema_type.__name__ == 'foo'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_plugin_path_validate_call.<locals>.foo')
            assert schema_kind == 'validate_call'
            return (CustomOnValidatePython(), None, None)

    plugin1: PydanticPluginProtocol = Plugin1()
    with install_plugin(plugin1):

        @validate_call()
        def foo(a: Any) -> Any:
            return a

    class Plugin2(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert schema_type.__name__ == 'my_wrapped_function'
            assert schema_type_path == SchemaTypePath(
                'tests.test_plugins', 'partial(test_plugin_path_validate_call.<locals>.my_wrapped_function)'
            )
            assert schema_kind == 'validate_call'
            return (CustomOnValidatePython(), None, None)

    plugin2: PydanticPluginProtocol = Plugin2()
    with install_plugin(plugin2):

        def my_wrapped_function(a: int, b: int, c: int) -> int:
            return a + b + c

        my_partial_function = partial(my_wrapped_function, c=3)
        validate_call(my_partial_function)


def test_plugin_path_create_model() -> None:
    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            assert schema_type.__name__ == 'FooModel'
            assert list(schema_type.model_fields.keys()) == ['foo', 'bar']
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'FooModel')
            assert schema_kind == 'create_model'
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):
        _ = create_model('FooModel', foo=(str, ...), bar=(int, 123))


def test_plugin_path_complex() -> None:
    paths: list[Tuple[str, SchemaTypePath, str]] = []

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin(PydanticPluginProtocol):
        def new_schema_validator(
            self,
            schema: Any,
            schema_type: Any,
            schema_type_path: SchemaTypePath,
            schema_kind: str,
            config: Any,
            plugin_settings: Any,
        ) -> Tuple[
            Optional[ValidatePythonHandlerProtocol],
            Optional[ValidateJsonHandlerProtocol],
            Optional[ValidateStringsHandlerProtocol],
        ]:
            paths.append((schema_type.__name__, schema_type_path, schema_kind))
            return (CustomOnValidatePython(), None, None)

    plugin: PydanticPluginProtocol = Plugin()
    with install_plugin(plugin):

        def foo() -> None:
            class Model1(BaseModel):
                pass

        def bar() -> None:
            class Model2(BaseModel):
                pass

        foo()
        bar()
    assert paths == [
        (
            'Model1',
            SchemaTypePath('tests.test_plugins', 'test_plugin_path_complex.<locals>.foo.<locals>.Model1'),
            'BaseModel',
        ),
        (
            'Model2',
            SchemaTypePath('tests.test_plugins', 'test_plugin_path_complex.<locals>.bar.<locals>.Model2'),
            'BaseModel',
        ),
    ]