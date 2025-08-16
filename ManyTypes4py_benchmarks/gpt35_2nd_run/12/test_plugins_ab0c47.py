from __future__ import annotations
import contextlib
from collections.abc import Generator
from functools import partial
from typing import Any
from pydantic_core import ValidationError
from pydantic import BaseModel, TypeAdapter, create_model, dataclasses, field_validator, validate_call
from pydantic.plugin import PydanticPluginProtocol, SchemaTypePath, ValidateJsonHandlerProtocol, ValidatePythonHandlerProtocol, ValidateStringsHandlerProtocol
from pydantic.plugin._loader import _plugins

@contextlib.contextmanager
def install_plugin(plugin: PydanticPluginProtocol) -> Generator:
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

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_on_validate_json_on_success.<locals>.Model')
            assert schema_kind == 'BaseModel'
            return (None, CustomOnValidateJson(), None)
    plugin = CustomPlugin()
    with install_plugin(plugin):

        class Model(BaseModel, plugin_settings={'observe': 'all'}):
            pass
        assert Model.model_validate({'a': 1}) == Model(a=1)
        assert Model.model_validate_json('{"a": 1}') == Model(a=1)
        assert Model.__pydantic_validator__.title == 'Model'

def test_on_validate_json_on_error() -> None:

    class CustomOnValidateJson:

        def on_enter(self, input: Any, *, strict: Any = None, context: Any = None, self_instance: Any = None) -> None:
            assert input == '{"a": "potato"}'
            assert strict is None
            assert context is None
            assert self_instance is None

        def on_error(self, error: Any) -> None:
            assert error.title == 'Model'
            assert error.errors(include_url=False) == [{'input': 'potato', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

    class Plugin(PydanticPluginProtocol):

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            return (None, CustomOnValidateJson(), None)
    plugin = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, plugin_settings={'observe': 'all'}):
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

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_kind == 'BaseModel'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, plugin_settings={'observe': 'all'}):
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
            assert error.errors(include_url=False) == [{'input': 'potato', 'loc': ('a',), 'msg': 'Input should be a valid integer, unable to parse string as an integer', 'type': 'int_parsing'}]

    class Plugin(PydanticPluginProtocol):

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert config == {'title': 'Model'}
            assert plugin_settings == {'observe': 'all'}
            assert schema_type.__name__ == 'Model'
            assert schema_kind == 'BaseModel'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel, plugin_settings={'observe': 'all'}):
            pass
        with contextlib.suppress(ValidationError):
            Model.model_validate({'a': 'potato'})
        assert Model.model_validate_json('{"a": 1}').model_dump() == {'a': 1}

def test_stateful_plugin() -> None:
    stack: list = []

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):

        def on_enter(self, input: Any, *, strict: Any = None, from_attributes: Any = None, context: Any = None, self_instance: Any = None) -> None:
            stack.append(input)

        def on_success(self, result: Any) -> None:
            stack.pop()

        def on_error(self, error: Any) -> None:
            stack.pop()

        def on_exception(self, exception: Any) -> None:
            stack.pop()

    class Plugin(PydanticPluginProtocol):

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()

    class MyException(Exception):
        pass
    with install_plugin(plugin):

        class Model(BaseModel, plugin_settings={'observe': 'all'}):

            @field_validator('a')
            def validate_a(cls, v: Any) -> Any:
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
    log: list = []

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

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            return (Python(), Json(), Strings())
    plugin = Plugin()
    with install_plugin(plugin):

        class Model(BaseModel):
            pass
        assert Model(a=1).model_dump() == {'a': 1}
        assert log == ["python enter input={'a': 1} kwargs={'self_instance': Model()}", 'python success result=a=1']
        log.clear()
        assert Model.model_validate_json('{"a": 2}', context={'c': 2}).model_dump() == {'a': 2}
        assert log == ['json enter input={"a": 2} kwargs={\'strict\': None, \'context\': {\'c\': 2}}', 'json success result=a=2']
        log.clear()
        assert Model.model_validate_strings({'a': '3'}, strict=True, context={'c': 3}).model_dump() == {'a': 3}
        assert log == ["strings enter input={'a': '3'} kwargs={'strict': True, 'context': {'c': 3}}", 'strings success result=a=3']

def test_plugin_path_dataclass() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert schema_type.__name__ == 'Bar'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_plugin_path_dataclass.<locals>.Bar')
            assert schema_kind == 'dataclass'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):

        @dataclasses.dataclass
        class Bar:
            pass

def test_plugin_path_type_adapter() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):
        adapter = TypeAdapter(list[str])
        adapter.validate_python(['a', 'b'])

def test_plugin_path_type_adapter_with_module() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('provided_module_by_type_adapter', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):
        TypeAdapter(list[str], module='provided_module_by_type_adapter')

def test_plugin_path_type_adapter_without_name_in_globals() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert str(schema_type) == 'list[str]'
            assert schema_type_path == SchemaTypePath('', 'list[str]')
            assert schema_kind == 'TypeAdapter'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):
        code = '\nimport pydantic\npydantic.TypeAdapter(list[str])\n'
        exec(code, {'bar': 'baz'})

def test_plugin_path_validate_call() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin1:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert schema_type.__name__ == 'foo'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'test_plugin_path_validate_call.<locals>.foo')
            assert schema_kind == 'validate_call'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin1()
    with install_plugin(plugin):

        @validate_call()
        def foo(a: Any) -> Any:
            return a

    class Plugin2:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert schema_type.__name__ == 'my_wrapped_function'
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'partial(test_plugin_path_validate_call.<locals>.my_wrapped_function)')
            assert schema_kind == 'validate_call'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin2()
    with install_plugin(plugin):

        def my_wrapped_function(a: Any, b: Any, c: Any) -> Any:
            return a + b + c
        my_partial_function = partial(my_wrapped_function, c=3)
        validate_call(my_partial_function)

def test_plugin_path_create_model() -> None:

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            assert schema_type.__name__ == 'FooModel'
            assert list(schema_type.model_fields.keys()) == ['foo', 'bar']
            assert schema_type_path == SchemaTypePath('tests.test_plugins', 'FooModel')
            assert schema_kind == 'create_model'
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):
        create_model('FooModel', foo=(str, ...), bar=(int, 123))

def test_plugin_path_complex() -> None:
    paths: list = []

    class CustomOnValidatePython(ValidatePythonHandlerProtocol):
        pass

    class Plugin:

        def new_schema_validator(self, schema: Any, schema_type: Any, schema_type_path: Any, schema_kind: Any, config: Any, plugin_settings: Any) -> Any:
            paths.append((schema_type.__name__, schema_type_path, schema_kind))
            return (CustomOnValidatePython(), None, None)
    plugin = Plugin()
    with install_plugin(plugin):

        def foo() -> None:

            class Model1(BaseModel):
                pass

        def bar() -> None:

            class Model2(BaseModel):
                pass
        foo()
        bar()
    assert paths == [('Model1', SchemaTypePath('tests.test_plugins', 'test_plugin_path_complex.<locals>.foo.<locals>.Model1'), 'BaseModel'), ('Model2', SchemaTypePath('tests.test_plugins', 'test_plugin_path_complex.<locals>.bar.<locals>.Model2'), 'BaseModel')]
