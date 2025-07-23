import datetime
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
import pendulum
import pydantic.version
import pytest
from pydantic import SecretStr
from prefect.exceptions import ParameterBindError
from prefect.utilities import callables


class TestFunctionToSchema:

    def test_simple_function_with_no_arguments(self) -> None:

        def f() -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'properties': {}, 'title': 'Parameters', 'type': 'object', 'definitions': {}}

    def test_function_with_pydantic_base_model_collisions(self) -> None:

        def f(
            json: Any,
            copy: Any,
            parse_obj: Any,
            parse_raw: Any,
            parse_file: Any,
            from_orm: Any,
            schema: Any,
            schema_json: Any,
            construct: Any,
            validate: Any,
            foo: Any,
        ) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'json': {'title': 'json', 'position': 0}, 'copy': {'title': 'copy', 'position': 1}, 'parse_obj': {'title': 'parse_obj', 'position': 2}, 'parse_raw': {'title': 'parse_raw', 'position': 3}, 'parse_file': {'title': 'parse_file', 'position': 4}, 'from_orm': {'title': 'from_orm', 'position': 5}, 'schema': {'title': 'schema', 'position': 6}, 'schema_json': {'title': 'schema_json', 'position': 7}, 'construct': {'title': 'construct', 'position': 8}, 'validate': {'title': 'validate', 'position': 9}, 'foo': {'title': 'foo', 'position': 10}}, 'required': ['json', 'copy', 'parse_obj', 'parse_raw', 'parse_file', 'from_orm', 'schema', 'schema_json', 'construct', 'validate', 'foo'], 'definitions': {}}
    
    def test_function_with_one_required_argument(self) -> None:

        def f(x: Any) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_one_optional_argument(self) -> None:

        def f(x: Any = 42) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'default': 42, 'position': 0, 'title': 'x'}}, 'definitions': {}}
    
    def test_function_with_one_optional_annotated_argument(self) -> None:

        def f(x: int = 42) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'default': 42, 'position': 0, 'title': 'x', 'type': 'integer'}}, 'definitions': {}}
    
    def test_function_with_two_arguments(self) -> None:

        def f(x: int, y: float = 5.0) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'type': 'integer', 'position': 0}, 'y': {'title': 'y', 'default': 5.0, 'type': 'number', 'position': 1}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_datetime_arguments(self) -> None:

        def f(
            x: Any,
            y: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
            z: datetime.timedelta = datetime.timedelta(seconds=5),
        ) -> None:
            pass
        schema = callables.parameter_schema(f)
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'format': 'date-time', 'position': 0, 'title': 'x', 'type': 'string'}, 'y': {'default': '2025-01-01T00:00:00Z', 'format': 'date-time', 'position': 1, 'title': 'y', 'type': 'string'}, 'z': {'default': 'PT5S', 'format': 'duration', 'position': 2, 'title': 'z', 'type': 'string'}}, 'required': ['x'], 'definitions': {}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_enum_argument(self) -> None:

        class Color(Enum):
            RED = 'RED'
            GREEN = 'GREEN'
            BLUE = 'BLUE'
    
        def f(x: Color = Color.RED) -> None:
            pass
        schema = callables.parameter_schema(f)
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'$ref': '#/definitions/Color', 'default': 'RED', 'position': 0, 'title': 'x'}}, 'definitions': {'Color': {'enum': ['RED', 'GREEN', 'BLUE'], 'title': 'Color', 'type': 'string'}}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_generic_arguments(self) -> None:

        def f(a: List[str], b: Dict[str, Any], c: Any, d: Tuple[int, float], e: Union[str, bytes, int]) -> None:
            pass
        schema = callables.parameter_schema(f)
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'a': {'items': {'type': 'string'}, 'position': 0, 'title': 'a', 'type': 'array'}, 'b': {'position': 1, 'title': 'b', 'type': 'object'}, 'c': {'position': 2, 'title': 'c'}, 'd': {'maxItems': 2, 'minItems': 2, 'position': 3, 'prefixItems': [{'type': 'integer'}, {'type': 'number'}], 'title': 'd', 'type': 'array'}, 'e': {'anyOf': [{'type': 'string'}, {'format': 'binary', 'type': 'string'}, {'type': 'integer'}], 'position': 4, 'title': 'e'}}, 'required': ['a', 'b', 'c', 'd', 'e'], 'definitions': {}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_user_defined_type(self) -> None:

        class Foo:
            pass
    
        def f(x: Foo) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'definitions': {}, 'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x']}
    
    def test_function_with_user_defined_pydantic_model(self) -> None:

        class Foo(pydantic.BaseModel):
            y: int
            z: str
    
        def f(x: Foo) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'definitions': {'Foo': {'properties': {'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'string'}}, 'required': ['y', 'z'], 'title': 'Foo', 'type': 'object'}}, 'properties': {'x': {'$ref': '#/definitions/Foo', 'title': 'x', 'position': 0}}, 'required': ['x'], 'title': 'Parameters', 'type': 'object'}
    
    def test_function_with_pydantic_model_default_across_v1_and_v2(self) -> None:
        import pydantic
    
        class Foo(pydantic.BaseModel):
            bar: str
    
        def f(foo: Foo = Foo(bar='baz')) -> None:
            ...
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'foo': {'$ref': '#/definitions/Foo', 'default': {'bar': 'baz'}, 'position': 0, 'title': 'foo'}}, 'definitions': {'Foo': {'properties': {'bar': {'title': 'Bar', 'type': 'string'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}}}
    
    def test_function_with_complex_args_across_v1_and_v2(self) -> None:
        import pydantic
    
        class Foo(pydantic.BaseModel):
            bar: str
    
        class Color(Enum):
            RED = 'RED'
            GREEN = 'GREEN'
            BLUE = 'BLUE'
    
        def f(
            a: int,
            s: List[Optional[Any]],
            m: Foo,
            i: int = 0,
            x: float = 1.0,
            model: Foo = Foo(bar='bar'),
            pdt: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
            pdate: pendulum.Date = pendulum.date(2025, 1, 1),
            pduration: pendulum.Duration = pendulum.duration(seconds=5),
            c: Color = Color.BLUE,
        ) -> None:
            ...
        datetime_schema: Dict[str, Any] = {'title': 'pdt', 'default': '2025-01-01T00:00:00+00:00', 'position': 6, 'type': 'string', 'format': 'date-time'}
        duration_schema: Dict[str, Any] = {'title': 'pduration', 'default': 5.0, 'position': 8, 'type': 'number', 'format': 'time-delta'}
        enum_schema: Dict[str, Any] = {'enum': ['RED', 'GREEN', 'BLUE'], 'title': 'Color', 'type': 'string', 'description': 'An enumeration.'}
        datetime_schema['default'] = '2025-01-01T00:00:00Z'
        duration_schema['default'] = 'PT5S'
        duration_schema['type'] = 'string'
        duration_schema['format'] = 'duration'
        enum_schema.pop('description')
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'a': {'position': 0, 'title': 'a', 'type': 'integer'}, 's': {'items': {'type': 'null'}, 'position': 1, 'title': 's', 'type': 'array'}, 'm': {'$ref': '#/definitions/Foo', 'position': 2, 'title': 'm'}, 'i': {'default': 0, 'position': 3, 'title': 'i', 'type': 'integer'}, 'x': {'default': 1.0, 'position': 4, 'title': 'x', 'type': 'number'}, 'model': {'$ref': '#/definitions/Foo', 'default': {'bar': 'bar'}, 'position': 5, 'title': 'model'}, 'pdt': datetime_schema, 'pdate': {'title': 'pdate', 'default': '2025-01-01', 'position': 7, 'type': 'string', 'format': 'date'}, 'pduration': duration_schema, 'c': {'title': 'c', 'default': 'BLUE', 'position': 9, '$ref': '#/definitions/Color'}}, 'required': ['a', 's', 'm'], 'definitions': {'Foo': {'properties': {'bar': {'title': 'Bar', 'type': 'string'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}, 'Color': enum_schema}}
    
    def test_function_with_secretstr(self) -> None:

        def f(x: SecretStr) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0, 'format': 'password', 'type': 'string', 'writeOnly': True}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_v1_secretstr_from_compat_module(self) -> None:
        import pydantic.v1 as pydantic
    
        def f(x: pydantic.SecretStr) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}


class TestMethodToSchema:

    def test_methods_with_no_arguments(self) -> None:

        class Foo:

            def f(self) -> None:
                pass

            @classmethod
            def g(cls) -> None:
                pass

            @staticmethod
            def h() -> None:
                pass
        for method in [Foo().f, Foo.g, Foo.h]:
            schema = callables.parameter_schema(method)
            assert schema.model_dump_for_openapi() == {'properties': {}, 'title': 'Parameters', 'type': 'object', 'definitions': {}}
    
    def test_methods_with_enum_arguments(self) -> None:

        class Color(Enum):
            RED = 'RED'
            GREEN = 'GREEN'
            BLUE = 'BLUE'
    
        class Foo:

            def f(self, color: Color = Color.RED) -> None:
                pass

            @classmethod
            def g(cls, color: Color = Color.RED) -> None:
                pass

            @staticmethod
            def h(color: Color = Color.RED) -> None:
                pass
        for method in [Foo().f, Foo.g, Foo.h]:
            schema = callables.parameter_schema(method)
            expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'color': {'$ref': '#/definitions/Color', 'default': 'RED', 'position': 0, 'title': 'color'}}, 'definitions': {'Color': {'enum': ['RED', 'GREEN', 'BLUE'], 'title': 'Color', 'type': 'string'}}}
            assert schema.model_dump_for_openapi() == expected_schema
    
    def test_methods_with_complex_arguments(self) -> None:

        class Foo:

            def f(self, x: datetime.datetime, y: int = 42, z: Optional[bool] = None) -> None:
                pass

            @classmethod
            def g(cls, x: datetime.datetime, y: int = 42, z: Optional[bool] = None) -> None:
                pass

            @staticmethod
            def h(x: datetime.datetime, y: int = 42, z: Optional[bool] = None) -> None:
                pass
        for method in [Foo().f, Foo.g, Foo.h]:
            schema = callables.parameter_schema(method)
            expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'format': 'date-time', 'position': 0, 'title': 'x', 'type': 'string'}, 'y': {'default': 42, 'position': 1, 'title': 'y', 'type': 'integer'}, 'z': {'default': None, 'position': 2, 'title': 'z', 'anyOf': [{'type': 'boolean'}, {'type': 'null'}]}}, 'required': ['x'], 'definitions': {}}
            assert schema.model_dump_for_openapi() == expected_schema
    
    def test_method_with_kwargs_only(self) -> None:

        def f(*, x: int) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'properties': {'x': {'title': 'x', 'position': 0, 'type': 'integer'}}, 'title': 'Parameters', 'type': 'object', 'definitions': {}, 'required': ['x']}


class TestParseFlowDescriptionToSchema:

    def test_flow_with_args_docstring(self) -> None:

        def f(x: int) -> None:
            """Function f.

            Args:
                x: required argument x
            """
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'description': 'required argument x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_without_docstring(self) -> None:

        def f(x: int) -> None:
            pass
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_without_args_docstring(self) -> None:

        def f(x: int) -> None:
            """Function f."""
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_with_complex_args_docstring(self) -> None:

        def f(x: int, y: str) -> None:
            """Function f.

            Second line of docstring.

            Args:
                x: required argument x
                y (str): required typed argument y
                  with second line

            Returns:
                None: nothing
            """
        schema = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'description': 'required argument x', 'position': 0}, 'y': {'title': 'y', 'description': 'required typed argument y\nwith second line', 'position': 1}}, 'required': ['x', 'y'], 'definitions': {}}


class TestGetCallParameters:

    def test_raises_parameter_bind_with_no_kwargs(self) -> None:

        def dog(x: Any) -> None:
            pass
        with pytest.raises(ParameterBindError):
            callables.get_call_parameters(dog, call_args=(), call_kwargs={})
    
    def test_raises_parameter_bind_with_wrong_kwargs_same_number(self) -> None:

        def dog(x: Any, y: Any) -> None:
            pass
        with pytest.raises(ParameterBindError):
            callables.get_call_parameters(dog, call_args=(), call_kwargs={'x': 2, 'a': 42})
    
    def test_raises_parameter_bind_with_missing_kwargs(self) -> None:

        def dog(x: Any, y: Any) -> None:
            pass
        with pytest.raises(ParameterBindError):
            callables.get_call_parameters(dog, call_args=(), call_kwargs={'x': 2})
    
    def test_raises_parameter_bind_error_with_excess_kwargs(self) -> None:

        def dog(x: Any) -> None:
            pass
        with pytest.raises(ParameterBindError):
            callables.get_call_parameters(dog, call_args=(), call_kwargs={'x': 'y', 'a': 'b'})
    
    def test_raises_parameter_bind_error_with_excess_kwargs_no_args(self) -> None:

        def dog() -> None:
            pass
        with pytest.raises(ParameterBindError):
            callables.get_call_parameters(dog, call_args=(), call_kwargs={'x': 'y'})


class TestExplodeVariadicParameter:

    def test_no_error_if_no_variadic_parameter(self) -> None:

        def foo(a: int, b: int) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2}
        new_params = callables.explode_variadic_parameter(foo, parameters)
        assert parameters == new_params
    
    def test_no_error_if_variadic_parameter_and_kwargs_provided(self) -> None:

        def foo(a: int, b: int, **kwargs: Any) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2, 'kwargs': {'c': 3, 'd': 4}}
        new_params = callables.explode_variadic_parameter(foo, parameters)
        assert new_params == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    
    def test_no_error_if_variadic_parameter_and_no_kwargs_provided(self) -> None:

        def foo(a: int, b: int, **kwargs: Any) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2}
        new_params = callables.explode_variadic_parameter(foo, parameters)
        assert new_params == parameters


class TestCollapseVariadicParameter:

    def test_no_error_if_no_variadic_parameter(self) -> None:

        def foo(a: int, b: int) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2}
        new_params = callables.collapse_variadic_parameters(foo, parameters)
        assert new_params == parameters
    
    def test_no_error_if_variadic_parameter_and_kwargs_provided(self) -> None:

        def foo(a: int, b: int, **kwargs: Any) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        new_params = callables.collapse_variadic_parameters(foo, parameters)
        assert new_params == {'a': 1, 'b': 2, 'kwargs': {'c': 3, 'd': 4}}
    
    def test_params_unchanged_if_variadic_parameter_and_no_kwargs_provided(self) -> None:

        def foo(a: int, b: int, **kwargs: Any) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2}
        new_params = callables.collapse_variadic_parameters(foo, parameters)
        assert new_params == parameters
    
    def test_value_error_raised_if_extra_args_but_no_variadic_parameter(self) -> None:

        def foo(a: int, b: int) -> None:
            pass
        parameters: Dict[str, Any] = {'a': 1, 'b': 2, 'kwargs': {'c': 3, 'd': 4}}
        with pytest.raises(ValueError):
            callables.collapse_variadic_parameters(foo, parameters)


class TestEntrypointToSchema:

    def test_function_not_found(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f():\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        with pytest.raises(ValueError):
            callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:g')
    
    def test_simple_function_with_no_arguments(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f():\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'properties': {}, 'title': 'Parameters', 'type': 'object', 'definitions': {}}
    
    def test_function_with_pydantic_base_model_collisions(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(\n            json,\n            copy,\n            parse_obj,\n            parse_raw,\n            parse_file,\n            from_orm,\n            schema,\n            schema_json,\n            construct,\n            validate,\n            foo,\n        ):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'foo': {'title': 'foo', 'position': 10}, 'json': {'title': 'json', 'position': 0}, 'copy': {'title': 'copy', 'position': 1}, 'parse_obj': {'title': 'parse_obj', 'position': 2}, 'parse_raw': {'title': 'parse_raw', 'position': 3}, 'parse_file': {'title': 'parse_file', 'position': 4}, 'from_orm': {'title': 'from_orm', 'position': 5}, 'schema': {'title': 'schema', 'position': 6}, 'schema_json': {'title': 'schema_json', 'position': 7}, 'construct': {'title': 'construct', 'position': 8}, 'validate': {'title': 'validate', 'position': 9}}, 'required': ['json', 'copy', 'parse_obj', 'parse_raw', 'parse_file', 'from_orm', 'schema', 'schema_json', 'construct', 'validate', 'foo'], 'definitions': {}}
    
    def test_function_with_one_required_argument(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_one_optional_argument(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x=42):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'default': 42, 'position': 0}}, 'definitions': {}}
    
    def test_function_with_one_optional_annotated_argument(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int = 42):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'default': 42, 'type': 'integer', 'position': 0}}, 'definitions': {}}
    
    def test_function_with_two_arguments(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int, y: float = 5.0):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'type': 'integer', 'position': 0}, 'y': {'title': 'y', 'default': 5.0, 'type': 'number', 'position': 1}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_datetime_arguments(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import pendulum\n        import datetime                 \n\n        def f(\n            x: datetime.datetime,\n            y: pendulum.DateTime = pendulum.datetime(2025, 1, 1),\n            z: datetime.timedelta = datetime.timedelta(seconds=5),\n        ):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'format': 'date-time', 'position': 0, 'title': 'x', 'type': 'string'}, 'y': {'default': '2025-01-01T00:00:00Z', 'format': 'date-time', 'position': 1, 'title': 'y', 'type': 'string'}, 'z': {'default': 'PT5S', 'format': 'duration', 'position': 2, 'title': 'z', 'type': 'string'}}, 'required': ['x'], 'definitions': {}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_enum_argument(self, tmp_path: Path) -> None:

        class Color(Enum):
            RED = 'RED'
            GREEN = 'GREEN'
            BLUE = 'BLUE'
        source_code = dedent('\n        from enum import Enum\n\n        class Color(Enum):\n            RED = "RED"\n            GREEN = "GREEN"\n            BLUE = "BLUE"\n\n        def f(x: Color = Color.RED):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'$ref': '#/definitions/Color', 'default': 'RED', 'position': 0, 'title': 'x'}}, 'definitions': {'Color': {'enum': ['RED', 'GREEN', 'BLUE'], 'title': 'Color', 'type': 'string'}}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_generic_arguments(self, tmp_path: Path) -> None:
        source_code = dedent('\n        from typing import List, Dict, Any, Tuple, Union\n\n        def f(\n            a: List[str],\n            b: Dict[str, Any],\n            c: Any,\n            d: Tuple[int, float],\n            e: Union[str, bytes, int],\n        ):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        expected_schema = {'title': 'Parameters', 'type': 'object', 'properties': {'a': {'items': {'type': 'string'}, 'position': 0, 'title': 'a', 'type': 'array'}, 'b': {'position': 1, 'title': 'b', 'type': 'object'}, 'c': {'position': 2, 'title': 'c'}, 'd': {'maxItems': 2, 'minItems': 2, 'position': 3, 'prefixItems': [{'type': 'integer'}, {'type': 'number'}], 'title': 'd', 'type': 'array'}, 'e': {'anyOf': [{'type': 'string'}, {'format': 'binary', 'type': 'string'}, {'type': 'integer'}], 'position': 4, 'title': 'e'}}, 'required': ['a', 'b', 'c', 'd', 'e'], 'definitions': {}}
        assert schema.model_dump_for_openapi() == expected_schema
    
    def test_function_with_user_defined_type(self, tmp_path: Path) -> None:
        source_code = dedent('\n        class Foo:\n            y: int\n\n        def f(x: Foo):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_user_defined_pydantic_model(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import pydantic\n\n        class Foo(pydantic.BaseModel):\n            y: int\n            z: str\n\n        def f(x: Foo):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'definitions': {'Foo': {'properties': {'y': {'title': 'Y', 'type': 'integer'}, 'z': {'title': 'Z', 'type': 'string'}}, 'required': ['y', 'z'], 'title': 'Foo', 'type': 'object'}}, 'properties': {'x': {'$ref': '#/definitions/Foo', 'title': 'x', 'position': 0}}, 'required': ['x'], 'title': 'Parameters', 'type': 'object'}
    
    def test_function_with_pydantic_model_default_across_v1_and_v2(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import pydantic\n\n        class Foo(pydantic.BaseModel):\n            bar: str\n\n        def f(foo: Foo = Foo(bar="baz")):\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'foo': {'$ref': '#/definitions/Foo', 'default': {'bar': 'baz'}, 'position': 0, 'title': 'foo'}}, 'definitions': {'Foo': {'properties': {'bar': {'title': 'Bar', 'type': 'string'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}}}
    
    def test_function_with_complex_args_across_v1_and_v2(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import pydantic\n        import pendulum\n        import datetime\n        from enum import Enum\n        from typing import List\n\n        class Foo(pydantic.BaseModel):\n            bar: str\n\n        class Color(Enum):\n            RED = "RED"\n            GREEN = "GREEN"\n            BLUE = "BLUE"\n\n        def f(\n            a: int,\n            s: List[None],\n            m: Foo,\n            i: int = 0,\n            x: float = 1.0,\n            model: Foo = Foo(bar="bar"),\n            pdt: pendulum.DateTime = pendulum.datetime(2025, 1, 1),\n            pdate: pendulum.Date = pendulum.date(2025, 1, 1),\n            pduration: pendulum.Duration = pendulum.duration(seconds=5),\n            c: Color = Color.BLUE,\n        ) -> None:\n            pass\n        ')
        datetime_schema: Dict[str, Any] = {'title': 'pdt', 'default': '2025-01-01T00:00:00+00:00', 'position': 6, 'type': 'string', 'format': 'date-time'}
        duration_schema: Dict[str, Any] = {'title': 'pduration', 'default': 5.0, 'position': 8, 'type': 'number', 'format': 'time-delta'}
        enum_schema: Dict[str, Any] = {'enum': ['RED', 'GREEN', 'BLUE'], 'title': 'Color', 'type': 'string', 'description': 'An enumeration.'}
        datetime_schema['default'] = '2025-01-01T00:00:00Z'
        duration_schema['default'] = 'PT5S'
        duration_schema['type'] = 'string'
        duration_schema['format'] = 'duration'
        enum_schema.pop('description')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'a': {'position': 0, 'title': 'a', 'type': 'integer'}, 's': {'items': {'type': 'null'}, 'position': 1, 'title': 's', 'type': 'array'}, 'm': {'$ref': '#/definitions/Foo', 'position': 2, 'title': 'm'}, 'i': {'default': 0, 'position': 3, 'title': 'i', 'type': 'integer'}, 'x': {'default': 1.0, 'position': 4, 'title': 'x', 'type': 'number'}, 'model': {'$ref': '#/definitions/Foo', 'default': {'bar': 'bar'}, 'position': 5, 'title': 'model'}, 'pdt': datetime_schema, 'pdate': {'title': 'pdate', 'default': '2025-01-01', 'position': 7, 'type': 'string', 'format': 'date'}, 'pduration': duration_schema, 'c': {'title': 'c', 'default': 'BLUE', 'position': 9, '$ref': '#/definitions/Color'}}, 'required': ['a', 's', 'm'], 'definitions': {'Foo': {'properties': {'bar': {'title': 'Bar', 'type': 'string'}}, 'required': ['bar'], 'title': 'Foo', 'type': 'object'}, 'Color': enum_schema}}
    
    def test_function_with_secretstr(self, tmp_path: Path) -> None:
        source_code = dedent('\n        from pydantic import SecretStr\n\n        def f(x: SecretStr) -> None:\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0, 'format': 'password', 'type': 'string', 'writeOnly': True}}, 'required': ['x'], 'definitions': {}}
    
    def test_function_with_v1_secretstr_from_compat_module(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import pydantic.v1 as pydantic\n\n        def f(x: pydantic.SecretStr) -> None:\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_with_args_docstring(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int) -> None:\n            """Function f.\n\n            Args:\n                x: required argument x\n            """\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'description': 'required argument x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_without_args_docstring(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int) -> None:\n            """Function f."""\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_flow_with_complex_args_docstring(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int, y: str) -> None:\n            """Function f.\n\n            Second line of docstring.\n\n            Args:\n                x: required argument x\n                y (str): required typed argument y\n                  with second line\n\n            Returns:\n                None: nothing\n            """\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'description': 'required argument x', 'position': 0}, 'y': {'title': 'y', 'description': 'required typed argument y\nwith second line', 'position': 1}}, 'required': ['x', 'y'], 'definitions': {}}
    
    def test_does_not_raise_when_missing_dependencies(self, tmp_path: Path) -> None:
        source_code = dedent('\n        import bipitty_boopity\n                             \n        def f(x: int) -> None:\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'x': {'title': 'x', 'position': 0}}, 'required': ['x'], 'definitions': {}}
    
    def test_handles_dynamically_created_models(self, tmp_path: Path) -> None:
        source_code = dedent('\n            from pydantic import BaseModel, create_model, Field\n\n\n            def get_model() -> BaseModel:\n                return create_model(\n                    "MyModel",\n                    param=(\n                        int,\n                        Field(\n                            title="param",\n                            default=1,\n                        ),\n                    ),\n                )\n\n\n            MyModel = get_model()\n\n\n            def f(\n                param: MyModel,\n            ) -> None:\n                pass        \n            ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'title': 'Parameters', 'type': 'object', 'properties': {'param': {'$ref': '#/definitions/MyModel', 'position': 0, 'title': 'param'}}, 'required': ['param'], 'definitions': {'MyModel': {'properties': {'param': {'default': 1, 'title': 'param', 'type': 'integer'}}, 'title': 'MyModel', 'type': 'object'}}}
    
    def test_function_with_kwargs_only(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(\n            *,\n            x: int = 42,\n        ) -> None:\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'properties': {'x': {'title': 'x', 'position': 0, 'type': 'integer', 'default': 42}}, 'title': 'Parameters', 'type': 'object', 'definitions': {}}
    
    def test_function_with_positional_only_args(self, tmp_path: Path) -> None:
        source_code = dedent('\n        def f(x: int = 1, /, y: int = 2, z: int = 3) -> None:\n            pass\n        ')
        tmp_path.joinpath('test.py').write_text(source_code)
        schema = callables.parameter_schema_from_entrypoint(f'{tmp_path}/test.py:f')
        assert schema.model_dump_for_openapi() == {'properties': {'x': {'title': 'x', 'position': 0, 'default': 1}, 'y': {'title': 'y', 'position': 1, 'default': 2}, 'z': {'title': 'z', 'position': 2, 'default': 3}}, 'title': 'Parameters', 'type': 'object', 'definitions': {}}
