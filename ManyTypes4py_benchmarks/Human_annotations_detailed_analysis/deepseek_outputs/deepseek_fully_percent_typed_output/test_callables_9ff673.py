import datetime
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable

import pendulum
import pydantic.version
import pytest
from pydantic import SecretStr, BaseModel
from prefect.exceptions import ParameterBindError
from prefect.utilities import callables


class TestFunctionToSchema:
    def test_simple_function_with_no_arguments(self) -> None:
        def f() -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "properties": {},
            "title": "Parameters",
            "type": "object",
            "definitions": {},
        }

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

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "json": {"title": "json", "position": 0},
                "copy": {"title": "copy", "position": 1},
                "parse_obj": {"title": "parse_obj", "position": 2},
                "parse_raw": {"title": "parse_raw", "position": 3},
                "parse_file": {"title": "parse_file", "position": 4},
                "from_orm": {"title": "from_orm", "position": 5},
                "schema": {"title": "schema", "position": 6},
                "schema_json": {"title": "schema_json", "position": 7},
                "construct": {"title": "construct", "position": 8},
                "validate": {"title": "validate", "position": 9},
                "foo": {"title": "foo", "position": 10},
            },
            "required": [
                "json",
                "copy",
                "parse_obj",
                "parse_raw",
                "parse_file",
                "from_orm",
                "schema",
                "schema_json",
                "construct",
                "validate",
                "foo",
            ],
            "definitions": {},
        }

    def test_function_with_one_required_argument(self) -> None:
        def f(x: Any) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {"x": {"title": "x", "position": 0}},
            "required": ["x"],
            "definitions": {},
        }

    def test_function_with_one_optional_argument(self) -> None:
        def f(x: int = 42) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {"x": {"default": 42, "position": 0, "title": "x"}},
            "definitions": {},
        }

    def test_function_with_one_optional_annotated_argument(self) -> None:
        def f(x: int = 42) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {
                    "default": 42,
                    "position": 0,
                    "title": "x",
                    "type": "integer",
                }
            },
            "definitions": {},
        }

    def test_function_with_two_arguments(self) -> None:
        def f(x: int, y: float = 5.0) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {"title": "x", "type": "integer", "position": 0},
                "y": {"title": "y", "default": 5.0, "type": "number", "position": 1},
            },
            "required": ["x"],
            "definitions": {},
        }

    def test_function_with_datetime_arguments(self) -> None:
        def f(
            x: datetime.datetime,
            y: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
            z: datetime.timedelta = datetime.timedelta(seconds=5),
        ) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        expected_schema: Dict[str, Any] = {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {
                    "format": "date-time",
                    "position": 0,
                    "title": "x",
                    "type": "string",
                },
                "y": {
                    "default": "2025-01-01T00:00:00Z",
                    "format": "date-time",
                    "position": 1,
                    "title": "y",
                    "type": "string",
                },
                "z": {
                    "default": "PT5S",
                    "format": "duration",
                    "position": 2,
                    "title": "z",
                    "type": "string",
                },
            },
            "required": ["x"],
            "definitions": {},
        }
        assert schema.model_dump_for_openapi() == expected_schema

    def test_function_with_enum_argument(self) -> None:
        class Color(Enum):
            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        def f(x: Color = "RED") -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)

        expected_schema: Dict[str, Any] = {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {
                    "$ref": "#/definitions/Color",
                    "default": "RED",
                    "position": 0,
                    "title": "x",
                }
            },
            "definitions": {
                "Color": {
                    "enum": ["RED", "GREEN", "BLUE"],
                    "title": "Color",
                    "type": "string",
                }
            },
        }

        assert schema.model_dump_for_openapi() == expected_schema

    def test_function_with_generic_arguments(self) -> None:
        def f(
            a: List[str],
            b: Dict[str, Any],
            c: Any,
            d: Tuple[int, float],
            e: Union[str, bytes, int],
        ) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)

        expected_schema: Dict[str, Any] = {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "a": {
                    "items": {"type": "string"},
                    "position": 0,
                    "title": "a",
                    "type": "array",
                },
                "b": {"position": 1, "title": "b", "type": "object"},
                "c": {"position": 2, "title": "c"},
                "d": {
                    "maxItems": 2,
                    "minItems": 2,
                    "position": 3,
                    "prefixItems": [{"type": "integer"}, {"type": "number"}],
                    "title": "d",
                    "type": "array",
                },
                "e": {
                    "anyOf": [
                        {"type": "string"},
                        {"format": "binary", "type": "string"},
                        {"type": "integer"},
                    ],
                    "position": 4,
                    "title": "e",
                },
            },
            "required": ["a", "b", "c", "d", "e"],
            "definitions": {},
        }

        assert schema.model_dump_for_openapi() == expected_schema

    def test_function_with_user_defined_type(self) -> None:
        class Foo:
            y: int

        def f(x: Foo) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "definitions": {},
            "title": "Parameters",
            "type": "object",
            "properties": {"x": {"title": "x", "position": 0}},
            "required": ["x"],
        }

    def test_function_with_user_defined_pydantic_model(self) -> None:
        class Foo(BaseModel):
            y: int
            z: str

        def f(x: Foo) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "definitions": {
                "Foo": {
                    "properties": {
                        "y": {"title": "Y", "type": "integer"},
                        "z": {"title": "Z", "type": "string"},
                    },
                    "required": ["y", "z"],
                    "title": "Foo",
                    "type": "object",
                }
            },
            "properties": {
                "x": {
                    "$ref": "#/definitions/Foo",
                    "title": "x",
                    "position": 0,
                }
            },
            "required": ["x"],
            "title": "Parameters",
            "type": "object",
        }

    def test_function_with_pydantic_model_default_across_v1_and_v2(self) -> None:
        import pydantic

        class Foo(pydantic.BaseModel):
            bar: str

        def f(foo: Foo = Foo(bar="baz")) -> None:
            ...

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "foo": {
                    "$ref": "#/definitions/Foo",
                    "default": {"bar": "baz"},
                    "position": 0,
                    "title": "foo",
                }
            },
            "definitions": {
                "Foo": {
                    "properties": {"bar": {"title": "Bar", "type": "string"}},
                    "required": ["bar"],
                    "title": "Foo",
                    "type": "object",
                }
            },
        }

    def test_function_with_complex_args_across_v1_and_v2(self) -> None:
        import pydantic

        class Foo(pydantic.BaseModel):
            bar: str

        class Color(Enum):
            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        def f(
            a: int,
            s: List[None],
            m: Foo,
            i: int = 0,
            x: float = 1.0,
            model: Foo = Foo(bar="bar"),
            pdt: pendulum.DateTime = pendulum.datetime(2025, 1, 1),
            pdate: pendulum.Date = pendulum.date(2025, 1, 1),
            pduration: pendulum.Duration = pendulum.duration(seconds=5),
            c: Color = Color.BLUE,
        ) -> None:
            ...

        datetime_schema: Dict[str, Any] = {
            "title": "pdt",
            "default": "2025-01-01T00:00:00+00:00",
            "position": 6,
            "type": "string",
            "format": "date-time",
        }
        duration_schema: Dict[str, Any] = {
            "title": "pduration",
            "default": 5.0,
            "position": 8,
            "type": "number",
            "format": "time-delta",
        }
        enum_schema: Dict[str, Any] = {
            "enum": ["RED", "GREEN", "BLUE"],
            "title": "Color",
            "type": "string",
            "description": "An enumeration.",
        }

        datetime_schema["default"] = "2025-01-01T00:00:00Z"
        duration_schema["default"] = "PT5S"
        duration_schema["type"] = "string"
        duration_schema["format"] = "duration"
        enum_schema.pop("description")

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "a": {"position": 0, "title": "a", "type": "integer"},
                "s": {
                    "items": {"type": "null"},
                    "position": 1,
                    "title": "s",
                    "type": "array",
                },
                "m": {
                    "$ref": "#/definitions/Foo",
                    "position": 2,
                    "title": "m",
                },
                "i": {"default": 0, "position": 3, "title": "i", "type": "integer"},
                "x": {"default": 1.0, "position": 4, "title": "x", "type": "number"},
                "model": {
                    "$ref": "#/definitions/Foo",
                    "default": {"bar": "bar"},
                    "position": 5,
                    "title": "model",
                },
                "pdt": datetime_schema,
                "pdate": {
                    "title": "pdate",
                    "default": "2025-01-01",
                    "position": 7,
                    "type": "string",
                    "format": "date",
                },
                "pduration": duration_schema,
                "c": {
                    "title": "c",
                    "default": "BLUE",
                    "position": 9,
                    "$ref": "#/definitions/Color",
                },
            },
            "required": ["a", "s", "m"],
            "definitions": {
                "Foo": {
                    "properties": {"bar": {"title": "Bar", "type": "string"}},
                    "required": ["bar"],
                    "title": "Foo",
                    "type": "object",
                },
                "Color": enum_schema,
            },
        }

    def test_function_with_secretstr(self) -> None:
        def f(x: SecretStr) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {
                    "title": "x",
                    "position": 0,
                    "format": "password",
                    "type": "string",
                    "writeOnly": True,
                },
            },
            "required": ["x"],
            "definitions": {},
        }

    def test_function_with_v1_secretstr_from_compat_module(self) -> None:
        import pydantic.v1 as pydantic

        def f(x: pydantic.SecretStr) -> None:
            pass

        schema: BaseModel = callables.parameter_schema(f)
        assert schema.model_dump_for_openapi() == {
            "title": "Parameters",
            "type": "object",
            "properties": {
                "x": {
                    "title": "x",
                    "position": 0,
                },
            },
            "required": ["x"],
            "definitions": {},
        }


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
            schema: BaseModel = callables.parameter_schema(method)
            assert schema.model_dump_for_openapi() == {
                "properties": {},
                "title": "Parameters",
                "type": "object",
                "definitions": {},
            }

    def test_methods_with_enum_arguments(self) -> None:
        class Color(Enum):
            RED = "RED"
            GREEN = "GREEN"
            BLUE = "BLUE"

        class Foo:
            def f(self, color: Color = "RED") -> None:
                pass

            @classmethod
            def g(cls, color: Color = "RED") -> None:
                pass

            @staticmethod
            def h(color: Color = "RED") -> None:
                pass

        for method in [Foo().f, Foo.g, Foo.h]:
            schema: BaseModel = callables.parameter_schema(method)

            expected_schema: Dict[str, Any] = {
                "title": "Parameters",
                "type": "object",
                "properties": {
                    "color": {
                        "$ref": "#/definitions/Color",
                        "default": "RED",
                        "position": 0,
                        "title": "color",
                    }
                },
                "definitions": {
                    "Color": {
                        "enum": ["RED", "GREEN", "BLUE"],
                        "title": "Color",
                        "type": "string",
                    }
                },
            }

            assert schema.model_dump_for_openapi() == expected_schema

    def test_methods_with_complex_arguments(self) -> None:
        class Foo:
            def f(
                self, x: datetime.datetime, y: int = 42, z: Optional[bool] = None
            ) -> None:
                pass

            @classmethod
            def g(
                cls, x: datetime.datetime, y: int = 42, z: Optional[bool] = None
            ) -> None:
                pass

            @staticmethod
            def h(
                x: datetime.datetime, y: int = 42, z: Optional[bool] = None
            ) -> None:
                pass

        for method in [Foo().f, Foo.g, Foo.h]:
            schema: BaseModel = callables.parameter_schema(method)
            expected_schema: Dict[str, Any] = {
                "