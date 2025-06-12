import pprint
import avro.schema
import yaml
from urllib.parse import urlsplit
from typing import Any, List, Set, Optional


class ValidationException(Exception):
    pass


INT_MIN_VALUE: int = -(1 << 31)
INT_MAX_VALUE: int = (1 << 31) - 1
LONG_MIN_VALUE: int = -(1 << 63)
LONG_MAX_VALUE: int = (1 << 63) - 1


def validate(
    expected_schema: avro.schema.Schema,
    datum: Any,
    identifiers: Optional[List[str]] = None,
    strict: bool = False,
    foreign_properties: Optional[Set[str]] = None
) -> bool:
    if identifiers is None:
        identifiers = []
    if foreign_properties is None:
        foreign_properties = set()
    try:
        return validate_ex(
            expected_schema,
            datum,
            identifiers,
            strict=strict,
            foreign_properties=foreign_properties
        )
    except ValidationException:
        return False


def indent(v: str, nolead: bool = False) -> str:
    if nolead:
        lines = v.splitlines()
        return lines[0] + ''.join(['  ' + l + '\n' for l in lines[1:]]).rstrip()
    else:
        return '\n'.join(['  ' + l for l in v.splitlines()])


def friendly(v: avro.schema.Schema) -> str:
    if isinstance(v, avro.schema.NamedSchema):
        return v.name
    if isinstance(v, avro.schema.ArraySchema):
        return f'array of <{friendly(v.items)}>'
    elif isinstance(v, avro.schema.PrimitiveSchema):
        return v.type
    elif isinstance(v, avro.schema.UnionSchema):
        return ' or '.join([friendly(s) for s in v.schemas])
    else:
        return str(v)


def multi(v: str, q: str = '') -> str:
    if '\n' in v:
        return f'{q}{v}{q}\n'
    else:
        return f'{q}{v}{q}'


def vpformat(datum: Any) -> str:
    a = pprint.pformat(datum)
    if len(a) > 160:
        a = a[0:160] + '[...]'
    return a


def validate_ex(
    expected_schema: avro.schema.Schema,
    datum: Any,
    identifiers: Set[str] = set(),
    strict: bool = False,
    foreign_properties: Set[str] = set()
) -> bool:
    """Determine if a python datum is an instance of a schema."""
    schema_type: str = expected_schema.type
    if schema_type == 'null':
        if datum is None:
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not null')
    elif schema_type == 'boolean':
        if isinstance(datum, bool):
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not boolean')
    elif schema_type == 'string':
        if isinstance(datum, str):
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not string')
    elif schema_type == 'bytes':
        if isinstance(datum, bytes):
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not bytes')
    elif schema_type == 'int':
        if isinstance(datum, int) and INT_MIN_VALUE <= datum <= INT_MAX_VALUE:
            return True
        else:
            raise ValidationException(f'`{vpformat(datum)}` is not int')
    elif schema_type == 'long':
        if isinstance(datum, int) and LONG_MIN_VALUE <= datum <= LONG_MAX_VALUE:
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not long')
    elif schema_type in ['float', 'double']:
        if isinstance(datum, (int, float)):
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not float or double')
    elif schema_type == 'fixed':
        if isinstance(datum, bytes) and len(datum) == expected_schema.size:
            return True
        else:
            raise ValidationException(f'the value `{vpformat(datum)}` is not fixed')
    elif schema_type == 'enum':
        if expected_schema.name == 'Any':
            if datum is not None:
                return True
            else:
                raise ValidationException('Any type must be non-null')
        if datum in expected_schema.symbols:
            return True
        else:
            symbols = "', '".join(expected_schema.symbols)
            raise ValidationException(
                f"the value `{vpformat(datum)}`\n is not a valid symbol in enum {expected_schema.name}, expected one of '{symbols}'"
            )
    elif schema_type == 'array':
        if isinstance(datum, list):
            for i, d in enumerate(datum):
                try:
                    validate_ex(expected_schema.items, d, identifiers, strict=strict, foreign_properties=foreign_properties)
                except ValidationException as v:
                    raise ValidationException(f'At position {i}\n{indent(str(v))}')
            return True
        else:
            friendly_items = friendly(expected_schema.items)
            raise ValidationException(f'the value `{vpformat(datum)}` is not a list, expected list of {friendly_items}')
    elif schema_type == 'map':
        if (
            isinstance(datum, dict)
            and all(isinstance(k, str) for k in datum.keys())
            and all(validate(expected_schema.values, v, strict=strict) for v in datum.values())
        ):
            return True
        else:
            raise ValidationException(f'`{vpformat(datum)}` is not a valid map value, expected\n {vpformat(expected_schema.values)}')
    elif schema_type in ['union', 'error_union']:
        if any(validate(s, datum, identifiers, strict=strict) for s in expected_schema.schemas):
            return True
        else:
            errors: List[str] = []
            for s in expected_schema.schemas:
                try:
                    validate_ex(s, datum, identifiers, strict=strict, foreign_properties=foreign_properties)
                except ValidationException as e:
                    errors.append(str(e))
            error_msgs = '\n'.join([
                f"- {friendly(expected_schema.schemas[i])}, but\n {indent(multi(errors[i]))}"
                for i in range(len(expected_schema.schemas))
            ])
            raise ValidationException(
                f'the value {multi(vpformat(datum), "`")} is not a valid type in the union, expected one of:\n{error_msgs}'
            )
    elif schema_type in ['record', 'error', 'request']:
        if not isinstance(datum, dict):
            raise ValidationException(f'`{vpformat(datum)}`\n is not a dict')
        errors: List[str] = []
        for f in expected_schema.fields:
            if f.name in datum:
                fieldval = datum[f.name]
            else:
                fieldval = f.default
            try:
                validate_ex(f.type, fieldval, identifiers, strict=strict, foreign_properties=foreign_properties)
            except ValidationException as v:
                if f.name not in datum:
                    errors.append(f'missing required field `{f.name}`')
                else:
                    errors.append(f'could not validate field `{f.name}` because\n{multi(indent(str(v)))}')
        if strict:
            for d in datum:
                if not any(d == f.name for f in expected_schema.fields):
                    if d not in identifiers and d not in foreign_properties and not d.startswith(('@', '$')):
                        split = urlsplit(d)
                        if split.scheme:
                            errors.append(
                                f'could not validate extension field `{d}` because it is not recognized and strict is True.  Did you include a $schemas section?'
                            )
                        else:
                            valid_fields = ', '.join(f.name for f in expected_schema.fields)
                            errors.append(
                                f'could not validate field `{d}` because it is not recognized and strict is True, valid fields are: {valid_fields}'
                            )
        if errors:
            raise ValidationException('\n'.join(errors))
        else:
            return True
    raise ValidationException(f'Unrecognized schema_type {schema_type}')
