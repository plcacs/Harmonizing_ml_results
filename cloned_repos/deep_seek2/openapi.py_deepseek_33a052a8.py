# Zulip's OpenAPI-based API documentation system is documented at
#   https://zulip.readthedocs.io/en/latest/documentation/api.html
#
# This file contains helper functions to interact with the OpenAPI
# definitions and validate that Zulip's implementation matches what is
# described in our documentation.

import json
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, Dict, List, Optional, Set, Tuple, Union

import orjson
from openapi_core import OpenAPI
from openapi_core.protocols import Request, Response
from openapi_core.testing import MockRequest, MockResponse
from openapi_core.validation.exceptions import ValidationError as OpenAPIValidationError
from pydantic import BaseModel

OPENAPI_SPEC_PATH: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../openapi/zulip.yaml")
)

# A list of endpoint-methods such that the endpoint
# has documentation but not with this particular method.
EXCLUDE_UNDOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = {
    ("/users", "patch"),
}
# Consists of endpoints with some documentation remaining.
# These are skipped but return true as the validator cannot exclude objects
EXCLUDE_DOCUMENTED_ENDPOINTS: Set[Tuple[str, str]] = set()


# Most of our code expects allOf to be preprocessed away because that is what
# yamole did.  Its algorithm for doing so is not standards compliant, but we
# replicate it here.
def naively_merge(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    ret: Dict[str, object] = a.copy()
    for key, b_value in b.items():
        if key == "example" or key not in ret:
            ret[key] = b_value
            continue
        a_value = ret[key]
        if isinstance(b_value, list):
            assert isinstance(a_value, list)
            ret[key] = a_value + b_value
        elif isinstance(b_value, dict):
            assert isinstance(a_value, dict)
            ret[key] = naively_merge(a_value, b_value)
    return ret


def naively_merge_allOf(obj: object) -> object:
    if isinstance(obj, dict):
        return naively_merge_allOf_dict(obj)
    elif isinstance(obj, list):
        return list(map(naively_merge_allOf, obj))
    else:
        return obj


def naively_merge_allOf_dict(obj: Dict[str, object]) -> Dict[str, object]:
    if "allOf" in obj:
        ret = obj.copy()
        subschemas = ret.pop("allOf")
        ret = naively_merge_allOf_dict(ret)
        assert isinstance(subschemas, list)
        for subschema in subschemas:
            assert isinstance(subschema, dict)
            ret = naively_merge(ret, naively_merge_allOf_dict(subschema))
        return ret
    return {key: naively_merge_allOf(value) for key, value in obj.items()}


class OpenAPISpec:
    def __init__(self, openapi_path: str) -> None:
        self.openapi_path: str = openapi_path
        self.mtime: Optional[float] = None
        self._openapi: Dict[str, Any] = {}
        self._endpoints_dict: Dict[str, str] = {}
        self._spec: Optional[OpenAPI] = None

    def check_reload(self) -> None:
        # Because importing yaml takes significant time, and we only
        # use python-yaml for our API docs, importing it lazily here
        # is a significant optimization to `manage.py` startup.
        #
        # There is a bit of a race here...we may have two processes
        # accessing this module level object and both trying to
        # populate self.data at the same time.  Hopefully this will
        # only cause some extra processing at startup and not data
        # corruption.

        import yaml
        from jsonref import JsonRef

        with open(self.openapi_path) as f:
            mtime = os.fstat(f.fileno()).st_mtime
            # Using == rather than >= to cover the corner case of users placing an
            # earlier version than the current one
            if self.mtime == mtime:
                return

            openapi = yaml.load(f, Loader=yaml.CSafeLoader)

        spec = OpenAPI.from_dict(openapi)
        self._spec = spec
        self._openapi = naively_merge_allOf_dict(JsonRef.replace_refs(openapi))
        self.create_endpoints_dict()
        self.mtime = mtime

    def create_endpoints_dict(self) -> None:
        # Algorithm description:
        # We have 2 types of endpoints
        # 1.with path arguments 2. without path arguments
        # In validate_against_openapi_schema we directly check
        # if we have a without path endpoint, since it does not
        # require regex. Hence they are not part of the regex dict
        # and now we are left with only:
        # endpoint with path arguments.
        # Now for this case, the regex has been created carefully,
        # numeric arguments are matched with [0-9] only and
        # emails are matched with their regex. This is why there are zero
        # collisions. Hence if this regex matches
        # an incorrect endpoint then there is some backend problem.
        # For example if we have users/{name}/presence then it will
        # conflict with users/me/presence even in the backend.
        # Care should be taken though that if we have special strings
        # such as email they must be substituted with proper regex.

        email_regex: str = r"([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})"
        self._endpoints_dict = {}
        for endpoint in self._openapi["paths"]:
            if "{" not in endpoint:
                continue
            path_regex: str = "^" + endpoint + "$"
            # Numeric arguments have id at their end
            # so find such arguments and replace them with numeric
            # regex
            path_regex = re.sub(r"{[^}]*id}", r"[0-9]*", path_regex)
            # Email arguments end with email
            path_regex = re.sub(r"{[^}]*email}", email_regex, path_regex)
            # All other types of arguments are supposed to be
            # all-encompassing string.
            path_regex = re.sub(r"{[^}]*}", r"[^\/]*", path_regex)
            path_regex = path_regex.replace(r"/", r"\/")
            self._endpoints_dict[path_regex] = endpoint

    def openapi(self) -> Dict[str, Any]:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the parsed data.
        """
        self.check_reload()
        assert len(self._openapi) > 0
        return self._openapi

    def endpoints_dict(self) -> Dict[str, str]:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the parsed data.
        """
        self.check_reload()
        assert len(self._endpoints_dict) > 0
        return self._endpoints_dict

    def spec(self) -> OpenAPI:
        """Reload the OpenAPI file if it has been modified after the last time
        it was read, and then return the openapi_core validator object. Similar
        to preceding functions. Used for proper access to OpenAPI objects.
        """
        self.check_reload()
        assert self._spec is not None
        return self._spec


class SchemaError(Exception):
    pass


openapi_spec: OpenAPISpec = OpenAPISpec(OPENAPI_SPEC_PATH)


def get_schema(endpoint: str, method: str, status_code: str) -> Dict[str, Any]:
    if len(status_code) == 3 and (
        "oneOf"
        in openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][status_code][
            "content"
        ]["application/json"]["schema"]
    ):
        # Currently at places where multiple schemas are defined they only
        # differ in example so either can be used.
        status_code += "_0"
    if len(status_code) == 3:
        schema = openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][
            status_code
        ]["content"]["application/json"]["schema"]
        return schema
    else:
        subschema_index = int(status_code[4])
        status_code = status_code[0:3]
        schema = openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][
            status_code
        ]["content"]["application/json"]["schema"]["oneOf"][subschema_index]
        return schema


def get_openapi_fixture(
    endpoint: str, method: str, status_code: str = "200"
) -> List[Dict[str, Any]]:
    """Fetch a fixture from the full spec object."""
    if "example" not in get_schema(endpoint, method, status_code):
        return openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][status_code][
            "content"
        ]["application/json"]["examples"].values()
    return [
        {
            "description": get_schema(endpoint, method, status_code)["description"],
            "value": get_schema(endpoint, method, status_code)["example"],
        }
    ]


def get_curl_include_exclude(endpoint: str, method: str) -> List[Dict[str, Any]]:
    """Fetch all the kinds of parameters required for curl examples."""
    if (
        "x-curl-examples-parameters"
        not in openapi_spec.openapi()["paths"][endpoint][method.lower()]
    ):
        return [{"type": "exclude", "parameters": {"enum": [""]}}]
    return openapi_spec.openapi()["paths"][endpoint][method.lower()]["x-curl-examples-parameters"][
        "oneOf"
    ]


def check_requires_administrator(endpoint: str, method: str) -> bool:
    """Fetch if the endpoint requires admin config."""
    return openapi_spec.openapi()["paths"][endpoint][method.lower()].get(
        "x-requires-administrator", False
    )


def check_additional_imports(endpoint: str, method: str) -> Optional[List[str]]:
    """Fetch the additional imports required for an endpoint."""
    return openapi_spec.openapi()["paths"][endpoint][method.lower()].get(
        "x-python-examples-extra-imports", None
    )


def get_responses_description(endpoint: str, method: str) -> str:
    """Fetch responses description of an endpoint."""
    return openapi_spec.openapi()["paths"][endpoint][method.lower()].get(
        "x-response-description", ""
    )


def get_parameters_description(endpoint: str, method: str) -> str:
    """Fetch parameters description of an endpoint."""
    return openapi_spec.openapi()["paths"][endpoint][method.lower()].get(
        "x-parameter-description", ""
    )


def generate_openapi_fixture(endpoint: str, method: str) -> List[str]:
    """Generate fixture to be rendered"""
    fixture: List[str] = []
    for status_code in sorted(
        openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"]
    ):
        if (
            "oneOf"
            in openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][status_code][
                "content"
            ]["application/json"]["schema"]
        ):
            subschema_count = len(
                openapi_spec.openapi()["paths"][endpoint][method.lower()]["responses"][status_code][
                    "content"
                ]["application/json"]["schema"]["oneOf"]
            )
        else:
            subschema_count = 1
        for subschema_index in range(subschema_count):
            if subschema_count != 1:
                subschema_status_code = status_code + "_" + str(subschema_index)
            else:
                subschema_status_code = status_code
            fixture_dict = get_openapi_fixture(endpoint, method, subschema_status_code)
            for example in fixture_dict:
                fixture_json = json.dumps(
                    example["value"], indent=4, sort_keys=True, separators=(",", ": ")
                )
                if "description" in example:
                    fixture.extend(example["description"].strip().splitlines())
                fixture.append("