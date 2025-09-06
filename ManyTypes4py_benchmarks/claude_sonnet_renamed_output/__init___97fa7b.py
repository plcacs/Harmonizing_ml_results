"""Component to allow running Python scripts."""
from collections.abc import Callable, Mapping, Sequence
import datetime
import glob
import logging
from numbers import Number
import operator
import os
import time
import types
from typing import Any, Dict, Optional
from RestrictedPython import compile_restricted_exec, limited_builtins, safe_builtins, utility_builtins
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
import voluptuous as vol
from homeassistant.const import CONF_DESCRIPTION, CONF_NAME, SERVICE_RELOAD
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers.service import async_set_service_schema
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass
from homeassistant.util import dt as dt_util, raise_if_invalid_filename
from homeassistant.util.yaml.loader import load_yaml_dict

_LOGGER: logging.Logger = logging.getLogger(__name__)
DOMAIN: str = 'python_script'
FOLDER: str = 'python_scripts'
CONFIG_SCHEMA: vol.Schema = vol.Schema({DOMAIN: vol.Schema(dict)}, extra=vol.ALLOW_EXTRA)
ALLOWED_HASS: set[str] = {'bus', 'services', 'states'}
ALLOWED_EVENTBUS: set[str] = {'fire'}
ALLOWED_STATEMACHINE: set[str] = {
    'entity_ids',
    'all',
    'get',
    'is_state',
    'is_state_attr',
    'remove',
    'set',
}
ALLOWED_SERVICEREGISTRY: set[str] = {'services', 'has_service', 'call'}
ALLOWED_TIME: set[str] = {
    'sleep',
    'strftime',
    'strptime',
    'gmtime',
    'localtime',
    'ctime',
    'time',
    'mktime',
}
ALLOWED_DATETIME: set[str] = {
    'date',
    'time',
    'datetime',
    'timedelta',
    'tzinfo',
}
ALLOWED_DT_UTIL: set[str] = {
    'utcnow',
    'now',
    'as_utc',
    'as_timestamp',
    'as_local',
    'utc_from_timestamp',
    'start_of_local_day',
    'parse_datetime',
    'parse_date',
    'get_age',
}
CONF_FIELDS: str = 'fields'


class ScriptError(HomeAssistantError):
    """When a script error occurs."""


def func_yw77nset(hass: HomeAssistant, config: ConfigType) -> bool:
    """Initialize the Python script component."""
    path: str = hass.config.path(FOLDER)
    if not os.path.isdir(path):
        _LOGGER.warning('Folder %s not found in configuration folder', FOLDER)
        return False
    discover_scripts(hass)

    def func_xmibxg7s(call: ServiceCall) -> None:
        """Handle reload service calls."""
        discover_scripts(hass)

    hass.services.register(DOMAIN, SERVICE_RELOAD, func_xmibxg7s)
    return True


def discover_scripts(hass: HomeAssistant) -> None:
    """Discover python scripts in folder."""
    path: str = hass.config.path(FOLDER)
    if not os.path.isdir(path):
        _LOGGER.warning('Folder %s not found in configuration folder', FOLDER)
        return

    def func_rz7ufoxl(call: ServiceCall) -> Any:
        """Handle python script service calls."""
        return execute_script(hass, call.service, call.data, call.return_response)

    existing_services = list(hass.services.services.get(DOMAIN, {}).keys())
    for existing_service in existing_services:
        if existing_service == SERVICE_RELOAD:
            continue
        hass.services.remove(DOMAIN, existing_service)
    services_yaml: str = os.path.join(path, 'services.yaml')
    if os.path.exists(services_yaml):
        services_dict: Dict[str, Any] = load_yaml_dict(services_yaml)
    else:
        services_dict = {}
    for fil in glob.iglob(os.path.join(path, '*.py')):
        name: str = os.path.splitext(os.path.basename(fil))[0]
        hass.services.register(
            DOMAIN,
            name,
            python_script_service_handler,
            supports_response=SupportsResponse.OPTIONAL,
        )
        service_desc: Dict[str, Any] = {
            CONF_NAME: services_dict.get(name, {}).get('name', name),
            CONF_DESCRIPTION: services_dict.get(name, {}).get('description', ''),
            CONF_FIELDS: services_dict.get(name, {}).get('fields', {}),
        }
        async_set_service_schema(hass, DOMAIN, name, service_desc)


IOPERATOR_TO_OPERATOR: Dict[str, Callable[[Any, Any], Any]] = {
    '%=': operator.mod,
    '&=': operator.and_,
    '**=': operator.pow,
    '*=': operator.mul,
    '+=': operator.add,
    '-=': operator.sub,
    '//=': operator.floordiv,
    '/=': operator.truediv,
    '<<=': operator.lshift,
    '>>=': operator.rshift,
    '@=': operator.matmul,
    '^=': operator.xor,
    '|=': operator.or_,
}


def guarded_import(
    name: str,
    globals: Optional[Dict[str, Any]] = None,
    locals: Optional[Dict[str, Any]] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    """Guard imports."""
    if name == '_strptime':
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f'Not allowed to import {name}')


def guarded_aug_assign(
    op: str, target: Any, operand: Any
) -> Any:
    """Implement augmented-assign (+=, -=, etc.) operators for restricted code.

    See RestrictedPython's `visit_AugAssign` for details.
    """
    if not isinstance(target, (list, Number, str)):
        raise ScriptError(
            f'The {op!r} operation is not allowed on a {type(target)}'
        )
    op_fun: Optional[Callable[[Any, Any], Any]] = IOPERATOR_TO_OPERATOR.get(op)
    if not op_fun:
        raise ScriptError(f'The {op!r} operation is not allowed')
    return op_fun(target, operand)


@bind_hass
def execute_script(
    hass: HomeAssistant,
    name: str,
    data: Optional[Dict[str, Any]] = None,
    return_response: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute a script."""
    filename: str = f'{name}.py'
    raise_if_invalid_filename(filename)
    with open(hass.config.path(FOLDER, filename), encoding='utf8') as fil:
        source: str = fil.read()
    return execute(
        hass, filename, source, data, return_response=return_response
    )


@bind_hass
def execute(
    hass: HomeAssistant,
    filename: str,
    source: str,
    data: Optional[Dict[str, Any]] = None,
    return_response: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute Python source."""
    compiled = compile_restricted_exec(source, filename=filename)
    if compiled.errors:
        _LOGGER.error(
            'Error loading script %s: %s',
            filename,
            ', '.join(compiled.errors),
        )
        return None
    if compiled.warnings:
        _LOGGER.warning(
            'Warning loading script %s: %s',
            filename,
            ', '.join(compiled.warnings),
        )

    def restricted_getattr(obj: Any, name: str, default: Any = None) -> Any:
        """Restricted method to get attributes."""
        if name.startswith('async_'):
            raise ScriptError('Not allowed to access async methods')
        if (
            obj is hass
            and name not in ALLOWED_HASS
            or obj is hass.bus
            and name not in ALLOWED_EVENTBUS
            or obj is hass.states
            and name not in ALLOWED_STATEMACHINE
            or obj is hass.services
            and name not in ALLOWED_SERVICEREGISTRY
            or obj is dt_util
            and name not in ALLOWED_DT_UTIL
            or obj is datetime
            and name not in ALLOWED_DATETIME
            or isinstance(obj, TimeWrapper)
            and name not in ALLOWED_TIME
        ):
            raise ScriptError(
                f'Not allowed to access {obj.__class__.__name__}.{name}'
            )
        return getattr(obj, name, default)

    extra_builtins: Dict[str, Any] = {
        '__import__': guarded_import,
        'datetime': datetime,
        'sorted': sorted,
        'time': TimeWrapper(),
        'dt_util': dt_util,
        'min': min,
        'max': max,
        'sum': sum,
        'any': any,
        'all': all,
        'enumerate': enumerate,
    }
    builtins = safe_builtins.copy()
    builtins.update(utility_builtins)
    builtins.update(limited_builtins)
    builtins.update(extra_builtins)
    logger: logging.Logger = logging.getLogger(f'{__name__}.{filename}')
    restricted_globals: Dict[str, Any] = {
        '__builtins__': builtins,
        '_print_': StubPrinter,
        '_getattr_': restricted_getattr,
        '_write_': full_write_guard,
        '_getiter_': iter,
        '_getitem_': default_guarded_getitem,
        '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
        '_unpack_sequence_': guarded_unpack_sequence,
        '_inplacevar_': guarded_aug_assign,
        'hass': hass,
        'data': data or {},
        'logger': logger,
        'output': {},
    }
    try:
        _LOGGER.info('Executing %s: %s', filename, data)
        exec(compiled.code, restricted_globals)
        _LOGGER.debug(
            'Output of python_script: `%s`:\n%s',
            filename,
            restricted_globals['output'],
        )
        if not isinstance(restricted_globals['output'], dict):
            output_type = type(restricted_globals['output'])
            restricted_globals['output'] = {}
            raise ScriptError(
                f'Expected `output` to be a dictionary, was {output_type}'
            )
    except ScriptError as err:
        if return_response:
            raise ServiceValidationError(
                f'Error executing script: {err}'
            ) from err
        logger.error('Error executing script: %s', err)
        return None
    except Exception as err:
        if return_response:
            raise HomeAssistantError(
                f'Error executing script ({type(err).__name__}): {err}'
            ) from err
        logger.exception('Error executing script')
        return None
    return restricted_globals['output']


class StubPrinter:
    """Class to handle printing inside scripts."""

    def __init__(self, _getattr_: Callable[[Any, str, Any], Any]) -> None:
        """Initialize our printer."""
        pass

    def __call__(self, *objects: Any, **kwargs: Any) -> None:
        """Print text."""
        _LOGGER.warning(
            "Don't use print() inside scripts. Use logger.info() instead"
        )


class TimeWrapper:
    """Wrap the time module."""

    warned: bool = False

    def sleep(self, seconds: float) -> None:
        """Sleep method that warns once."""
        if not TimeWrapper.warned:
            TimeWrapper.warned = True
            _LOGGER.warning(
                'Using time.sleep can reduce the performance of Home Assistant'
            )
        time.sleep(seconds)

    def __getattr__(self, attr: str) -> Any:
        """Fetch an attribute from Time module."""
        attribute = getattr(time, attr)
        if callable(attribute):

            def wrapper(*args: Any, **kw: Any) -> Any:
                """Wrap to return callable method if callable."""
                return attribute(*args, **kw)

            return wrapper
        return attribute
