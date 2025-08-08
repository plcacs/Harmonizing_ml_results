from __future__ import annotations
from functools import partial
import json
import re
from typing import Any
import voluptuous as vol
from voluptuous.humanize import humanize_error
import homeassistant.helpers.config_validation as cv
from script.translations import upload
from .model import Config, Integration

UNDEFINED: int = 0
REQUIRED: int = 1
REMOVED: int = 2
RE_REFERENCE: str = '\\[\\%key:(.+)\\%'
RE_TRANSLATION_KEY: re.Pattern = re.compile('^(?!.+[_-]{2})(?![_-])[a-z0-9-_]+(?<![_-])$')
RE_COMBINED_REFERENCE: re.Pattern = re.compile('(.+\\[%)|(%\\].+)')
RE_PLACEHOLDER_IN_SINGLE_QUOTES: re.Pattern = re.compile("'{\\w+}'")
ALLOW_NAME_TRANSLATION: set[str] = {'cert_expiry', 'cpuspeed', 'emulated_roku', 'faa_delays', 'garages_amsterdam', 'generic', 'google_travel_time', 'holiday', 'homekit_controller', 'islamic_prayer_times', 'local_calendar', 'local_ip', 'local_todo', 'nmap_tracker', 'rpi_power', 'swiss_public_transport', 'waze_travel_time', 'zodiac'}
REMOVED_TITLE_MSG: str = 'config.title key has been moved out of config and into the root of strings.json. Starting Home Assistant 0.109 you only need to define this key in the root if the title needs to be different than the name of your integration in the manifest.'
MOVED_TRANSLATIONS_DIRECTORY_MSG: str = "The '.translations' directory has been moved, the new name is 'translations', starting with Home Assistant 0.112 your translations will no longer load if you do not move/rename this "

def allow_name_translation(integration: Integration) -> bool:
    ...

def check_translations_directory_name(integration: Integration) -> None:
    ...

def find_references(strings: dict[str, Any], prefix: str, found: list[dict[str, str]]) -> None:
    ...

def removed_title_validator(config: Config, integration: Integration, value: Any) -> Any:
    ...

def translation_key_validator(value: str) -> str:
    ...

def translation_value_validator(value: str) -> str:
    ...

def string_no_single_quoted_placeholders(value: str) -> str:
    ...

def gen_data_entry_schema(*, config: Config, integration: Integration, flow_title: int, require_step_title: bool, mandatory_description: str = None) -> vol.Schema:
    ...

def gen_issues_schema(config: Config, integration: Integration) -> vol.Schema:
    ...

_EXCEPTIONS_SCHEMA: vol.Schema = {vol.Optional('exceptions'): cv.schema_with_slug_keys({vol.Optional('message'): translation_value_validator}, slug_validator=cv.slug)}

def gen_strings_schema(config: Config, integration: Integration) -> vol.Schema:
    ...

def gen_auth_schema(config: Config, integration: Integration) -> vol.Schema:
    ...

def gen_ha_hardware_schema(config: Config, integration: Integration) -> vol.Schema:
    ...

ONBOARDING_SCHEMA: vol.Schema = vol.Schema({vol.Required('area'): {str: translation_value_validator}, vol.Required('dashboard'): {str: {'title': translation_value_validator}}})

def validate_translation_file(config: Config, integration: Integration, all_strings: dict[str, Any]) -> None:
    ...

def validate(integrations: dict[str, Integration], config: Config) -> None:
    ...
