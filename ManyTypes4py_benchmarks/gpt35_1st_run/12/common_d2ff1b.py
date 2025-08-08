from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

def config_entry_hub() -> ConfigEntry:
def config_entry_salt() -> ConfigEntry:
def config_entry_alert() -> ConfigEntry:
def config_entry_leak() -> ConfigEntry:
def config_entry_softener() -> ConfigEntry:
def config_entry_filter() -> ConfigEntry:
def config_entry_protection_valve() -> ConfigEntry:
def config_entry_pump_controller() -> ConfigEntry:
def config_entry_ro_filter() -> ConfigEntry:

def help_assert_entries(hass: HomeAssistant, entity_registry: er, snapshot: SnapshotAssertion, config_entry: ConfigEntry, step: str, assert_unknown: bool = False) -> None:
