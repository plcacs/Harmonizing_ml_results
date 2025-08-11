"""Selectors for KNX."""
from enum import Enum
from typing import Any
import voluptuous as vol
from ..validation import ga_validator, maybe_ga_validator
from .const import CONF_DPT, CONF_GA_PASSIVE, CONF_GA_STATE, CONF_GA_WRITE

class GASelector:
    """Selector for a KNX group address structure."""

    def __init__(self, write: bool=True, state: bool=True, passive: bool=True, write_required: bool=False, state_required: bool=False, dpt: Union[None, bool]=None) -> None:
        """Initialize the group address selector."""
        self.write = write
        self.state = state
        self.passive = passive
        self.write_required = write_required
        self.state_required = state_required
        self.dpt = dpt
        self.schema = self.build_schema()

    def __call__(self, data: Union[dict, dict[str, typing.Any], T]) -> Union[dict, bytes, list[str]]:
        """Validate the passed data."""
        return self.schema(data)

    def build_schema(self):
        """Create the schema based on configuration."""
        schema = {}
        self._add_group_addresses(schema)
        self._add_passive(schema)
        self._add_dpt(schema)
        return vol.Schema(schema)

    def _add_group_addresses(self, schema: Union[dict[str, typing.Any], dict, list[dict]]) -> None:
        """Add basic group address items to the schema."""

        def add_ga_item(key: Any, allowed: Any, required: Any) -> None:
            """Add a group address item validator to the schema."""
            if not allowed:
                schema[vol.Remove(key)] = object
                return
            if required:
                schema[vol.Required(key)] = ga_validator
            else:
                schema[vol.Optional(key, default=None)] = maybe_ga_validator
        add_ga_item(CONF_GA_WRITE, self.write, self.write_required)
        add_ga_item(CONF_GA_STATE, self.state, self.state_required)

    def _add_passive(self, schema: Union[str, bool]) -> None:
        """Add passive group addresses validator to the schema."""
        if self.passive:
            schema[vol.Optional(CONF_GA_PASSIVE, default=list)] = vol.Any([ga_validator], vol.All(vol.IsFalse(), vol.SetTo(list)))
        else:
            schema[vol.Remove(CONF_GA_PASSIVE)] = object

    def _add_dpt(self, schema: Union[dict[str, typing.Any], str]) -> None:
        """Add DPT validator to the schema."""
        if self.dpt is not None:
            schema[vol.Required(CONF_DPT)] = vol.In({item.value for item in self.dpt})
        else:
            schema[vol.Remove(CONF_DPT)] = object