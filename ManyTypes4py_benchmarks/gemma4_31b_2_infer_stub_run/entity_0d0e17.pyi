"""Provide entity classes for group entities."""
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Optional, Union, overload
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_component import EntityComponent
from .registry import GroupIntegrationRegistry, SingleStateType

ENTITY_ID_FORMAT: str = ...

class GroupEntity(Entity):
    """Representation of a Group of entities."""
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool

    @callback
    def async_start_preview(self, preview_callback: Callable[[str, Mapping[str, Any]], Any]) -> Any:
        """Render a preview."""
        ...

    async def async_added_to_hass(self) -> None:
        """Register listeners."""
        ...

    @callback
    def _update_at_start(self, _: Any) -> None:
        """Update the group state at start."""
        ...

    @callback
    def async_defer_or_update_ha_state(self) -> None:
        """Only update once at start."""
        ...

    @abstractmethod
    @callback
    def async_update_group_state(self) -> None:
        """Abstract method to update the entity."""
        ...

    @callback
    def async_update_supported_features(self, entity_id: str, new_state: State) -> None:
        """Update dictionaries with supported features."""
        ...

class Group(Entity):
    """Track a group of entity ids."""
    _unrecorded_attributes: frozenset[str]
    _attr_should_poll: bool
    hass: HomeAssistant
    created_by_service: bool
    mode: Callable[[Iterable[Any]], bool]
    tracking: tuple[str, ...]
    trackable: tuple[str, ...]
    single_state_type_key: Optional[SingleStateType]

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: Optional[bool],
        order: Optional[int],
    ) -> None:
        """Initialize a group."""
        ...

    @staticmethod
    @callback
    def async_create_group_entity(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: Optional[bool],
        object_id: Optional[str],
        order: Optional[int],
    ) -> Group:
        """Create a group entity."""
        ...

    @staticmethod
    async def async_create_group(
        hass: HomeAssistant,
        name: str,
        *,
        created_by_service: bool,
        entity_ids: Collection[str],
        icon: Optional[str],
        mode: Optional[bool],
        object_id: Optional[str],
        order: Optional[int],
    ) -> Group:
        """Initialize a group."""
        ...

    def set_name(self, value: str) -> None:
        """Set Group name."""
        ...

    @property
    def state(self) -> Optional[str]:
        """Return the state of the group."""
        ...

    def set_icon(self, value: str) -> None:
        """Set Icon for group."""
        ...

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        """Return the state attributes for the group."""
        ...

    @property
    def assumed_state(self) -> bool:
        """Test if any member has an assumed state."""
        ...

    @callback
    def async_update_tracked_entity_ids(self, entity_ids: Collection[str]) -> None:
        """Update the member entity IDs."""
        ...

    def _set_tracked(self, entity_ids: Collection[str]) -> None:
        """Tuple of entities to be tracked."""
        ...

    @callback
    def _async_deregister(self) -> None:
        """Deregister group entity from the registry."""
        ...

    @callback
    def _async_start(self, _: Optional[Any] = None) -> None:
        """Start tracking members and write state."""
        ...

    @callback
    def _async_start_tracking(self) -> None:
        """Start tracking members."""
        ...

    @callback
    def _async_stop(self) -> None:
        """Unregister the group from Home Assistant."""
        ...

    @callback
    def async_update_group_state(self) -> None:
        """Query all members and determine current group state."""
        ...

    async def async_added_to_hass(self) -> None:
        """Handle addition to Home Assistant."""
        ...

    async def async_will_remove_from_hass(self) -> None:
        """Handle removal from Home Assistant."""
        ...

    async def _async_state_changed_listener(self, event: Event) -> None:
        """Respond to a member state changing."""
        ...

    def _reset_tracked_state(self) -> None:
        """Reset tracked state."""
        ...

    def _see_state(self, new_state: State) -> None:
        """Keep track of the state."""
        ...

    @callback
    def _async_update_group_state(self, tr_state: Optional[State] = None) -> None:
        """Update group state."""
        ...

def async_get_component(hass: HomeAssistant) -> EntityComponent[Group]:
    """Get the group entity component."""
    ...