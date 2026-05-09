from __future__ import annotations
from collections import deque
from collections.abc import Mapping
import logging
import math
from typing import Any, ConfigType, DiscoveryInfoType

class SensorTrend(BinarySensorEntity, RestoreEntity):
    """Representation of a trend Sensor."""
    _attr_should_poll: bool
    _gradient: float
    _state: str

    def __init__(self, 
                 name: str, 
                 entity_id: str, 
                 attribute: str | None, 
                 invert: bool, 
                 sample_duration: int, 
                 min_gradient: float, 
                 min_samples: int, 
                 max_samples: int, 
                 unique_id: str | None, 
                 device_class: BinarySensorDeviceClass | None, 
                 sensor_entity_id: str | None, 
                 device_info: Mapping[str, Any] | None):
        """Initialize the sensor."""
        self._entity_id: str = entity_id
        self._attribute: str | None = attribute
        self._invert: bool = invert
        self._sample_duration: int = sample_duration
        self._min_gradient: float = min_gradient
        self._min_samples: int = min_samples
        self.samples: deque[tuple[float, float]] = deque(maxlen=int(max_samples))
        self._attr_name: str = name
        self._attr_device_class: BinarySensorDeviceClass | None = device_class
        self._attr_unique_id: str | None = unique_id
        self._attr_device_info: Mapping[str, Any] | None = device_info
        if sensor_entity_id:
            self.entity_id: str = sensor_entity_id

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        """Return the state attributes of the sensor."""
        return {ATTR_ENTITY_ID: self._entity_id, ATTR_FRIENDLY_NAME: self._attr_name, ATTR_GRADIENT: self._gradient, ATTR_INVERT: self._invert, ATTR_MIN_GRADIENT: self._min_gradient, ATTR_SAMPLE_COUNT: len(self.samples), ATTR_SAMPLE_DURATION: self._sample_duration}

    async def async_added_to_hass(self) -> None:
        """Complete device setup after being added to hass."""

        @callback
        def trend_sensor_state_listener(event: Event) -> None:
            """Handle state changes on the observed device."""
            if (new_state := event.data['new_state']) is None:
                return
            try:
                if self._attribute:
                    state = new_state.attributes.get(self._attribute)
                else:
                    state = new_state.state
                if state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                    self._attr_available = False
                else:
                    self._attr_available = True
                    sample = (new_state.last_updated.timestamp(), float(state))
                    self.samples.append(sample)
                self.async_schedule_update_ha_state(True)
            except (ValueError, TypeError) as ex:
                _LOGGER.error(ex)
        self.async_on_remove(async_track_state_change_event(self.hass, [self._entity_id], trend_sensor_state_listener))
        if not (state := (await self.async_get_last_state())):
            return
        if state.state in {STATE_UNKNOWN, STATE_UNAVAILABLE}:
            return
        self._attr_is_on = state.state == STATE_ON

    async def async_update(self) -> None:
        """Get the latest data and update the states."""
        if self._sample_duration > 0:
            cutoff = utcnow().timestamp() - self._sample_duration
            while self.samples and self.samples[0][0] < cutoff:
                self.samples.popleft()
        if len(self.samples) < self._min_samples:
            return
        await self.hass.async_add_executor_job(self._calculate_gradient)
        self._attr_is_on = abs(self._gradient) > abs(self._min_gradient) and math.copysign(self._gradient, self._min_gradient) == self._gradient
        if self._invert:
            self._attr_is_on = not self._attr_is_on

    def _calculate_gradient(self) -> None:
        """Compute the linear trend gradient of the current samples.

        This need run inside executor.
        """
        timestamps = np.array([t for t, _ in self.samples])
        values = np.array([s for _, s in self.samples])
        coeffs = np.polyfit(timestamps, values, 1)
        self._gradient = coeffs[0]
