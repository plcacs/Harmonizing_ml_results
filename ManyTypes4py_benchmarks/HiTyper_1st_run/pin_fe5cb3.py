"""Code to handle pins on a Firmata board."""
from __future__ import annotations
from collections.abc import Callable
import logging
from typing import cast
from .board import FirmataBoard, FirmataPinType
from .const import PIN_MODE_INPUT, PIN_MODE_PULLUP, PIN_TYPE_ANALOG
_LOGGER = logging.getLogger(__name__)

class FirmataPinUsedException(Exception):
    """Represents an exception when a pin is already in use."""

class FirmataBoardPin:
    """Manages a single Firmata board pin."""

    def __init__(self, board: Union[str, board.FirmataBoard, board.FirmataPinType], pin: Union[str, board.FirmataBoard, board.FirmataPinType], pin_mode: Union[str, board.FirmataBoard, board.FirmataPinType]) -> None:
        """Initialize the pin."""
        self.board = board
        self._pin = pin
        self._pin_mode = pin_mode
        self._pin_type, self._firmata_pin = self.board.get_pin_type(self._pin)
        self._state = None
        self._analog_pin = None
        if self._pin_type == PIN_TYPE_ANALOG:
            self._analog_pin = int(cast(str, self._pin)[1:])

    def setup(self) -> None:
        """Set up a pin and make sure it is valid."""
        if not self.board.mark_pin_used(self._pin):
            raise FirmataPinUsedException(f'Pin {self._pin} already used!')

class FirmataBinaryDigitalOutput(FirmataBoardPin):
    """Representation of a Firmata Digital Output Pin."""

    def __init__(self, board: Union[str, board.FirmataBoard, board.FirmataPinType], pin: Union[str, board.FirmataBoard, board.FirmataPinType], pin_mode: Union[str, board.FirmataBoard, board.FirmataPinType], initial: Union[bool, typing.Sequence[T]], negate) -> None:
        """Initialize the digital output pin."""
        self._initial = initial
        self._negate = negate
        super().__init__(board, pin, pin_mode)

    async def start_pin(self):
        """Set initial state on a pin."""
        _LOGGER.debug('Setting initial state for digital output pin %s on board %s', self._pin, self.board.name)
        api = self.board.api
        await api.set_pin_mode_digital_output(self._firmata_pin)
        if self._initial:
            new_pin_state = not self._negate
        else:
            new_pin_state = self._negate
        await api.digital_pin_write(self._firmata_pin, int(new_pin_state))
        self._state = self._initial

    @property
    def is_on(self):
        """Return true if digital output is on."""
        return self._state

    async def turn_on(self):
        """Turn on digital output."""
        _LOGGER.debug('Turning digital output on pin %s on', self._pin)
        new_pin_state = not self._negate
        await self.board.api.digital_pin_write(self._firmata_pin, int(new_pin_state))
        self._state = True

    async def turn_off(self):
        """Turn off digital output."""
        _LOGGER.debug('Turning digital output on pin %s off', self._pin)
        new_pin_state = self._negate
        await self.board.api.digital_pin_write(self._firmata_pin, int(new_pin_state))
        self._state = False

class FirmataPWMOutput(FirmataBoardPin):
    """Representation of a Firmata PWM/analog Output Pin."""

    def __init__(self, board: Union[str, board.FirmataBoard, board.FirmataPinType], pin: Union[str, board.FirmataBoard, board.FirmataPinType], pin_mode: Union[str, board.FirmataBoard, board.FirmataPinType], initial: Union[bool, typing.Sequence[T]], minimum: Union[int, float], maximum: Union[int, float]) -> None:
        """Initialize the PWM/analog output pin."""
        self._initial = initial
        self._min = minimum
        self._max = maximum
        self._range = self._max - self._min
        super().__init__(board, pin, pin_mode)

    async def start_pin(self):
        """Set initial state on a pin."""
        _LOGGER.debug('Setting initial state for PWM/analog output pin %s on board %s to %d', self._pin, self.board.name, self._initial)
        api = self.board.api
        await api.set_pin_mode_pwm_output(self._firmata_pin)
        new_pin_state = round(self._initial * self._range / 255) + self._min
        await api.pwm_write(self._firmata_pin, new_pin_state)
        self._state = self._initial

    @property
    def state(self):
        """Return PWM/analog state."""
        return self._state

    async def set_level(self, level):
        """Set PWM/analog output."""
        _LOGGER.debug('Setting PWM/analog output on pin %s to %d', self._pin, level)
        new_pin_state = round(level * self._range / 255) + self._min
        await self.board.api.pwm_write(self._firmata_pin, new_pin_state)
        self._state = level

class FirmataBinaryDigitalInput(FirmataBoardPin):
    """Representation of a Firmata Digital Input Pin."""

    def __init__(self, board: Union[str, board.FirmataBoard, board.FirmataPinType], pin: Union[str, board.FirmataBoard, board.FirmataPinType], pin_mode: Union[str, board.FirmataBoard, board.FirmataPinType], negate) -> None:
        """Initialize the digital input pin."""
        self._negate = negate
        super().__init__(board, pin, pin_mode)

    async def start_pin(self, forward_callback):
        """Get initial state and start reporting a pin."""
        _LOGGER.debug('Starting reporting updates for digital input pin %s on board %s', self._pin, self.board.name)
        self._forward_callback = forward_callback
        api = self.board.api
        if self._pin_mode == PIN_MODE_INPUT:
            await api.set_pin_mode_digital_input(self._pin, self.latch_callback)
        elif self._pin_mode == PIN_MODE_PULLUP:
            await api.set_pin_mode_digital_input_pullup(self._pin, self.latch_callback)
        new_state = bool((await self.board.api.digital_read(self._firmata_pin))[0])
        if self._negate:
            new_state = not new_state
        self._state = new_state
        await api.enable_digital_reporting(self._pin)
        self._forward_callback()

    async def stop_pin(self):
        """Stop reporting digital input pin."""
        _LOGGER.debug('Stopping reporting updates for digital input pin %s on board %s', self._pin, self.board.name)
        api = self.board.api
        await api.disable_digital_reporting(self._pin)

    @property
    def is_on(self):
        """Return true if digital input is on."""
        return self._state

    async def latch_callback(self, data):
        """Update pin state on callback."""
        if data[1] != self._firmata_pin:
            return
        _LOGGER.debug('Received latch %d for digital input pin %d on board %s', data[2], self._firmata_pin, self.board.name)
        new_state = bool(data[2])
        if self._negate:
            new_state = not new_state
        if self._state == new_state:
            return
        self._state = new_state
        self._forward_callback()

class FirmataAnalogInput(FirmataBoardPin):
    """Representation of a Firmata Analog Input Pin."""

    def __init__(self, board: Union[str, board.FirmataBoard, board.FirmataPinType], pin: Union[str, board.FirmataBoard, board.FirmataPinType], pin_mode: Union[str, board.FirmataBoard, board.FirmataPinType], differential) -> None:
        """Initialize the analog input pin."""
        self._differential = differential
        super().__init__(board, pin, pin_mode)

    async def start_pin(self, forward_callback):
        """Get initial state and start reporting a pin."""
        _LOGGER.debug('Starting reporting updates for analog input pin %s on board %s', self._pin, self.board.name)
        self._forward_callback = forward_callback
        api = self.board.api
        await api.set_pin_mode_analog_input(self._analog_pin, self.latch_callback, self._differential)
        self._state = (await self.board.api.analog_read(self._analog_pin))[0]
        self._forward_callback()

    async def stop_pin(self):
        """Stop reporting analog input pin."""
        _LOGGER.debug('Stopping reporting updates for analog input pin %s on board %s', self._pin, self.board.name)
        api = self.board.api
        await api.disable_analog_reporting(self._analog_pin)

    @property
    def state(self):
        """Return sensor state."""
        return self._state

    async def latch_callback(self, data):
        """Update pin state on callback."""
        if data[1] != self._analog_pin:
            return
        _LOGGER.debug('Received latch %d for analog input pin %s on board %s', data[2], self._pin, self.board.name)
        new_state = data[2]
        if self._state == new_state:
            _LOGGER.debug('stopping')
            return
        self._state = new_state
        self._forward_callback()