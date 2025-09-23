"""Alexa Resources and Assets."""
from typing import Any

class AlexaGlobalCatalog:
    """The Global Alexa catalog.

    https://developer.amazon.com/docs/device-apis/resources-and-assets.html#global-alexa-catalog

    You can use the global Alexa catalog for pre-defined names of devices, settings,
    values, and units.

    This catalog is localized into all the languages that Alexa supports.
    You can reference the following catalog of pre-defined friendly names.

    Each item in the following list is an asset identifier followed by its
    supported friendly names. The first friendly name for each identifier is
    the one displayed in the Alexa mobile app.
    """
    DEVICE_NAME_AIR_PURIFIER = 'Alexa.DeviceName.AirPurifier'
    DEVICE_NAME_FAN = 'Alexa.DeviceName.Fan'
    DEVICE_NAME_ROUTER = 'Alexa.DeviceName.Router'
    DEVICE_NAME_SHADE = 'Alexa.DeviceName.Shade'
    DEVICE_NAME_SHOWER = 'Alexa.DeviceName.Shower'
    DEVICE_NAME_SPACE_HEATER = 'Alexa.DeviceName.SpaceHeater'
    DEVICE_NAME_WASHER = 'Alexa.DeviceName.Washer'
    SETTING_2G_GUEST_WIFI = 'Alexa.Setting.2GGuestWiFi'
    SETTING_5G_GUEST_WIFI = 'Alexa.Setting.5GGuestWiFi'
    SETTING_AUTO = 'Alexa.Setting.Auto'
    SETTING_DIRECTION = 'Alexa.Setting.Direction'
    SETTING_DRY_CYCLE = 'Alexa.Setting.DryCycle'
    SETTING_FAN_SPEED = 'Alexa.Setting.FanSpeed'
    SETTING_GUEST_WIFI = 'Alexa.Setting.GuestWiFi'
    SETTING_HEAT = 'Alexa.Setting.Heat'
    SETTING_MODE = 'Alexa.Setting.Mode'
    SETTING_NIGHT = 'Alexa.Setting.Night'
    SETTING_OPENING = 'Alexa.Setting.Opening'
    SETTING_OSCILLATE = 'Alexa.Setting.Oscillate'
    SETTING_PRESET = 'Alexa.Setting.Preset'
    SETTING_QUIET = 'Alexa.Setting.Quiet'
    SETTING_TEMPERATURE = 'Alexa.Setting.Temperature'
    SETTING_WASH_CYCLE = 'Alexa.Setting.WashCycle'
    SETTING_WATER_TEMPERATURE = 'Alexa.Setting.WaterTemperature'
    SHOWER_HAND_HELD = 'Alexa.Shower.HandHeld'
    SHOWER_RAIN_HEAD = 'Alexa.Shower.RainHead'
    UNIT_ANGLE_DEGREES = 'Alexa.Unit.Angle.Degrees'
    UNIT_ANGLE_RADIANS = 'Alexa.Unit.Angle.Radians'
    UNIT_DISTANCE_FEET = 'Alexa.Unit.Distance.Feet'
    UNIT_DISTANCE_INCHES = 'Alexa.Unit.Distance.Inches'
    UNIT_DISTANCE_KILOMETERS = 'Alexa.Unit.Distance.Kilometers'
    UNIT_DISTANCE_METERS = 'Alexa.Unit.Distance.Meters'
    UNIT_DISTANCE_MILES = 'Alexa.Unit.Distance.Miles'
    UNIT_DISTANCE_YARDS = 'Alexa.Unit.Distance.Yards'
    UNIT_MASS_GRAMS = 'Alexa.Unit.Mass.Grams'
    UNIT_MASS_KILOGRAMS = 'Alexa.Unit.Mass.Kilograms'
    UNIT_PERCENT = 'Alexa.Unit.Percent'
    UNIT_TEMPERATURE_CELSIUS = 'Alexa.Unit.Temperature.Celsius'
    UNIT_TEMPERATURE_DEGREES = 'Alexa.Unit.Temperature.Degrees'
    UNIT_TEMPERATURE_FAHRENHEIT = 'Alexa.Unit.Temperature.Fahrenheit'
    UNIT_TEMPERATURE_KELVIN = 'Alexa.Unit.Temperature.Kelvin'
    UNIT_VOLUME_CUBIC_FEET = 'Alexa.Unit.Volume.CubicFeet'
    UNIT_VOLUME_CUBIC_METERS = 'Alexa.Unit.Volume.CubicMeters'
    UNIT_VOLUME_GALLONS = 'Alexa.Unit.Volume.Gallons'
    UNIT_VOLUME_LITERS = 'Alexa.Unit.Volume.Liters'
    UNIT_VOLUME_PINTS = 'Alexa.Unit.Volume.Pints'
    UNIT_VOLUME_QUARTS = 'Alexa.Unit.Volume.Quarts'
    UNIT_WEIGHT_OUNCES = 'Alexa.Unit.Weight.Ounces'
    UNIT_WEIGHT_POUNDS = 'Alexa.Unit.Weight.Pounds'
    VALUE_CLOSE = 'Alexa.Value.Close'
    VALUE_DELICATE = 'Alexa.Value.Delicate'
    VALUE_HIGH = 'Alexa.Value.High'
    VALUE_LOW = 'Alexa.Value.Low'
    VALUE_MAXIMUM = 'Alexa.Value.Maximum'
    VALUE_MEDIUM = 'Alexa.Value.Medium'
    VALUE_MINIMUM = 'Alexa.Value.Minimum'
    VALUE_OPEN = 'Alexa.Value.Open'
    VALUE_QUICK_WASH = 'Alexa.Value.QuickWash'

class AlexaCapabilityResource:
    """Base class for Alexa capabilityResources, modeResources, and presetResources.

    Resources objects labels must be unique across all modeResources and
    presetResources within the same device. To provide support for all
    supported locales, include one label from the AlexaGlobalCatalog in the
    labels array.

    You cannot use any words from the following list as friendly names:
    https://developer.amazon.com/docs/alexa/device-apis/resources-and-assets.html#names-you-cannot-use

    https://developer.amazon.com/docs/device-apis/resources-and-assets.html#capability-resources
    """

    def __init__(self, labels):
        """Initialize an Alexa resource."""
        self._resource_labels = []
        for label in labels:
            self._resource_labels.append(label)

    def serialize_capability_resources(self):
        """Return capabilityResources object serialized for an API response."""
        return self.serialize_labels(self._resource_labels)

    def serialize_configuration(self):
        """Return serialized configuration for an API response.

        Return ModeResources, PresetResources friendlyNames serialized.
        """
        raise NotImplementedError

    def serialize_labels(self, resources):
        """Return serialized labels for an API response.

        Returns resource label objects for friendlyNames serialized.
        """
        labels: list[dict[str, Any]] = []
        label_dict: dict[str, Any]
        for label in resources:
            if label in AlexaGlobalCatalog.__dict__.values():
                label_dict = {'@type': 'asset', 'value': {'assetId': label}}
            else:
                label_dict = {'@type': 'text', 'value': {'text': label, 'locale': 'en-US'}}
            labels.append(label_dict)
        return {'friendlyNames': labels}

class AlexaModeResource(AlexaCapabilityResource):
    """Implements Alexa ModeResources.

    https://developer.amazon.com/docs/device-apis/resources-and-assets.html#capability-resources
    """

    def __init__(self, labels, ordered=False):
        """Initialize an Alexa modeResource."""
        super().__init__(labels)
        self._supported_modes: list[dict[str, Any]] = []
        self._mode_ordered: bool = ordered

    def add_mode(self, value, labels) -> None:
        """Add mode to the supportedModes object."""
        self._supported_modes.append({'value': value, 'labels': labels})

    def serialize_configuration(self) -> dict[str, Any]:
        """Return serialized configuration for an API response.

        Returns configuration for ModeResources friendlyNames serialized.
        """
        mode_resources: list[dict[str, Any]] = []
        for mode in self._supported_modes:
            result = {'value': mode['value'], 'modeResources': self.serialize_labels(mode['labels'])}
            mode_resources.append(result)
        return {'ordered': self._mode_ordered, 'supportedModes': mode_resources}

class AlexaPresetResource(AlexaCapabilityResource):
    """Implements Alexa PresetResources.

    Use presetResources with RangeController to provide a set of
    friendlyNames for each RangeController preset.

    https://developer.amazon.com/docs/device-apis/resources-and-assets.html#presetresources
    """

    def __init__(self, labels: list[str], min_value: float, max_value: float, precision: float, unit: str | None=None) -> None:
        """Initialize an Alexa presetResource."""
        super().__init__(labels)
        self._presets: list[dict[str, Any]] = []
        self._minimum_value = min_value
        self._maximum_value = max_value
        self._precision = precision
        self._unit_of_measure = None
        if unit in AlexaGlobalCatalog.__dict__.values():
            self._unit_of_measure = unit

    def add_preset(self, value: float, labels: list[str]) -> None:
        """Add preset to configuration presets array."""
        self._presets.append({'value': value, 'labels': labels})

    def serialize_configuration(self) -> dict[str, Any]:
        """Return serialized configuration for an API response.

        Returns configuration for PresetResources friendlyNames serialized.
        """
        configuration: dict[str, Any] = {'supportedRange': {'minimumValue': self._minimum_value, 'maximumValue': self._maximum_value, 'precision': self._precision}}
        if self._unit_of_measure:
            configuration['unitOfMeasure'] = self._unit_of_measure
        if self._presets:
            preset_resources = [{'rangeValue': preset['value'], 'presetResources': self.serialize_labels(preset['labels'])} for preset in self._presets]
            configuration['presets'] = preset_resources
        return configuration

class AlexaSemantics:
    """Class for Alexa Semantics Object.

    You can optionally enable additional utterances by using semantics. When
    you use semantics, you manually map the phrases "open", "close", "raise",
    and "lower" to directives.

    Semantics is supported for the following interfaces only: ModeController,
    RangeController, and ToggleController.

    Semantics stateMappings are only supported for one interface of the same
    type on the same device. If a device has multiple RangeControllers only
    one interface may use stateMappings otherwise discovery will fail.

    You can support semantics actionMappings on different controllers for the
    same device, however each controller must support different phrases.
    For example, you can support "raise" on a RangeController, and "open"
    on a ModeController, but you can't support "open" on both RangeController
    and ModeController. Semantics stateMappings are only supported for one
    interface on the same device.

    https://developer.amazon.com/docs/device-apis/alexa-discovery.html#semantics-object
    """
    MAPPINGS_ACTION = 'actionMappings'
    MAPPINGS_STATE = 'stateMappings'
    ACTIONS_TO_DIRECTIVE = 'ActionsToDirective'
    STATES_TO_VALUE = 'StatesToValue'
    STATES_TO_RANGE = 'StatesToRange'
    ACTION_CLOSE = 'Alexa.Actions.Close'
    ACTION_LOWER = 'Alexa.Actions.Lower'
    ACTION_OPEN = 'Alexa.Actions.Open'
    ACTION_RAISE = 'Alexa.Actions.Raise'
    STATES_OPEN = 'Alexa.States.Open'
    STATES_CLOSED = 'Alexa.States.Closed'
    DIRECTIVE_RANGE_SET_VALUE = 'SetRangeValue'
    DIRECTIVE_RANGE_ADJUST_VALUE = 'AdjustRangeValue'
    DIRECTIVE_TOGGLE_TURN_ON = 'TurnOn'
    DIRECTIVE_TOGGLE_TURN_OFF = 'TurnOff'
    DIRECTIVE_MODE_SET_MODE = 'SetMode'
    DIRECTIVE_MODE_ADJUST_MODE = 'AdjustMode'

    def __init__(self) -> None:
        """Initialize an Alexa modeResource."""
        self._action_mappings: list[dict[str, Any]] = []
        self._state_mappings: list[dict[str, Any]] = []

    def _add_action_mapping(self, semantics: dict[str, Any]) -> None:
        """Add action mapping between actions and interface directives."""
        self._action_mappings.append(semantics)

    def _add_state_mapping(self, semantics: dict[str, Any]) -> None:
        """Add state mapping between states and interface directives."""
        self._state_mappings.append(semantics)

    def add_states_to_value(self, states: list[str], value: Any) -> None:
        """Add StatesToValue stateMappings."""
        self._add_state_mapping({'@type': self.STATES_TO_VALUE, 'states': states, 'value': value})

    def add_states_to_range(self, states: list[str], min_value: float, max_value: float) -> None:
        """Add StatesToRange stateMappings."""
        self._add_state_mapping({'@type': self.STATES_TO_RANGE, 'states': states, 'range': {'minimumValue': min_value, 'maximumValue': max_value}})

    def add_action_to_directive(self, actions: list[str], directive: str, payload: dict[str, Any]) -> None:
        """Add ActionsToDirective actionMappings."""
        self._add_action_mapping({'@type': self.ACTIONS_TO_DIRECTIVE, 'actions': actions, 'directive': {'name': directive, 'payload': payload}})

    def serialize_semantics(self) -> dict[str, Any]:
        """Return semantics object serialized for an API response."""
        semantics: dict[str, Any] = {}
        if self._action_mappings:
            semantics[self.MAPPINGS_ACTION] = self._action_mappings
        if self._state_mappings:
            semantics[self.MAPPINGS_STATE] = self._state_mappings
        return semantics