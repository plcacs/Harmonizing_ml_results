from typing import Any, Dict, List, Optional

class AlexaGlobalCatalog:
    DEVICE_NAME_AIR_PURIFIER: str = 'Alexa.DeviceName.AirPurifier'
    DEVICE_NAME_FAN: str = 'Alexa.DeviceName.Fan'
    DEVICE_NAME_ROUTER: str = 'Alexa.DeviceName.Router'
    DEVICE_NAME_SHADE: str = 'Alexa.DeviceName.Shade'
    DEVICE_NAME_SHOWER: str = 'Alexa.DeviceName.Shower'
    DEVICE_NAME_SPACE_HEATER: str = 'Alexa.DeviceName.SpaceHeater'
    DEVICE_NAME_WASHER: str = 'Alexa.DeviceName.Washer'
    SETTING_2G_GUEST_WIFI: str = 'Alexa.Setting.2GGuestWiFi'
    SETTING_5G_GUEST_WIFI: str = 'Alexa.Setting.5GGuestWiFi'
    SETTING_AUTO: str = 'Alexa.Setting.Auto'
    SETTING_DIRECTION: str = 'Alexa.Setting.Direction'
    SETTING_DRY_CYCLE: str = 'Alexa.Setting.DryCycle'
    SETTING_FAN_SPEED: str = 'Alexa.Setting.FanSpeed'
    SETTING_GUEST_WIFI: str = 'Alexa.Setting.GuestWiFi'
    SETTING_HEAT: str = 'Alexa.Setting.Heat'
    SETTING_MODE: str = 'Alexa.Setting.Mode'
    SETTING_NIGHT: str = 'Alexa.Setting.Night'
    SETTING_OPENING: str = 'Alexa.Setting.Opening'
    SETTING_OSCILLATE: str = 'Alexa.Setting.Oscillate'
    SETTING_PRESET: str = 'Alexa.Setting.Preset'
    SETTING_QUIET: str = 'Alexa.Setting.Quiet'
    SETTING_TEMPERATURE: str = 'Alexa.Setting.Temperature'
    SETTING_WASH_CYCLE: str = 'Alexa.Setting.WashCycle'
    SETTING_WATER_TEMPERATURE: str = 'Alexa.Setting.WaterTemperature'
    SHOWER_HAND_HELD: str = 'Alexa.Shower.HandHeld'
    SHOWER_RAIN_HEAD: str = 'Alexa.Shower.RainHead'
    UNIT_ANGLE_DEGREES: str = 'Alexa.Unit.Angle.Degrees'
    UNIT_ANGLE_RADIANS: str = 'Alexa.Unit.Angle.Radians'
    UNIT_DISTANCE_FEET: str = 'Alexa.Unit.Distance.Feet'
    UNIT_DISTANCE_INCHES: str = 'Alexa.Unit.Distance.Inches'
    UNIT_DISTANCE_KILOMETERS: str = 'Alexa.Unit.Distance.Kilometers'
    UNIT_DISTANCE_METERS: str = 'Alexa.Unit.Distance.Meters'
    UNIT_DISTANCE_MILES: str = 'Alexa.Unit.Distance.Miles'
    UNIT_DISTANCE_YARDS: str = 'Alexa.Unit.Distance.Yards'
    UNIT_MASS_GRAMS: str = 'Alexa.Unit.Mass.Grams'
    UNIT_MASS_KILOGRAMS: str = 'Alexa.Unit.Mass.Kilograms'
    UNIT_PERCENT: str = 'Alexa.Unit.Percent'
    UNIT_TEMPERATURE_CELSIUS: str = 'Alexa.Unit.Temperature.Celsius'
    UNIT_TEMPERATURE_DEGREES: str = 'Alexa.Unit.Temperature.Degrees'
    UNIT_TEMPERATURE_FAHRENHEIT: str = 'Alexa.Unit.Temperature.Fahrenheit'
    UNIT_TEMPERATURE_KELVIN: str = 'Alexa.Unit.Temperature.Kelvin'
    UNIT_VOLUME_CUBIC_FEET: str = 'Alexa.Unit.Volume.CubicFeet'
    UNIT_VOLUME_CUBIC_METERS: str = 'Alexa.Unit.Volume.CubicMeters'
    UNIT_VOLUME_GALLONS: str = 'Alexa.Unit.Volume.Gallons'
    UNIT_VOLUME_LITERS: str = 'Alexa.Unit.Volume.Liters'
    UNIT_VOLUME_PINTS: str = 'Alexa.Unit.Volume.Pints'
    UNIT_VOLUME_QUARTS: str = 'Alexa.Unit.Volume.Quarts'
    UNIT_WEIGHT_OUNCES: str = 'Alexa.Unit.Weight.Ounces'
    UNIT_WEIGHT_POUNDS: str = 'Alexa.Unit.Weight.Pounds'
    VALUE_CLOSE: str = 'Alexa.Value.Close'
    VALUE_DELICATE: str = 'Alexa.Value.Delicate'
    VALUE_HIGH: str = 'Alexa.Value.High'
    VALUE_LOW: str = 'Alexa.Value.Low'
    VALUE_MAXIMUM: str = 'Alexa.Value.Maximum'
    VALUE_MEDIUM: str = 'Alexa.Value.Medium'
    VALUE_MINIMUM: str = 'Alexa.Value.Minimum'
    VALUE_OPEN: str = 'Alexa.Value.Open'
    VALUE_QUICK_WASH: str = 'Alexa.Value.QuickWash'

class AlexaCapabilityResource:
    def __init__(self, labels: List[str]) -> None:
        self._resource_labels: List[str] = []
        for label in labels:
            self._resource_labels.append(label)

    def serialize_capability_resources(self) -> Dict[str, Any]:
        return self.serialize_labels(self._resource_labels)

    def serialize_configuration(self) -> Dict[str, Any]:
        raise NotImplementedError

    def serialize_labels(self, resources: List[str]) -> Dict[str, Any]:
        labels: List[Dict[str, Any]] = []
        for label in resources:
            if label in AlexaGlobalCatalog.__dict__.values():
                label_dict: Dict[str, Any] = {'@type': 'asset', 'value': {'assetId': label}}
            else:
                label_dict = {'@type': 'text', 'value': {'text': label, 'locale': 'en-US'}}
            labels.append(label_dict)
        return {'friendlyNames': labels}

class AlexaModeResource(AlexaCapabilityResource):
    def __init__(self, labels: List[str], ordered: bool = False) -> None:
        super().__init__(labels)
        self._supported_modes: List[Dict[str, Any]] = []
        self._mode_ordered: bool = ordered

    def add_mode(self, value: str, labels: List[str]) -> None:
        self._supported_modes.append({'value': value, 'labels': labels})

    def serialize_configuration(self) -> Dict[str, Any]:
        mode_resources: List[Dict[str, Any]] = []
        for mode in self._supported_modes:
            result: Dict[str, Any] = {
                'value': mode['value'],
                'modeResources': self.serialize_labels(mode['labels'])
            }
            mode_resources.append(result)
        return {'ordered': self._mode_ordered, 'supportedModes': mode_resources}

class AlexaPresetResource(AlexaCapabilityResource):
    def __init__(self, labels: List[str], min_value: float, max_value: float, precision: float, unit: Optional[str] = None) -> None:
        super().__init__(labels)
        self._presets: List[Dict[str, Any]] = []
        self._minimum_value: float = min_value
        self._maximum_value: float = max_value
        self._precision: float = precision
        self._unit_of_measure: Optional[str] = None
        if unit in AlexaGlobalCatalog.__dict__.values():
            self._unit_of_measure = unit

    def add_preset(self, value: float, labels: List[str]) -> None:
        self._presets.append({'value': value, 'labels': labels})

    def serialize_configuration(self) -> Dict[str, Any]:
        configuration: Dict[str, Any] = {
            'supportedRange': {
                'minimumValue': self._minimum_value,
                'maximumValue': self._maximum_value,
                'precision': self._precision
            }
        }
        if self._unit_of_measure:
            configuration['unitOfMeasure'] = self._unit_of_measure
        if self._presets:
            preset_resources: List[Dict[str, Any]] = [
                {
                    'rangeValue': preset['value'],
                    'presetResources': self.serialize_labels(preset['labels'])
                } for preset in self._presets
            ]
            configuration['presets'] = preset_resources
        return configuration

class AlexaSemantics:
    MAPPINGS_ACTION: str = 'actionMappings'
    MAPPINGS_STATE: str = 'stateMappings'
    ACTIONS_TO_DIRECTIVE: str = 'ActionsToDirective'
    STATES_TO_VALUE: str = 'StatesToValue'
    STATES_TO_RANGE: str = 'StatesToRange'
    ACTION_CLOSE: str = 'Alexa.Actions.Close'
    ACTION_LOWER: str = 'Alexa.Actions.Lower'
    ACTION_OPEN: str = 'Alexa.Actions.Open'
    ACTION_RAISE: str = 'Alexa.Actions.Raise'
    STATES_OPEN: str = 'Alexa.States.Open'
    STATES_CLOSED: str = 'Alexa.States.Closed'
    DIRECTIVE_RANGE_SET_VALUE: str = 'SetRangeValue'
    DIRECTIVE_RANGE_ADJUST_VALUE: str = 'AdjustRangeValue'
    DIRECTIVE_TOGGLE_TURN_ON: str = 'TurnOn'
    DIRECTIVE_TOGGLE_TURN_OFF: str = 'TurnOff'
    DIRECTIVE_MODE_SET_MODE: str = 'SetMode'
    DIRECTIVE_MODE_ADJUST_MODE: str = 'AdjustMode'

    def __init__(self) -> None:
        self._action_mappings: List[Dict[str, Any]] = []
        self._state_mappings: List[Dict[str, Any]] = []

    def _add_action_mapping(self, semantics: Dict[str, Any]) -> None:
        self._action_mappings.append(semantics)

    def _add_state_mapping(self, semantics: Dict[str, Any]) -> None:
        self._state_mappings.append(semantics)

    def add_states_to_value(self, states: List[str], value: Any) -> None:
        self._add_state_mapping({
            '@type': self.STATES_TO_VALUE,
            'states': states,
            'value': value
        })

    def add_states_to_range(self, states: List[str], min_value: float, max_value: float) -> None:
        self._add_state_mapping({
            '@type': self.STATES_TO_RANGE,
            'states': states,
            'range': {
                'minimumValue': min_value,
                'maximumValue': max_value
            }
        })

    def add_action_to_directive(self, actions: List[str], directive: str, payload: Any) -> None:
        self._add_action_mapping({
            '@type': self.ACTIONS_TO_DIRECTIVE,
            'actions': actions,
            'directive': {
                'name': directive,
                'payload': payload
            }
        })

    def serialize_semantics(self) -> Dict[str, Any]:
        semantics: Dict[str, Any] = {}
        if self._action_mappings:
            semantics[self.MAPPINGS_ACTION] = self._action_mappings
        if self._state_mappings:
            semantics[self.MAPPINGS_STATE] = self._state_mappings
        return semantics