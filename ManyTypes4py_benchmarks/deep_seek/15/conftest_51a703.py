"""Provide common Z-Wave JS fixtures."""
import asyncio
import copy
import io
from typing import Any, Dict, Generator, cast
from unittest.mock import DEFAULT, AsyncMock, MagicMock, patch
import pytest
from zwave_js_server.event import Event
from zwave_js_server.model.driver import Driver
from zwave_js_server.model.node import Node
from zwave_js_server.model.node.data_model import NodeDataType
from zwave_js_server.version import VersionInfo
from homeassistant.components.zwave_js.const import DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonArrayType
from tests.common import MockConfigEntry, load_json_array_fixture, load_json_object_fixture

@pytest.fixture(name='controller_state', scope='package')
def controller_state_fixture() -> Dict[str, Any]:
    """Load the controller state fixture data."""
    return load_json_object_fixture('controller_state.json', DOMAIN)

@pytest.fixture(name='controller_node_state', scope='package')
def controller_node_state_fixture() -> Dict[str, Any]:
    """Load the controller node state fixture data."""
    return load_json_object_fixture('controller_node_state.json', DOMAIN)

@pytest.fixture(name='version_state', scope='package')
def version_state_fixture() -> Dict[str, Any]:
    """Load the version state fixture data."""
    return {'type': 'version', 'driverVersion': '6.0.0-beta.0', 'serverVersion': '1.0.0', 'homeId': 1234567890}

@pytest.fixture(name='log_config_state')
def log_config_state_fixture() -> Dict[str, Any]:
    """Return log config state fixture data."""
    return {'enabled': True, 'level': 'info', 'logToFile': False, 'filename': '', 'forceConsole': False}

@pytest.fixture(name='config_entry_diagnostics', scope='package')
def config_entry_diagnostics_fixture() -> JsonArrayType:
    """Load the config entry diagnostics fixture data."""
    return load_json_array_fixture('config_entry_diagnostics.json', DOMAIN)

@pytest.fixture(name='config_entry_diagnostics_redacted', scope='package')
def config_entry_diagnostics_redacted_fixture() -> Dict[str, Any]:
    """Load the redacted config entry diagnostics fixture data."""
    return load_json_object_fixture('config_entry_diagnostics_redacted.json', DOMAIN)

@pytest.fixture(name='multisensor_6_state', scope='package')
def multisensor_6_state_fixture() -> Dict[str, Any]:
    """Load the multisensor 6 node state fixture data."""
    return load_json_object_fixture('multisensor_6_state.json', DOMAIN)

@pytest.fixture(name='ecolink_door_sensor_state', scope='package')
def ecolink_door_sensor_state_fixture() -> Dict[str, Any]:
    """Load the Ecolink Door/Window Sensor node state fixture data."""
    return load_json_object_fixture('ecolink_door_sensor_state.json', DOMAIN)

@pytest.fixture(name='hank_binary_switch_state', scope='package')
def binary_switch_state_fixture() -> Dict[str, Any]:
    """Load the hank binary switch node state fixture data."""
    return load_json_object_fixture('hank_binary_switch_state.json', DOMAIN)

@pytest.fixture(name='bulb_6_multi_color_state', scope='package')
def bulb_6_multi_color_state_fixture() -> Dict[str, Any]:
    """Load the bulb 6 multi-color node state fixture data."""
    return load_json_object_fixture('bulb_6_multi_color_state.json', DOMAIN)

@pytest.fixture(name='light_color_null_values_state', scope='package')
def light_color_null_values_state_fixture() -> Dict[str, Any]:
    """Load the light color null values node state fixture data."""
    return load_json_object_fixture('light_color_null_values_state.json', DOMAIN)

@pytest.fixture(name='eaton_rf9640_dimmer_state', scope='package')
def eaton_rf9640_dimmer_state_fixture() -> Dict[str, Any]:
    """Load the eaton rf9640 dimmer node state fixture data."""
    return load_json_object_fixture('eaton_rf9640_dimmer_state.json', DOMAIN)

@pytest.fixture(name='lock_schlage_be469_state', scope='package')
def lock_schlage_be469_state_fixture() -> Dict[str, Any]:
    """Load the schlage lock node state fixture data."""
    return load_json_object_fixture('lock_schlage_be469_state.json', DOMAIN)

@pytest.fixture(name='lock_august_asl03_state', scope='package')
def lock_august_asl03_state_fixture() -> Dict[str, Any]:
    """Load the August Pro lock node state fixture data."""
    return load_json_object_fixture('lock_august_asl03_state.json', DOMAIN)

@pytest.fixture(name='climate_radio_thermostat_ct100_plus_state', scope='package')
def climate_radio_thermostat_ct100_plus_state_fixture() -> Dict[str, Any]:
    """Load the climate radio thermostat ct100 plus node state fixture data."""
    return load_json_object_fixture('climate_radio_thermostat_ct100_plus_state.json', DOMAIN)

@pytest.fixture(name='climate_radio_thermostat_ct100_plus_different_endpoints_state', scope='package')
def climate_radio_thermostat_ct100_plus_different_endpoints_state_fixture() -> Dict[str, Any]:
    """Load the thermostat fixture state with values on different endpoints.

    This device is a radio thermostat ct100.
    """
    return load_json_object_fixture('climate_radio_thermostat_ct100_plus_different_endpoints_state.json', DOMAIN)

@pytest.fixture(name='climate_adc_t3000_state', scope='package')
def climate_adc_t3000_state_fixture() -> Dict[str, Any]:
    """Load the climate ADC-T3000 node state fixture data."""
    return load_json_object_fixture('climate_adc_t3000_state.json', DOMAIN)

@pytest.fixture(name='climate_airzone_aidoo_control_hvac_unit_state', scope='package')
def climate_airzone_aidoo_control_hvac_unit_state_fixture() -> Dict[str, Any]:
    """Load the climate Airzone Aidoo Control HVAC Unit state fixture data."""
    return load_json_object_fixture('climate_airzone_aidoo_control_hvac_unit_state.json', DOMAIN)

@pytest.fixture(name='climate_danfoss_lc_13_state', scope='package')
def climate_danfoss_lc_13_state_fixture() -> Dict[str, Any]:
    """Load Danfoss (LC-13) electronic radiator thermostat node state fixture data."""
    return load_json_object_fixture('climate_danfoss_lc_13_state.json', DOMAIN)

@pytest.fixture(name='climate_eurotronic_spirit_z_state', scope='package')
def climate_eurotronic_spirit_z_state_fixture() -> Dict[str, Any]:
    """Load the climate Eurotronic Spirit Z thermostat node state fixture data."""
    return load_json_object_fixture('climate_eurotronic_spirit_z_state.json', DOMAIN)

@pytest.fixture(name='climate_heatit_z_trm6_state', scope='package')
def climate_heatit_z_trm6_state_fixture() -> Dict[str, Any]:
    """Load the climate HEATIT Z-TRM6 thermostat node state fixture data."""
    return load_json_object_fixture('climate_heatit_z_trm6_state.json', DOMAIN)

@pytest.fixture(name='climate_heatit_z_trm3_state', scope='package')
def climate_heatit_z_trm3_state_fixture() -> Dict[str, Any]:
    """Load the climate HEATIT Z-TRM3 thermostat node state fixture data."""
    return load_json_object_fixture('climate_heatit_z_trm3_state.json', DOMAIN)

@pytest.fixture(name='climate_heatit_z_trm2fx_state', scope='package')
def climate_heatit_z_trm2fx_state_fixture() -> Dict[str, Any]:
    """Load the climate HEATIT Z-TRM2fx thermostat node state fixture data."""
    return load_json_object_fixture('climate_heatit_z_trm2fx_state.json', DOMAIN)

@pytest.fixture(name='climate_heatit_z_trm3_no_value_state', scope='package')
def climate_heatit_z_trm3_no_value_state_fixture() -> Dict[str, Any]:
    """Load the climate HEATIT Z-TRM3 thermostat node w/no value state fixture data."""
    return load_json_object_fixture('climate_heatit_z_trm3_no_value_state.json', DOMAIN)

@pytest.fixture(name='nortek_thermostat_state', scope='package')
def nortek_thermostat_state_fixture() -> Dict[str, Any]:
    """Load the nortek thermostat node state fixture data."""
    return load_json_object_fixture('nortek_thermostat_state.json', DOMAIN)

@pytest.fixture(name='srt321_hrt4_zw_state', scope='package')
def srt321_hrt4_zw_state_fixture() -> Dict[str, Any]:
    """Load the climate HRT4-ZW / SRT321 / SRT322 thermostat node state fixture data."""
    return load_json_object_fixture('srt321_hrt4_zw_state.json', DOMAIN)

@pytest.fixture(name='chain_actuator_zws12_state', scope='package')
def window_cover_state_fixture() -> Dict[str, Any]:
    """Load the window cover node state fixture data."""
    return load_json_object_fixture('chain_actuator_zws12_state.json', DOMAIN)

@pytest.fixture(name='fan_generic_state', scope='package')
def fan_generic_state_fixture() -> Dict[str, Any]:
    """Load the fan node state fixture data."""
    return load_json_object_fixture('fan_generic_state.json', DOMAIN)

@pytest.fixture(name='hs_fc200_state', scope='package')
def hs_fc200_state_fixture() -> Dict[str, Any]:
    """Load the HS FC200+ node state fixture data."""
    return load_json_object_fixture('fan_hs_fc200_state.json', DOMAIN)

@pytest.fixture(name='leviton_zw4sf_state', scope='package')
def leviton_zw4sf_state_fixture() -> Dict[str, Any]:
    """Load the Leviton ZW4SF node state fixture data."""
    return load_json_object_fixture('leviton_zw4sf_state.json', DOMAIN)

@pytest.fixture(name='fan_honeywell_39358_state', scope='package')
def fan_honeywell_39358_state_fixture() -> Dict[str, Any]:
    """Load the fan node state fixture data."""
    return load_json_object_fixture('fan_honeywell_39358_state.json', DOMAIN)

@pytest.fixture(name='gdc_zw062_state', scope='package')
def motorized_barrier_cover_state_fixture() -> Dict[str, Any]:
    """Load the motorized barrier cover node state fixture data."""
    return load_json_object_fixture('cover_zw062_state.json', DOMAIN)

@pytest.fixture(name='iblinds_v2_state', scope='package')
def iblinds_v2_state_fixture() -> Dict[str, Any]:
    """Load the iBlinds v2 node state fixture data."""
    return load_json_object_fixture('cover_iblinds_v2_state.json', DOMAIN)

@pytest.fixture(name='iblinds_v3_state', scope='package')
def iblinds_v3_state_fixture() -> Dict[str, Any]:
    """Load the iBlinds v3 node state fixture data."""
    return load_json_object_fixture('cover_iblinds_v3_state.json', DOMAIN)

@pytest.fixture(name='zvidar_state', scope='package')
def zvidar_state_fixture() -> Dict[str, Any]:
    """Load the ZVIDAR node state fixture data."""
    return load_json_object_fixture('cover_zvidar_state.json', DOMAIN)

@pytest.fixture(name='qubino_shutter_state', scope='package')
def qubino_shutter_state_fixture() -> Dict[str, Any]:
    """Load the Qubino Shutter node state fixture data."""
    return load_json_object_fixture('cover_qubino_shutter_state.json', DOMAIN)

@pytest.fixture(name='aeotec_nano_shutter_state', scope='package')
def aeotec_nano_shutter_state_fixture() -> Dict[str, Any]:
    """Load the Aeotec Nano Shutter node state fixture data."""
    return load_json_object_fixture('cover_aeotec_nano_shutter_state.json', DOMAIN)

@pytest.fixture(name='fibaro_fgr222_shutter_state', scope='package')
def fibaro_fgr222_shutter_state_fixture() -> Dict[str, Any]:
    """Load the Fibaro FGR222 node state fixture data."""
    return load_json_object_fixture('cover_fibaro_fgr222_state.json', DOMAIN)

@pytest.fixture(name='fibaro_fgr223_shutter_state', scope='package')
def fibaro_fgr223_shutter_state_fixture() -> Dict[str, Any]:
    """Load the Fibaro FGR223 node state fixture data."""
    return load_json_object_fixture('cover_fibaro_fgr223_state.json', DOMAIN)

@pytest.fixture(name='shelly_europe_ltd_qnsh_001p10_state', scope='package')
def shelly_europe_ltd_qnsh_001p10_state_fixture() -> Dict[str, Any]:
    """Load the Shelly QNSH 001P10 node state fixture data."""
    return load_json_object_fixture('shelly_europe_ltd_qnsh_001p10_state.json', DOMAIN)

@pytest.fixture(name='merten_507801_state', scope='package')
def merten_507801_state_fixture() -> Dict[str, Any]:
    """Load the Merten 507801 Shutter node state fixture data."""
    return load_json_object_fixture('cover_merten_507801_state.json', DOMAIN)

@pytest.fixture(name='aeon_smart_switch_6_state', scope='package')
def aeon_smart_switch_6_state_fixture() -> Dict[str, Any]:
    """Load the AEON Labs (ZW096) Smart Switch 6 node state fixture data."""
    return load_json_object_fixture('aeon_smart_switch_6_state.json', DOMAIN)

@pytest.fixture(name='ge_12730_state', scope='package')
def ge_12730_state_fixture() -> Dict[str, Any]:
    """Load the GE 12730 node state fixture data."""
    return load_json_object_fixture('fan_ge_12730_state.json', DOMAIN)

@pytest.fixture(name='aeotec_radiator_thermostat_state', scope='package')
def aeotec_radiator_thermostat_state_fixture() -> Dict[str, Any]:
    """Load the Aeotec Radiator Thermostat node state fixture data."""
    return load_json_object_fixture('aeotec_radiator_thermostat_state.json', DOMAIN)

@pytest.fixture(name='inovelli_lzw36_state', scope='package')
def inovelli_lzw36_state_fixture() -> Dict[str, Any]:
    """Load the Inovelli LZW36 node state fixture data."""
    return load_json_object_fixture('inovelli_lzw36_state.json', DOMAIN)

@pytest.fixture(name='null_name_check_state', scope='package')
def null_name_check_state_fixture() -> Dict[str, Any]:
    """Load the null name check node state fixture data."""
    return load_json_object_fixture('null_name_check_state.json', DOMAIN)

@pytest.fixture(name='lock_id_lock_as_id150_state', scope='package')
def lock_id_lock_as_id150_state_fixture() -> Dict[str, Any]:
    """Load the id lock id-150 lock node state fixture data."""
    return load_json_object_fixture('lock_id_lock_as_id150_state.json', DOMAIN)

@pytest.fixture(name='climate_radio_thermostat_ct101_multiple_temp_units_state', scope='package')
def climate_radio_thermostat_ct101_multiple_temp_units_state_fixture() -> Dict[str, Any]:
    """Load the climate multiple temp units node state fixture data."""
    return load_json_object_fixture('climate_radio_thermostat_ct101_multiple_temp_units_state.json', DOMAIN)

@pytest.fixture(name='climate_radio_thermostat_ct100_mode_and_setpoint_on_different_endpoints_state', scope='package')
def climate_radio_thermostat_ct100_mode_and_setpoint_on_different_endpoints_state_fixture() -> Dict[str, Any]:
    """Load climate device w/ mode+setpoint on diff endpoints node state fixture data."""
    return load_json_object_fixture('climate_radio_thermostat_ct100_mode_and_setpoint_on_different_endpoints_state.json', DOMAIN)

@pytest.fixture(name='vision_security_zl7432_state', scope='package')
def vision_security_zl7432_state_fixture() -> Dict[str, Any]:
    """Load the vision security zl7432 switch node state fixture data."""
    return load_json_object_fixture('vision_security_zl7432_state.json', DOMAIN)

@pytest.fixture(name='zen_31_state', scope='package')
def zem_31_state_fixture() -> Dict[str, Any]:
    """Load the zen_31 node state fixture data."""
    return load_json_object_fixture('zen_31_state.json', DOMAIN)

@pytest.fixture(name='wallmote_central_scene_state', scope='package')
def wallmote_central_scene_state_fixture() -> Dict[str, Any]:
    """Load the wallmote central scene node state fixture data."""
    return load_json_object_fixture('wallmote_central_scene_state.json', DOMAIN)

@pytest.fixture(name='ge_in_wall_dimmer_switch_state', scope='package')
def ge_in_wall_dimmer_switch_state_fixture() -> Dict[str, Any]:
    """Load the ge in-wall dimmer switch node