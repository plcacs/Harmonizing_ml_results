from typing import Any, Dict, Set, List

def recursive_flatten(prefix: str, data: Dict[str, Any]) -> Dict[str, Any]:
def _load_translations_files_by_language(translation_files: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
def build_resources(translation_strings: Dict[str, Dict[str, Dict[str, Any]]], components: Set[str], category: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
async def _async_get_component_strings(hass: HomeAssistant, languages: List[str], components: Set[str], integrations: Dict[str, Integration]) -> Dict[str, Dict[str, Dict[str, Any]]]:
def async_get_translations(hass: HomeAssistant, language: str, category: str, integrations: List[str] = None, config_flow: bool = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
def async_get_cached_translations(hass: HomeAssistant, language: str, category: str, integration: str = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
def _async_get_translations_cache(hass: HomeAssistant) -> _TranslationCache:
def async_setup(hass: HomeAssistant):
async def async_load_integrations(hass: HomeAssistant, integrations: Set[str]):
def async_translations_loaded(hass: HomeAssistant, components: Set[str]) -> bool:
def async_get_exception_message(translation_domain: str, translation_key: str, translation_placeholders: Dict[str, Any] = None) -> str:
def async_translate_state(hass: HomeAssistant, state: str, domain: str, platform: str, translation_key: str, device_class: str) -> str:
