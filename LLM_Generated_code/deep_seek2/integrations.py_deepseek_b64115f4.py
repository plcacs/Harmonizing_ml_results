from typing import Dict, List, Optional, Tuple, Union

def get_all_event_types_for_integration(integration: Integration) -> Optional[List[str]]:
    integration = INTEGRATIONS[integration.name]
    if isinstance(integration, WebhookIntegration):
        if integration.name == "githubsponsors":
            return import_string("zerver.webhooks.github.view.SPONSORS_EVENT_TYPES")
        function = integration.get_function()
        if hasattr(function, "_all_event_types"):
            return function._all_event_types
    return None
