from typing import List

def get_confirmations(types: List[str], object_ids: List[int], hostname: str = None) -> List[dict]:
    ...

def get_remote_servers_for_support(email_to_search: str, uuid_to_search: str, hostname_to_search: str) -> List[RemoteZulipServer]:
    ...

def remote_servers_support(request, *, query: str = None, remote_server_id: int = None, remote_realm_id: int = None, monthly_discounted_price: float = None, annual_discounted_price: float = None, minimum_licenses: int = None, required_plan_tier: int = None, fixed_price: float = None, sent_invoice_id: str = None, sponsorship_pending: bool = None, approve_sponsorship: bool = False, billing_modality: str = None, plan_end_date: str = None, modify_plan: str = None, delete_fixed_price_next_plan: bool = False, remote_server_status: str = None, complimentary_access_plan: str = None) -> HttpResponse:
    ...
