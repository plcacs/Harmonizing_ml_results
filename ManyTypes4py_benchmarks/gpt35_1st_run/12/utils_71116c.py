def generate_password_hash(password: str) -> str:
def check_password_hash(pwhash: str, password: str) -> bool:
def not_authorized(allowed_setting: str, groups: List[str]) -> bool:
def get_customers(login: str, groups: List[str]) -> List[Customer]:
def create_token(user_id: str, name: str, login: str, provider: str, customers: List[Customer], scopes: List[str], email: str = None, email_verified: bool = None, picture: str = None, **kwargs: Any) -> Jwt:
def link(base_url: str, *parts: str) -> str:
def send_confirmation(user: User, token: str) -> None:
def send_password_reset(user: User, token: str) -> None:
def generate_email_token(email: str, salt: str = None) -> str:
def confirm_email_token(token: str, salt: str = None, expiration: int = 900) -> str:
