from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, List, cast
from urllib.parse import urljoin
from uuid import uuid4
from flask import current_app, request
from itsdangerous import BadData, SignatureExpired, URLSafeTimedSerializer
from alerta.app import mailer
from alerta.exceptions import ApiError, NoCustomerMatch
from alerta.models.customer import Customer
from alerta.models.token import Jwt
from alerta.utils.response import absolute_url
if TYPE_CHECKING:
    from alerta.models.user import User
try:
    import bcrypt

    def generate_password_hash(password: Any):
        if isinstance(password, str):
            password = password.encode('utf-8')
        return bcrypt.hashpw(password, bcrypt.gensalt(prefix=b'2a')).decode('utf-8')

    def check_password_hash(pwhash: Any, password: Any):
        return bcrypt.checkpw(password.encode('utf-8'), pwhash.encode('utf-8'))
except ImportError:
    from werkzeug.security import check_password_hash
    from werkzeug.security import generate_password_hash

def not_authorized(allowed_setting: Union[str, list[str], bool], groups: Union[str, list[str]]) -> bool:
    return current_app.config['AUTH_REQUIRED'] and (not ('*' in current_app.config[allowed_setting] or set(current_app.config[allowed_setting]).intersection(set(groups))))

def get_customers(login: Union[str, list[str], arxiv.submission.User], groups: Union[str, list[str], arxiv.submission.User]) -> list:
    if current_app.config['CUSTOMER_VIEWS']:
        try:
            return Customer.lookup(login, groups)
        except NoCustomerMatch as e:
            raise ApiError(str(e), 403)
    else:
        return []

def create_token(user_id: Union[str, list[str], bool], name: Union[str, list[str], bool], login: Union[str, list[str], bool], provider: Union[str, list[str], bool], customers: Union[str, list[str], bool], scopes: Union[str, list[str], bool], email: Union[None, str, list[str], bool]=None, email_verified: Union[None, str, list[str], bool]=None, picture: Union[None, str, list[str], bool]=None, **kwargs) -> Jwt:
    now = datetime.utcnow()
    return Jwt(iss=request.url_root, typ='Bearer', sub=user_id, aud=current_app.config.get('OAUTH2_CLIENT_ID') or current_app.config.get('SAML2_ENTITY_ID') or absolute_url(), exp=now + timedelta(days=current_app.config['TOKEN_EXPIRE_DAYS']), nbf=now, iat=now, jti=str(uuid4()), name=name, preferred_username=login, email=email, email_verified=email_verified, provider=provider, scopes=scopes, customers=customers, picture=picture, **kwargs)

def link(base_url: str, *parts) -> str:
    if base_url.endswith('/'):
        return urljoin(base_url, '/'.join(('#',) + parts))
    else:
        return urljoin(base_url, '/'.join(parts))

def send_confirmation(user: Union[str, arxiv.users.domain.User], token: Union[str, User, zerver.models.UserProfile]) -> None:
    subject = f"[Alerta] Please verify your email '{user.email}'"
    text = "Hello {name}!\n\nPlease verify your email address is {email} by clicking on the link below:\n\n{url}\n\nYou're receiving this email because you recently created a new Alerta account. If this wasn't you, please ignore this email.".format(name=user.name, email=user.email, url=link(request.referrer, 'confirm', token))
    mailer.send_email(user.email, subject, body=text)

def send_password_reset(user: Union[User, str], token: Union[str, User, zerver.models.Realm]) -> None:
    subject = '[Alerta] Reset password request'
    text = "You forgot your password. Reset it by clicking on the link below:\n\n{url}\n\nYou're receiving this email because you asked for a password reset of an Alerta account. If this wasn't you, please ignore this email.".format(url=link(request.referrer, 'reset', token))
    mailer.send_email(user.email, subject, body=text)

def generate_email_token(email: str, salt: Union[None, str]=None) -> str:
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return cast(str, serializer.dumps(email, salt))

def confirm_email_token(token: Union[str, int], salt: Union[None, str, int]=None, expiration: int=900) -> Union[str, None]:
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token, salt=salt, max_age=expiration)
    except SignatureExpired as e:
        raise ApiError('confirmation token has expired', 401, errors=['invalid_token', str(e)])
    except BadData as e:
        raise ApiError('confirmation token invalid', 400, errors=['invalid_request', str(e)])
    return email