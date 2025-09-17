import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar
import jwt
from flask import current_app
from jwt import DecodeError, ExpiredSignatureError, InvalidAudienceError
from alerta.utils.response import absolute_url

T = TypeVar('T', bound='Jwt')
dt = datetime.datetime

class Jwt:
    """
    JSON Web Token (JWT): https://tools.ietf.org/html/rfc7519
    """

    def __init__(self, 
                 iss: Optional[str], 
                 typ: Optional[str], 
                 sub: Optional[str], 
                 aud: Optional[str], 
                 exp: Optional[int], 
                 nbf: Optional[int], 
                 iat: Optional[int], 
                 jti: Optional[str] = None, 
                 **kwargs: Any) -> None:
        self.issuer: Optional[str] = iss
        self.type: Optional[str] = typ
        self.subject: Optional[str] = sub
        self.audience: Optional[str] = aud
        self.expiration: Optional[int] = exp
        self.not_before: Optional[int] = nbf
        self.issued_at: Optional[int] = iat
        self.jwt_id: Optional[str] = jti
        self.name: Optional[str] = kwargs.get('name')
        self.preferred_username: Optional[str] = kwargs.get('preferred_username')
        self.email: Optional[str] = kwargs.get('email')
        self.provider: Optional[str] = kwargs.get('provider')
        self.orgs: List[Any] = kwargs.get('orgs', list())
        self.groups: List[Any] = kwargs.get('groups', list())
        self.roles: List[Any] = kwargs.get('roles', list())
        self.scopes: List[str] = kwargs.get('scopes', list())
        self.email_verified: Optional[Any] = kwargs.get('email_verified')
        self.picture: Optional[Any] = kwargs.get('picture')
        self.customers: List[Any] = kwargs.get('customers', list())
        self.oid: Optional[Any] = kwargs.get('oid')

    @classmethod
    def parse(cls: Type[T], token: str, key: Optional[str] = None, verify: bool = True, algorithm: str = 'HS256') -> T:
        try:
            decoded_json: Dict[str, Any] = jwt.decode(
                token,
                key=key or current_app.config['SECRET_KEY'],
                options={'verify_signature': verify},
                algorithms=[algorithm],
                audience=current_app.config['OAUTH2_CLIENT_ID'] or current_app.config['SAML2_ENTITY_ID'] or absolute_url()
            )
        except (DecodeError, ExpiredSignatureError, InvalidAudienceError):
            raise
        return Jwt(
            iss=decoded_json.get('iss', None),
            typ=decoded_json.get('typ', None),
            sub=decoded_json.get('sub', None),
            aud=decoded_json.get('aud', None),
            exp=decoded_json.get('exp', None),
            nbf=decoded_json.get('nbf', None),
            iat=decoded_json.get('iat', None),
            jti=decoded_json.get('jti', None),
            name=decoded_json.get('name', None),
            preferred_username=decoded_json.get('preferred_username', None),
            email=decoded_json.get('email', None),
            provider=decoded_json.get('provider', None),
            orgs=decoded_json.get('orgs', list()),
            groups=decoded_json.get('groups', list()),
            roles=decoded_json.get('roles', list()),
            scopes=decoded_json.get('scope', '').split(' '),
            email_verified=decoded_json.get('email_verified', None),
            picture=decoded_json.get('picture', None),
            customers=[decoded_json['customer']] if 'customer' in decoded_json else decoded_json.get('customers', list()),
            oid=decoded_json.get('oid')
        )

    @property
    def serialize(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            'iss': self.issuer,
            'typ': self.type,
            'sub': self.subject,
            'aud': self.audience,
            'exp': self.expiration,
            'nbf': self.not_before,
            'iat': self.issued_at,
            'jti': self.jwt_id
        }
        if self.name:
            data['name'] = self.name
        if self.preferred_username:
            data['preferred_username'] = self.preferred_username
        if self.email:
            data['email'] = self.email
        if self.provider:
            data['provider'] = self.provider
        if self.orgs:
            data['orgs'] = self.orgs
        if self.groups:
            data['groups'] = self.groups
        if self.roles:
            data['roles'] = self.roles
        if self.scopes:
            data['scope'] = ' '.join(self.scopes)
        if self.email_verified is not None:
            data['email_verified'] = self.email_verified
        if self.picture is not None:
            data['picture'] = self.picture
        if current_app.config['CUSTOMER_VIEWS']:
            data['customers'] = self.customers
        if self.oid:
            data['oid'] = self.oid
        return data

    def tokenize(self, algorithm: str = 'HS256') -> str:
        return jwt.encode(self.serialize, key=current_app.config['SECRET_KEY'], algorithm=algorithm)

    def __repr__(self) -> str:
        return 'Jwt(iss={!r}, sub={!r}, aud={!r}, exp={!r}, name={!r}, preferred_username={!r}, customers={!r})'.format(
            self.issuer, self.subject, self.audience, self.expiration, self.name, self.preferred_username, self.customers
        )