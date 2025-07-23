import datetime
from typing import Any, Dict, List, Optional
import jwt
from flask import current_app
from jwt import DecodeError, ExpiredSignatureError, InvalidAudienceError
from alerta.utils.response import absolute_url

dt = datetime.datetime

class Jwt:
    """
    JSON Web Token (JWT): https://tools.ietf.org/html/rfc7519
    """

    issuer: str
    type: str
    subject: str
    audience: str
    expiration: int
    not_before: int
    issued_at: int
    jwt_id: Optional[str]
    name: Optional[str]
    preferred_username: Optional[str]
    email: Optional[str]
    provider: Optional[str]
    orgs: List[str]
    groups: List[str]
    roles: List[str]
    scopes: List[str]
    email_verified: Optional[bool]
    picture: Optional[str]
    customers: List[str]
    oid: Optional[str]

    def __init__(
        self,
        iss: str,
        typ: str,
        sub: str,
        aud: str,
        exp: int,
        nbf: int,
        iat: int,
        jti: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.issuer = iss
        self.type = typ
        self.subject = sub
        self.audience = aud
        self.expiration = exp
        self.not_before = nbf
        self.issued_at = iat
        self.jwt_id = jti
        self.name = kwargs.get('name')
        self.preferred_username = kwargs.get('preferred_username')
        self.email = kwargs.get('email')
        self.provider = kwargs.get('provider')
        self.orgs = kwargs.get('orgs', list())
        self.groups = kwargs.get('groups', list())
        self.roles = kwargs.get('roles', list())
        self.scopes = kwargs.get('scopes', list())
        self.email_verified = kwargs.get('email_verified')
        self.picture = kwargs.get('picture')
        self.customers = kwargs.get('customers')
        self.oid = kwargs.get('oid')

    @classmethod
    def parse(
        cls,
        token: str,
        key: Optional[str] = None,
        verify: bool = True,
        algorithm: str = 'HS256'
    ) -> 'Jwt':
        try:
            json = jwt.decode(
                token,
                key=key or current_app.config['SECRET_KEY'],
                options={'verify_signature': verify},
                algorithms=[algorithm],
                audience=(
                    current_app.config['OAUTH2_CLIENT_ID']
                    or current_app.config['SAML2_ENTITY_ID']
                    or absolute_url()
                )
            )
        except (DecodeError, ExpiredSignatureError, InvalidAudienceError):
            raise
        return Jwt(
            iss=json.get('iss'),
            typ=json.get('typ'),
            sub=json.get('sub'),
            aud=json.get('aud'),
            exp=json.get('exp'),
            nbf=json.get('nbf'),
            iat=json.get('iat'),
            jti=json.get('jti'),
            name=json.get('name'),
            preferred_username=json.get('preferred_username'),
            email=json.get('email'),
            provider=json.get('provider'),
            orgs=json.get('orgs', []),
            groups=json.get('groups', []),
            roles=json.get('roles', []),
            scopes=json.get('scope', '').split(' '),
            email_verified=json.get('email_verified'),
            picture=json.get('picture'),
            customers=[json['customer']] if 'customer' in json else json.get('customers', []),
            oid=json.get('oid')
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
        if current_app.config.get('CUSTOMER_VIEWS'):
            data['customers'] = self.customers
        if self.oid:
            data['oid'] = self.oid
        return data

    def tokenize(self, algorithm: str = 'HS256') -> str:
        return jwt.encode(
            self.serialize,
            key=current_app.config['SECRET_KEY'],
            algorithm=algorithm
        )

    def __repr__(self) -> str:
        return (
            f"Jwt(iss={self.issuer!r}, sub={self.subject!r}, aud={self.audience!r}, "
            f"exp={self.expiration!r}, name={self.name!r}, "
            f"preferred_username={self.preferred_username!r}, customers={self.customers!r})"
        )
