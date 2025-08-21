import datetime
from typing import Any, Dict, List, Optional, Union
import jwt
from flask import current_app
from jwt import DecodeError, ExpiredSignatureError, InvalidAudienceError
from alerta.utils.response import absolute_url
dt = datetime.datetime


class Jwt:
    """
    JSON Web Token (JWT): https://tools.ietf.org/html/rfc7519
    """

    def __init__(
        self,
        iss: Optional[str],
        typ: Optional[str],
        sub: Optional[str],
        aud: Optional[Union[str, List[str]]],
        exp: Optional[int],
        nbf: Optional[int],
        iat: Optional[int],
        jti: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.issuer: Optional[str] = iss
        self.type: Optional[str] = typ
        self.subject: Optional[str] = sub
        self.audience: Optional[Union[str, List[str]]] = aud
        self.expiration: Optional[int] = exp
        self.not_before: Optional[int] = nbf
        self.issued_at: Optional[int] = iat
        self.jwt_id: Optional[str] = jti
        self.name: Optional[str] = kwargs.get('name')
        self.preferred_username: Optional[str] = kwargs.get('preferred_username')
        self.email: Optional[str] = kwargs.get('email')
        self.provider: Optional[str] = kwargs.get('provider')
        self.orgs: List[str] = kwargs.get('orgs', list())
        self.groups: List[str] = kwargs.get('groups', list())
        self.roles: List[str] = kwargs.get('roles', list())
        self.scopes: List[str] = kwargs.get('scopes', list())
        self.email_verified: Optional[bool] = kwargs.get('email_verified')
        self.picture: Optional[str] = kwargs.get('picture')
        self.customers: Optional[List[str]] = kwargs.get('customers')
        self.oid: Optional[str] = kwargs.get('oid')

    @classmethod
    def parse(
        cls,
        token: str,
        key: Optional[Union[str, bytes]] = None,
        verify: bool = True,
        algorithm: str = 'HS256'
    ) -> 'Jwt':
        try:
            payload: Dict[str, Any] = jwt.decode(
                token,
                key=key or current_app.config['SECRET_KEY'],
                options={'verify_signature': verify},
                algorithms=[algorithm],
                audience=current_app.config['OAUTH2_CLIENT_ID']
                or current_app.config['SAML2_ENTITY_ID']
                or absolute_url()
            )
        except (DecodeError, ExpiredSignatureError, InvalidAudienceError):
            raise
        return Jwt(
            iss=payload.get('iss', None),
            typ=payload.get('typ', None),
            sub=payload.get('sub', None),
            aud=payload.get('aud', None),
            exp=payload.get('exp', None),
            nbf=payload.get('nbf', None),
            iat=payload.get('iat', None),
            jti=payload.get('jti', None),
            name=payload.get('name', None),
            preferred_username=payload.get('preferred_username', None),
            email=payload.get('email', None),
            provider=payload.get('provider', None),
            orgs=payload.get('orgs', list()),
            groups=payload.get('groups', list()),
            roles=payload.get('roles', list()),
            scopes=payload.get('scope', '').split(' '),
            email_verified=payload.get('email_verified', None),
            picture=payload.get('picture', None),
            customers=[payload['customer']] if 'customer' in payload else payload.get('customers', list()),
            oid=payload.get('oid')
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

    def tokenize(self, algorithm: str = 'HS256') -> Union[str, bytes]:
        return jwt.encode(self.serialize, key=current_app.config['SECRET_KEY'], algorithm=algorithm)

    def __repr__(self) -> str:
        return 'Jwt(iss={!r}, sub={!r}, aud={!r}, exp={!r}, name={!r}, preferred_username={!r}, customers={!r})'.format(
            self.issuer, self.subject, self.audience, self.expiration, self.name, self.preferred_username, self.customers
        )