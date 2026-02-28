import datetime
from typing import Any, Dict, Optional, List
import jwt
from flask import current_app
from jwt import DecodeError, ExpiredSignatureError, InvalidAudienceError
from alerta.utils.response import absolute_url

dt = datetime.datetime

class Jwt:
    """
    JSON Web Token (JWT): https://tools.ietf.org/html/rfc7519
    """

    def __init__(self, 
                 iss: str, 
                 typ: str, 
                 sub: str, 
                 aud: str, 
                 exp: int, 
                 nbf: int, 
                 iat: int, 
                 jti: Optional[str] = None, 
                 name: Optional[str] = None, 
                 preferred_username: Optional[str] = None, 
                 email: Optional[str] = None, 
                 provider: Optional[str] = None, 
                 orgs: Optional[List[str]] = list(), 
                 groups: Optional[List[str]] = list(), 
                 roles: Optional[List[str]] = list(), 
                 scopes: Optional[List[str]] = list(), 
                 email_verified: Optional[bool] = None, 
                 picture: Optional[str] = None, 
                 customers: Optional[List[str]] = list(), 
                 oid: Optional[str] = None) -> None:
        self.issuer = iss
        self.type = typ
        self.subject = sub
        self.audience = aud
        self.expiration = exp
        self.not_before = nbf
        self.issued_at = iat
        self.jwt_id = jti
        self.name = name
        self.preferred_username = preferred_username
        self.email = email
        self.provider = provider
        self.orgs = orgs
        self.groups = groups
        self.roles = roles
        self.scopes = scopes
        self.email_verified = email_verified
        self.picture = picture
        self.customers = customers
        self.oid = oid

    @classmethod
    def parse(cls, token: str, key: Optional[str] = None, verify: bool = True, algorithm: str = 'HS256') -> 'Jwt':
        try:
            json = jwt.decode(token, key=key or current_app.config['SECRET_KEY'], options={'verify_signature': verify}, algorithms=[algorithm], audience=current_app.config['OAUTH2_CLIENT_ID'] or current_app.config['SAML2_ENTITY_ID'] or absolute_url())
        except (DecodeError, ExpiredSignatureError, InvalidAudienceError):
            raise
        return Jwt(iss=json.get('iss', None), typ=json.get('typ', None), sub=json.get('sub', None), aud=json.get('aud', None), exp=json.get('exp', None), nbf=json.get('nbf', None), iat=json.get('iat', None), jti=json.get('jti', None), name=json.get('name', None), preferred_username=json.get('preferred_username', None), email=json.get('email', None), provider=json.get('provider', None), orgs=json.get('orgs', list()), groups=json.get('groups', list()), roles=json.get('roles', list()), scopes=json.get('scope', '').split(' '), email_verified=json.get('email_verified', None), picture=json.get('picture', None), customers=[json['customer']] if 'customer' in json else json.get('customers', list()), oid=json.get('oid'))

    @property
    def serialize(self) -> Dict[str, Any]:
        data = {'iss': self.issuer, 'typ': self.type, 'sub': self.subject, 'aud': self.audience, 'exp': self.expiration, 'nbf': self.not_before, 'iat': self.issued_at, 'jti': self.jwt_id}
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
        return 'Jwt(iss={!r}, sub={!r}, aud={!r}, exp={!r}, name={!r}, preferred_username={!r}, customers={!r})'.format(self.issuer, self.subject, self.audience, self.expiration, self.name, self.preferred_username, self.customers)
