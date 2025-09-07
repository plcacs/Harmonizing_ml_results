import base64
import binascii
import hashlib
import json
import os
from Cryptodome.Cipher import AES
from typing import Any, Dict

__all__ = ['encrypted_id', 'encrypted_request']

MODULUS: str = '00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7'
PUBKEY: str = '010001'
NONCE: bytes = b'0CoJUm6Qyw8W8jud'

def encrypted_id(id: str) -> str:
    magic: bytearray = bytearray('3go8&$8*3*3h0k(2)2', 'u8')
    song_id: bytearray = bytearray(id, 'u8')
    magic_len: int = len(magic)
    for i, sid in enumerate(song_id):
        song_id[i] = sid ^ magic[i % magic_len]
    m = hashlib.md5(song_id)
    result: bytes = m.digest()
    result = base64.b64encode(result).replace(b'/', b'_').replace(b'+', b'-')
    return result.decode('utf-8')

def encrypted_request(text: Any) -> Dict[str, str]:
    data: bytes = json.dumps(text).encode('utf-8')
    secret: bytes = create_key(16)
    params_bytes: bytes = aes(aes(data, NONCE), secret)
    params: str = params_bytes.decode('utf-8')
    encseckey: str = rsa(secret, PUBKEY, MODULUS)
    return {'params': params, 'encSecKey': encseckey}

def aes(text: bytes, key: bytes) -> bytes:
    pad: int = 16 - len(text) % 16
    text += bytearray([pad] * pad)
    encryptor = AES.new(key, AES.MODE_CBC, b'0102030405060708')
    ciphertext: bytes = encryptor.encrypt(text)
    return base64.b64encode(ciphertext)

def rsa(text: bytes, pubkey: str, modulus: str) -> str:
    text = text[::-1]
    num: int = int(binascii.hexlify(text), 16)
    exponent: int = int(pubkey, 16)
    mod: int = int(modulus, 16)
    rs: int = pow(num, exponent, mod)
    return format(rs, 'x').zfill(256)

def create_key(size: int) -> bytes:
    # Generate random bytes, hexlify them, then take the first 16 bytes.
    return binascii.hexlify(os.urandom(size))[:16]