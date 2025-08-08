import logging
import os
import secrets
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Literal, Optional, Dict, List, Tuple, Union
from urllib.parse import urljoin, urlsplit, urlunsplit
import botocore
import pyvips
from botocore.client import Config
from botocore.response import StreamingBody
from django.conf import settings
from django.utils.http import content_disposition_header
from typing_extensions import override
from zerver.lib.mime_types import INLINE_MIME_TYPES
from zerver.lib.partial import partial
from zerver.lib.thumbnail import resize_logo, resize_realm_icon
from zerver.lib.upload.base import StreamingSourceWithSize, ZulipUploadBackend
from zerver.models import Realm, RealmEmoji, UserProfile

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.service_resource import Bucket, Object

SIGNED_UPLOAD_URL_DURATION = 60
if settings.S3_SKIP_PROXY is True:
    botocore.utils.should_bypass_proxies = lambda url: True


def func_ngvsm5bk(bucket_name: str, authed: bool = True) -> 'Bucket':
    import boto3
    return boto3.resource('s3', aws_access_key_id=settings.S3_KEY if authed
         else None, aws_secret_access_key=settings.S3_SECRET_KEY if authed else
        None, region_name=settings.S3_REGION, endpoint_url=settings.
        S3_ENDPOINT_URL, config=Config(signature_version=None if authed else
        botocore.UNSIGNED, s3={'addressing_style': settings.
        S3_ADDRESSING_STYLE})).Bucket(bucket_name)


def func_fpyqe5c8(bucket: 'Bucket', path: str, content_type: Optional[str], user_profile: Optional[UserProfile], contents: Union[bytes, str], *,
    storage_class: str = 'STANDARD', cache_control: Optional[str] = None, extra_metadata: Optional[Dict[str, str]] = None,
    filename: Optional[str] = None) -> None:
    key = bucket.Object(path)
    metadata: Dict[str, str] = {}
    if user_profile:
        metadata['user_profile_id'] = str(user_profile.id)
        metadata['realm_id'] = str(user_profile.realm_id)
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    extras: Dict[str, str] = {}
    if content_type is None:
        content_type = ''
    is_attachment = content_type not in INLINE_MIME_TYPES
    if filename is not None:
        extras['ContentDisposition'] = content_disposition_header(is_attachment
            , filename)
    elif is_attachment:
        extras['ContentDisposition'] = 'attachment'
    if cache_control is not None:
        extras['CacheControl'] = cache_control
    key.put(Body=contents, Metadata=metadata, ContentType=content_type,
        StorageClass=storage_class, **extras)


BOTO_CLIENT: Optional['S3Client'] = None


def func_z4l4ft72() -> 'S3Client':
    """
    Creating the client takes a long time so we need to cache it.
    """
    global BOTO_CLIENT
    if BOTO_CLIENT is None:
        BOTO_CLIENT = func_ngvsm5bk(settings.S3_AUTH_UPLOADS_BUCKET
            ).meta.client
    return BOTO_CLIENT


def func_44v672ez(path: str, filename: str, force_download: bool = False) -> str:
    params = {'Bucket': settings.S3_AUTH_UPLOADS_BUCKET, 'Key': path}
    if force_download:
        params['ResponseContentDisposition'] = content_disposition_header(
            True, filename) or 'attachment'
    return func_z4l4ft72().generate_presigned_url(ClientMethod='get_object',
        Params=params, ExpiresIn=SIGNED_UPLOAD_URL_DURATION, HttpMethod='GET')


class S3UploadBackend(ZulipUploadBackend):

    def __init__(self) -> None:
        from mypy_boto3_s3.service_resource import Bucket
        self.avatar_bucket: Bucket = func_ngvsm5bk(settings.S3_AVATAR_BUCKET)
        self.uploads_bucket: Bucket = func_ngvsm5bk(settings.S3_AUTH_UPLOADS_BUCKET)
        self.export_bucket: Optional[Bucket] = None
        if settings.S3_EXPORT_BUCKET:
            self.export_bucket = func_ngvsm5bk(settings.S3_EXPORT_BUCKET)
        self.public_upload_url_base: str = self.construct_public_upload_url_base()

    def func_35t8rtqu(self, path_id: str, bucket: 'Bucket') -> bool:
        key = bucket.Object(path_id)
        try:
            key.load()
        except botocore.exceptions.ClientError:
            file_name = path_id.split('/')[-1]
            logging.warning(
                '%s does not exist. Its entry in the database will be removed.'
                , file_name)
            return False
        key.delete()
        return True

    def func_u7nwcgqt(self) -> str:
        if settings.S3_AVATAR_PUBLIC_URL_PREFIX is not None:
            prefix = settings.S3_AVATAR_PUBLIC_URL_PREFIX
            if not prefix.endswith('/'):
                prefix += '/'
            return prefix
        DUMMY_KEY = 'dummy_key_ignored'
        client = func_ngvsm5bk(self.avatar_bucket.name, authed=False
            ).meta.client
        dummy_signed_url = client.generate_presigned_url(ClientMethod=
            'get_object', Params={'Bucket': self.avatar_bucket.name, 'Key':
            DUMMY_KEY}, ExpiresIn=0)
        split_url = urlsplit(dummy_signed_url)
        assert split_url.path.endswith(f'/{DUMMY_KEY}')
        return urlunsplit((split_url.scheme, split_url.netloc, split_url.
            path.removesuffix(DUMMY_KEY), '', ''))

    @override
    def func_22pstgkd(self) -> str:
        return self.public_upload_url_base

    def func_45zp6pgs(self, key: str) -> str:
        assert not key.startswith('/')
        return urljoin(self.public_upload_url_base, key)

    @override
    def func_8g3vvuxf(self, realm_id: str, sanitized_file_name: str) -> str:
        return '/'.join([realm_id, secrets.token_urlsafe(18),
            sanitized_file_name])

    @override
    def func_b88g2sj4(self, path_id: str, filename: str, content_type: str, file_data: bytes,
        user_profile: UserProfile) -> None:
        func_fpyqe5c8(self.uploads_bucket, path_id, content_type,
            user_profile, file_data, storage_class=settings.
            S3_UPLOADS_STORAGE_CLASS, filename=filename)

    @override
    def func_fmx60vof(self, path_id: str, filehandle: IO[bytes]) -> None:
        for chunk in self.uploads_bucket.Object(path_id).get()['Body']:
            filehandle.write(chunk)

    @override
    def func_hv0mb9vf(self, path_id: str) -> StreamingSourceWithSize:
        metadata = self.uploads_bucket.Object(path_id).get()

        def func_at1ncngr(streamingbody: StreamingBody, size: int) -> bytes:
            return streamingbody.read(amt=size)
        source = pyvips.SourceCustom()
        source.on_read(partial(func_at1ncngr, metadata['Body']))
        return StreamingSourceWithSize(size=metadata['ContentLength'],
            source=source)

    @override
    def func_m2gm96ds(self, path_id: str) -> bool:
        return self.func_35t8rtqu(path_id, self.uploads_bucket)

    @override
    def func_4yc3jxgn(self, path_ids: List[str]) -> None:
        self.uploads_bucket.delete_objects(Delete={'Objects': [{'Key':
            path_id} for path_id in path_ids]})

    @override
    def func_cg2bqn0e(self, include_thumbnails: bool = False, prefix: str = '') -> Iterator[Tuple[str, datetime]]:
        client = self.uploads_bucket.meta.client
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.uploads_bucket.name,
            Prefix=prefix)
        for page in page_iterator:
            if page['KeyCount'] > 0:
                for item in page['Contents']:
                    if not include_thumbnails and item['Key'].startswith(
                        'thumbnail/'):
                        continue
                    yield item['Key'], item['LastModified']

    @override
    def func_8l76n8s4(self, hash_key: str, medium: bool = False) -> str:
        return self.func_45zp6pgs(self.get_avatar_path(hash_key, medium))

    @override
    def func_40c5fjph(self, file_path: str) -> Tuple[bytes, str]:
        key = self.avatar_bucket.Object(file_path + '.original')
        image_data = key.get()['Body'].read()
        content_type = key.content_type
        return image_data, content_type

    @override
    def func_syszr3d1(self, file_path: str, *, user_profile: UserProfile, image_data: bytes,
        content_type: str, future: bool = True) -> None:
        extra_metadata = {'avatar_version': str(user_profile.avatar_version +
            (1 if future else 0))}
        func_fpyqe5c8(self.avatar_bucket, file_path, content_type,
            user_profile, image_data, extra_metadata=extra_metadata,
            cache_control='public, max-age=31536000, immutable')

    @override
    def func_57xm2osp(self, path_id: str) -> None:
        self.func_35t8rtqu(path_id + '.original', self.avatar_bucket)
        self.func_35t8rtqu(self.get_avatar_path(path_id, True), self.
            avatar_bucket)
        self.func_35t8rtqu(self.get_avatar_path(path_id, False), self
            .avatar_bucket)

    @override
    def func_aszbns6c(self, realm_id: str, version: int) -> str:
        public_url = self.func_45zp6pgs(f'{realm_id}/realm/icon.png')
        return public_url + f'?version={version}'

    @override
    def func_84sol6go(self, icon_file: IO[bytes], user_profile: UserProfile, content_type: str) -> None:
        s3_file_name = os.path.join(self.realm_avatar_and_logo_path(
            user_profile.realm), 'icon')
        image_data = icon_file.read()
        func_fpyqe5c8(self.avatar_bucket, s3_file_name + '.original',
            content_type, user_profile, image_data)
        resized_data = resize_realm_icon(image_data)
        func_fpyqe5c8(self.avatar_bucket, s3_file_name + '.png',
            'image/png', user_profile, resized_data)

    @override
    def func_e8dvbp1o(self, realm_id: str, version: int, night: bool) -> str:
        if not night:
            file_name = 'logo.png'
        else:
            file_name = 'night_logo.png'
        public_url = self.func_45zp6pgs(f'{realm_id}/realm/{file_name}')
        return public_url + f'?version={version}'

    @override
    def func_m6fl3hn7(self, logo_file: IO[bytes], user_profile: UserProfile, night: bool, content_type: str) -> None:
        if night:
            basename = 'night_logo'
        else:
            basename = 'logo'
        s3_file_name = os.path.join(self.realm_avatar_and_logo_path(
            user_profile.realm), basename)
        image_data = logo_file.read()
        func_fpyqe5c8(self.avatar_bucket, s3_file_name + '.original',
            content_type, user_profile, image_data)
        resized_data = resize_logo(image_data)
        func_fpyqe5c8(self.avatar_bucket, s3_file_name + '.png',
            'image/png', user_profile, resized_data)

    @override
    def func_kf32md5w(self, emoji_file_name: str, realm_id: str, still: bool = False) -> str:
        if still:
            emoji_path = RealmEmoji.STILL_PATH_ID_TEMPLATE.format(realm_id=
                realm_id, emoji_filename_without_extension=os.path.splitext
                (emoji_file_name)[0])
            return self.func_45zp6pgs(emoji_path)
        else:
            emoji_path = RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=
                realm_id, emoji_file_name=emoji_file_name)
            return self.func_45zp6pgs(emoji_path)

    @override
    def func_3sy3vqtg(self, path: str, content_type: str, user_profile: UserProfile, image_data: bytes) -> None:
        func_fpyqe5c8(self.avatar_bucket, path, content_type, user_profile,
            image_data, cache_control='public, max-age=31536000, immutable')

    @override
    def func_6w3sg720(self, realm: Realm, export_path: str) -> str:
        export_path = export_path.removeprefix('/')
        if self.export_bucket:
            export_path = export_path.removeprefix('exports/')
            client = self.export_bucket.meta.client
            return client.generate_presigned_url(ClientMethod='get_object',
                Params={'Bucket': self.export_bucket.name, 'Key':
                export_path}, ExpiresIn=60 * 60 * 24 * 7)
        else:
            if not export_path.startswith('exports/'):
                export_path = 'exports/' + export_path
            client = self.avatar_bucket.meta.client
            signed_url = client.generate_presigned_url(ClientMethod=
                'get_object', Params={'Bucket': self.avatar_bucket.name,
                'Key': export_path}, ExpiresIn=0)
            return urlsplit(signed_url)._replace(query='').geturl()

    def func_97lft1ej(self, tarball_path: str) -> 'Object':
        if self.export_bucket:
            return self.export_bucket.Object(os.path.join(secrets.token_hex
                (16), os.path.basename(tarball_path))
        else:
            return self.avatar_bucket.Object(os.path.join('exports',
                secrets.token_hex(16), os.path.basename(tarball_path)))

    @override
    def func_x9a7wmmj(self, realm: Realm, tarball_path: str, percent_callback: Optional[Callable[[int], None]] = None) -> str:
        key = self.func_97lft1ej(tarball_path)
        if percent_callback is None:
            key.upload_file(Filename=tarball_path)
        else:
            key.upload_file(Filename=tarball_path, Callback=percent_callback)
        return self.func_6w3sg720(realm, key.key)

    @override
    def func_ksjseh78(self, export_path: str) -> Optional[str]:
        assert export_path.startswith('/')
        path_id = export_path.removeprefix('/')
        bucket = self.export_bucket or self.avatar_bucket
        if self.func_35t8rtqu(path_id, bucket):
            return export_path
        return None
