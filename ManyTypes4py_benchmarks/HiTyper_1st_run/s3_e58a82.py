import logging
import os
import secrets
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Literal
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

def get_bucket(bucket_name: str, authed: bool=True) -> Union[bool, str, dict]:
    import boto3
    return boto3.resource('s3', aws_access_key_id=settings.S3_KEY if authed else None, aws_secret_access_key=settings.S3_SECRET_KEY if authed else None, region_name=settings.S3_REGION, endpoint_url=settings.S3_ENDPOINT_URL, config=Config(signature_version=None if authed else botocore.UNSIGNED, s3={'addressing_style': settings.S3_ADDRESSING_STYLE})).Bucket(bucket_name)

def upload_content_to_s3(bucket: Union[str, dict, dict[str, typing.Any]], path: Union[str, dict, dict[str, typing.Any]], content_type: Union[str, typing.Any, None], user_profile: Union[str, int], contents: Union[str, dict[str, str], typing.Any, None], *, storage_class: typing.Text='STANDARD', cache_control: Union[None, str]=None, extra_metadata: Union[str, None, dict[str, typing.Any], list[annofabapi.models.LabelV1]]=None, filename: Union[None, str]=None) -> None:
    key = bucket.Object(path)
    metadata = {}
    if user_profile:
        metadata['user_profile_id'] = str(user_profile.id)
        metadata['realm_id'] = str(user_profile.realm_id)
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    extras = {}
    if content_type is None:
        content_type = ''
    is_attachment = content_type not in INLINE_MIME_TYPES
    if filename is not None:
        extras['ContentDisposition'] = content_disposition_header(is_attachment, filename)
    elif is_attachment:
        extras['ContentDisposition'] = 'attachment'
    if cache_control is not None:
        extras['CacheControl'] = cache_control
    key.put(Body=contents, Metadata=metadata, ContentType=content_type, StorageClass=storage_class, **extras)
BOTO_CLIENT = None

def get_boto_client() -> Union[list, set]:
    """
    Creating the client takes a long time so we need to cache it.
    """
    global BOTO_CLIENT
    if BOTO_CLIENT is None:
        BOTO_CLIENT = get_bucket(settings.S3_AUTH_UPLOADS_BUCKET).meta.client
    return BOTO_CLIENT

def get_signed_upload_url(path: Union[str, bool], filename: Union[bool, str, typing.BinaryIO], force_download: bool=False) -> Union[str, dict]:
    params = {'Bucket': settings.S3_AUTH_UPLOADS_BUCKET, 'Key': path}
    if force_download:
        params['ResponseContentDisposition'] = content_disposition_header(True, filename) or 'attachment'
    return get_boto_client().generate_presigned_url(ClientMethod='get_object', Params=params, ExpiresIn=SIGNED_UPLOAD_URL_DURATION, HttpMethod='GET')

class S3UploadBackend(ZulipUploadBackend):

    def __init__(self) -> None:
        from mypy_boto3_s3.service_resource import Bucket
        self.avatar_bucket = get_bucket(settings.S3_AVATAR_BUCKET)
        self.uploads_bucket = get_bucket(settings.S3_AUTH_UPLOADS_BUCKET)
        self.export_bucket = None
        if settings.S3_EXPORT_BUCKET:
            self.export_bucket = get_bucket(settings.S3_EXPORT_BUCKET)
        self.public_upload_url_base = self.construct_public_upload_url_base()

    def delete_file_from_s3(self, path_id: str, bucket: Union[str, boto3.resources.base.ServiceResource, None]) -> bool:
        key = bucket.Object(path_id)
        try:
            key.load()
        except botocore.exceptions.ClientError:
            file_name = path_id.split('/')[-1]
            logging.warning('%s does not exist. Its entry in the database will be removed.', file_name)
            return False
        key.delete()
        return True

    def construct_public_upload_url_base(self) -> str:
        if settings.S3_AVATAR_PUBLIC_URL_PREFIX is not None:
            prefix = settings.S3_AVATAR_PUBLIC_URL_PREFIX
            if not prefix.endswith('/'):
                prefix += '/'
            return prefix
        DUMMY_KEY = 'dummy_key_ignored'
        client = get_bucket(self.avatar_bucket.name, authed=False).meta.client
        dummy_signed_url = client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': self.avatar_bucket.name, 'Key': DUMMY_KEY}, ExpiresIn=0)
        split_url = urlsplit(dummy_signed_url)
        assert split_url.path.endswith(f'/{DUMMY_KEY}')
        return urlunsplit((split_url.scheme, split_url.netloc, split_url.path.removesuffix(DUMMY_KEY), '', ''))

    @override
    def get_public_upload_root_url(self):
        return self.public_upload_url_base

    def get_public_upload_url(self, key: str):
        assert not key.startswith('/')
        return urljoin(self.public_upload_url_base, key)

    @override
    def generate_message_upload_path(self, realm_id: Union[str, int, tuple[str]], sanitized_file_name: Union[str, int, tuple[str]]) -> str:
        return '/'.join([realm_id, secrets.token_urlsafe(18), sanitized_file_name])

    @override
    def upload_message_attachment(self, path_id: Union[str, bool], filename: Union[str, bool], content_type: Union[str, bool], file_data: Union[str, bool], user_profile: Union[str, bool]) -> None:
        upload_content_to_s3(self.uploads_bucket, path_id, content_type, user_profile, file_data, storage_class=settings.S3_UPLOADS_STORAGE_CLASS, filename=filename)

    @override
    def save_attachment_contents(self, path_id: str, filehandle: Union[str, bytes]) -> None:
        for chunk in self.uploads_bucket.Object(path_id).get()['Body']:
            filehandle.write(chunk)

    @override
    def attachment_vips_source(self, path_id: Union[str, bool, list[str]]) -> StreamingSourceWithSize:
        metadata = self.uploads_bucket.Object(path_id).get()

        def s3_read(streamingbody: Any, size: Any):
            return streamingbody.read(amt=size)
        source = pyvips.SourceCustom()
        source.on_read(partial(s3_read, metadata['Body']))
        return StreamingSourceWithSize(size=metadata['ContentLength'], source=source)

    @override
    def delete_message_attachment(self, path_id: str):
        return self.delete_file_from_s3(path_id, self.uploads_bucket)

    @override
    def delete_message_attachments(self, path_ids: Union[str, list[int]]) -> None:
        self.uploads_bucket.delete_objects(Delete={'Objects': [{'Key': path_id} for path_id in path_ids]})

    @override
    def all_message_attachments(self, include_thumbnails: bool=False, prefix: typing.Text='') -> typing.Generator[tuple]:
        client = self.uploads_bucket.meta.client
        paginator = client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.uploads_bucket.name, Prefix=prefix)
        for page in page_iterator:
            if page['KeyCount'] > 0:
                for item in page['Contents']:
                    if not include_thumbnails and item['Key'].startswith('thumbnail/'):
                        continue
                    yield (item['Key'], item['LastModified'])

    @override
    def get_avatar_url(self, hash_key: Union[str, bytes, bool], medium: bool=False) -> Union[str, None]:
        return self.get_public_upload_url(self.get_avatar_path(hash_key, medium))

    @override
    def get_avatar_contents(self, file_path: Union[str, list[int]]) -> tuple:
        key = self.avatar_bucket.Object(file_path + '.original')
        image_data = key.get()['Body'].read()
        content_type = key.content_type
        return (image_data, content_type)

    @override
    def upload_single_avatar_image(self, file_path: Union[str, bool], *, user_profile: Union[str, bool, None], image_data: Union[str, bool], content_type: Union[str, bool], future: bool=True) -> None:
        extra_metadata = {'avatar_version': str(user_profile.avatar_version + (1 if future else 0))}
        upload_content_to_s3(self.avatar_bucket, file_path, content_type, user_profile, image_data, extra_metadata=extra_metadata, cache_control='public, max-age=31536000, immutable')

    @override
    def delete_avatar_image(self, path_id: str) -> None:
        self.delete_file_from_s3(path_id + '.original', self.avatar_bucket)
        self.delete_file_from_s3(self.get_avatar_path(path_id, True), self.avatar_bucket)
        self.delete_file_from_s3(self.get_avatar_path(path_id, False), self.avatar_bucket)

    @override
    def get_realm_icon_url(self, realm_id: int, version: int) -> str:
        public_url = self.get_public_upload_url(f'{realm_id}/realm/icon.png')
        return public_url + f'?version={version}'

    @override
    def upload_realm_icon_image(self, icon_file: django.core.files.File, user_profile: Union[zerver.models.UserProfile, str, zerver.models.Stream], content_type: Union[str, zerver.models.UserProfile, zerver.models.Stream]) -> None:
        s3_file_name = os.path.join(self.realm_avatar_and_logo_path(user_profile.realm), 'icon')
        image_data = icon_file.read()
        upload_content_to_s3(self.avatar_bucket, s3_file_name + '.original', content_type, user_profile, image_data)
        resized_data = resize_realm_icon(image_data)
        upload_content_to_s3(self.avatar_bucket, s3_file_name + '.png', 'image/png', user_profile, resized_data)

    @override
    def get_realm_logo_url(self, realm_id: Union[int, str], version: int, night: bool) -> str:
        if not night:
            file_name = 'logo.png'
        else:
            file_name = 'night_logo.png'
        public_url = self.get_public_upload_url(f'{realm_id}/realm/{file_name}')
        return public_url + f'?version={version}'

    @override
    def upload_realm_logo_image(self, logo_file: django.core.files.File, user_profile: Union[str, zerver.models.UserProfile, bool], night: Union[bool, zerver.models.Realm, str], content_type: Union[str, bool]) -> None:
        if night:
            basename = 'night_logo'
        else:
            basename = 'logo'
        s3_file_name = os.path.join(self.realm_avatar_and_logo_path(user_profile.realm), basename)
        image_data = logo_file.read()
        upload_content_to_s3(self.avatar_bucket, s3_file_name + '.original', content_type, user_profile, image_data)
        resized_data = resize_logo(image_data)
        upload_content_to_s3(self.avatar_bucket, s3_file_name + '.png', 'image/png', user_profile, resized_data)

    @override
    def get_emoji_url(self, emoji_file_name: Union[int, str, list[int]], realm_id: Union[int, str, list[int]], still: bool=False) -> Union[str, typing.BinaryIO, None]:
        if still:
            emoji_path = RealmEmoji.STILL_PATH_ID_TEMPLATE.format(realm_id=realm_id, emoji_filename_without_extension=os.path.splitext(emoji_file_name)[0])
            return self.get_public_upload_url(emoji_path)
        else:
            emoji_path = RealmEmoji.PATH_ID_TEMPLATE.format(realm_id=realm_id, emoji_file_name=emoji_file_name)
            return self.get_public_upload_url(emoji_path)

    @override
    def upload_single_emoji_image(self, path: Union[str, bytes, None], content_type: Union[str, bytes, None], user_profile: Union[str, bytes, None], image_data: Union[str, bytes, None]) -> None:
        upload_content_to_s3(self.avatar_bucket, path, content_type, user_profile, image_data, cache_control='public, max-age=31536000, immutable')

    @override
    def get_export_tarball_url(self, realm: Union[zerver.models.Realm, str, zerver.models.UserProfile], export_path: Union[str, zerver.models.Realm, abilian.core.models.subjects.User]) -> Union[str, int, None]:
        export_path = export_path.removeprefix('/')
        if self.export_bucket:
            export_path = export_path.removeprefix('exports/')
            client = self.export_bucket.meta.client
            return client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': self.export_bucket.name, 'Key': export_path}, ExpiresIn=60 * 60 * 24 * 7)
        else:
            if not export_path.startswith('exports/'):
                export_path = 'exports/' + export_path
            client = self.avatar_bucket.meta.client
            signed_url = client.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': self.avatar_bucket.name, 'Key': export_path}, ExpiresIn=0)
            return urlsplit(signed_url)._replace(query='').geturl()

    def export_object(self, tarball_path: str) -> Union[str, tuple[str]]:
        if self.export_bucket:
            return self.export_bucket.Object(os.path.join(secrets.token_hex(16), os.path.basename(tarball_path)))
        else:
            return self.avatar_bucket.Object(os.path.join('exports', secrets.token_hex(16), os.path.basename(tarball_path)))

    @override
    def upload_export_tarball(self, realm: Union[str, zerver.models.Realm], tarball_path: Union[typing.Callable[typing.Any, None], None, str, bool], percent_callback: Union[None, typing.Callable[typing.Any, None], bool, str]=None):
        key = self.export_object(tarball_path)
        if percent_callback is None:
            key.upload_file(Filename=tarball_path)
        else:
            key.upload_file(Filename=tarball_path, Callback=percent_callback)
        return self.get_export_tarball_url(realm, key.key)

    @override
    def delete_export_tarball(self, export_path: str) -> Union[str, None]:
        assert export_path.startswith('/')
        path_id = export_path.removeprefix('/')
        bucket = self.export_bucket or self.avatar_bucket
        if self.delete_file_from_s3(path_id, bucket):
            return export_path
        return None