import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar
import pyvips
from bs4 import BeautifulSoup
from bs4.formatter import EntitySubstitution, HTMLFormatter
from django.utils.translation import gettext as _
from typing_extensions import override
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.mime_types import INLINE_MIME_TYPES
from zerver.lib.queue import queue_event_on_commit
from zerver.models import ImageAttachment
DEFAULT_AVATAR_SIZE = 100
MEDIUM_AVATAR_SIZE = 500
DEFAULT_EMOJI_SIZE = 64
IMAGE_BOMB_TOTAL_PIXELS = 90000000
IMAGE_MAX_ANIMATED_PIXELS = IMAGE_BOMB_TOTAL_PIXELS / 3
MAX_EMOJI_GIF_FILE_SIZE_BYTES = 128 * 1024
T = TypeVar('T', bound='BaseThumbnailFormat')

@dataclass(frozen=True)
class BaseThumbnailFormat:

    @override
    def __eq__(self, other):
        if not isinstance(other, BaseThumbnailFormat):
            return False
        return str(self) == str(other)

    @override
    def __str__(self):
        animated = '-anim' if self.animated else ''
        return f'{self.max_width}x{self.max_height}{animated}.{self.extension}'

    @classmethod
    def from_string(cls, format_string):
        format_parts = re.match('(\\d+)x(\\d+)(-anim)?\\.(\\w+)$', format_string)
        if format_parts is None:
            return None
        return cls(max_width=int(format_parts[1]), max_height=int(format_parts[2]), animated=format_parts[3] is not None, extension=format_parts[4])

@dataclass(frozen=True, eq=False)
class ThumbnailFormat(BaseThumbnailFormat):
    opts = ''

@dataclass(frozen=True, eq=False)
class StoredThumbnailFormat(BaseThumbnailFormat):
    pass
THUMBNAIL_OUTPUT_FORMATS = (ThumbnailFormat('webp', 840, 560, animated=True), ThumbnailFormat('webp', 840, 560, animated=False))
TRANSCODED_IMAGE_FORMAT = ThumbnailFormat('webp', 4032, 3024, animated=False)
THUMBNAIL_ACCEPT_IMAGE_TYPES = frozenset(['image/avif', 'image/gif', 'image/heic', 'image/jpeg', 'image/png', 'image/tiff', 'image/webp'])
pyvips.operation_block_set('VipsForeignLoad', True)
pyvips.operation_block_set('VipsForeignLoadHeif', False)
pyvips.operation_block_set('VipsForeignLoadNsgif', False)
pyvips.operation_block_set('VipsForeignLoadJpeg', False)
pyvips.operation_block_set('VipsForeignLoadPng', False)
pyvips.operation_block_set('VipsForeignLoadTiff', False)
pyvips.operation_block_set('VipsForeignLoadWebp', False)
pyvips.block_untrusted_set(True)
pyvips.voperation.cache_set_max(0)

class BadImageError(JsonableError):
    code = ErrorCode.BAD_IMAGE

@contextmanager
def libvips_check_image(image_data, truncated_animation=False):
    try:
        if isinstance(image_data, bytes):
            source_image = pyvips.Image.new_from_buffer(image_data, '')
        else:
            source_image = pyvips.Image.new_from_source(image_data, '', access='sequential')
    except pyvips.Error:
        raise BadImageError(_('Could not decode image; did you upload an image file?'))
    if not truncated_animation:
        if source_image.width * source_image.height * source_image.get_n_pages() > IMAGE_BOMB_TOTAL_PIXELS:
            raise BadImageError(_('Image size exceeds limit.'))
    elif source_image.get_n_pages() == 1:
        if source_image.width * source_image.height > IMAGE_BOMB_TOTAL_PIXELS:
            raise BadImageError(_('Image size exceeds limit.'))
    elif source_image.width * source_image.height * min(3, source_image.get_n_pages()) > IMAGE_MAX_ANIMATED_PIXELS:
        raise BadImageError(_('Image size exceeds limit.'))
    try:
        yield source_image
    except pyvips.Error as e:
        logging.exception(e)
        raise BadImageError(_('Image is corrupted or truncated'))

def resize_avatar(image_data, size=DEFAULT_AVATAR_SIZE):
    with libvips_check_image(image_data):
        return pyvips.Image.thumbnail_buffer(image_data, size, height=size, crop=pyvips.Interesting.CENTRE).write_to_buffer('.png')

def resize_realm_icon(image_data):
    return resize_avatar(image_data)

def resize_logo(image_data):
    with libvips_check_image(image_data):
        return pyvips.Image.thumbnail_buffer(image_data, 8 * DEFAULT_AVATAR_SIZE, height=DEFAULT_AVATAR_SIZE, size=pyvips.Size.DOWN).write_to_buffer('.png')

def resize_emoji(image_data, emoji_file_name, size=DEFAULT_EMOJI_SIZE):
    write_file_ext = os.path.splitext(emoji_file_name)[1]
    assert '[' not in write_file_ext
    with libvips_check_image(image_data) as source_image:
        if source_image.get_n_pages() == 1:
            return (pyvips.Image.thumbnail_buffer(image_data, size, height=size, crop=pyvips.Interesting.CENTRE).write_to_buffer(write_file_ext), None)
        first_still = pyvips.Image.thumbnail_buffer(image_data, size, height=size, crop=pyvips.Interesting.CENTRE).write_to_buffer('.png')
        animated = pyvips.Image.thumbnail_buffer(image_data, size, height=size, option_string='n=-1')
        if animated.width != animated.get('page-height'):
            if not animated.hasalpha():
                animated = animated.addalpha()
            frames = [frame.gravity(pyvips.CompassDirection.CENTRE, size, size, extend=pyvips.Extend.BACKGROUND, background=[0, 0, 0, 0]) for frame in animated.pagesplit()]
            animated = frames[0].pagejoin(frames[1:])
        return (animated.write_to_buffer(write_file_ext), first_still)

def missing_thumbnails(image_attachment):
    seen_thumbnails = set()
    for existing_thumbnail in image_attachment.thumbnail_metadata:
        seen_thumbnails.add(StoredThumbnailFormat(**existing_thumbnail))
    potential_output_formats = list(THUMBNAIL_OUTPUT_FORMATS)
    if image_attachment.content_type not in INLINE_MIME_TYPES:
        if image_attachment.original_width_px >= image_attachment.original_height_px:
            additional_format = ThumbnailFormat(TRANSCODED_IMAGE_FORMAT.extension, TRANSCODED_IMAGE_FORMAT.max_width, TRANSCODED_IMAGE_FORMAT.max_height, TRANSCODED_IMAGE_FORMAT.animated)
        else:
            additional_format = ThumbnailFormat(TRANSCODED_IMAGE_FORMAT.extension, TRANSCODED_IMAGE_FORMAT.max_height, TRANSCODED_IMAGE_FORMAT.max_width, TRANSCODED_IMAGE_FORMAT.animated)
        potential_output_formats.append(additional_format)
    needed_thumbnails = [thumbnail_format for thumbnail_format in potential_output_formats if thumbnail_format not in seen_thumbnails]
    if image_attachment.frames == 1:
        needed_thumbnails = [thumbnail_format for thumbnail_format in needed_thumbnails if not thumbnail_format.animated]
    return needed_thumbnails

def maybe_thumbnail(content, content_type, path_id, realm_id, skip_events=False):
    if content_type not in THUMBNAIL_ACCEPT_IMAGE_TYPES:
        return None
    try:
        with libvips_check_image(content, truncated_animation=True) as image:
            if 'orientation' in image.get_fields() and image.get('orientation') >= 5 and (image.get('orientation') <= 8):
                width, height = (image.height, image.width)
            else:
                width, height = (image.width, image.height)
            image_row = ImageAttachment.objects.create(realm_id=realm_id, path_id=path_id, original_width_px=width, original_height_px=height, frames=image.get_n_pages(), thumbnail_metadata=[], content_type=content_type)
            if not skip_events:
                queue_event_on_commit('thumbnail', {'id': image_row.id})
            return image_row
    except BadImageError:
        return None

def get_image_thumbnail_path(image_attachment, thumbnail_format):
    return f'thumbnail/{image_attachment.path_id}/{thumbnail_format!s}'

def split_thumbnail_path(file_path):
    assert file_path.startswith('thumbnail/')
    path_parts = file_path.split('/')
    thumbnail_format = BaseThumbnailFormat.from_string(path_parts.pop())
    assert thumbnail_format is not None
    path_id = '/'.join(path_parts[1:])
    return (path_id, thumbnail_format)

@dataclass
class MarkdownImageMetadata:
    transcoded_image = None

def get_user_upload_previews(realm_id, content, lock=False, enqueue=True, path_ids=None):
    if path_ids is None:
        path_ids = re.findall('/user_uploads/(\\d+/[/\\w.-]+)', content)
    if not path_ids:
        return {}
    upload_preview_data = {}
    image_attachments = ImageAttachment.objects.filter(realm_id=realm_id, path_id__in=path_ids).order_by('id')
    if lock:
        image_attachments = image_attachments.select_for_update(of=('self',))
    for image_attachment in image_attachments:
        if image_attachment.thumbnail_metadata == []:
            upload_preview_data[image_attachment.path_id] = MarkdownImageMetadata(url=None, is_animated=False, original_width_px=image_attachment.original_width_px, original_height_px=image_attachment.original_height_px, original_content_type=image_attachment.content_type)
            if enqueue:
                queue_event_on_commit('thumbnail', {'id': image_attachment.id})
        else:
            url, is_animated = get_default_thumbnail_url(image_attachment)
            upload_preview_data[image_attachment.path_id] = MarkdownImageMetadata(url=url, is_animated=is_animated, original_width_px=image_attachment.original_width_px, original_height_px=image_attachment.original_height_px, original_content_type=image_attachment.content_type, transcoded_image=get_transcoded_format(image_attachment))
    return upload_preview_data

def get_default_thumbnail_url(image_attachment):
    found_format = None
    for thumbnail_format in THUMBNAIL_OUTPUT_FORMATS:
        if thumbnail_format.animated == (image_attachment.frames > 1):
            found_format = thumbnail_format
            break
    if found_format is None:
        found_format = THUMBNAIL_OUTPUT_FORMATS[0]
    return ('/user_uploads/' + get_image_thumbnail_path(image_attachment, found_format), found_format.animated)

def get_transcoded_format(image_attachment):
    if image_attachment.content_type is None or image_attachment.content_type in INLINE_MIME_TYPES:
        return None
    thumbs_by_size = sorted((StoredThumbnailFormat(**d) for d in image_attachment.thumbnail_metadata), key=lambda t: t.width * t.height)
    return thumbs_by_size.pop() if thumbs_by_size else None
html_formatter = HTMLFormatter(entity_substitution=EntitySubstitution.substitute_xml, void_element_close_prefix='', empty_attributes_are_booleans=True)

def rewrite_thumbnailed_images(rendered_content, images, to_delete=None):
    if not images and (not to_delete):
        return (None, set())
    remaining_thumbnails = set()
    parsed_message = BeautifulSoup(rendered_content, 'html.parser')
    changed = False
    for inline_image_div in parsed_message.find_all('div', class_='message_inline_image'):
        image_link = inline_image_div.find('a')
        if image_link is None or image_link['href'] is None or (not image_link['href'].startswith('/user_uploads/')):
            continue
        image_tag = image_link.find('img', class_='image-loading-placeholder')
        if image_tag is None:
            continue
        path_id = image_link['href'].removeprefix('/user_uploads/')
        if to_delete and path_id in to_delete:
            inline_image_div.decompose()
            changed = True
            continue
        image_data = images.get(path_id)
        if image_data is None:
            remaining_thumbnails.add(path_id)
        elif image_data.url is None:
            remaining_thumbnails.add(path_id)
        else:
            changed = True
            del image_tag['class']
            image_tag['src'] = image_data.url
            image_tag['data-original-dimensions'] = f'{image_data.original_width_px}x{image_data.original_height_px}'
            image_tag['data-original-content-type'] = image_data.original_content_type
            if image_data.is_animated:
                image_tag['data-animated'] = 'true'
            if image_data.transcoded_image is not None:
                image_tag['data-transcoded-image'] = str(image_data.transcoded_image)
    if changed:
        return (parsed_message.encode(formatter=html_formatter).decode().strip(), remaining_thumbnails)
    else:
        return (None, remaining_thumbnails)