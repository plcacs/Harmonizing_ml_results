"""Minio helper methods."""
from __future__ import annotations
from collections.abc import Iterable
import json
import logging
from queue import Queue
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Self
from urllib.parse import unquote
from minio import Minio
from urllib3.exceptions import HTTPError

_LOGGER = logging.getLogger(__name__)
_METADATA_RE = re.compile(r'x-amz-meta-(.*)', re.IGNORECASE)

def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize object metadata by stripping the prefix."""
    new_metadata: Dict[str, Any] = {}
    for meta_key, meta_value in metadata.items():
        match = _METADATA_RE.match(meta_key)
        if not match:
            continue
        new_metadata[match.group(1).lower()] = meta_value
    return new_metadata

def create_minio_client(endpoint: str, access_key: str, secret_key: str, secure: bool) -> Minio:
    """Create Minio client."""
    return Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

def get_minio_notification_response(
    minio_client: Minio,
    bucket_name: str,
    prefix: str,
    suffix: str,
    events: List[str]
) -> Any:
    """Start listening to minio events. Copied from minio-py."""
    query: Dict[str, Any] = {'prefix': prefix, 'suffix': suffix, 'events': events}
    return minio_client._url_open('GET', bucket_name=bucket_name, query=query, preload_content=False)

class MinioEventStreamIterator(Iterable[Dict[str, Any]]):
    """Iterator wrapper over notification HTTP response stream."""

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return self."""
        return self

    def __init__(self, response: Any) -> None:
        """Initialize the stream iterator."""
        self._response = response
        self._stream = response.stream()

    def __next__(self) -> Dict[str, Any]:
        """Get the next non-empty event."""
        while True:
            line = next(self._stream)
            if line.strip():
                event = json.loads(line.decode('utf-8'))
                if event.get('Records') is not None:
                    return event
        raise StopIteration

    def close(self) -> None:
        """Close the response."""
        self._response.close()

class MinioEventThread(threading.Thread):
    """Thread wrapper around Minio notification blocking stream."""

    def __init__(
        self,
        queue: Queue[Dict[str, Any]],
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
        bucket_name: str,
        prefix: str,
        suffix: str,
        events: List[str]
    ) -> None:
        """Initialize the event thread with Minio client options."""
        super().__init__()
        self._queue = queue
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._bucket_name = bucket_name
        self._prefix = prefix
        self._suffix = suffix
        self._events = events
        self._event_stream_it: Optional[MinioEventStreamIterator] = None
        self._should_stop: bool = False

    def __enter__(self) -> Self:
        """Start the thread upon entering the context."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop and join the thread upon exiting the context."""
        self.stop()

    def run(self) -> None:
        """Create Minio client and run the event listening loop."""
        _LOGGER.debug('Running MinioEventThread')
        self._should_stop = False
        minio_client = create_minio_client(
            self._endpoint,
            self._access_key,
            self._secret_key,
            self._secure
        )
        while not self._should_stop:
            _LOGGER.debug('Connecting to Minio event stream')
            response: Optional[Any] = None
            try:
                response = get_minio_notification_response(
                    minio_client,
                    self._bucket_name,
                    self._prefix,
                    self._suffix,
                    self._events
                )
                self._event_stream_it = MinioEventStreamIterator(response)
                self._iterate_event_stream(self._event_stream_it, minio_client)
            except json.JSONDecodeError:
                if response:
                    response.close()
            except HTTPError as error:
                _LOGGER.error('Failed to connect to Minio endpoint: %s', error)
                time.sleep(1)
            except AttributeError:
                break

    def _iterate_event_stream(
        self,
        event_stream_it: MinioEventStreamIterator,
        minio_client: Minio
    ) -> None:
        """Process events from the event stream iterator."""
        for event in event_stream_it:
            for event_name, bucket, key, metadata in iterate_objects(event):
                presigned_url: str = ''
                try:
                    presigned_url = minio_client.presigned_get_object(bucket, key)
                except Exception as error:
                    _LOGGER.error('Failed to generate presigned URL: %s', error)
                queue_entry: Dict[str, Any] = {
                    'event_name': event_name,
                    'bucket': bucket,
                    'key': key,
                    'presigned_url': presigned_url,
                    'metadata': metadata
                }
                _LOGGER.debug('Queue entry: %s', queue_entry)
                self._queue.put(queue_entry)

    def stop(self) -> None:
        """Cancel event stream and join the thread."""
        _LOGGER.debug('Stopping event thread')
        self._should_stop = True
        if self._event_stream_it is not None:
            self._event_stream_it.close()
            self._event_stream_it = None
        _LOGGER.debug('Joining event thread')
        self.join()
        _LOGGER.debug('Event thread joined')

def iterate_objects(event: Dict[str, Any]) -> Iterator[Tuple[str, str, str, Dict[str, Any]]]:
    """Iterate over file records of a notification event.

    Most of the time it should still be only one record.
    """
    records = event.get('Records', [])
    for record in records:
        event_name: Optional[str] = record.get('eventName')
        bucket: Optional[str] = record.get('s3', {}).get('bucket', {}).get('name')
        key: Optional[str] = record.get('s3', {}).get('object', {}).get('key')
        metadata: Dict[str, Any] = normalize_metadata(
            record.get('s3', {}).get('object', {}).get('userMetadata', {})
        )
        if not bucket or not key:
            _LOGGER.warning('Invalid bucket and/or key: %s, %s', bucket, key)
            continue
        key = unquote(key)
        yield (event_name, bucket, key, metadata)
