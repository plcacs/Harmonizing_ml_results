import io
from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlencode
from multidict import MultiDict, MultiDictProxy
from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload, BytesPayload
__all__ = ('FormData',)

class FormData:
    """Helper class for form body generation.

    Supports multipart/form-data and application/x-www-form-urlencoded.
    """

    def __init__(
        self,
        fields: Optional[Union[dict, Iterable[Any]]] = (),
        quote_fields: bool = True,
        charset: Optional[str] = None,
        boundary: Optional[str] = None,
        *,
        default_to_multipart: bool = False
    ) -> None:
        self._boundary: Optional[str] = boundary
        self._writer: multipart.MultipartWriter = multipart.MultipartWriter('form-data', boundary=self._boundary)
        self._fields: List[Tuple[MultiDict, dict, Any]] = []
        self._is_multipart: bool = default_to_multipart
        self._is_processed: bool = False
        self._quote_fields: bool = quote_fields
        self._charset: Optional[str] = charset
        if isinstance(fields, dict):
            fields = list(fields.items())
        elif not isinstance(fields, (list, tuple)):
            fields = (fields,)
        self.add_fields(*fields)

    @property
    def is_multipart(self) -> bool:
        return self._is_multipart

    def add_field(
        self,
        name: str,
        value: Union[str, bytes, bytearray, memoryview, io.IOBase],
        *,
        content_type: Optional[str] = None,
        filename: Optional[str] = None
    ) -> None:
        if isinstance(value, (io.IOBase, bytes, bytearray, memoryview)):
            self._is_multipart = True
        type_options: MultiDict = MultiDict({'name': name})
        if filename is not None and not isinstance(filename, str):
            raise TypeError(f'filename must be an instance of str. Got: {filename}')
        if filename is None and isinstance(value, io.IOBase):
            filename = guess_filename(value, name)
        if filename is not None:
            type_options['filename'] = filename
            self._is_multipart = True
        headers: dict = {}
        if content_type is not None:
            if not isinstance(content_type, str):
                raise TypeError(f'content_type must be an instance of str. Got: {content_type}')
            headers[hdrs.CONTENT_TYPE] = content_type
            self._is_multipart = True
        self._fields.append((type_options, headers, value))

    def add_fields(self, *fields: Any) -> None:
        to_add: List[Any] = list(fields)
        while to_add:
            rec: Any = to_add.pop(0)
            if isinstance(rec, io.IOBase):
                k: str = guess_filename(rec, 'unknown')
                self.add_field(k, rec)
            elif isinstance(rec, (MultiDictProxy, MultiDict)):
                to_add.extend(rec.items())
            elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                k, fp = rec
                self.add_field(k, fp)
            else:
                raise TypeError(
                    f'Only io.IOBase, multidict and (name, file) pairs allowed, use .add_field() for passing more complex parameters, got {rec!r}'
                )

    def _gen_form_urlencoded(self) -> Payload:
        data: List[Tuple[str, str]] = []
        for type_options, _, value in self._fields:
            if not isinstance(value, str):
                raise TypeError(f'expected str, got {value!r}')
            data.append((type_options['name'], value))
        charset: str = self._charset if self._charset is not None else 'utf-8'
        if charset == 'utf-8':
            content_type: str = 'application/x-www-form-urlencoded'
        else:
            content_type = f'application/x-www-form-urlencoded; charset={charset}'
        return payload.BytesPayload(urlencode(data, doseq=True, encoding=charset).encode(), content_type=content_type)

    def _gen_form_data(self) -> multipart.MultipartWriter:
        """Encode a list of fields using the multipart/form-data MIME format"""
        if self._is_processed:
            raise RuntimeError('Form data has been processed already')
        for dispparams, headers, value in self._fields:
            try:
                if hdrs.CONTENT_TYPE in headers:
                    part = payload.get_payload(
                        value,
                        content_type=headers[hdrs.CONTENT_TYPE],
                        headers=headers,
                        encoding=self._charset
                    )
                else:
                    part = payload.get_payload(
                        value,
                        headers=headers,
                        encoding=self._charset
                    )
            except Exception as exc:
                raise TypeError(
                    f'Can not serialize value type: {type(value)!r}\n headers: {headers!r}\n value: {value!r}'
                ) from exc
            if dispparams:
                part.set_content_disposition('form-data', quote_fields=self._quote_fields, **dispparams)
                assert part.headers is not None
                part.headers.popall(hdrs.CONTENT_LENGTH, None)
            self._writer.append_payload(part)
        self._is_processed = True
        return self._writer

    def __call__(self) -> Union[multipart.MultipartWriter, Payload]:
        if self._is_multipart:
            return self._gen_form_data()
        else:
            return self._gen_form_urlencoded()
