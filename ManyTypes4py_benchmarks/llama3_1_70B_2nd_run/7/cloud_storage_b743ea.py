@task
@sync_compatible
async def cloud_storage_create_bucket(
    bucket: str, 
    gcp_credentials: GcpCredentials, 
    project: Optional[str] = None, 
    location: Optional[str] = None, 
    **create_kwargs: Any
) -> str:
    ...

@task
@sync_compatible
async def cloud_storage_download_blob_as_bytes(
    bucket: str, 
    blob: str, 
    gcp_credentials: GcpCredentials, 
    chunk_size: Optional[int] = None, 
    encryption_key: Optional[str] = None, 
    timeout: Union[int, Tuple[int, int]] = 60, 
    project: Optional[str] = None, 
    **download_kwargs: Any
) -> Union[bytes, str]:
    ...

@task
@sync_compatible
async def cloud_storage_download_blob_to_file(
    bucket: str, 
    blob: str, 
    path: Union[str, Path], 
    gcp_credentials: GcpCredentials, 
    chunk_size: Optional[int] = None, 
    encryption_key: Optional[str] = None, 
    timeout: Union[int, Tuple[int, int]] = 60, 
    project: Optional[str] = None, 
    **download_kwargs: Any
) -> str:
    ...

@task
@sync_compatible
async def cloud_storage_upload_blob_from_string(
    data: Union[str, bytes], 
    bucket: str, 
    blob: str, 
    gcp_credentials: GcpCredentials, 
    content_type: Optional[str] = None, 
    chunk_size: Optional[int] = None, 
    encryption_key: Optional[str] = None, 
    timeout: Union[int, Tuple[int, int]] = 60, 
    project: Optional[str] = None, 
    **upload_kwargs: Any
) -> str:
    ...

@task
@sync_compatible
async def cloud_storage_upload_blob_from_file(
    file: Union[str, Path, BinaryIO], 
    bucket: str, 
    blob: str, 
    gcp_credentials: GcpCredentials, 
    content_type: Optional[str] = None, 
    chunk_size: Optional[int] = None, 
    encryption_key: Optional[str] = None, 
    timeout: Union[int, Tuple[int, int]] = 60, 
    project: Optional[str] = None, 
    **upload_kwargs: Any
) -> str:
    ...

@task
def cloud_storage_copy_blob(
    source_bucket: str, 
    dest_bucket: str, 
    source_blob: str, 
    gcp_credentials: GcpCredentials, 
    dest_blob: Optional[str] = None, 
    timeout: Union[int, Tuple[int, int]] = 60, 
    project: Optional[str] = None, 
    **copy_kwargs: Any
) -> str:
    ...

class DataFrameSerializationFormat(Enum):
    ...

class GcsBucket(WritableDeploymentStorage, WritableFileSystem, ObjectStorageBlock):
    ...

    @sync_compatible
    async def get_directory(
        self, 
        from_path: Optional[str] = None, 
        local_path: Optional[str] = None
    ) -> List[str]:
        ...

    @sync_compatible
    async def put_directory(
        self, 
        local_path: Optional[str] = None, 
        to_path: Optional[str] = None, 
        ignore_file: Optional[str] = None
    ) -> int:
        ...

    @sync_compatible
    async def read_path(self, path: str) -> Union[bytes, str]:
        ...

    @sync_compatible
    async def write_path(
        self, 
        path: str, 
        content: Union[str, bytes]
    ) -> str:
        ...

    def _join_bucket_folder(self, bucket_path: str = '') -> Optional[str]:
        ...

    @sync_compatible
    async def create_bucket(
        self, 
        location: Optional[str] = None, 
        **create_kwargs: Any
    ) -> Bucket:
        ...

    @sync_compatible
    async def get_bucket(self) -> Bucket:
        ...

    @sync_compatible
    async def list_blobs(self, folder: str = '') -> List[Blob]:
        ...

    @sync_compatible
    async def list_folders(self, folder: str = '') -> List[str]:
        ...

    @sync_compatible
    async def download_object_to_path(
        self, 
        from_path: str, 
        to_path: Optional[str] = None, 
        **download_kwargs: Any
    ) -> Path:
        ...

    @sync_compatible
    async def download_object_to_file_object(
        self, 
        from_path: str, 
        to_file_object: BinaryIO, 
        **download_kwargs: Any
    ) -> BinaryIO:
        ...

    @sync_compatible
    async def download_folder_to_path(
        self, 
        from_folder: str, 
        to_folder: Optional[str] = None, 
        **download_kwargs: Any
    ) -> Path:
        ...

    @sync_compatible
    async def upload_from_path(
        self, 
        from_path: Union[str, Path], 
        to_path: Optional[str] = None, 
        **upload_kwargs: Any
    ) -> str:
        ...

    @sync_compatible
    async def upload_from_file_object(
        self, 
        from_file_object: BinaryIO, 
        to_path: str, 
        **upload_kwargs: Any
    ) -> str:
        ...

    @sync_compatible
    async def upload_from_folder(
        self, 
        from_folder: Union[str, Path], 
        to_folder: Optional[str] = None, 
        **upload_kwargs: Any
    ) -> str:
        ...

    @sync_compatible
    async def upload_from_dataframe(
        self, 
        df: DataFrame, 
        to_path: str, 
        serialization_format: Union[DataFrameSerializationFormat, str] = DataFrameSerializationFormat.CSV_GZIP, 
        **upload_kwargs: Any
    ) -> str:
        ...
