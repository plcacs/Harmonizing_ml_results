from typing import Dict, Any, Generator
import pytest
import pandas as pd
import boto3
import requests
import subprocess
import shlex
import time
import uuid

def compression_to_extension() -> Dict[str, str]:
    return {value: key for key, value in icom.extension_to_compression.items()}

def tips_file(datapath: Any) -> str:
    return datapath('io', 'data', 'csv', 'tips.csv')

def jsonl_file(datapath: Any) -> str:
    return datapath('io', 'parser', 'data', 'items.jsonl')

def salaries_table(datapath: Any) -> pd.DataFrame:
    return pd.read_csv(datapath('io', 'parser', 'data', 'salaries.csv'), sep='\t')

def feather_file(datapath: Any) -> str:
    return datapath('io', 'data', 'feather', 'feather-0_3_1.feather')

def xml_file(datapath: Any) -> str:
    return datapath('io', 'data', 'xml', 'books.xml')

def s3_base(worker_id: str, monkeypatch: Any) -> Generator[str, None, None]:
    ...

def s3so(s3_base: str) -> Dict[str, Any]:
    return {'client_kwargs': {'endpoint_url': s3_base}}

def s3_resource(s3_base: str) -> boto3.resources.base.ServiceResource:
    s3 = boto3.resource('s3', endpoint_url=s3_base)
    return s3

def s3_public_bucket(s3_resource: boto3.resources.base.ServiceResource) -> boto3.resources.factory.s3.Bucket:
    ...

def s3_public_bucket_with_data(s3_public_bucket: boto3.resources.factory.s3.Bucket, tips_file: str, jsonl_file: str, feather_file: str, xml_file: str) -> boto3.resources.factory.s3.Bucket:
    ...

def s3_private_bucket(s3_resource: boto3.resources.base.ServiceResource) -> boto3.resources.factory.s3.Bucket:
    ...

def s3_private_bucket_with_data(s3_private_bucket: boto3.resources.factory.s3.Bucket, tips_file: str, jsonl_file: str, feather_file: str, xml_file: str) -> boto3.resources.factory.s3.Bucket:
    ...

def compression_format(request: Any) -> Any:
    ...

def compression_ext(request: Any) -> Any:
    ...
