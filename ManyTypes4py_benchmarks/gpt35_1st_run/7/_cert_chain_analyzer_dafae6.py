from dataclasses import dataclass
from typing import Optional, List, Union
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import ExtensionNotFound, ExtensionOID, Certificate, load_pem_x509_certificate, TLSFeature
from cryptography.x509.ocsp import OCSPResponseStatus, OCSPResponse
from sslyze.plugins.certificate_info.trust_stores.trust_store import TrustStore, PathValidationResult

@dataclass(frozen=True)
class CertificateDeploymentAnalysisResult:
    received_certificate_chain: List[Certificate]
    verified_certificate_chain: Optional[List[Certificate]]
    path_validation_results: List[PathValidationResult]
    leaf_certificate_is_ev: bool
    leaf_certificate_has_must_staple_extension: bool
    leaf_certificate_signed_certificate_timestamps_count: Optional[int]
    received_chain_has_valid_order: Optional[bool]
    received_chain_contains_anchor_certificate: Optional[bool]
    verified_chain_has_sha1_signature: Optional[bool]
    verified_chain_has_legacy_symantec_anchor: Optional[bool]
    ocsp_response: Optional[OCSPResponse]
    ocsp_response_is_trusted: Optional[bool]

    @property
    def verified_certificate_chain(self) -> Optional[List[Certificate]]:
        ...

    @property
    def verified_certificate_chain_as_pem(self) -> Optional[List[str]]:
        ...

    @property
    def received_certificate_chain_as_pem(self) -> List[str]:
        ...

class CertificateDeploymentAnalyzer:
    def __init__(self, server_subject: str, server_certificate_chain_as_pem: List[str], server_ocsp_response: bytes, trust_stores_for_validation: List[TrustStore]):
        ...

    def perform(self) -> CertificateDeploymentAnalysisResult:
        ...
