from dataclasses import dataclass
from typing import Optional, List, Union, cast, Any

import cryptography
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.x509 import (
    ExtensionNotFound,
    ExtensionOID,
    Certificate,
    load_pem_x509_certificate,
    TLSFeature,
    DNSName,
    IPAddress,
    Name,
)
from cryptography.x509.ocsp import load_der_ocsp_response, OCSPResponseStatus, OCSPResponse
import nassl.ocsp_response
from nassl._nassl import OCSP_RESPONSE

from sslyze.plugins.certificate_info._symantec import SymantecDistructTester
from sslyze.plugins.certificate_info.trust_stores.trust_store import TrustStore, PathValidationResult


@dataclass(frozen=True)
class CertificateDeploymentAnalysisResult:
    """The result of analyzing a server's certificate to verify its validity."""

    received_certificate_chain: List[Certificate]
    leaf_certificate_has_must_staple_extension: bool
    leaf_certificate_is_ev: bool
    leaf_certificate_signed_certificate_timestamps_count: Optional[int]
    received_chain_contains_anchor_certificate: Optional[bool]
    received_chain_has_valid_order: Optional[bool]

    path_validation_results: List[PathValidationResult]
    verified_chain_has_sha1_signature: Optional[bool]
    verified_chain_has_legacy_symantec_anchor: Optional[bool]

    ocsp_response: Optional[OCSPResponse]
    ocsp_response_is_trusted: Optional[bool]

    @property
    def verified_certificate_chain(self) -> Optional[List[Certificate]]:
        """Get one of the verified certificate chains if one was successfully built using any of the trust stores."""
        for path_result in self.path_validation_results:
            if path_result.was_validation_successful:
                return path_result.verified_certificate_chain
        return None

    @property
    def verified_certificate_chain_as_pem(self) -> Optional[List[str]]:
        if self.verified_certificate_chain is None:
            return None

        pem_certs: List[str] = []
        for certificate in self.verified_certificate_chain:
            pem_certs.append(certificate.public_bytes(Encoding.PEM).decode("ascii"))
        return pem_certs

    @property
    def received_certificate_chain_as_pem(self) -> List[str]:
        pem_certs: List[str] = []
        for certificate in self.received_certificate_chain:
            pem_certs.append(certificate.public_bytes(Encoding.PEM).decode("ascii"))
        return pem_certs


class CertificateDeploymentAnalyzer:
    """Utility class for analyzing a certificate chain as deployed on a specific server."""

    def __init__(
        self,
        server_subject: Union[IPAddress, DNSName],
        server_certificate_chain_as_pem: List[str],
        server_ocsp_response: Optional[OCSP_RESPONSE],
        trust_stores_for_validation: List[TrustStore],
    ) -> None:
        self.server_subject = server_subject
        self.server_certificate_chain_as_pem = server_certificate_chain_as_pem
        self.server_ocsp_response = server_ocsp_response
        self.trust_stores_for_validation = trust_stores_for_validation

    def perform(self) -> CertificateDeploymentAnalysisResult:
        received_certificate_chain: List[Certificate] = [
            load_pem_x509_certificate(pem_cert.encode("ascii"), default_backend())
            for pem_cert in self.server_certificate_chain_as_pem
        ]
        leaf_cert: Certificate = received_certificate_chain[0]

        # OCSP Must-Staple
        has_ocsp_must_staple: bool = False
        try:
            tls_feature_ext = leaf_cert.extensions.get_extension_for_oid(ExtensionOID.TLS_FEATURE)
            tls_feature_value = cast(TLSFeature, tls_feature_ext.value)
            for feature_type in tls_feature_value:
                if feature_type == cryptography.x509.TLSFeatureType.status_request:
                    has_ocsp_must_staple = True
                    break
        except ExtensionNotFound:
            pass

        # Received chain order
        is_chain_order_valid: Optional[bool] = True
        previous_issuer: Optional[Name] = None
        for index, cert in enumerate(received_certificate_chain):
            try:
                current_subject: Name = cert.subject
            except ValueError:
                is_chain_order_valid = None
                break

            if index > 0:
                if current_subject != previous_issuer:
                    is_chain_order_valid = False
                    break
            try:
                previous_issuer = cert.issuer
            except KeyError:
                previous_issuer = None
            except ValueError:
                is_chain_order_valid = None
                break

        # Check if the leaf certificate is Extended Validation
        is_leaf_certificate_ev: bool = False
        for trust_store in self.trust_stores_for_validation:
            if trust_store.ev_oids is None:
                continue

            is_leaf_certificate_ev = trust_store.is_certificate_extended_validation(leaf_cert)

        # Check for Signed Timestamps
        number_of_scts: Optional[int] = 0
        try:
            sct_ext = leaf_cert.extensions.get_extension_for_oid(ExtensionOID.PRECERT_SIGNED_CERTIFICATE_TIMESTAMPS)
            if isinstance(sct_ext.value, cryptography.x509.UnrecognizedExtension):
                number_of_scts = None

            sct_ext_value = cast(cryptography.x509.PrecertificateSignedCertificateTimestamps, sct_ext.value)
            number_of_scts = len(sct_ext_value)
        except ExtensionNotFound:
            pass

        # Try to generate the verified certificate chain using each trust store
        all_path_validation_results: List[PathValidationResult] = []
        for trust_store in self.trust_stores_for_validation:
            path_validation_result = trust_store.verify_certificate_chain(
                self.server_certificate_chain_as_pem, self.server_subject
            )
            all_path_validation_results.append(path_validation_result)

        # Keep one trust store that was able to build the verified chain to then run additional checks
        trust_store_that_can_build_verified_chain: Optional[TrustStore] = None
        verified_certificate_chain: Optional[List[Certificate]] = None

        def sort_function(path_validation: PathValidationResult) -> str:
            return path_validation.trust_store.name.lower()

        all_path_validation_results.sort(key=sort_function)

        for path_validation_result in all_path_validation_results:
            if path_validation_result.was_validation_successful:
                trust_store_that_can_build_verified_chain = path_validation_result.trust_store
                verified_certificate_chain = path_validation_result.verified_certificate_chain
                break

        # Check if the anchor was sent by the server
        has_anchor_in_certificate_chain: Optional[bool] = None
        if verified_certificate_chain:
            has_anchor_in_certificate_chain = verified_certificate_chain[-1] in received_certificate_chain

        # Check if a SHA1-signed certificate is in the chain
        has_sha1_in_certificate_chain: Optional[bool] = None
        if verified_certificate_chain:
            has_sha1_in_certificate_chain = False
            for cert in verified_certificate_chain[:-1]:
                if isinstance(cert.signature_hash_algorithm, hashes.SHA1):
                    has_sha1_in_certificate_chain = True
                    break

        # Check if this is a distrusted Symantec-issued chain
        verified_chain_has_legacy_symantec_anchor: Optional[bool] = None
        if verified_certificate_chain:
            symantec_distrust_timeline = SymantecDistructTester.get_distrust_timeline(verified_certificate_chain)
            verified_chain_has_legacy_symantec_anchor = True if symantec_distrust_timeline else False

        # Check the OCSP response if there is one
        is_ocsp_response_trusted: Optional[bool] = None
        final_ocsp_response: Optional[OCSPResponse] = None
        if self.server_ocsp_response:
            final_ocsp_response = load_der_ocsp_response(self.server_ocsp_response.as_der_bytes())

            if (
                trust_store_that_can_build_verified_chain
                and final_ocsp_response.response_status == OCSPResponseStatus.SUCCESSFUL
            ):
                try:
                    nassl.ocsp_response.verify_ocsp_response(
                        self.server_ocsp_response, trust_store_that_can_build_verified_chain.path
                    )
                    is_ocsp_response_trusted = True
                except nassl.ocsp_response.OcspResponseNotTrustedError:
                    is_ocsp_response_trusted = False

        return CertificateDeploymentAnalysisResult(
            received_certificate_chain=received_certificate_chain,
            leaf_certificate_has_must_staple_extension=has_ocsp_must_staple,
            leaf_certificate_is_ev=is_leaf_certificate_ev,
            leaf_certificate_signed_certificate_timestamps_count=number_of_scts,
            received_chain_contains_anchor_certificate=has_anchor_in_certificate_chain,
            received_chain_has_valid_order=is_chain_order_valid,
            verified_chain_has_sha1_signature=has_sha1_in_certificate_chain,
            verified_chain_has_legacy_symantec_anchor=verified_chain_has_legacy_symantec_anchor,
            path_validation_results=all_path_validation_results,
            ocsp_response=final_ocsp_response,
            ocsp_response_is_trusted=is_ocsp_response_trusted,
        )
